#!/usr/bin/env python3
"""QWEN-016 decision gate, stage 2: decompose the NVLS EP collectives.

Runs the production AllGather-V (3-tensor) and ReduceScatter-V at the exact
Qwen3-30B-A3B EP4 decode shapes on 4 GB200s, in lockstep (so inter-rank skew is
~0), under CUDA graphs (matching `full_iteration_inference`), and separates:

  launch      — an empty Triton kernel replayed in the same graph,
  barrier     — `symm_mem_sync` alone at the same grid/block shape,
  transfer    — the full collective minus the barrier.

It also sweeps the CTA count. The library exposes `max_num_blocks` as a kwarg
but the dispatcher never passes it, so every collective runs with 128 CTAs of
which only `ep_max_tokens` (64 at BS256/EP4) do work — and the barrier is
*per CTA*, so its cost scales with the number of active CTAs.

Run (inside the container, 4 GPUs):
  python -m torch.distributed.run --nproc-per-node 4 dev/moe_fused/harness_comm.py
"""
import argparse
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from megatron.core.inference.communication.torch_symm_triton.barrier import symm_mem_sync
from megatron.core.inference.communication.torch_symm_triton.utils import sync_threads
from megatron.core.inference.communication.torch_symm_triton.variable_collectives import (
    multimem_all_gatherv_3tensor,
    multimem_reduce_scatter_v,
)
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager


@triton.jit
def _barrier_only_kernel(
    signal_pad_ptrs,
    ep_max_tokens_ptr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Just the per-CTA symmetric-memory barrier, same shape as the collectives."""
    pid = tl.program_id(axis=0)
    ep_max_tokens = tl.load(ep_max_tokens_ptr)
    if pid >= ep_max_tokens:
        return
    sync_threads()
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )


@triton.jit
def _empty_kernel(ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    if pid < 0:
        tl.store(ptr + tl.arange(0, BLOCK_SIZE), 0)


def graph_time_us(fn, iters_per_graph=20, replays=50, warmup=5):
    """Median per-call device time of `fn` replayed inside a CUDA graph."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            for _ in range(iters_per_graph):
                fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()

    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()

    times = []
    for _ in range(5):
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        dist.barrier()
        torch.cuda.synchronize()
        ev0.record()
        for _ in range(replays):
            g.replay()
        ev1.record()
        torch.cuda.synchronize()
        times.append(ev0.elapsed_time(ev1) * 1e3 / (replays * iters_per_graph))
    times.sort()
    return times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--local-tokens", type=int, default=64)
    ap.add_argument("--per-rank-max", type=int, default=2048)
    ap.add_argument("--blocks", default="4,8,16,32,64,128")
    ap.add_argument("--rsv-dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    ep_group = dist.group.WORLD

    H, topk, LT = args.hidden, args.topk, args.local_tokens
    global_max = args.per_rank_max * world
    rsv_dtype = getattr(torch, args.rsv_dtype)

    def _mb(shape, dtype):
        n = 1
        for s in shape:
            n *= s
        return max(1, (n * torch.tensor([], dtype=dtype).element_size() + 2**20 - 1) // 2**20)

    agv_h = SymmetricMemoryManager.get_buffer(
        "b_h", process_group=ep_group, size_mb=_mb([global_max, H], torch.bfloat16)
    ).maybe_get_tensor([global_max, H], dtype=torch.bfloat16)
    agv_r = SymmetricMemoryManager.get_buffer(
        "b_r", process_group=ep_group, size_mb=_mb([global_max, topk], torch.int64)
    ).maybe_get_tensor([global_max, topk], dtype=torch.int64)
    agv_p = SymmetricMemoryManager.get_buffer(
        "b_p", process_group=ep_group, size_mb=_mb([global_max, topk], torch.float32)
    ).maybe_get_tensor([global_max, topk], dtype=torch.float32)
    rsv = SymmetricMemoryManager.get_buffer(
        "b_rsv", process_group=ep_group, size_mb=_mb([global_max, H], rsv_dtype)
    ).maybe_get_tensor([global_max, H], dtype=rsv_dtype)
    for nm, b in (("h", agv_h), ("r", agv_r), ("p", agv_p), ("rsv", rsv)):
        assert b["handle"] is not None, f"symm buffer {nm} failed to init"

    dev = torch.cuda.current_device()
    rank_token_offset = torch.tensor([rank * LT], dtype=torch.int32, device=dev)
    ep_max_tokens = torch.tensor([LT], dtype=torch.int32, device=dev)

    loc_h = torch.randn(LT, H, device=dev, dtype=torch.bfloat16)
    loc_r = torch.randint(0, 128, (LT, topk), device=dev, dtype=torch.int64)
    loc_p = torch.rand(LT, topk, device=dev, dtype=torch.float32)
    rsv_out = torch.empty(LT, H, device=dev, dtype=rsv_dtype)
    scratch = torch.zeros(1024, device=dev, dtype=torch.float32)
    # Zero the reduce buffer: uninitialised symmetric memory can hold denormals
    # or NaNs, which would distort the ld_reduce timing.
    rsv["tensor"].zero_()
    torch.cuda.synchronize()

    agv_bytes = LT * (H * 2 + topk * 8 + topk * 4)
    rsv_bytes = LT * H * rsv["tensor"].element_size()

    def agv(nb):
        multimem_all_gatherv_3tensor(
            agv_h["tensor"], agv_r["tensor"], agv_p["tensor"],
            loc_h, loc_r, loc_p,
            agv_h["handle"], agv_r["handle"], agv_p["handle"],
            rank_token_offset=rank_token_offset,
            ep_max_tokens=ep_max_tokens,
            per_rank_max_tokens=args.per_rank_max,
            max_num_blocks=nb,
        )

    def rsv_call(nb):
        multimem_reduce_scatter_v(
            rsv_out, rsv["tensor"], rsv["handle"],
            rank_token_offset=rank_token_offset,
            ep_max_tokens=ep_max_tokens,
            per_rank_max_tokens=args.per_rank_max,
            max_num_blocks=nb,
        )

    def barrier_only(nb, block_size):
        _barrier_only_kernel[(nb, 1, 1)](
            agv_h["handle"].signal_pad_ptrs_dev,
            ep_max_tokens,
            RANK=rank,
            WORLD_SIZE=world,
            BLOCK_SIZE=block_size,
            num_warps=max(1, block_size // 32),
        )

    def empty(nb, block_size):
        _empty_kernel[(nb, 1, 1)](
            scratch, BLOCK_SIZE=block_size, num_warps=max(1, block_size // 32)
        )

    # Correctness gate: a barrier that releases too early shows up here as a
    # torn gather or a partial reduction, so this must pass before any timing
    # number from this run is trusted.
    ref_h = torch.empty(world * LT, H, device=dev, dtype=torch.bfloat16)
    dist.all_gather_into_tensor(ref_h, loc_h)
    # Every rank contributes to every output slice, so each fills the whole
    # gathered region with a rank-scaled integer pattern. Values stay under
    # 2**24, so the fp32 reduction is exact and any mismatch is a real race.
    base = torch.arange(
        1, world * LT * H + 1, device=dev, dtype=rsv_dtype
    ).view(world * LT, H)
    fill = (rank + 1) * base
    ref_rsv = (world * (world + 1) // 2) * base[rank * LT : (rank + 1) * LT]
    agv_err = rsv_err = 0.0
    for _ in range(64):
        agv_h["tensor"].zero_()
        rsv["tensor"].zero_()
        rsv["tensor"][: world * LT].copy_(fill)
        dist.barrier()
        agv(128)
        rsv_call(128)
        torch.cuda.synchronize()
        got_h = agv_h["tensor"][: world * LT].float()
        agv_err = max(agv_err, (got_h - ref_h.float()).abs().max().item())
        rsv_err = max(
            rsv_err,
            ((rsv_out.float() - ref_rsv.float()).abs() / ref_rsv.float().abs())
            .max().item(),
        )
    if rank == 0:
        print(f"correctness over 64 trials: AGV max|diff| {agv_err:.3e}  "
              f"RSV max rel err {rsv_err:.3e}")
    assert agv_err == 0.0, f"AGV mismatch {agv_err}"
    assert rsv_err == 0.0, f"RSV mismatch {rsv_err}"
    rsv["tensor"].zero_()
    torch.cuda.synchronize()

    blocks = [int(b) for b in args.blocks.split(",")]
    results = {}

    if rank == 0:
        print(f"world={world} hidden={H} topk={topk} local_tokens={LT} "
              f"per_rank_max={args.per_rank_max} rsv_dtype={args.rsv_dtype}")
        print(f"AGV payload {agv_bytes} B/rank   RSV payload {rsv_bytes} B/rank")
        print(f"NVLink floor @900 GB/s/dir: AGV ingress "
              f"{(world-1)*agv_bytes/0.9e12*1e6:.3f} us, RSV egress "
              f"{(world-1)*rsv_bytes/0.9e12*1e6:.3f} us")
        print()
        print(f"{'CTAs':>5} {'AGV_us':>9} {'RSV_us':>9} {'bar256_us':>10} "
              f"{'bar512_us':>10} {'empty_us':>9}")

    for nb in blocks:
        t_agv = graph_time_us(lambda nb=nb: agv(nb))
        t_rsv = graph_time_us(lambda nb=nb: rsv_call(nb))
        t_b256 = graph_time_us(lambda nb=nb: barrier_only(nb, 256))
        t_b512 = graph_time_us(lambda nb=nb: barrier_only(nb, 512))
        t_e = graph_time_us(lambda nb=nb: empty(nb, 256))
        results[nb] = (t_agv, t_rsv, t_b256, t_b512, t_e)
        if rank == 0:
            print(f"{nb:>5} {t_agv:9.3f} {t_rsv:9.3f} {t_b256:10.3f} "
                  f"{t_b512:10.3f} {t_e:9.3f}")

    if rank == 0:
        prod = results.get(128)
        if prod:
            a, r, b256, b512, e = prod
            print("\n--- production shape (128 CTAs launched, "
                  f"{LT} active) decomposition ---")
            print(f"  AGV total {a:.3f} us = launch {e:.3f} + barrier "
                  f"{b256-e:.3f} + transfer {a-b256:.3f}")
            print(f"  RSV total {r:.3f} us = launch {e:.3f} + barrier "
                  f"{b512-e:.3f} + transfer {r-b512:.3f}")
            print(f"  per layer {a+r:.3f} us -> per step (48 layers) "
                  f"{48*(a+r):.1f} us")
        best_a = min(results.items(), key=lambda kv: kv[1][0])
        best_r = min(results.items(), key=lambda kv: kv[1][1])
        print(f"\n  best AGV: {best_a[1][0]:.3f} us at {best_a[0]} CTAs "
              f"({results[128][0]/best_a[1][0]:.3f}x vs production 128)")
        print(f"  best RSV: {best_r[1][1]:.3f} us at {best_r[0]} CTAs "
              f"({results[128][1]/best_r[1][1]:.3f}x vs production 128)")
        best_sum = min(48 * (results[a][0] + results[b][1])
                       for a in results for b in results)
        print(f"  best per-step comm: {best_sum:.1f} us vs production "
              f"{48*(results[128][0]+results[128][1]):.1f} us")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
