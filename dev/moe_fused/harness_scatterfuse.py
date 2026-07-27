#!/usr/bin/env python3
"""QWEN-022 A/B: fold the scatter into the fused indirection-table kernel.

reference  = `_moe_align_block_size_count_fused` (2 launches: QWEN-019's
             `_count_prefix_fill_init_kernel`, then `_scatter_token_indices_kernel`)
candidate  = `_moe_align_block_size_single`      (1 launch: `_align_single_kernel`,
             which streams the pairs a second time and places its own expert's
             rows itself, with no global atomics)

Row order within an expert block differs by construction (ascending pair index
instead of atomically permuted), so the table check compares the sorted multiset
per expert block, and the whole-MoE output check is the one that has to be
bit-exact -- rows are indexed by pair id, so a permutation inside a block must
not change any value.

Usage (inside the container):
  python dev/moe_fused/harness_scatterfuse.py --iters 300
"""
import argparse
import importlib

import torch

from megatron.core.inference.moe.fused_moe import ActivationType
from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

vfm = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")
assert hasattr(vfm, "_USE_FUSED_SCATTER"), "vfm is not the module"

H = 2048
NF = 768
TOPK = 8
NUM_GLOBAL_EXPERTS = 128
NUM_LOCAL_EXPERTS = 32


def build(valid, max_tokens, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    hidden = (torch.randn(max_tokens, H, generator=g, device=dev) * 0.1).to(torch.bfloat16)
    fc1 = (torch.randn(NUM_LOCAL_EXPERTS, 2 * NF, H, generator=g, device=dev) * 0.02).to(
        torch.bfloat16
    )
    fc2 = (torch.randn(NUM_LOCAL_EXPERTS, H, NF, generator=g, device=dev) * 0.02).to(
        torch.bfloat16
    )
    routing = torch.full((max_tokens, TOPK), -1, device=dev, dtype=torch.int64)
    for t in range(valid):
        routing[t] = torch.randperm(NUM_GLOBAL_EXPERTS, generator=g, device=dev)[:TOPK]
    probs = torch.zeros(max_tokens, TOPK, device=dev, dtype=torch.float32)
    probs[:valid] = torch.softmax(torch.randn(valid, TOPK, generator=g, device=dev), dim=-1)
    valid_t = torch.tensor([valid], device=dev, dtype=torch.int32)
    return hidden, probs, fc1, fc2, routing, valid_t


def check_tables(args):
    print("=== indirection-table equality ===")
    ok = True
    for valid in args.tokens:
        for block_m in (16, 64):
            for start in (0, 32):
                *_, routing, valid_t = build(valid, valid * 4, seed=valid + start)
                ref = vfm._moe_align_block_size_count_fused(
                    routing, block_m, NUM_LOCAL_EXPERTS, start, valid_t
                )
                cand = vfm._moe_align_block_size_single(
                    routing, block_m, NUM_LOCAL_EXPERTS, start, valid_t
                )
                torch.cuda.synchronize()
                npp_r, npp_c = int(ref[2].item()), int(cand[2].item())
                nb = npp_r // block_m
                eq_npp = npp_r == npp_c
                eq_eids = torch.equal(ref[1][:nb], cand[1][:nb])
                # Whole-table multiset equality. Rows are only grouped by expert,
                # not by BLOCK_M block: an expert's rows span several blocks and
                # the two paths order them differently inside that range, so a
                # per-block comparison would flag a difference that does not exist.
                eq_rows = torch.equal(
                    torch.sort(ref[0][:npp_r].int()).values,
                    torch.sort(cand[0][:npp_c].int()).values,
                )
                # Stronger check: every non-sentinel row must sit in a block whose
                # expert id actually owns that pair.
                placement_ok = True
                for tbl in (ref, cand):
                    rows = tbl[0][:npp_r].long()
                    blk_expert = tbl[1][:nb].long().repeat_interleave(block_m)
                    real = rows < (routing.shape[0] * TOPK)
                    pair_expert = routing.view(-1)[rows.clamp_max(routing.numel() - 1)]
                    placement_ok &= bool(
                        ((pair_expert - start) == blk_expert)[real].all().item()
                    )
                good = eq_npp and eq_eids and eq_rows and placement_ok
                ok &= good
                print(
                    f"  valid={valid:<5} BLOCK_M={block_m:<3} expert_start={start:<3} "
                    f"npp {npp_r}/{npp_c}  expert_ids={eq_eids}  rows={eq_rows}  "
                    f"placement={placement_ok}  -> {'OK' if good else 'MISMATCH'}"
                )
    print(f"table equality: {'PASS' if ok else 'FAIL'}\n")
    return ok


def _common(valid, routing, valid_t):
    return dict(
        activation_type=ActivationType.SWIGLU,
        num_local_experts=NUM_LOCAL_EXPERTS,
        local_expert_start=0,
        valid_tokens=valid_t,
        routing_map=routing,
        num_tokens_hint=valid,
        fuse_fc1_activation=True,
    )


def check_output(args):
    print("=== whole-MoE output equality (expect bit-exact) ===")
    ok = True
    for valid in args.tokens:
        hidden, probs, fc1, fc2, routing, valid_t = build(valid, valid * 4, seed=valid)
        common = _common(valid, routing, valid_t)
        outs = {}
        for sc in (False, True):
            vfm._USE_FUSED_SCATTER = sc
            outs[sc] = vllm_fused_moe(hidden, probs, fc1, fc2, **common)[:valid].float().clone()
        torch.cuda.synchronize()
        vfm._USE_FUSED_SCATTER = False
        r, c = outs[False], outs[True]
        d = (r - c).abs()
        rel = (d / r.abs().clamp_min(1e-6)).max().item()
        good = d.max().item() == 0.0 and rel == 0.0
        ok &= good
        print(
            f"  valid={valid:<5} max_abs={d.max().item():.3e}  max_rel={rel:.3e}  "
            f"norm {r.norm().item():.4f}/{c.norm().item():.4f}  "
            f"-> {'BIT-EXACT' if good else 'DIFFERS'}"
        )
    print(f"output equality: {'PASS (bit-exact)' if ok else 'NOT BIT-EXACT'}\n")
    return ok


def _graph_time(fn, iters, repeats):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / iters * 1e3)
    del g
    ts.sort()
    return ts[len(ts) // 2]


def timing(args):
    print("=== CUDA-graph replay device time ===")
    print(f"device={torch.cuda.get_device_name()}")
    print(
        f"{'valid':>6} {'align ref':>10} {'align cand':>11} {'x':>7}   "
        f"{'moe ref':>9} {'moe cand':>9} {'x':>7}"
    )
    for valid in args.tokens:
        hidden, probs, fc1, fc2, routing, valid_t = build(valid, valid * 4, seed=valid)
        block_m = 16
        a_ref = _graph_time(
            lambda: vfm._moe_align_block_size_count_fused(
                routing, block_m, NUM_LOCAL_EXPERTS, 0, valid_t
            ),
            args.iters,
            args.repeats,
        )
        a_cand = _graph_time(
            lambda: vfm._moe_align_block_size_single(
                routing, block_m, NUM_LOCAL_EXPERTS, 0, valid_t
            ),
            args.iters,
            args.repeats,
        )
        common = _common(valid, routing, valid_t)
        m = {}
        for sc in (False, True):
            vfm._USE_FUSED_SCATTER = sc
            m[sc] = _graph_time(
                lambda: vllm_fused_moe(hidden, probs, fc1, fc2, **common),
                args.iters,
                args.repeats,
            )
        vfm._USE_FUSED_SCATTER = False
        print(
            f"{valid:>6} {a_ref:10.2f} {a_cand:11.2f} {a_ref / a_cand:6.3f}x   "
            f"{m[False]:9.2f} {m[True]:9.2f} {m[False] / m[True]:6.3f}x"
        )


def breakdown(args):
    from torch.profiler import ProfilerActivity, profile

    valid = 256
    *_, routing, valid_t = build(valid, valid * 4, seed=valid)
    n = 100
    for label, fn in (
        ("reference (2 launches)", vfm._moe_align_block_size_count_fused),
        ("candidate (1 launch)", vfm._moe_align_block_size_single),
    ):
        for _ in range(20):
            fn(routing, 16, NUM_LOCAL_EXPERTS, 0, valid_t)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                fn(routing, 16, NUM_LOCAL_EXPERTS, 0, valid_t)
            torch.cuda.synchronize()
        evs = [e for e in prof.key_averages() if e.device_time_total > 0]
        evs.sort(key=lambda e: -e.device_time_total)
        print(f"\n===== {label} =====")
        for e in evs[:8]:
            print(f"{e.key[:52]:<52} {e.count / n:7.1f} {e.device_time_total / n:9.2f}")
        print(
            f"{'TOTAL device':<52} {'':>7} {sum(e.device_time_total for e in evs) / n:9.2f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tokens", type=int, nargs="+", default=[128, 256, 384, 512])
    args = ap.parse_args()

    vfm._USE_FUSED_ALIGN = True
    vfm._USE_FUSED_COUNT = True
    vfm._TUNE_DECODE_GEMM = True

    ok = check_tables(args)
    ok &= check_output(args)
    timing(args)
    breakdown(args)
    print(f"\nCORRECTNESS: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
