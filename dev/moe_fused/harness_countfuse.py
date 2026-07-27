#!/usr/bin/env python3
"""QWEN-018 A/B: fold the MoE token count into `_prefix_fill_init_kernel`.

reference  = `_moe_align_block_size_fused`        (4 launches: zeros fill,
             `_count_local_tokens_kernel_persistent`, `_prefix_fill_init_kernel`,
             `_scatter_token_indices_kernel`)
candidate  = `_moe_align_block_size_count_fused`  (2 launches: the count folded
             into `_count_prefix_fill_init_kernel`, then the same scatter)

Three measurements, in order:
  1. table equality -- num_tokens_post_padded, expert_ids, and the multiset of
     sorted_token_ids must match exactly (the scatter's atomics permute rows
     within an expert block on both sides, so compare sorted).
  2. whole-MoE output equality, expected bit-exact.
  3. CUDA-graph replay device time of the align call alone and of the whole
     `vllm_fused_moe` call. Eager wall time is launch-bound (~12 us/launch) and
     would flatter a launch-count change; production captures the full decode
     iteration, so only device time counts (QWEN-014's lesson).

Shapes mirror Qwen3-30B-A3B decode on EP4: H=2048, moe_ffn=768, 32 local
experts, top-8, 256 tokens.

Usage (inside the container):
  python dev/moe_fused/harness_countfuse.py --iters 300
"""
import argparse
import importlib

import torch

from megatron.core.inference.moe.fused_moe import ActivationType
from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

# `import ... as vfm` yields the *function*: `moe/__init__.py` rebinds that
# attribute on the package. Patching module flags on it silently no-ops and the
# A/B then compares one path against itself (QWEN-013).
vfm = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")
assert hasattr(vfm, "_USE_FUSED_COUNT"), "vfm is not the module"

H = 2048
NF = 768
TOPK = 8
NUM_GLOBAL_EXPERTS = 128
NUM_LOCAL_EXPERTS = 32


def build(valid, max_tokens, seed=0, expert_start=0):
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
    """Table-level equality across token counts, expert offsets and block sizes."""
    print("=== indirection-table equality ===")
    ok = True
    for valid in args.tokens:
        for block_m in (16, 64):
            for start in (0, 32):
                max_tokens = valid * 4
                *_, routing, valid_t = build(valid, max_tokens, seed=valid + start)
                ref = vfm._moe_align_block_size_fused(
                    routing, block_m, NUM_LOCAL_EXPERTS, start, valid_t
                )
                cand = vfm._moe_align_block_size_count_fused(
                    routing, block_m, NUM_LOCAL_EXPERTS, start, valid_t
                )
                torch.cuda.synchronize()
                npp_r, npp_c = int(ref[2].item()), int(cand[2].item())
                nb = npp_r // block_m
                eq_npp = npp_r == npp_c
                eq_eids = torch.equal(ref[1][:nb], cand[1][:nb])
                eq_sorted = torch.equal(
                    torch.sort(ref[0][:npp_r].int()).values,
                    torch.sort(cand[0][:npp_c].int()).values,
                )
                good = eq_npp and eq_eids and eq_sorted
                ok &= good
                print(f"  valid={valid:<5} BLOCK_M={block_m:<3} expert_start={start:<3} "
                      f"npp {npp_r}/{npp_c}  expert_ids={eq_eids}  "
                      f"sorted_ids={eq_sorted}  -> {'OK' if good else 'MISMATCH'}")
    print(f"table equality: {'PASS' if ok else 'FAIL'}\n")
    return ok


def check_output(args):
    """Whole-MoE output equality; expected bit-exact (max_abs and max_rel both 0)."""
    print("=== whole-MoE output equality ===")
    ok = True
    for valid in args.tokens:
        max_tokens = valid * 4
        hidden, probs, fc1, fc2, routing, valid_t = build(valid, max_tokens, seed=valid)
        common = dict(
            activation_type=ActivationType.SWIGLU,
            num_local_experts=NUM_LOCAL_EXPERTS,
            local_expert_start=0,
            valid_tokens=valid_t,
            routing_map=routing,
            num_tokens_hint=valid,
            fuse_fc1_activation=True,
        )
        vfm._USE_FUSED_ALIGN = True
        vfm._TUNE_DECODE_GEMM = True
        outs = {}
        for fc in (False, True):
            vfm._USE_FUSED_COUNT = fc
            outs[fc] = vllm_fused_moe(hidden, probs, fc1, fc2, **common)[:valid].float().clone()
        torch.cuda.synchronize()
        r, c = outs[False], outs[True]
        d = (r - c).abs()
        # QWEN-014: check max RELATIVE error, not a loose allclose. A loose
        # allclose(2e-2) passed on numerically wrong results there.
        rel = (d / r.abs().clamp_min(1e-6)).max().item()
        good = d.max().item() == 0.0 and rel == 0.0
        ok &= good
        print(f"  valid={valid:<5} max_abs={d.max().item():.3e}  max_rel={rel:.3e}  "
              f"norm {r.norm().item():.4f}/{c.norm().item():.4f}  "
              f"-> {'BIT-EXACT' if good else 'DIFFERS'}")
    vfm._USE_FUSED_COUNT = False
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
    vfm._USE_FUSED_ALIGN = True
    vfm._TUNE_DECODE_GEMM = True
    print(f"{'valid':>6} {'align ref':>10} {'align cand':>11} {'x':>7}   "
          f"{'moe ref':>9} {'moe cand':>9} {'x':>7}")
    for valid in args.tokens:
        max_tokens = valid * 4
        hidden, probs, fc1, fc2, routing, valid_t = build(valid, max_tokens, seed=valid)
        block_m = 16
        a_ref = _graph_time(
            lambda: vfm._moe_align_block_size_fused(
                routing, block_m, NUM_LOCAL_EXPERTS, 0, valid_t),
            args.iters, args.repeats)
        a_cand = _graph_time(
            lambda: vfm._moe_align_block_size_count_fused(
                routing, block_m, NUM_LOCAL_EXPERTS, 0, valid_t),
            args.iters, args.repeats)

        common = dict(
            activation_type=ActivationType.SWIGLU,
            num_local_experts=NUM_LOCAL_EXPERTS,
            local_expert_start=0,
            valid_tokens=valid_t,
            routing_map=routing,
            num_tokens_hint=valid,
            fuse_fc1_activation=True,
        )
        m = {}
        for fc in (False, True):
            vfm._USE_FUSED_COUNT = fc
            m[fc] = _graph_time(
                lambda: vllm_fused_moe(hidden, probs, fc1, fc2, **common),
                args.iters, args.repeats)
        vfm._USE_FUSED_COUNT = False
        print(f"{valid:>6} {a_ref:10.2f} {a_cand:11.2f} {a_ref/a_cand:6.3f}x   "
              f"{m[False]:9.2f} {m[True]:9.2f} {m[False]/m[True]:6.3f}x")


def breakdown(args):
    """Per-kernel device time inside the align call, ref vs candidate."""
    from torch.profiler import ProfilerActivity, profile

    valid = 256
    max_tokens = valid * 4
    *_, routing, valid_t = build(valid, max_tokens, seed=valid)
    n = 100
    for label, fn in (
        ("reference (4 launches)", vfm._moe_align_block_size_fused),
        ("candidate (2 launches)", vfm._moe_align_block_size_count_fused),
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
        total = sum(e.device_time_total for e in evs)
        print(f"\n===== {label} =====")
        print(f"{'kernel':<52} {'calls':>7} {'us/iter':>9}")
        for e in evs[:8]:
            print(f"{e.key[:52]:<52} {e.count/n:7.1f} {e.device_time_total/n:9.2f}")
        print(f"{'TOTAL device (sum of kernels)':<52} {'':>7} {total/n:9.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tokens", type=int, nargs="+", default=[128, 256, 384, 512])
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()

    ok = check_tables(args)
    ok &= check_output(args)
    if not args.skip_timing:
        timing(args)
        breakdown(args)
    print(f"\nCORRECTNESS: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
