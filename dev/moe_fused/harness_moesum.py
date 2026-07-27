#!/usr/bin/env python3
"""QWEN-020 A/B: predicated `_moe_sum_kernel_fast` vs the branched `_moe_sum_kernel`.

reference  = `_moe_sum_kernel`      (BLOCK_K=1024, uniform scalar branch per topk slot)
candidate  = `_moe_sum_kernel_fast` (BLOCK_K=2048, locality test predicated into
             the load mask, num_warps=8)

The reduction order over topk is identical and masked-off slots contribute an
exact fp32 zero, so the output must be bit-exact. What changes is whether the
CTA can overlap slot t's data load with slot t+1's index load.

QWEN-018 measured the branched kernel at 7.79 us/layer in the BS256 decode
graph while it moves only ~6.3 MB (~1.05 us at the 6.08 TB/s of QWEN-012).

Usage (inside the container):
  python dev/moe_fused/harness_moesum.py --iters 300
"""
import argparse
import importlib

import torch

from megatron.core.inference.moe.fused_moe import ActivationType
from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

vfm = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")
assert hasattr(vfm, "_USE_FAST_MOE_SUM"), "vfm is not the module"

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tokens", type=int, nargs="+", default=[128, 256, 384, 512])
    args = ap.parse_args()

    vfm._USE_FUSED_ALIGN = True
    vfm._USE_FUSED_COUNT = True
    vfm._TUNE_DECODE_GEMM = True

    print("=== whole-MoE output equality (expect bit-exact) ===")
    ok = True
    for valid in args.tokens:
        max_tokens = valid * 4
        hidden, probs, fc1, fc2, routing, valid_t = build(valid, max_tokens, seed=valid)
        common = _common(valid, routing, valid_t)
        outs = {}
        for fast in (False, True):
            vfm._USE_FAST_MOE_SUM = fast
            outs[fast] = vllm_fused_moe(hidden, probs, fc1, fc2, **common)[:valid].float().clone()
        torch.cuda.synchronize()
        r, c = outs[False], outs[True]
        d = (r - c).abs()
        # QWEN-014: max RELATIVE error, not a loose allclose.
        rel = (d / r.abs().clamp_min(1e-6)).max().item()
        good = d.max().item() == 0.0 and rel == 0.0
        ok &= good
        print(f"  valid={valid:<5} max_abs={d.max().item():.3e}  max_rel={rel:.3e}  "
              f"-> {'BIT-EXACT' if good else 'DIFFERS'}")
    vfm._USE_FAST_MOE_SUM = False
    print(f"output equality: {'PASS (bit-exact)' if ok else 'NOT BIT-EXACT'}\n")

    print("=== CUDA-graph replay device time, whole vllm_fused_moe ===")
    print(f"device={torch.cuda.get_device_name()}")
    print(f"{'valid':>6} {'moe ref':>9} {'moe fast':>9} {'x':>8}")
    for valid in args.tokens:
        max_tokens = valid * 4
        hidden, probs, fc1, fc2, routing, valid_t = build(valid, max_tokens, seed=valid)
        common = _common(valid, routing, valid_t)
        m = {}
        for fast in (False, True):
            vfm._USE_FAST_MOE_SUM = fast
            m[fast] = _graph_time(
                lambda: vllm_fused_moe(hidden, probs, fc1, fc2, **common),
                args.iters, args.repeats)
        vfm._USE_FAST_MOE_SUM = False
        print(f"{valid:>6} {m[False]:9.2f} {m[True]:9.2f} {m[False]/m[True]:7.3f}x")

    print("\n=== per-kernel device time inside the MoE call ===")
    from torch.profiler import ProfilerActivity, profile

    valid = 256
    hidden, probs, fc1, fc2, routing, valid_t = build(valid, valid * 4, seed=valid)
    common = _common(valid, routing, valid_t)
    n = 100
    for fast in (False, True):
        vfm._USE_FAST_MOE_SUM = fast
        for _ in range(20):
            vllm_fused_moe(hidden, probs, fc1, fc2, **common)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                vllm_fused_moe(hidden, probs, fc1, fc2, **common)
            torch.cuda.synchronize()
        evs = [e for e in prof.key_averages() if e.device_time_total > 0]
        evs.sort(key=lambda e: -e.device_time_total)
        print(f"\n----- fast_moe_sum={fast} -----")
        for e in evs[:8]:
            print(f"{e.key[:52]:<52} {e.count/n:7.1f} {e.device_time_total/n:9.2f}")
        print(f"{'TOTAL device':<52} {'':>7} "
              f"{sum(e.device_time_total for e in evs)/n:9.2f}")
    vfm._USE_FAST_MOE_SUM = False
    print(f"\nCORRECTNESS: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
