#!/usr/bin/env python3
"""Correctness + timing A/B for the decode grouped-GEMM retune.

reference  = vllm_fused_moe with `_get_default_config` shared across FC1/FC2
candidate  = vllm_fused_moe with `_get_decode_tuned_configs` (per-GEMM tiles,
             shared BLOCK_SIZE_M), toggled via the module flag.

Both run with the shipping decode config (fused FC1+SwiGLU epilogue, fused
3-kernel align). The retune changes only tile sizes, so outputs differ solely
in fp32 accumulation order — expect ~1e-3 relative, never a structural change.

Shapes mirror Qwen3-30B-A3B decode on EP4: H=2048, moe_ffn=768, 32 local
experts, top-8. Also sweeps the token count so we can see whether the tuned
config holds away from the nominal 256.

Usage (inside the container):
  python dev/moe_fused/harness_gemmtune.py --iters 300
"""
import argparse
import importlib

import torch

from megatron.core.inference.moe.fused_moe import ActivationType
from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

# NOTE: `import megatron.core.inference.moe.vllm_fused_moe as vfm` does NOT give
# the module here — `moe/__init__.py` does `from .vllm_fused_moe import
# vllm_fused_moe`, which rebinds that attribute on the package to the *function*.
# Patching flags on it would silently no-op and the A/B would compare the default
# path against itself. importlib returns the real module from sys.modules.
vfm = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")
assert hasattr(vfm, "_get_default_config"), "vfm is not the module"

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


def graph_timing(args):
    """Time the MoE call as a captured CUDA graph replay.

    Production captures the whole decode iteration with
    `--cuda-graph-scope full_iteration_inference`, so host launch overhead is
    amortized and only device time matters. The eager per-call wall time is
    launch-bound (~12 us/launch here) and hides GEMM improvements entirely.
    """
    print(f"device={torch.cuda.get_device_name()}  (CUDA-graph replay timing)")
    print(f"{'valid':>6} {'ref us':>9} {'tuned us':>9} {'speedup':>8}")
    for valid in args.tokens:
        max_tokens = valid * 4
        hidden, probs, fc1, fc2, routing, valid_t = build(valid, max_tokens)
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
        res = {}
        for tuned in (False, True):
            vfm._TUNE_DECODE_GEMM = tuned
            # Warm up (Triton JIT + autotune) on a side stream before capture.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(5):
                    vllm_fused_moe(hidden, probs, fc1, fc2, **common)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                vllm_fused_moe(hidden, probs, fc1, fc2, **common)
            for _ in range(10):
                g.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(args.repeats):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                for _ in range(args.iters):
                    g.replay()
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) / args.iters * 1e3)
            ts.sort()
            res[tuned] = ts[len(ts) // 2]
            del g
        print(f"{valid:>6} {res[False]:9.2f} {res[True]:9.2f} "
              f"{res[False]/res[True]:7.3f}x")
    vfm._TUNE_DECODE_GEMM = False


def profile_breakdown(args):
    """Per-kernel CUDA time inside the full vllm_fused_moe call, ref vs tuned."""
    from torch.profiler import ProfilerActivity, profile

    valid = 256
    max_tokens = valid * 4
    hidden, probs, fc1, fc2, routing, valid_t = build(valid, max_tokens)
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
    n = 50
    for tuned in (False, True):
        vfm._TUNE_DECODE_GEMM = tuned
        cfg = (
            vfm._get_decode_tuned_configs(valid, NUM_LOCAL_EXPERTS, TOPK)
            if tuned
            else (vfm._get_default_config(valid, NUM_LOCAL_EXPERTS, TOPK),) * 2
        )
        print(f"\n===== tuned={tuned} =====")
        print(f"fc1 cfg: {cfg[0]}")
        print(f"fc2 cfg: {cfg[1]}")
        for _ in range(20):
            vllm_fused_moe(hidden, probs, fc1, fc2, **common)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                vllm_fused_moe(hidden, probs, fc1, fc2, **common)
            torch.cuda.synchronize()
        evs = [e for e in prof.key_averages() if e.device_time_total > 0]
        evs.sort(key=lambda e: -e.device_time_total)
        total = sum(e.device_time_total for e in evs)
        print(f"{'kernel':<52} {'calls':>7} {'us/iter':>9} {'share':>7}")
        for e in evs[:12]:
            print(f"{e.key[:52]:<52} {e.count/n:7.1f} "
                  f"{e.device_time_total/n:9.2f} "
                  f"{e.device_time_total/total*100:6.1f}%")
        print(f"{'TOTAL GPU (sum of kernels)':<52} {'':>7} {total/n:9.2f}")
    vfm._TUNE_DECODE_GEMM = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tokens", type=int, nargs="+", default=[128, 256, 384, 512])
    ap.add_argument("--profile", action="store_true",
                    help="per-kernel attribution of the full call, both configs")
    ap.add_argument("--graph", action="store_true",
                    help="time CUDA-graph replays (what production actually sees)")
    args = ap.parse_args()

    if args.profile:
        profile_breakdown(args)
        return
    if args.graph:
        graph_timing(args)
        return

    print(f"device={torch.cuda.get_device_name()}")
    print(f"{'valid':>6} {'ref us':>9} {'tuned us':>9} {'speedup':>8} "
          f"{'max_abs':>10} {'max_rel':>10} {'allclose':>9}")

    for valid in args.tokens:
        max_tokens = valid * 4
        hidden, probs, fc1, fc2, routing, valid_t = build(valid, max_tokens)
        common = dict(
            activation_type=ActivationType.SWIGLU,
            num_local_experts=NUM_LOCAL_EXPERTS,
            local_expert_start=0,
            valid_tokens=valid_t,
            routing_map=routing,
            num_tokens_hint=valid,
            fuse_fc1_activation=True,
        )
        # The fused 3-kernel align is what ships alongside this change.
        vfm._USE_FUSED_ALIGN = True

        def run(tuned):
            vfm._TUNE_DECODE_GEMM = tuned
            return vllm_fused_moe(hidden, probs, fc1, fc2, **common)

        ref = run(False).clone()
        cand = run(True).clone()
        torch.cuda.synchronize()
        r, c = ref[:valid].float(), cand[:valid].float()
        d = (r - c).abs()
        rel = d / r.abs().clamp_min(1e-4)

        def bench(tuned):
            ts = []
            for _ in range(args.repeats):
                for _ in range(10):
                    run(tuned)
                torch.cuda.synchronize()
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                for _ in range(args.iters):
                    run(tuned)
                e.record()
                torch.cuda.synchronize()
                ts.append(s.elapsed_time(e) / args.iters * 1e3)
            ts.sort()
            return ts[len(ts) // 2]

        t_ref = bench(False)
        t_cand = bench(True)
        print(f"{valid:>6} {t_ref:9.2f} {t_cand:9.2f} {t_ref/t_cand:7.3f}x "
              f"{d.max().item():10.3e} {rel.max().item():10.3e} "
              f"{str(torch.allclose(r, c, rtol=2e-2, atol=2e-2)):>9}")

    vfm._TUNE_DECODE_GEMM = False


if __name__ == "__main__":
    main()
