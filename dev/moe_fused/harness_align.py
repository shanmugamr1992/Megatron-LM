#!/usr/bin/env python3
"""Correctness + timing A/B for the fused MoE indirection-table build.

reference  = vllm_fused_moe with the default 5-kernel align
             (init + count + prefix + fill + scatter)
candidate  = vllm_fused_moe with the 3-kernel fused align
             (count + prefix_fill_init + scatter), toggled via the module flag.

Both run with fuse_fc1_activation=True (the shipping decode config). The align
change only affects the indirection table, so outputs must match bit-for-bit up
to atomic-scatter ordering (which the grouped-GEMM reduction is invariant to).

Shapes mirror Qwen3-30B-A3B decode on EP4: H=2048, moe_ffn=768, 32 local
experts, top-8. Usage (inside the container):
  python dev/moe_fused/harness_align.py --valid 256 --iters 100
"""
import argparse
import importlib

import torch

from megatron.core.inference.moe.fused_moe import ActivationType
from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

# `import megatron.core.inference.moe.vllm_fused_moe as vfm` returns the
# *function*, not the module: `moe/__init__.py` does `from .vllm_fused_moe import
# vllm_fused_moe`, rebinding that attribute on the package. Patching a flag on it
# silently no-ops, which makes the A/B below compare one path against itself.
vfm = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")
assert hasattr(vfm, "_USE_FUSED_ALIGN"), "vfm is not the module"


def build_inputs(valid, max_tokens, H, Nf, num_local_experts, num_global_experts, topk, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    hidden = (
        torch.randn(max_tokens, H, generator=g, device=dev, dtype=torch.float32) * 0.1
    ).to(torch.bfloat16)
    fc1 = (
        torch.randn(num_local_experts, 2 * Nf, H, generator=g, device=dev, dtype=torch.float32)
        * 0.02
    ).to(torch.bfloat16)
    fc2 = (
        torch.randn(num_local_experts, H, Nf, generator=g, device=dev, dtype=torch.float32) * 0.02
    ).to(torch.bfloat16)
    routing = torch.full((max_tokens, topk), -1, device=dev, dtype=torch.int64)
    for t in range(valid):
        routing[t] = torch.randperm(num_global_experts, generator=g, device=dev)[:topk]
    probs = torch.zeros(max_tokens, topk, device=dev, dtype=torch.float32)
    probs[:valid] = torch.softmax(
        torch.randn(valid, topk, generator=g, device=dev, dtype=torch.float32), dim=-1
    )
    valid_t = torch.tensor([valid], device=dev, dtype=torch.int32)
    return hidden, probs, fc1, fc2, routing, valid_t


def run(fused_align, hidden, probs, fc1, fc2, common):
    vfm._USE_FUSED_ALIGN = fused_align
    return vllm_fused_moe(hidden, probs, fc1, fc2, fuse_fc1_activation=True, **common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--iters", type=int, default=0)
    args = ap.parse_args()

    H, Nf, topk = 2048, 768, 8
    num_local_experts, num_global_experts = 32, 128
    # Mirror the decode buffer: after EP4 all-gather a rank can hold up to
    # valid*ep tokens; use 4x to exercise padding/tail handling.
    max_tokens = args.max_tokens or args.valid * 4

    hidden, probs, fc1, fc2, routing, valid_t = build_inputs(
        args.valid, max_tokens, H, Nf, num_local_experts, num_global_experts, topk
    )
    common = dict(
        activation_type=ActivationType.SWIGLU,
        num_local_experts=num_local_experts,
        local_expert_start=0,
        valid_tokens=valid_t,
        routing_map=routing,
        num_tokens_hint=args.valid,
    )

    ref = run(False, hidden, probs, fc1, fc2, common)
    cand = run(True, hidden, probs, fc1, fc2, common)
    torch.cuda.synchronize()

    r = ref[: args.valid].float()
    c = cand[: args.valid].float()
    abs_diff = (r - c).abs()
    rel = abs_diff / r.abs().clamp_min(1e-4)
    print(f"max_tokens={max_tokens} valid={args.valid}")
    print(f"ref norm={r.norm().item():.4f} cand norm={c.norm().item():.4f}")
    print(f"max_abs_diff={abs_diff.max().item():.6e}  mean_abs={abs_diff.mean().item():.6e}")
    print(f"max_rel_diff={rel.max().item():.6e}  mean_rel={rel.mean().item():.6e}")
    print(f"ALLCLOSE(rtol=1e-2,atol=1e-2): {torch.allclose(r, c, rtol=1e-2, atol=1e-2)}")

    if args.iters:

        def bench(fused_align):
            fn = lambda: run(fused_align, hidden, probs, fc1, fc2, common)
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            s = torch.cuda.Event(True)
            e = torch.cuda.Event(True)
            s.record()
            for _ in range(args.iters):
                fn()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / args.iters

        t_ref = bench(False)
        t_cand = bench(True)
        print(f"\nreference (5-kernel align) : {t_ref*1e3:.2f} us/call")
        print(f"candidate (3-kernel align) : {t_cand*1e3:.2f} us/call")
        print(f"speedup                    : {t_ref/t_cand:.3f}x")


if __name__ == "__main__":
    main()
