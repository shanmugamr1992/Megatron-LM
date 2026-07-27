#!/usr/bin/env python3
"""Stage-2/3 pre-check: does flashinfer's BF16 fused MoE work at mcore's decode
contract, and is it faster than the shipping vLLM-Triton grouped GEMM?

Reference = `vllm_fused_moe` (the shipping decode path, fused FC1+SwiGLU +
fused align). Candidates:

  A. `flashinfer.fused_moe.cutlass_fused_moe` exactly as
     `InferenceGroupedMLP._flashinfer_forward` calls it (experts.py:1057).
  B. `flashinfer.fused_moe.trtllm_bf16_routed_moe` as vLLM's
     `TrtLlmBf16ExpertsMonolithic` modular variant calls it — the kernel vLLM
     actually wins with.

Both are probed for: import/JIT success, numerics vs the reference (including a
gate|up vs up|gate weight-layout swap test, since the concatenated mcore FC1
weight ordering may not match TRT-LLM's), and per-call latency.

Shapes are one EP4 rank of Qwen3-30B-A3B decode: H=2048, moe_ffn=768,
128 global / 32 local experts, top-8.

Usage (inside the container):
  python dev/moe_fused/harness_flashinfer.py --valid 256 --iters 100
"""
import argparse
import traceback

import torch

from megatron.core.inference.moe.fused_moe import ActivationType
from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

H = 2048
NF = 768
TOPK = 8
NUM_GLOBAL_EXPERTS = 128
NUM_LOCAL_EXPERTS = 32


def build(valid, max_tokens, ep_rank, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    hidden = (torch.randn(max_tokens, H, generator=g, device=dev) * 0.1).to(torch.bfloat16)
    # [E, 2*NF, H] with the gate half first, then the up half (TE GroupedLinear order).
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


def timeit(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


def compare(name, ref, cand, valid):
    r = ref[:valid].float()
    c = cand[:valid].float()
    d = (r - c).abs()
    rel = d / r.abs().clamp_min(1e-4)
    print(f"  {name:<28} max_abs={d.max().item():.4e} mean_abs={d.mean().item():.4e} "
          f"max_rel={rel.max().item():.3e} "
          f"allclose(2e-2)={torch.allclose(r, c, rtol=2e-2, atol=2e-2)} "
          f"|ref|={r.norm().item():.4f} |cand|={c.norm().item():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--ep-size", type=int, default=4)
    ap.add_argument("--ep-rank", type=int, default=0)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--dense-only", action="store_true",
                    help="set max_tokens == valid (no -1 padded rows)")
    args = ap.parse_args()

    valid = args.valid
    max_tokens = valid if args.dense_only else (args.max_tokens or valid * 4)
    ep_rank = args.ep_rank
    local_start = ep_rank * NUM_LOCAL_EXPERTS

    hidden, probs, fc1, fc2, routing, valid_t = build(valid, max_tokens, ep_rank)
    print(f"device={torch.cuda.get_device_name()} valid={valid} max_tokens={max_tokens} "
          f"ep_size={args.ep_size} ep_rank={ep_rank} local_expert_start={local_start}")

    # ---------------- reference: shipping vLLM Triton path ----------------
    ref_kwargs = dict(
        activation_type=ActivationType.SWIGLU,
        num_local_experts=NUM_LOCAL_EXPERTS,
        local_expert_start=local_start,
        valid_tokens=valid_t,
        routing_map=routing,
        num_tokens_hint=valid,
        fuse_fc1_activation=True,
    )
    ref = vllm_fused_moe(hidden, probs, fc1, fc2, **ref_kwargs)
    torch.cuda.synchronize()
    t_ref = timeit(lambda: vllm_fused_moe(hidden, probs, fc1, fc2, **ref_kwargs), args.iters)
    print(f"\nREFERENCE vllm_fused_moe        : {t_ref:8.2f} us/call")

    try:
        from flashinfer import fused_moe
        from flashinfer.tllm_enums import ActivationType as FiAct
    except Exception:
        print("flashinfer import failed:")
        traceback.print_exc()
        return
    print(f"flashinfer ActivationType members: "
          f"{[m for m in dir(FiAct) if not m.startswith('_') and m[0].isupper()]}")

    # ---------------- candidate A: cutlass_fused_moe (mcore's wiring) ----------------
    for act_name in ("Swiglu", "Silu"):
        act = getattr(FiAct, act_name, None)
        if act is None:
            print(f"\n[A] cutlass_fused_moe act={act_name}: enum member missing, skipped")
            continue
        for swap in (False, True):
            w1 = fc1
            if swap:
                gate, up = fc1[:, :NF], fc1[:, NF:]
                w1 = torch.cat([up, gate], dim=1).contiguous()
            tag = f"[A] cutlass act={act_name}{' swap(up|gate)' if swap else ' (gate|up)'}"
            try:
                out = fused_moe.cutlass_fused_moe(
                    hidden,
                    routing.int(),
                    probs,
                    w1,
                    fc2,
                    hidden.dtype,
                    quant_scales=None,
                    activation_type=act,
                    ep_size=args.ep_size,
                    ep_rank=ep_rank,
                    output=None,
                )[0]
                torch.cuda.synchronize()
                print(f"\n{tag}: OK")
                compare("numerics vs reference", ref, out, valid)
                t = timeit(
                    lambda: fused_moe.cutlass_fused_moe(
                        hidden, routing.int(), probs, w1, fc2, hidden.dtype,
                        quant_scales=None, activation_type=act,
                        ep_size=args.ep_size, ep_rank=ep_rank, output=None,
                    )[0],
                    args.iters,
                )
                print(f"  latency {t:8.2f} us/call  ({t_ref/t:.3f}x vs reference)")
            except Exception as exc:
                print(f"\n{tag}: FAILED {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=6)

    # -------- graph-level device time: the only fair comparison ----------------
    # Eager wall-clock above is launch-count dominated (~12 us/launch), which
    # flatters any single-launch kernel. Production captures the whole decode
    # iteration in a CUDA graph, so device time is what actually competes.
    act = getattr(FiAct, "Swiglu")
    gate, up = fc1[:, :NF], fc1[:, NF:]
    w1_swapped = torch.cat([up, gate], dim=1).contiguous()

    def graph_time(fn, iters=300, repeats=3):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
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
        ts.sort()
        return ts[len(ts) // 2], None

    print("\n===== CUDA-graph device time (the decisive comparison) =====")
    import importlib
    vfm = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")
    vfm._USE_FUSED_ALIGN = True
    results = {}
    for label, tuned in (("vllm_fused_moe (default tiles)", False),
                         ("vllm_fused_moe (QWEN-013 tiles)", True)):
        vfm._TUNE_DECODE_GEMM = tuned
        t, err = graph_time(lambda: vllm_fused_moe(hidden, probs, fc1, fc2, **ref_kwargs))
        results[label] = t
        print(f"  {label:<34} {t:8.2f} us" if t else f"  {label:<34} FAILED {err}")
    vfm._TUNE_DECODE_GEMM = False

    t, err = graph_time(
        lambda: fused_moe.cutlass_fused_moe(
            hidden, routing.int(), probs, w1_swapped, fc2, hidden.dtype,
            quant_scales=None, activation_type=act,
            ep_size=args.ep_size, ep_rank=ep_rank, output=None,
        )[0]
    )
    print(f"  {'cutlass_fused_moe (up|gate)':<34} "
          + (f"{t:8.2f} us" if t else f"FAILED {err}"))
    if t:
        for label, ref_t in results.items():
            if ref_t:
                print(f"      vs {label}: {ref_t/t:.3f}x")

    # ---------------- candidate B: trtllm_bf16_routed_moe (vLLM's kernel) ------------
    RM = fused_moe.RoutingMethodType
    WL = fused_moe.WeightLayout
    # vLLM passes already-normalized topk ids/weights with RoutingMethodType.TopK.
    topk_ids = routing.int()
    for use_shuffled, layout_name in ((True, "BlockMajorK"), (False, "MajorK")):
        layout = getattr(WL, layout_name)
        tag = f"[B] trtllm_bf16_routed_moe shuffled={use_shuffled} layout={layout_name}"
        try:
            out = fused_moe.trtllm_bf16_routed_moe(
                topk_ids,
                hidden,
                fc1,
                fc2,
                NUM_GLOBAL_EXPERTS,
                TOPK,
                None,
                None,
                NF,
                local_start,
                NUM_LOCAL_EXPERTS,
                routing_method_type=int(RM.TopK),
                use_shuffled_weight=use_shuffled,
                weight_layout=int(layout),
                do_finalize=True,
            )
            out = out[0] if isinstance(out, (list, tuple)) else out
            torch.cuda.synchronize()
            print(f"\n{tag}: OK  out.shape={tuple(out.shape)} dtype={out.dtype}")
            compare("numerics vs reference", ref, out, valid)
        except Exception as exc:
            print(f"\n{tag}: FAILED {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=6)


if __name__ == "__main__":
    main()
