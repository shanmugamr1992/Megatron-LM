#!/usr/bin/env python3
"""QWEN-021 A/B: fused softmax+top-k router selection vs `torch.softmax` + `torch.topk`.

reference  = torch.softmax(logits, dim=-1, dtype=fp32) then
             torch.topk(scores, k=8, dim=1, sorted=False)   [what the compiled
             InferenceTopKRouter runs today: a softmax kernel + aten gatherTopK]
candidate  = `fused_softmax_topk` — one CTA per token, softmax in registers then
             8 max-then-mask selection passes.

QWEN-018 measured the pair at 6.05 us (gatherTopK) + 1.94 us (softmax) per layer
in the BS256 decode graph, for a [256, 128] fp32 reduction.

`torch.topk(sorted=False)` leaves the order of the k results unspecified, so the
correctness criterion is set equality of the (expert_id, prob) pairs per token,
plus the fp32 softmax value agreeing to within bf16 rounding.

Usage (inside the container):
  python dev/moe_fused/harness_routertopk.py --iters 500
"""
import argparse
import importlib

import torch

rt = importlib.import_module("megatron.core.inference.moe.router_topk")
assert hasattr(rt, "USE_FUSED_ROUTER_TOPK"), "rt is not the module"
# the fused path is wired into InferenceTopKRouter at module import; make sure that
# import still resolves (no cycle between transformer.moe and inference.moe).
importlib.import_module("megatron.core.transformer.moe.router")

NUM_EXPERTS = 128
TOPK = 8


def reference(logits, topk):
    scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    probs, idx = torch.topk(scores, k=topk, dim=1, sorted=False)
    return probs.type_as(logits), idx


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


def check(tokens, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = (torch.randn(tokens, NUM_EXPERTS, generator=g, device="cuda")).to(torch.bfloat16)
    rp, ri = reference(logits, TOPK)
    fp, fi = rt.fused_softmax_topk(logits, TOPK)
    torch.cuda.synchronize()

    # set equality of expert ids, per token
    ri_s, ri_o = torch.sort(ri, dim=1)
    fi_s, fi_o = torch.sort(fi, dim=1)
    idx_match = bool(torch.equal(ri_s, fi_s))
    # align probs by the sorted-index permutation before comparing
    rp_a = torch.gather(rp.float(), 1, ri_o)
    fp_a = torch.gather(fp.float(), 1, fi_o)
    d = (rp_a - fp_a).abs()
    rel = (d / rp_a.abs().clamp_min(1e-9)).max().item()
    exact = d.max().item() == 0.0
    # a duplicate expert id would silently break routing
    dup = bool((torch.sort(fi, dim=1).values.diff(dim=1) == 0).any())
    return idx_match, exact, rel, d.max().item(), dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tokens", type=int, nargs="+", default=[128, 256, 384, 512])
    args = ap.parse_args()

    print("=== routing equality vs torch softmax+topk ===")
    ok = True
    for tokens in args.tokens:
        for seed in (tokens, tokens + 1000):
            m, exact, rel, mx, dup = check(tokens, seed)
            good = m and rel == 0.0 and not dup
            ok &= good
            print(
                f"  tokens={tokens:<5} seed={seed:<6} same_expert_set={m}  dup_expert={dup}  "
                f"prob max_abs={mx:.3e} max_rel={rel:.3e}  "
                f"-> {'BIT-EXACT' if exact and m else ('MATCH' if good else 'DIFFERS')}"
            )
    print(f"routing equality: {'PASS' if ok else 'FAIL'}\n")

    print("=== CUDA-graph replay device time, router selection only ===")
    print(f"device={torch.cuda.get_device_name()}")
    print(f"{'tokens':>7} {'torch us':>9} {'fused us':>9} {'x':>8}")
    for tokens in args.tokens:
        g = torch.Generator(device="cuda").manual_seed(tokens)
        logits = torch.randn(tokens, NUM_EXPERTS, generator=g, device="cuda").to(torch.bfloat16)
        t_ref = _graph_time(lambda: reference(logits, TOPK), args.iters, args.repeats)
        t_new = _graph_time(lambda: rt.fused_softmax_topk(logits, TOPK), args.iters, args.repeats)
        print(f"{tokens:>7} {t_ref:9.2f} {t_new:9.2f} {t_ref / t_new:7.3f}x")

    print("\n=== per-kernel device time (tokens=256) ===")
    from torch.profiler import ProfilerActivity, profile

    g = torch.Generator(device="cuda").manual_seed(256)
    logits = torch.randn(256, NUM_EXPERTS, generator=g, device="cuda").to(torch.bfloat16)
    n = 200
    for name, fn in (("torch", lambda: reference(logits, TOPK)),
                     ("fused", lambda: rt.fused_softmax_topk(logits, TOPK))):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
        evs = [e for e in prof.key_averages() if e.device_time_total > 0]
        evs.sort(key=lambda e: -e.device_time_total)
        print(f"\n----- {name} -----")
        for e in evs[:6]:
            print(f"{e.key[:52]:<52} {e.count / n:7.1f} {e.device_time_total / n:9.2f}")
        print(f"{'TOTAL device':<52} {'':>7} {sum(e.device_time_total for e in evs) / n:9.2f}")

    print(f"\nCORRECTNESS: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
