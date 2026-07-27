#!/usr/bin/env python3
"""Stage-1 roofline / padding-waste analysis for the decode MoE grouped GEMM.

Answers, at the *real* Qwen3-30B-A3B EP4 decode shapes, whether the ~40% of
decode GPU time spent in `_fused_moe_kernel` is recoverable inefficiency or
irreducible expert-weight traffic:

  1. Indirection-table padding waste per BLOCK_SIZE_M
     (`num_tokens_post_padded` vs the number of real local token-expert pairs).
  2. Isolated FC1 (fused SwiGLU) and FC2 timing.
  3. Achieved TFLOP/s on both valid and padded FLOPs.
  4. Achieved *expert-weight* bandwidth, compared against the HBM bandwidth
     measured on this very GPU (never an assumed peak), plus the resulting
     bandwidth-bound time floor.
  5. A BLOCK_SIZE_* sweep, to check whether the shipping config is even the
     best Triton config at these shapes.

All shapes are per-EP-rank, matching what a decode step actually sees:
hidden=2048, moe_ffn=768, 128 global / 32 local experts, top-8. The valid token
count defaults to `num_tokens_hint` = local_tokens * ep_size (log the real value
from the server rather than trusting this default).

Usage (inside the container):
  python dev/moe_fused/harness_roofline.py --valid 256 --iters 200
"""
import argparse

import torch

import megatron.core.inference.moe.vllm_fused_moe as vfm
from megatron.core.inference.moe.vllm_fused_moe import (
    _get_default_config,
    _invoke_fused_moe_kernel,
    _moe_align_block_size_fused,
)

H = 2048
NF = 768
TOPK = 8
NUM_GLOBAL_EXPERTS = 128
NUM_LOCAL_EXPERTS = 32


def build_inputs(valid, max_tokens, seed=0):
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
    probs[:valid] = torch.softmax(
        torch.randn(valid, TOPK, generator=g, device=dev), dim=-1
    )
    valid_t = torch.tensor([valid], device=dev, dtype=torch.int32)
    return hidden, probs, fc1, fc2, routing, valid_t


def measure_hbm_bandwidth(nbytes=2 << 30, iters=30):
    """Measured device-to-device copy bandwidth on this GPU (read+write)."""
    n = nbytes // 2
    src = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        dst.copy_(src)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    # copy touches 2*nbytes (one read + one write)
    return 2 * nbytes / (ms * 1e-3) / 1e12  # TB/s


def measure_stream_read_bandwidth(fc1, fc2, iters=50):
    """Vectorized pure-streaming-read ceiling over the exact weight tensors.

    A Triton kernel that reads every element once (128-bit loads, grid-stride)
    and writes one scalar per CTA. This is the honest read ceiling; torch's
    `.sum()` is not bandwidth-optimal and understates it.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def _read_kernel(p, out, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        base = pid * BLOCK
        step = nprog * BLOCK
        for off in tl.range(base, n, step):
            idx = off + tl.arange(0, BLOCK)
            acc += tl.load(p + idx, mask=idx < n, other=0.0).to(tl.float32)
        tl.store(out + pid, tl.sum(acc))

    nbytes = fc1.numel() * 2 + fc2.numel() * 2
    grid = 148 * 8
    scratch = torch.empty(grid, dtype=torch.float32, device="cuda")

    def once():
        for t in (fc1, fc2):
            f = t.view(-1)
            _read_kernel[(grid,)](f, scratch, f.numel(), BLOCK=2048, num_warps=8)

    for _ in range(5):
        once()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        once()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    return nbytes / (ms * 1e-3) / 1e12


def measure_weight_read_bandwidth(fc1, fc2, iters=30):
    """Pure streaming-read bandwidth over the exact expert-weight tensors.

    This is the relevant ceiling for the grouped GEMM: every decode step must
    read every local expert weight once (all 32 local experts are hit at
    256 tokens x top-8).
    """
    nbytes = fc1.numel() * 2 + fc2.numel() * 2
    f1 = fc1.view(-1)
    f2 = fc2.view(-1)
    for _ in range(5):
        f1.sum()
        f2.sum()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        f1.sum()
        f2.sum()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    return nbytes, nbytes / (ms * 1e-3) / 1e12  # TB/s


def table_stats(routing, valid_t, valid, block_m):
    sorted_ids, expert_ids, num_post = _moe_align_block_size_fused(
        routing, block_m, NUM_LOCAL_EXPERTS, 0, valid_t
    )
    torch.cuda.synchronize()
    npp = int(num_post.item())
    live = routing[:valid]
    local_pairs = int(((live >= 0) & (live < NUM_LOCAL_EXPERTS)).sum().item())
    rows = torch.bincount(
        live[(live >= 0) & (live < NUM_LOCAL_EXPERTS)], minlength=NUM_LOCAL_EXPERTS
    )
    return {
        "block_m": block_m,
        "num_tokens_post_padded": npp,
        "real_local_pairs": local_pairs,
        "padding_factor": npp / max(local_pairs, 1),
        "dead_row_frac": 1.0 - local_pairs / max(npp, 1),
        "rows_per_expert_min": int(rows.min().item()),
        "rows_per_expert_med": int(rows.median().item()),
        "rows_per_expert_max": int(rows.max().item()),
    }


def bench_gemms(hidden, probs, fc1, fc2, routing, valid_t, valid, config, iters):
    """Time FC1(+SwiGLU) and FC2 separately with the production launch config."""
    block_m = config["BLOCK_SIZE_M"]
    sorted_ids, expert_ids, num_post = _moe_align_block_size_fused(
        routing, block_m, NUM_LOCAL_EXPERTS, 0, valid_t
    )
    max_tokens = hidden.size(0)
    num_valid = max_tokens * TOPK
    em_hint = valid * TOPK + block_m * NUM_LOCAL_EXPERTS
    num_pid_m_hint = -(-em_hint // block_m)
    grid_fc1 = num_pid_m_hint * (-(-NF // config["BLOCK_SIZE_N"]))
    grid_fc2 = num_pid_m_hint * (-(-H // config["BLOCK_SIZE_N"]))

    inter1 = torch.empty(num_valid, NF, dtype=torch.bfloat16, device="cuda")
    inter3 = torch.empty(num_valid, H, dtype=torch.bfloat16, device="cuda")
    probs_flat = probs.reshape(-1).contiguous()

    def fc1_call():
        _invoke_fused_moe_kernel(
            hidden, fc1, inter1, probs_flat, sorted_ids, expert_ids, num_post,
            mul_routed_weight=False, top_k=TOPK, config=config, grid_size=grid_fc1,
            fuse_squared_relu=False, fuse_swiglu=True,
        )

    def fc2_call():
        _invoke_fused_moe_kernel(
            inter1, fc2, inter3, probs_flat, sorted_ids, expert_ids, num_post,
            mul_routed_weight=False, top_k=1, config=config, grid_size=grid_fc2,
        )

    def timeit(fn):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters * 1e3  # us

    return timeit(fc1_call), timeit(fc2_call), int(num_post.item()), grid_fc1, grid_fc2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", type=int, default=256, help="num_tokens_hint (local_tokens*ep)")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--sweep", action="store_true", help="exhaustive BLOCK_SIZE sweep")
    ap.add_argument("--confirm", action="store_true",
                    help="re-bench the sweep's top candidates with repeats")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    max_tokens = args.max_tokens or args.valid * 4
    valid = args.valid
    hidden, probs, fc1, fc2, routing, valid_t = build_inputs(valid, max_tokens)

    print(f"=== device: {torch.cuda.get_device_name()} ===")
    print(f"shapes: H={H} moe_ffn={NF} local_experts={NUM_LOCAL_EXPERTS} "
          f"global_experts={NUM_GLOBAL_EXPERTS} topk={TOPK} valid={valid} "
          f"max_tokens={max_tokens}")

    d2d = measure_hbm_bandwidth()
    wbytes, wbw_sum = measure_weight_read_bandwidth(fc1, fc2)
    wbw = measure_stream_read_bandwidth(fc1, fc2)
    print(f"\n=== measured bandwidth on THIS gpu ===")
    print(f"d2d copy (read+write, aggregate) : {d2d:.3f} TB/s")
    print(f"torch .sum() read (weak proxy)   : {wbw_sum:.3f} TB/s over {wbytes/1e6:.1f} MB")
    print(f"triton vectorized read (CEILING) : {wbw:.3f} TB/s over {wbytes/1e6:.1f} MB")
    print(f"weight-bw floor for FC1+FC2      : {wbytes/(wbw*1e12)*1e6:.2f} us")

    print(f"\n=== indirection-table padding waste (per BLOCK_SIZE_M) ===")
    print(f"{'block_m':>8} {'npp':>8} {'real':>8} {'pad_x':>7} {'dead%':>7} "
          f"{'rows/e min/med/max':>22}")
    for bm in (16, 32, 64, 128):
        st = table_stats(routing, valid_t, valid, bm)
        print(f"{st['block_m']:>8} {st['num_tokens_post_padded']:>8} "
              f"{st['real_local_pairs']:>8} {st['padding_factor']:>7.2f} "
              f"{st['dead_row_frac']*100:>6.1f}% "
              f"{st['rows_per_expert_min']:>7}/{st['rows_per_expert_med']}/"
              f"{st['rows_per_expert_max']}")

    prod = _get_default_config(M=valid, E=NUM_LOCAL_EXPERTS, top_k=TOPK)
    print(f"\nproduction config (_get_default_config M={valid}): {prod}")

    # FLOP / byte accounting.  Weight traffic is mandatory and identical for
    # every config: all 32 local experts are active every step.
    fc1_flops_row = 2 * (2 * NF) * H
    fc2_flops_row = 2 * H * NF
    st_prod = table_stats(routing, valid_t, valid, prod["BLOCK_SIZE_M"])
    real_rows = st_prod["real_local_pairs"]
    fc1_w = fc1.numel() * 2
    fc2_w = fc2.numel() * 2

    def report(tag, config):
        t1, t2, npp, g1, g2 = bench_gemms(
            hidden, probs, fc1, fc2, routing, valid_t, valid, config, args.iters
        )
        tot = t1 + t2
        f1_valid = real_rows * fc1_flops_row
        f2_valid = real_rows * fc2_flops_row
        f1_pad = npp * fc1_flops_row
        f2_pad = npp * fc2_flops_row
        print(f"\n--- {tag}  {config}")
        print(f"  grid fc1/fc2               : {g1} / {g2} CTAs "
              f"(npp={npp}, real_rows={real_rows})")
        print(f"  FC1 (fused SwiGLU)         : {t1:8.2f} us   "
              f"valid {f1_valid/(t1*1e-6)/1e12:7.1f} TFLOP/s   "
              f"padded {f1_pad/(t1*1e-6)/1e12:7.1f} TFLOP/s   "
              f"w-bw {fc1_w/(t1*1e-6)/1e12:6.3f} TB/s "
              f"({fc1_w/(t1*1e-6)/1e12/wbw*100:5.1f}% of measured)")
        print(f"  FC2                        : {t2:8.2f} us   "
              f"valid {f2_valid/(t2*1e-6)/1e12:7.1f} TFLOP/s   "
              f"padded {f2_pad/(t2*1e-6)/1e12:7.1f} TFLOP/s   "
              f"w-bw {fc2_w/(t2*1e-6)/1e12:6.3f} TB/s "
              f"({fc2_w/(t2*1e-6)/1e12/wbw*100:5.1f}% of measured)")
        floor = (fc1_w + fc2_w) / (wbw * 1e12) * 1e6
        print(f"  FC1+FC2                    : {tot:8.2f} us  vs weight-bw floor "
              f"{floor:7.2f} us  -> {tot/floor:.2f}x off roofline")
        return tot

    base = report("PRODUCTION", prod)

    if args.confirm:
        # Shortlist from the exhaustive sweep, re-timed with repeats so a
        # config change is not accepted on single-shot noise.  BLOCK_SIZE_M is
        # shared between FC1 and FC2 (one indirection table), so FC1/FC2 best
        # per-GEMM configs are reported within each bm group.
        shortlist = [
            ("prod  bm64 bn128 bk64 w8 s3", 64, 128, 64, 8, 3),
            ("      bm64 bn64  bk64 w4 s3", 64, 64, 64, 4, 3),
            ("      bm64 bn64  bk64 w8 s3", 64, 64, 64, 8, 3),
            ("      bm32 bn128 bk64 w4 s3", 32, 128, 64, 4, 3),
            ("      bm32 bn256 bk64 w8 s4", 32, 256, 64, 8, 4),
            ("      bm32 bn64  bk64 w4 s3", 32, 64, 64, 4, 3),
            ("      bm16 bn128 bk64 w8 s4", 16, 128, 64, 8, 4),
            ("      bm16 bn64  bk64 w4 s3", 16, 64, 64, 4, 3),
            ("      bm16 bn256 bk64 w8 s4", 16, 256, 64, 8, 4),
        ]
        print(f"\n=== CONFIRM: {args.repeats} repeats x {args.iters} iters ===")
        print(f"{'config':<30} {'fc1 us (min/med/max)':>28} {'fc2 us (min/med/max)':>28} "
              f"{'tot':>8} {'vs prod':>8}")
        rows = []
        for name, bm, bn, bk, nw, ns in shortlist:
            cfg = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                       GROUP_SIZE_M=1, num_warps=nw, num_stages=ns)
            t1s, t2s = [], []
            for _ in range(args.repeats):
                t1, t2, _, _, _ = bench_gemms(
                    hidden, probs, fc1, fc2, routing, valid_t, valid, cfg, args.iters
                )
                t1s.append(t1)
                t2s.append(t2)
            t1s.sort()
            t2s.sort()
            m1, m2 = t1s[len(t1s) // 2], t2s[len(t2s) // 2]
            tot = m1 + m2
            rows.append((tot, name, m1, m2))
            print(f"{name:<30} {t1s[0]:8.2f}/{m1:7.2f}/{t1s[-1]:7.2f} "
                  f"{t2s[0]:8.2f}/{m2:7.2f}/{t2s[-1]:7.2f} {tot:8.2f} "
                  f"{base/tot:7.3f}x")
        rows.sort()
        print(f"\nBEST (median tot): {rows[0][1].strip()} at {rows[0][0]:.2f} us "
              f"(fc1 {rows[0][2]:.2f} + fc2 {rows[0][3]:.2f})")
        print(f"per-GEMM best: fc1 {min(r[2] for r in rows):.2f} us, "
              f"fc2 {min(r[3] for r in rows):.2f} us")
        floor = (fc1_w + fc2_w) / (wbw * 1e12) * 1e6
        print(f"vectorized-read floor: {floor:.2f} us -> best is "
              f"{rows[0][0]/floor:.2f}x off, production {base/floor:.2f}x off")

    if args.sweep:
        print(f"\n=== BLOCK_SIZE sweep (same numerics, different tiling) ===")
        best = (base, "PRODUCTION")
        for bm in (16, 32, 64):
            for bn in (64, 128, 256):
                for bk in (64, 128):
                    for nw, ns in ((4, 3), (8, 3), (8, 4)):
                        cfg = dict(
                            BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                            GROUP_SIZE_M=1, num_warps=nw, num_stages=ns,
                        )
                        try:
                            t1, t2, npp, _, _ = bench_gemms(
                                hidden, probs, fc1, fc2, routing, valid_t, valid,
                                cfg, max(args.iters // 4, 30),
                            )
                        except Exception as exc:  # OOM shmem etc.
                            print(f"  bm={bm} bn={bn} bk={bk} w={nw} s={ns}: FAIL "
                                  f"{type(exc).__name__}")
                            continue
                        tot = t1 + t2
                        flag = " <-- best" if tot < best[0] else ""
                        print(f"  bm={bm:>3} bn={bn:>3} bk={bk:>3} warps={nw} stages={ns}: "
                              f"fc1={t1:7.2f} fc2={t2:7.2f} tot={tot:7.2f} us "
                              f"({base/tot:.3f}x vs prod){flag}")
                        if tot < best[0]:
                            best = (tot, f"bm={bm} bn={bn} bk={bk} warps={nw} stages={ns}")
        print(f"\nBEST: {best[1]} at {best[0]:.2f} us "
              f"({base/best[0]:.3f}x vs production {base:.2f} us)")


if __name__ == "__main__":
    main()
