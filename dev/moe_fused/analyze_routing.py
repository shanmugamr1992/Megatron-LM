#!/usr/bin/env python3
"""QWEN-018 decision gate: per-kernel breakdown of the routing/permute chain.

PROFILE-TUNED reports routing/permute at 1251 us/step over 242 kernels
(5.2 us/kernel). Before building any fusion we need to know, per kernel name:

  * how many launches per decode step,
  * device duration per launch,
  * the dispatch gap that follows it (in a single-stream full-iteration CUDA
    graph the per-step idle is the sum of these gaps, so removing a launch
    removes its gap too -- that is real wall time, not just kernel time),
  * how much of the duration is irreducible fixed cost (QWEN-016 measured an
    empty kernel at 0.72 us on this machine).

From that we can compute, per candidate fusion, the wall-time ceiling as
    launches_removed * (duration + following_gap)
and reject anything below ~1% of the step.

Usage: python3 analyze_routing.py <profile.sqlite> [--device N] [--steps N]
"""
import argparse
import re
import sqlite3
import statistics
from collections import defaultdict

K = "CUPTI_ACTIVITY_KIND_KERNEL"
COMM_PATTERNS = ("multimem_all_gatherv", "multimem_reduce_scatter", "nccl")

# Category regexes, ordered: first match wins.
CATEGORIES = [
    ("comm", r"multimem_all_gatherv|multimem_reduce_scatter|nccl"),
    ("moe_gemm", r"_fused_moe_kernel"),
    (
        "moe_routing",
        r"_count_local_tokens|_prefix_fill_init|_scatter_token_indices|_moe_sum"
        r"|_init_sorted_ids|_prefix_sum_kernel|_fill_expert_block_ids"
        r"|_permute_tokens|_unpermute_tokens|_zero_output_rows|_init_permutation_map"
        r"|bounded_silu_mul|topk|softmax|_routing|router",
    ),
    ("attention", r"flash|fmha|attention|paged|kv_cache|rope|rotary"),
    ("dense_gemm", r"nvjet|cutlass|gemm|sm100|sm90|ampere|volta|xmma|matmul"),
    ("norm", r"norm|rms|layer_norm"),
    ("elementwise", r"elementwise|vectorized|copy|cast|add|mul|fill|memset|scatter|gather"),
]


def categorize(name):
    low = name.lower()
    for cat, pat in CATEGORIES:
        if re.search(pat, low):
            return cat
    return "other"


def union_list(intervals):
    if not intervals:
        return []
    ivs = sorted(intervals)
    out = []
    cs, ce = ivs[0]
    for s, e in ivs[1:]:
        if s > ce:
            out.append((cs, ce))
            cs, ce = s, e
        else:
            ce = max(ce, e)
    out.append((cs, ce))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--device", type=int, default=3)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--steps", type=int, default=20, help="steady-state steps to aggregate")
    ap.add_argument("--stream", type=int, default=None)
    ap.add_argument("--window", default=None,
                    help="t0,t1 in seconds bounding the steady-state decode region "
                         "(from forward_pass.py); required, the AGV anchor alone also "
                         "matches the prefill/warmup region")
    ap.add_argument("--empty-kernel-us", type=float, default=0.72,
                    help="measured empty-kernel floor on this machine (QWEN-016)")
    args = ap.parse_args()
    win = None
    if args.window:
        a, b = args.window.split(",")
        win = (int(float(a) * 1e9), int(float(b) * 1e9))

    con = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    cur = con.cursor()
    names = {i: v for i, v in cur.execute("SELECT id, value FROM StringIds")}

    agv_id = [i for i, v in names.items() if "multimem_all_gatherv" in v]
    agv_set = ",".join(str(i) for i in agv_id)
    _wq0 = (f" AND start >= {int(float(args.window.split(',')[0])*1e9)}"
            f" AND start < {int(float(args.window.split(',')[1])*1e9)}") if args.window else ""
    rows = cur.execute(
        f"SELECT deviceId, streamId, COUNT(*) FROM {K} WHERE shortName IN ({agv_set}){_wq0} "
        "GROUP BY deviceId, streamId ORDER BY COUNT(*) DESC"
    ).fetchall()
    stream = args.stream if args.stream is not None else rows[0][1]
    dev = args.device
    print(f"decode stream={stream} device={dev}")

    wq = f" AND start >= {win[0]} AND start < {win[1]}" if win else ""
    starts = [r[0] for r in cur.execute(
        f"SELECT start FROM {K} WHERE shortName IN ({agv_set}) AND deviceId=? AND streamId=?"
        f"{wq} ORDER BY start", (dev, stream))]
    nsteps = len(starts) // args.layers
    periods = [starts[args.layers * (k + 1)] - starts[args.layers * k]
               for k in range(nsteps - 1)]
    lo, hi = max(2, nsteps // 4), min(nsteps - 2, 3 * nsteps // 4)
    med_period = statistics.median(periods[lo:hi])
    # Steady-state steps closest to the median period.
    cand = sorted(range(lo, hi), key=lambda k: abs(periods[k] - med_period))[: args.steps]
    cand.sort()
    print(f"steps in trace={nsteps}  median period={med_period/1e3:.3f} us  "
          f"aggregating {len(cand)} steady-state steps")

    # Per kernel-name accumulators, aggregated over the sampled steps.
    n_launch = defaultdict(int)
    dur_sum = defaultdict(int)
    durs = defaultdict(list)
    gap_after = defaultdict(list)
    grid = {}
    step_busy, step_idle, step_wall, step_nk = [], [], [], []

    for k_rep in cand:
        t0 = starts[args.layers * k_rep]
        t1 = starts[args.layers * (k_rep + 1)]
        ks = cur.execute(
            f"SELECT start, end, shortName, gridX, blockX FROM {K} "
            "WHERE deviceId=? AND start >= ? AND start < ? ORDER BY start",
            (dev, t0, t1)).fetchall()
        merged = union_list([(s, e) for s, e, *_ in ks])
        busy = sum(e - s for s, e in merged)
        step_wall.append(t1 - t0)
        step_busy.append(busy)
        step_idle.append(t1 - t0 - busy)
        step_nk.append(len(ks))

        # Gap that follows each kernel on the serial timeline. Kernels are
        # effectively serial in this graph; attribute the gap between the end of
        # kernel i and the start of kernel i+1 to kernel i.
        prev_end = None
        for idx, (s, e, sn, gx, bx) in enumerate(ks):
            nm = names[sn]
            n_launch[nm] += 1
            dur_sum[nm] += e - s
            durs[nm].append(e - s)
            grid[nm] = (gx, bx)
            if idx + 1 < len(ks):
                nxt = ks[idx + 1][0]
                g = nxt - max(e, prev_end or e)
                if g >= 0:
                    gap_after[nm].append(g)
            prev_end = max(e, prev_end or e)

    ns = len(cand)
    wall = statistics.mean(step_wall) / 1e3
    busy = statistics.mean(step_busy) / 1e3
    idle = statistics.mean(step_idle) / 1e3
    print(f"\nper step: wall {wall:.1f} us | GPU-busy {busy:.1f} us | "
          f"idle {idle:.1f} us ({100*idle/wall:.1f}%) | "
          f"kernels {statistics.mean(step_nk):.0f}")

    # ---- per-kernel-name table ----
    rowsout = []
    for nm, n in n_launch.items():
        per_step = n / ns
        d_mean = statistics.mean(durs[nm]) / 1e3
        d_step = dur_sum[nm] / ns / 1e3
        g_mean = statistics.mean(gap_after[nm]) / 1e3 if gap_after[nm] else 0.0
        g_step = sum(gap_after[nm]) / ns / 1e3 if gap_after[nm] else 0.0
        rowsout.append(dict(
            name=nm, cat=categorize(nm), per_step=per_step, d_mean=d_mean,
            d_step=d_step, g_mean=g_mean, g_step=g_step,
            wall_step=d_step + g_step, grid=grid[nm]))
    rowsout.sort(key=lambda r: -r["wall_step"])

    print(f"\n{'kernel':<52}{'cat':<13}{'#/step':>7}{'us/k':>8}{'us/step':>9}"
          f"{'gap/k':>7}{'gap/step':>9}{'WALL/step':>10}{'grid':>7}")
    print("-" * 124)
    for r in rowsout:
        if r["wall_step"] < 1.0:
            continue
        print(f"{r['name'][:51]:<52}{r['cat']:<13}{r['per_step']:>7.1f}{r['d_mean']:>8.2f}"
              f"{r['d_step']:>9.1f}{r['g_mean']:>7.2f}{r['g_step']:>9.1f}"
              f"{r['wall_step']:>10.1f}{r['grid'][0]:>7}")

    # ---- category rollup ----
    catd, catg, catn = defaultdict(float), defaultdict(float), defaultdict(float)
    for r in rowsout:
        catd[r["cat"]] += r["d_step"]
        catg[r["cat"]] += r["g_step"]
        catn[r["cat"]] += r["per_step"]
    print(f"\n{'category':<14}{'#/step':>8}{'kern_us':>10}{'gap_us':>9}{'WALL_us':>9}{'%wall':>8}")
    print("-" * 58)
    for cat in sorted(catd, key=lambda c: -(catd[c] + catg[c])):
        w = catd[cat] + catg[cat]
        print(f"{cat:<14}{catn[cat]:>8.1f}{catd[cat]:>10.1f}{catg[cat]:>9.1f}"
              f"{w:>9.1f}{100*w/wall:>7.1f}%")
    print(f"{'TOTAL':<14}{sum(catn.values()):>8.1f}{sum(catd.values()):>10.1f}"
          f"{sum(catg.values()):>9.1f}{sum(catd.values())+sum(catg.values()):>9.1f}")

    # ---- fixed-overhead accounting for the routing chain ----
    floor = args.empty_kernel_us
    print(f"\n=== routing/permute fixed-overhead accounting "
          f"(empty-kernel floor {floor} us) ===")
    print(f"{'kernel':<44}{'#/step':>7}{'us/k':>8}{'fixed':>8}{'work':>8}"
          f"{'gap/step':>9}{'REMOVE=':>9}")
    print("-" * 93)
    tot_remove = 0.0
    for r in rowsout:
        if r["cat"] != "moe_routing":
            continue
        fixed = min(r["d_mean"], floor) * r["per_step"]
        work = r["d_step"] - fixed
        remove = r["wall_step"]
        tot_remove += remove
        print(f"{r['name'][:43]:<44}{r['per_step']:>7.1f}{r['d_mean']:>8.2f}"
              f"{fixed:>8.1f}{work:>8.1f}{r['g_step']:>9.1f}{remove:>9.1f}")
    print(f"{'total routing if every launch vanished':<44}{'':>7}{'':>8}{'':>8}{'':>8}"
          f"{'':>9}{tot_remove:>9.1f}  ({100*tot_remove/wall:.2f}% of step)")

    # ---- one-layer ordered sequence ----
    k_rep = cand[len(cand) // 2]
    a0 = args.layers * k_rep
    la, lb = starts[a0 + 20], starts[a0 + 21]
    seq = cur.execute(
        f"SELECT start, end, shortName, streamId, gridX, blockX FROM {K} "
        "WHERE deviceId=? AND start >= ? AND start < ? ORDER BY start",
        (dev, la, lb)).fetchall()
    print(f"\n--- one layer (device {dev}, step {k_rep}): "
          f"{(lb-la)/1e3:.2f} us, {len(seq)} kernels ---")
    prev = la
    for s, e, sn, st, gx, bx in seq:
        print(f"  +{(s-la)/1e3:7.2f}  dur={(e-s)/1e3:6.2f}  gap={(s-prev)/1e3:5.2f}  "
              f"grid={gx:5d} blk={bx:4d} str={st:<5} {names[sn][:58]}")
        prev = max(prev, e)


if __name__ == "__main__":
    main()
