#!/usr/bin/env python3
"""QWEN-016 decision gate: quantify exposed NVLS EP comm in a decode step.

Anchors on the per-layer AllGather-V launches of the BS256 decode CUDA graph,
extracts one steady-state decode step per device, and reports:

  * forward-pass period (cross-check against TPOT),
  * GPU-busy interval union vs idle,
  * comm kernel Σ-duration, comm-only union, and the part of comm that is
    genuinely exposed (no non-comm kernel running concurrently),
  * per-collective duration distribution (AGV vs RSV) and per-device skew,
  * the ordered kernel sequence of one MoE layer, with inter-kernel gaps, so the
    serial critical path is visible,
  * measured bytes moved per collective vs the NVLink ceiling.

Usage: python3 analyze_comm.py <profile.sqlite> [--device N]
"""
import argparse
import sqlite3
import statistics

K = "CUPTI_ACTIVITY_KIND_KERNEL"
COMM_PATTERNS = ("multimem_all_gatherv", "multimem_reduce_scatter", "nccl")


def union(intervals):
    if not intervals:
        return 0
    ivs = sorted(intervals)
    total = 0
    cs, ce = ivs[0]
    for s, e in ivs[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + ce - cs


def subtract(intervals, holes):
    """Total length of `intervals` (already a union) not covered by `holes`."""
    if not intervals:
        return 0
    hs = sorted(holes)
    total = 0
    for s, e in intervals:
        cur = s
        for hstart, hend in hs:
            if hend <= cur:
                continue
            if hstart >= e:
                break
            if hstart > cur:
                total += min(hstart, e) - cur
            cur = max(cur, hend)
            if cur >= e:
                break
        if cur < e:
            total += e - cur
    return total


def to_union_list(intervals):
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
    ap.add_argument("--devices", default="0,1,2,3")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--dump-layer-device", type=int, default=3)
    ap.add_argument("--stream", type=int, default=None,
                    help="force the decode CUDA-graph stream (else the busiest, "
                         "which may be a warmup/client-wait stream)")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    cur = con.cursor()

    names = {i: v for i, v in cur.execute("SELECT id, value FROM StringIds")}

    # Locate the BS256 decode graph: the (device, stream) carrying the most
    # AllGather-V launches, which is the graph replayed every decode step.
    agv_id = [i for i, v in names.items() if "multimem_all_gatherv" in v]
    rsv_id = [i for i, v in names.items() if "multimem_reduce_scatter" in v]
    print(f"AGV shortName ids: {[(i, names[i]) for i in agv_id]}")
    print(f"RSV shortName ids: {[(i, names[i]) for i in rsv_id]}")

    agv_set = ",".join(str(i) for i in agv_id)
    q = (f"SELECT deviceId, streamId, COUNT(*) FROM {K} WHERE shortName IN ({agv_set}) "
         "GROUP BY deviceId, streamId ORDER BY COUNT(*) DESC")
    rows = cur.execute(q).fetchall()
    print("\nAGV launches per (device, stream):")
    for r in rows[:10]:
        print("   ", r)
    decode_stream = args.stream if args.stream is not None else rows[0][1]
    nsteps = max(c for _, s, c in rows if s == decode_stream) // args.layers
    print(f"\ndecode stream = {decode_stream}, steps = {nsteps}")

    devices = [int(d) for d in args.devices.split(",")]
    summary = {}

    for dev in devices:
        starts = [r[0] for r in cur.execute(
            f"SELECT start FROM {K} WHERE shortName IN ({agv_set}) AND deviceId=? "
            "AND streamId=? ORDER BY start", (dev, decode_stream))]
        n = len(starts) // args.layers
        # steady state: middle of the decode region
        periods = [starts[args.layers * (k + 1)] - starts[args.layers * k]
                   for k in range(n - 1)]
        mid = n // 2
        window = [k for k in range(max(2, n // 4), min(n - 2, 3 * n // 4))]
        med_period = statistics.median([periods[k] for k in window])
        # representative step = the one whose period is closest to the median
        k_rep = min(window, key=lambda k: abs(periods[k] - med_period))
        t0 = starts[args.layers * k_rep]
        t1 = starts[args.layers * (k_rep + 1)]

        ks = cur.execute(
            f"SELECT start, end, shortName, streamId, gridX, blockX FROM {K} "
            "WHERE deviceId=? AND start >= ? AND start < ? ORDER BY start",
            (dev, t0, t1)).fetchall()

        comm = [(s, e, sn, st) for s, e, sn, st in
                [(a, b, c, d_) for a, b, c, d_, _, _ in ks]
                if any(p in names[sn] for p in COMM_PATTERNS)]
        noncomm = [(s, e) for s, e, sn, _, _, _ in ks
                   if not any(p in names[sn] for p in COMM_PATTERNS)]

        busy = union([(s, e) for s, e, *_ in ks])
        comm_u = to_union_list([(s, e) for s, e, _, _ in comm])
        noncomm_u = to_union_list(noncomm)
        comm_sum = sum(e - s for s, e, _, _ in comm)
        comm_union = sum(e - s for s, e in comm_u)
        comm_exposed = subtract(comm_u, noncomm_u)

        agv = [e - s for s, e, sn, _ in comm if "gatherv" in names[sn]]
        rsv = [e - s for s, e, sn, _ in comm if "reduce_scatter" in names[sn]]

        summary[dev] = dict(
            period=t1 - t0, busy=busy, idle=(t1 - t0) - busy, nkern=len(ks),
            comm_sum=comm_sum, comm_union=comm_union, comm_exposed=comm_exposed,
            agv=agv, rsv=rsv, k_rep=k_rep)

        print(f"\n=== device {dev}  step #{k_rep} of {n} ===")
        print(f"forward-pass period : {(t1-t0)/1e3:9.1f} us")
        print(f"GPU busy (union)    : {busy/1e3:9.1f} us   "
              f"idle {(t1-t0-busy)/1e3:.1f} us ({100*(t1-t0-busy)/(t1-t0):.1f}%)")
        print(f"kernels in step     : {len(ks)}")
        print(f"comm  sum-durations : {comm_sum/1e3:9.1f} us  ({len(comm)} kernels)")
        print(f"comm  union         : {comm_union/1e3:9.1f} us")
        print(f"comm  EXPOSED       : {comm_exposed/1e3:9.1f} us  "
              f"({100*comm_exposed/(t1-t0):.1f}% of the step, "
              f"{100*comm_exposed/max(1,comm_union):.1f}% of comm union)")
        for label, vals in (("AGV", agv), ("RSV", rsv)):
            if vals:
                vals_s = sorted(vals)
                print(f"  {label}: n={len(vals):3d} "
                      f"sum={sum(vals)/1e3:7.1f} us  mean={statistics.mean(vals)/1e3:6.2f} "
                      f"p10={vals_s[len(vals)//10]/1e3:6.2f} "
                      f"med={statistics.median(vals)/1e3:6.2f} "
                      f"p90={vals_s[9*len(vals)//10]/1e3:6.2f} "
                      f"max={max(vals)/1e3:6.2f} us")

        # Inter-kernel gap accounting: in a single-stream full-iteration CUDA
        # graph the idle time is the sum of node-dispatch gaps, so it scales
        # with kernel count. Split it into per-kernel gaps vs a few large ones.
        merged = to_union_list([(s, e) for s, e, *_ in ks])
        gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
        gaps.append(t1 - merged[-1][1])
        gaps.sort()
        small = [g for g in gaps if g < 5000]
        big = [g for g in gaps if g >= 5000]
        print(f"  gaps: n={len(gaps)}  total={sum(gaps)/1e3:.1f} us | "
              f"<5us: n={len(small)} sum={sum(small)/1e3:.1f} us "
              f"median={statistics.median(small)/1e3:.2f} us | "
              f">=5us: n={len(big)} sum={sum(big)/1e3:.1f} us")

        if dev == args.dump_layer_device:
            # Dump the ordered kernel sequence spanning one MoE layer:
            # from the 21st AGV to the 22nd AGV of this step.
            a_idx = args.layers * k_rep
            la, lb = starts[a_idx + 20], starts[a_idx + 21]
            seq = cur.execute(
                f"SELECT start, end, shortName, streamId, gridX, blockX FROM {K} "
                "WHERE deviceId=? AND start >= ? AND start < ? ORDER BY start",
                (dev, la, lb)).fetchall()
            print(f"\n--- one layer on device {dev} "
                  f"({(lb-la)/1e3:.2f} us, {len(seq)} kernels) ---")
            prev_end = la
            for s, e, sn, st, gx, bx in seq:
                gap = s - prev_end
                print(f"  +{(s-la)/1e3:8.2f} us  dur={(e-s)/1e3:7.2f} "
                      f"gap={gap/1e3:6.2f}  grid={gx:5d} blk={bx:5d} "
                      f"str={st:4d}  {names[sn][:60]}")
                prev_end = max(prev_end, e)

    print("\n\n================ cross-device summary ================")
    print(f"{'dev':>3} {'period_us':>10} {'busy_us':>9} {'idle_us':>9} "
          f"{'comm_sum':>9} {'comm_exp':>9} {'AGV_mean':>9} {'RSV_mean':>9}")
    for dev, s in summary.items():
        print(f"{dev:>3} {s['period']/1e3:10.1f} {s['busy']/1e3:9.1f} "
              f"{s['idle']/1e3:9.1f} {s['comm_sum']/1e3:9.1f} "
              f"{s['comm_exposed']/1e3:9.1f} "
              f"{statistics.mean(s['agv'])/1e3:9.2f} "
              f"{statistics.mean(s['rsv'])/1e3:9.2f}")

    # Bytes / NVLink roofline for the measured shapes.
    print("\n================ byte-movement floor ================")
    local_tokens, hidden, topk = 64, 2048, 8
    agv_bytes = local_tokens * (hidden * 2 + topk * 8 + topk * 4)
    rsv_bytes = local_tokens * hidden * 4
    for bw_tbs in (0.9, 1.8):
        print(f"  at {bw_tbs} TB/s per-GPU NVLink: "
              f"AGV {agv_bytes} B -> {agv_bytes/(bw_tbs*1e12)*1e6:.3f} us, "
              f"RSV {rsv_bytes} B -> {rsv_bytes/(bw_tbs*1e12)*1e6:.3f} us")
    print(f"  per step (48 layers) at 0.9 TB/s: "
          f"{48*(agv_bytes+rsv_bytes)/(0.9e12)*1e6:.1f} us")


if __name__ == "__main__":
    main()
