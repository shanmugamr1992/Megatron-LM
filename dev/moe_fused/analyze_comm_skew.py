#!/usr/bin/env python3
"""QWEN-016: separate inter-rank arrival skew from intrinsic collective cost.

All four EP ranks are profiled by one nsys instance on one node, so their kernel
timestamps share a timebase. Both NVLS collectives end with (AGV) or begin with
(RSV) a barrier that every rank must reach, so for a given (step, layer):

    duration[rank] = release_time - launch_time[rank]

and the release time is common. Therefore

    intrinsic = release - max(launch)      # what the collective actually costs
    skew[rank] = max(launch) - launch[rank]  # time this rank spent waiting

Reports both, aggregated over a range of steady-state decode steps.

Usage: python3 analyze_comm_skew.py <profile.sqlite> --stream 257
"""
import argparse
import sqlite3
import statistics

K = "CUPTI_ACTIVITY_KIND_KERNEL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--stream", type=int, required=True)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--first-step", type=int, default=32)
    ap.add_argument("--last-step", type=int, default=96)
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    cur = con.cursor()
    names = {i: v for i, v in cur.execute("SELECT id, value FROM StringIds")}

    def ids(sub):
        return ",".join(str(i) for i, v in names.items() if sub in v)

    for label, kid in (("AGV", ids("multimem_all_gatherv")),
                       ("RSV", ids("multimem_reduce_scatter"))):
        per_dev = {}
        for dev in (0, 1, 2, 3):
            per_dev[dev] = cur.execute(
                f"SELECT start, end FROM {K} WHERE shortName IN ({kid}) "
                "AND deviceId=? AND streamId=? ORDER BY start",
                (dev, args.stream)).fetchall()
        n = min(len(v) for v in per_dev.values())
        lo = args.first_step * args.layers
        hi = min(n, args.last_step * args.layers)

        intrinsic, skews, durs, exit_spread = [], {d: [] for d in per_dev}, [], []
        for i in range(lo, hi):
            launches = {d: per_dev[d][i][0] for d in per_dev}
            ends = {d: per_dev[d][i][1] for d in per_dev}
            last_in = max(launches.values())
            release = statistics.median(ends.values())
            intrinsic.append(release - last_in)
            exit_spread.append(max(ends.values()) - min(ends.values()))
            for d in per_dev:
                skews[d].append(last_in - launches[d])
                durs.append(ends[d] - launches[d])

        print(f"\n=== {label}  ({hi-lo} collectives per rank, "
              f"steps {args.first_step}..{args.last_step}) ===")
        print(f"  mean duration over all ranks : {statistics.mean(durs)/1e3:7.3f} us")
        print(f"  intrinsic (release - last arrival):"
              f" mean {statistics.mean(intrinsic)/1e3:7.3f} us  "
              f"median {statistics.median(intrinsic)/1e3:7.3f}  "
              f"p10 {sorted(intrinsic)[len(intrinsic)//10]/1e3:7.3f}")
        print(f"  exit spread across ranks     : "
              f"mean {statistics.mean(exit_spread)/1e3:7.3f} us  "
              f"median {statistics.median(exit_spread)/1e3:7.3f}")
        for d in per_dev:
            print(f"  rank {d} arrival skew (waiting): "
                  f"mean {statistics.mean(skews[d])/1e3:7.3f} us  "
                  f"median {statistics.median(skews[d])/1e3:7.3f}")
        tot_int = statistics.mean(intrinsic) * args.layers
        tot_dur = statistics.mean(durs) * args.layers
        print(f"  per step (x{args.layers}): total {tot_dur/1e3:7.1f} us = "
              f"intrinsic {tot_int/1e3:7.1f} us + skew "
              f"{(tot_dur-tot_int)/1e3:7.1f} us")


if __name__ == "__main__":
    main()
