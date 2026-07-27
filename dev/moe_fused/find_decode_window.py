#!/usr/bin/env python3
"""Locate the steady-state decode window in an nsys sqlite trace.

`analyze_hostgaps.py` needs an explicit `--window t0,t1`. The decode loop is the
longest run of near-constant intervals between successive launches of the
once-per-step LM-head GEMM, so find that run and print the window in the form
the other script expects.

Usage:
    python dev/moe_fused/find_decode_window.py <trace>.sqlite --device 3
"""

import argparse
import sqlite3
import statistics
import sys

ANCHOR = "nvjet_sm100_tst_512x64_64x3_2x1_2cta_v_bz_TNT"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sqlite")
    p.add_argument("--device", type=int, default=3)
    p.add_argument("--anchor", default=ANCHOR)
    p.add_argument("--lo-ms", type=float, default=4.0)
    p.add_argument("--hi-ms", type=float, default=20.0)
    args = p.parse_args()

    con = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    rows = con.execute(
        """
        SELECT k.start
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.demangledName
        WHERE k.deviceId = ? AND s.value LIKE ?
        ORDER BY k.start
        """,
        (args.device, f"%{args.anchor}%"),
    ).fetchall()
    starts = [r[0] for r in rows]
    if len(starts) < 10:
        sys.exit(f"ERROR: only {len(starts)} anchor launches found for {args.anchor!r}")

    lo, hi = args.lo_ms * 1e6, args.hi_ms * 1e6
    best = cur = None
    for i in range(1, len(starts)):
        d = starts[i] - starts[i - 1]
        if lo <= d <= hi:
            cur = (cur[0], i) if cur else (i - 1, i)
            if best is None or cur[1] - cur[0] > best[1] - best[0]:
                best = cur
        else:
            cur = None
    if best is None:
        sys.exit("ERROR: no periodic anchor run found")

    i0, i1 = best
    seg = starts[i0 : i1 + 1]
    diffs = [b - a for a, b in zip(seg, seg[1:])]
    print(f"trace   : {args.sqlite}")
    print(f"anchor  : {args.anchor}  launches {len(starts)}")
    print(f"steps   : {len(diffs)}  period median {statistics.median(diffs) / 1e6:.3f} ms")
    print(f"window  : {seg[0] / 1e9:.4f},{seg[-1] / 1e9:.4f}")


if __name__ == "__main__":
    main()
