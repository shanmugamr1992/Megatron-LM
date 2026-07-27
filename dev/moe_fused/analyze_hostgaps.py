# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Bracket and attribute inter-kernel host gaps in an nsys decode trace.

Stage 1 (`gaps`)  : per decode step, compute the complement of the kernel
                    interval union and bracket each gap by the kernel that ends
                    at its start and the kernel that begins at its end.
Stage 2 (`attrib`): for one bracketed gap class, aggregate what the host is
                    doing inside those windows -- CUDA runtime API calls, OSRT
                    calls, and native sampling backtraces.

Works on GPU-only traces (stage 1 only) and host-visibility traces (both).
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
from collections import defaultdict


def con_ro(path):
    return sqlite3.connect("file:" + path + "?mode=ro", uri=True)


def table_exists(con, name):
    return (
        con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def kernels(con, dev, t0, t1):
    return con.execute(
        """SELECT k.start, k.end,
                  COALESCE((SELECT value FROM StringIds WHERE id=k.demangledName),'')
           FROM CUPTI_ACTIVITY_KIND_KERNEL k
           WHERE k.deviceId=? AND k.start>=? AND k.start<?
           ORDER BY k.start""",
        (dev, t0, t1),
    ).fetchall()


def short(name, n=52):
    """Collapse a C++ template kernel name to a readable identifier."""
    s = name
    for pre in ("void ", "at::native::", "<unnamed>::", "cutlass::", "flashinfer::"):
        s = s.replace(pre, "")
    s = s.split("<")[0].split("(")[0]
    s = s.strip()
    return s[:n] if s else name[:n]


def find_anchor(ks, forced):
    per = defaultdict(list)
    for s, _e, n in ks:
        per[n].append(s)
    best = None
    for n, ss in per.items():
        if len(ss) < 8:
            continue
        g = [b - a for a, b in zip(ss, ss[1:])]
        m = statistics.median(g)
        if m <= 0:
            continue
        cv = statistics.pstdev(g) / m
        if forced:
            if forced.lower() not in n.lower():
                continue
        elif cv > 0.6:
            continue
        if best is None or m > best[1]:
            best = (n, m, cv, sorted(ss))
    if best is None:
        sys.exit("ERROR: no anchor found; pass --anchor")
    return best


def step_gaps(ks, t0, t1):
    """Return (busy_ns, [(gap_start, gap_end, prev_name, next_name)])."""
    iv = sorted((s, e) for s, e, _ in ks)
    if not iv:
        return 0, []
    # name lookup by exact end / exact start
    ends = defaultdict(list)
    starts = defaultdict(list)
    for s, e, n in ks:
        ends[e].append(n)
        starts[s].append(n)
    merged = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    busy = sum(e - s for s, e in merged)
    gaps = []
    for (a_s, a_e), (b_s, _b_e) in zip(merged, merged[1:]):
        pn = ends.get(a_e, ["?"])[-1]
        nn = starts.get(b_s, ["?"])[0]
        gaps.append((a_e, b_s, pn, nn))
    return busy, gaps


def cmd_gaps(args):
    con = con_ro(args.sqlite)
    t0, t1 = (int(float(x) * 1e9) for x in args.window.split(","))
    ks = kernels(con, args.device, t0, t1)
    if not ks:
        sys.exit("ERROR: no kernels in window on that device")
    aname, period, acv, astarts = find_anchor(ks, args.anchor)
    astarts = [s for s in astarts if t0 <= s < t1]
    nstep = len(astarts) - 1
    print("trace   : %s" % args.sqlite)
    print("device  : %d   window %.3f-%.3f s   kernels %d" % (args.device, t0 / 1e9, t1 / 1e9, len(ks)))
    print("anchor  : %s" % short(aname, 70))
    print("period  : median %.3f ms  cv %.3f   steps %d" % (period / 1e6, acv, nstep))

    tot_wall = tot_busy = 0
    agg = defaultdict(list)
    kern_per_step = []
    for a, b in zip(astarts, astarts[1:]):
        sk = [k for k in ks if a <= k[0] < b]
        busy, gaps = step_gaps(sk, a, b)
        tot_wall += b - a
        tot_busy += busy
        kern_per_step.append(len(sk))
        for gs, ge, pn, nn in gaps:
            agg[(short(pn, args.namelen), short(nn, args.namelen))].append(ge - gs)

    print("\nper step: wall %.3f ms   GPU-busy %.3f ms   idle %.3f ms (%.1f%%)   kernels %.0f"
          % (tot_wall / nstep / 1e6, tot_busy / nstep / 1e6,
             (tot_wall - tot_busy) / nstep / 1e6,
             100.0 * (tot_wall - tot_busy) / tot_wall,
             sum(kern_per_step) / len(kern_per_step)))

    rows = []
    for (pn, nn), v in agg.items():
        rows.append((sum(v) / nstep, len(v) / nstep, statistics.median(v), sum(v) / len(v), pn, nn))
    rows.sort(reverse=True)
    print("\n%-10s %7s %9s %9s  %s" % ("us/step", "n/step", "med_us", "mean_us", "prev_kernel -> next_kernel"))
    for tot, n, med, mean, pn, nn in rows[: args.top]:
        print("%10.1f %7.2f %9.2f %9.2f  %s -> %s" % (tot / 1e3, n, med / 1e3, mean / 1e3, pn, nn))
    idle_us = (tot_wall - tot_busy) / nstep / 1e3
    listed_us = sum(r[0] for r in rows[: args.top]) / 1e3
    print("\ntotal idle/step = %.1f us ; top-%d listed = %.1f us (%.1f%% of idle)"
          % (idle_us, args.top, listed_us, 100.0 * listed_us / idle_us))


def gap_instances(con, dev, t0, t1, anchor, prev, nxt, lo_us, hi_us):
    ks = kernels(con, dev, t0, t1)
    aname, period, acv, astarts = find_anchor(ks, anchor)
    astarts = [s for s in astarts if t0 <= s < t1]
    out = []
    nstep = len(astarts) - 1
    for a, b in zip(astarts, astarts[1:]):
        _busy, gaps = step_gaps([k for k in ks if a <= k[0] < b], a, b)
        for gs, ge, pn, nn in gaps:
            d = (ge - gs) / 1e3
            if prev.lower() in pn.lower() and nxt.lower() in nn.lower() and lo_us <= d <= hi_us:
                out.append((gs, ge))
    return out, nstep, period


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def cmd_attrib(args):
    con = con_ro(args.sqlite)
    t0, t1 = (int(float(x) * 1e9) for x in args.window.split(","))
    wins, nstep, period = gap_instances(
        con, args.device, t0, t1, args.anchor, args.prev, args.next, args.min_us, args.max_us
    )
    if not wins:
        sys.exit("ERROR: no gap instances matched")
    durs = [(b - a) / 1e3 for a, b in wins]
    durs_s = sorted(durs)
    print("=" * 100)
    print("GAP  %s -> %s   [%.0f, %.0f] us" % (args.prev, args.next, args.min_us, args.max_us))
    print("device %d  window %.3f-%.3f s  steps %d  step period %.3f ms" % (args.device, t0 / 1e9, t1 / 1e9, nstep, period / 1e6))
    print("instances %d (%.2f/step)  median %.1f us  mean %.1f us  p10 %.1f  p90 %.1f  total %.1f us/step"
          % (len(wins), len(wins) / nstep, statistics.median(durs), sum(durs) / len(durs),
             durs_s[int(0.1 * len(durs_s))], durs_s[int(0.9 * len(durs_s))], sum(durs) / nstep))

    gpid = con.execute(
        "SELECT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=? AND start>=? AND start<? LIMIT 1",
        (args.device, t0, t1),
    ).fetchone()[0]
    tlo, thi = gpid, gpid + 2 ** 24
    W0, W1 = wins[0][0], wins[-1][1]
    tot_gap_ns = sum(b - a for a, b in wins)

    # The engine thread is the one emitting the NVTX decode ranges.
    main_gtid = con.execute(
        """SELECT globalTid, COUNT(*) c FROM NVTX_EVENTS
           WHERE globalTid>=? AND globalTid<? AND start>=? AND start<?
             AND (text IS NOT NULL OR textId IS NOT NULL)
           GROUP BY globalTid ORDER BY c DESC LIMIT 1""",
        (tlo, thi, W0, W1),
    ).fetchone()[0]
    print("engine thread: tid %d (of rank pid %d)" % (main_gtid & 0xFFFFFF, (gpid >> 24) & 0xFFFFFF))

    def bucket(rows, label, namefn):
        agg = defaultdict(lambda: [0, 0])
        for s, e, nm in rows:
            for a, b in wins:
                ov = overlap(s, e, a, b)
                if ov:
                    agg[namefn(nm)][0] += 1
                    agg[namefn(nm)][1] += ov
        rs = sorted(agg.items(), key=lambda kv: -kv[1][1])
        print("\n--- %s (overlap-clipped inside the gap windows) ---" % label)
        print("  %-46s %8s %11s %8s" % ("name", "n/gap", "us/gap", "%gap"))
        for n, (c, d) in rs[: args.top]:
            print("  %-46s %8.2f %11.2f %7.1f%%" % (n[:46], c / len(wins), d / 1e3 / len(wins), 100.0 * d / tot_gap_ns))
        cov = sum(v[1] for v in agg.values())
        print("  %-46s %8s %11.2f %7.1f%%" % ("TOTAL (may double-count nesting)", "", cov / 1e3 / len(wins), 100.0 * cov / tot_gap_ns))

    rt = con.execute(
        """SELECT r.start,r.end,COALESCE((SELECT value FROM StringIds WHERE id=r.nameId),'?')
           FROM CUPTI_ACTIVITY_KIND_RUNTIME r
           WHERE r.globalTid>=? AND r.globalTid<? AND r.end>=? AND r.start<?""",
        (tlo, thi, W0, W1),
    ).fetchall()
    bucket(rt, "CUDA runtime API (CUPTI_ACTIVITY_KIND_RUNTIME)", lambda x: x)

    os_ = con.execute(
        """SELECT o.start,o.end,COALESCE((SELECT value FROM StringIds WHERE id=o.nameId),'?')
           FROM OSRT_API o WHERE o.globalTid=? AND o.end>=? AND o.start<?""",
        (main_gtid, W0, W1),
    ).fetchall()
    bucket(os_, "OS runtime (OSRT_API) on the ENGINE THREAD only", lambda x: x)

    nv = con.execute(
        """SELECT e.start,COALESCE(e.end,e.start),COALESCE(e.text,(SELECT value FROM StringIds WHERE id=e.textId))
           FROM NVTX_EVENTS e WHERE e.globalTid>=? AND e.globalTid<? AND COALESCE(e.end,e.start)>=? AND e.start<?""",
        (tlo, thi, W0, W1),
    ).fetchall()
    bucket([r for r in nv if r[2]], "NVTX ranges", lambda x: x)

    # ---- native sampling backtraces ----
    ce = con.execute(
        "SELECT id,start,globalTid,threadState FROM COMPOSITE_EVENTS WHERE globalTid>=? AND globalTid<? AND start>=? AND start<?",
        (tlo, thi, W0, W1),
    ).fetchall()
    inwin = []
    for cid, st, gtid, tstate in ce:
        for a, b in wins:
            if a <= st < b:
                inwin.append((cid, gtid, tstate))
                break
    print("\n--- CPU sampling: %d samples land in the %d gap windows (%.1f/gap) ---" % (len(inwin), len(wins), len(inwin) / len(wins)))
    if not inwin:
        return
    bytid = defaultdict(int)
    for _c, g, _t in inwin:
        bytid[g & 0xFFFFFF] += 1
    tn = {}
    for r in con.execute("SELECT globalTid,COALESCE((SELECT value FROM StringIds WHERE id=nameId),'?') FROM ThreadNames WHERE globalTid>=? AND globalTid<?", (tlo, thi)):
        tn[r[0] & 0xFFFFFF] = r[1]
    print("  by thread: " + ", ".join("%d(%s)=%d" % (t, tn.get(t, "?")[:16], c) for t, c in sorted(bytid.items(), key=lambda kv: -kv[1])[:6]))
    ts = defaultdict(int)
    for _c, _g, t in inwin:
        ts[t] += 1
    print("  threadState histogram: %s" % dict(ts))

    main_tid = max(bytid, key=lambda k: bytid[k]) if args.tid is None else args.tid
    ids = [c for c, g, _t in inwin if (g & 0xFFFFFF) == main_tid]
    print("  attributing %d samples on thread %d (%s)" % (len(ids), main_tid, tn.get(main_tid, "?")))
    leaf = defaultdict(int)
    anyframe = defaultdict(int)
    for i in range(0, len(ids), 400):
        chunk = ids[i : i + 400]
        qm = ",".join("?" * len(chunk))
        rows = con.execute(
            """SELECT sc.id,sc.stackDepth,COALESCE((SELECT value FROM StringIds WHERE id=sc.symbol),'?'),
                      COALESCE((SELECT value FROM StringIds WHERE id=sc.module),'?'),sc.unresolved
               FROM SAMPLING_CALLCHAINS sc WHERE sc.id IN (%s)""" % qm,
            chunk,
        ).fetchall()
        per = defaultdict(list)
        for cid, sd, sym, mod, unres in rows:
            per[cid].append((sd, sym, mod, unres))
        for cid, fr in per.items():
            fr.sort()
            named = [(sd, sym, mod) for sd, sym, mod, unres in fr if not unres and sym != "[Broken backtraces]"]
            if named:
                leaf["%s  [%s]" % (named[0][1], named[0][2].split("/")[-1])] += 1
            seen = set()
            for _sd, sym, mod in named:
                k = "%s  [%s]" % (sym, mod.split("/")[-1])
                if k not in seen:
                    seen.add(k)
                    anyframe[k] += 1
    print("\n  deepest RESOLVED frame (leaf) -- top %d:" % args.top)
    for n, c in sorted(leaf.items(), key=lambda kv: -kv[1])[: args.top]:
        print("    %6.1f%%  %s" % (100.0 * c / max(1, len(ids)), n[:96]))
    print("\n  any-frame presence (how often a symbol appears anywhere in the stack) -- top %d:" % args.top)
    for n, c in sorted(anyframe.items(), key=lambda kv: -kv[1])[: args.top]:
        print("    %6.1f%%  %s" % (100.0 * c / max(1, len(ids)), n[:96]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    at = sub.add_parser("attrib")
    at.add_argument("sqlite")
    at.add_argument("--device", type=int, required=True)
    at.add_argument("--window", required=True)
    at.add_argument("--anchor", default=None)
    at.add_argument("--prev", required=True)
    at.add_argument("--next", required=True)
    at.add_argument("--min-us", type=float, default=0.0)
    at.add_argument("--max-us", type=float, default=1e9)
    at.add_argument("--top", type=int, default=12)
    at.add_argument("--tid", type=int, default=None)
    at.set_defaults(func=cmd_attrib)
    g = sub.add_parser("gaps")
    g.add_argument("sqlite")
    g.add_argument("--device", type=int, required=True)
    g.add_argument("--window", required=True, help="t0,t1 in seconds")
    g.add_argument("--anchor", default=None)
    g.add_argument("--top", type=int, default=20)
    g.add_argument("--namelen", type=int, default=46)
    g.set_defaults(func=cmd_gaps)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
