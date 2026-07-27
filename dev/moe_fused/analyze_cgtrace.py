#!/usr/bin/env python3
"""Control analysis for `--cuda-graph-trace=node` vs `=graph`.

HOSTGAP-S6 attributed ~900 us/step of the decode step to CUDA-graph machinery
(a 199 us median `cudaGraphLaunch` plus inter-node dispatch gaps) but could not
say whether that cost is real or an artifact of CUPTI instrumenting all 1158
graph nodes. This compares two otherwise-identical captures and reports, for a
steady-state decode window on one device:

  * step period            inter-arrival of graph launches
  * host cudaGraphLaunch   the submit cost HOSTGAP-S6 measured at 199 us
  * GPU busy per step      union of kernel intervals (node) or the graph
                           execution range (graph)

If the graph-mode step period and launch cost match node mode, the machinery is
real and the graph-node-count lever is live. If they collapse, node mode was
inflating them and the lever is an instrumentation artifact.

Usage:
    python dev/moe_fused/analyze_cgtrace.py <node.sqlite> <graph.sqlite>
"""

import sqlite3
import statistics
import sys

NS_PER_US = 1000.0
# Skip the load/warmup/prefill head and the client-wait tail; the BS256 OSL128
# decode loop is the long periodic run in the middle.
TRIM_FRACTION = 0.25


def _table_exists(conn, name):
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _launch_starts(conn, tid):
    """Host-side `cudaGraphLaunch` calls on one rank: (start_ns, dur_ns)."""
    return conn.execute(
        """
        SELECT r.start, r.end - r.start
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id = r.nameId
        WHERE s.value LIKE 'cudaGraphLaunch%' AND r.globalTid = ?
        ORDER BY r.start
        """,
        (tid,),
    ).fetchall()


def _busiest_launch_tid(conn):
    """The four ranks share the trace; step period is per-rank, so pick one."""
    return conn.execute(
        """
        SELECT r.globalTid, COUNT(*) c
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id = r.nameId
        WHERE s.value LIKE 'cudaGraphLaunch%'
        GROUP BY r.globalTid ORDER BY c DESC LIMIT 1
        """
    ).fetchone()[0]


def _busiest_device(conn):
    row = conn.execute(
        "SELECT deviceId, COUNT(*) c FROM CUPTI_ACTIVITY_KIND_KERNEL"
        " GROUP BY deviceId ORDER BY c DESC LIMIT 1"
    ).fetchone()
    return row[0]


def _busy_union(rows):
    if not rows:
        return 0
    total = 0
    cur_s, cur_e = rows[0]
    for s, e in rows[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s)


def _steady_window(starts):
    """Trim the load/warmup head and client-wait tail, then keep the periodic run.

    Returns the kept inter-arrivals and the launch starts that bound them, so
    the caller can slice per-step GPU work between consecutive launches.
    """
    if len(starts) < 8:
        raise SystemExit("too few graph launches to find a steady window")
    lo = int(len(starts) * TRIM_FRACTION)
    hi = int(len(starts) * (1.0 - TRIM_FRACTION))
    window = starts[lo:hi]
    deltas = [b - a for a, b in zip(window, window[1:])]
    median = statistics.median(deltas)
    kept, boundaries = [], []
    for a, b in zip(window, window[1:]):
        if 0.5 * median <= b - a <= 2.0 * median:
            kept.append(b - a)
            boundaries.append(a)
    boundaries.append(window[-1])
    return kept, boundaries


def _busy_per_step(conn, device, boundaries):
    """Median union of kernel intervals between consecutive graph launches."""
    per_step = []
    n_kern = 0
    for t0, t1 in zip(boundaries, boundaries[1:]):
        rows = conn.execute(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"
            " WHERE deviceId = ? AND start >= ? AND start < ? ORDER BY start",
            (device, t0, t1),
        ).fetchall()
        n_kern += len(rows)
        per_step.append(_busy_union(rows))
    if not per_step:
        return None, 0, 0
    return statistics.median(per_step), n_kern / len(per_step), len(per_step)


def _graph_exec_durations(conn, device, t0, t1):
    rows = conn.execute(
        "SELECT end - start FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
        " WHERE deviceId = ? AND start >= ? AND end <= ?",
        (device, t0, t1),
    ).fetchall()
    return [r[0] for r in rows]


def analyze(path, label):
    conn = sqlite3.connect(path)
    device = _busiest_device(conn)
    tid = _busiest_launch_tid(conn)
    launches = _launch_starts(conn, tid)
    deltas, boundaries = _steady_window([r[0] for r in launches])
    t0, t1 = boundaries[0], boundaries[-1]

    durs = [d for s, d in launches if t0 <= s <= t1]
    busy, kern_per_step, n_steps = _busy_per_step(conn, device, boundaries)

    print(f"--- {label}  ({path.rsplit('/', 2)[-2]}) ---")
    print(f"  device / launch tid     : {device} / {tid}")
    print(f"  steady decode steps     : {n_steps}")
    print(
        f"  step period             : {statistics.median(deltas) / NS_PER_US:8.1f} us"
        f"   (mean {statistics.mean(deltas) / NS_PER_US:.1f})"
    )
    print(
        f"  host cudaGraphLaunch    : {statistics.median(durs) / NS_PER_US:8.1f} us"
        f"   (mean {statistics.mean(durs) / NS_PER_US:.1f}, n={len(durs)})"
    )
    if busy is not None:
        print(
            f"  GPU busy / step (union) : {busy / NS_PER_US:8.1f} us"
            f"   ({kern_per_step:.0f} traced kernel rows/step)"
        )
    if _table_exists(conn, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
        g = _graph_exec_durations(conn, device, t0, t1)
        if g:
            print(
                f"  graph exec duration     : {statistics.median(g) / NS_PER_US:8.1f} us"
                f"   (n={len(g)})"
            )
    conn.close()
    return statistics.median(deltas), statistics.median(durs)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    period_node, launch_node = analyze(sys.argv[1], "cuda-graph-trace=node")
    print()
    period_graph, launch_graph = analyze(sys.argv[2], "cuda-graph-trace=graph")

    print("\n=== control verdict ===")
    d_period = (period_node - period_graph) / NS_PER_US
    d_launch = (launch_node - launch_graph) / NS_PER_US
    print(
        f"  step period       node − graph = {d_period:+8.1f} us"
        f"  ({100 * d_period * NS_PER_US / period_node:+.2f}% of the node step)"
    )
    print(f"  cudaGraphLaunch   node − graph = {d_launch:+8.1f} us")
    print()
    if abs(d_period) < 0.03 * period_node / NS_PER_US:  # 3% of the step
        print("  Step period is unchanged by the trace mode ⇒ the per-step graph")
        print("  machinery HOSTGAP-S6 measured is REAL, not CUPTI overhead.")
    else:
        print("  Step period moves with the trace mode ⇒ node-mode instrumentation")
        print("  inflated it; HOSTGAP-S6's graph-machinery figure is an artifact.")


if __name__ == "__main__":
    main()
