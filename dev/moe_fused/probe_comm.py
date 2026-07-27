#!/usr/bin/env python3
"""Probe an nsys sqlite for the decode-step comm structure.

Stage 1 of the QWEN-016 exposed-EP-comm decision gate: list the kernels present,
their per-device counts/durations, and the stream layout, so the follow-up
analysis can be built on measured names rather than guessed ones.

Usage: python3 probe_comm.py <profile.sqlite>
"""
import sqlite3
import sys

db = sys.argv[1]
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
cur = con.cursor()

tables = [r[0] for r in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
print("tables with KERNEL/RUNTIME/GRAPH:",
      [t for t in tables if any(k in t.upper()
                                for k in ("KERNEL", "RUNTIME", "GRAPH", "NVTX"))])

K = "CUPTI_ACTIVITY_KIND_KERNEL"
cols = [r[1] for r in cur.execute(f"PRAGMA table_info({K})")]
print("\nkernel cols:", cols)

print("\n--- global extent ---")
print(cur.execute(f"SELECT MIN(start), MAX(end), COUNT(*) FROM {K}").fetchone())

print("\n--- per device kernel count / total dur (ns) ---")
for row in cur.execute(
        f"SELECT deviceId, COUNT(*), SUM(end-start) FROM {K} GROUP BY deviceId"):
    print(row)

print("\n--- top 40 kernels by total time (all devices) ---")
q = f"""
SELECT s.value AS name, COUNT(*) AS n, SUM(k.end-k.start) AS tot,
       AVG(k.end-k.start) AS avg
FROM {K} k JOIN StringIds s ON k.shortName = s.id
GROUP BY s.value ORDER BY tot DESC LIMIT 40
"""
for name, n, tot, avg in cur.execute(q):
    print(f"{tot/1e6:12.3f} ms  n={n:8d}  avg={avg/1e3:9.3f} us  {name[:90]}")

print("\n--- candidate comm kernels (name match) ---")
pat = ("multimem", "gatherv", "reduce_scatter", "symm", "metadata", "nccl", "all_gather")
q2 = f"""
SELECT s.value AS name, COUNT(*) AS n, SUM(k.end-k.start) AS tot,
       AVG(k.end-k.start) AS avg, k.deviceId, k.streamId
FROM {K} k JOIN StringIds s ON k.shortName = s.id
GROUP BY s.value, k.deviceId, k.streamId ORDER BY tot DESC
"""
for name, n, tot, avg, dev, stream in cur.execute(q2):
    if any(p in name.lower() for p in pat):
        print(f"dev={dev} stream={stream} {tot/1e6:10.3f} ms n={n:7d} "
              f"avg={avg/1e3:8.3f} us  {name[:80]}")

print("\n--- streams per device ---")
for row in cur.execute(
        f"SELECT deviceId, streamId, COUNT(*), SUM(end-start) FROM {K} "
        "GROUP BY deviceId, streamId ORDER BY deviceId, SUM(end-start) DESC"):
    print(row)
