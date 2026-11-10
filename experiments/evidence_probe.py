#!/usr/bin/env python3
"""Show what the reader is given for a question: each record's page, trust label and whether it holds a string.

    python experiments/evidence_probe.py "question" [needle ...]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "solution")]
from run_eval import load_snapshots  # noqa: E402
import solution  # noqa: E402
from reader import Reader, _trust  # noqa: E402

q, needles = sys.argv[1], sys.argv[2:]
ks = solution.build()
for s in load_snapshots(ROOT / "snapshots"):
    ks.ingest(s)
snap = ks.store.snapshot_for(None)
recs = Reader().evidence(ks.store, snap, q)
print(f"{len(recs)} records, {sum(len(c.text) for c in recs)} chars")
for i, c in enumerate(recs, 1):
    hit = [n for n in needles if n.lower() in c.text.lower()]
    print(f"[{i:2d}] {c.kind:4s} {_trust(c)[:9]:9s} {c.url[23:85]:62s} {c.text[:50]!r} {'<== ' + ','.join(hit) if hit else ''}")
missing = [n for n in needles if not any(n.lower() in c.text.lower() for c in recs)]
print("not in evidence:", missing or "-")
