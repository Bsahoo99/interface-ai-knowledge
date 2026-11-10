#!/usr/bin/env python3
"""Run one variant of the system and log its numbers, so variants can be compared before/after.

    python experiments/run.py --variant solution --label "D: load-time (current)"
    python experiments/run.py --variant experiments/variants/query_time --label "A: question-time"
    python experiments/run.py --variant solution --scale 4 --label "D at 4x data"

Scoring uses the kit's own harness/scoring.py, so accuracy, consistency and speed match the grader.
Extra measurements: exact latencies (not rounded), first-answer latency, ingest time, peak memory.
Every run is appended to experiments/results.csv; --show prints the log as a table.
"""
from __future__ import annotations

import argparse
import csv
import json
import resource
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))

from contract import Snapshot  # noqa: E402
from run_eval import load_snapshots, load_submission, to_answer  # noqa: E402
from scoring import score_item, summarize  # noqa: E402

LOG = ROOT / "experiments" / "results.csv"
FIELDS = ["when", "label", "variant", "scale", "eval", "questions", "accuracy", "consistency",
          "speed_score", "score_no_conflicts", "source_hit_rate", "p50_ms", "p95_ms", "max_ms", "first_answer_ms",
          "ingest_s", "peak_mem_mb", "conflicts_found", "notes"]


def scaled(snaps: list[Snapshot], factor: int) -> list[Snapshot]:
    """Simulate a bigger website: each page gets (factor-1) extra copies under different URLs.

    The copies are real extra pages to the system (different URL, so not de-duplicated), which grows
    the number of records and the search index the way a larger site would.
    """
    if factor <= 1:
        return snaps
    out = []
    for s in snaps:
        pages = list(s.pages)
        for k in range(1, factor):
            pages += [replace(p, url=f"{p.url.rstrip('/')}/copy-{k}") for p in s.pages]
        out.append(Snapshot(s.snapshot_id, s.taken_on, pages))
    return out


def peak_mem_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 * 1024) if sys.platform == "darwin" else r / 1024, 1)   # bytes on macOS, KB on Linux


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=Path, help="folder containing solution.py")
    ap.add_argument("--label", default="")
    ap.add_argument("--eval", type=Path, default=ROOT / "eval" / "public_eval.jsonl")
    ap.add_argument("--snapshots", type=Path, default=ROOT / "snapshots")
    ap.add_argument("--scale", type=int, default=1, help="multiply the website size (synthetic copies)")
    ap.add_argument("--repeat-sample", type=int, default=20)
    ap.add_argument("--notes", default="")
    ap.add_argument("--show", action="store_true", help="print the results log and exit")
    args = ap.parse_args()

    if args.show:
        show()
        return

    items = [json.loads(l) for l in args.eval.read_text().splitlines() if l.strip()]
    snaps = scaled(load_snapshots(args.snapshots), args.scale)
    ks = load_submission((ROOT / args.variant).resolve() if not args.variant.is_absolute() else args.variant)

    t0 = time.perf_counter()
    for s in snaps:
        ks.ingest(s)
    ingest_s = time.perf_counter() - t0

    results, lat = [], []
    for it in items:
        t = time.perf_counter()
        ans = to_answer(ks.answer(it["question"], it.get("as_of")))
        dt = time.perf_counter() - t
        lat.append(dt)
        r = score_item(it, ans)
        r.update(latency_s=dt, error=None, answer={"text": ans.text})
        results.append(r)

    repeats = []
    for it in items[:: max(1, len(items) // args.repeat_sample)]:
        repeats.append({"id": it["id"], "text": to_answer(ks.answer(it["question"], it.get("as_of"))).text})

    try:
        n_conf = len(ks.conflicts())
    except Exception:
        n_conf = -1

    rep = summarize(results, None, repeats)
    xs = sorted(lat)
    pct = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]
    row = {
        "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "label": args.label, "variant": str(args.variant), "scale": args.scale, "eval": args.eval.name,
        "questions": len(items),
        "accuracy": round(rep["qa_score"], 4),
        "consistency": rep["consistency"]["score"],
        "speed_score": rep["speed"]["score"],
        "score_no_conflicts": rep["final_score"],
        "source_hit_rate": round(sum(1 for r in results if r["source_hit"]) /
                                 max(1, sum(1 for r in results if r["source_hit"] is not None)), 3),
        "p50_ms": round(pct(0.5) * 1000, 1), "p95_ms": round(pct(0.95) * 1000, 1),
        "max_ms": round(xs[-1] * 1000, 1), "first_answer_ms": round(lat[0] * 1000, 1),
        "ingest_s": round(ingest_s, 2), "peak_mem_mb": peak_mem_mb(),
        "conflicts_found": n_conf, "notes": args.notes,
    }
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row, indent=2))
    print("by category:", {k: v["score"] for k, v in rep["by_category"].items()})


def show() -> None:
    if not LOG.exists():
        print("no runs yet")
        return
    rows = list(csv.DictReader(LOG.open()))
    cols = ["label", "scale", "accuracy", "source_hit_rate", "consistency", "speed_score", "p50_ms", "p95_ms", "max_ms",
            "first_answer_ms", "ingest_s", "peak_mem_mb", "conflicts_found"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")


if __name__ == "__main__":
    main()
