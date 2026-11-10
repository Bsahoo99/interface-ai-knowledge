#!/usr/bin/env python3
"""Refusals: how well does the system decide when to refuse?

Reads the kit runner's per-question results for one or more runs (experiments/scratch_runs/<tag>-<set>/)
and classifies every question:

- correct refusal:   a should_refuse question that was refused
- missed refusal:    a should_refuse question that was answered (the costly error for a bank)
- over-refusal:      an answerable question that was refused although every required value was in the
                     evidence the reader was shown (the model had the answer)
- blind refusal:     an answerable question refused when the evidence lacked the answer (a search miss,
                     so refusing was the honest outcome)

    python experiments/refusal_report.py d5_text1500 d5_text1500_2 [--variant DIR] [--show]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "solution"))
from run_eval import load_snapshots  # noqa: E402
from scoring import contains, REFUSAL_RE  # noqa: E402

SETS = {"public_eval": ROOT / "eval" / "public_eval.jsonl"}
SETS.update({f"{n}_eval": ROOT / "experiments" / "eval_sets" / f"{n}_eval.jsonl"
             for n in ("reworded", "heldout", "heldout2")})


def load_items():
    items = {}
    for s, f in SETS.items():
        for l in f.read_text().splitlines():
            if l.strip():
                it = json.loads(l)
                items[it["id"]] = (s, it)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--variant", default="solution", help="solution whose evidence the runs used")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("sub", ROOT / args.variant / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ks = mod.build()
    for s in load_snapshots(ROOT / "snapshots"):
        ks.ingest(s)
    import reader as rmod
    rd = rmod.Reader()
    store = ks.store
    items = load_items()
    in_ev = {}
    for i, (s, it) in items.items():
        if it["category"] == "should_refuse" or not it.get("must_include"):
            continue
        snap = store.snapshot_for(it.get("as_of"))
        shown = rd.render(rd.evidence(store, snap, it["question"])) if snap is not None else ""
        in_ev[i] = all(contains(shown, t) for t in it["must_include"])

    for tag in args.tags:
        c = Counter()
        rows = []
        for s in SETS:
            f = ROOT / "experiments" / "scratch_runs" / f"{tag}-{s}" / "per_question.jsonl"
            for l in f.read_text().splitlines():
                r = json.loads(l)
                it = items[r["id"]][1]
                text = r["answer"]["text"]
                refused = (not r["answer"].get("confident", True)) or bool(REFUSAL_RE.search(text))
                if it["category"] == "should_refuse":
                    k = "correct refusal" if r["correctness"] == 1 else "missed refusal"
                elif refused:
                    k = "over-refusal" if in_ev.get(r["id"]) else "blind refusal"
                else:
                    k = "answered"
                c[k] += 1
                if k in ("missed refusal", "over-refusal", "blind refusal"):
                    rows.append((k, r["id"], it["question"][:80], text[:100]))
        total_refuse = c["correct refusal"] + c["missed refusal"]
        refused_all = c["correct refusal"] + c["over-refusal"] + c["blind refusal"]
        print(f"[{tag}] " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
        print(f"   refusal recall {c['correct refusal']}/{total_refuse}, refusal precision "
              f"{c['correct refusal']}/{refused_all} (blind refusals are honest misses, not wrong)")
        if args.show:
            for row in sorted(rows):
                print("   ", " | ".join(row))


if __name__ == "__main__":
    main()
