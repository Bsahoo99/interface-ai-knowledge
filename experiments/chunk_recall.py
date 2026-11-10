#!/usr/bin/env python3
"""Record size: how each way of splitting pages into records changes what search finds (no API calls).

For every question with required values (all four question sets), per splitting variant:
- found in top 1 / 5 / 20: every required value appears in the top-k records of the answerer's ranking
- in reader evidence: every required value appears in the records the reader would be shown
- evidence size: average characters of evidence the reader would be sent (cost and latency)

    python experiments/chunk_recall.py [variant ...]      (default: all variants in _chunk_variant.CONFIGS)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "variants"))
import _chunk_variant as cv  # noqa: E402  (puts harness/ and solution/ on sys.path)
from run_eval import load_snapshots  # noqa: E402
from scoring import contains  # noqa: E402
import answerer  # noqa: E402
import kb  # noqa: E402
import reader  # noqa: E402

SETS = [ROOT / "eval" / "public_eval.jsonl"] + [
    ROOT / "experiments" / "eval_sets" / f"{n}_eval.jsonl" for n in ("reworded", "heldout", "heldout2")]
KS = (1, 5, 20)


def main():
    names = sys.argv[1:] or list(cv.CONFIGS)
    items = []
    for f in SETS:
        for l in f.read_text().splitlines():
            if l.strip():
                it = json.loads(l)
                if it.get("must_include") and it["category"] != "should_refuse":
                    items.append(it)
    snaps = load_snapshots(ROOT / "snapshots")
    print(f"{len(items)} questions with required values across {len(SETS)} sets\n")
    print("| variant | records (latest) | " + " | ".join(f"top {k}" for k in KS)
          + " | in reader evidence | evidence chars |")
    print("|---|---|" + "---|" * len(KS) + "---|---|")
    misses = {}
    for name in names:
        cv.install(name)
        store = kb.Store()
        for s in snaps:
            store.add_snapshot(s)
        rd = reader.Reader()
        found = {k: 0 for k in KS}
        ev_found, ev_chars, miss = 0, 0, []
        for it in items:
            snap = store.snapshot_for(it.get("as_of"))
            ix = store.index(snap)
            q = it["question"]
            qt = answerer._expand(kb.tokens(q))
            ranked, _ = answerer._rank(ix, qt, answerer._is_rate_question(q), k=40)
            for k in KS:
                if all(contains(" ".join(c.text for _, c in ranked[:k]), t) for t in it["must_include"]):
                    found[k] += 1
            ev = rd.evidence(store, snap, q)
            shown = rd.render(ev)
            ev_chars += len(shown)
            if all(contains(shown, t) for t in it["must_include"]):
                ev_found += 1
            else:
                miss.append(it["id"])
        misses[name] = miss
        n = len(items)
        latest = len(store.index(len(store.snapshots) - 1).chunks)
        print(f"| {name} | {latest} | " + " | ".join(f"{found[k]}/{n}" for k in KS)
              + f" | {ev_found}/{n} | {ev_chars // n} |")
    print()
    for name, m in misses.items():
        print(f"{name}: not in reader evidence: {m}")


if __name__ == "__main__":
    main()
