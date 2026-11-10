#!/usr/bin/env python3
"""Score a solution's conflicts() with the kit's own scorer, against our checked conflict list.

The hidden answer list's granularity is unknown, so we score against two versions of it:
- page level: experiments/eval_sets/conflicts_gold.json, one entry per stale page (8 entries)
- row level: the same stale tables expanded to one entry per table row and rate column, plus the branch count

    python experiments/score_conflicts.py                       # the main solution
    python experiments/score_conflicts.py --variant experiments/variants/conflicts_v1 --show
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from run_eval import load_snapshots, load_submission  # noqa: E402
from scoring import contains, score_conflicts  # noqa: E402

GOLD = ROOT / "experiments" / "eval_sets" / "conflicts_gold.json"
PCT = re.compile(r"\d+\.\d+%")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("–", "-").replace("—", "-").replace("\\", "")).lower()


def table_rows(path: Path, row: str) -> list[list[str]]:
    """All data rows of the table that contains `row`."""
    lines = path.read_text(encoding="utf-8").replace("\\r", "").replace("\\n", "\n").split("\n")
    parts = [p.strip() for p in norm(row).split("|")]
    for i, l in enumerate(lines):
        if l.lstrip().startswith("|") and all(p in norm(l) for p in parts):
            a = i
            while a > 0 and lines[a - 1].lstrip().startswith("|"):
                a -= 1
            b = i
            while b + 1 < len(lines) and lines[b + 1].lstrip().startswith("|"):
                b += 1
            rows = [[c.strip() for c in x.strip().strip("|").split("|")] for x in lines[a:b + 1]]
            return [r for r in rows[2:] if any(PCT.search(c) for c in r)]  # skip header and --- line
    return []


def row_level(gold: list[dict], pages: dict[str, Path]) -> list[dict]:
    out = []
    for g in gold:
        if "stale" not in g:
            out.append({k: g[k] for k in ("id", "topic_keywords", "values", "correct_value")})
            continue
        st = table_rows(pages[g["stale"]["url"]], g["stale"]["row"])
        of = table_rows(pages[g["official"]["url"]], g["official"]["row"])
        if g["official"]["row"].count("|"):  # official table may hold other products: keep the same product
            prod = norm(g["official"]["row"].split("|")[0])
            of = [r for r in of if prod in norm(r[0])] or of
        for k, (s, o) in enumerate(zip(st, of)):
            sv = [c for c in s if PCT.search(c)]
            ov = [c for c in o if PCT.search(c)]
            for j, (a, b) in enumerate(zip(sv, ov)):
                a, b = PCT.search(a).group(), PCT.search(b).group()
                if contains(a, b):
                    continue
                out.append({"id": f"{g['id']}-r{k}c{j}", "topic_keywords": g["topic_keywords"],
                            "values": [a, b], "correct_value": b})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="solution")
    ap.add_argument("--show", action="store_true", help="print the submitted conflicts")
    args = ap.parse_args()

    snaps = load_snapshots(ROOT / "snapshots")
    d = sorted(p for p in (ROOT / "snapshots").iterdir() if (p / "manifest.json").exists())[-1]
    m = json.loads((d / "manifest.json").read_text())
    pages = {p["url"]: d / p["file"] for p in m["pages"]}
    pages.update({u.rstrip("/"): f for u, f in list(pages.items())})

    ks = load_submission(ROOT / args.variant)
    for s in snaps:
        ks.ingest(s)
    subs = ks.conflicts()
    gold_p = json.loads(GOLD.read_text())
    gold_r = row_level(gold_p, pages)
    for name, gold in (("page level", gold_p), ("row level", gold_r)):
        r = score_conflicts(gold, subs)
        pillar = 0.7 * r["f1"] + 0.3 * (r["resolution_accuracy"] or 0)
        print(f"{name:10s}  gold {r['gold']:3d}  submitted {r['submitted']:3d}  matched {r['matched']:3d}  "
              f"precision {r['precision']:.2f}  recall {r['recall']:.2f}  F1 {r['f1']:.2f}  "
              f"right winner {r['resolution_accuracy']}  conflicts pillar {pillar:.3f}")
    if args.show:
        for i, c in enumerate(subs, 1):
            print(f"{i:3d}. {c.topic[:60]} | {c.values} -> {c.resolution} | "
                  + " vs ".join(s.url.replace("https://www.dupaco.com", "") for s in c.sources))


if __name__ == "__main__":
    main()
