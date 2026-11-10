#!/usr/bin/env python3
"""Check our hand-made conflict list (experiments/eval_sets/conflicts_gold.json) against the latest snapshot.

For each table conflict: the stale value must sit on the named row of the stale page, the official value
on the named row of the official page, each page must print its effective date, and the two values must differ.
For the prose conflict: every count the pattern finds is listed with the newest page date that states it,
so you can see that the newest dated pages give the 'current' value.

    python experiments/verify_conflicts.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from scoring import contains  # noqa: E402

GOLD = ROOT / "experiments" / "eval_sets" / "conflicts_gold.json"
DATE = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}), (20\d\d)")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("–", "-").replace("—", "-").replace("\\", "")).lower()


def latest_snapshot():
    d = sorted(p for p in (ROOT / "snapshots").iterdir() if (p / "manifest.json").exists())[-1]
    m = json.loads((d / "manifest.json").read_text())
    pages = {p["url"].rstrip("/"): d / p["file"] for p in m["pages"]}
    return m["snapshot_id"], pages


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\\r", "").replace("\\n", "\n")


def check_side(pages, side) -> list[str]:
    f = pages.get(side["url"].rstrip("/"))
    if not f:
        return [f"page not in snapshot: {side['url']}"]
    t = text(f)
    lines = [l for l in t.split("\n") if l.lstrip().startswith("|")]
    row = norm(side["row"])
    parts = [p.strip() for p in row.split("|")]
    hits = [l for l in lines if all(p in norm(l) for p in parts)]
    errs = []
    if not hits:
        errs.append(f"row {side['row']!r} not on {side['url']}")
    elif not any(contains(h, side["value"]) for h in hits):
        errs.append(f"{side['value']} not on row {side['row']!r} of {side['url']}: {hits[0].strip()[:120]}")
    if side["effective"] not in t:
        errs.append(f"effective date {side['effective']} not printed on {side['url']}")
    return errs


def main():
    sid, pages = latest_snapshot()
    gold = json.loads(GOLD.read_text())
    ok_all = True
    print(f"latest snapshot: {sid}\n")
    for g in gold:
        errs = []
        if "stale" in g:
            errs += check_side(pages, g["stale"]) + check_side(pages, g["official"])
            if contains(g["stale"]["value"], g["official"]["value"]):
                errs.append("stale and official values are the same")
        else:
            pat = re.compile(g["prose"]["pattern"])
            newest: dict[str, tuple] = {}
            for url, f in pages.items():
                t = text(f)
                m = pat.search(t)
                d = DATE.search(t)
                if m and d:
                    when = datetime.strptime(" ".join(d.groups()), "%B %d %Y")
                    if m.group(1) not in newest or when > newest[m.group(1)][0]:
                        newest[m.group(1)] = (when, url)
            for v, (when, url) in sorted(newest.items(), key=lambda kv: kv[1][0], reverse=True):
                print(f"      {v:>3}: newest page {when:%Y-%m-%d}  {url.replace('https://www.dupaco.com', '')[:80]}")
            top = max(newest.items(), key=lambda kv: kv[1][0])[0] if newest else None
            if top != g["prose"]["current"]:
                errs.append(f"newest dated value is {top}, gold says {g['prose']['current']}")
        ok_all &= not errs
        print(f"{'ok  ' if not errs else 'FAIL'}  {g['id']}  {g['note']}")
        for e in errs:
            print(f"      {e}")
    print(f"\n{sum(1 for _ in gold)} conflicts, {'all verified' if ok_all else 'SOME FAILED'}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
