"""Find places where the website contradicts itself, and say which value is right.

We only look at the newest snapshot (that is where the *open* contradictions are) and only at
table rows, because a rate table is where a stale copy shows up as a clean, comparable number.

Two kinds of disagreement are reported:

1. A secondary page (calculator, promo, press release...) shows a rate for a product whose
   official page shows a different rate.  The official page wins.
2. Two equally official pages disagree.  The one with the later "effective as of" date wins.

Rows that differ only because they belong to different tiers (different balance, term or model-year
range) are NOT conflicts, which is the main source of false alarms in these tables.
"""
from __future__ import annotations

import re
from collections import defaultdict

from contract import Conflict, Source

PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
YEARS = re.compile(r"\b((?:19|20)\d\d)\s*-\s*((?:19|20)?\d\d)\b")
FOOT = re.compile(r"(?<=[A-Za-z%])\d{1,2}$")
OFFICIAL = 0.8


GENERIC = re.compile(r"^(none|n/?a|\d+\s*(months?|years?|mo\.?))?$", re.I)


def _canon(url: str) -> str:
    return url.split("#")[0].rstrip("/").lower().replace("://www.", "://")


def _years(label: str):
    m = YEARS.search(label)
    if not m:
        return None
    lo, hi = int(m.group(1)), m.group(2)
    hi = int(hi) if len(hi) == 4 else int(str(lo)[:2] + hi)
    return lo, hi


def _overlap(a, b) -> int:
    if not a or not b:
        return 0
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def _metric(header: str) -> str:
    h = header.lower()
    if "apy" in h or "percentage yield" in h:
        return "APY"
    if "apr" in h or "annual percentage rate" in h:
        return "APR"
    if "dividend" in h:
        return "dividend rate"
    return "rate"


def _facts(chunk):
    """Yield (label, key_without_years, years, metric, value) for each % cell of a table row."""
    def _is_label(v: str) -> bool:
        if not v or PCT.search(v):
            return False
        if re.fullmatch(r"[\s$\d.,]+", v):
            return False          # a bare number or amount: minimum balance, points, payment example
        return True

    label_cells = [v for v in chunk.values.values() if _is_label(v)]
    label = " ".join(label_cells).strip()
    if not label or GENERIC.match(label) or label.lower().startswith("none"):
        return              # a bare "36 Months" says nothing about which product it belongs to
    years = _years(label)
    key = YEARS.sub("", label.lower())
    key = re.sub(r"[^a-z0-9 ]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    for header, cell in chunk.values.items():
        for m in PCT.finditer(cell or ""):
            yield label, key, years, _metric(header), float(m.group(1)), label_cells
            break


def find(store) -> list[Conflict]:
    if not store.snapshots:
        return []
    snap = len(store.snapshots) - 1
    sid = store.snapshots[snap][0]
    rows = [c for c in store.chunks if c.kind == "row" and snap in c.snaps and c.values]

    # fact: (key, metric) -> list of (chunk, label, years, value)
    facts = defaultdict(list)
    for c in rows:
        for label, key, years, metric, value, _ in _facts(c):
            facts[(key, metric)].append((c, label, years, value))

    out: list[Conflict] = []
    seen = set()

    for (key, metric), items in facts.items():
        official = [it for it in items if it[0].auth >= OFFICIAL]
        other = [it for it in items if it[0].auth < OFFICIAL]
        if not official:
            continue

        # (1) secondary page vs the official row that covers the same tier
        for c, label, years, value in other:
            cands = official
            if years:
                cands = sorted(official, key=lambda it: (-_overlap(years, it[2]), it[1] != label))
                if _overlap(years, cands[0][2]) == 0:
                    continue
                cands = [cands[0]]
            else:
                cands = [it for it in official if it[1].lower() == label.lower()] or []
            for oc, olabel, oyears, ovalue in cands:
                if abs(ovalue - value) < 1e-9 or value == 0.0 or ovalue == 0.0:
                    continue
                sig = (key, metric, round(value, 4), round(ovalue, 4))
                if sig in seen:
                    continue
                seen.add(sig)
                out.append(Conflict(
                    topic=f"{olabel} {metric}",
                    values=[f"{value:.2f}%", f"{ovalue:.2f}%"],
                    sources=[Source(url=c.url, snapshot_id=sid, quote=c.text[:160]),
                             Source(url=oc.url, snapshot_id=sid, quote=oc.text[:160])],
                    resolution=f"{ovalue:.2f}%",
                    reason=f"{oc.url.replace('https://www.dupaco.com', '') or '/'} is the official product/rate page"
                           f"; {c.url.replace('https://www.dupaco.com', '') or '/'} is a secondary page that was not updated",
                ))

        # (2) two official pages with the same tier label but different values
        by_label = defaultdict(list)
        for it in official:
            by_label[it[1].lower()].append(it)
        for label, group in by_label.items():
            vals = {round(it[3], 4) for it in group}
            urls = {_canon(it[0].url) for it in group}
            if len(vals) < 2 or len(urls) < 2:
                continue
            group.sort(key=lambda it: (it[0].effective or "", it[0].auth), reverse=True)
            win = group[0]
            lose = [it for it in group if round(it[3], 4) != round(win[3], 4)]
            sig = (key, metric, "official", tuple(sorted(vals)))
            if sig in seen:
                continue
            seen.add(sig)
            out.append(Conflict(
                topic=f"{win[1]} {metric}",
                values=[f"{lose[0][3]:.2f}%", f"{win[3]:.2f}%"],
                sources=[Source(url=lose[0][0].url, snapshot_id=sid), Source(url=win[0].url, snapshot_id=sid)],
                resolution=f"{win[3]:.2f}%",
                reason="two official pages disagree; the one with the later effective date is used",
            ))
    return out
