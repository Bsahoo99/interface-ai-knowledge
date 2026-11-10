"""Branch hours, gathered from every page of a snapshot into one record text per kind (lobby, drive-up).

The site has no general hours page: each branch's lobby and drive-up hours appear only in the news
release announcing that branch, as a bold label followed by one "Day: time" line per day. Search
brings up one or two of those releases at a time, so a general question ("what are your lobby
hours?") was answered from whichever branch came up, or refused. This module lists every distinct
schedule with how many releases give it, so the most common hours are stated, not guessed.
"""
from __future__ import annotations

import re
from collections import defaultdict

from kb import published_on

KINDS = {"lobby": "lobby", "drive-up": "drive-up", "drive up": "drive-up", "drive-thru": "drive-up",
         "drive-through": "drive-up", "drive thru": "drive-up", "drive through": "drive-up"}
LABEL = re.compile(r"^\*\*\s*(lobby|drive[- ]?(?:up|thru|through))\s*:?\s*\*\*\s*:?\s*(.*)$", re.I)
DAY_LINE = re.compile(r"^((?:mon|tue|wed|thu|fri|sat|sun)[a-z]*(?:\s*[-–]\s*(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*)?)"
                      r"\s*:\s*(.+)$", re.I)
CONTEXT = {"lobby": "general branch lobby hours, when branches and branch lobbies are open and close",
           "drive-up": "general drive-up drive-through drive-thru hours, when the drive-up is open and closes"}


def _schedules(md: str) -> dict[str, str]:
    """kind -> "Day: time; Day: time" for each labelled hours block on one page."""
    out: dict[str, str] = {}
    lines = [ln.strip() for ln in md.splitlines()]
    for i, ln in enumerate(lines):
        m = LABEL.match(ln)
        if not m:
            continue
        kind = KINDS[re.sub(r"\s+", " ", m.group(1).lower())]
        days = [m.group(2)] if m.group(2) else []
        for nxt in lines[i + 1:]:
            if not nxt:
                continue
            if not DAY_LINE.match(nxt):
                break
            days.append(nxt)
        days = [d for d in days if DAY_LINE.match(d)]
        if days and kind not in out:
            out[kind] = "; ".join(days)
    return out


def _key(schedule: str) -> str:
    return re.sub(r"\s+", " ", schedule.replace("–", "-").replace("—", "-")).lower()


def records(pages) -> list[dict]:
    """For each kind of hours found on two or more pages of a snapshot, one record text, stored once per
    distinct schedule on the newest page that gives it, so answers can cite the pages behind each schedule."""
    found: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))   # kind -> key -> [(date, url, text)]
    for p in pages:
        for kind, sched in _schedules(p.markdown).items():
            found[kind][_key(sched)].append((published_on(p.markdown) or "", p.url, sched))
    out = []
    for kind, groups in found.items():
        pages_n = sum(len(v) for v in groups.values())
        if pages_n < 2:
            continue
        dates = sorted(d for v in groups.values() for d, _, _ in v if d)
        ranked = sorted(groups.values(), key=lambda v: (-len(v), max(d for d, _, _ in v)), reverse=False)
        parts = []
        for v in ranked:
            text = max(v)[2]   # the newest page's own wording
            parts.append(f"({len(parts) + 1}) {text} ({len(v)} of {pages_n} branches)")
        span = f", published {dates[0]} to {dates[-1]}" if dates else ""
        text = (f"Branch {kind} hours, from the {pages_n} branch announcements that list them{span}. "
                f"Hours vary by branch: there are {len(parts)} different schedules, the most common first: "
                + " ".join(parts))
        # the same record once per schedule, on the newest page that gives it, most common first: the reader
        # sees one record ("same text also on ...") and the answer cites the pages behind the schedules it
        # names, not only the newest announcement
        for v in ranked:
            out.append({"url": max(v)[1], "title": f"Branch {kind} hours (they vary by branch)", "text": text,
                        "context": CONTEXT[kind], "key": ""})
    return out
