"""Find places where the website contradicts itself, and say which value is right.

Only the newest snapshot is checked: that is where the open contradictions are.
A difference between snapshots is a change over time, not a contradiction.

Three kinds of contradiction are reported:

1. Rate tables. A secondary page (promotion, landing page, calculator, blog) shows a rate for a
   product and tier whose official page shows a different rate. The official page wins.
   Two official pages that disagree are also reported; the later "effective as of" date wins.
   One conflict is reported per stale table (a stale page usually copies a whole table), with the
   table's first differing row as the example and the number of differing rows in the reason.
2. Rates stated in sentences. A secondary page's sentence gives a rate for a product and tier
   ("6-month certificate APY 2.95%") that differs from the official table. The official page wins.
3. Facts repeated in dated page boilerplate, such as the number of branch offices in the
   "About Dupaco" paragraph of every news release. The newest dated page wins.

Rows that differ only because they belong to different tiers (balance, term, model-year range) or
different products (personal vs business checking) are not conflicts: that is the main source of
false alarms in these tables.
"""
from __future__ import annotations

import re
from collections import defaultdict

from contract import Conflict, Source

PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
YEARS = re.compile(r"\b((?:19|20)\d\d)\s*-\s*((?:19|20)?\d\d)\b")
NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![a-z])", re.I)   # "1st" is not a tier number
WORD = re.compile(r"\b[a-z][a-z0-9]+\b")   # whole words only: "1st" gives nothing, not "st"
OFFICIAL = 0.8
GROUP_BY_TABLE = True  # False: one conflict per row and rate column (a tested alternative)

# words that describe a table layout, not a product
LAYOUT = set("""account type balance tier dividend rate annual percentage yield apy apr min minimum open earn
to of the and or as low none months month term loan amount program points payment example rates""".split())

# facts that describe the credit union *now*, stated in the boilerplate of dated pages:
# (name, the plain word members use, how dated pages state it, how an answer states a count of it)
BOILERPLATE = [("branch office locations", "branches", re.compile(r"\b(\d{1,3}) branch office locations"),
                re.compile(r"\b(\d{1,3})\s+(?:[A-Za-z-]+\s+){0,2}branch", re.I))]
RESOLVED = "resolved"   # Chunk.key of the records made from these facts (see current_facts)


def _canon(url: str) -> str:
    return url.split("#")[0].rstrip("/").lower().replace("://www.", "://")


def _short(url: str) -> str:
    return url.replace("https://www.dupaco.com", "").split("#")[0] or "/"


def _years(label: str):
    m = YEARS.search(label)
    if not m:
        return None
    lo, hi = int(m.group(1)), m.group(2)
    hi = int(hi) if len(hi) == 4 else int(str(lo)[:2] + hi)
    return (min(lo, hi), max(lo, hi))   # "Model Year 2026-23" is the same range as "2023-26"


def _overlap(a, b) -> int:
    if not a or not b:
        return 0
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def _metric(header: str, cell: str = "") -> str:
    c = cell.lower()   # a cell that says "5.34% APR" wins over a mislabelled column header
    if re.search(r"%\s*apr\b", c):
        return "APR"
    if re.search(r"%\s*apy\b", c):
        return "APY"
    h = header.lower()
    if "apy" in h or "percentage yield" in h:
        return "APY"
    if "apr" in h or "annual percentage rate" in h:
        return "APR"
    if "dividend" in h:
        return "dividend rate"
    return "rate"


def _is_label(v: str) -> bool:
    return bool(v) and not PCT.search(v) and not re.fullmatch(r"[\s$\d.,\-–—]*", v)


class Fact:
    """One % cell of a table row, with what it is about."""

    def __init__(self, chunk, header, value, rank):
        self.c, self.metric, self.value, self.rank = chunk, _metric(header, chunk.values.get(header, "")), value, rank
        cells = [v for h, v in chunk.values.items() if _is_label(v) or (v and not PCT.search(v) and h.lower().startswith(("balance", "loan amount", "term")))]
        self.label = " ".join(v for v in cells if v).strip()
        self.years = _years(self.label)
        tier_text = YEARS.sub(" ", self.label)
        self.tier = frozenset(n.replace(",", "") for n in NUM.findall(tier_text))
        own = set(WORD.findall(YEARS.sub(" ", self.label.lower()))) - LAYOUT
        # a tier-only row ("$5,000 - $24,999.99") takes its product words from the table's heading
        self.product = own or (set(WORD.findall(chunk.context.lower())) - LAYOUT)
        self.context = set(WORD.findall(f"{chunk.title} {chunk.context} {self.label}".lower()))

    def same_tier(self, o: "Fact") -> bool:
        if self.metric != o.metric or self.tier != o.tier:
            return False
        if self.years or o.years:
            return _overlap(self.years, o.years) > 0
        return True

    def product_fit(self, official: "Fact"):
        """(missing, shared): official product words absent from this row's page context, and shared words."""
        return len(official.product - self.context), len(official.product & self.context)


PROSE_RATE = re.compile(
    r"(\d+\.\d+)\s*%\s*(APY|APR|annual percentage (?:yield|rate)|dividend)|\b(APY|APR)\b[:\s]+(\d+\.\d+)\s*%", re.I)
DATES = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:19|20)\d\d\b")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$*])")
PROSE_STOP = set("""a an the of to in on for and with is are was at by up our your you we it its this that as be
from or will can may more than just only per new now get earn earns special offer ended expires limited time
summer winter spring fall holiday""".split())


class ProseFact(Fact):
    """One rate stated in a sentence ("6-month certificate APY 2.95%"), with what it is about."""

    def __init__(self, chunk, sentence, value, metric):
        self.c, self.metric, self.value, self.rank = chunk, metric, value, 0
        self.label = sentence.strip()[:160]
        rest = PROSE_RATE.sub(" ", DATES.sub(" ", sentence))
        self.years = None
        self.tier = frozenset(n.replace(",", "") for n in NUM.findall(rest))
        self.product = set(WORD.findall(rest.lower())) - LAYOUT - PROSE_STOP
        self.context = set(WORD.findall(f"{chunk.title} {chunk.context} {sentence}".lower()))


def _prose_facts(chunks):
    out = []
    for c in chunks:
        for s in SENT.split(c.text):
            # several rates in one sentence: each rate belongs to the words since the previous rate
            # ("6-month certificate APY: 3.50% 12-month certificate APY: 3.10%")
            start = 0
            for m in PROSE_RATE.finditer(s):
                part = s[start:m.end()]
                start = m.end()
                value = float(m.group(1) or m.group(4))
                unit = (m.group(2) or m.group(3) or "").lower()
                metric = ("APY" if "apy" in unit or "yield" in unit else "APR" if "apr" in unit or "rate" in unit
                          else "dividend rate")
                f = ProseFact(c, part, value, metric)
                f.context |= set(WORD.findall(s.lower()))
                if f.product:
                    out.append(f)
    return out


def _facts(rows):
    out = []
    for c in rows:
        rank = 0
        for header, cell in c.values.items():
            m = PCT.search(cell or "")
            if m:
                out.append(Fact(c, header, float(m.group(1)), rank))
                rank += 1
    return [f for f in out if f.label and f.tier | f.product]


def _best_official(f: Fact, official: list[Fact]):
    cands = [o for o in official if f.same_tier(o)]
    if not cands:
        return None
    # best product fit first; among equally good fits, the closest value (a stale copy of an auto
    # table is near the auto rates, not the recreational ones), then the widest model-year overlap
    scored = sorted(cands, key=lambda o: (f.product_fit(o)[0], -f.product_fit(o)[1], abs(o.value - f.value),
                                          -_overlap(f.years, o.years), o.label.lower() != f.label.lower(),
                                          "/rates" not in o.c.url, len(o.c.url)))   # cite the rate sheet when it agrees
    o = scored[0]
    missing, shared = f.product_fit(o)
    if missing or not shared:
        return None
    return o


def _table_conflicts(store, snap, sid) -> list[Conflict]:
    rows = [c for c in store.chunks if c.kind == "row" and snap in c.snaps and c.values]
    facts = _facts(rows)
    official = [f for f in facts if f.c.auth >= OFFICIAL]
    secondary = [f for f in facts if f.c.auth < OFFICIAL]

    # pairs of (stale fact, official fact) that disagree
    pairs = []
    for f in secondary:
        o = _best_official(f, official)
        if o and abs(o.value - f.value) > 1e-9 and o.value and f.value:
            pairs.append((f, o))
    # a rate stated in a sentence on a secondary page, against official tables and official sentences;
    # a page that already has a stale table is reported once, for its table
    texts = [c for c in store.chunks if c.kind == "text" and snap in c.snaps and "%" in c.text]
    prose = _prose_facts(texts)
    official_all = official + [p for p in prose if p.c.auth >= OFFICIAL]
    stale_pages = {_canon(f.c.url) for f, _ in pairs}
    for f in (p for p in prose if p.c.auth < OFFICIAL and _canon(p.c.url) not in stale_pages):
        o = _best_official(f, official_all)
        if o and abs(o.value - f.value) > 1e-9 and o.value and f.value:
            pairs.append((f, o))
    # official vs official: same product and tier on two pages, different value; later effective date wins
    for i, a in enumerate(official):
        for b in official[i + 1:]:
            if _canon(a.c.url) == _canon(b.c.url) or abs(a.value - b.value) < 1e-9:
                continue
            if a.label.lower() == b.label.lower() and a.same_tier(b) and (a.c.effective or "") != (b.c.effective or ""):
                old, new = sorted((a, b), key=lambda x: x.c.effective or "")
                pairs.append((old, new))

    # group: one conflict per stale table (page + table heading + effective date)
    groups = defaultdict(list)
    for f, o in pairs:
        key = (_canon(f.c.url), f.c.context, f.c.effective) if GROUP_BY_TABLE else (f.c.cid, f.metric)
        groups[key].append((f, o))

    out = []
    for key, items in groups.items():
        items.sort(key=lambda p: (p[0].c.cid, p[0].rank))
        f, o = items[0]
        n_rows = len({p[0].c.cid for p in items})
        official_side = f.c.auth >= OFFICIAL
        when_f = f"effective {f.c.effective}" if f.c.effective else "no effective date"
        when_o = f"effective {o.c.effective}" if o.c.effective else "no effective date"
        reason = (f"{_short(o.c.url)} ({when_o}) and {_short(f.c.url)} ({when_f}) are both official; the later effective date wins"
                  if official_side else
                  f"{_short(o.c.url)} is the official rate page ({when_o}); {_short(f.c.url)} is a secondary page "
                  f"({when_f}) whose copy was not updated")
        if n_rows > 1:
            reason += f"; {n_rows} rows of that table differ"
        # the example row's every differing rate column (e.g. dividend rate and APY), on both sides
        same_row = [p for p in items if p[0].c.cid == f.c.cid]
        stale_v = " / ".join(f"{p[0].value:.2f}% {p[0].metric}" for p in same_row)
        right_v = " / ".join(f"{p[1].value:.2f}% {p[1].metric}" for p in same_row)
        out.append(Conflict(
            topic=f"{o.label} {' and '.join(dict.fromkeys(p[1].metric for p in same_row))}",
            values=[stale_v, right_v],
            sources=[Source(url=f.c.url, snapshot_id=sid, quote=f.c.text[:160]),
                     Source(url=o.c.url, snapshot_id=sid, quote=o.c.text[:160])],
            resolution=right_v,
            reason=reason,
        ))
    return out


def _boilerplate_values(store, snap):
    """Per boilerplate fact: [(value, (newest date stating it, chunk))], newest first."""
    dates = store.published.get(snap, {})
    out = []
    for name, noun, pat, answer_pat in BOILERPLATE:
        seen: dict[str, tuple[str, object]] = {}   # value -> (newest date, chunk)
        for c in store.chunks:
            if c.kind != "text" or c.key == RESOLVED or snap not in c.snaps or c.url not in dates:
                continue
            for m in pat.finditer(c.text):
                v, d = m.group(1), dates[c.url]
                if v not in seen or d > seen[v][0]:
                    seen[v] = (d, c)
        if seen:
            out.append((name, noun, answer_pat, sorted(seen.items(), key=lambda kv: kv[1][0], reverse=True)))
    return out


def current_facts(store, snap) -> list[dict]:
    """The value the newest dated page gives for each boilerplate fact in one snapshot.

    The answerers use these so that answer() and conflicts() never disagree: each becomes a search
    record that states only the current value (no outdated number, which would score zero), and an
    answer that gives a different count of the same thing is rejected (solution.py).
    The record quotes the newest page's own paragraph (the "About Dupaco" boilerplate), so it also
    carries that page's other current facts, such as the counties and states the credit union serves."""
    out = []
    for name, noun, answer_pat, order in _boilerplate_values(store, snap):
        value, (date, chunk) = order[0]
        para = next((p for p in chunk.text.split("\n") if f"{value} {name}" in p), "").strip()
        out.append({"name": name, "noun": noun, "value": value, "date": date, "url": chunk.url,
                    "outdated": {v for v, _ in order[1:]}, "answer_pattern": answer_pat,
                    "text": f"Current number of {name} ({noun}): {value}, as stated on the newest dated page "
                            f"(published {date}). Pages that give a different number are older and out of date.",
                    "paragraph": para})
    return out


def _boilerplate_conflicts(store, snap, sid) -> list[Conflict]:
    out = []
    for name, _, _, order in _boilerplate_values(store, snap):
        if len(order) < 2:
            continue
        cur, (cur_date, cur_chunk) = order[0]
        old = order[1:]
        out.append(Conflict(
            topic=f"number of {name}",
            values=[f"{v} {name}" for v, _ in order],
            sources=[Source(url=cur_chunk.url, snapshot_id=sid, quote=cur_chunk.text[:160])]
                    + [Source(url=c.url, snapshot_id=sid, quote=c.text[:160]) for _, (_, c) in old[:2]],
            resolution=f"{cur} {name}",
            reason=f"dated pages disagree; the newest ({_short(cur_chunk.url)}, {cur_date}) says {cur}, "
                   + ", ".join(f"{v} last stated {d}" for v, (d, _) in old),
        ))
    return out


def find(store) -> list[Conflict]:
    if not store.snapshots:
        return []
    snap = store.snapshot_for(None)
    sid = store.snapshots[snap][0]
    return _table_conflicts(store, snap, sid) + _boilerplate_conflicts(store, snap, sid)
