"""Chunking, normalisation and the versioned chunk store.

A *chunk* is either one table row (with its header labels, section heading and the
"effective as of" date found next to the table) or a run of paragraph text.  Every chunk
remembers which snapshots it appeared in, so we can answer "as of" questions and tell
stale copies from current ones.

This module also holds page trust (official product and rate pages over promotions, calculators and
news), snapshot choice by date, and the BM25 keyword index built for each snapshot at load time.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

# ---------------------------------------------------------------- normalisation

_HSLANG = re.compile(r"[?&]hsLang=[a-z-]+", re.I)
_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_ESC_NL = re.compile(r"(?:\\r)?\\n|\\r")
# the branch locator prints "Open" or "Closed" on a line of its own under each address: that is whether the
# branch was open at the moment the page was crawled, not a fact about the branch, and answers repeated it as
# "listed as closed". Hours lines ("Sunday: Closed") are not standalone and are kept.
_LIVE_STATUS = re.compile(r"^[ \t]*(?:open|closed)(?: now)?[ \t]*$\n?", re.I | re.M)
_FOOT = re.compile(r"(%|APR|APY|apr|apy)(\d{1,2})(?!\d|\.\d)")
_EFFECTIVE = re.compile(
    r"effective(?:\s+as\s+of|\s+on|:)?\s+(\d{1,2})/(\d{1,2})/(\d{2,4})", re.I
)
_EFFECTIVE_ISO = re.compile(r"effective(?:\s+as\s+of|\s+on)?\s+(\d{4})-(\d{2})-(\d{2})", re.I)
MONTHS = {m: i + 1 for i, m in enumerate(
    "january february march april may june july august september october november december".split())}
_PUBLISHED = re.compile(r"\b(" + "|".join(m.capitalize() for m in (
    "january february march april may june july august september october november december".split()))
    + r") (\d{1,2}), (20\d\d)\b")
_EFFECTIVE_LONG = re.compile(
    r"effective(?:\s+as\s+of|\s+on)?\s+(" + "|".join(MONTHS) + r")\s+(\d{1,2}),?\s+(\d{4})", re.I)


# hosts that are the website itself, a development copy, or file and redirect infrastructure, not a member service
_NOT_A_SERVICE = re.compile(r"^(?:www\d*|m|[a-z]*dev|staging|test|uploads|cdn|links|static|images?)\.")


def _own_service(target: str, page_url: str) -> Optional[str]:
    """The address of a link to another part of the same organisation (join.dupaco.com from www.dupaco.com),
    without tracking parameters; None for links within the same site or to other organisations."""
    t = re.match(r"https?://([^/?#]+)([^?#]*)", target.strip())
    p = re.match(r"https?://([^/?#]+)", page_url or "")
    if not t or not p:
        return None
    host, home = t.group(1).lower(), p.group(1).lower()
    domain = ".".join(home.split(".")[-2:])
    if host == home or not host.endswith("." + domain) or _NOT_A_SERVICE.match(host):
        return None
    return f"https://{host}{t.group(2)}"


def clean_text(s: str, page_url: str = "") -> str:
    s = _ESC_NL.sub("\n", s)
    s = _LIVE_STATUS.sub("", s)
    s = _HSLANG.sub("", s)
    s = _IMG.sub("", s)
    # keep the anchor text and drop the URL, except for links to the organisation's own services on another
    # host ("Become a member" -> join.dupaco.com): that address is the answer to "how do I join online"
    s = _LINK.sub(lambda m: m.group(1) + (f" ({u})" if (u := _own_service(m.group(2), page_url)) else ""), s)
    s = _FOOT.sub(r"\1", s)                          # "5.39% APR1" -> "5.39% APR"
    s = s.replace("–", "-").replace("—", "-").replace("**", "").replace("\\*", "*")
    s = re.sub(r"[ \t]+", " ", s)
    return s


def published_on(md: str) -> Optional[str]:
    """The date printed at the top of a dated page (news release, blog post), as ISO, else None."""
    m = _PUBLISHED.search(md[:600])
    if not m:
        return None
    try:
        return date(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2))).isoformat()
    except ValueError:
        return None


def parse_effective(text: str) -> Optional[str]:
    """Return the latest 'effective as of' date in text as ISO, or None."""
    found = []
    for m in _EFFECTIVE.finditer(text):
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y += 2000 if y < 100 else 0
        found.append((y, mo, d))
    for m in _EFFECTIVE_ISO.finditer(text):
        found.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in _EFFECTIVE_LONG.finditer(text):
        found.append((int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2))))
    ok = []
    for y, mo, d in found:
        try:
            date(y, mo, d)
            ok.append((y, mo, d))
        except ValueError:
            pass
    if not ok:
        return None
    y, mo, d = max(ok)
    return f"{y:04d}-{mo:02d}-{d:02d}"


# ---------------------------------------------------------------- tokenising

STOP = set("""a an the of to in on for and or is are was were be been do does did can could i you your my me we our
us it its this that these those with at by from as what whats which how when where who whom any there their about
please tell know like want would should have has had get give provide current currently right now today dupaco
offer offers available much many""".split())

NUMWORDS = {w: str(i + 2) for i, w in enumerate(      # "one" is left alone: too common as a plain word
    "two three four five six seven eight nine ten eleven twelve".split())}
RATEISH = {"apr", "apy", "dividend", "interest", "yield", "percent"}
_TOKEN = re.compile(r"[a-z0-9$%.:/#-]+")
SYN = {
    "cd": "certificate", "cds": "certificates", "apr": "apr", "apy": "apy", "mortgage": "home",
    "mortgages": "home", "car": "auto", "vehicle": "auto", "mobile": "mobile", "wallet": "wallet",
}


def tokens(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        t = NUMWORDS.get(t, t)
        t = t.strip(".-:/")
        if not t or t in STOP:
            continue
        t = SYN.get(t, t)
        out.append(t)
        if t in RATEISH:
            out.append("rate")
        # "17-month" -> also index "17" and "month"; "60-month" etc.
        if "-" in t:
            out.extend(p for p in t.split("-") if p and p not in STOP)
        if len(t) > 4 and t.endswith("ies"):
            out.append(t[:-3] + "y")          # "lobbies" -> "lobby", "policies" -> "policy"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            out.append(t[:-1])
            if t.endswith(("ches", "shes", "sses", "xes", "zes")):
                out.append(t[:-2])            # "branches" -> "branch", "taxes" -> "tax", "addresses" -> "address"
    return out


# ---------------------------------------------------------------- authority

def authority(url: str) -> float:
    """How much we trust a page as the *source of truth* for a fact."""
    u = re.sub(r"^[a-z]+://[^/]+", "", url.lower())   # judge by the path, whatever the site
    if re.match(r"^/\d{4}/\d{2}/", u):
        return 0.35                                   # dated blog permalinks
    if u.startswith("/learn") or "/resources" in u:
        return 0.45
    if "calculator" in u:
        return 0.40
    if u.startswith("/newsroom"):
        return 0.55
    if u.startswith("/switch-and-earn") or "promo" in u or "sprint" in u:
        return 0.6
    if u in ("", "/") or u.startswith(("/contact", "/personal", "/business", "/about", "/rates")):
        return 1.0
    if u.rstrip("/") in ("/checking", "/savings-account") or u.startswith("/savings-account/"):
        return 1.0                                    # product pages that live outside /personal
    return 0.6                                        # landing pages, promos, one-off pages


# ---------------------------------------------------------------- chunking

@dataclass
class Chunk:
    cid: int
    url: str
    title: str
    kind: str                    # "row" | "text"
    text: str                    # human readable (what we answer with)
    context: str                 # heading path + table header, used only for retrieval
    key: str = ""                # normalised fact key for rows (product + tier + term)
    values: dict = field(default_factory=dict)   # header -> cell, for rows
    effective: Optional[str] = None
    auth: float = 1.0
    snaps: set = field(default_factory=set)      # indexes of snapshots containing it
    toks: Counter = field(default_factory=Counter)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$])")

# Paragraph text is cut at sentence ends into pieces of about this many characters (about 375 tokens).
# 1500 keeps a fact and the sentence that names it together; 500 and 2000 both tested worse.
TEXT_CHARS = 1500


def _caption(lines: list[str], i: int) -> Optional[str]:
    """The short title line right above a table that has no markdown heading of its own ("ATV, Snowmobile & Jet
    Ski Loans" on the rates page). Without it a row reads "Model Year 2022-21 | 60 Months | 8.54%" with no product."""
    k = i - 1
    while k >= 0 and not lines[k].strip():
        k -= 1
    if k < 0:
        return None
    ln = lines[k].strip()
    if (ln.startswith(("#", "|", "[", "!", "-", "*", "_")) or len(ln) > 60 or ln.endswith((".", ":", "!", "?"))
            or re.search(r"\d\.\d+\s*%", ln)):
        return None
    return ln


def chunk_page(url: str, title: str, md: str):
    """Yield (kind, text, context, key, values, effective) tuples for one page."""
    md = clean_text(md, url)
    lines = md.split("\n")
    heads: list[str] = []
    i, n = 0, len(lines)
    para: list[str] = []

    def flush_para():
        if not para:
            return
        units = []
        for p in para:
            p = p.strip()
            if not p:
                continue
            p = re.sub(r"^[-*]\s+", "", p)
            units.extend(x.strip() for x in _SENT.split(p) if x.strip())
        para.clear()
        ctx = " > ".join(heads[-2:])
        cur = ""
        for u in units:
            if cur and len(cur) + len(u) > TEXT_CHARS:
                _emit(cur, ctx)
                cur = ""
            cur += u + " "
        _emit(cur, ctx)

    def _emit(text, ctx):
        text = text.strip()
        if len(text) >= 40 or (len(text) >= 15 and re.search(r"[$%]\s?\d|\d\s?%", text)):
            out.append(("text", text, ctx, "", {}, parse_effective(text)))

    out: list = []
    while i < n:
        ln = lines[i].rstrip()
        if ln.startswith("#"):
            flush_para()
            level = len(ln) - len(ln.lstrip("#"))
            heads = heads[: level - 1] + [ln.lstrip("# ").strip()]
            i += 1
            continue
        if ln.lstrip().startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            flush_para()
            header = _cells(ln)
            j = i + 2
            rows = []
            while j < n and lines[j].lstrip().startswith("|"):
                rows.append(_cells(lines[j]))
                j += 1
            # effective date: text right after the table (a few lines)
            tail = " ".join(lines[j: j + 6])
            eff = parse_effective(tail) or parse_effective(" ".join(lines[max(0, i - 4): i]))
            ctx = " > ".join(heads[-2:])
            cap = _caption(lines, i)
            if cap and cap.lower() not in ctx.lower():
                ctx = f"{ctx} > {cap}" if ctx else cap
            for r in rows:
                vals = {header[k] if k < len(header) else f"c{k}": r[k] for k in range(len(r))}
                label = " | ".join(f"{h}: {v}" for h, v in vals.items() if v)
                key_cells = [r[k] for k in range(len(r)) if not re.search(r"\d+\.\d+%|\$[\d,]+\.\d\d$", r[k])]
                key = " ".join(key_cells).lower()
                out.append(("row", label, ctx + " " + " ".join(header), key, vals, eff))
            i = j
            continue
        if ln.strip():
            para.append(ln)
        i += 1
    flush_para()
    return out


class Store:
    """All unique chunks, each tagged with the snapshots it appeared in."""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self._by_sig: dict[tuple, Chunk] = {}
        self.snapshots: list[tuple[str, str]] = []      # (snapshot_id, taken_on)
        self.df: Counter = Counter()
        self._index_cache: dict[int, "Index"] = {}
        self._links: dict[int, Counter] = {}
        self.page_links: dict[int, dict[str, list]] = {}
        self.vocab: set = set()
        self.published: dict[int, dict[str, str]] = {}   # snapshot -> url -> date printed on the page

    def links(self, snap_idx: int) -> Counter:
        return self._links.get(snap_idx, Counter())

    def add_snapshot(self, snapshot) -> None:
        idx = len(self.snapshots)
        self.snapshots.append((snapshot.snapshot_id, snapshot.taken_on))
        lc: Counter = Counter()
        pl: dict[str, list] = {}
        for p in snapshot.pages:
            for anchor, target in _LINK.findall(_HSLANG.sub("", p.markdown)):
                target = target.strip()
                if target.startswith("/"):
                    target = "https://www.dupaco.com" + target
                if target.startswith("http") and anchor.strip():
                    key = (anchor.strip().lower(), target.rstrip("/") if "#" not in target else target)
                    lc[key] += 1
                    pl.setdefault(p.url, []).append(key)
        self._links[idx] = lc
        self.page_links[idx] = pl
        self.published[idx] = {p.url: d for p in snapshot.pages if (d := published_on(p.markdown))}
        for p in snapshot.pages:
            for kind, text, ctx, key, vals, eff in chunk_page(p.url, p.title, p.markdown):
                sig = (p.url, kind, text, eff)   # same row under a new effective date is a new record
                ch = self._by_sig.get(sig)
                if ch is None:
                    ch = Chunk(len(self.chunks), p.url, p.title, kind, text, ctx, key, vals, eff,
                               authority(p.url))
                    ch.toks = Counter(tokens(f"{p.title} {ctx} {text} {key}"))
                    if "%" in text:
                        ch.toks["rate"] += 1
                    self.vocab.update(ch.toks)
                    self._by_sig[sig] = ch
                    self.chunks.append(ch)
                ch.snaps.add(idx)
        # Build this snapshot's search index now, while loading (untimed), instead of on the first
        # question that needs it. Earlier snapshots' indexes stay valid: their chunk sets never change.
        self.index(idx)

    def add_records(self, snap_idx: int, records: list[dict]) -> None:
        """Add derived text records (url, title, text, context, key) to one snapshot and rebuild its index."""
        for r in records:
            ch = Chunk(len(self.chunks), r["url"], r["title"], "text", r["text"], r["context"], key=r["key"],
                       auth=1.0)
            ch.toks = Counter(tokens(f"{r['title']} {r['context']} {r['text']}"))
            ch.snaps.add(snap_idx)
            self.vocab.update(ch.toks)
            self.chunks.append(ch)
        self._index_cache.pop(snap_idx, None)
        self.index(snap_idx)

    def snapshot_for(self, as_of=None) -> Optional[int]:
        """The snapshot that was current on as_of (an ISO date or timestamp, or a date object); None = newest.
        Chosen by date, so the answer does not depend on the order the snapshots were loaded in."""
        if not self.snapshots:
            return None
        order = sorted(range(len(self.snapshots)), key=lambda i: self.snapshots[i][1])
        if not as_of:
            return order[-1]
        day = (as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)).strip()[:10]
        best = None
        for i in order:
            if self.snapshots[i][1][:10] <= day:
                best = i
        return best

    def index(self, snap_idx: int) -> "Index":
        ix = self._index_cache.get(snap_idx)
        if ix is None:
            ix = Index([c for c in self.chunks if snap_idx in c.snaps])
            self._index_cache[snap_idx] = ix
        return ix


class Index:
    """Plain BM25 over the chunks that exist in one snapshot."""

    K1, B = 1.4, 0.75

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.df: Counter = Counter()
        total = 0
        for c in chunks:
            total += sum(c.toks.values())
            for t in c.toks:
                self.df[t] += 1
        self.avg = total / max(1, len(chunks))
        self.N = len(chunks)
        self.by_text: dict[str, list] = defaultdict(list)
        self.copies: dict[str, list[Chunk]] = defaultdict(list)   # text -> every record with exactly that text
        for c in chunks:
            if c.kind == "row":
                self.by_text[c.text].append(c)
            self.copies[c.text].append(c)
        self.inv: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            for t in c.toks:
                self.inv[t].append(c)

    def idf(self, t: str) -> float:
        df = self.df.get(t, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, qtoks: list[str], k: int = 12):
        qset = list(dict.fromkeys(qtoks))
        scores: dict[int, float] = defaultdict(float)
        by_id = {}
        for t in qset:
            idf = self.idf(t)
            for c in self.inv.get(t, ()):
                tf = c.toks[t]
                dl = sum(c.toks.values())
                s = idf * tf * (self.K1 + 1) / (tf + self.K1 * (1 - self.B + self.B * dl / self.avg))
                scores[c.cid] += s
                by_id[c.cid] = c
        ranked = sorted(((s, by_id[i]) for i, s in scores.items()), key=lambda x: -x[0])
        return ranked[:k]
