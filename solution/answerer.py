"""Rule-based question answering over the versioned store (deterministic, no LLM).

This answers every question when no API key is set, and is the fallback whenever the AI reader
(reader.py) fails a check or the API fails. It picks the matching table row or sentence for the
question, and refuses when the evidence cannot back the term, name or hours the question asks about.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from contract import Answer, Source
from kb import Chunk, Store, tokens, _SENT

RATE_WORDS = {"rate", "rates", "apr", "apy", "interest", "dividend", "yield", "percent"}
LINK_Q = re.compile(r"\b(link|url|website|web address|webpage|where (?:can|do) i (?:go|find|see))\b", re.I)
NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")
BASE = "https://www.dupaco.com"

# NOTE: the refusal must not contain any number (the scorer treats a number as an answer).
REFUSAL = ("I don't see that information on the website, so I can't confirm it. "
           "Please contact the credit union or visit a branch for help.")

EXPIRED = re.compile(r"\b(?:offer|promotion|promo|special|rate)s?\s+(?:has\s+|have\s+)?(?:ended|expired)|\bexpired\b|\bhas ended\b|\bended on\b", re.I)

CALENDAR = set("january february march april may june july august september october november december "
               "monday tuesday wednesday thursday friday saturday sunday".split())

EXPAND = {
    "lobby": ["hours", "branch"], "drive": ["window", "hours"], "wallet": ["apple", "pay"],
    "student": ["loan"], "routing": ["number"],
    "mortgage": ["loan", "purchase", "conventional", "fixed"], "mortgages": ["loan", "purchase", "conventional"],
    "cards": ["card"], "eligible": ["member", "live", "work", "iowa"], "eligibility": ["member", "live", "work", "iowa"],
}


def _money(s: str) -> list[float]:
    out = []
    for m in re.finditer(r"\$\s?(\d[\d,]*(?:\.\d+)?)", s):
        out.append(float(m.group(1).replace(",", "")))
    return out


def _range_of(cell: str):
    """Parse a balance/loan tier like '$10,000 - $24,999.99', '$25,000+', '$100,000 or less'."""
    vals = _money(cell)
    if not vals:
        return None
    low = cell.lower()
    if "or less" in low or "up to" in low or "under" in low:
        return (0.0, max(vals))
    if "+" in cell or "or more" in low:
        return (min(vals), math.inf)
    if len(vals) >= 2:
        return (min(vals), max(vals))
    return None


def _in_tier(row: Chunk, amount: float) -> bool:
    for v in row.values.values():
        r = _range_of(v)
        if r and r[0] <= amount <= r[1]:
            return True
    return False


def _year_in(row: Chunk, year: int) -> bool:
    for v in row.values.values():
        m = re.search(r"\b((?:19|20)\d\d)\s*-\s*((?:19|20)?\d\d)\b", v)
        if m:
            lo = int(m.group(1))
            hi = m.group(2)
            hi = int(hi) if len(hi) == 4 else int(str(lo)[:2] + hi)
            if lo <= year <= hi:
                return True
    return False


def _is_rate_question(q: str) -> bool:
    words = set(re.findall(r"[a-z]+", q.lower()))
    return bool((words | {w[:-1] for w in words if w.endswith("s")}) & RATE_WORDS)   # "APRs", "APYs", "yields"


def _coverage(ix, qtoks: list[str], chunk: Chunk) -> float:
    """Share of the question's most distinctive (rarest, non-numeric) known tokens the chunk contains."""
    uniq = [t for t in dict.fromkeys(qtoks) if not re.search(r"\d", t) and ix.df.get(t, 0) > 0]
    uniq.sort(key=lambda t: -ix.idf(t))
    top = uniq[:3]
    if not top:
        return 1.0
    return sum(1 for t in top if t in chunk.toks) / len(top)


def _pick_chunk(ix, qtoks, ranked):
    """Among near-best retrieval hits prefer authority, then newer effective date."""
    if not ranked:
        return None
    best = ranked[0][0]
    pool = [(s, c) for s, c in ranked if s >= 0.80 * best]
    pool.sort(key=lambda sc: (round(sc[1].auth, 1), sc[1].effective or "", sc[0]), reverse=True)
    return pool[0][1]


def _rows_for(ix, qtoks, q: str, top: Chunk) -> list[Chunk]:
    """The rows of the same table as `top` that actually match the question's qualifiers."""
    table = [c for c in ix.chunks if c.kind == "row" and c.url == top.url and c.context == top.context
             and c.effective == top.effective]
    if not table:
        return [top]
    amounts = _money(q)
    years = [int(y) for y in re.findall(r"\b(20[0-3]\d)\b", q)]
    uniq = list(dict.fromkeys(qtoks))
    scored = []
    for r in table:
        s = sum(ix.idf(t) for t in uniq if t in r.toks) + sum(ix.idf(t) for t in uniq if t in set(tokens(r.key)))
        if amounts and any(_in_tier(r, a) for a in amounts):
            s += 6
        if years and any(_year_in(r, y) for y in years):
            s += 6
        scored.append((s, r))
    scored.sort(key=lambda x: -x[0])
    m = scored[0][0]
    keep = [r for s, r in scored if s >= 0.97 * m][:6]
    return keep


def _best_sentences(ix, qtoks, text: str, limit=420) -> str:
    sents = [s.strip() for s in _SENT.split(text) if s.strip()]
    if len(sents) <= 1:
        return text[:limit]
    uniq = list(dict.fromkeys(qtoks))
    sc = []
    for i, s in enumerate(sents):
        st = set(tokens(s))
        sc.append((sum(ix.idf(t) for t in uniq if t in st), i))
    top = sorted(sorted(sc, key=lambda x: -x[0])[:2], key=lambda x: x[1])
    out = " ".join(sents[i] for _, i in top)
    return out[:limit]


def _unseen_name(store: Store, question: str) -> bool:
    """True if the question names something (a capitalised word or a ratio like 7/1) the site never mentions."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", question)
    for k, w in enumerate(words):
        if k == 0 or w == "I" or not w[0].isupper() or len(w) < 4:
            continue
        if w.lower() in CALENDAR:
            continue
        tk = set(tokens(w))
        if tk and not tk & store.vocab:
            return True
    for t in re.findall(r"\b\d+/\d+\b", question):
        if t not in store.vocab:
            return True
    return False


def _expand(qtoks: list[str]) -> list[str]:
    out = list(qtoks)
    for t in qtoks:
        out.extend(EXPAND.get(t, ()))
    return out


def _rank(ix, qtoks, rate_q=False, k=40):
    ranked = ix.search(qtoks, k=k)
    out = []
    for s, c in ranked:
        # Page trust decides between copies that disagree, which happens with rates (stale tables on
        # promotions and calculators). Other facts often live only on a news or blog page, so for
        # those questions every page ranks equally and the reader's official/secondary labels decide.
        w = s * (c.auth ** 2.0) if rate_q else s
        if rate_q:
            if not re.search(r"\d\.\d+\s*%", c.text):
                continue                      # a rate question needs a decimal rate as evidence
            if c.kind == "row":
                w *= 2.2
        out.append((w, c))
    return sorted(out, key=lambda x: -x[0]), ranked


def _link_answer(store, snap, ix, question, qtoks, ranked):
    rate_q = _is_rate_question(question)
    uniq = set(qtoks)
    pages = list(dict.fromkeys(c.url for _, c in ranked[:8]))[:2]
    found = []
    for page in pages:
        cands = {(page.lower(), page): 0.1}
        for key in store.page_links.get(snap, {}).get(page, []):
            cands.setdefault(key, 0.0)
        best, bs = None, -1.0
        for (anchor, url), base in cands.items():
            words = set(tokens(anchor) + tokens(url.replace(BASE, "").replace("/", " ").replace(".", " ").replace("-", " ")))
            sc = base + sum(ix.idf(t) for t in uniq if t in words)
            if "#rates" in url and rate_q:
                sc += 2
            if sc > bs:
                best, bs = url, sc
        if best:
            found.append(best)
    return list(dict.fromkeys(found))


TERM = re.compile(r"\b(\d{1,3})[\s-]?(months?|mo|years?|yr)\b", re.I)


def _term_missing(question: str, text: str) -> bool:
    """True if the question names a term (72-month, 40-year) that the answer never states, in months or years."""
    for n, unit in TERM.findall(question):
        months = int(n) if unit.lower().startswith("mo") else int(n) * 12
        forms = [rf"\b{months}\s*-?\s*(?:months?|mo)\b"]
        if months % 12 == 0:
            forms.append(rf"\b{months // 12}\s*-?\s*(?:years?|yr)\b")
        if not any(re.search(f, text, re.I) for f in forms):
            return True
    return False


# Words that say what kind of thing is asked about, not which one; never treated as a name.
GENERIC = set(tokens(
    "rate rates apr aprs apy apys interest dividend yield percent loan loans account accounts card cards hours hour open "
    "close time branch branches location locations lobby drive up fee fees minimum balance term month months "
    "year years cost charge pay get give need much number phone website link page online member members "
    "credit union dupaco purchase new used find best every each all other same different nearest local main"))
PRODUCT_NOUN = re.compile(r"\b([A-Za-z][A-Za-z-]+)\s+(?:loans?|cards?|accounts?|certificates?|branch(?:es)?|mortgages?)\b",
                          re.I)
TIME = re.compile(r"\b\d{1,2}(?::\d\d)?\s*(?:a\.?m\.?|p\.?m\.?)|\bnoon\b", re.I)
HOURS_Q = re.compile(r"\b(hours?|what time)\b", re.I)


def _names(question: str) -> list[str]:
    """What the question names: capitalised words (not the first word or a date) and the word before a product noun
    ("Mullan Avenue branch", "Visa Signature card", "7/1 ARM", "construction loans")."""
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+/\d+", question)
    out = [w for k, w in enumerate(words) if k > 0 and (w[0].isupper() or "/" in w) and w.lower() not in CALENDAR
           and w not in ("I", "Dupaco", "Dupaco's")]
    out += [m.group(1) for m in PRODUCT_NOUN.finditer(question)]
    return list(dict.fromkeys(t for w in out for t in tokens(w) if t not in GENERIC))


def _name_missing(store: Store, snap: int, question: str, a: Answer) -> bool:
    """True if something the question names is in neither the answer nor the titles and headings of its pages."""
    if a.text.startswith("You can use http"):
        return False                       # link answers are a URL, chosen by their own matching
    urls = {s.url for s in a.sources or []}
    have = set(tokens(a.text))
    ix = store.index(snap)
    for c in ix.chunks:
        if c.url in urls:
            have.update(tokens(f"{c.title} {c.context}"))
    for t in _names(question):
        if ix.df.get(t, 0) and not {t, t.rstrip("s"), t + "s"} & have:
            return True                    # (a name the site never uses is refused earlier, by _unseen_name)
    return False


def answer(store: Store, question: str, as_of: Optional[str] = None) -> Answer:
    # Without these checks, a question about something the site does not cover gets the nearest record, which
    # is a confidently wrong answer: a 72-month certificate gets the 17-month rate, the Mullan Avenue branch
    # gets another branch's hours, a 7/1 ARM gets a certificate rate. Refusing is the right answer.
    a = _answer(store, question, as_of)
    if not a.confident:
        return a
    snap = store.snapshot_for(as_of)
    if (_term_missing(question, a.text) or _name_missing(store, snap, question, a)
            or (HOURS_Q.search(question) and not TIME.search(a.text))):
        return Answer(text=REFUSAL, confident=False)
    return a


def _answer(store: Store, question: str, as_of: Optional[str] = None) -> Answer:
    snap = store.snapshot_for(as_of)
    if snap is None:
        return Answer(text=REFUSAL, confident=False)
    sid = store.snapshots[snap][0]
    ix = store.index(snap)
    qtoks = _expand(tokens(question))
    if not qtoks or _unseen_name(store, question):
        return Answer(text=REFUSAL, confident=False)

    rate_q = _is_rate_question(question)
    link_q = bool(LINK_Q.search(question) or re.search(
        r"\bhow (?:do|can) i (?:join|apply|open|enroll|sign up)\b", question, re.I))
    ranked, raw = _rank(ix, qtoks, rate_q and not link_q)
    if not ranked:
        return Answer(text=REFUSAL, confident=False)

    if link_q:
        urls = _link_answer(store, snap, ix, question, qtoks, ranked)
        if urls:
            return Answer(text="You can use " + " or ".join(urls),
                          sources=[Source(url=u, snapshot_id=sid) for u in urls], confident=True)

    top = ranked[0][1]
    cov = _coverage(ix, qtoks, top)
    if cov < 0.5:
        return Answer(text=REFUSAL, confident=False)

    if top.kind == "row":
        listy = not re.search(r"\d", question) and re.search(
            r"\b(rates|cards|accounts|all|offer|certificates?)\b", question, re.I)
        tables = [top]
        if listy:
            bestrow = ranked[0][0]
            seen = {(top.url, top.context, top.effective)}
            for s, c in ranked:
                if c.kind == "row" and s >= 0.7 * bestrow and c.auth >= 0.8:
                    sig = (c.url, c.context, c.effective)
                    if sig not in seen:
                        seen.add(sig)
                        tables.append(c)
                if len(tables) >= 4:
                    break
        lines, srcs = [], []
        for t in tables:
            rows = _rows_for(ix, qtoks, question, t)
            lines.extend(_row_text(r) for r in rows)
            for u in dict.fromkeys([t.url] + [c.url for r in rows for c in ix.by_text.get(r.text, ()) if c.auth >= 0.8]):
                srcs.append(Source(url=u, snapshot_id=sid, quote=rows[0].text[:200]))
        text = "; ".join(dict.fromkeys(lines))
        if top.effective:
            text += f" (rates effective {top.effective})"
        return Answer(text=text, sources=srcs, confident=True)

    return _text_answer(ix, qtoks, ranked, sid, rate_q)


def _text_answer(ix, qtoks, ranked, sid, rate_q) -> Answer:
    """Sentence-level answer: best-matching sentences from the top few text chunks."""
    uniq = list(dict.fromkeys(qtoks))
    tot = sum(ix.idf(t) for t in uniq) or 1.0
    best_chunk = ranked[0][0] or 1.0
    cands = []
    for cs, c in ranked[:10]:
        if c.kind != "text":
            continue
        sents = [x.strip() for x in _SENT.split(c.text) if x.strip()] or [c.text]
        for k, snt in enumerate(sents):
            if EXPIRED.search(snt):
                continue                      # an ended promotion is not a current fact
            st = set(tokens(snt)) | (set(c.toks) & set(tokens(c.title)))
            if "%" in snt:
                st.add("rate")
            frac = sum(ix.idf(t) for t in uniq if t in st) / tot
            if rate_q and not re.search(r"\d\.\d+\s*%", snt):
                frac *= 0.3
            cands.append((frac + 0.25 * cs / best_chunk, frac, snt, c))
    if not cands:
        return Answer(text=REFUSAL, confident=False)
    cands.sort(key=lambda x: -x[0])
    if cands[0][1] < 0.30:
        return Answer(text=REFUSAL, confident=False)
    picked, seen_txt = [], set()
    for sc, frac, snt, c in cands:
        if snt in seen_txt or sc < 0.6 * cands[0][0]:
            continue
        seen_txt.add(snt)
        picked.append((snt, c))
        if len(picked) >= 3:
            break
    text = " ".join(x[0] for x in picked)[:700]
    srcs, seen = [], set()
    for snt, c in picked:
        if c.url not in seen:
            seen.add(c.url)
            srcs.append(Source(url=c.url, snapshot_id=sid, quote=snt[:200]))
    return Answer(text=text, sources=srcs, confident=True)


def _row_text(r: Chunk) -> str:
    return r.text
