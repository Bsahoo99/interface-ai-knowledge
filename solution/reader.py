"""Optional AI reader: a Claude model writes the answer from the top search results.

Search, page trust, "as of" snapshot choice and refusal wording stay deterministic (kb.py,
answerer.py). The model only reads the evidence records it is given and writes a short answer.
Guardrails keep the scorer's hard rules intact:

- the model must say whether the evidence answers the question; if not, the fixed refusal text
  (which contains no numbers) is returned
- every number in the model's answer must appear in the evidence or the question, otherwise the
  answer is discarded and the rule-based answer is used instead
- any API error, timeout, model refusal or missing API key falls back to the rule-based answer
- answers are cached per (model, snapshot, question), so a repeated question gets the same text
- after MAX_API_FAILURES API errors in a row the reader stops calling the API for the rest of the run,
  so an outage costs a few slow answers instead of a timeout on every question

solution.py turns the reader on when ANTHROPIC_API_KEY is set and the anthropic package is installed;
KNOWLEDGE_READER=off forces the rule-based answers.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

from contract import Answer, Source
from kb import Chunk, Store, tokens
import answerer

PRICES = {  # USD per million tokens (input, output), from the Claude API reference, 2026-10
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
SERVER_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}  # support fallbacks="default"

SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answerable", "answer", "evidence_ids"],
    "additionalProperties": False,
}

SYSTEM = """You answer members' questions about Dupaco Community Credit Union using only the evidence records you are given. The records come from one crawl of dupaco.com.

How to answer:
- Use only facts stated in the evidence. If the evidence does not answer the question, set answerable to false. When the question names one specific product, term, branch or card, do not substitute a similar one; if that exact thing is not in the evidence, it is not answerable.
- A kind of deposit account named in everyday words, however it is phrased (certificates, "the certificate of accounts", CDs, savings, checking), is not the name of a specific product: answer with that kind's rates or details from the evidence.
- When the question asks generally (for example "what are your lobby hours" or "what are the drive-up hours") and the evidence gives the answer for specific branches, products or accounts, answer with those as examples and say that it can vary; that is an answer, not a refusal.
- When the question asks how a member can get something done with their own account or membership (for example update their address or contact details, or ask about a transaction) and no record gives a specific procedure, an official record that tells members how to reach the credit union with account questions answers it: answer directly with those ways, with the phone number as written, and do not say that a procedure is missing.
- Match the question to the evidence the way a knowledgeable member-service employee would: a range such as "2024-27 Model Year" or "$10,000 - $24,999.99" includes every value inside it; terms in years and months are the same (5 years = 60 months); common abbreviations mean the full name (for example CD = certificate).
- If the evidence answers the question only in part, or only for some branches, accounts or conditions, give that answer and state the condition, rather than refusing. Refuse only when nothing in the evidence addresses the question. When the question names a specific product and asks for a value (a rate, fee, amount or count) for it, and the evidence describes that product but never states that value, set answerable to false; describing the product or giving other figures about it is not an answer. A matching row or sentence is an answer even when it uses different units or words (120 months = 10 years).
- When the evidence says how many times something has happened (for example "the fifth time"), include that count.
- When the question asks whether something is the same everywhere or for all (branches, accounts, cards), answer yes or no and then list every distinct value or schedule the evidence gives, each once, not just examples.
- When the question asks whether any account or tier of a general kind (savings, checking, certificates) pays a specific rate, name the accounts and tiers that pay it, and also give the highest rate of that kind's basic account (for savings, Share Savings) when it does not reach that rate.
- When the question gives no amount and a product's rate differs by amount tier, give the range from the lowest to the highest rate and name the tier of each end.
- When the evidence describes the same kind of fact for several dates or years (an award, a count, a rate), answer with the most recent one, and do not list the older ones. News releases and blog posts carry the date they were published; a later published date is more recent. Use published dates only to judge this, and do not write them in the answer.
- The website contains outdated copies of some values (calculators, blog posts, newsroom posts, promotion and landing pages). Each record is labelled official or secondary. When records disagree, trust the official record, and among official records the later effective date. Never present a value that appears only on a secondary page when an official record gives a different value. When only secondary pages give a value and no official record gives a different one, answer with it and say it comes from a promotional page that may be outdated; do not refuse because of that.
- Promotions or rates described as ended or expired are not current.
- If your answer says members must live or work in an area (such as a charter area), add the counties and states the evidence says the credit union serves.
- When the question names an amount, balance, term, model year or loan size, give only the value from the row that matches it, and do not list the values of other rows. When the question asks generally (for example "what are your mortgage rates"), lead with the main rate for each product and keep the list short.
- When the question asks for a list (for example which cards or wallets), name every matching item in the evidence.
- When the question asks for a link or web page, give the full URL of the most relevant official page from the evidence. For a product's rates or details, that is the product's own page that shows them (a record's page or one of the pages its text is also on), for example the savings accounts page for certificate rates, rather than the general rates page; you may name the general rates page after it.
- Copy numbers, dates, times, phone numbers and names exactly as written in the evidence, including % and $ signs. Do not round, convert or calculate.
- Write one to three plain sentences for a member. Mention the effective date when the record gives one. Do not mention records, evidence, crawls or "these facts".
- evidence_ids lists the ids of the records the answer relies on."""

MAX_API_FAILURES = 3

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# An answer that opens by saying the value asked for is not published is a refusal, worded around other
# figures ("no 72-month rate is listed for a 2015 motorcycle; the 48-month rate is 8.59%"). The model
# does this even when told not to, and a refusal must carry no figures.
SAYS_MISSING = re.compile(
    r"\b(?:is|are) not (?:listed|stated|given|shown|published)\b"
    r"|\b(?:does|do) not (?:state|list|give|show|include|publish) (?:(?:a|an|any) [\w\-/ ]{0,30})?(?:rate|apr|apy|fee|term|price|amount)s?\b"
    r"|\bbut not (?:a|an|any|the|its) [\w\-/ ]{0,30}(?:rate|apr|apy|fee|price)s?\b"
    r"|\bno [\w\-/ ]{0,40}(?:rate|apr|apy|fee|term|price|amount) is (?:listed|stated|given|shown|published)\b"
    r"|\b(?:gives|lists|shows|states) no [\w\-/ ]{0,30}(?:rate|apr|apy|fee|term|price|amount)\b",
    re.I)
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")
CLAUSE = re.compile(r";\s+")
ASKS_RATE = re.compile(r"\bwhat(?:'s| is| are)?\b[^?]*\b(?:rates?|aprs?|apys?|interest)\b", re.I)
ASKS_PLACE = re.compile(r"\b(?:link|url|website|web ?page|page|where)\b", re.I)   # "which page shows the rates"
RATE_VALUE = re.compile(r"\d+\.\d+\s*%")
# "Is there any savings tier earning 0.35% APY?"
ANY_SAVINGS = re.compile(r"\bany\b[^?]*\bsavings\b[^?]*?(\d+\.\d+)\s*%", re.I)
SHARE_SAVINGS_ROW = re.compile(r"^Account Type: Share Savings \|.*?Annual Percentage Yield[^:|]*: (\d+\.\d+)%")
# "the $40,000+ tier is 7.34%, which does not apply to a $30,000 loan"
NOT_APPLY = re.compile(r"\b(?:does|do|would) not apply\b|\bnot (?:applicable|apply) to\b", re.I)
AMOUNT = re.compile(r"\$\d[\d,]*")
RELATIVE_YEAR = re.compile(r"\b(last|this) year\b", re.I)


def dated_question(question: str, as_of: Optional[str], taken_on: str) -> str:
    """The question with "last year" or "this year" replaced by the calendar year it means on the question's date.

    The model cannot know what "last year" means without the date, and answered the same question three ways
    (the 2024 figure, a refusal, a 2025 figure of a different measure). Site figures lag, so when that year is
    not given the most recent earlier year is the answer, with its year stated. Done in the question rather
    than the shared prompt, so no other question's answer can change.
    """
    year = int((as_of or taken_on)[:4])

    def sub(m: re.Match) -> str:
        y = year - 1 if m.group(1).lower() == "last" else year
        return (f"in {y} (if the evidence gives no {y} figure for it, the most recent earlier year it gives, "
                f"with that year stated)")
    return RELATIVE_YEAR.sub(sub, question)
# The prompt's own words for its inputs (evidence records, official or secondary, crawled on a date) mean
# nothing to a member, and the model still uses them in about 1 answer in 150. Each is rewritten to what the
# member can see; "record" only after a determiner, so "a record 5,000 hours" in a news answer is kept.
INTERNAL_WORDS = [
    (re.compile(r"\b(the|this|that|these|those)\s+evidence\b", re.I), r"\1 information"),
    (re.compile(r"\bevidence\b", re.I), lambda m: "Information" if m.group(0)[0].isupper() else "information"),
    (re.compile(r"\b((?:the|this|that|an?|any|each|no|its|our|their)\s+(?:(?:official|secondary|promotion|promotional|most|recent|latest|newest|Dupaco|newsroom|rate|\d{4})\s+){0,4})"
                r"records?\b(?!\s*(?:[$\d]|high|low|number|year|-))", re.I),
     lambda m: m.group(1) + ("pages" if m.group(0)[-1] in "sS" else "page")),
    (re.compile(r"\s+is an? secondary (?:promotion(?:al)? |landing )?(?:page|source) and(?= may\b)", re.I), ""),
    (re.compile(r"\bsecondary (?:promotion(?:al)? |landing )?(?:page|source)s?\b", re.I),
     lambda m: ("P" if m.group(0)[0].isupper() else "p") + "romotional page" + ("s" if m.group(0)[-1] in "sS" else "")),
    (re.compile(r"\bsecondary(?=(?:[\s,]+[\w-]+){0,3}?\s+(?:page|source|record)s?\b)", re.I),
     lambda m: "Promotional" if m.group(0)[0].isupper() else "promotional"),
    # any other "secondary" thing ("a secondary fee schedule") is an older copy; "post-secondary education" is kept
    (re.compile(r"(?<![-\w])(?:(an?)\s+)?(secondary)\b(?!\s+(?:education|school|schools|market|students?)\b)", re.I),
     lambda m: ((("An" if m.group(1)[0].isupper() else "an") + " ") if m.group(1) else "")
     + ("O" if m.group(2)[0].isupper() and not m.group(1) else "o") + "lder"),
    (re.compile(r",?\s*crawled (?:on )?\d{4}-\d{2}-\d{2}\b", re.I), ""),
    (re.compile(r"\b[Tt]he (?:latest |most recent )?crawl(?: taken| dated)?(?: on)?\s+(?=[A-Z][a-z]+ \d|\d{4}-)"), ""),
    (re.compile(r"\bthe (?:current |latest |most recent )?crawl's\b", re.I), "the website's"),
    (re.compile(r"\bthe (?:latest |most recent )?crawl\b", re.I), "the website"),
]


def _plain_words(text: str) -> str:
    for pat, rep in INTERNAL_WORDS:
        text = pat.sub(rep, text)
    return text


# A rate in an answer carries a label ("1.80% APY", "7.26% APR"), and the number check only proves the number is
# somewhere in the evidence. Rate tables print a dividend or note rate next to its APY or APR, and when the model
# summarises a range it sometimes takes the number from the wrong column: "from 1.79% APY" for a 60-month
# certificate whose dividend rate is 1.79% and APY 1.80%.
_PCT = r"(\d+\.\d+)%"
LABELLED = re.compile(
    _PCT + r"\s*(?:-|–|to)\s*" + _PCT + r"\s*(APY|APR)\b"       # 1.79% to 3.40% APY: both are APYs
    r"|" + _PCT + r"\s*(APY|APR)\b(?!\)?\s*(?:of|is|was|at|:)?\s*\d)"   # 1.80% APY, but in "1.79% APY 1.80%" the label is the next number's
    r"|\b(APY|APR)\)?\s*(?:of|is|was|at|:)?\s*" + _PCT,          # APY 1.80%, (APY) of 1.80%
    re.I)
_FIELD_APR = re.compile(r"\bAPR\b|annual percentage rate", re.I)
_FIELD_APY = re.compile(r"\bAPY\b|annual percentage yield", re.I)


def _rate_labels(records: list[Chunk]):
    """From the evidence: the numbers that are an APY or an APR, and, per table row, which APY or APR goes with
    each dividend or note rate on that row (rate -> {(label, value)})."""
    known = {"APY": set(), "APR": set()}
    pairs: dict[str, set] = {}
    for c in records:
        for m in LABELLED.finditer(c.text):
            if m.group(3):
                known[m.group(3).upper()].update((m.group(1), m.group(2)))
            elif m.group(5):
                known[m.group(5).upper()].add(m.group(4))
            else:
                known[m.group(6).upper()].add(m.group(7))
        rates, labelled = [], {"APY": [], "APR": []}
        for field in c.text.split(" | "):
            name, _, value = field.partition(":")
            nums = re.findall(_PCT, value)
            if not nums:
                continue
            if _FIELD_APR.search(name) or _FIELD_APR.search(value):    # "Annual Percentage Yield: 8.54% APR" is an APR
                labelled["APR"].append(nums)
                known["APR"].update(nums)
            elif _FIELD_APY.search(name):
                labelled["APY"].append(nums)
                known["APY"].update(nums)
            elif re.search(r"\brate\b", name, re.I):
                rates.append(nums)
        for label, vals in labelled.items():
            if len(vals) == len(rates):                  # the n-th rate goes with the n-th APY or APR
                for r, v in zip(rates, vals):
                    if len(r) == 1 and len(v) == 1:
                        pairs.setdefault(r[0], set()).add((label, v[0]))
    return known, pairs


def _fix_rate_labels(text: str, records: list[Chunk]) -> tuple[str, Optional[str]]:
    """Each number the answer calls an APY or APR must be one in the evidence. A dividend or note rate given that
    label is replaced by the APY or APR on its row when the evidence pairs it with exactly one. When it pairs it
    with several (one note rate, a different APR per loan amount), "7.13% APR" loses the wrong label and keeps the
    right number; in other wordings ("an APR of 7.13%") the answer is not used. Returns (text, problem)."""
    known, pairs = _rate_labels(records)
    edits = []     # (number start, number end, label, end of a trailing label that can be dropped, or None)
    for m in LABELLED.finditer(text):
        if m.group(3):
            edits += [(m.start(1), m.end(1), m.group(3), None), (m.start(2), m.end(2), m.group(3), None)]
        elif m.group(5):
            edits.append((m.start(4), m.end(4), m.group(5), m.end(5)))
        else:
            edits.append((m.start(7), m.end(7), m.group(6), None))
    out, last = [], 0
    for start, end, label, label_end in edits:
        label, value = label.upper(), text[start:end]
        if value in known[label] or value not in pairs:
            continue        # right label, or a number the tables do not give as a rate (left to the model)
        right = {v for lab, v in pairs[value] if lab == label}
        if len(right) == 1:
            out += [text[last:start], right.pop()]
            last = end
        elif label_end is not None:
            out += [text[last:end], "%"]
            last = label_end
        else:
            return text, f"{value}% called {label}, no single {label} on its row"
    return "".join(out) + text[last:], None


def _numbers(text: str) -> set[str]:
    out = set()
    for m in NUMBER.finditer(text or ""):
        v = m.group().replace(",", "").rstrip(".")
        if "." in v:
            v = v.rstrip("0").rstrip(".") or "0"
        out.add(v.lstrip("0") or "0")
    return out


def _trust(c: Chunk) -> str:
    if c.auth >= 0.8:
        return "official"
    u = c.url.lower()
    kind = ("calculator" if "calculator" in u else "blog post" if re.search(r"/20\d\d/|/learn|/resources", u)
            else "newsroom post" if "/newsroom" in u else "promotion or landing page")
    return f"secondary ({kind}, may be outdated)"


class Reader:
    def __init__(self, model: str = "claude-haiku-5-5", effort: Optional[str] = None,
                 thinking_off: bool = False, k: int = 15, max_records: int = 48, timeout_s: float = 25.0):
        self.model, self.effort, self.thinking_off = model, effort, thinking_off
        self.k, self.max_records, self.timeout_s = k, max_records, timeout_s
        self.cache: dict = {}
        self.log: list[dict] = []
        self.log_path = os.getenv("READER_LOG")
        self._client = None
        self.api_failures = 0  # consecutive API errors; reset by any successful call

    # ---- evidence -------------------------------------------------------------------------
    def evidence(self, store: Store, snap: int, question: str) -> list[Chunk]:
        ix = store.index(snap)
        qtoks = answerer._expand(tokens(question))
        rate_q = answerer._is_rate_question(question)
        # Search deep: the same text often sits on many pages (a rate table copied to promotions, a newsroom
        # menu on every tag page), and the slots below count distinct records, not search hits.
        primary, _ = answerer._rank(ix, qtoks, rate_q, k=400)
        general, _ = answerer._rank(ix, qtoks, False, k=400)
        picked: list[Chunk] = []
        seen: set[str] = set()

        def add(c: Chunk) -> bool:
            if c.text not in seen and len(picked) < self.max_records:
                seen.add(c.text)
                picked.append(c)
                return True
            return False

        def add_distinct(ranked, n: int):
            added = 0
            for _, c in ranked:
                if added >= n:
                    break
                added += add(c)

        add_distinct(primary, self.k)
        add_distinct(general, 10)
        # a retrieved table row brings the rest of its table, so the matching tier is always there
        for c in list(picked):
            if c.kind == "row":
                for r in ix.chunks:
                    if r.kind == "row" and r.url == c.url and r.context == c.context and r.effective == c.effective:
                        add(r)
        # the best official rate page also brings its other small tables (a product's sibling table,
        # e.g. a line of credit next to a fixed loan, is often missed by keyword search)
        row_pages = list(dict.fromkeys(c.url for c in picked if c.kind == "row" and c.auth >= 0.8))[:1]
        for page in row_pages:
            sib = [r for r in ix.chunks if r.kind == "row" and r.url == page]
            if len(sib) <= 40:
                for r in sib:
                    add(r)
        return picked

    @staticmethod
    def pages(store: Store, snap: int, c: Chunk, question: str) -> list[Chunk]:
        """Every page that carries exactly this record's text, best first: official pages, then (unless the question
        says "business") personal before business pages, then pages whose address shares words with the question.
        A rate row is often copied to several product pages; the member should see the page about their product."""
        copies = list({x.url: x for x in store.index(snap).copies.get(c.text, [c])}.values())
        q = set(tokens(question))
        business_q = "business" in q

        def key(x: Chunk):
            path = re.sub(r"^[a-z]+://[^/]+", "", x.url.lower())
            mismatch = ("/business" in path or "business-" in path) != business_q
            return (x.auth < 0.8, mismatch, -len(q & set(tokens(path.replace("/", " ").replace("-", " ")))))
        return sorted(copies, key=key)

    @staticmethod
    def render(records: list[Chunk], published: Optional[dict] = None, copies: Optional[list] = None) -> str:
        """published: url -> date printed on a news release or blog post, so "the most recent" can be judged.
        copies: per record, every page that carries the same text (Reader.pages), best first."""
        out = []
        for i, c in enumerate(records, 1):
            same = copies[i - 1] if copies else [c]
            c0 = same[0]
            head = f"[{i}] page: {c0.url} | {_trust(c0)}"
            if len(same) > 1:
                more = [x.url for x in same[1:4]]
                head += " | same text also on: " + ", ".join(more) + (f" and {len(same) - 4} more" if len(same) > 4 else "")
            if c0.effective:
                head += f" | effective {c0.effective}"
            if published and c0.url in published:
                head += f" | published {published[c0.url]}"
            section = " > ".join(x for x in (c0.title, c0.context) if x)   # the label of the page shown
            if section:
                head += f" | section: {section[:160]}"
            out.append(f"{head}\n{c.text[:2500]}")   # longest record is about 2,250 chars: shown whole
        return "\n\n".join(out)

    # ---- model call -----------------------------------------------------------------------
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(timeout=self.timeout_s, max_retries=1)
        return self._client

    def call(self, user: str) -> tuple[dict, dict]:
        kwargs = dict(
            model=self.model, max_tokens=4000, system=SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA},
                           **({"effort": self.effort} if self.effort else {})},
        )
        if self.thinking_off:
            kwargs["thinking"] = {"type": "disabled"}
        c = self.client()
        if self.model in SERVER_FALLBACK_MODELS:
            resp = c.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            resp = c.messages.create(**kwargs)
        u = resp.usage
        meta = {"stop_reason": resp.stop_reason, "model_served": resp.model,
                "input_tokens": u.input_tokens, "output_tokens": u.output_tokens}
        if resp.stop_reason != "end_turn":
            raise RuntimeError(f"stop_reason={resp.stop_reason}")
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text), meta

    # ---- answer ---------------------------------------------------------------------------
    def answer(self, store: Store, question: str, as_of: Optional[str] = None) -> Answer:
        snap = store.snapshot_for(as_of)
        if snap is None:
            return Answer(text=answerer.REFUSAL, confident=False)
        key = (self.model, snap, question.strip().lower())
        if key in self.cache:
            return self.cache[key]
        sid, taken_on = store.snapshots[snap]
        records = self.evidence(store, snap, question)
        entry = {"question": question, "as_of": as_of, "model": self.model, "records": len(records)}
        t0 = time.perf_counter()
        ans = None
        try:
            if not records:
                raise RuntimeError("no evidence")
            if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("ANTHROPIC_AUTH_TOKEN"):
                raise RuntimeError("no API key")
            if self.api_failures >= MAX_API_FAILURES:
                raise RuntimeError(f"API off after {MAX_API_FAILURES} failures in a row")
            when = self.when(as_of, taken_on)
            published = store.published.get(snap, {})
            copies = [self.pages(store, snap, c, question) for c in records]
            asked = dated_question(question, as_of, taken_on)
            user = f"{when}\n\nEvidence records:\n\n{self.render(records, published, copies)}\n\nQuestion: {asked}"
            try:
                out, meta = self.call(user)
            except Exception as e:
                if type(e).__module__.startswith("anthropic"):  # network, timeout, auth, 5xx; not a bad reply
                    self.api_failures += 1
                raise
            self.api_failures = 0
            entry.update(meta)
            ans, reason = self._to_answer(out, records, asked + " " + when, sid, published, copies, asked=question)
            entry["outcome"] = reason
        except Exception as e:  # any failure: the deterministic answer is still a valid answer
            entry["outcome"] = f"fallback: {type(e).__name__}: {str(e)[:120]}"
        if ans is None:
            ans = answerer.answer(store, question, as_of)
        entry["latency_s"] = round(time.perf_counter() - t0, 3)
        p = PRICES.get(self.model)
        if p and "input_tokens" in entry:
            entry["cost_usd"] = round((entry["input_tokens"] * p[0] + entry["output_tokens"] * p[1]) / 1e6, 6)
        entry["answer"] = ans.text[:300]
        self.log.append(entry)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        self.cache[key] = ans
        return ans

    @staticmethod
    def when(as_of: Optional[str], taken_on: str) -> str:
        """The line that tells the model which date the evidence describes."""
        if as_of:  # said plainly: a softer wording made the model refuse past-date questions
            return (f"The question asks about {as_of}. The evidence below is the website exactly as it stood on that "
                    f"date (the most recent crawl before it, taken {taken_on}); treat its values as the ones in force "
                    f"on {as_of}.")
        return f"The evidence is from the current website (crawled {taken_on})."

    def _to_answer(self, out: dict, records: list[Chunk], question: str, sid: str, published: Optional[dict] = None,
                   copies: Optional[list] = None, asked: str = ""):
        if not out.get("answerable"):
            return Answer(text=answerer.REFUSAL, confident=False), "refused"
        text = (out.get("answer") or "").strip()
        if not text:
            return None, "fallback: empty answer"
        # Led by "not listed", the answer is a refusal; a later "other terms are not listed" is an aside
        # after a real answer, so only that sentence is dropped. Clauses joined by a semicolon
        # count as sentences: "8.59% APR on a 48-month term; other terms are not listed" is an answer.
        sents = [[c for c in CLAUSE.split(s) if c] for s in SENTENCE.split(text)]
        if SAYS_MISSING.search(sents[0][0]):
            return Answer(text=answerer.REFUSAL, confident=False), "refused: answer says the value is not listed"
        kept = []
        for clauses in sents:
            keep = [c for c in clauses if not SAYS_MISSING.search(c)]
            if keep:
                s = "; ".join(keep)
                kept.append(s if len(keep) == len(clauses) or s[-1] in ".!?" else s.rstrip(",:;") + ".")
        text = _plain_words(" ".join(kept))
        # A question that asks what a rate is, answered with no rate at all ("the construction loan is a 1/1 ARM
        # that can adjust up to 2% a year"), describes the product instead of answering: the rate is not
        # published, so it is a refusal. Table rates always carry decimals, so caps like "2%" do not count.
        # Checked after the sentences that say a value is missing are dropped, since a dropped sentence may
        # hold the only rate.
        if ASKS_RATE.search(asked) and not ASKS_PLACE.search(asked) and not RATE_VALUE.search(text):
            return Answer(text=answerer.REFUSAL, confident=False), "refused: rate question answered without a rate"
        # A question with an amount wants the one matching row: a sentence naming another row's rate as not
        # applying adds a second rate the member may misread, so it is dropped.
        if AMOUNT.search(asked):
            sents = SENTENCE.split(text)
            keep = [x for x in sents if not NOT_APPLY.search(x)]
            if keep and len(keep) < len(sents):
                text = " ".join(keep)
        ids = [i for i in out.get("evidence_ids", []) if isinstance(i, int) and 1 <= i <= len(records)]
        allowed = _numbers(question)
        for c in records:
            allowed |= _numbers(c.text) | _numbers(c.url) | _numbers(c.effective or "")
            allowed |= _numbers((published or {}).get(c.url, ""))   # the published date shown with the record
        unknown = _numbers(text) - allowed
        if unknown:
            return None, f"fallback: numbers not in evidence {sorted(unknown)[:5]}"
        text, mislabel = _fix_rate_labels(text, records)
        if mislabel:
            return None, f"fallback: {mislabel}"
        # Asked whether any savings tier pays a rate, the member is comparing with the basic account: when the
        # answer leaves Share Savings out and its top official APY is below the rate asked about, that APY is
        # added from the evidence (the prompt asks for it, the model leaves it out about 1 in 4).
        m = ANY_SAVINGS.search(asked)
        if m and not re.search(r"\bShare Savings\b", text):
            rows = [(float(r.group(1)), r.group(1), i) for i, c in enumerate(records) if c.auth >= 0.8
                    for r in [SHARE_SAVINGS_ROW.search(c.text)] if r]
            if rows and max(rows)[0] < float(m.group(1)):
                _, apy, i = max(rows)
                text += f" Share Savings, the basic savings account, pays up to {apy}% APY."
                ids.append(i + 1)
        # sources: the pages of the records the answer relies on, each record's best page first, then its other
        # official copies (the same rate on the product page and the rates page), at most three in all
        cited = [i - 1 for i in ids] or [0]
        pages = []
        for depth in range(4):
            for i in cited:
                same = copies[i] if copies else [records[i]]
                if depth < len(same) and (depth == 0 or same[depth].auth >= 0.8):
                    pages.append(same[depth])
        urls = list(dict.fromkeys(x.url for x in pages))[:3]
        srcs = [Source(url=u, snapshot_id=sid, quote=next(x.text for x in pages if x.url == u)[:200]) for u in urls]
        return Answer(text=text, sources=srcs, confident=True), "ok"
