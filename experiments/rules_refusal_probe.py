#!/usr/bin/env python3
"""Refusals, rules only: try refusal checks offline on all four question sets (no API calls).

Builds the store once, answers every question with the rule-based answerer, then applies each candidate
check on top and scores with the kit's own score_item. Prints, per check: all-117 accuracy, should-refuse
questions refused, and which questions it changed (a new refusal of an answerable question is a loss).

    python experiments/rules_refusal_probe.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "solution"))
from run_eval import load_snapshots  # noqa: E402
from scoring import score_item  # noqa: E402
from contract import Answer  # noqa: E402
import answerer  # noqa: E402
from kb import Store, tokens  # noqa: E402

SETS = [ROOT / "eval" / "public_eval.jsonl"] + [
    ROOT / "experiments" / "eval_sets" / f"{n}_eval.jsonl" for n in ("reworded", "heldout", "heldout2")]

# words that describe the kind of question, not the thing asked about
GENERIC = set(tokens(
    "rate rates apr apy interest dividend yield percent loan loans account accounts card cards hours hour open "
    "close time branch branches location locations lobby drive up fee fees minimum balance term month months "
    "year years cost charge pay get give need much number phone website link page online member members "
    "credit union dupaco purchase new used find best"))
TIME = re.compile(r"\b\d{1,2}(?::\d\d)?\s*(?:a\.?m\.?|p\.?m\.?)|\bnoon\b", re.I)
HOURS_Q = re.compile(r"\b(hours?|open|close[sd]?|what time)\b", re.I)


def evidence_words(store, snap, ans: Answer) -> set:
    """Words of the answer, plus the titles and headings of the pages it cites."""
    words = set(tokens(ans.text))
    urls = {s.url for s in ans.sources or []}
    for c in store.index(snap).chunks:
        if c.url in urls:
            words |= set(tokens(f"{c.title} {c.context}"))
    return words


def name_missing(store, snap, q: str, ans: Answer, top: int) -> bool:
    ix = store.index(snap)
    qt = [t for t in dict.fromkeys(tokens(q)) if t not in GENERIC and not re.fullmatch(r"[\d$%.,]+", t)]
    qt = [t for t in qt if ix.df.get(t, 0) > 0]
    qt.sort(key=lambda t: -ix.idf(t))
    have = evidence_words(store, snap, ans)
    return any(t not in have for t in qt[:top])


def time_missing(q: str, ans: Answer) -> bool:
    return bool(HOURS_Q.search(q)) and not TIME.search(ans.text)


PRODUCT_NOUN = re.compile(r"\b([A-Za-z][A-Za-z-]+)\s+(?:loans?|cards?|accounts?|certificates?|branch(?:es)?|mortgages?)\b", re.I)
CALENDAR = set("january february march april may june july august september october november december "
               "monday tuesday wednesday thursday friday saturday sunday".split())
HOURS_Q2 = re.compile(r"\b(hours?|what time)\b", re.I)


def names(q: str) -> list[str]:
    """What the question names: capitalised words (not the first word or a date) and the word before a product noun."""
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+/\d+", q)
    out = [w for k, w in enumerate(words) if k > 0 and (w[0].isupper() or "/" in w) and w.lower() not in CALENDAR
           and w not in ("I", "Dupaco", "Dupaco's")]
    out += [m.group(1) for m in PRODUCT_NOUN.finditer(q)]
    toks = []
    for w in out:
        toks += [t for t in tokens(w) if t not in GENERIC]
    return list(dict.fromkeys(toks))


def name_missing2(store, snap, q: str, ans: Answer) -> bool:
    if ans.text.startswith("You can use http"):
        return False                       # link answers are a URL, checked elsewhere
    have = evidence_words(store, snap, ans)
    ix = store.index(snap)
    for t in names(q):
        if ix.df.get(t, 0) == 0:
            continue                       # never on the site: the existing unseen-name check handles it
        forms = {t, t.rstrip("s"), t + "s"}
        if not forms & have:
            return True
    return False


def time_missing2(q: str, ans: Answer) -> bool:
    return bool(HOURS_Q2.search(q)) and not TIME.search(ans.text)


CHECKS = {
    "shipped": lambda st, sn, q, a: False,
    "names": lambda st, sn, q, a: name_missing2(st, sn, q, a),
    "time2": lambda st, sn, q, a: time_missing2(q, a),
    "names + time2": lambda st, sn, q, a: name_missing2(st, sn, q, a) or time_missing2(q, a),
    "name top1": lambda st, sn, q, a: name_missing(st, sn, q, a, 1),
    "name top2": lambda st, sn, q, a: name_missing(st, sn, q, a, 2),
    "time": lambda st, sn, q, a: time_missing(q, a),
    "name top1 + time": lambda st, sn, q, a: name_missing(st, sn, q, a, 1) or time_missing(q, a),
    "name top2 + time": lambda st, sn, q, a: name_missing(st, sn, q, a, 2) or time_missing(q, a),
}


def main():
    items = [json.loads(l) for f in SETS for l in f.read_text().splitlines() if l.strip()]
    store = Store()
    for s in load_snapshots(ROOT / "snapshots"):
        store.add_snapshot(s)
    base = []
    for it in items:
        a = answerer.answer(store, it["question"], it.get("as_of"))
        base.append((it, store.snapshot_for(it.get("as_of")), a))
    ref = {}
    for name, check in CHECKS.items():
        tot = 0.0
        refused = 0
        changes = []
        for it, snap, a in base:
            b = a
            if a.confident and snap is not None and check(store, snap, it["question"], a):
                b = Answer(text=answerer.REFUSAL, confident=False)
            r = score_item(it, b)
            tot += r["score"]
            if it["category"] == "should_refuse" and r["correctness"] == 1:
                refused += 1
            if name != "shipped" and abs(r["score"] - ref[it["id"]]) > 1e-9:
                changes.append(f"{it['id']} {ref[it['id']]:.2f}->{r['score']:.2f}")
            if name == "shipped":
                ref[it["id"]] = r["score"]
        n_ref = sum(1 for it in items if it["category"] == "should_refuse")
        print(f"{name:18s} all {len(items)}: {tot / len(items):.3f}  should-refuse refused {refused}/{n_ref}  "
              f"changed: {', '.join(changes) or '-'}")


if __name__ == "__main__":
    main()
