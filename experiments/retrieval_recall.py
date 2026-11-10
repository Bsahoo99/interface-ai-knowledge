#!/usr/bin/env python3
"""Does the needed fact appear in the top-k search results at all?

For each public question that has required values, a question counts as "found at k" when every
required value (matched with the scorer's own `contains`) appears somewhere in the text of the top-k
records the search returns, after the same page-trust weighting the answerer uses.
This separates search failures from answer-building failures.

    python experiments/retrieval_recall.py [eval.jsonl]   (default: the public questions)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "solution"))
sys.path.insert(0, str(ROOT / "experiments" / "variants"))

from run_eval import load_snapshots  # noqa: E402
from scoring import contains  # noqa: E402
import kb  # noqa: E402
import answerer  # noqa: E402
from kb import Store, tokens  # noqa: E402
import _search_variant as sv  # noqa: E402

METHODS = {"BM25": kb.Index.search, "TF-IDF": sv.search_tfidf, "Word overlap": sv.search_overlap}
KS = (1, 3, 5, 10, 20)

try:  # embedding methods only exist in the experiment environment (.venv-embed)
    import _embed_variant as ev
    METHODS.update({"Embeddings": ev.search_embed, "Hybrid (BM25 + embeddings)": ev.search_hybrid})
except ImportError:
    ev = None


def main():
    eval_file = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "eval" / "public_eval.jsonl"
    items = [json.loads(l) for l in eval_file.read_text().splitlines() if l.strip()]
    items = [it for it in items if it.get("must_include") and it["category"] != "should_refuse"]
    if ev:
        ev.install(kb.Index.search)  # vectors get built with each index; search is set per method below
    store = Store()
    for s in load_snapshots(ROOT / "snapshots"):
        store.add_snapshot(s)

    print(f"{len(items)} questions with required values\n")
    print("| method | " + " | ".join(f"found in top {k}" for k in KS) + " |")
    print("|---|" + "---|" * len(KS))
    misses = {}
    for name, fn in METHODS.items():
        kb.Index.search = fn
        found = {k: 0 for k in KS}
        miss = []
        for it in items:
            ix = store.index(store.snapshot_for(it.get("as_of")))
            q = it["question"]
            if ev:
                ev.set_question(q)
            qt = answerer._expand(tokens(q))
            ranked, _ = answerer._rank(ix, qt, answerer._is_rate_question(q), k=40)
            for k in KS:
                text = " ".join(c.text for _, c in ranked[:k])
                if all(contains(text, t) for t in it["must_include"]):
                    found[k] += 1
            if not all(contains(" ".join(c.text for _, c in ranked[:20]), t) for t in it["must_include"]):
                miss.append(it["id"])
        misses[name] = miss
        print(f"| {name} | " + " | ".join(f"{found[k]}/{len(items)}" for k in KS) + " |")
    print()
    for name, m in misses.items():
        print(f"{name}: not found even in top 20: {m}")


if __name__ == "__main__":
    main()
