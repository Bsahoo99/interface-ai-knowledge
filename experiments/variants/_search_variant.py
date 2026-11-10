"""Shared helper for the search-method experiments: the main solution with only the search scoring swapped.

Everything else (cleaning, splitting, page trust, row picking, refusals) stays identical, so any
difference in results comes from the search method alone.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "solution"))

import kb  # noqa: E402


def _load_main():
    spec = importlib.util.spec_from_file_location("main_solution", ROOT / "solution" / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rank(scores: dict, by_id: dict, k: int):
    return sorted(((s, by_id[i]) for i, s in scores.items()), key=lambda x: -x[0])[:k]


def search_overlap(self, qtoks, k=12):
    """Count how many distinct question words a record contains (what the starter does)."""
    scores, by_id = defaultdict(float), {}
    for t in dict.fromkeys(qtoks):
        for c in self.inv.get(t, ()):
            scores[c.cid] += 1.0
            by_id[c.cid] = c
    return _rank(scores, by_id, k)


def search_tfidf(self, qtoks, k=12):
    """Classic TF-IDF: rare words count more, repeated words count a bit more, long records are normalised."""
    scores, by_id = defaultdict(float), {}
    for t in dict.fromkeys(qtoks):
        idf = math.log((self.N + 1) / (self.df.get(t, 0) + 1)) + 1
        for c in self.inv.get(t, ()):
            scores[c.cid] += (1 + math.log(c.toks[t])) * idf
            by_id[c.cid] = c
    for cid, c in by_id.items():
        scores[cid] /= math.sqrt(sum(c.toks.values()))
    return _rank(scores, by_id, k)


def build_with(search_fn):
    kb.Index.search = search_fn
    return _load_main().build()
