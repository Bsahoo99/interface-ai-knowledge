"""AI reader: the main solution with the AI reader (solution/reader.py) writing answers.

Only the answer-writing step changes; ingest, search, page trust and snapshot choice are the main
solution's. With search="hybrid" the BM25 + embeddings search from the search-method experiments is used (needs .venv-embed).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _search_variant as sv  # noqa: E402  (puts harness/ and solution/ on sys.path)
import reader as reader_mod  # noqa: E402


def build_with(search: str = "bm25", **reader_kwargs):
    ev = None
    if search == "hybrid":
        import _embed_variant as ev
        ev.install(ev.search_hybrid)
    ks = sv._load_main().build()
    rd = reader_mod.Reader(**reader_kwargs)

    def answer(question, as_of=None):
        if ev:
            ev.set_question(question)
        return rd.answer(ks.store, question, as_of)

    ks.answer = answer
    ks.reader = rd
    return ks
