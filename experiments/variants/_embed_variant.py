"""Search method experiments: embedding search and a keyword + embedding hybrid.

Needs the experiment-only environment (`.venv-embed`, with `fastembed` installed); the main
solution does not depend on any of this.

Only the search step changes, exactly like `_search_variant.py`:
- `search_embed`: rank records by cosine similarity between the question vector and each record
  vector (BAAI/bge-small-en-v1.5, run locally on CPU through ONNX).
- `search_hybrid`: combine the BM25 ranking and the embedding ranking with reciprocal rank fusion
  (score = sum of 1 / (60 + rank) over both lists), the standard way to merge two rankings whose
  raw scores are on different scales.

Record vectors are computed while loading (untimed) and cached on disk by text hash, so repeated
experiment runs do not re-embed. The question vector is computed at answer time (timed).
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "variants"))

import _search_variant as sv  # noqa: E402  (also puts harness/ and solution/ on sys.path)
import answerer  # noqa: E402
import kb  # noqa: E402

MODEL = "BAAI/bge-small-en-v1.5"
CACHE = ROOT / "experiments" / ".embed_cache"
RRF_K = 60

_model = None
_vecs: dict[str, np.ndarray] = {}
_current_question = {"text": ""}
_bm25 = kb.Index.search
STATS = {"embed_s": 0.0, "embedded": 0}


def model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(MODEL, cache_dir=str(CACHE / "model"))
    return _model


def _doc_text(c) -> str:
    # The record plus where it lives, so a bare table row still carries its product name.
    return f"{c.title}. {c.context}. {c.text}"[:1500]


def _key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def _load_disk_cache():
    f = CACHE / f"{MODEL.replace('/', '_')}.npz"
    if f.exists() and not _vecs:
        d = np.load(f)
        _vecs.update(zip(d["keys"].tolist(), d["vecs"]))


def _save_disk_cache():
    CACHE.mkdir(parents=True, exist_ok=True)
    keys = list(_vecs)
    np.savez(CACHE / f"{MODEL.replace('/', '_')}.npz", keys=np.array(keys), vecs=np.stack([_vecs[k] for k in keys]))


_orig_init = kb.Index.__init__


def _init_with_vectors(self, chunks):
    _orig_init(self, chunks)
    _load_disk_cache()
    texts = [_doc_text(c) for c in chunks]
    keys = [_key(t) for t in texts]
    todo = [(k, t) for k, t in zip(keys, texts) if k not in _vecs]
    if todo:
        t0 = time.perf_counter()
        for (k, _), v in zip(todo, model().embed([t for _, t in todo], batch_size=64)):
            _vecs[k] = v
        STATS["embed_s"] += time.perf_counter() - t0
        STATS["embedded"] += len(todo)
        _save_disk_cache()
    m = np.stack([_vecs[k] for k in keys]).astype(np.float32)
    self.emb = m / np.linalg.norm(m, axis=1, keepdims=True)


def _qvec():
    v = next(iter(model().query_embed([_current_question["text"]])))
    return v / np.linalg.norm(v)


def search_embed(self, qtoks, k=12):
    sims = self.emb @ _qvec()
    top = np.argsort(-sims)[:k]
    return [(float(sims[i]), self.chunks[i]) for i in top]


def search_hybrid(self, qtoks, k=12):
    depth = max(k, 100)
    lists = [_bm25(self, qtoks, k=depth), search_embed(self, qtoks, k=depth)]
    scores, by_id = {}, {}
    for ranked in lists:
        for r, (_, c) in enumerate(ranked):
            scores[c.cid] = scores.get(c.cid, 0.0) + 1.0 / (RRF_K + r + 1)
            by_id[c.cid] = c
    return sv._rank(scores, by_id, k)


_orig_answer = answerer.answer


def _answer_remembering_question(store, question, as_of=None):
    _current_question["text"] = question
    return _orig_answer(store, question, as_of)


def install(search_fn):
    """Patch the main solution in place: vectors at index build, chosen search, question capture."""
    kb.Index.__init__ = _init_with_vectors
    kb.Index.search = search_fn
    answerer.answer = _answer_remembering_question


def set_question(q: str):
    _current_question["text"] = q


def build_with(search_fn):
    install(search_fn)
    return sv._load_main().build()
