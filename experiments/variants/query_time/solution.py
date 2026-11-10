"""Load time vs question time, alternative A: do the work at QUESTION time instead of load time.

ingest() only keeps the raw pages. Every answer() call cleans and splits the pages of the snapshot it
needs and builds a fresh search index, then answers with exactly the same logic as the main solution.
Accuracy should therefore match the main solution; only where the time is spent changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "solution"))

from kb import Store  # noqa: E402
import answerer  # noqa: E402
import conflicts as conflict_finder  # noqa: E402


class QuestionTimeSystem:
    def __init__(self):
        self.snaps = []          # raw snapshots, untouched

    def ingest(self, snapshot) -> None:
        self.snaps.append(snapshot)

    def _store_for(self, as_of):
        idx = len(self.snaps) - 1
        if as_of:
            idx = max((i for i, s in enumerate(self.snaps) if s.taken_on <= as_of), default=None)
            if idx is None:
                return Store()
        store = Store()
        store.add_snapshot(self.snaps[idx])     # all the preparation happens here, per question
        return store

    def answer(self, question, as_of=None):
        return answerer.answer(self._store_for(as_of), question, None)

    def conflicts(self):
        store = Store()
        for s in self.snaps:
            store.add_snapshot(s)
        return conflict_finder.find(store)


def build():
    return QuestionTimeSystem()
