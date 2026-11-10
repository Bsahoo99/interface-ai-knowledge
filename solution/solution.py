"""Knowledge system that answers from the *current* website, with sources and honest refusals.

Entry point for the grader: build() -> object with ingest / answer / conflicts.
Search, page trust, snapshot choice and conflicts are deterministic and standard-library only.
When ANTHROPIC_API_KEY is set and the `anthropic` package is installed, an AI reader (reader.py,
Claude Haiku 5.5) writes the answer from the search results; any failure falls back to the rules.
Set KNOWLEDGE_READER=off to force the rules-only answers.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(next(d / "harness" for d in Path(__file__).resolve().parents
                            if (d / "harness" / "contract.py").exists())))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import Answer, Snapshot  # noqa: E402
from kb import Store  # noqa: E402
import answerer  # noqa: E402
import conflicts as conflict_finder  # noqa: E402
import hours  # noqa: E402

READER_MODEL = "claude-haiku-5-5"
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")
READER_TIMEOUT_S = 10.0  # per attempt; the client retries once, well inside the grader's 60 s limit


def make_reader():
    """The AI reader when it can run (key set, package installed, not switched off), else None."""
    if os.getenv("KNOWLEDGE_READER", "").strip().lower() in ("off", "0", "false", "no"):
        return None
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return None
    from reader import Reader
    return Reader(model=READER_MODEL, thinking_off=True, timeout_s=READER_TIMEOUT_S)


class KnowledgeSystem:
    def __init__(self):
        self.store = Store()
        self._conflicts = None
        self.facts: dict[int, list[dict]] = {}   # snapshot -> facts settled by its newest dated page
        self.reader = make_reader()

    def ingest(self, snapshot: Snapshot) -> None:
        self.store.add_snapshot(snapshot)
        self._conflicts = None
        # Facts that dated pages disagree on (the branch count) become records stating only the current
        # value, so search finds the settled value instead of one old news release.
        snap = len(self.store.snapshots) - 1
        self.facts[snap] = conflict_finder.current_facts(self.store, snap)
        records = []
        for f in self.facts[snap]:
            # a short record that search finds for "how many branches", and the newest page's own paragraph,
            # which also says where the credit union serves members
            records.append({"url": f["url"], "title": f"Current number of {f['name']}", "text": f["text"],
                            "context": f"how many {f['noun']}", "key": conflict_finder.RESOLVED})
            serves = next((x for x in SENTENCE.split(f["paragraph"]) if re.search(r"\bserves\b", x)), "")
            if serves:
                # short, so a one-word question ("who can join?") finds it; the paragraph below is long
                records.append({"url": f["url"], "title": "Where the credit union serves members",
                                "text": serves.strip(), "key": conflict_finder.RESOLVED,
                                "context": "who can join, where members live or work"})
            if f["paragraph"]:
                records.append({"url": f["url"], "title": "About the credit union, newest version",
                                "text": f["paragraph"], "key": conflict_finder.RESOLVED,
                                "context": "where the credit union serves members: its counties and states, "
                                           "the area members live or work in to be eligible to join"})
        records += hours.records(snapshot.pages)   # general lobby and drive-up hours
        self.store.add_records(snap, records)

    def answer(self, question: str, as_of=None) -> Answer:
        # answer() and conflicts() must never disagree: a sentence giving a count that the conflict check
        # settled differently ("23 branches" when the newest dated page says 22) is dropped; if nothing is
        # left, the rule-based answer is tried the same way, then the refusal.
        ans = self._without_outdated(self._answer(question, as_of), question, as_of)
        if ans is None and self.reader is not None:
            ans = self._without_outdated(answerer.answer(self.store, question, as_of), question, as_of)
        return ans or Answer(text=answerer.REFUSAL, confident=False)

    def _answer(self, question: str, as_of=None) -> Answer:
        if self.reader is not None:
            return self.reader.answer(self.store, question, as_of)
        return answerer.answer(self.store, question, as_of)

    def _without_outdated(self, ans: Answer, question: str, as_of):
        """The answer minus sentences that give an outdated count; None if no sentence is left."""
        facts = self.facts.get(self.store.snapshot_for(as_of), ())
        if not ans.confident or not facts:
            return ans

        def outdated(sentence: str) -> bool:
            for f in facts:
                asks_count = re.search(rf"\bhow many\b.*\b{f['name'].split()[0]}", question, re.I)   # "branch"
                for m in f["answer_pattern"].finditer(sentence):
                    n = m.group(1)
                    if n in f["outdated"] or (asks_count and n != f["value"]):
                        return True
            return False

        sents = SENTENCE.split(ans.text.strip())
        keep = [s for s in sents if not outdated(s)]
        if len(keep) == len(sents):
            return ans
        if not keep:
            return None
        return Answer(text=" ".join(keep), sources=ans.sources, confident=True, conflict_note=ans.conflict_note)

    def conflicts(self):
        if self._conflicts is None:
            self._conflicts = conflict_finder.find(self.store)
        return self._conflicts


def build():
    return KnowledgeSystem()
