"""Refusals: the main solution with changes to when the reader refuses.

    examples   resolves a contradiction in the prompt: "do not substitute a similar branch" stays for a
               question that names one branch or product, but a general question ("what are your lobby
               hours") is answered with the specific examples the evidence gives, saying they vary
    asof       examples + the date line says the evidence IS the website as it stood on the asked date
    draft      asof + the model writes its answer before deciding whether the evidence answers the
               question (answer first, answerable last in the JSON), instead of deciding first
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _search_variant as sv  # noqa: E402  (puts harness/ and solution/ on sys.path)
import reader  # noqa: E402

OLD_SUB = ("- Use only facts stated in the evidence. If the evidence does not answer the question, set answerable to false. "
           "Do not substitute a similar product, term, branch or card for the one asked about; if the exact thing asked "
           "about is not in the evidence, it is not answerable.\n")
NEW_SUB = ("- Use only facts stated in the evidence. If the evidence does not answer the question, set answerable to false. "
           "When the question names one specific product, term, branch or card, do not substitute a similar one; if that "
           "exact thing is not in the evidence, it is not answerable.\n"
           "- When the question asks generally (for example \"what are your lobby hours\" or \"what are the drive-up hours\") "
           "and the evidence gives the answer for specific branches, products or accounts, answer with those as examples "
           "and say that it can vary; that is an answer, not a refusal.\n")
OLD_PART = "Refuse only when nothing in the evidence addresses the question."
NEW_PART = ("Refuse only when nothing in the evidence addresses the question. A matching row or sentence is an answer "
            "even when it uses different units or words (120 months = 10 years).")


def when_asof(as_of, taken_on):
    if as_of:
        return (f"The question asks about {as_of}. The evidence below is the website exactly as it stood on that date "
                f"(the most recent crawl before it, taken {taken_on}); treat its values as the ones in force on {as_of}.")
    return f"The evidence is from the current website (crawled {taken_on})."


DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "integer"}},
        "answerable": {"type": "boolean"},
    },
    "required": ["answer", "evidence_ids", "answerable"],
    "additionalProperties": False,
}
DRAFT_NOTE = ("\n- Write the answer first, then set answerable: true if the answer is supported by the evidence, "
              "false if the evidence does not address the question (then the answer text is discarded).")


def install(name: str):
    # The asof variant was adopted (solution/reader.py now has these lines); the swap is a no-op then.
    reader.SYSTEM = reader.SYSTEM.replace(OLD_SUB, NEW_SUB).replace(OLD_PART, NEW_PART)
    if name in ("asof", "draft"):
        reader.Reader.when = staticmethod(when_asof)
    if name == "draft":
        reader.SCHEMA = DRAFT_SCHEMA
        reader.SYSTEM += DRAFT_NOTE
    elif name != "examples" and name != "asof":
        raise ValueError(name)


def build_with(name: str):
    install(name)
    return sv._load_main().build()


# ---- rules only: refuse when the answer does not carry the term the question names ----------------
import re  # noqa: E402

import answerer  # noqa: E402
from contract import Answer  # noqa: E402

TERM = re.compile(r"\b(\d{1,3})[\s-]?(months?|mo|years?|yr)\b", re.I)


def term_missing(question: str, text: str) -> bool:
    """True if the question names a term (72-month, 40-year) that the answer never states, in months or years."""
    for n, unit in TERM.findall(question):
        n = int(n)
        months = n if unit.lower().startswith("mo") else n * 12
        forms = [rf"\b{months}\s*-?\s*(?:months?|mo)\b"]
        if months % 12 == 0:
            forms.append(rf"\b{months // 12}\s*-?\s*(?:years?|yr)\b")
        if not any(re.search(f, text, re.I) for f in forms):
            return True
    return False


def install_term_check():
    orig = answerer.answer

    def answer(store, question, as_of=None):
        a = orig(store, question, as_of)
        if a.confident and term_missing(question, a.text):
            return Answer(text=answerer.REFUSAL, confident=False)
        return a
    answerer.answer = answer
