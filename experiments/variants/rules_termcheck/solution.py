"""Refusals variant: rules only, refuse when the answer lacks the term the question names (see _refusal_variant.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _search_variant as sv  # noqa: E402
from _refusal_variant import install_term_check  # noqa: E402


def build():
    install_term_check()
    return sv._load_main().build()
