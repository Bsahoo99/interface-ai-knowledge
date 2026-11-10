"""Search method, alternative: plain word overlap (the starter's method) instead of BM25."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _search_variant import build_with, search_overlap  # noqa: E402


def build():
    return build_with(search_overlap)
