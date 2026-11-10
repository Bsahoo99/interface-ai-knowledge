"""AI reader: reader on Claude Sonnet 5.5, effort low, BM25 search"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _reader_variant import build_with  # noqa: E402


def build():
    return build_with(search="bm25", model="claude-sonnet-5-5", effort="low")
