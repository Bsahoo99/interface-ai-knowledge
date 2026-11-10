"""AI reader: reader on Claude Haiku 5.5, thinking off, BM25 search"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _reader_variant import build_with  # noqa: E402


def build():
    return build_with(search="bm25", model="claude-haiku-5-5", thinking_off=True)
