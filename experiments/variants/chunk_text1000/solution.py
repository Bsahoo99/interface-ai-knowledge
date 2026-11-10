"""Record size variant: text1000 (see _chunk_variant.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _chunk_variant import build_with  # noqa: E402


def build():
    return build_with("text1000")
