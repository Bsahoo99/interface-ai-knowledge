"""Search method, alternative: embedding search (needs .venv-embed with fastembed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _embed_variant import build_with, search_embed  # noqa: E402


def build():
    return build_with(search_embed)
