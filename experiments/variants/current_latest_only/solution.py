"""Current-rules ablation: latest_only (see _current_variant.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _current_variant import build_with  # noqa: E402


def build():
    return build_with("latest_only")
