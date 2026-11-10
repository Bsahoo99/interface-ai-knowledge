"""Conflict detection baseline: the main solution with the first conflict detector (one entry per table row and rate type)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _search_variant as sv  # noqa: E402  (puts harness/ and solution/ on sys.path)
import conflicts_v1  # noqa: E402


def build():
    ks = sv._load_main().build()
    ks.conflicts = lambda: conflicts_v1.find(ks.store)
    return ks
