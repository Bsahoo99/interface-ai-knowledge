"""Conflict detection: the main solution with the new detector, but one conflict per table row and rate column."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _search_variant as sv  # noqa: E402  (puts harness/ and solution/ on sys.path)
import conflicts  # noqa: E402

conflicts.GROUP_BY_TABLE = False


def build():
    return sv._load_main().build()
