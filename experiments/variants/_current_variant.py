"""Current-rules: the main solution (rules, or the reader when a key is set) with one "current" rule turned off.

    no_trust      every page is trusted equally (no authority weighting, every record labelled official)
    no_dates      no "effective as of" dates are read, so nothing can prefer the later date
    no_expired    ended or expired promotions are no longer filtered out (rules) or flagged (reader prompt)
    latest_only   as_of is ignored: every question is answered from the newest snapshot
    trust_rates   page trust weights search only for rate questions (where stale copies exist);
                  other questions rank all pages equally, so a fact found only on a news page can win
    trust_soft    page trust squared for rate questions, page trust (not squared) for the rest
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _search_variant as sv  # noqa: E402  (puts harness/ and solution/ on sys.path)
import answerer  # noqa: E402
import kb  # noqa: E402
import reader  # noqa: E402


def build_with(ablation: str):
    if ablation == "no_trust":
        kb.authority = lambda url: 1.0
    elif ablation == "no_dates":
        kb.parse_effective = lambda text: None
    elif ablation == "no_expired":
        answerer.EXPIRED = re.compile(r"(?!x)x")
        reader.SYSTEM = reader.SYSTEM.replace("- Promotions or rates described as ended or expired are not current.\n", "")
    elif ablation == "latest_only":
        orig = kb.Store.snapshot_for
        kb.Store.snapshot_for = lambda self, as_of=None: orig(self, None)
    elif ablation in ("trust_rates", "trust_soft"):
        other = 0.0 if ablation == "trust_rates" else 1.0

        def _rank(ix, qtoks, rate_q=False, k=40):
            ranked = ix.search(qtoks, k=k)
            out = []
            for s, c in ranked:
                w = s * (c.auth ** (2.0 if rate_q else other))
                if rate_q:
                    if not re.search(r"\d\.\d+\s*%", c.text):
                        continue
                    if c.kind == "row":
                        w *= 2.2
                out.append((w, c))
            return sorted(out, key=lambda x: -x[0]), ranked
        answerer._rank = _rank
    else:
        raise ValueError(ablation)
    return sv._load_main().build()
