#!/usr/bin/env python3
"""Show the evidence behind every question in our own test sets, straight from the snapshots.

For each question it picks the snapshot that was current on the question's date (the latest one
when there is no date), then prints the exact page file and line where each required value appears.
Matching uses the kit's own `contains`, so "7.59%" matches "7.59%1" the same way the scorer would.

Checks:
- every required value appears in that snapshot (on the expected source page, when one is given)
- for table questions with a `verify_row` label, the value sits on that row, not just on the page
- trap values are shown with where they appear, so you can see why they are wrong for this question
- refusal questions list `verify_absent` phrases that must appear nowhere in that snapshot
- reworded questions have exactly the same required and trap values as their public original

    python experiments/verify_eval_sets.py              # all sets, short output
    python experiments/verify_eval_sets.py --id ho-13   # one question in full
    python experiments/verify_eval_sets.py --all-lines  # every matching line, not just the best

Exit code is 1 if any question fails, so it can be used as a check before trusting a run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))

from scoring import contains  # noqa: E402

SNAPS = ROOT / "snapshots"
SETS = ROOT / "experiments" / "eval_sets"
PUBLIC = ROOT / "eval" / "public_eval.jsonl"
STOP = set("what which does do the a an on in of for is are your you my with how much can i at to get dupaco "
           "and or was did as there any now it me tell".split())


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


class Snapshot:
    def __init__(self, d: Path):
        m = json.loads((d / "manifest.json").read_text())
        self.id, self.taken_on, self.dir = m["snapshot_id"], m["taken_on"], d
        self.pages = {p["url"].rstrip("/"): p["file"] for p in m["pages"]}
        self._lines: dict[str, list[str]] = {}

    def lines(self, file: str) -> list[str]:
        if file not in self._lines:
            text = (self.dir / file).read_text(encoding="utf-8").replace("\\r\\n", "\n").replace("\\n", "\n")
            self._lines[file] = text.splitlines()
        return self._lines[file]


def snapshot_for(snaps: list[Snapshot], as_of: str | None) -> Snapshot | None:
    if not as_of:
        return snaps[-1]
    ok = [s for s in snaps if s.taken_on <= as_of]
    return ok[-1] if ok else None


def qwords(q: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9$,.]+", q.lower().replace("-", " ")) if w not in STOP and len(w) > 1}


def find(snap: Snapshot, value: str, files: list[str], question: str):
    """All (file, line_no, line, relevance) where `value` appears; most question-like lines first."""
    words = qwords(question)
    hits = []
    for f in files:
        for i, line in enumerate(snap.lines(f), 1):
            if contains(line, value):
                rel = sum(1 for w in words if w in line.lower())
                hits.append((f, i, line.strip(), rel))
    return sorted(hits, key=lambda h: -h[3])


def _dash(s: str) -> str:
    return s.replace("–", "-").replace("—", "-")


def show(hit, width=150) -> str:
    f, i, line, _ = hit
    line = line if len(line) <= width else line[:width] + " ..."
    return f"{f}:{i}  {line}"


def check_item(it, snaps, all_lines: bool, verbose: bool) -> bool:
    snap = snapshot_for(snaps, it.get("as_of"))
    head = f"{it['id']}  [{it['category']}]  {it['question']}"
    if snap is None:
        print(f"FAIL  {head}\n      no snapshot existed on {it['as_of']}")
        return False
    out, ok = [], True
    out.append(f"      snapshot: {snap.id}" + (f"  (as of {it['as_of']})" if it.get("as_of") else "  (latest)"))
    all_files = sorted(set(snap.pages.values()))
    src_files = []
    for u in it.get("expected_sources", []):
        f = snap.pages.get(u.split("#")[0].rstrip("/"))
        if f:
            src_files.append(f)
        else:
            ok = False
            out.append(f"      MISSING expected source page in this snapshot: {u}")
    n = None if all_lines else 2

    for v in it.get("must_include", []):
        if v.startswith("dupaco.com/") or v.startswith("join."):
            url = "https://www." + v if v.startswith("dupaco.com/") else "https://" + v
            found = url.rstrip("/") in snap.pages
            out.append(f"      {'ok  ' if found else 'MISS'} link {v}: {'a crawled page' if found else 'not a crawled page'}")
            ok &= found
            continue
        hits = find(snap, v, src_files or all_files, it["question"])
        where = "expected source page" if src_files else "snapshot"
        row = it.get("verify_row")
        if row and hits:  # the value must sit on the named table row, not just anywhere on the page
            on_row = [h for h in hits if _dash(row.lower()) in _dash(h[2].lower())]
            if not on_row:
                out.append(f"      MISS {v!r} is in the {where} but never on a line containing row {row!r}; e.g. {show(hits[0], 110)}")
                ok = False
                continue
            out.append(f"      ok   {v!r} on row {row!r}:")
            out += [f"             {show(h)}" for h in on_row[:n]]
            continue
        if not hits and src_files:  # say whether it exists elsewhere, to tell a wrong source from a wrong value
            elsewhere = find(snap, v, all_files, it["question"])
            out.append(f"      MISS {v!r} not on the expected source page"
                       + (f"; found on {len({h[0] for h in elsewhere})} other pages" if elsewhere else "; not in snapshot at all"))
            ok = False
            continue
        if not hits:
            out.append(f"      MISS {v!r} not found anywhere in the snapshot")
            ok = False
            continue
        out.append(f"      ok   {v!r} in {where}: {len(hits)} line(s); best match:")
        out += [f"             {show(h)}" for h in hits[:n]]

    for v in it.get("must_not_include", []):
        hits = find(snap, v, src_files or all_files, it["question"])
        note = f"{len(hits)} line(s), e.g. {show(hits[0], 110)}" if hits else "not on that page"
        out.append(f"      trap {v!r}: {note}")

    if it["category"] == "should_refuse":
        terms = it.get("verify_absent", [])
        if not terms:
            out.append("      WARN refusal question has no verify_absent phrases to check")
        for t in terms:  # whole-word match, so "9 month" does not match "19 months"
            pat = re.compile(r"(?<![\w.])" + re.escape(t) + (r"(?!\w)" if t[-1].isalnum() else ""), re.I)
            pages = [f for f in all_files if any(pat.search(l) for l in snap.lines(f))]
            out.append(f"      {'ok  ' if not pages else 'FAIL'} {t!r} appears on {len(pages)} page(s)"
                       + (f", e.g. {pages[0]}" if pages else ""))
            ok &= not pages

    print(f"{'ok  ' if ok else 'FAIL'}  {head}")
    if verbose or not ok:
        print("\n".join(out))
    return ok


def original_of(public: list[dict], rw: dict) -> dict | None:
    """rw-016 is the rewording of dup-016."""
    return next((p for p in public if p["id"] == "dup-" + rw["id"][3:]), None)


def check_reworded(public: list[dict], reworded: list[dict], verbose: bool) -> bool:
    ok_all = True
    for it in reworded:
        p = original_of(public, it)
        same = p and all(it.get(k) == p.get(k) for k in
                         ("must_include", "must_not_include", "as_of", "category", "expected_sources"))
        ok_all &= bool(same)
        if verbose or not same:
            print(f"{'ok  ' if same else 'FAIL'}  {it['id']}  {it['question']}\n"
                  f"      original: {p['question'] if p else '(no original!)'}")
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", help="check one question and print everything")
    ap.add_argument("--all-lines", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="only print failures and the summary")
    args = ap.parse_args()

    snaps = sorted((Snapshot(d) for d in SNAPS.iterdir() if (d / "manifest.json").exists()), key=lambda s: s.taken_on)
    heldout = [it for p in sorted(SETS.glob("heldout*_eval.jsonl")) for it in load(p)]  # every held-out set
    reworded = load(SETS / "reworded_eval.jsonl")
    public = load(PUBLIC)

    if args.id:
        it = next((x for x in heldout + reworded + public if x["id"] == args.id), None)
        if not it:
            sys.exit(f"no question with id {args.id}")
        sys.exit(0 if check_item(it, snaps, args.all_lines, True) else 1)

    print("== Held-out questions: is every expected value really in the snapshot?\n")
    ok_h = [check_item(it, snaps, args.all_lines, not args.quiet) for it in heldout]
    print("\n== Reworded questions: same expected values as the public original?\n")
    ok_r = check_reworded(public, reworded, not args.quiet)
    print(f"\nHeld-out: {sum(ok_h)}/{len(ok_h)} verified.  Reworded: {'all match their originals' if ok_r else 'MISMATCH, see above'}.")
    sys.exit(0 if all(ok_h) and ok_r else 1)


if __name__ == "__main__":
    main()
