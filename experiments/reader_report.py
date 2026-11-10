#!/usr/bin/env python3
"""Summarise reader logs and list the questions a reader variant got wrong.

    python experiments/reader_report.py reader_haiku            # cost, latency, outcomes per set
    python experiments/reader_report.py reader_haiku --fails     # plus each wrong answer
"""
import json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from scoring import score_item  # noqa: E402
from contract import Answer  # noqa: E402

SETS = {"public_eval": ROOT / "eval/public_eval.jsonl", "reworded_eval": ROOT / "experiments/eval_sets/reworded_eval.jsonl",
        "heldout_eval": ROOT / "experiments/eval_sets/heldout_eval.jsonl",
        "heldout2_eval": ROOT / "experiments/eval_sets/heldout2_eval.jsonl"}


def main():
    v, fails = sys.argv[1], "--fails" in sys.argv
    for name, path in SETS.items():
        f = ROOT / f"experiments/reader_logs/{v}__{name}.jsonl"
        if not f.exists():
            continue
        log = [json.loads(l) for l in f.read_text().splitlines()]
        items = {json.loads(l)["question"].strip().lower(): json.loads(l) for l in path.read_text().splitlines() if l.strip()}
        lat = sorted(e["latency_s"] for e in log)
        cost = sum(e.get("cost_usd", 0) for e in log)
        outc = Counter(e["outcome"].split(":")[0] if e["outcome"].startswith("fallback") else e["outcome"] for e in log)
        print(f"\n== {v} on {name}: {len(log)} calls, total ${cost:.4f} (${cost/len(log):.5f}/question), "
              f"latency p50 {lat[len(lat)//2]:.2f}s p95 {lat[min(len(lat)-1, int(.95*len(lat)))]:.2f}s max {lat[-1]:.2f}s, "
              f"avg input {sum(e.get('input_tokens',0) for e in log)/len(log):.0f} tok, outcomes {dict(outc)}")
        for e in log:
            if e["outcome"].startswith("fallback"):
                print(f"   FALLBACK {e['outcome']} | {e['question']}")
        if fails:
            for e in log:
                it = items.get(e["question"].strip().lower())
                if not it:
                    continue
                r = score_item(it, Answer(text=e["answer"], confident=e["outcome"] != "refused"))
                if r["correctness"] < 0.999:
                    print(f"   x {it['id']} [{it['category']}] {r['correctness']:.2f} miss={r['missed']} trap={r['trap_hits']}\n"
                          f"     Q: {it['question']}\n     A: {e['answer'][:260]}")


main()
