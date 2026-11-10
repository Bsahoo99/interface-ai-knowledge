# Experiments

The code and question sets used to compare design choices; the solution itself does not depend on anything here.

- `eval_sets/`: our own question sets in the kit's format.
  `reworded_eval.jsonl` asks each public question differently, `heldout_eval.jsonl` and `heldout2_eval.jsonl` cover products and dates the public set does not, `blind_eval.jsonl` covers topics never used for tuning, and `conflicts_gold.json` lists the 8 contradictions we found by hand in the newest snapshot.
- `verify_eval_sets.py` and `verify_conflicts.py` check every expected value against the snapshot file and line it comes from.
- `variants/`: copies of the solution with one choice changed (search method, record size, "current" rules, refusal behaviour, conflict granularity, reader model); each folder's `solution.py` says what it changes.
- `run.py` scores one variant with the kit's own scoring code and logs it; `run_reader.sh` and `run_variant_sets.sh` run a variant with the reader on all question sets.
- `results_*.csv`: logged runs for load time against question time, search methods, the question sets and reader models; new runs of `run.py` are appended to `results.csv`.
- The other scripts measure one thing each without API calls (search recall, refusals, evidence given to the reader); each says what it measures at the top.

Run a variant on a question set, from the repo root:

```bash
python experiments/run.py --variant experiments/variants/search_tfidf --eval experiments/eval_sets/reworded_eval.jsonl --label "TF-IDF, reworded"
```

Two small debugging helpers sit in `scripts/` at the repo root: `dbg.py` asks the solution ad-hoc questions, and `fails.py` prints the questions the last kit run got wrong.

The embedding variants need a separate environment with `fastembed`; the reader variants need `ANTHROPIC_API_KEY`.
