#!/usr/bin/env bash
# Run one reader variant on the three question sets. Usage: experiments/run_reader.sh reader_haiku
# Reads ANTHROPIC_API_KEY from .env and ignores any ANTHROPIC_BASE_URL proxy, so the key only goes to api.anthropic.com.
set -euo pipefail
cd "$(dirname "$0")/.."
v="$1"
for e in eval/public_eval.jsonl experiments/eval_sets/reworded_eval.jsonl experiments/eval_sets/heldout_eval.jsonl experiments/eval_sets/heldout2_eval.jsonl; do
  n=$(basename "$e" .jsonl)
  env -u ANTHROPIC_BASE_URL -u ANTHROPIC_AUTH_TOKEN bash -c "set -a; . ./.env; set +a; \
    READER_LOG=experiments/reader_logs/${v}__${n}.jsonl .venv-embed/bin/python experiments/run.py \
    --variant experiments/variants/$v --eval $e --label '$v on $n'" | grep -E '"(accuracy|consistency|speed_score|score_no_conflicts|p50_ms|p95_ms)"|by category' | sed "s/^/[$n] /"
done
