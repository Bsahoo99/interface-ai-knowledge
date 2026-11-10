#!/usr/bin/env bash
# Run one variant with the reader (key from .env, proxy variables removed) on the four question sets
# through the kit's runner, and print accuracy and p95 per set. Usage: experiments/run_variant_sets.sh <variant dir> <tag>
set -euo pipefail
cd "$(dirname "$0")/.."
v="$1"; tag="$2"
mkdir -p "experiments/reader_logs/$tag"
for e in eval/public_eval.jsonl experiments/eval_sets/reworded_eval.jsonl experiments/eval_sets/heldout_eval.jsonl experiments/eval_sets/heldout2_eval.jsonl; do
  n=$(basename "$e" .jsonl)
  r=$(env -u ANTHROPIC_BASE_URL -u ANTHROPIC_AUTH_TOKEN bash -c "set -a; . ./.env; set +a; \
    READER_LOG=experiments/reader_logs/$tag/$n.jsonl .venv-embed/bin/python harness/run_eval.py \
    --submission $v --eval $e --out experiments/scratch_runs/$tag-$n" 2>&1 | grep -E '"accuracy"|"p95_s"' | tr -s ' ' | tr '\n' ' ')
  echo "[$tag] $n $r"
done
