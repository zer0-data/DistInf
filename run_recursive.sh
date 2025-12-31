#!/usr/bin/env bash
# Runs evaluation over RMT-team/babilong tasks qa1..qa10
# Configs: 16k (budget 4k) and 64k (budget 8k)
# Methods: exact, lsh (frequency_rank, magicpig_baseline), hybrid permutations
# Compression mode: recursive only
# Block size: 1024

set -euo pipefail

MODEL_PATH=${1:-"${MODEL_PATH:-}"}
if [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 <model_path>"
  echo "Or set MODEL_PATH environment variable."
  exit 1
fi

PY=${PYTHON:-python}

TASKS=(qa1 qa2 qa3 qa4 qa5 qa6 qa7 qa8 qa9 qa10)
CONFIGS=(16k 64k)

run_cmd() {
  echo "+ $*"
  # Do not fail the whole batch on a single run; continue
  "$@" || true
}

for cfg in "${CONFIGS[@]}"; do
  if [ "$cfg" = "16k" ]; then
    BUDGET=4096
  else
    BUDGET=8192
  fi

  for task in "${TASKS[@]}"; do
    echo "\n=== CONFIG=$cfg TASK=$task BUDGET=$BUDGET ==="

    # 1) Exact (eager)
    run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
      --method exact --backend eager --budget $BUDGET --compression_mode recursive \
      --dataset_config $cfg --dataset_split $task --block_size 1024

    # 2) LSH: frequency_rank (flash)
    run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
      --method lsh --lsh_mode frequency_rank --backend flash --budget $BUDGET --compression_mode recursive \
      --dataset_config $cfg --dataset_split $task --block_size 1024

    # 3) LSH: magicpig_baseline (flash)
    run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
      --method lsh --lsh_mode magicpig_baseline --backend flash --budget $BUDGET --compression_mode recursive \
      --dataset_config $cfg --dataset_split $task --block_size 1024

    # 4) Hybrid permutations
    # a) exact + frequency_rank
    run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
      --method hybrid --hybrid_primary exact --hybrid_secondary lsh --lsh_mode frequency_rank \
      --hybrid_ratio 0.5 --backend eager --budget $BUDGET --compression_mode recursive \
      --dataset_config $cfg --dataset_split $task --block_size 1024

    # For ratio 0.75, run both orders (primary/secondary swapped)
    run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
      --method hybrid --hybrid_primary exact --hybrid_secondary lsh --lsh_mode frequency_rank \
      --hybrid_ratio 0.75 --backend eager --budget $BUDGET --compression_mode recursive \
      --dataset_config $cfg --dataset_split $task --block_size 1024
    run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
      --method hybrid --hybrid_primary lsh --hybrid_secondary exact --lsh_mode frequency_rank \
      --hybrid_ratio 0.75 --backend eager --budget $BUDGET --compression_mode recursive \
      --dataset_config $cfg --dataset_split $task --block_size 1024

    # b) exact + magicpig_baseline
    run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
      --method hybrid --hybrid_primary exact --hybrid_secondary lsh --lsh_mode magicpig_baseline \
      --hybrid_ratio 0.5 --backend eager --budget $BUDGET --compression_mode recursive \
      --dataset_config $cfg --dataset_split $task --block_size 1024

    run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
      --method hybrid --hybrid_primary exact --hybrid_secondary lsh --lsh_mode magicpig_baseline \
      --hybrid_ratio 0.75 --backend eager --budget $BUDGET --compression_mode recursive \
      --dataset_config $cfg --dataset_split $task --block_size 1024
    run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
      --method hybrid --hybrid_primary lsh --hybrid_secondary exact --lsh_mode magicpig_baseline \
      --hybrid_ratio 0.75 --backend eager --budget $BUDGET --compression_mode recursive \
      --dataset_config $cfg --dataset_split $task --block_size 1024

    # c) magicpig + frequency_rank
    # Per instructions, use flash backend for this hybrid. We'll run variants to approximate both orders.
    # Run with primary magicpig (lsh_mode magicpig_baseline)
    run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
      --method hybrid --hybrid_primary lsh --hybrid_secondary lsh --lsh_mode magicpig_baseline \
      --hybrid_ratio 0.5 --backend flash --budget $BUDGET --compression_mode recursive \
      --dataset_config $cfg --dataset_split $task --block_size 1024

    # Run with primary frequency_rank (lsh_mode frequency_rank)
    run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
      --method hybrid --hybrid_primary lsh --hybrid_secondary lsh --lsh_mode frequency_rank \
      --hybrid_ratio 0.5 --backend flash --budget $BUDGET --compression_mode recursive \
      --dataset_config $cfg --dataset_split $task --block_size 1024

    # Ratio 0.75 both orders (approximate by switching lsh_mode between runs)
    run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
      --method hybrid --hybrid_primary lsh --hybrid_secondary lsh --lsh_mode magicpig_baseline \
      --hybrid_ratio 0.75 --backend flash --budget $BUDGET --compression_mode recursive \
      --dataset_config $cfg --dataset_split $task --block_size 1024
    run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
      --method hybrid --hybrid_primary lsh --hybrid_secondary lsh --lsh_mode frequency_rank \
      --hybrid_ratio 0.75 --backend flash --budget $BUDGET --compression_mode recursive \
      --dataset_config $cfg --dataset_split $task --block_size 1024

    # small pause
    sleep 1
  done
done

echo "\nAll runs queued/completed (errors ignored to continue batch)."
