#!/usr/bin/env bash
# Simplified evaluation script based on user request.

set -euo pipefail

# Set MODEL_PATH to the default value from run_single_sample.py
MODEL_PATH="gradientai/Llama-3-8B-Instruct-Gradient-1048k"

PY=${PYTHON:-python}

TASKS=(qa1 qa2 qa6)

run_cmd() {
  echo "+ $*"
  # Do not fail the whole batch on a single run; continue
  "$@" || true
}

echo "Starting evaluation with MODEL_PATH: $MODEL_PATH"

# 1) Method: exact, on qa1, qa2, qa6
echo "\n=== Running with method: exact ==="
for task in "${TASKS[@]}"; do
  echo "\n--- CONFIG=128k TASK=$task BUDGET=1024 ---"
  run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
    --method exact \
    --dataset_config 128k --dataset_split "$task"
  sleep 1 # small pause
done

# 2) Method: hybrid, lsh_mode: frequency_rank, on qa1, qa2, qa6
echo "\n=== Running with method: hybrid, lsh_mode: frequency_rank ==="
for task in "${TASKS[@]}"; do
  echo "\n--- CONFIG=128k TASK=$task BUDGET=1024 ---"
  run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
    --method hybrid --lsh_mode frequency_rank \
    --dataset_config 128k --dataset_split "$task"
  sleep 1 # small pause
done

# 3) Method: hybrid, lsh_mode: magicpig_baseline, on qa1, qa2, qa6
echo "\n=== Running with method: hybrid, lsh_mode: magicpig_baseline ==="
for task in "${TASKS[@]}"; do
  echo "\n--- CONFIG=128k TASK=$task BUDGET=1024 ---"
  run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
    --method hybrid --lsh_mode magicpig_baseline \
    --dataset_config 128k --dataset_split "$task"
  sleep 1 # small pause
done

echo "\nAll specified runs completed."
