#!/usr/bin/env bash
# Runs evaluation over RMT-team/babilong tasks qa1..qa10
# Configs: 16k (budget 4k) and 64k (budget 8k)
# Methods: exact, lsh (frequency_rank, magicpig_baseline), hybrid permutations
# Compression mode: recursive only
# Block size: 1024

set -euo pipefail

# Default values
MODEL_PATH="${MODEL_PATH:-}"
NUM_SAMPLES=100
TARGET_TASK="all"
TARGET_CONFIG="all"
TARGET_METHOD="all"
BLOCK_SIZE=1024

usage() {
  echo "Usage: $0 [options] <model_path>"
  echo "Options:"
  echo "  --method <method>    Target specific method (e.g., exact, lsh_freq, hybrid_exact_freq_0.5). Default: all"
  echo "  --task <task>        Target specific task (e.g., qa1). Default: all"
  echo "  --config <config>    Target specific config (16k or 64k). Default: all"
  echo "  --samples <num>      Number of samples per test. Default: 100"
  echo "  --block_size <size>  Block size. Default: 1024"
  echo "  --help               Show this help message"
  exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --method)
      TARGET_METHOD="$2"
      shift 2
      ;;
    --task)
      TARGET_TASK="$2"
      shift 2
      ;;
    --config)
      TARGET_CONFIG="$2"
      shift 2
      ;;
    --samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --block_size)
      BLOCK_SIZE="$2"
      shift 2
      ;;
    --help)
      usage
      ;;
    *)
      if [ -z "$MODEL_PATH" ]; then
        MODEL_PATH="$1"
        shift
      else
        echo "Unknown argument: $1"
        usage
      fi
      ;;
  esac
done

if [ -z "$MODEL_PATH" ]; then
  echo "Error: MODEL_PATH is required."
  usage
fi

PY=${PYTHON:-python3}

TASKS=(qa1 qa2 qa3 qa4 qa5 qa6 qa7 qa8 qa9 qa10)
if [ "$TARGET_TASK" != "all" ]; then
  TASKS=($TARGET_TASK)
fi

CONFIGS=(16k 64k)
if [ "$TARGET_CONFIG" != "all" ]; then
  CONFIGS=($TARGET_CONFIG)
fi

run_cmd() {
  echo "+ $*"
  "$@" || true
}

# $1: current_id (e.g., "lsh_freq")
# $2: target_filter (e.g., "lsh")
should_run() {
  local current="$1"
  local target="$2"
  
  if [ "$target" = "all" ]; then
    return 0
  fi
  
  # Exact match
  if [ "$current" = "$target" ]; then
    return 0
  fi
  
  # Prefix match (e.g. target="lsh" matches "lsh_freq")
  if [[ "$current" == "$target"* ]]; then
     return 0
  fi
  
  return 1
}

for cfg in "${CONFIGS[@]}"; do
  if [ "$cfg" = "16k" ]; then
    BUDGET=4096
  else
    BUDGET=8192
  fi

  for task in "${TASKS[@]}"; do
    echo -e "\n=== CONFIG=$cfg TASK=$task BUDGET=$BUDGET ==="

    # 1) Exact (eager)
    ID="exact"
    if should_run "$ID" "$TARGET_METHOD"; then
      echo "--- Method: Exact ---"
      run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
        --method exact --backend eager --budget $BUDGET --compression_mode recursive \
        --dataset_config $cfg --dataset_split $task --block_size "$BLOCK_SIZE" --num_samples "$NUM_SAMPLES"
    fi

    # 2) LSH: frequency_rank (flash)
    ID="lsh_freq"
    if should_run "$ID" "$TARGET_METHOD"; then
      echo "--- Method: LSH (Frequency Rank) ---"
      run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
        --method lsh --lsh_mode frequency_rank --backend flash --budget $BUDGET --compression_mode recursive \
        --dataset_config $cfg --dataset_split $task --block_size "$BLOCK_SIZE" --num_samples "$NUM_SAMPLES"
    fi

    # 3) LSH: magicpig_baseline (flash)
    ID="lsh_magicpig"
    if should_run "$ID" "$TARGET_METHOD"; then
      echo "--- Method: LSH (MagicPIG) ---"
      run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
        --method lsh --lsh_mode magicpig_baseline --backend flash --budget $BUDGET --compression_mode recursive \
        --dataset_config $cfg --dataset_split $task --block_size "$BLOCK_SIZE" --num_samples "$NUM_SAMPLES"
    fi

    # 4) Hybrid: Exact + FreqRank
    # 0.5
    ID="hybrid_exact_freq_0.5"
    if should_run "$ID" "$TARGET_METHOD"; then
      echo "--- Method: Hybrid (Exact + FreqRank 0.5) ---"
      run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
        --method hybrid --hybrid_primary exact --hybrid_secondary lsh --lsh_mode frequency_rank \
        --hybrid_ratio 0.5 --backend eager --budget $BUDGET --compression_mode recursive \
        --dataset_config $cfg --dataset_split $task --block_size "$BLOCK_SIZE" --num_samples "$NUM_SAMPLES"
    fi

    # 0.75
    ID="hybrid_exact_freq_0.75"
    if should_run "$ID" "$TARGET_METHOD"; then
      echo "--- Method: Hybrid (Exact + FreqRank 0.75) ---"
      run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
        --method hybrid --hybrid_primary exact --hybrid_secondary lsh --lsh_mode frequency_rank \
        --hybrid_ratio 0.75 --backend eager --budget $BUDGET --compression_mode recursive \
        --dataset_config $cfg --dataset_split $task --block_size "$BLOCK_SIZE" --num_samples "$NUM_SAMPLES"
    fi
      
    ID="hybrid_freq_exact_0.75"
    if should_run "$ID" "$TARGET_METHOD"; then
      echo "--- Method: Hybrid (FreqRank + Exact 0.75) ---"
      run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
        --method hybrid --hybrid_primary lsh --hybrid_secondary exact --lsh_mode frequency_rank \
        --hybrid_ratio 0.75 --backend eager --budget $BUDGET --compression_mode recursive \
        --dataset_config $cfg --dataset_split $task --block_size "$BLOCK_SIZE" --num_samples "$NUM_SAMPLES"
    fi

    # 5) Hybrid: Exact + MagicPIG
    # 0.5
    ID="hybrid_exact_magic_0.5"
    if should_run "$ID" "$TARGET_METHOD"; then
        echo "--- Method: Hybrid (Exact + MagicPIG 0.5) ---"
        run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
          --method hybrid --hybrid_primary exact --hybrid_secondary lsh --lsh_mode magicpig_baseline \
          --hybrid_ratio 0.5 --backend eager --budget $BUDGET --compression_mode recursive \
          --dataset_config $cfg --dataset_split $task --block_size "$BLOCK_SIZE" --num_samples "$NUM_SAMPLES"
    fi

    # 0.75
    ID="hybrid_exact_magic_0.75"
    if should_run "$ID" "$TARGET_METHOD"; then
        echo "--- Method: Hybrid (Exact + MagicPIG 0.75) ---"
        run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
          --method hybrid --hybrid_primary exact --hybrid_secondary lsh --lsh_mode magicpig_baseline \
          --hybrid_ratio 0.75 --backend eager --budget $BUDGET --compression_mode recursive \
          --dataset_config $cfg --dataset_split $task --block_size "$BLOCK_SIZE" --num_samples "$NUM_SAMPLES"
    fi

    ID="hybrid_magic_exact_0.75"
    if should_run "$ID" "$TARGET_METHOD"; then
        echo "--- Method: Hybrid (MagicPIG + Exact 0.75) ---"
        run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
          --method hybrid --hybrid_primary lsh --hybrid_secondary exact --lsh_mode magicpig_baseline \
          --hybrid_ratio 0.75 --backend eager --budget $BUDGET --compression_mode recursive \
          --dataset_config $cfg --dataset_split $task --block_size "$BLOCK_SIZE" --num_samples "$NUM_SAMPLES"
    fi

    # 6) Hybrid: MagicPIG + FreqRank
    # 0.5
    ID="hybrid_magic_freq_0.5"
    if should_run "$ID" "$TARGET_METHOD"; then
        echo "--- Method: Hybrid (MagicPIG + FreqRank 0.5) ---"
        run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
          --method hybrid --hybrid_primary lsh --hybrid_secondary lsh --lsh_mode magicpig_baseline \
          --hybrid_ratio 0.5 --backend flash --budget $BUDGET --compression_mode recursive \
          --dataset_config $cfg --dataset_split $task --block_size "$BLOCK_SIZE" --num_samples "$NUM_SAMPLES"
    fi
    ID="hybrid_freq_magic_0.5"
    if should_run "$ID" "$TARGET_METHOD"; then
        echo "--- Method: Hybrid (FreqRank + MagicPIG 0.5) ---"
        run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
          --method hybrid --hybrid_primary lsh --hybrid_secondary lsh --lsh_mode frequency_rank \
          --hybrid_ratio 0.5 --backend flash --budget $BUDGET --compression_mode recursive \
          --dataset_config $cfg --dataset_split $task --block_size "$BLOCK_SIZE" --num_samples "$NUM_SAMPLES"
    fi

    # 0.75
    ID="hybrid_magic_freq_0.75"
    if should_run "$ID" "$TARGET_METHOD"; then
        echo "--- Method: Hybrid (MagicPIG + FreqRank 0.75) ---"
        run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
          --method hybrid --hybrid_primary lsh --hybrid_secondary lsh --lsh_mode magicpig_baseline \
          --hybrid_ratio 0.75 --backend flash --budget $BUDGET --compression_mode recursive \
          --dataset_config $cfg --dataset_split $task --block_size "$BLOCK_SIZE" --num_samples "$NUM_SAMPLES"
    fi
          
    ID="hybrid_freq_magic_0.75"
    if should_run "$ID" "$TARGET_METHOD"; then
        echo "--- Method: Hybrid (FreqRank + MagicPIG 0.75) ---"
        run_cmd $PY run_single_sample.py --model_path "$MODEL_PATH" \
          --method hybrid --hybrid_primary lsh --hybrid_secondary lsh --lsh_mode frequency_rank \
          --hybrid_ratio 0.75 --backend flash --budget $BUDGET --compression_mode recursive \
          --dataset_config $cfg --dataset_split $task --block_size "$BLOCK_SIZE" --num_samples "$NUM_SAMPLES"
    fi

    # small pause
    sleep 1
  done
done

echo -e "\nAll runs queued/completed (errors ignored to continue batch)."
