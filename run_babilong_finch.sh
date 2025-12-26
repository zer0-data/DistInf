#!/bin/bash

# Define the python script path
SCRIPT_PATH="tests/babilong_finch.py"
OUTPUT_FILE="results.txt"

# Ensure the tests directory exists, just in case
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Could not find $SCRIPT_PATH"
    exit 1
fi

echo "Starting FINCH experiments on Babilong..."
echo "Results will be appended to $OUTPUT_FILE"
echo "----------------------------------------"

# --- Experiment 1: 16k Context ---
# Target retentions: 8k (0.5), 4k (0.25), 2k (0.125)
CONTEXT="16k"
RATIOS=(0.5 0.25 0.125)
CHUNKS=(256 1024)

for chunk in "${CHUNKS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
        echo "[Running] Context: $CONTEXT | Chunk: $chunk | Ratio: $ratio"
        python "$SCRIPT_PATH" --context_size "$CONTEXT" --compression_ratio "$ratio" --chunk_size "$chunk"
    done
done

# --- Experiment 2: 32k Context ---
# Target retentions: 8k (0.25), 4k (0.125), 2k (0.0625)
CONTEXT="32k"
RATIOS=(0.25 0.125 0.0625)
CHUNKS=(256 1024)

for chunk in "${CHUNKS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
        echo "[Running] Context: $CONTEXT | Chunk: $chunk | Ratio: $ratio"
        python "$SCRIPT_PATH" --context_size "$CONTEXT" --compression_ratio "$ratio" --chunk_size "$chunk"
    done
done

# --- Experiment 3: 64k Context ---
# Target retentions: 8k (0.125), 4k (0.0625), 2k (0.03125)
CONTEXT="64k"
RATIOS=(0.125 0.0625 0.03125)
CHUNKS=(256 1024)

for chunk in "${CHUNKS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
        echo "[Running] Context: $CONTEXT | Chunk: $chunk | Ratio: $ratio"
        python "$SCRIPT_PATH" --context_size "$CONTEXT" --compression_ratio "$ratio" --chunk_size "$chunk"
    done
done

echo "----------------------------------------"
echo "All experiments completed."
