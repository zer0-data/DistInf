#!/bin/bash

MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
BLOCK_SIZE=2048
MAX_NEW_TOKENS=100

for DATASET_CONFIG in 16k 32k 64k 128k; do
  for TOP_K in 64 128 256; do
    echo "Running: dataset_config=$DATASET_CONFIG, top_k=$TOP_K"
    python run_topk_single_sample.py \
      --model_path "$MODEL_PATH" \
      --block_size $BLOCK_SIZE \
      --max_new_tokens $MAX_NEW_TOKENS \
      --dataset_config $DATASET_CONFIG \
      --top_k $TOP_K
  done
done
