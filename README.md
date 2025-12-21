# DistInf
Sequential Top-K attention with query-guided token selection for efficient long-context language model inference.

## Features
- **Query-Guided Top-K Selection**: Select most relevant tokens from each block based on attention from query tokens to block tokens
- **Memory-Efficient Accumulation**: Accumulates attention scores layer-by-layer using `AttentionScoreAccumulator` (stores only running sum, not full attention matrices)
- **Original Position Preservation**: Summary tokens retain their original position IDs for correct attention matrix construction
- **Single-pass Sparse KV Construction**: Builds a compressed KV cache by extracting KVs for all summary tokens in a single forward pass

## Pipeline Overview

```
Phase 1 - Sequential Sampling (with context propagation, ALL blocks):
┌──────────────────────────────────────────────────────────────────────────┐
│ Block1 + Query → Accumulate attn (query→block) → Top-K → Summary1        │
│ Summary1 + Block2 + Query → Accumulate attn → Top-K → Summary2           │
│ Summary1 + Summary2 + Block3 + Query → Accumulate → Top-K → Summary3     │
│ ...                                                                      │

## Batch Experiments: Run Across Multiple Context Lengths and Top-K

To run `run_topk_single_sample.py` for all combinations of context lengths (16k, 32k, 64k, 128k) and top_k values (64, 128, 256), use the provided shell script:

```bash
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
```

Save this as `run_all_topk.sh`, make it executable (`chmod +x run_all_topk.sh`), and run it in your project directory.

---

└──────────────────────────────────────────────────────────────────────────┘
                              ↓
         Only summary tokens (top-K per block) are kept; all other tokens are masked out.
         Summary tokens retain their ORIGINAL position IDs.

Phase 2 - Build KV Cache (Summary tokens only):
┌──────────────────────────────────────────────────────────────────────────┐
│ Concatenate all summary tokens from all blocks.                          │
│ Build the KV cache using ONLY these summary tokens and their original    │
│ position IDs. All other tokens are excluded from the cache.              │
└──────────────────────────────────────────────────────────────────────────┘
                              ↓
    Final KV Cache = All summary tokens' K/Vs (with original position IDs)

Phase 3 - Generation:
┌──────────────────────────────────────────────────────────────────────────┐
│ Query (positions start at cache_len) → Attend to Final KV Cache          │
│                                      → Autoregressive Generation         │
└──────────────────────────────────────────────────────────────────────────┘
```

## Usage

### Sequential Top-K (run_topk_single_sample.py)

Test on a single sample from the BABILong dataset:

```bash
python run_topk_single_sample.py \
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --top_k 256 \
    --block_size 2048 \
    --max_new_tokens 100 \
    --dataset_config 16k \
    --sample_index 0
```

**Command Line Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | Required | HuggingFace model path |
| `--top_k` | 256 | Tokens to select per block |
| `--block_size` | 2048 | Size of each context block |
| `--max_new_tokens` | 100 | Max tokens to generate |
| `--stop_words` | "" | Comma-separated stop words |
| `--dataset_config` | "16k" | Dataset config (16k, 32k, 64k, 128k) |
| `--dataset_split` | "qa1" | Dataset split |
| `--sample_index` | 0 | Sample index to test |

### K-Means Clustering (run_single_sample.py)

Use K-Means clustering on token hidden states for selection:

```bash
python run_single_sample.py \
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --summary_method kmeans \
    --k_summary_size 128 \
    --block_size 4096 \
    --max_new_tokens 100
```

Or use attention-based top-k selection:

```bash
python run_single_sample.py \
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --summary_method top_k \
    --k_summary_size 128 \
    --block_size 4096 \
    --max_new_tokens 100
```

**Command Line Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | Required | HuggingFace model path |
| `--summary_method` | "top_k" | Selection method: `top_k` or `kmeans` |
| `--k_summary_size` | 128 | Number of tokens to select per block |
| `--block_size` | 4096 | Size of each context block |
| `--max_new_tokens` | 100 | Max tokens to generate |

**Note:** K-Means requires `scikit-learn`:
```bash
pip install scikit-learn
```

### Python API

```python
from topk_attention import SequentialTopKProcessor

processor = SequentialTopKProcessor(
    model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    top_k=256,           # Tokens to select per block
    block_size=2048,     # Size of each context block
    max_new_tokens=100,  # Max tokens to generate
)

result = processor(
    prompt_context="<your long context here>",
    prompt_query="What is the main topic of the document?"
)

print(result['text'][0])
```

### API Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | Required | HuggingFace model path |
| `top_k` | 256 | Number of tokens to select from each block |
| `block_size` | 2048 | Size of each context block |
| `max_new_tokens` | 100 | Maximum tokens to generate |
| `stop_words` | None | List of stop words for generation |

## Architecture


### Phase 1: Sequential Query-Guided Sampling
Process ALL blocks sequentially with context propagation:

1. **For Block 1**: Input = `[Block_1] + [Query]`
2. **For Block i (i > 1)**: Input = `[Summary_1 + ... + Summary_{i-1}] + [Block_i] + [Query]`
3. Run forward pass with `output_attentions=True`
4. **Accumulate attention scores** across ALL layers using `AttentionScoreAccumulator`:
    - Extract attention FROM query tokens TO block tokens only
    - Uses `start_block_with_prefix()` to track prefix/block/query boundaries
    - Sum across layers, heads, and query positions
    - Memory-efficient: only stores running sum `(bsz, block_len)`
5. Select top-K tokens based on accumulated scores (only from current block)
6. **Return both Summary_i token IDs AND their original position indices**
7. Store summary token IDs and their original positions for Phase 2

**After Phase 1:**
- Only the summary tokens (top-K per block) are kept; all other tokens are masked out and excluded from further processing.
- Summary tokens retain their original position IDs from their source blocks.

### Phase 2: KV Cache Construction (Summary tokens only)
- Concatenate all summary tokens and their original position IDs.
- Build the KV cache using ONLY these summary tokens and their original position IDs.
- No other tokens are included in the cache.

**Benefits:**
- Only summary tokens are included in the final cache, preserving memory and efficiency.
- Position ID alignment is preserved: each summary token appears at its original position in the attention matrix.

Final KV Cache = KV cache built from all summary tokens (single forward pass)

### Phase 3: Generation
- Query position IDs start at `cache_len` (length of final KV cache)
- Uses `DynamicCache` for efficient autoregressive generation
- Autoregressive token generation with early stopping on EOS

### Memory Efficiency

The `AttentionScoreAccumulator` is memory-efficient because it:
- Stores only a **running sum** of attention scores `(bsz, block_len)`
- Does NOT store full attention matrices from all layers
- Processes attention layer-by-layer during accumulation with in-place addition
- Explicitly frees each layer's attention weights after accumulation

### Key Classes

| Class | Description |
|-------|-------------|
| `AttentionScoreAccumulator` | Memory-efficient attention score accumulation across layers |
| `SequentialTopKProcessor` | Full pipeline: Phase 1 sampling → Phase 2 KV cache → Phase 3 generation |

### Key Methods

| Method | Description |
|--------|-------------|
| `start_block_with_prefix()` | Initialize accumulator with prefix/block/query boundaries |
| `accumulate()` | Add attention from one layer to running sum |
| `select_top_k()` | Select top-K tokens based on accumulated scores |
| `_sample_topk_from_block_with_context()` | Phase 1: Sample tokens with context propagation |
| `_build_kv_cache_sparse()` | Phase 2: Build sparse KV cache (single forward pass) |
| `_generate()` | Phase 3: Autoregressive generation |

## Project Structure

```
DistInf/
├── topk_attention.py           # Sequential Top-K implementation
│   ├── AttentionScoreAccumulator   # Memory-efficient score accumulation with prefix support
│   ├── SequentialTopKProcessor     # Full pipeline with position-aware KV cache
│   └── get_or_create_accumulator() # Utility to attach accumulator to model
├── model.py                    # CustomAccuracyModel (supports top_k & kmeans with position tracking)
├── run_topk_single_sample.py   # Test script for SequentialTopKProcessor
├── run_single_sample.py        # Test script for CustomAccuracyModel (kmeans/top_k)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

**Required packages:**
- `torch`
- `transformers`
- `datasets` (for running the test script)

**Note**: For eager attention (required for `output_attentions=True`), no special setup needed.
For flash attention in other components:
```bash
pip install flash-attn --no-build-isolation
```
