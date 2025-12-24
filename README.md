


# DistInf: Parallel Query-Guided Top-K Inference (BM25 + TF-IDF, Flash Attention 2)

Efficient long-context language model inference using a parallel, query-guided summary selection pipeline with hybrid BM25 + TF-IDF scoring and Flash Attention 2 support.

---

## Features

- **Hybrid BM25 + TF-IDF Scoring**: Sentences in each block are scored by a mix of BM25 (query relevance) and global TF-IDF (context importance).
- **Parallel Block Processing**: All context blocks are processed in parallel for speed and efficiency.
- **Anchor & Local Windows**: Always keep the first N (anchor) and last N (local) tokens in each block for context stability.
- **Dense, Position-Aware Summary**: Selected tokens retain their original position IDs for correct RoPE/Flash Attention.
- **Flash Attention 2 Optimized**: Dense summary tensors enable efficient use of Flash Attention 2 in HuggingFace models.
- **Query-Guided**: The query directly influences which tokens are selected from each block.

---

## Pipeline Overview

```
Phase 1 - Parallel Block Scoring & Selection:
┌─────────────────────────────────────────────────────────────────────────────┐
│ For each block:                                                             │
│   - Score sentences by (Query_BM25 * 10) + Context_TFIDF                    │
│   - Always keep anchor tokens (first N) and local tokens (last N)           │
│   - Select top-K tokens from highest-scoring sentences                      │
│   - Retain original position IDs                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
         All blocks are processed in parallel; summary tokens are concatenated.
         Position IDs are preserved for RoPE/Flash Attention.

Phase 2 - Build Dense KV Cache (Summary tokens only):
┌─────────────────────────────────────────────────────────────────────────────┐
│ All summary tokens (from all blocks) → Build the KV cache using ONLY these  │
│ tokens and their original position IDs (single forward pass, Flash Attn 2)  │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
    Final KV Cache = All summary tokens' K/Vs (with original position IDs)

Phase 3 - Generation:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Query (positions start at cache_len) → Attend to Final KV Cache             │
│                                      → Autoregressive Generation            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Usage

### Python API

```python
from topk_attention import ParallelSmartSummaryProcessor

processor = ParallelSmartSummaryProcessor(
    model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    top_k=256,           # Tokens to select per block
    block_size=2048,     # Size of each context block
    max_new_tokens=100,  # Max tokens to generate
    anchor_size=64,      # Always keep first 64 tokens per block
    local_window_size=64 # Always keep last 64 tokens per block
)

result = processor(
    prompt_context="<your long context here>",
    prompt_query="What is the main topic of the document?"
)

print(result['text'][0])
```

### CLI Example

```bash
python run_topk_single_sample.py \
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --top_k 256 \
    --block_size 2048 \
    --max_new_tokens 100 \
    --anchor_size 64 \
    --local_window_size 64 \
    --sample_index 0
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `model_path` | Required | HuggingFace model path |
| `top_k` | 256 | Top-K tokens to select per block (in addition to anchor/local) |
| `block_size` | 2048 | Size of each context block |
| `max_new_tokens` | 100 | Max tokens to generate |
| `anchor_size` | 64 | Always keep first N tokens per block |
| `local_window_size` | 64 | Always keep last N tokens per block |
| `stop_words` | None | List of stop words |

---

## Architecture

### Key Classes
| Class | Description |
|-------|-------------|
| `ParallelSmartSummaryProcessor` | Implements the full pipeline: parallel block processing, hybrid BM25+TF-IDF summary selection, anchor/local token retention, dense summary construction, and Flash Attention 2 cache prefill |
| `TextScorer` | Hybrid BM25 + TF-IDF scoring for query-guided sentence/token selection |

### Key Methods
| Method | Description |
|--------|-------------|
| `_sample_bm25_from_block()` | Phase 1: Block-wise sampling with anchor/local and BM25+TF-IDF scoring |
| `_build_kv_cache()` | Phase 2: Build the KV cache using only summary tokens and their original positions |
| `_generate()` | Phase 3: Autoregressive generation using the constructed cache |

---

## Project Structure

```
DistInf/
├── topk_attention.py           # ParallelSmartSummaryProcessor (BM25+TF-IDF, Flash Attention 2)
│   ├── TextScorer                  # Hybrid BM25 + TF-IDF scoring
│   └── ParallelSmartSummaryProcessor # Full pipeline
├── model.py                    # (Legacy) CustomAccuracyModel (sequential, kmeans, etc.)
├── run_topk_single_sample.py   # CLI for ParallelSmartSummaryProcessor
├── run_single_sample.py        # (Legacy) Sequential/kmeans pipeline
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Required packages:**
- `torch`
- `transformers`
- `datasets` (for running the test script)

**Note**: For Flash Attention 2 support:
```bash
pip install flash-attn --no-build-isolation
```

---

## Legacy: Sequential Top-K (TF-IDF, Per Block)

The previous sequential, per-block TF-IDF pipeline is still available for reference in older scripts and classes. It is not recommended for new projects.



## Features
- **TF-IDF Top-K Summary Selection (Per Block)**: For each context block, select the most relevant tokens using TF-IDF scores computed over the block's tokens. Only block tokens are eligible for selection.
- **Summary Token Propagation**: At each block, only the top-K summary tokens (by TF-IDF) are retained and propagated as prefix for the next block.
- **Original Position Preservation**: Summary tokens retain their original position IDs for correct attention and cache construction.
- **Single-Pass Sparse KV Construction**: After all blocks, builds a compressed KV cache using only the selected summary tokens and their original positions.



## Pipeline Overview

### Parallel Query-Guided Top-K (BM25 + TF-IDF, Flash Attention 2)

```
Phase 1 - Parallel Block Scoring & Selection:
┌─────────────────────────────────────────────────────────────────────────────┐
│ For each block:                                                             │
│   - Score sentences by (Query_BM25 * 10) + Context_TFIDF                    │
│   - Always keep anchor tokens (first N) and local tokens (last N)           │
│   - Select top-K tokens from highest-scoring sentences                      │
│   - Retain original position IDs                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
         All blocks are processed in parallel; summary tokens are concatenated.
         Position IDs are preserved for RoPE/Flash Attention.

Phase 2 - Build Dense KV Cache (Summary tokens only):
┌─────────────────────────────────────────────────────────────────────────────┐
│ All summary tokens (from all blocks) → Build the KV cache using ONLY these  │
│ tokens and their original position IDs (single forward pass, Flash Attn 2)  │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
    Final KV Cache = All summary tokens' K/Vs (with original position IDs)

Phase 3 - Generation:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Query (positions start at cache_len) → Attend to Final KV Cache             │
│                                      → Autoregressive Generation            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Sequential Top-K (TF-IDF, Per Block)

```
Phase 1 - Sequential TF-IDF Top-K (ALL blocks):
┌─────────────────────────────────────────────────────────────────────────────┐
│ Block1 + Query → Select Top-K by TF-IDF → Summary1                          │
│ Summary1 + Block2 + Query → Select Top-K by TF-IDF → Summary2               │
│ Summary1 + Summary2 + Block3 + Query → Select Top-K by TF-IDF → Summary3    │
│ ...                                                                         │
│ At each step, only K summary tokens (from the current block, by TF-IDF)     │
│ are retained and propagated as prefix for the next block.                   │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
         Only summary tokens (top-K per block) are kept; all other tokens are masked out.
         Summary tokens retain their ORIGINAL position IDs.

Phase 2 - Build KV Cache (Summary tokens only):
┌─────────────────────────────────────────────────────────────────────────────┐
│ All summary tokens (from all blocks) → Build the KV cache using ONLY these  │
│ tokens and their original position IDs. All other tokens are excluded.      │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
    Final KV Cache = All summary tokens' K/Vs (with original position IDs)

Phase 3 - Generation:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Query (positions start at cache_len) → Attend to Final KV Cache             │
│                                      → Autoregressive Generation            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

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
            Efficient long-context language model inference using **sequential, per-block Top-K summary selection** with multiple methods (TF-IDF, RAKE, YAKE, Length-Normalized TF-IDF, TextRank).
            --max_new_tokens $MAX_NEW_TOKENS \
            --dataset_config $DATASET_CONFIG \

            ## Features
            - **Flexible Top-K Summary Selection (Per Block)**: For each context block, select the most relevant tokens using one of several methods:
                - **TF-IDF**: Standard TF-IDF score per token
                - **RAKE**: Rapid Automatic Keyword Extraction
                - **YAKE**: Yet Another Keyword Extractor
                - **Length-Normalized TF-IDF**: TF-IDF score divided by token length
                - **TextRank**: Graph-based ranking using token similarity
            - **Summary Token Propagation**: At each block, only the top-K summary tokens (by the chosen method) are retained and propagated as prefix for the next block.
            - **Original Position Preservation**: Summary tokens retain their original position IDs for correct attention and cache construction.
            - **Single-Pass Sparse KV Construction**: After all blocks, builds a compressed KV cache using only the selected summary tokens and their original positions.
Save this as `run_all_topk.sh`, make it executable (`chmod +x run_all_topk.sh`), and run it in your project directory.

---



## Usage

### Sequential Top-K (run_topk_single_sample.py)

Test on a single sample from the BABILong dataset:

```bash
python run_topk_single_sample.py \
            Phase 1 - Sequential Top-K (ALL blocks):
            ┌─────────────────────────────────────────────────────────────────────────────┐
            │ Block1 + Query → Select Top-K by <summary_method> → Summary1                │
            │ Summary1 + Block2 + Query → Select Top-K by <summary_method> → Summary2     │
            │ Summary1 + Summary2 + Block3 + Query → Select Top-K by <summary_method> → Summary3 │
            │ ...                                                                         │
            │ At each step, only K summary tokens (from the current block, by the chosen method)  │
            │ are retained and propagated as prefix for the next block.                   │
            └─────────────────────────────────────────────────────────────────────────────┘
                                          ↓
                     Only summary tokens (top-K per block) are kept; all other tokens are masked out.
                     Summary tokens retain their ORIGINAL position IDs.
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --top_k 256 \
    --block_size 2048 \
    --max_new_tokens 100 \
    --dataset_config 16k \
    --sample_index 0
```

**Command Line Arguments:**

| Argument | Default | Description |
            **After Phase 1:**
            - Only the summary tokens (top-K per block) are kept; all other tokens are masked out and excluded from further processing.
            - Summary tokens retain their original position IDs from their source blocks.
|----------|---------|-------------|
| `--model_path` | Required | HuggingFace model path |
| `--top_k` | 256 | Tokens to select per block |
| `--block_size` | 2048 | Size of each context block |
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




### Phase 1: Sequential TF-IDF Top-K Sampling
Process all blocks sequentially, propagating summary tokens:

1. **Block 1**: Input = `[Block_1] + [Query]`
2. **Block i (i > 1)**: Input = `[Summary_1 + ... + Summary_{i-1}] + [Block_i] + [Query]`
3. For each block, decode block tokens to text and compute TF-IDF scores for each token (treating each token as a document).
4. Select top-K tokens from the current block based on TF-IDF scores.
5. Return both summary token IDs and their original position indices.
6. Store summary token IDs and their original positions for the next block and for Phase 2.

**After Phase 1:**
- Only the summary tokens (top-K per block) are kept; all other tokens are masked out and excluded from further processing.
- Summary tokens retain their original position IDs from their source blocks.

### Phase 2: KV Cache Construction (Summary tokens only)
- Concatenate all summary tokens and their original position IDs from all blocks.
- Build the KV cache using ONLY these summary tokens and their original position IDs (single forward pass).
- No other tokens are included in the cache.

**Benefits:**
- Only summary tokens are included in the final cache, preserving memory and efficiency.
- Position ID alignment is preserved: each summary token appears at its original position in the attention matrix.

Final KV Cache = KV cache built from all summary tokens (single forward pass)

### Phase 3: Generation
- Query position IDs start at `cache_len` (length of final KV cache)
- Uses `DynamicCache` for efficient autoregressive generation
- Autoregressive token generation with early stopping on EOS



### Simplicity and Efficiency

This approach is memory-efficient and simple:
- No attention matrices are stored or accumulated.
- Only TF-IDF scores are computed for each block, and top-K tokens are selected directly.


### Key Classes

| Class | Description |
|-------|-------------|
| `SequentialTopKProcessor` | Implements the full pipeline: sequential block processing, TF-IDF-based top-K summary selection, summary propagation, and final generation |


### Key Methods

| Method | Description |
|--------|-------------|
| `_sample_topk_from_block_with_context()` | Phase 1: Run block-wise sampling with context propagation and summary token selection using TF-IDF |
| `_build_kv_cache_sequential()` | Phase 2: Build the KV cache using only summary tokens and their original positions |
| `_generate()` | Phase 3: Autoregressive generation using the constructed cache |

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
