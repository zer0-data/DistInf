# DistInf
Sequential Top-K attention with query-guided token selection for efficient long-context language model inference.

## Features
- **Query-Guided Top-K Selection**: Select most relevant tokens from each block based on attention to query
- **Memory-Efficient Accumulation**: Accumulates attention scores layer-by-layer (not all at once)
- **Sequential KV Cache Construction**: Builds compressed KV cache with context propagation across blocks

## Pipeline Overview

```
Phase 1 - Sampling (Independent per block):
┌─────────────────────────────────────────────────────────────┐
│ Block1 + Query → Accumulate attn scores → Top-K → Summary1  │
│ Block2 + Query → Accumulate attn scores → Top-K → Summary2  │
│ Block3 + Query → Accumulate attn scores → Top-K → Summary3  │
│ Block4 + Query → Accumulate attn scores → Top-K → Summary4  │
└─────────────────────────────────────────────────────────────┘

Phase 2 - Build KV Cache (Sequential):
┌─────────────────────────────────────────────────────────────┐
│ Block1                              → KV1                   │
│ Summary1 + Block2                   → KV2                   │
│ Summary1 + Summary2 + Block3        → KV3                   │
│ Summary1 + Summary2 + Summary3 + Block4 → KV4               │
└─────────────────────────────────────────────────────────────┘
                            ↓
         Final KV Cache = KV1 + KV2 + KV3 + KV4

Phase 3 - Generation:
┌─────────────────────────────────────────────────────────────┐
│ Query → Attend to Final KV Cache → Generate Response        │
└─────────────────────────────────────────────────────────────┘
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

### Phase 1: Query-Guided Sampling
For each block independently:
1. Concatenate `[Block_i] + [Query]`
2. Run forward pass with `output_attentions=True`
3. Accumulate attention scores across **ALL layers**:
   - Extract attention FROM query tokens TO block tokens
   - Sum across layers, heads, and query positions
4. Select top-K tokens based on accumulated scores
5. Output: `Summary_i` = top-K token IDs from Block_i

### Phase 2: Sequential KV Cache Construction
Process blocks sequentially with accumulated summaries:
- Step 1: `Block1` → KV cache contains Block1
- Step 2: `Summary1 + Block2` → KV cache contains Summary1 + Block2
- Step 3: `Summary1 + Summary2 + Block3` → KV cache contains all
- ...

Final KV Cache = Concatenation of all step outputs

### Phase 3: Generation
- Project query onto the final sparse KV cache
- Autoregressive token generation

### Memory Efficiency

The `AttentionScoreAccumulator` is memory-efficient because it:
- Stores only a **running sum** of attention scores `(bsz, seq_len)`
- Does NOT store full attention matrices from all layers
- Processes attention layer-by-layer during accumulation

## Project Structure

```
DistInf/
├── topk_attention.py           # Sequential Top-K implementation
│   ├── AttentionScoreAccumulator   # Memory-efficient score accumulation
│   └── SequentialTopKProcessor     # Full pipeline
├── model.py                    # CustomAccuracyModel (supports top_k & kmeans)
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
