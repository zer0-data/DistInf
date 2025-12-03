# DistInf
A distributed inference framework supporting Star Attention with Top-K token selection for long-context language model inference.

## Features
- **Top-K Attention Selection**: Block-wise attention accumulation across all layers for intelligent token selection
- **Star Attention**: Distributed inference with sparse KV cache
- **Custom Accuracy Models**: Single-GPU models with various summarization methods

## Scripts

### 1. Top-K Single Sample Test (Distributed)
Test the Star Attention model with Top-K token selection on a single sample.

```bash
# Single GPU
torchrun --nproc_per_node=1 run_topk_single_sample.py \
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --top_k 256 \
    --num_blocks 4 \
    --max_new_tokens 100

# Multi-GPU (e.g., 2 GPUs)
torchrun --nproc_per_node=2 run_topk_single_sample.py \
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --top_k 256 \
    --num_blocks 4
```

**Key Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | Required | Path to model |
| `--top_k` | 256 | Tokens to select per block |
| `--num_blocks` | 4 | Number of context blocks |
| `--block_size` | -1 | Block size (-1 uses num_blocks) |
| `--dataset_config` | 16k | Dataset context length (16k, 32k, 64k, 128k) |
| `--sample_index` | 0 | Which sample to test |

### 2. Star Attention Inference (Distributed)
Run full benchmark inference on JSONL files.

```bash
torchrun --nproc_per_node=2 run_star_attention_inference.py \
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --attn_type star \
    --top_k 256 \
    --num_blocks 4 \
    --tokens_to_generate 100 \
    --input_path data/input.jsonl \
    --output_path data/output.jsonl
```

### 3. Custom Accuracy Model (Single GPU)
Single-GPU model with attention-based or k-means summarization.

```bash
# Using Top-K summarization
python run_single_sample.py \
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --block_size 4096 \
    --k_summary_size 128 \
    --max_new_tokens 100 \
    --summary_method top_k

# Using K-means summarization
python run_single_sample.py \
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --block_size 4096 \
    --k_summary_size 128 \
    --max_new_tokens 100 \
    --summary_method kmeans
```

## Architecture

### Top-K Token Selection
The Top-K attention mechanism works as follows:
1. Context is split into blocks
2. During prefill, attention scores are **accumulated across ALL layers**
3. At the final layer, the top-K most attended tokens are selected
4. Only selected tokens' KV pairs are kept in the sparse cache
5. Generation uses the sparse cache for efficient inference

This approach provides better token selection compared to layer-specific methods by considering the full attention pattern across the model.

## Project Structure

```
DistInf/
├── model.py                    # Main model classes (StarAttentionModel, etc.)
├── topk_attention.py           # Top-K attention implementation
├── run_topk_single_sample.py   # Single sample test (distributed)
├── run_star_attention_inference.py  # Full benchmark inference
├── run_single_sample.py        # Single GPU custom accuracy model
├── requirements.txt
└── star_attention/             # Star Attention distributed components
    ├── modeling_llama.py
    ├── modeling_flash_attention_utils.py
    └── star_flash_attn/
```

## Installation

```bash
pip install -r requirements.txt
```

**Note**: Flash Attention requires specific CUDA setup:
```bash
pip install flash-attn --no-build-isolation
```