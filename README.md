# BMRT: Bounded-Memory Recursive Transformer

Enable infinite-context reasoning within a fixed memory budget by iteratively compressing the KV cache using modular selection strategies.

## Features
- **Recursive Compression**: `State_N = Compress(State_{N-1} + Chunk_N)`.
- **Dynamic Budgeting**: Total budget is dynamically split using a `protection_divisor`:
    - **Anchors**: Stabilize attention.
    - **Local Window**: Preserve recent context.
    - **Global Budget**: Selectable token capacity.
- **Dual Strategies**:
    1.  **Exact Attention**: Uses chunked score accumulation (Memory Optimized Eager Mode).
    2.  **LSH Sampling**: Locality Sensitive Hashing with two modes (`frequency_rank` vs `magicpig_baseline`) and **Flash Attention** compatibility.
    3.  **Hybrid Strategy**: Combines two selectors (e.g., Exact + LSH) with configurable diversity ratios.

## Architecture

### Hybrid Cache Composition
Total Budget $B$ is split based on `protection_divisor` ($n$):
1.  **Anchors** ($B/n$): Fixed initial tokens.
2.  **Local Window** ($B/n$): Most recent tokens.
3.  **Global Memory** ($B - 2B/n$): Selected via the configured Strategy.

### Components
*   `bmrt.processor`: Main `RecursiveCompressionEngine` handling the loop and state management.
*   `bmrt.selectors.eager_exact`: Exact attention score computation (Option 1).
*   `bmrt.selectors.lsh_core`: LSH-based selection (Option 2) with "Frequency Rank" (Ours) or "MagicPIG Baseline".
*   `bmrt.accumulator`: Helper for aggregating attention scores in Eager mode.

## Installation

```bash
pip install -e .
```

**Dependencies**:
- `torch` >= 2.2.0
- `transformers`
- `datasets`
- `accelerate`
- `flash-attn` (Optional, for Flash backend)

## Usage

### 1. Exact Strategy (Eager Mode)
The default "Gold Standard" method. Uses exact attention scores to pick the most relevant global tokens.

```bash
python run_single_sample.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --method exact \
    --backend eager \
    --budget 4096 \
    --protection_divisor 4
```

### 2. LSH Strategy (Flash Attention)
Optimized for speed. Uses Hash Collision frequency to approximate relevance.

```bash
python run_single_sample.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --method lsh \
    --lsh_mode frequency_rank \
    --backend flash \
    --budget 4096 \
    --protection_divisor 4
```

### 3. LSH Baseline (MagicPIG Style)
Uses standard probabilistic sampling from LSH buckets.

```bash
python run_single_sample.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --method lsh \
    --lsh_mode magicpig_baseline \
    --backend eager \
    --budget 4096
```

### 4. Hybrid Strategy
Combines Exact attention (for quality) with LSH (for diversity) using a configurable ratio.

```bash
python run_single_sample.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --method hybrid \
    --hybrid_primary exact \
    --hybrid_secondary lsh \
    --hybrid_ratio 0.6 \
    --compression_mode recursive \
    --budget 4096
```

### Ablation Strategy
To exhaustively test effectiveness, we recommend the following ablation grid (~14 core runs per dataset):

| Strategy | Mode | Backend | Est. Runs | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Exact (Accumulate) | Eager | 1 | "Gold Standard" |
| **Recursive** | Exact (Recursive) | Eager | 1 | Full history scoring |
| **LSH** | Freq / MagicPIG | Flash / Eager | 4 | 2 Modes x (Flash vs Eager*) |
| **Hybrid** | Exact + LSH | Eager | 8 | 2 Ratios (0.5, 0.75) x 2 LSH Modes x 2 Compr. Modes |

*Note: Flash backend is recommended for LSH speed, but Eager is required if mixing with Exact (Hybrid).*

### Python API

```python
from bmrt import RecursiveCompressionEngine

engine = RecursiveCompressionEngine(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    selector_type='exact',   # or 'lsh'
    backend='eager',         # or 'flash'
    budget=4096,             # Total tokens to keep
    protection_divisor=4     # 1/4th Anchors, 1/4th Window
)

result = engine(
    prompt_context="<your long context here>",
    prompt_query="What is the main topic?"
)

print(result['text'][0])
```
