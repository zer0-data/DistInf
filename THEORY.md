# Theory of Operation: Bounded-Memory Recursive Transformer (BMRT)

## 1. The Core Problem
Large Language Models (LLMs) have a "Finite Context Window" limitation, primarily driven by the $O(N^2)$ complexity of attention and the linear growth of Key-Value (KV) cache memory. To handle "infinite" streams of data (e.g., books, logs, long conversations) within a fixed GPU memory budget, we cannot simply append tokens forever.

**BMRT** solves this by treating the KV Cache as a fixed-size buffer that is *recursively compressed* rather than just truncated.

---

## 2. Memory Architecture
We partition the fixed token budget $B$ into three distinct functional regions. The size of these regions is dynamic, controlled by a `protection_divisor` ($n$).

### A. Global Anchors ($B/n$)
*   **Purpose**: Stabilize the attention mechanism.
*   **Mechanism**: The very first few tokens (e.g., indices 0-1024) are *always* preserved.
*   **Why?**: "Attention Sinks" theory suggests that initial tokens collect disproportionate attention mass. Removing them destabilizes the entire model output.

### B. Local Window ($B/n$)
*   **Purpose**: Maintain immediate syntactic coherence.
*   **Mechanism**: A sliding window of the most recent tokens is always kept.
*   **Why?**: Language is locally structured. To predict the next word, the immediately preceding words are the most critical.

### C. Global Long-Term Memory ($B - 2B/n$)
*   **Purpose**: Retrieve relevant information from the distant past.
*   **Mechanism**: A "Heavy Hitter" selection process that keeps the most relevant tokens from the entire history.
*   **Why?**: This allows the model to recall specific facts (e.g., a name mentioned 50 pages ago) even if they are far outside the local window.

---

## 3. The Recursive Compression Loop
Instead of compressing the prompt once, we perform **Block-Wise Recursive Compression**.

$$ \text{State}_t = \text{Compress}(\text{State}_{t-1} \cup \text{Block}_t) $$

### The Process
1.  **Input**: We have a compressed `History` (from previous steps) and a new raw `Block` of tokens.
2.  **Pooling**: We create a "Candidate Pool" consisting of:
    *   **History Candidates**: All non-anchor tokens currently in memory.
    *   **Current Candidates**: All non-local tokens in the new block.
3.  **Selection**: We select the top $K$ tokens from this *combined* pool that maximize relevance to the current block.
4.  **Update**:
    *   **Anchors**: Kept (Static).
    *   **Local Window**: The tail of the new `Block` becomes the new protected local window.
    *   **Global Memory**: Filled with the $K$ selected/surviving tokens.
5.  **Result**: A new KV cache state that fits exactly within budget $B$.

**Key Property**: Tokens from the distant past are not "safe". If a clearer, more relevant fact appears in the new block, an old fact may be evicted. Conversely, if an old fact remains highly relevant, it can survive indefinitely ("Resurrection").

---

## 4. Selection Strategies
How do we decide which tokens are "relevant"? We implement multiple strategies.

### Strategy 1: Exact Attention (Ground Truth)
*   **Method**: We compute the *actual* attention scores between the current block (Query) and all candidates (Keys).
*   **Metric**: $\sum \text{Softmax}(Q \cdot K^T)$.
*   **Pros**: Guaranteed to keep the mathematically most important tokens.
*   **Cons**: Requires computing the full attention matrix, which is what we tried to avoid. Good for quality upper-bound benchmarks.
*   **Implementation**: `ExactSelector` using an `AttentionScoreAccumulator`.

### Strategy 2: LSH Approximation (Fast)
To avoid $O(N^2)$ compute, we use **Locality Sensitive Hashing (LSH)**.
*   **Concept**: We hash Query and Key vectors such that similar vectors have the same hash with high probability.
*   **Metric**: Collision Count. If a candidate token hashes to the same bucket as a query token, it is likely relevant.

#### Mode A: Frequency Rank (Ours)
*   We use multiple hash tables (SimHash).
*   We count exactly how many times a candidate collides with *any* query token.
*   We sort candidates by `(Collision_Count, Attention_Score)` and pick the top.
#### Mode A: Frequency Rank (Ours)
*   We use multiple hash tables (SimHash).
*   We count exactly how many times a candidate collides with *any* query token (Collision_Count).
*   Selection sorts primarily by `Collision_Count`. A configurable tie-breaking hyperparameter `mode` controls secondary ordering:
    - `attention_it`: (default) use attention scores as the secondary key (higher is better), preserving a stable attention-driven order.
    - `l2`: use the L2 (Euclidean) distance between the candidate vector and the (mean) query vector as the secondary key (lower is better).
    - `none`: no secondary key; pure collision-count ordering.
*   Decoupling tie-breaking from the primary LSH ranking allows experimenting with alternative heuristics (e.g., distance-based) without changing the collision counting logic.

#### Mode B: MagicPIG Baseline
*   **Deterministic Probability Scoring**: Instead of random sampling, we assign a theoretical selection probability $u_i$ to each token $i$ based on its Hamming Distance $D(q, k)$ from the query.
*   **Scoring Formula**: 
    $$ p = 1 - \frac{D(q, k)}{L \cdot K} $$
    $$ u_i = 1 - (1 - p^K)^L - L \cdot p^K \cdot (1 - p^K)^{L-1} $$
    Where $K$ is bits per hash and $L$ is number of tables.
*   **Filtering**: A candidate is only valid if it collides with the query in $\ge 2$ tables.
*   **Fallback**: If strict filtering yields insufficient candidates, we fall back to temporal selection (most recent tokens).

### Strategy 3: Hybrid Selection
We observe a trade-off: **Exact** selection provides high quality but low diversity (keeps redundant tokens). **LSH** provides high diversity but lower individual precision.
*   **Mechanism**: We combine two selectors (Primary and Secondary) with a configurable Ratio $R$.
    *   **Step 1**: Primary selector picks $R \cdot B$ tokens.
    *   **Step 2**: Secondary selector picks $(1-R) \cdot B$ tokens from the *remaining* pool.
*   **Significance**: This allows "filling the gaps" where the primary selector fails, boosting overall recall.

---

## 5. Technical Implementation Details
*   **Absolute Indexing**: Even though tokens are moved around in memory, we track their original "Absolute Position IDs". This ensures that RoPE (Rotary Embeddings) remain correct. The token at position 100 is always mathematically treated as "Position 100", even if it is stored at index 5 in the cache.
*   **GPU Optimization**: Our LSH implementation avoids CPU-GPU transfers. It uses matrix multiplications on the GPU to compute hash codes for thousands of tokens in parallel.
