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
*   We count exactly how many times a candidate collides with *any* query token (Collision_Count).
*   Selection sorts primarily by `Collision_Count` (descending). A configurable tie-breaking hyperparameter `mode` controls secondary ordering when collision counts are equal.
*   Decoupling tie-breaking from the primary LSH ranking allows experimenting with alternative heuristics without changing the collision counting logic.

##### Tie-Breaking Strategies for Frequency Rank

When multiple candidates have the same collision count, the following tie-breaker modes determine their final order:

**1. `l2` (L2 Distance to Mean Query)**
- **Method**: Compute the mean of all query vectors, then rank candidates by Euclidean distance to this mean.
- **Pros**: Simple, interpretable, low computational cost (O(C·D)).
- **Cons**: Loses sequence structure by averaging; poor for diverse, multi-topic queries; biased toward mid-range vectors.
- **Best for**: Focused, unimodal queries with coherent semantics.

**2. `max_sim` (Maximum Pairwise Similarity)**
- **Method**: For each candidate, compute its distance to *every* query token and take the minimum distance. Rank by this min-distance.
- **Pros**: Captures "best-case relevance"; handles diverse queries; avoids centroid curse; finds local matches.
- **Cons**: Computationally expensive (O(Q·C) distance matrix); can favor outlier matches; memory-heavy with `torch.cdist`.
- **Best for**: Hybrid scenarios (especially Exact + LSH); capturing query diversity.

**3. `mahalanobis` (Variance-Weighted Distance)**
- **Method**: Compute query mean and variance, then rank candidates by weighted distance: $$d = \sqrt{\sum_d \frac{(c_d - \mu_d)^2}{\sigma_d^2}}$$
- **Pros**: Statistical grounding with second-order information; adapts to query variance; penalizes outliers more heavily; similar cost to `l2` (O(C·D)).
- **Cons**: Assumes Gaussian distribution (embeddings aren't Gaussian); uses diagonal covariance only; unstable with short queries; numerical sensitivity in low-variance dimensions.
- **Best for**: Coherent, variance-aware queries; statistical selection scenarios.

**4. `partitioned_centroid` (Multi-View Partition Centroids)**
- **Method**: Partition the query into k chunks (where k = max(1, Q/16), capped at 8), compute a centroid for each chunk, then rank candidates by minimum distance to any centroid.
- **Pros**: Captures temporal query structure; auto-scales with query length; balances cost (O(C·k) with k ≤ 8) vs. precision; avoids single-centroid collapse; handles long diverse queries.
- **Cons**: Arbitrary partitioning may not align with semantic boundaries; boundary effects at partition edges; still lossy (multiple centroids); hyperparameters (chunk size, k-cap) are fixed.
- **Best for**: Long-sequence queries; temporal structure preservation; scenarios requiring multi-view relevance.

**5. `none` (Pure Collision Count, No Tie-Breaking)**
- **Method**: Sort candidates purely by collision count with no secondary ranking. Ties are broken arbitrarily by the stable sort.
- **Pros**: Maximum speed (O(1) per candidate); pure LSH semantics with no contamination; cache-friendly; fully reproducible.
- **Cons**: Loses all tie-breaking information; LSH approximation errors exposed; all-or-nothing ranking; brittle to poor hash functions; tends toward low diversity.
- **Best for**: Speed-critical applications with very large candidate pools; when LSH collision count is sufficiently informative.



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
