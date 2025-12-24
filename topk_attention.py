# topk_attention.py
# Sequential Top-K attention with query-guided token selection
# Memory-efficient: accumulates attention scores layer-by-layer
# Cache construction: Global Anchor Tokens + Local Window + Global Top-K

from typing import Optional, Tuple, List, Dict

import torch
from transformers.cache_utils import DynamicCache


# =============================================================================
# ATTENTION SCORE ACCUMULATOR
# =============================================================================

class AttentionScoreAccumulator:
    """
    Accumulates attention scores across all layers during block-wise prefill.
    Performs top-K selection based on the sum of attention scores from all layers.
    
    Supports query-guided selection: when query tokens are appended to a block,
    only the block tokens (not query tokens) are considered for top-K selection.
    
    Memory efficient: only stores running sum, not full attention matrices.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the accumulator for a new block."""
        self.accumulated_scores: Optional[torch.Tensor] = None
        self.layer_count: int = 0
        self.is_active: bool = False
        # Number of tokens in the block that are eligible for selection (excludes query)
        self.block_token_count: int = 0
        # Total sequence length including query
        self.total_seq_len: int = 0
        # Prefix length (summaries from previous blocks)
        self.prefix_len: int = 0
        # Query length
        self.query_len: int = 0
    
    def start_block(self, total_seq_len: int, block_token_count: int = -1):
        """
        Start accumulating scores for a new block (legacy method for simple cases).
        
        Sequence layout: [block] + [query]
        Only tokens in [block] are eligible for top-K selection.
        
        Args:
            total_seq_len: Total sequence length (block + query)
            block_token_count: Number of tokens in the block that are eligible for selection.
                              If -1, all tokens are eligible (no query appended).
        """
        self.reset()
        self.is_active = True
        self.total_seq_len = total_seq_len
        self.prefix_len = 0  # No prefix in legacy mode
        
        if block_token_count > 0:
            self.block_token_count = block_token_count
            # Query length is the remaining tokens after the block
            self.query_len = total_seq_len - block_token_count
        else:
            # No query - all tokens are block tokens
            self.block_token_count = total_seq_len
            self.query_len = 0
    
    def start_block_with_prefix(
        self, 
        total_seq_len: int, 
        prefix_len: int,
        block_len: int,
        query_len: int,
    ):
        """
        Start accumulating scores for a block with prefix context.
        
        Sequence layout: [prefix_summaries] + [block] + [query]
        Only tokens in [block] are eligible for top-K selection.
        
        Args:
            total_seq_len: Total sequence length (prefix + block + query)
            prefix_len: Length of prefix summaries (not eligible for selection)
            block_len: Length of the block (eligible for selection)
            query_len: Length of query tokens (not eligible for selection)
        
        Raises:
            ValueError: If parameters are invalid or inconsistent
        """
        # Validate inputs
        if prefix_len < 0:
            raise ValueError(f"prefix_len cannot be negative, got {prefix_len}")
        if block_len <= 0:
            raise ValueError(f"block_len must be positive, got {block_len}")
        if query_len < 0:
            raise ValueError(f"query_len cannot be negative, got {query_len}")
        
        expected_total = prefix_len + block_len + query_len
        if total_seq_len != expected_total:
            raise ValueError(
                f"Length mismatch: total_seq_len ({total_seq_len}) != "
                f"prefix_len ({prefix_len}) + block_len ({block_len}) + query_len ({query_len}) = {expected_total}"
            )
        
        self.reset()
        self.is_active = True
        self.total_seq_len = total_seq_len
        self.prefix_len = prefix_len
        self.block_token_count = block_len  # Only block tokens are eligible
        self.query_len = query_len
    
    def accumulate(self, attn_weights: torch.Tensor, layer_idx: int):
        """
        Add attention scores from a layer to the accumulator.
        
        Supports two layouts:
        1. Legacy: [block] + [query] - query attends to block
        2. With prefix: [prefix] + [block] + [query] - query attends to block only
        
        In both cases, we use attention FROM query tokens TO block tokens for scoring.
        
        Args:
            attn_weights: Attention weights of shape (bsz, num_heads, q_len, kv_seq_len)
                          This is the standard shape for HuggingFace transformer attention outputs.
                          - bsz: batch size (must be 1)
                          - num_heads: number of attention heads
                          - q_len: query sequence length (== kv_seq_len for self-attention)
                          - kv_seq_len: key/value sequence length
            layer_idx: The layer index (used for error reporting)
        
        Raises:
            ValueError: If attention tensor has unexpected shape or dimensions
        """
        if not self.is_active:
            return
        
        # Validate attention tensor shape
        if attn_weights.dim() != 4:
            raise ValueError(
                f"Expected attention weights to have 4 dimensions (bsz, num_heads, q_len, kv_seq_len), "
                f"got {attn_weights.dim()} dimensions with shape {attn_weights.shape} at layer {layer_idx}"
            )
        
        bsz, num_heads, q_len, kv_seq_len = attn_weights.shape
        
        if bsz != 1:
            raise ValueError(
                f"Expected batch size of 1, got {bsz} at layer {layer_idx}. "
                f"This implementation only supports single-sequence processing."
            )
        
        # Validate sequence length matches expected total
        if q_len != self.total_seq_len:
            raise ValueError(
                f"Attention q_len ({q_len}) doesn't match expected total_seq_len ({self.total_seq_len}) "
                f"at layer {layer_idx}. This may indicate a configuration error."
            )
        
        # Determine the slice of KV positions corresponding to the block
        block_start = self.prefix_len
        block_end = self.prefix_len + self.block_token_count
        
        # Use stored query_len to determine if query tokens exist
        # Query tokens are at the END of the sequence: positions [seq_len - query_len, seq_len)
        if self.query_len > 0:
            # Query tokens exist - use query attention for scoring block tokens
            # Query tokens start at: total_seq_len - query_len
            query_start_in_seq = q_len - self.query_len
            # Attention FROM query tokens TO block tokens
            # Sum over num_heads (dim=1) and query positions (dim=2) to get score per block token
            layer_scores = attn_weights[:, :, query_start_in_seq:, block_start:block_end].sum(dim=(1, 2))
        else:
            # No query tokens - use all attention to block tokens
            # Sum over num_heads (dim=1) and all query positions (dim=2)
            layer_scores = attn_weights[:, :, :, block_start:block_end].sum(dim=(1, 2))
        
        # layer_scores shape: (bsz, block_token_count) = (1, block_token_count)
        
        if self.accumulated_scores is None:
            self.accumulated_scores = layer_scores  # No clone needed, we own this tensor
        else:
            # Validate shape consistency - all layers should produce same-shaped scores
            if self.accumulated_scores.shape[-1] != layer_scores.shape[-1]:
                raise RuntimeError(
                    f"Shape mismatch in attention score accumulation at layer {layer_idx}: "
                    f"accumulated shape {self.accumulated_scores.shape[-1]} vs "
                    f"layer scores shape {layer_scores.shape[-1]}. "
                    f"This indicates a bug - all layers should produce consistent shapes."
                )
            # In-place addition to avoid allocating new tensor
            self.accumulated_scores.add_(layer_scores)
            del layer_scores  # Explicitly free
        
        self.layer_count += 1
    
    def select_top_k(self, top_k: int) -> List[int]:
        """
        Select top-K tokens based on accumulated attention scores.
        
        Args:
            top_k: Number of tokens to select
            
        Returns:
            List of selected indices (sorted to maintain sequence order)
        
        Raises:
            ValueError: If top_k is not positive
            RuntimeError: If no layers have been accumulated
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        
        if self.accumulated_scores is None:
            return []
        
        if self.layer_count == 0:
            raise RuntimeError(
                "select_top_k called but no layers have been accumulated. "
                "Make sure accumulate() was called for each layer."
            )
        
        num_block_tokens = self.accumulated_scores.shape[-1]
        token_budget = min(num_block_tokens, top_k)
        
        # Select top-K indices (tensor will be on same device as accumulated_scores)
        _, top_k_indices = torch.topk(self.accumulated_scores[0], k=token_budget)
        
        # Sort to maintain sequence order
        top_k_indices_sorted, _ = torch.sort(top_k_indices)
        
        # Transfer to CPU and convert to Python list
        # This is necessary because the indices will be used for Python indexing
        return top_k_indices_sorted.cpu().tolist()
    
    def select_top_k_with_exclusion(self, top_k: int, exclude_indices: List[int]) -> List[int]:
        """
        Select top-K tokens based on accumulated attention scores, excluding specified indices.
        
        Args:
            top_k: Number of tokens to select
            exclude_indices: Indices to exclude from selection (e.g., anchor and local window tokens)
            
        Returns:
            List of selected indices (sorted to maintain sequence order)
        
        Raises:
            ValueError: If top_k is not positive
            RuntimeError: If no layers have been accumulated
        """
        if top_k <= 0:
            return []
        
        if self.accumulated_scores is None:
            return []
        
        if self.layer_count == 0:
            raise RuntimeError(
                "select_top_k_with_exclusion called but no layers have been accumulated. "
                "Make sure accumulate() was called for each layer."
            )
        
        num_block_tokens = self.accumulated_scores.shape[-1]
        
        # Create a mask for excluded indices
        scores = self.accumulated_scores[0].clone()
        if exclude_indices:
            exclude_tensor = torch.tensor(exclude_indices, device=scores.device, dtype=torch.long)
            # Set excluded indices to -inf so they won't be selected
            scores[exclude_tensor] = float('-inf')
        
        # Count available tokens (not excluded)
        available_tokens = num_block_tokens - len(exclude_indices)
        token_budget = min(available_tokens, top_k)
        
        if token_budget <= 0:
            return []
        
        # Select top-K indices from non-excluded tokens
        _, top_k_indices = torch.topk(scores, k=token_budget)
        
        # Sort to maintain sequence order
        top_k_indices_sorted, _ = torch.sort(top_k_indices)
        
        # Transfer to CPU and convert to Python list
        return top_k_indices_sorted.cpu().tolist()
    
    def finish_block(self) -> None:
        """
        Clean up after finishing a block.
        
        Resets all state to prepare for the next block. Does NOT call
        torch.cuda.empty_cache() - callers should manage cache clearing
        at appropriate points (e.g., after processing multiple blocks).
        """
        # Clear accumulated scores tensor
        self.accumulated_scores = None
        
        # Reset all state variables
        self.layer_count = 0
        self.block_token_count = 0
        self.total_seq_len = 0
        self.prefix_len = 0
        self.query_len = 0
        self.is_active = False


def get_or_create_accumulator(model) -> AttentionScoreAccumulator:
    """Get or create an attention score accumulator for a model."""
    if not hasattr(model, '_topk_accumulator'):
        model._topk_accumulator = AttentionScoreAccumulator()
    return model._topk_accumulator


# =============================================================================
# SEQUENTIAL TOP-K PROCESSOR
# =============================================================================

class SequentialTopKProcessor:
    """
    Implements the sequential block processing pipeline with query-guided top-K selection.
    
    Cache Construction Strategy:
        Summary = Global Anchor Tokens + Local Window + Global Top-K
        
        - Global Anchor Tokens: First N tokens of the ENTIRE context (only from first block)
        - Local Window: Last M tokens of each block (recent context)
        - Global Top-K: Top-K tokens by query attention (excluding anchor and local)
    
    Pipeline:
    Phase 1 - Sequential Sampling (with context propagation):
        Block1 + Query → Select (Global Anchor + Local + Top-K) from Block1 → Summary1
        Summary1 + Block2 + Query → Select (Local + Top-K) from Block2 → Summary2
        ...
    
    Phase 2 - Build KV Cache (Summary tokens only):
        Concatenate all summaries with their original positions
    
    Phase 3 - Generation:
        Query → Attend to Final Cache → Generate
    """
    
    def __init__(
        self,
        model_path: str,
        top_k: int = 256,
        block_size: int = 2048,
        max_new_tokens: int = 100,
        stop_words: Optional[List[str]] = None,
        # New parameters for Anchor + Local + Top-K strategy
        anchor_size: int = 64,
        local_window_size: int = 64,
    ):
        """
        Args:
            model_path: Path to the HuggingFace model
            top_k: Number of tokens to select via attention (excluding anchor and local)
            block_size: Size of each context block
            max_new_tokens: Maximum tokens to generate
            stop_words: List of stop words for generation
            anchor_size: Number of anchor tokens from the start of the entire context
            local_window_size: Number of tokens from the end of each block (local window)
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.model_path = model_path
        self.top_k = top_k
        self.block_size = block_size
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words or []
        self.anchor_size = anchor_size
        self.local_window_size = local_window_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"\n[SequentialTopKProcessor] Initializing...")
        print(f"  Model: {model_path}")
        print(f"  Block Size: {block_size}")
        print(f"  Cache Strategy: Global Anchor({anchor_size}) + Local({local_window_size}) + Top-K({top_k})")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with eager attention (needed for output_attentions=True)
        print("  Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation='eager',
        )
        self.model.eval()
        
        print("[SequentialTopKProcessor] Initialization complete.\n")
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text and return tensor on device."""
        return self.tokenizer.encode(
            text, return_tensors='pt', add_special_tokens=False
        ).to(self.device)
    
    def _tokenize_query_with_chat_template(self, query_text: str) -> torch.Tensor:
        """Tokenize query using chat template for consistency with generation."""
        messages = [{"role": "user", "content": query_text}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
    
    def _split_into_blocks(self, token_ids: torch.Tensor) -> List[torch.Tensor]:
        """Split token IDs into blocks of block_size."""
        return list(token_ids.split(self.block_size, dim=1))
    
    def _compute_anchor_local_indices(
        self, 
        block_len: int, 
        block_start_position: int,
    ) -> Tuple[List[int], List[int]]:
        """
        Compute anchor and local window indices for a block.
        
        Anchor tokens are the first N tokens of the ENTIRE context, not per-block.
        So only the first block (block_start_position == 0) will have anchor tokens,
        and only if anchor_size > 0.
        
        Args:
            block_len: Length of the block
            block_start_position: Starting position of this block in the full context
            
        Returns:
            Tuple of (anchor_indices, local_indices) - indices relative to the block
        """
        anchor_indices = []
        
        # Anchor tokens: first anchor_size tokens of the ENTIRE context
        # These only exist in the first block (or blocks if anchor_size > block_size)
        if block_start_position < self.anchor_size:
            # This block contains some anchor tokens
            # anchor tokens in this block: positions [0, min(anchor_size - block_start_position, block_len))
            anchor_end_in_block = min(self.anchor_size - block_start_position, block_len)
            anchor_indices = list(range(anchor_end_in_block))
        
        # Local window: last local_window_size tokens of this block
        # Make sure we don't overlap with anchor tokens in this block
        anchor_end = len(anchor_indices)
        local_start = max(anchor_end, block_len - self.local_window_size)
        local_indices = list(range(local_start, block_len))
        
        return anchor_indices, local_indices
    
    @torch.no_grad()
    def _sample_topk_from_block_with_context(
        self,
        prefix_summaries: List[torch.Tensor],
        block_ids: torch.Tensor,
        query_ids: torch.Tensor,
        block_start_position: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Phase 1: Sample tokens from a block using Anchor + Local + Top-K strategy.
        
        Cache = Global Anchor Tokens + Local Window + Global Top-K
        
        Global Anchor: Only from the first block(s) - first N tokens of entire context
        Local Window: Last M tokens of this block
        Top-K: Selected via accumulated attention scores
        
        Args:
            prefix_summaries: List of summary token IDs from previous blocks
            block_ids: Token IDs of the current block
            query_ids: Query token IDs
            block_start_position: The starting position of this block in the full context
        
        Returns:
            Tuple of (summary_ids, summary_positions):
            - summary_ids: Selected token IDs shape (1, num_selected)
            - summary_positions: Original position IDs shape (1, num_selected)
        """
        # Validate inputs
        if block_ids.dim() != 2 or block_ids.shape[0] != 1:
            raise ValueError(f"block_ids must have shape (1, seq_len), got {block_ids.shape}")
        if query_ids.dim() != 2 or query_ids.shape[0] != 1:
            raise ValueError(f"query_ids must have shape (1, seq_len), got {query_ids.shape}")
        
        block_len = block_ids.shape[1]
        
        # Compute anchor and local window indices (anchor is global, only in first block(s))
        anchor_indices, local_indices = self._compute_anchor_local_indices(
            block_len, block_start_position
        )
        
        # Combine anchor and local indices (they might overlap for small blocks)
        fixed_indices_set = set(anchor_indices) | set(local_indices)
        fixed_indices = sorted(list(fixed_indices_set))
        
        # Compute prefix length
        prefix_len = sum(s.shape[1] for s in prefix_summaries) if prefix_summaries else 0
        
        # Combine: [prefix_summaries] + block + query
        if prefix_summaries:
            input_ids = torch.cat(prefix_summaries + [block_ids, query_ids], dim=1)
        else:
            input_ids = torch.cat([block_ids, query_ids], dim=1)
        
        total_len = input_ids.shape[1]
        query_len = query_ids.shape[1]
        
        # Start accumulation for Top-K selection
        accumulator = get_or_create_accumulator(self.model)
        accumulator.start_block_with_prefix(
            total_seq_len=total_len,
            prefix_len=prefix_len,
            block_len=block_len,
            query_len=query_len,
        )
        
        # Forward pass with output_attentions to get attention weights
        outputs = self.model(
            input_ids=input_ids,
            output_attentions=True,
            use_cache=False,
        )
        
        # Extract attentions and free other outputs immediately
        attentions = list(outputs.attentions)
        del outputs
        del input_ids
        
        # Accumulate attention scores from all layers
        num_layers = len(attentions)
        for layer_idx in range(num_layers):
            attn_weights = attentions[layer_idx]
            attentions[layer_idx] = None
            accumulator.accumulate(attn_weights, layer_idx)
            del attn_weights
        
        del attentions
        torch.cuda.empty_cache()
        
        # Get Top-K selection excluding anchor and local indices
        topk_indices = accumulator.select_top_k_with_exclusion(self.top_k, fixed_indices)
        
        # Clean up accumulator
        accumulator.finish_block()
        
        # Combine all indices: anchor + local + top-k (sorted)
        all_indices_set = set(fixed_indices) | set(topk_indices)
        all_selected_indices = sorted(list(all_indices_set))
        
        if not all_selected_indices:
            raise RuntimeError(
                f"No tokens were selected for block with {block_len} tokens. "
                f"This indicates a bug in the selection logic."
            )
        
        # Create tensor for indexing
        indices_tensor = torch.tensor(all_selected_indices, device=block_ids.device, dtype=torch.long)
        summary_ids = block_ids.index_select(dim=1, index=indices_tensor)
        
        # Compute original position IDs
        summary_positions = (block_start_position + indices_tensor).unsqueeze(0)
        
        del indices_tensor
        
        return summary_ids, summary_positions
    
    @torch.no_grad()
    def _build_kv_cache_from_summaries(
        self,
        summaries: List[torch.Tensor],
        summary_original_positions: List[torch.Tensor],
    ) -> Tuple:
        """
        Build the KV cache using only the summary tokens.
        Concatenate all summary tokens and use their original positions for position_ids.
        """
        # Concatenate all summary tokens and their positions
        all_summary_token_ids = torch.cat(summaries, dim=1)
        all_summary_positions = torch.cat(summary_original_positions, dim=1)
        
        # Forward pass to build the cache for summary tokens only
        outputs = self.model(
            input_ids=all_summary_token_ids,
            position_ids=all_summary_positions,
            use_cache=True,
            output_attentions=False,
        )
        return outputs.past_key_values, all_summary_token_ids, all_summary_positions
    
    @torch.no_grad()
    def _generate(
        self,
        query_ids: torch.Tensor,
        kv_cache: Tuple,
    ) -> torch.Tensor:
        """
        Phase 3: Generate response by projecting query onto sparse KV cache.
        """
        cache_len = kv_cache[0][0].shape[2]
        
        position_ids = torch.arange(
            cache_len, cache_len + query_ids.shape[1], device=self.device
        ).unsqueeze(0)
        
        past_key_values = DynamicCache.from_legacy_cache(kv_cache)
        
        outputs = self.model(
            input_ids=query_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        current_cache = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        generated_tokens = [next_token]
        
        for _ in range(self.max_new_tokens - 1):
            outputs = self.model(
                input_ids=next_token,
                past_key_values=current_cache,
                use_cache=True,
            )
            
            current_cache = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_tokens.append(next_token)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return torch.cat(generated_tokens, dim=1)
    
    def _get_output_text(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text and apply stop words."""
        text = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
        for stop_word in self.stop_words:
            text = text.split(stop_word)[0]
        return text.strip()
    
    def __call__(
        self,
        prompt_context: str,
        prompt_query: str,
    ) -> Dict[str, List[str]]:
        """
        Run the full pipeline.
        
        Args:
            prompt_context: The long context to process
            prompt_query: The query/question
            
        Returns:
            Dict with 'text' key containing generated response
        """
        print("=" * 60)
        print("Sequential Top-K Processing Pipeline (Global Anchor + Local + Top-K)")
        print("=" * 60)
        
        # Tokenize
        context_ids = self._tokenize(prompt_context)
        query_ids = self._tokenize_query_with_chat_template(prompt_query)
        
        print(f"\nContext length: {context_ids.shape[1]} tokens")
        print(f"Query length: {query_ids.shape[1]} tokens (with chat template)")
        
        # Split context into blocks
        blocks = self._split_into_blocks(context_ids)
        print(f"Split into {len(blocks)} blocks of ~{self.block_size} tokens each")
        print(f"Cache strategy: Global Anchor({self.anchor_size}) + Local({self.local_window_size}/block) + Top-K({self.top_k}/block)")
        
        # === PHASE 1: Sequential Sampling with Context ===
        print(f"\n--- Phase 1: Sequential Sampling ---")
        summaries = []
        summary_positions = []
        
        block_start_position = 0
        for i, block in enumerate(blocks):
            summary_ids, summary_pos = self._sample_topk_from_block_with_context(
                prefix_summaries=summaries,
                block_ids=block,
                query_ids=query_ids,
                block_start_position=block_start_position,
            )
            summaries.append(summary_ids)
            summary_positions.append(summary_pos)
            
            prefix_info = f"prefix={sum(s.shape[1] for s in summaries[:-1])}+" if summaries[:-1] else ""
            anchor_info = f" (includes global anchor)" if block_start_position < self.anchor_size else ""
            print(f"  Block {i+1}: {prefix_info}{block.shape[1]} tokens → Summary: {summary_ids.shape[1]} tokens (pos {block_start_position}-{block_start_position + block.shape[1] - 1}){anchor_info}")
            block_start_position += block.shape[1]
        
        # === PHASE 2: Build KV Cache ===
        print(f"\n--- Phase 2: Building KV Cache (Summary tokens only) ---")
        kv_cache, all_summary_token_ids, all_summary_positions = self._build_kv_cache_from_summaries(
            summaries, summary_positions
        )
        
        if all_summary_token_ids is None or all_summary_token_ids.numel() == 0:
            raise RuntimeError("No summary tokens were selected. Cannot build KV cache.")
        
        total_cache_len = kv_cache[0][0].shape[2]
        print(f"  Total KV cache length: {total_cache_len} tokens")

        # Free blocks and summaries
        del blocks, summaries, summary_positions, context_ids
        torch.cuda.empty_cache()    

        # === PHASE 3: Generation ===
        print(f"\n--- Phase 3: Generation ---")
        generated_ids = self._generate(query_ids, kv_cache)
        generated_text = self._get_output_text(generated_ids)
        print(f"\nGenerated {generated_ids.shape[1]} tokens")
        print("=" * 60)
        
        return {'text': [generated_text]}
