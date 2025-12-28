# topk_attention.py
# Sequential Top-K attention with query-guided token selection
# Memory-efficient: accumulates attention scores layer-by-layer
# Cache construction: Global Anchor Tokens + Local Window + Global Top-K
#
# IMPORTANT: This implementation uses CONTIGUOUS position IDs for the final
# KV cache to avoid RoPE mismatch between selection and generation phases.

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
    
    Pipeline (Two-Pass per Block to Avoid RoPE Mismatch):
    
    Phase 1 - Selection Pass (per block):
        For each block:
          - Forward pass with [prefix_summaries] + [block] + [query] (using prefix KV cache)
          - Accumulate attention scores across layers
          - Select tokens (anchor + local + top-k)
          - NOTE: KV cache from this pass is DISCARDED (has wrong positions for final cache)
    
    Phase 2 - KV Recomputation Pass (per block):
        For selected tokens from each block:
          - Recompute KV cache with CONTIGUOUS position IDs
          - Position IDs are: [current_final_cache_len, current_final_cache_len + num_selected)
          - This ensures selected tokens have positions matching their order in final cache
    
    Phase 3 - Generation:
        Query → Attend to accumulated KV Cache → Generate
        Query positions continue contiguously from final cache length
    
    Why Two Passes?
        RoPE (Rotary Position Embedding) bakes position information into K/V vectors.
        If we compute KV with position X during selection but then use it at position Y
        during generation, the attention patterns will be incorrect.
        
        By recomputing KV with final contiguous positions, we ensure consistency.
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
    
    def _merge_kv_caches(
        self,
        cache1: Optional[Tuple],
        cache2: Tuple,
    ) -> Tuple:
        """
        Merge two KV caches by concatenating along the sequence dimension.
        
        Args:
            cache1: First KV cache (can be None for first block)
            cache2: Second KV cache to append
            
        Returns:
            Merged KV cache
        """
        if cache1 is None:
            return cache2
        
        merged_kv = []
        for (k1, v1), (k2, v2) in zip(cache1, cache2):
            merged_key = torch.cat([k1, k2], dim=2)
            merged_value = torch.cat([v1, v2], dim=2)
            merged_kv.append((merged_key, merged_value))
        
        return tuple(merged_kv)
    
    @torch.no_grad()
    def _select_tokens_from_block(
        self,
        prefix_kv_cache: Optional[Tuple],
        block_ids: torch.Tensor,
        query_ids: torch.Tensor,
        block_start_position: int,
        current_cache_len: int,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Phase 1: Select tokens from a block using query-guided attention scoring.
        
        This pass computes attention scores to select important tokens.
        The KV cache from this pass is DISCARDED because it has position IDs
        that won't match the final contiguous cache.
        
        Args:
            prefix_kv_cache: KV cache from previous blocks' summaries (for context)
            block_ids: Token IDs of the current block
            query_ids: Query token IDs
            block_start_position: The starting position of this block in the full context
            current_cache_len: Current length of the accumulated KV cache
        
        Returns:
            Tuple of (anchor_indices, local_indices, topk_indices) - all relative to block
        """
        block_len = block_ids.shape[1]
        query_len = query_ids.shape[1]
        
        # Compute anchor and local window indices
        anchor_indices, local_indices = self._compute_anchor_local_indices(
            block_len, block_start_position
        )
        
        # Combine anchor and local indices for exclusion from top-k
        fixed_indices_set = set(anchor_indices) | set(local_indices)
        fixed_indices = sorted(list(fixed_indices_set))
        
        # Build input: [block] + [query] (prefix is handled via KV cache)
        input_ids = torch.cat([block_ids, query_ids], dim=1)
        total_input_len = input_ids.shape[1]
        
        # Position IDs for selection pass - these are temporary and will be discarded
        # We use positions continuing from current cache for proper attention computation
        position_ids = torch.arange(
            current_cache_len,
            current_cache_len + total_input_len,
            device=block_ids.device
        ).unsqueeze(0)
        
        # Convert prefix KV cache to DynamicCache if provided
        past_key_values = None
        if prefix_kv_cache is not None:
            past_key_values = DynamicCache.from_legacy_cache(prefix_kv_cache)
        
        # Start accumulation for Top-K selection
        accumulator = get_or_create_accumulator(self.model)
        accumulator.start_block_with_prefix(
            total_seq_len=total_input_len,
            prefix_len=0,  # Prefix is in KV cache
            block_len=block_len,
            query_len=query_len,
        )
        
        # Forward pass - we only need attention weights, KV cache will be discarded
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=True,
            use_cache=False,  # Don't need KV cache from selection pass
        )
        
        # Extract attention weights
        attentions = list(outputs.attentions)
        del outputs
        
        # Accumulate attention scores from all layers
        num_layers = len(attentions)
        for layer_idx in range(num_layers):
            attn_weights = attentions[layer_idx]
            attentions[layer_idx] = None
            accumulator.accumulate(attn_weights, layer_idx)
            del attn_weights
        
        del attentions
        
        # Get Top-K selection excluding anchor and local indices
        topk_indices = accumulator.select_top_k_with_exclusion(self.top_k, fixed_indices)
        
        # Clean up accumulator
        accumulator.finish_block()
        
        return anchor_indices, local_indices, topk_indices
    
    @torch.no_grad()
    def _compute_kv_for_selected_tokens(
        self,
        selected_token_ids: torch.Tensor,
        start_position: int,
    ) -> Tuple:
        """
        Phase 2: Compute KV cache for selected tokens with correct contiguous positions.
        
        This is the key to avoiding RoPE mismatch. We recompute the KV cache for
        selected tokens using their final positions in the accumulated cache.
        
        Args:
            selected_token_ids: Token IDs to compute KV for, shape (1, num_selected)
            start_position: Starting position ID for these tokens in the final cache
        
        Returns:
            KV cache tuple for the selected tokens
        """
        num_tokens = selected_token_ids.shape[1]
        
        # Position IDs are CONTIGUOUS starting from start_position
        # These will be the final positions in the accumulated cache
        position_ids = torch.arange(
            start_position,
            start_position + num_tokens,
            device=selected_token_ids.device
        ).unsqueeze(0)
        
        # Forward pass to compute KV cache with correct positions
        # No attention output needed, no past_key_values (fresh computation)
        outputs = self.model(
            input_ids=selected_token_ids,
            position_ids=position_ids,
            past_key_values=None,
            output_attentions=False,
            use_cache=True,
        )
        
        kv_cache = outputs.past_key_values
        
        # Convert DynamicCache back to legacy format
        if hasattr(kv_cache, 'to_legacy_cache'):
            kv_cache = kv_cache.to_legacy_cache()
        
        del outputs
        
        return kv_cache
    
    @torch.no_grad()
    def _process_block(
        self,
        prefix_kv_cache: Optional[Tuple],
        block_ids: torch.Tensor,
        query_ids: torch.Tensor,
        block_start_position: int,
        current_cache_len: int,
    ) -> Tuple[torch.Tensor, List[int], Tuple]:
        """
        Process a single block: select tokens and compute their KV cache.
        
        This combines the two phases:
        1. Selection pass: Find important tokens using attention scores
        2. KV computation pass: Compute KV cache with correct positions
        
        Args:
            prefix_kv_cache: KV cache from previous blocks' summaries
            block_ids: Token IDs of the current block
            query_ids: Query token IDs
            block_start_position: Starting position of this block in full context
            current_cache_len: Current length of the accumulated KV cache
        
        Returns:
            Tuple of (summary_ids, selected_indices, kv_cache):
            - summary_ids: Selected token IDs shape (1, num_selected)
            - selected_indices: List of indices that were selected from the block
            - kv_cache: KV cache for selected tokens with correct positions
        """
        # Validate inputs
        if block_ids.dim() != 2 or block_ids.shape[0] != 1:
            raise ValueError(f"block_ids must have shape (1, seq_len), got {block_ids.shape}")
        if query_ids.dim() != 2 or query_ids.shape[0] != 1:
            raise ValueError(f"query_ids must have shape (1, seq_len), got {query_ids.shape}")
        
        # Phase 1: Select tokens
        anchor_indices, local_indices, topk_indices = self._select_tokens_from_block(
            prefix_kv_cache=prefix_kv_cache,
            block_ids=block_ids,
            query_ids=query_ids,
            block_start_position=block_start_position,
            current_cache_len=current_cache_len,
        )
        
        # Combine all selected indices (sorted for sequence order)
        all_indices_set = set(anchor_indices) | set(local_indices) | set(topk_indices)
        all_selected_indices = sorted(list(all_indices_set))
        
        if not all_selected_indices:
            raise RuntimeError(
                f"No tokens were selected for block with {block_ids.shape[1]} tokens. "
                f"This indicates a bug in the selection logic."
            )
        
        # Extract token IDs for selected indices
        indices_tensor = torch.tensor(all_selected_indices, device=block_ids.device, dtype=torch.long)
        summary_ids = block_ids.index_select(dim=1, index=indices_tensor)
        
        # Phase 2: Compute KV cache with correct contiguous positions
        # The selected tokens will have positions [current_cache_len, current_cache_len + num_selected)
        selected_kv_cache = self._compute_kv_for_selected_tokens(
            selected_token_ids=summary_ids,
            start_position=current_cache_len,
        )
        
        # Clean up
        del indices_tensor
        torch.cuda.empty_cache()
        
        return summary_ids, all_selected_indices, selected_kv_cache
    
    @torch.no_grad()
    def _generate(
        self,
        query_ids: torch.Tensor,
        kv_cache: Tuple,
    ) -> torch.Tensor:
        """
        Phase 3: Generate response using the accumulated KV cache.
        
        Args:
            query_ids: Query token IDs
            kv_cache: Accumulated KV cache from all selected summary tokens
        """
        cache_len = kv_cache[0][0].shape[2]
        query_len = query_ids.shape[1]
        
        # Query positions continue from cache length (CONTIGUOUS)
        position_ids = torch.arange(
            cache_len,
            cache_len + query_len,
            device=self.device
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
        
        # Track the next position for generated tokens (CONTIGUOUS)
        next_position = cache_len + query_len
        
        for _ in range(self.max_new_tokens - 1):
            token_position = torch.tensor([[next_position]], device=self.device)
            
            outputs = self.model(
                input_ids=next_token,
                position_ids=token_position,
                past_key_values=current_cache,
                use_cache=True,
            )
            
            current_cache = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_tokens.append(next_token)
            
            next_position += 1
            
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
        print("Sequential Top-K Processing Pipeline (RoPE-Corrected)")
        print("=" * 60)
        
        # Tokenize
        context_ids = self._tokenize(prompt_context)
        query_ids = self._tokenize_query_with_chat_template(prompt_query)
        
        total_context_length = context_ids.shape[1]
        
        print(f"\nContext length: {total_context_length} tokens")
        print(f"Query length: {query_ids.shape[1]} tokens (with chat template)")
        
        # Split context into blocks
        blocks = self._split_into_blocks(context_ids)
        print(f"Split into {len(blocks)} blocks of ~{self.block_size} tokens each")
        print(f"Cache strategy: Global Anchor({self.anchor_size}) + Local({self.local_window_size}/block) + Top-K({self.top_k}/block)")
        print(f"\nNote: Two-pass processing to ensure RoPE consistency")
        
        # === PHASES 1 & 2: Selection + KV Recomputation ===
        print(f"\n--- Phases 1 & 2: Selection + KV Recomputation ---")
        
        accumulated_kv_cache = None
        current_cache_len = 0
        
        block_start_position = 0
        for i, block in enumerate(blocks):
            summary_ids, selected_indices, block_kv = self._process_block(
                prefix_kv_cache=accumulated_kv_cache,
                block_ids=block,
                query_ids=query_ids,
                block_start_position=block_start_position,
                current_cache_len=current_cache_len,
            )
            
            # Merge the KV cache (already has correct positions)
            accumulated_kv_cache = self._merge_kv_caches(accumulated_kv_cache, block_kv)
            current_cache_len = accumulated_kv_cache[0][0].shape[2]
            
            anchor_info = f" (includes global anchor)" if block_start_position < self.anchor_size else ""
            print(f"  Block {i+1}: {block.shape[1]} tokens → Selected: {len(selected_indices)} → Cache: {current_cache_len} (pos 0-{current_cache_len-1}){anchor_info}")
            
            block_start_position += block.shape[1]
            
            # Clean up
            del block_kv
        
        if accumulated_kv_cache is None:
            raise RuntimeError("No KV cache was accumulated. Cannot proceed to generation.")
        
        total_cache_len = accumulated_kv_cache[0][0].shape[2]
        print(f"\n  Final KV cache: {total_cache_len} tokens (contiguous positions 0-{total_cache_len-1})")

        # Free blocks and context
        del blocks, context_ids
        torch.cuda.empty_cache()

        # === PHASE 3: Generation ===
        print(f"\n--- Phase 3: Generation ---")
        print(f"  Query positions: {total_cache_len} to {total_cache_len + query_ids.shape[1] - 1}")
        generated_ids = self._generate(query_ids, accumulated_kv_cache)
        generated_text = self._get_output_text(generated_ids)
        print(f"\nGenerated {generated_ids.shape[1]} tokens")
        print("=" * 60)
        
        return {'text': [generated_text]}
