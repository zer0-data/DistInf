# topk_attention.py
# Sequential Top-K attention with query-guided token selection
# Memory-efficient: accumulates attention scores layer-by-layer

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
    
    Pipeline:
    Phase 1 - Sequential Sampling (with context propagation):
        Block1 + Query → Select Top-K from Block1 → Summary1
        Summary1 + Block2 + Query → Select Top-K from Block2 → Summary2
        Summary1 + Summary2 + Block3 + Query → Select Top-K from Block3 → Summary3
        ... (skip last block - its summary is not used)
    
    Phase 2 - Build KV Cache (Sequential):
        Block1 → KV1
        Summary1 + Block2 → KV2 (full)
        Summary1 + Summary2 + Block3 → KV3 (full)
        ...
        Final Cache = KV1 + KV2 + KV3 + ...
    
    Phase 3 - Generation:
        Query → Attend to Final Cache → Generate
    
    Uses memory-efficient attention score accumulation (layer-by-layer).
    """
    
    def __init__(
        self,
        model_path: str,
        top_k: int = 256,
        block_size: int = 2048,
        max_new_tokens: int = 100,
        stop_words: Optional[List[str]] = None,
    ):
        """
        Args:
            model_path: Path to the HuggingFace model
            top_k: Number of tokens to select from each block
            block_size: Size of each context block
            max_new_tokens: Maximum tokens to generate
            stop_words: List of stop words for generation
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.model_path = model_path
        self.top_k = top_k
        self.block_size = block_size
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words or []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"\n[SequentialTopKProcessor] Initializing...")
        print(f"  Model: {model_path}")
        print(f"  Top-K: {top_k}, Block Size: {block_size}")
        
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
    
    @torch.no_grad()
    def _sample_topk_from_block_with_context(
        self,
        prefix_summaries: List[torch.Tensor],
        block_ids: torch.Tensor,
        query_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Phase 1: Sample top-K tokens from a block using query-guided selection with context.
        
        Input: [Summary1 + ... + Summary_{i-1}] + Block_i + Query
        Output: Summary_i (top-K token IDs from Block_i only)
        
        The attention scores are accumulated across all layers, but tokens are only
        selected from Block_i (not from prefix summaries or query).
        
        Uses memory-efficient accumulation across all layers.
        
        Raises:
            ValueError: If inputs have invalid shapes or dimensions
        """
        # Validate inputs
        if block_ids.dim() != 2 or block_ids.shape[0] != 1:
            raise ValueError(f"block_ids must have shape (1, seq_len), got {block_ids.shape}")
        if query_ids.dim() != 2 or query_ids.shape[0] != 1:
            raise ValueError(f"query_ids must have shape (1, seq_len), got {query_ids.shape}")
        if block_ids.shape[1] == 0:
            raise ValueError("block_ids cannot be empty")
        if query_ids.shape[1] == 0:
            raise ValueError("query_ids cannot be empty")
        
        for i, summary in enumerate(prefix_summaries):
            if summary.dim() != 2 or summary.shape[0] != 1:
                raise ValueError(f"prefix_summaries[{i}] must have shape (1, seq_len), got {summary.shape}")
            if summary.shape[1] == 0:
                raise ValueError(f"prefix_summaries[{i}] cannot be empty")
        
        block_len = block_ids.shape[1]
        
        if self.top_k > block_len:
            import warnings
            warnings.warn(
                f"top_k ({self.top_k}) exceeds block length ({block_len}). "
                f"Will select all {block_len} tokens from this block."
            )
        
        # Compute prefix length (all summaries before this block)
        prefix_len = sum(s.shape[1] for s in prefix_summaries) if prefix_summaries else 0
        
        # Combine: [prefix_summaries] + block + query
        if prefix_summaries:
            input_ids = torch.cat(prefix_summaries + [block_ids, query_ids], dim=1)
        else:
            input_ids = torch.cat([block_ids, query_ids], dim=1)
        
        total_len = input_ids.shape[1]
        query_len = query_ids.shape[1]
        
        # Start accumulation
        # block_token_count is set to select only from block (not prefix or query)
        # We need to track: prefix_len, block_len, query_len
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
        # Note: outputs.attentions is a tuple, so we convert to list for proper cleanup
        attentions = list(outputs.attentions)
        del outputs  # Free logits and other outputs early
        del input_ids  # Free combined input
        
        # Accumulate attention scores from all layers
        # Process and delete each layer's attention immediately to save memory
        num_layers = len(attentions)
        for layer_idx in range(num_layers):
            attn_weights = attentions[layer_idx]
            attentions[layer_idx] = None  # Remove reference from list
            accumulator.accumulate(attn_weights, layer_idx)
            del attn_weights  # Explicitly free this layer's attention weights
        
        del attentions
        torch.cuda.empty_cache()
        
        # Get top-K selection
        selected_indices = accumulator.select_top_k(self.top_k)
        
        # Clean up
        accumulator.finish_block()
        
        if not selected_indices:
            # Empty selection indicates a bug - attention accumulation failed
            raise RuntimeError(
                f"Top-K selection returned no indices for block with {block_len} tokens. "
                f"This indicates a bug in attention accumulation. "
                f"Check that the model's attention outputs are valid and non-empty."
            )
        
        # Convert to tensor and index directly (avoid storing indices separately)
        indices_tensor = torch.tensor(selected_indices, device=self.device, dtype=torch.long)
        summary_ids = block_ids[:, indices_tensor]
        del indices_tensor  # Free immediately
        
        return summary_ids
    
    @torch.no_grad()
    def _build_kv_cache_sequential(
        self,
        blocks: List[torch.Tensor],
        summaries: List[torch.Tensor],
    ) -> Tuple:
        """
        Phase 2: Build KV cache sequentially with correct position IDs.
        
        Step 1: Block1 → KV1 (positions 0 to len(Block1)-1)
        Step 2: Summary1 + Block2 → KV2 (positions continue from where KV1 ended)
        Step 3: Summary1 + Summary2 + Block3 → KV3 (positions continue)
        ...
        
        Final Cache = concatenation of all KVs with continuous position IDs
        """
        all_kv_caches = []
        cumulative_position = 0  # Track position offset for continuous positioning
        
        for i, block in enumerate(blocks):
            if i == 0:
                # First block: just process Block1
                input_ids = block
                seq_len = block.shape[1]
            else:
                # Subsequent blocks: Summary1 + ... + Summary_{i-1} + Block_i
                prefix_summaries = summaries[:i]
                input_ids = torch.cat(prefix_summaries + [block], dim=1)
                seq_len = input_ids.shape[1]
            
            # Create position IDs starting from cumulative position
            position_ids = torch.arange(
                cumulative_position, 
                cumulative_position + seq_len,
                device=self.device
            ).unsqueeze(0)
            
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=True,
                output_attentions=False,
            )
            
            # Store the full KV cache from this step
            all_kv_caches.append(outputs.past_key_values)
            
            # Update cumulative position for next iteration
            cumulative_position += seq_len
            
            print(f"  Step {i+1}: {seq_len} tokens (positions {cumulative_position - seq_len}-{cumulative_position - 1}) → KV cache built")
            
            del outputs.logits  # Free logits immediately (large tensor)
            del outputs
            if i > 0:
                del input_ids  # Free concatenated input
            torch.cuda.empty_cache()
        
        # Concatenate all KV caches and free individual caches
        final_kv_cache = self._concatenate_kv_caches(all_kv_caches)
        
        # Free individual caches after concatenation
        for cache in all_kv_caches:
            for layer_kv in cache:
                del layer_kv
        del all_kv_caches
        torch.cuda.empty_cache()
        
        return final_kv_cache
    
    def _concatenate_kv_caches(self, kv_caches: List[Tuple]) -> Tuple:
        """
        Concatenate multiple KV caches along the sequence dimension.
        
        Args:
            kv_caches: List of KV caches, each is a tuple of (key, value) tuples per layer
                      Shape per layer: key/value are (bsz, num_heads, seq_len, head_dim)
        
        Returns:
            Combined KV cache with all sequences concatenated on dim=2
        
        Raises:
            ValueError: If caches have inconsistent structure or shapes
        """
        if not kv_caches:
            return None
        
        if len(kv_caches) == 1:
            return kv_caches[0]  # No concatenation needed
        
        # Validate all caches have same number of layers
        num_layers = len(kv_caches[0])
        for i, cache in enumerate(kv_caches[1:], start=1):
            if len(cache) != num_layers:
                raise ValueError(
                    f"KV cache {i} has {len(cache)} layers, expected {num_layers} layers. "
                    f"All caches must have the same number of layers."
                )
        
        # Get reference shapes from first cache for validation
        ref_cache = kv_caches[0]
        ref_device = ref_cache[0][0].device
        ref_dtype = ref_cache[0][0].dtype
        
        combined_cache = []
        
        for layer_idx in range(num_layers):
            # Collect keys and values for this layer
            all_keys = []
            all_values = []
            
            for cache_idx, cache in enumerate(kv_caches):
                key, value = cache[layer_idx]
                
                # Validate device consistency
                if key.device != ref_device:
                    raise ValueError(
                        f"KV cache {cache_idx}, layer {layer_idx}: key is on {key.device}, "
                        f"expected {ref_device}. All tensors must be on the same device."
                    )
                if value.device != ref_device:
                    raise ValueError(
                        f"KV cache {cache_idx}, layer {layer_idx}: value is on {value.device}, "
                        f"expected {ref_device}. All tensors must be on the same device."
                    )
                
                # Validate shape compatibility (all dims except seq_len must match)
                # Shape: (bsz, num_heads, seq_len, head_dim)
                if key.dim() != 4:
                    raise ValueError(
                        f"KV cache {cache_idx}, layer {layer_idx}: key has {key.dim()} dims, "
                        f"expected 4 dims (bsz, num_heads, seq_len, head_dim)."
                    )
                
                if cache_idx > 0:
                    ref_key = kv_caches[0][layer_idx][0]
                    # Check bsz, num_heads, head_dim match (dims 0, 1, 3)
                    if key.shape[0] != ref_key.shape[0]:
                        raise ValueError(
                            f"KV cache {cache_idx}, layer {layer_idx}: batch size {key.shape[0]} "
                            f"doesn't match reference batch size {ref_key.shape[0]}."
                        )
                    if key.shape[1] != ref_key.shape[1]:
                        raise ValueError(
                            f"KV cache {cache_idx}, layer {layer_idx}: num_heads {key.shape[1]} "
                            f"doesn't match reference num_heads {ref_key.shape[1]}."
                        )
                    if key.shape[3] != ref_key.shape[3]:
                        raise ValueError(
                            f"KV cache {cache_idx}, layer {layer_idx}: head_dim {key.shape[3]} "
                            f"doesn't match reference head_dim {ref_key.shape[3]}."
                        )
                
                all_keys.append(key)
                all_values.append(value)
            
            # Concatenate along sequence dimension (dim=2)
            combined_key = torch.cat(all_keys, dim=2)
            combined_value = torch.cat(all_values, dim=2)
            
            # Clear references to allow GC
            del all_keys, all_values
            
            combined_cache.append((combined_key, combined_value))
        
        return tuple(combined_cache)
    
    @torch.no_grad()
    def _generate(
        self,
        query_ids: torch.Tensor,
        kv_cache: Tuple,
    ) -> torch.Tensor:
        """
        Phase 3: Generate response by projecting query onto sparse KV cache.
        
        Args:
            query_ids: Pre-tokenized query (with chat template applied)
            kv_cache: The KV cache from Phase 2
        """
        # Get the length of the KV cache
        cache_len = kv_cache[0][0].shape[2]
        
        # Position IDs for query start after the cache
        position_ids = torch.arange(
            cache_len, cache_len + query_ids.shape[1], device=self.device
        ).unsqueeze(0)
        
        # Convert to DynamicCache
        past_key_values = DynamicCache.from_legacy_cache(kv_cache)
        
        # Prefill with query
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
        
        # Autoregressive generation
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
        print("Sequential Top-K Processing Pipeline")
        print("=" * 60)
        
        # Tokenize
        context_ids = self._tokenize(prompt_context)
        # Use chat template for query to match Phase 3 generation format
        query_ids = self._tokenize_query_with_chat_template(prompt_query)
        
        print(f"\nContext length: {context_ids.shape[1]} tokens")
        print(f"Query length: {query_ids.shape[1]} tokens (with chat template)")
        
        # Split context into blocks
        blocks = self._split_into_blocks(context_ids)
        print(f"Split into {len(blocks)} blocks of ~{self.block_size} tokens each")
        
        # === PHASE 1: Sequential Sampling with Context ===
        print(f"\n--- Phase 1: Sequential Sampling (Top-{self.top_k} per block) ---")
        summaries = []
        num_blocks = len(blocks)
        
        # Skip sampling for the last block (its summary is not used anywhere)
        blocks_to_sample = num_blocks - 1 if num_blocks > 1 else 0
        
        for i in range(blocks_to_sample):
            block = blocks[i]
            # Use all previous summaries as prefix context
            summary_ids = self._sample_topk_from_block_with_context(
                prefix_summaries=summaries,  # All summaries so far
                block_ids=block,
                query_ids=query_ids,
            )
            summaries.append(summary_ids)
            prefix_info = f"prefix={sum(s.shape[1] for s in summaries[:-1])}+" if summaries[:-1] else ""
            print(f"  Block {i+1}: {prefix_info}{block.shape[1]} tokens → Summary: {summary_ids.shape[1]} tokens")
        
        if num_blocks > 1:
            print(f"  Block {num_blocks}: {blocks[-1].shape[1]} tokens (no sampling - last block)")
        
        # === PHASE 2: Build KV Cache ===
        print(f"\n--- Phase 2: Building KV Cache (Sequential) ---")
        kv_cache = self._build_kv_cache_sequential(blocks, summaries)
        total_cache_len = kv_cache[0][0].shape[2]
        print(f"  Total KV cache length: {total_cache_len} tokens")
        
        # Free blocks and summaries after KV cache is built
        del blocks, summaries, context_ids
        torch.cuda.empty_cache()
        
        # === PHASE 3: Generation ===
        print(f"\n--- Phase 3: Generation ---")
        # Reuse the same chat-templated query_ids from Phase 1
        generated_ids = self._generate(query_ids, kv_cache)
        
        del query_ids
        generated_text = self._get_output_text(generated_ids)
        
        print(f"\nGenerated {generated_ids.shape[1]} tokens")
        print("=" * 60)
        
        return {'text': [generated_text]}
