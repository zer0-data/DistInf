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
    Implements the iterative single-summary block processing pipeline with query-guided top-K selection.
    
    Pipeline:
    Phase 1 - Iterative Summary Update:
        Block1 + Query → Select Top-K from Block1 → Summary1
        Summary1 + Block2 + Query → Select Top-K from (Summary1 + Block2) → Summary2
        Summary2 + Block3 + Query → Select Top-K from (Summary2 + Block3) → Summary3
        ...
        At each step, only K summary tokens are retained for the next step.
    
    Phase 2 - Build KV Cache:
        Final summary tokens (K) → Build sparse KV cache (single forward pass)
    
    Phase 3 - Generation:
        Query → Attend to Final Summary KV Cache → Generate
    
    Uses memory-efficient attention score accumulation (layer-by-layer).
    """
    @torch.no_grad()
    def _build_kv_cache_from_summary(
        self,
        summary_ids: torch.Tensor,
        summary_positions: torch.Tensor,
    ) -> Tuple:
        """
        Build the KV cache using only the final summary tokens (mask all others).
        Args:
            summary_ids: (1, k) tensor of summary token ids
            summary_positions: (1, k) tensor of original position ids
        Returns:
            (kv_cache, summary_ids, summary_positions)
        """
        outputs = self.model(
            input_ids=summary_ids,
            position_ids=summary_positions,
            use_cache=True,
            output_attentions=False,
        )
        return outputs.past_key_values, summary_ids, summary_positions
    
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
    def _sample_topk_from_summary_and_block(
        self,
        summary_ids: torch.Tensor,
        summary_positions: torch.Tensor,
        block_ids: torch.Tensor,
        block_start_position: int,
        query_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Iteratively update the summary: select top-K from (summary + block) using query-guided attention.
        Args:
            summary_ids: (1, k_prev) tensor of previous summary token ids (empty for first block)
            summary_positions: (1, k_prev) tensor of previous summary positions (empty for first block)
            block_ids: (1, block_len) tensor of current block token ids
            block_start_position: int, starting position of block in full context
            query_ids: (1, q_len) tensor of query token ids
        Returns:
            (summary_ids, summary_positions): both (1, k) tensors for new summary
        """
        # Concatenate summary and block
        if summary_ids is not None and summary_ids.shape[1] > 0:
            input_ids = torch.cat([summary_ids, block_ids, query_ids], dim=1)
            prefix_len = 0
            block_len = summary_ids.shape[1] + block_ids.shape[1]
        else:
            input_ids = torch.cat([block_ids, query_ids], dim=1)
            prefix_len = 0
            block_len = block_ids.shape[1]
        total_len = input_ids.shape[1]
        query_len = query_ids.shape[1]
        # Prepare position ids
        if summary_positions is not None and summary_positions.shape[1] > 0:
            all_positions = torch.cat([
                summary_positions,
                torch.arange(block_start_position, block_start_position + block_ids.shape[1], device=block_ids.device).unsqueeze(0)
            ], dim=1)
        else:
            all_positions = torch.arange(block_start_position, block_start_position + block_ids.shape[1], device=block_ids.device).unsqueeze(0)
        # Accumulate attention
        accumulator = get_or_create_accumulator(self.model)
        accumulator.start_block_with_prefix(
            total_seq_len=total_len,
            prefix_len=prefix_len,
            block_len=block_len,
            query_len=query_len,
        )
        outputs = self.model(
            input_ids=input_ids,
            position_ids=all_positions,
            output_attentions=True,
            use_cache=False,
        )
        attentions = list(outputs.attentions)
        del outputs
        del input_ids
        num_layers = len(attentions)
        for layer_idx in range(num_layers):
            attn_weights = attentions[layer_idx]
            attentions[layer_idx] = None
            accumulator.accumulate(attn_weights, layer_idx)
            del attn_weights
        del attentions
        torch.cuda.empty_cache()
        selected_indices = accumulator.select_top_k(self.top_k)
        accumulator.finish_block()
        if not selected_indices:
            raise RuntimeError("Top-K selection returned no indices for summary+block. Check attention accumulation.")
        indices_tensor = torch.tensor(selected_indices, device=block_ids.device, dtype=torch.long)
        new_summary_ids = torch.cat([summary_ids, block_ids], dim=1) if (summary_ids is not None and summary_ids.shape[1] > 0) else block_ids
        new_summary_ids = new_summary_ids.index_select(dim=1, index=indices_tensor)
        new_summary_positions = all_positions.index_select(dim=1, index=indices_tensor)
        del indices_tensor
        return new_summary_ids, new_summary_positions
    
    # ...existing code...
    
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
        Run the full iterative single-summary pipeline.
        Args:
            prompt_context: The long context to process
            prompt_query: The query/question
        Returns:
            Dict with 'text' key containing generated response
        """
        print("=" * 60)
        print("Iterative Single-Summary Top-K Pipeline")
        # Tokenize context and query
        context_ids = self._tokenize(prompt_context)
        query_ids = self._tokenize_query_with_chat_template(prompt_query)
        blocks = self._split_into_blocks(context_ids)
        summary_ids = None
        summary_positions = None
        for i, block_ids in enumerate(blocks):
            block_start = i * self.block_size
            if i == 0:
                # First block: summary is just from block 1
                summary_ids, summary_positions = self._sample_topk_from_summary_and_block(
                    summary_ids=None,
                    summary_positions=None,
                    block_ids=block_ids,
                    block_start_position=block_start,
                    query_ids=query_ids,
                )
            else:
                summary_ids, summary_positions = self._sample_topk_from_summary_and_block(
                    summary_ids=summary_ids,
                    summary_positions=summary_positions,
                    block_ids=block_ids,
                    block_start_position=block_start,
                    query_ids=query_ids,
                )
        # Final summary_ids, summary_positions are the K tokens for the whole context
        kv_cache, final_summary_ids, final_summary_positions = self._build_kv_cache_from_summary(
            summary_ids=summary_ids,
            summary_positions=summary_positions,
        )
        # Generate
        generated = self._generate(query_ids=query_ids, kv_cache=kv_cache)
        output_text = self._get_output_text(generated)
        return {"text": [output_text]}
