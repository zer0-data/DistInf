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
    
    def start_block(self, total_seq_len: int, block_token_count: int = -1):
        """
        Start accumulating scores for a new block.
        
        Args:
            total_seq_len: Total sequence length (block + query)
            block_token_count: Number of tokens in the block that are eligible for selection.
                              If -1, all tokens are eligible (no query appended).
        """
        self.reset()
        self.is_active = True
        self.total_seq_len = total_seq_len
        self.block_token_count = block_token_count if block_token_count > 0 else total_seq_len
    
    def accumulate(self, attn_weights: torch.Tensor, layer_idx: int):
        """
        Add attention scores from a layer to the accumulator.
        
        When query tokens are appended to the block, we use the attention from
        query tokens to score the block tokens. This enables query-guided selection.
        
        Args:
            attn_weights: Attention weights of shape (bsz, num_heads, q_len, kv_seq_len)
            layer_idx: The layer index
        """
        if not self.is_active:
            return
        
        bsz, num_heads, q_len, kv_seq_len = attn_weights.shape
        
        if self.block_token_count < kv_seq_len:
            # Query tokens are appended - use query attention for scoring block tokens
            query_start_idx = self.block_token_count
            
            # Get attention FROM query tokens TO block tokens (slice, don't copy)
            # Sum directly to minimize intermediate tensors
            layer_scores = attn_weights[:, :, query_start_idx:, :self.block_token_count].sum(dim=(1, 2))
        else:
            # No query tokens - use all attention
            layer_scores = attn_weights.sum(dim=(1, 2))
        
        if self.accumulated_scores is None:
            self.accumulated_scores = layer_scores  # No clone needed, we own this tensor
        else:
            # In-place addition to avoid allocating new tensor
            if self.accumulated_scores.shape[-1] == layer_scores.shape[-1]:
                self.accumulated_scores.add_(layer_scores)
            else:
                min_len = min(self.accumulated_scores.shape[-1], layer_scores.shape[-1])
                self.accumulated_scores[..., :min_len].add_(layer_scores[..., :min_len])
            del layer_scores  # Explicitly free
        
        self.layer_count += 1
    
    def select_top_k(self, top_k: int) -> List[int]:
        """
        Select top-K tokens based on accumulated attention scores.
        
        Args:
            top_k: Number of tokens to select
            
        Returns:
            List of selected indices (sorted to maintain sequence order)
        """
        if self.accumulated_scores is None:
            return []
        
        num_block_tokens = self.accumulated_scores.shape[-1]
        token_budget = min(num_block_tokens, top_k)
        
        # Select top-K indices
        _, top_k_indices = torch.topk(self.accumulated_scores[0], k=token_budget)
        
        # Sort to maintain sequence order
        top_k_indices_sorted, _ = torch.sort(top_k_indices)
        
        return top_k_indices_sorted.tolist()
    
    def finish_block(self) -> None:
        """Clean up after finishing a block."""
        if self.accumulated_scores is not None:
            del self.accumulated_scores
        self.accumulated_scores = None
        self.layer_count = 0
        self.block_token_count = 0
        self.total_seq_len = 0
        self.is_active = False
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


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
    Phase 1 - Sampling (Independent per block):
        Block1 + Query → Select Top-K from Block1 → Summary1
        Block2 + Query → Select Top-K from Block2 → Summary2
        ...
    
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
    
    def _split_into_blocks(self, token_ids: torch.Tensor) -> List[torch.Tensor]:
        """Split token IDs into blocks of block_size."""
        return list(token_ids.split(self.block_size, dim=1))
    
    @torch.no_grad()
    def _sample_topk_from_block(
        self,
        block_ids: torch.Tensor,
        query_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Phase 1: Sample top-K tokens from a block using query-guided selection.
        
        Input: Block + Query
        Output: Summary (top-K token IDs)
        
        Uses memory-efficient accumulation across all layers.
        """
        block_len = block_ids.shape[1]
        
        # Combine block + query
        input_ids = torch.cat([block_ids, query_ids], dim=1)
        total_len = input_ids.shape[1]
        
        # Start accumulation
        accumulator = get_or_create_accumulator(self.model)
        accumulator.start_block(total_seq_len=total_len, block_token_count=block_len)
        
        # Forward pass with output_attentions to get attention weights
        outputs = self.model(
            input_ids=input_ids,
            output_attentions=True,
            use_cache=False,
        )
        
        # Accumulate attention scores from all layers
        # Process and delete each layer's attention immediately to save memory
        attentions = outputs.attentions
        del outputs  # Free logits and other outputs early
        del input_ids  # Free combined input
        
        for layer_idx in range(len(attentions)):
            accumulator.accumulate(attentions[layer_idx], layer_idx)
            # Explicitly delete this layer's attention weights after processing
            attentions[layer_idx] = None
        
        del attentions
        torch.cuda.empty_cache()
        
        # Get top-K selection
        selected_indices = accumulator.select_top_k(self.top_k)
        
        # Clean up
        accumulator.finish_block()
        
        if not selected_indices:
            # Fallback: take first K tokens
            k = min(self.top_k, block_len)
            selected_indices = list(range(k))
        
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
        Phase 2: Build KV cache sequentially.
        
        Step 1: Block1 → KV1
        Step 2: Summary1 + Block2 → KV2 (full)
        Step 3: Summary1 + Summary2 + Block3 → KV3 (full)
        ...
        
        Final Cache = concatenation of all KVs
        """
        all_kv_caches = []
        
        for i, block in enumerate(blocks):
            if i == 0:
                # First block: just process Block1
                input_ids = block
            else:
                # Subsequent blocks: Summary1 + ... + Summary_{i-1} + Block_i
                prefix_summaries = summaries[:i]
                input_ids = torch.cat(prefix_summaries + [block], dim=1)
            
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                output_attentions=False,
            )
            
            # Store the full KV cache from this step
            all_kv_caches.append(outputs.past_key_values)
            
            del outputs.logits  # Free logits immediately (large tensor)
            del outputs
            if i > 0:
                del input_ids  # Free concatenated input
            torch.cuda.empty_cache()
            
            print(f"  Step {i+1}: Input length {input_ids.shape[1] if i == 0 else 'freed'} → KV cache built")
        
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
        """Concatenate multiple KV caches along the sequence dimension."""
        if not kv_caches:
            return None
        
        num_layers = len(kv_caches[0])
        combined_cache = []
        
        for layer_idx in range(num_layers):
            # Collect keys and values for this layer
            all_keys = [cache[layer_idx][0] for cache in kv_caches]
            all_values = [cache[layer_idx][1] for cache in kv_caches]
            
            combined_key = torch.cat(all_keys, dim=2)
            combined_value = torch.cat(all_values, dim=2)
            
            # Clear references to allow GC
            del all_keys, all_values
            
            combined_cache.append((combined_key, combined_value))
        
        return tuple(combined_cache)
    
    @torch.no_grad()
    def _generate(
        self,
        query_text: str,
        kv_cache: Tuple,
    ) -> torch.Tensor:
        """
        Phase 3: Generate response by projecting query onto sparse KV cache.
        Uses chat template for proper formatting with instruction-tuned models.
        """
        # Get the length of the KV cache
        cache_len = kv_cache[0][0].shape[2]
        
        # Apply chat template to query for proper formatting
        messages = [{"role": "user", "content": query_text}]
        query_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        
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
        query_ids = self._tokenize(prompt_query)
        
        print(f"\nContext length: {context_ids.shape[1]} tokens")
        print(f"Query length: {query_ids.shape[1]} tokens")
        
        # Split context into blocks
        blocks = self._split_into_blocks(context_ids)
        print(f"Split into {len(blocks)} blocks of ~{self.block_size} tokens each")
        
        # === PHASE 1: Sampling ===
        print(f"\n--- Phase 1: Sampling (Top-{self.top_k} per block) ---")
        summaries = []
        for i, block in enumerate(blocks):
            summary_ids = self._sample_topk_from_block(block, query_ids)
            summaries.append(summary_ids)
            print(f"  Block {i+1}: {block.shape[1]} tokens → Summary: {summary_ids.shape[1]} tokens")
        
        # === PHASE 2: Build KV Cache ===
        print(f"\n--- Phase 2: Building KV Cache (Sequential) ---")
        kv_cache = self._build_kv_cache_sequential(blocks, summaries)
        total_cache_len = kv_cache[0][0].shape[2]
        print(f"  Total KV cache length: {total_cache_len} tokens")
        
        # Free blocks and summaries after KV cache is built
        del blocks, summaries, context_ids, query_ids
        torch.cuda.empty_cache()
        
        # === PHASE 3: Generation ===
        print(f"\n--- Phase 3: Generation ---")
        generated_ids = self._generate(prompt_query, kv_cache)
        generated_text = self._get_output_text(generated_ids)
        
        print(f"\nGenerated {generated_ids.shape[1]} tokens")
        print("=" * 60)
        
        return {'text': [generated_text]}
