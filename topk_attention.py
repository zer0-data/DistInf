# topk_attention.py
# Sequential Top-K attention with query-guided token selection
# Memory-efficient: accumulates attention scores layer-by-layer
# Cache construction: Global Anchor Tokens + Local Window + Global Top-K
# POSITION HANDLING: Selected tokens KEEP their original position IDs.

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
    
    Can operate in two modes:
    1. Manual mode: Call accumulate() explicitly with attention weights
    2. Hook mode: Register hooks on attention layers to accumulate automatically
    """
    
    def __init__(self):
        self.reset()
        self._hooks = []
    
    def reset(self):
        """Reset the accumulator for a new block."""
        self.accumulated_scores: Optional[torch.Tensor] = None
        self.layer_count: int = 0
        self.is_active: bool = False
        self.block_token_count: int = 0
        self.total_seq_len: int = 0
        self.prefix_len: int = 0
        self.query_len: int = 0
        self.expected_q_len: int = 0
        self.prefix_in_kv_cache: int = 0
    
    def start_block(self, total_seq_len: int, block_token_count: int = -1):
        self.reset()
        self.is_active = True
        self.total_seq_len = total_seq_len
        self.prefix_len = 0
        self.prefix_in_kv_cache = 0
        
        if block_token_count > 0:
            self.block_token_count = block_token_count
            self.query_len = total_seq_len - block_token_count
        else:
            self.block_token_count = total_seq_len
            self.query_len = 0
        
        self.expected_q_len = self.block_token_count + self.query_len
    
    def start_block_with_prefix(
        self, 
        total_seq_len: int, 
        prefix_len: int,
        block_len: int,
        query_len: int,
        prefix_in_kv_cache: int = 0,
    ):
        if prefix_len < 0:
            raise ValueError(f"prefix_len cannot be negative, got {prefix_len}")
        if block_len <= 0:
            raise ValueError(f"block_len must be positive, got {block_len}")
        if query_len < 0:
            raise ValueError(f"query_len cannot be negative, got {query_len}")
        if prefix_in_kv_cache < 0:
            raise ValueError(f"prefix_in_kv_cache cannot be negative, got {prefix_in_kv_cache}")
        
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
        self.block_token_count = block_len
        self.query_len = query_len
        self.prefix_in_kv_cache = prefix_in_kv_cache
        
        self.expected_q_len = total_seq_len - prefix_in_kv_cache
    
    def _create_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            # output is typically (attn_output, attn_weights, past_key_value) when output_attentions=True
            # or just (attn_output, past_key_value) when output_attentions=False
            if not self.is_active:
                return
            
            # Try to extract attention weights from output
            attn_weights = None
            if isinstance(output, tuple) and len(output) >= 2:
                # Check if second element looks like attention weights (4D tensor)
                candidate = output[1]
                if candidate is not None and isinstance(candidate, torch.Tensor) and candidate.dim() == 4:
                    attn_weights = candidate
            
            if attn_weights is not None:
                self.accumulate(attn_weights, layer_idx)
        
        return hook
    
    def register_hooks(self, model):
        """Register forward hooks on all attention layers."""
        self.remove_hooks()  # Clean up any existing hooks
        
        # Find attention layers - this works for most HuggingFace models
        attention_modules = {}
        for name, module in model.named_modules():
            # Common attention module names across different architectures
            if any(attn_name in name.lower() for attn_name in ['attention', 'attn']) and \
               not any(skip in name.lower() for skip in ['layernorm', 'ln', 'norm', 'dropout']):
                # Check if this is an attention module (not a container)
                if hasattr(module, 'forward') and 'Attention' in type(module).__name__:
                    attention_modules[name] = module
        
        # Filter out parent modules - only keep leaf attention modules
        # A module is a parent if another module's name starts with its name + "."
        leaf_attention_modules = []
        sorted_names = sorted(attention_modules.keys())
        
        for name in sorted_names:
            is_parent = False
            for other_name in sorted_names:
                if other_name != name and other_name.startswith(name + "."):
                    is_parent = True
                    break
            if not is_parent:
                leaf_attention_modules.append((name, attention_modules[name]))
        
        for layer_idx, (name, module) in enumerate(leaf_attention_modules):
            hook = module.register_forward_hook(self._create_hook(layer_idx))
            self._hooks.append(hook)
        
        return len(self._hooks)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def accumulate(self, attn_weights: torch.Tensor, layer_idx: int):
        if not self.is_active:
            return
        
        if attn_weights.dim() != 4:
            raise ValueError(
                f"Expected attention weights to have 4 dimensions (bsz, num_heads, q_len, kv_seq_len), "
                f"got {attn_weights.dim()} dimensions with shape {attn_weights.shape} at layer {layer_idx}"
            )
        
        bsz, num_heads, q_len, kv_seq_len = attn_weights.shape
        
        if bsz != 1:
            raise ValueError(f"Expected batch size of 1, got {bsz} at layer {layer_idx}.")
        
        # Validate q_len matches expected
        # Note: num_heads can vary between layers in some architectures, and GQA models
        if q_len != self.expected_q_len:
            raise ValueError(
                f"Attention q_len ({q_len}) doesn't match expected_q_len ({self.expected_q_len}) "
                f"at layer {layer_idx}. "
                f"(total_seq_len={self.total_seq_len}, prefix_in_kv_cache={self.prefix_in_kv_cache})"
            )
        
        # Validate kv_seq_len covers the block we're trying to score
        expected_kv_len = self.prefix_in_kv_cache + self.block_token_count + self.query_len
        if kv_seq_len < expected_kv_len:
            raise ValueError(
                f"Attention kv_seq_len ({kv_seq_len}) is smaller than expected ({expected_kv_len}) "
                f"at layer {layer_idx}. Cannot extract block scores. "
                f"(prefix_in_kv_cache={self.prefix_in_kv_cache}, block_token_count={self.block_token_count}, "
                f"query_len={self.query_len})"
            )
        
        block_start_in_kv = self.prefix_in_kv_cache
        block_end_in_kv = self.prefix_in_kv_cache + self.block_token_count
        
        # Extract attention scores for block tokens from query positions
        # Sum over all heads (dim=1) and query positions (dim=2) to get per-token importance
        if self.query_len > 0:
            query_start_in_q = q_len - self.query_len
            layer_scores = attn_weights[:, :, query_start_in_q:, block_start_in_kv:block_end_in_kv].sum(dim=(1, 2))
        else:
            layer_scores = attn_weights[:, :, :, block_start_in_kv:block_end_in_kv].sum(dim=(1, 2))
        
        if self.accumulated_scores is None:
            self.accumulated_scores = layer_scores
        else:
            if self.accumulated_scores.shape[-1] != layer_scores.shape[-1]:
                raise RuntimeError(
                    f"Shape mismatch in attention score accumulation at layer {layer_idx}: "
                    f"accumulated shape {self.accumulated_scores.shape}, new layer shape {layer_scores.shape}"
                )
            self.accumulated_scores.add_(layer_scores)
            del layer_scores
        
        self.layer_count += 1
    
    def select_top_k(self, top_k: int) -> List[int]:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        
        if self.accumulated_scores is None:
            return []
        
        if self.layer_count == 0:
            raise RuntimeError("select_top_k called but no layers have been accumulated.")
        
        num_block_tokens = self.accumulated_scores.shape[-1]
        token_budget = min(num_block_tokens, top_k)
        
        _, top_k_indices = torch.topk(self.accumulated_scores[0], k=token_budget)
        top_k_indices_sorted, _ = torch.sort(top_k_indices)
        
        return top_k_indices_sorted.cpu().tolist()
    
    def select_top_k_with_exclusion(self, top_k: int, exclude_indices: List[int]) -> List[int]:
        if top_k <= 0:
            return []
        
        if self.accumulated_scores is None:
            return []
        
        if self.layer_count == 0:
            raise RuntimeError("select_top_k_with_exclusion called but no layers have been accumulated.")
        
        num_block_tokens = self.accumulated_scores.shape[-1]
        
        scores = self.accumulated_scores[0].clone()
        if exclude_indices:
            exclude_tensor = torch.tensor(exclude_indices, device=scores.device, dtype=torch.long)
            scores[exclude_tensor] = float('-inf')
        
        available_tokens = num_block_tokens - len(exclude_indices)
        token_budget = min(available_tokens, top_k)
        
        if token_budget <= 0:
            return []
        
        _, top_k_indices = torch.topk(scores, k=token_budget)
        top_k_indices_sorted, _ = torch.sort(top_k_indices)
        
        return top_k_indices_sorted.cpu().tolist()
    
    def finish_block(self) -> None:
        self.accumulated_scores = None
        self.layer_count = 0
        self.block_token_count = 0
        self.total_seq_len = 0
        self.prefix_len = 0
        self.query_len = 0
        self.expected_q_len = 0
        self.prefix_in_kv_cache = 0
        self.is_active = False
        # Note: hooks are NOT removed here - they persist across blocks


def get_or_create_accumulator(model) -> AttentionScoreAccumulator:
    if not hasattr(model, '_topk_accumulator'):
        model._topk_accumulator = AttentionScoreAccumulator()
    return model._topk_accumulator


# =============================================================================
# SEQUENTIAL TOP-K PROCESSOR
# =============================================================================

class SequentialTopKProcessor:
    """
    Implements sequential block processing with query-guided top-K selection.
    
    Pipeline:
        For each block:
          - Forward pass with [block] + [query], using accumulated KV cache as context
          - Block tokens use their ORIGINAL positions in the full context
          - Accumulate attention scores, select tokens
          - Extract KV cache for selected tokens
          - Append to accumulated cache
        
        Generation:
          - Query attends to accumulated KV cache
    """
    
    def __init__(
        self,
        model_path: str,
        top_k: int = 256,
        block_size: int = 2048,
        max_new_tokens: int = 100,
        stop_words: Optional[List[str]] = None,
        anchor_size: int = 64,
        local_window_size: int = 64,
    ):
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
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("  Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation='eager',
        )
        self.model.eval()
        
        # Register hooks once for attention score accumulation
        self.accumulator = get_or_create_accumulator(self.model)
        self.num_hooks = self.accumulator.register_hooks(self.model)
        print(f"  Registered {self.num_hooks} attention hooks")
        
        print("[SequentialTopKProcessor] Initialization complete.\n")
    
    def _tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(
            text, return_tensors='pt', add_special_tokens=False
        ).to(self.device)
    
    def _tokenize_query_with_chat_template(self, query_text: str) -> torch.Tensor:
        messages = [{"role": "user", "content": query_text}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
    
    def _split_into_blocks(self, token_ids: torch.Tensor) -> List[torch.Tensor]:
        return list(token_ids.split(self.block_size, dim=1))
    
    def _compute_anchor_local_indices(
        self, 
        block_len: int, 
        block_start_position: int,
    ) -> Tuple[List[int], List[int]]:
        anchor_indices = []
        
        if block_start_position < self.anchor_size:
            anchor_end_in_block = min(self.anchor_size - block_start_position, block_len)
            anchor_indices = list(range(anchor_end_in_block))
        
        anchor_end = len(anchor_indices)
        local_start = max(anchor_end, block_len - self.local_window_size)
        local_indices = list(range(local_start, block_len))
        
        return anchor_indices, local_indices
    
    def _extract_kv_for_indices(
        self,
        past_key_values: Tuple,
        indices: List[int],
        offset: int = 0,
    ) -> Tuple:
        """Extract KV cache entries for specific indices."""
        absolute_indices = [offset + idx for idx in indices]
        indices_tensor = torch.tensor(absolute_indices, device=self.device, dtype=torch.long)
        
        extracted_kv = []
        for layer_kv in past_key_values:
            key, value = layer_kv
            extracted_key = key.index_select(dim=2, index=indices_tensor)
            extracted_value = value.index_select(dim=2, index=indices_tensor)
            extracted_kv.append((extracted_key, extracted_value))
        
        return tuple(extracted_kv)
    
    def _merge_kv_caches(
        self,
        cache1: Optional[Tuple],
        cache2: Tuple,
    ) -> Tuple:
        if cache1 is None:
            return cache2
        
        merged_kv = []
        for (k1, v1), (k2, v2) in zip(cache1, cache2):
            merged_key = torch.cat([k1, k2], dim=2)
            merged_value = torch.cat([v1, v2], dim=2)
            merged_kv.append((merged_key, merged_value))
        
        return tuple(merged_kv)
    
    def cleanup(self):
        """Remove registered hooks and free resources."""
        if hasattr(self, 'accumulator'):
            self.accumulator.remove_hooks()
            print("Attention hooks removed.")
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.cleanup()
    
    @torch.no_grad()
    def _process_block(
        self,
        prefix_kv_cache: Optional[Tuple],
        block_ids: torch.Tensor,
        query_ids: torch.Tensor,
        block_start_position: int,
        total_context_length: int,
    ) -> Tuple[torch.Tensor, List[int], Tuple]:
        """
        Process a block: select tokens and extract their KV cache.
        
        Args:
            prefix_kv_cache: KV cache from previous blocks' summaries
            block_ids: Token IDs of the current block
            query_ids: Query token IDs  
            block_start_position: Starting position of this block in full context
            total_context_length: Total length of the original context
        
        Returns:
            Tuple of (summary_ids, selected_indices, kv_cache)
        """
        if block_ids.dim() != 2 or block_ids.shape[0] != 1:
            raise ValueError(f"block_ids must have shape (1, seq_len), got {block_ids.shape}")
        
        block_len = block_ids.shape[1]
        query_len = query_ids.shape[1]
        
        # Compute anchor and local window indices
        anchor_indices, local_indices = self._compute_anchor_local_indices(
            block_len, block_start_position
        )
        fixed_indices = sorted(list(set(anchor_indices) | set(local_indices)))
        
        # Build input: [block] + [query]
        input_ids = torch.cat([block_ids, query_ids], dim=1)
        total_input_len = input_ids.shape[1]
        
        # Position IDs: block tokens get their ORIGINAL positions
        # Query tokens get positions after the full context
        block_positions = torch.arange(
            block_start_position,
            block_start_position + block_len,
            device=self.device
        )
        query_positions = torch.arange(
            total_context_length,
            total_context_length + query_len,
            device=self.device
        )
        position_ids = torch.cat([block_positions, query_positions]).unsqueeze(0)
        
        # Convert prefix KV cache to DynamicCache if provided
        past_key_values = None
        prefix_cache_len = 0
        if prefix_kv_cache is not None:
            past_key_values = DynamicCache.from_legacy_cache(prefix_kv_cache)
            prefix_cache_len = prefix_kv_cache[0][0].shape[2]
        
        # Start accumulation for Top-K selection (hooks already registered)
        self.accumulator.start_block_with_prefix(
            total_seq_len=total_input_len,
            prefix_len=0,
            block_len=block_len,
            query_len=query_len,
            prefix_in_kv_cache=prefix_cache_len,
        )
        
        # Forward pass - hooks capture attention weights automatically
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True,
        )
        
        new_kv_cache = outputs.past_key_values
        
        if hasattr(new_kv_cache, 'to_legacy_cache'):
            new_kv_cache = new_kv_cache.to_legacy_cache()
        
        del outputs
        
        # Select top-K (excluding anchor and local)
        topk_indices = self.accumulator.select_top_k_with_exclusion(self.top_k, fixed_indices)
        self.accumulator.finish_block()
        
        # Combine all selected indices
        all_selected_indices = sorted(list(set(fixed_indices) | set(topk_indices)))
        
        # Validate indices are within block bounds (not selecting query tokens)
        all_selected_indices = [idx for idx in all_selected_indices if 0 <= idx < block_len]
        
        if not all_selected_indices:
            raise RuntimeError(f"No tokens selected for block with {block_len} tokens.")
        
        # Extract token IDs
        indices_tensor = torch.tensor(all_selected_indices, device=self.device, dtype=torch.long)
        summary_ids = block_ids.index_select(dim=1, index=indices_tensor)
        
        # Extract KV cache for selected block tokens
        extracted_kv = self._extract_kv_for_indices(
            new_kv_cache,
            all_selected_indices,
            offset=prefix_cache_len,
        )
        
        del new_kv_cache, indices_tensor
        torch.cuda.empty_cache()
        
        return summary_ids, all_selected_indices, extracted_kv
    
    @torch.no_grad()
    def _generate(
        self,
        query_ids: torch.Tensor,
        kv_cache: Tuple,
        total_context_length: int,
    ) -> torch.Tensor:
        """
        Generate response using accumulated KV cache.
        """
        query_len = query_ids.shape[1]
        
        # Query positions continue from end of original context
        position_ids = torch.arange(
            total_context_length,
            total_context_length + query_len,
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
        next_position = total_context_length + query_len
        
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
        text = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
        for stop_word in self.stop_words:
            text = text.split(stop_word)[0]
        return text.strip()
    
    def __call__(
        self,
        prompt_context: str,
        prompt_query: str,
    ) -> Dict[str, List[str]]:
        print("=" * 60)
        print("Sequential Top-K Processing Pipeline")
        print("=" * 60)
        
        context_ids = self._tokenize(prompt_context)
        query_ids = self._tokenize_query_with_chat_template(prompt_query)
        
        total_context_length = context_ids.shape[1]
        
        print(f"\nContext length: {total_context_length} tokens")
        print(f"Query length: {query_ids.shape[1]} tokens (with chat template)")
        
        blocks = self._split_into_blocks(context_ids)
        print(f"Split into {len(blocks)} blocks of ~{self.block_size} tokens each")
        print(f"Cache strategy: Global Anchor({self.anchor_size}) + Local({self.local_window_size}/block) + Top-K({self.top_k}/block)")
        
        print(f"\n--- Processing Blocks ---")
        
        accumulated_kv_cache = None
        block_start_position = 0
        
        for i, block in enumerate(blocks):
            summary_ids, selected_indices, block_kv = self._process_block(
                prefix_kv_cache=accumulated_kv_cache,
                block_ids=block,
                query_ids=query_ids,
                block_start_position=block_start_position,
                total_context_length=total_context_length,
            )
            
            accumulated_kv_cache = self._merge_kv_caches(accumulated_kv_cache, block_kv)
            cache_len = accumulated_kv_cache[0][0].shape[2]
            
            anchor_info = " (includes global anchor)" if block_start_position < self.anchor_size else ""
            print(f"  Block {i+1}: {block.shape[1]} tokens → Selected: {len(selected_indices)} → Cache: {cache_len}{anchor_info}")
            
            block_start_position += block.shape[1]
            del block_kv
        
        if accumulated_kv_cache is None:
            raise RuntimeError("No KV cache accumulated.")
        
        final_cache_len = accumulated_kv_cache[0][0].shape[2]
        print(f"\n  Final KV cache: {final_cache_len} tokens")

        del blocks, context_ids
        torch.cuda.empty_cache()

        print(f"\n--- Generation ---")
        print(f"  Query positions: {total_context_length} to {total_context_length + query_ids.shape[1] - 1}")
        
        generated_ids = self._generate(query_ids, accumulated_kv_cache, total_context_length)
        generated_text = self._get_output_text(generated_ids)
        
        print(f"\nGenerated {generated_ids.shape[1]} tokens")
        print("=" * 60)
        
        return {'text': [generated_text]}
