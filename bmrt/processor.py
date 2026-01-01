from typing import Optional, Tuple, List, Dict
import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoTokenizer, AutoModelForCausalLM

from .selectors.base import BaseSelector
from .selectors.eager_exact import ExactSelector
from .selectors.lsh_core import LSHSelector
from .selectors.hybrid import HybridSelector
from .utils import compute_anchor_local_indices, extract_kv_for_indices, merge_kv_caches

class RecursiveCompressionEngine:
    """
    Manages the recursive compression loop:
    State_N = Compress(State_N-1 + Chunk_N)
    
    Implemented specs:
    - Dynamic Budgeting via protection_divisor
    - Hardware agnostic selection loop
    """
    
    def __init__(
        self,
        model_path: str,
        selector_type: str = 'exact', # 'exact', 'lsh', or 'hybrid'
        lsh_mode: str = 'frequency_rank', # 'frequency_rank' or 'magicpig_baseline'
        selector_mode: str = 'l2', # tie-breaker: 'l2','max_sim','mahalanobis','partitioned_centroid','none'
        compression_mode: str = 'accumulate', # 'accumulate' or 'recursive'
        backend: str = 'eager', # 'eager' or 'flash'
        budget: int = 4096, # Total budget
        protection_divisor: int = 4, # n
        block_size: int = 4096,
        max_new_tokens: int = 100,
        stop_words: Optional[List[str]] = None,
        # LSH Config
        num_bits: int = 12,
        num_tables: int = 20,
        # Hybrid Config
        hybrid_primary: str = 'exact',
        hybrid_secondary: str = 'lsh',
        hybrid_ratio: float = 0.5,
    ):
        
        self.model_path = model_path
        self.selector_type = selector_type
        self.lsh_mode = lsh_mode
        self.selector_mode = selector_mode
        self.compression_mode = compression_mode
        self.backend = backend
        self.budget = budget
        self.protection_divisor = protection_divisor
        self.block_size = block_size
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words or []
        self.num_bits = num_bits
        self.num_tables = num_tables
        
        # Valid modes
        if compression_mode not in ['accumulate', 'recursive']:
            raise ValueError(f"Invalid compression_mode '{compression_mode}'. Must be 'accumulate' or 'recursive'.")
        
        # Dynamic Budget Calculation
        self.anchor_size = budget // protection_divisor
        self.local_window_size = budget // protection_divisor
        self.global_budget = budget - (self.anchor_size + self.local_window_size)
        
        if self.global_budget <= 0:
            raise ValueError(
                 f"Invalid protection_divisor {protection_divisor} for budget {budget}. "
                 f"Anchor({self.anchor_size}) + Window({self.local_window_size}) consumes all budget."
            )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Bookkeeping for the previous block's local tail (not persisted into accumulated cache
        # except the final tail which will be merged after processing all blocks)
        self.prev_local_tail_kv: Optional[Tuple] = None
        self.prev_local_tail_len: int = 0
        
        print(f"\n[RecursiveCompressionEngine] Initializing...")
        print(f"  Model: {model_path}")
        print(f"  Backend: {backend}")
        print(f"  Budget: {budget} (Divisor={protection_divisor})")
        print(f"  -> Allocation: Anchor={self.anchor_size} | Local={self.local_window_size} | Global={self.global_budget}")
        print(f"  Selector: {selector_type} (Mode: {lsh_mode})")
        print(f"  Selector tie-breaker mode: {self.selector_mode}")
        print(f"  Compression Mode: {compression_mode}")
        
        self.validate_config()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize Model
        print("  Loading model...")
        attn_impl = 'eager'
        if backend == 'flash':
             attn_impl = 'flash_attention_2'

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
        self.model.eval()

        # Initialize Selector
        if selector_type == 'hybrid':
            print(f"  Hybrid: {hybrid_primary} ({hybrid_ratio}) + {hybrid_secondary} ({1-hybrid_ratio})")
            primary = self._create_selector(hybrid_primary)
            secondary = self._create_selector(hybrid_secondary)
            self.selector = HybridSelector(primary, secondary, primary_ratio=hybrid_ratio)
        else:
            self.selector = self._create_selector(selector_type)
            
        self.selector.setup(self.model)
        
    def _create_selector(self, s_type: str):
        if s_type == 'exact':
            return ExactSelector()
        elif s_type == 'lsh':
            config = self.model.config
            head_dim = config.hidden_size // config.num_attention_heads
            return LSHSelector(
                head_dim=head_dim, 
                lsh_mode=self.lsh_mode, 
                num_bits=self.num_bits,
                num_tables=self.num_tables,
                device=self.device,
                mode=self.selector_mode
            )
        else:
            raise ValueError(f"Unknown selector type: {s_type}")
        
    def validate_config(self):
        """Enforce hardware compatibility matrix."""
        # Option 1 (Exact) + Flash -> ERROR
        if self.selector_type == 'exact' and self.backend == 'flash':
             raise ValueError("Configuration Error: ExactSelector (Option 1) requires Eager Attention. Flash Attention does not support chunked score access.")
        
        # Check protection divisor
        if self.protection_divisor <= 1:
             raise ValueError("protection_divisor must be > 1 to allow for global tokens.")

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
    
    def cleanup(self):
        if hasattr(self, 'selector'):
            self.selector.cleanup()

    def __del__(self):
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
        """
        block_len = block_ids.shape[1]
        query_len = query_ids.shape[1]
        
        anchor_indices, local_indices = compute_anchor_local_indices(
            block_len, 
            block_start_position, 
            self.anchor_size, 
            self.local_window_size
        )
        
        fixed_indices = sorted(list(set(anchor_indices) | set(local_indices)))
        input_ids = torch.cat([block_ids, query_ids], dim=1)
        total_input_len = input_ids.shape[1]
        
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
        
        # Prepare Cache
        past_key_values = None
        prefix_cache_len = 0
        if prefix_kv_cache is not None:
            prefix_cache_len = prefix_kv_cache[0][0].shape[2]

        # If we have a previous local tail (accumulate mode), temporarily
        # merge it into the prefix cache for scoring/selection only. We do
        # NOT persist this merge into the running `accumulated_kv_cache`.
        temp_prefix_kv = prefix_kv_cache
        temp_prefix_cache_len = prefix_cache_len
        if self.compression_mode == 'accumulate' and self.prev_local_tail_kv is not None and self.prev_local_tail_len > 0:
            temp_prefix_kv = merge_kv_caches(prefix_kv_cache, self.prev_local_tail_kv)
            temp_prefix_cache_len = prefix_cache_len + self.prev_local_tail_len

        if temp_prefix_kv is not None:
            past_key_values = DynamicCache.from_legacy_cache(temp_prefix_kv)
        
        # Prepare Selector (for eager exact accumulator)
        score_history = (self.compression_mode == 'recursive')
        prev_tail_len = 0 if score_history else self.prev_local_tail_len
        if hasattr(self.selector, 'prepare_block'):
            self.selector.prepare_block(
                total_seq_len=total_input_len,
                prefix_len=0,
                block_len=block_len,
                query_len=query_len,
                prefix_in_kv=temp_prefix_cache_len,
                score_history=score_history,
                prev_local_tail_len=prev_tail_len
            )

        from .patch_fa import ApplyFlashAttentionPatch
        with ApplyFlashAttentionPatch():
             outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        
        new_kv_cache = outputs.past_key_values
        if hasattr(new_kv_cache, 'to_legacy_cache'):
            new_kv_cache = new_kv_cache.to_legacy_cache()
        del outputs
        
        # Selection Phase
        candidate_indices_set = set(range(block_len)) - set(fixed_indices)
        candidate_indices = sorted(list(candidate_indices_set))

        selected_indices_only = []
        if candidate_indices and self.global_budget > 0:
            # Prepare data for selection
            last_layer_key = new_kv_cache[-1][0]

            # Using temp_prefix_cache_len because temp_prefix_kv may include prev tail
            cache_block_start = temp_prefix_cache_len
            cache_query_start = temp_prefix_cache_len + block_len

            # Build absolute candidate indices list
            if self.compression_mode == 'recursive':
                # For recursive, use history logic (absolute indices)
                # Include all non-anchor tokens from history as candidates for re-evaluation
                history_candidates = list(range(self.anchor_size, prefix_cache_len))
                cand_abs_indices = history_candidates + [prefix_cache_len + idx for idx in candidate_indices]
            else:
                cand_abs_indices = []
                # Include previous local tail absolute indices if present
                if self.compression_mode == 'accumulate' and self.prev_local_tail_kv is not None and self.prev_local_tail_len > 0:
                    cand_abs_indices.extend(list(range(prefix_cache_len, prefix_cache_len + self.prev_local_tail_len)))
                # Current-block candidate absolutes (block comes after temp prefix)
                cand_abs_indices.extend([cache_block_start + idx for idx in candidate_indices])

            cand_tensor_indices = torch.tensor(cand_abs_indices, device=self.device, dtype=torch.long)

            # Candidate vectors (Mean over heads)
            candidate_vectors = last_layer_key.index_select(dim=2, index=cand_tensor_indices)[0].mean(dim=0)

            # Query vectors
            query_vectors = last_layer_key[:, :, cache_query_start:, :][0].mean(dim=0)

            selected_indices_only = self.selector.select(
                query_ids=query_ids,
                query_vectors=query_vectors,
                candidate_vectors=candidate_vectors,
                candidate_indices=cand_abs_indices,
                budget=self.global_budget
            )

        if hasattr(self.selector, 'finish_block'):
            self.selector.finish_block()

        # 8. Update Phase
        # After selection, we need to handle mapping selected indices back to
        # either previous-tail entries or current-block entries. For accumulate
        # mode we may have prev-selected items stored separately.
        if self.compression_mode == 'recursive':
             # Reconstruct absolute indices to keep:
             # 1. Global Anchor
             # 2. Local Window (current block, strictly preserved)
             # 3. Selected Global Candidates
             
             # A. Global Anchor
             global_anchor_end = min(self.anchor_size, prefix_cache_len + block_len)
             global_anchor_indices = list(range(global_anchor_end))
             
             # B. Local Window (Convert current block's relative fixed indices to absolute)
             # Note: fixed_indices includes the block's own anchor/window regions.
             current_block_fixed_absolute = [prefix_cache_len + idx for idx in fixed_indices]
             
             # C. Candidates (Already Absolute)
             
             # Merge and Sort
             final_indices_set = set(global_anchor_indices) | set(current_block_fixed_absolute) | set(selected_indices_only)
             all_selected_indices_absolute = sorted(list(final_indices_set))
            
             extraction_indices = all_selected_indices_absolute
             extraction_offset = 0 # Offset 0 because indices are absolute into the full cache
             
             # Re-map for summary_ids (visualization/debugging)
             # Extract only those indices that correspond to the current block (i.e., >= prefix_len)
             kept_relative_indices = [idx - prefix_cache_len for idx in all_selected_indices_absolute if idx >= prefix_cache_len]
             summary_ids = block_ids.index_select(dim=1, index=torch.tensor(kept_relative_indices, device=self.device))
             
             # Extract KV for the selected absolute indices from new_kv_cache
             extracted_kv = extract_kv_for_indices(
                 new_kv_cache,
                 extraction_indices,
                 offset=extraction_offset,
                 device=self.device
             )
             
        else:
            # Standard Accumulate behavior
            # selected_indices_only contains absolute indices into new_kv_cache
            cache_block_start = temp_prefix_cache_len

            # Partition selected absolutes into prev-tail (from previous tail) and current-block selections
            prev_selected_abs = []
            curr_selected_abs = []
            if self.compression_mode == 'accumulate' and self.prev_local_tail_kv is not None and self.prev_local_tail_len > 0:
                prev_range_start = prefix_cache_len
                prev_range_end = prefix_cache_len + self.prev_local_tail_len
                for si in selected_indices_only:
                    if prev_range_start <= si < prev_range_end:
                        prev_selected_abs.append(int(si))
                    elif si >= cache_block_start:
                        curr_selected_abs.append(int(si))
            else:
                for si in selected_indices_only:
                    if si >= cache_block_start:
                        curr_selected_abs.append(int(si))

            # Current-block kept indices: ANCHORS (not local tail) U selected-from-current-block
            curr_selected_relative = [i - cache_block_start for i in curr_selected_abs]
            anchor_relative = anchor_indices
            all_selected_relative = sorted(list(set(anchor_relative) | set(curr_selected_relative)))
            all_selected_relative = [idx for idx in all_selected_relative if 0 <= idx < block_len]
            summary_ids = block_ids.index_select(dim=1, index=torch.tensor(all_selected_relative, device=self.device))

            # Extract KV from two sources:
            # 1. prev_selected_abs must be extracted from self.prev_local_tail_kv (not new_kv_cache)
            # 2. current block indices are extracted from new_kv_cache
            if prev_selected_abs:
                # Convert prev_selected_abs (absolute into temp_prefix_kv) to relative indices into prev_local_tail_kv
                prev_selected_relative_tail = [si - prefix_cache_len for si in prev_selected_abs]
                prev_extracted_kv = extract_kv_for_indices(
                    self.prev_local_tail_kv,
                    prev_selected_relative_tail,
                    offset=0,
                    device=self.device
                )
            else:
                prev_extracted_kv = None

            extraction_abs_current = [cache_block_start + idx for idx in all_selected_relative]
            extraction_indices = extraction_abs_current  # Store for return value
            
            # Extract current block KV from new_kv_cache
            curr_extracted_kv = extract_kv_for_indices(
                new_kv_cache,
                extraction_abs_current,
                offset=0,
                device=self.device
            )
            
            # Merge prev-selected and current-block extracted KVs
            extracted_kv = merge_kv_caches(prev_extracted_kv, curr_extracted_kv)

        # Store current block's local tail KV for use by the next block's selection.
        # We do NOT merge this tail into the accumulated cache now; it will only be
        # merged after processing the final block.
        if len(local_indices) > 0:
            # local_indices are relative to the block; convert to absolute positions
            cache_block_start = temp_prefix_cache_len
            local_abs_indices = [cache_block_start + idx for idx in local_indices]
            self.prev_local_tail_kv = extract_kv_for_indices(
                new_kv_cache,
                local_abs_indices,
                offset=0,
                device=self.device
            )
            self.prev_local_tail_len = len(local_indices)
        else:
            self.prev_local_tail_kv = None
            self.prev_local_tail_len = 0
        
        # Return something meaningful for `selected_indices`?
        # The return value `all_selected_indices` in signature is usually used for debugging.
        # Let's return the final list.
        if self.compression_mode == 'recursive':
             ret_indices = extraction_indices
        else:
             ret_indices = extraction_indices 
             
        del new_kv_cache
        torch.cuda.empty_cache()
        
        return summary_ids, ret_indices, extracted_kv

    @torch.no_grad()
    def _generate(self, query_ids, kv_cache, total_context_length) -> torch.Tensor:
        query_len = query_ids.shape[1]
        position_ids = torch.arange(total_context_length, total_context_length + query_len, device=self.device).unsqueeze(0)
        
        past_key_values = DynamicCache.from_legacy_cache(kv_cache)
        
        outputs = self.model(input_ids=query_ids, position_ids=position_ids, past_key_values=past_key_values, use_cache=True)
        current_cache = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated_tokens = [next_token]
        
        next_position = total_context_length + query_len
        for _ in range(self.max_new_tokens - 1):
            outputs = self.model(input_ids=next_token, position_ids=torch.tensor([[next_position]], device=self.device), past_key_values=current_cache, use_cache=True)
            current_cache = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated_tokens.append(next_token)
            next_position += 1
            if next_token.item() == self.tokenizer.eos_token_id: break
            
        return torch.cat(generated_tokens, dim=1)

    def _get_output_text(self, token_ids) -> str:
        text = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
        for stop_word in self.stop_words:
            text = text.split(stop_word)[0]
        return text.strip()

    def __call__(self, prompt_context: str, prompt_query: str) -> Dict[str, List[str]]:
        context_ids = self._tokenize(prompt_context)
        query_ids = self._tokenize_query_with_chat_template(prompt_query)
        blocks = self._split_into_blocks(context_ids)
        
        print(f"Split into {len(blocks)} blocks. Processing...")
        
        accumulated_kv_cache = None
        block_start_position = 0
        total_context_length = context_ids.shape[1]
        
        for i, block in enumerate(blocks):
            _, selected_indices, block_kv = self._process_block(
                prefix_kv_cache=accumulated_kv_cache,
                block_ids=block,
                query_ids=query_ids,
                block_start_position=block_start_position,
                total_context_length=total_context_length
            )
            if self.compression_mode == 'recursive':
                accumulated_kv_cache = block_kv
            else:
                accumulated_kv_cache = merge_kv_caches(accumulated_kv_cache, block_kv)
            block_start_position += block.shape[1]
            print(f"  Block {i+1} processed. Cache size: {accumulated_kv_cache[0][0].shape[2]}")
        # After processing all blocks, merge the final local tail (if any) into the
        # accumulated cache so the final window is preserved for generation.
        if self.compression_mode == 'accumulate' and self.prev_local_tail_kv is not None:
            accumulated_kv_cache = merge_kv_caches(accumulated_kv_cache, self.prev_local_tail_kv)

        generated_ids = self._generate(query_ids, accumulated_kv_cache, total_context_length)
        return {'text': [self._get_output_text(generated_ids)]}
