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
        compression_mode: str = 'accumulate', # 'accumulate' or 'recursive'
        backend: str = 'eager', # 'eager' or 'flash'
        budget: int = 4096, # Total budget
        protection_divisor: int = 4, # n
        block_size: int = 4096,
        max_new_tokens: int = 100,
        stop_words: Optional[List[str]] = None,
        # Hybrid Config
        hybrid_primary: str = 'exact',
        hybrid_secondary: str = 'lsh',
        hybrid_ratio: float = 0.5,
    ):
        
        self.model_path = model_path
        self.selector_type = selector_type
        self.lsh_mode = lsh_mode
        self.compression_mode = compression_mode
        self.backend = backend
        self.budget = budget
        self.protection_divisor = protection_divisor
        self.block_size = block_size
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words or []
        
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
        
        print(f"\n[RecursiveCompressionEngine] Initializing...")
        print(f"  Model: {model_path}")
        print(f"  Backend: {backend}")
        print(f"  Budget: {budget} (Divisor={protection_divisor})")
        print(f"  -> Allocation: Anchor={self.anchor_size} | Local={self.local_window_size} | Global={self.global_budget}")
        print(f"  Selector: {selector_type} (Mode: {lsh_mode})")
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
                device=self.device
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
            past_key_values = DynamicCache.from_legacy_cache(prefix_kv_cache)
            prefix_cache_len = prefix_kv_cache[0][0].shape[2]
        
        # Prepare Selector (for eager exact accumulator)
        score_history = (self.compression_mode == 'recursive')
        if hasattr(self.selector, 'prepare_block'):
            self.selector.prepare_block(
                total_seq_len=total_input_len,
                prefix_len=0,
                block_len=block_len,
                query_len=query_len,
                prefix_in_kv=prefix_cache_len,
                score_history=score_history
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
        
        # In Recursive Mode, we also consider history candidates!
        if self.compression_mode == 'recursive' and self.global_budget > 0:
             # Transform current block candidates to absolute
             current_block_candidates = [prefix_cache_len + idx for idx in candidate_indices]
             
             # History Range: [0, prefix_cache_len)
             history_candidates = []
             if prefix_cache_len > 0:
                 start_cand = min(self.anchor_size, prefix_cache_len)
                 history_candidates = list(range(start_cand, prefix_cache_len))

             pool_candidates_absolute = history_candidates + current_block_candidates
             candidate_indices = sorted(pool_candidates_absolute)
        
        selected_indices_only = []
        if candidate_indices and self.global_budget > 0:
            # Prepare data for selection
            last_layer_key = new_kv_cache[-1][0] 
            
            cache_block_start = prefix_cache_len
            cache_query_start = prefix_cache_len + block_len
            
            # Candidate vectors (Mean over heads)
            if self.compression_mode == 'recursive':
                 cand_tensor_indices = torch.tensor(candidate_indices, device=self.device, dtype=torch.long)
            else:
                 cand_abs_indices = [cache_block_start + idx for idx in candidate_indices]
                 cand_tensor_indices = torch.tensor(cand_abs_indices, device=self.device, dtype=torch.long)
            
            candidate_vectors = last_layer_key.index_select(dim=2, index=cand_tensor_indices)[0].mean(dim=0)
            
            # Query vectors
            query_vectors = last_layer_key[:, :, cache_query_start:, :][0].mean(dim=0)
            
            selected_indices_only = self.selector.select(
                query_ids=query_ids,
                query_vectors=query_vectors,
                candidate_vectors=candidate_vectors,
                candidate_indices=candidate_indices,
                budget=self.global_budget
            )

        if hasattr(self.selector, 'finish_block'):
            self.selector.finish_block()
        
        # 8. Update Phase
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
             
        else:
            # Standard Accumulate behavior
            all_selected_indices = sorted(list(set(fixed_indices) | set(selected_indices_only)))
            all_selected_indices = [idx for idx in all_selected_indices if 0 <= idx < block_len] # Safety
            summary_ids = block_ids.index_select(dim=1, index=torch.tensor(all_selected_indices, device=self.device))
            
            extraction_indices = all_selected_indices
            extraction_offset = prefix_cache_len
        
        extracted_kv = extract_kv_for_indices(
            new_kv_cache,
            extraction_indices,
            offset=extraction_offset,
            device=self.device
        )
        
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
            
        generated_ids = self._generate(query_ids, accumulated_kv_cache, total_context_length)
        return {'text': [self._get_output_text(generated_ids)]}
