# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Dict, List, Optional, Tuple, Set

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

# --- NEW IMPORTS for Tidal-Prefill ---
# We import our custom 'load' function from the src package
# This load function will return the modified LlamaForCausalLM
# that is capable of sampling tokens during prefill.
try:
    from src.utils import load as load_tidal_model
    from src.models.llama_tidaldecoding import LlamaForCausalLM as TidalLlamaForCausalLM
except ImportError:
    print("Warning: Could not import tidal model components from 'src' package.")
    print("Please ensure 'src' is in your PYTHONPATH and contains:")
    print("- utils.py")
    print("- enable_tidal.py")
    print("- models/llama_tidaldecoding.py")
    print("- tidal_build/modify_llama_lim.py")
    # Define a placeholder if imports fail to avoid crashing
    load_tidal_model = None
    TidalLlamaForCausalLM = None

# --- ORIGINAL IMPORT for RingAttentionModel ---
# This is the LlamaForCausalLM from the Star Attention repo
# We rename it to avoid conflicts.
try:
    from star_attention import LlamaForCausalLM as StarLlamaForCausalLM
except ImportError:
    print("Warning: Could not import StarLlamaForCausalLM from 'star_attention' package.")
    StarLlamaForCausalLM = None


class DistributedInferenceBaseModel:
    """
    Base class for distributed models (RingAttentionModel).
    This uses the original star_attention.LlamaForCausalLM.
    """
    def __init__(
        self,
        path: str,
        max_new_tokens: int,
        stop_words: Optional[List[str]] = None,
        block_size: int = -1,
        anchor_block_size: int = -1,
    ):
        if StarLlamaForCausalLM is None:
            raise ImportError("star_attention.LlamaForCausalLM not found.")
            
        self._init_distributed()

        # Setup the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        # Define the model
        self.model = StarLlamaForCausalLM.from_pretrained(
            path,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            max_memory=self.max_memory,
            attn_implementation='flash_attention_2',
        )
        self.block_size = block_size if block_size > 0 else None
        self.anchor_block_size = anchor_block_size if anchor_block_size > 0 else None

        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words if stop_words else []

    def _init_distributed(self):
        """Initialize the distributed environment"""

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))

            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))

            # Assign each rank, its own set of GPUs
            # This is done so that the sharded model for each rank can be loaded on separate GPUs
            num_devices_per_rank = torch.cuda.device_count() // self.local_world_size
            device_id_start = self.local_rank * num_devices_per_rank
            self.max_memory = {
                x: f'{round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))}GB'
                for x in range(device_id_start, device_id_start + num_devices_per_rank)
            }
            print(
                '[model._init_distributed] '
                f'World size: {self.world_size}, Rank: {self.rank}, '
                f'Local World Size: {self.local_world_size}, Local rank: {self.local_rank}, '
                f'GPUs Assigned: {", ".join([str(x) for x in self.max_memory.keys()])}'
            )
        else:
            raise RuntimeError('Distributed environment is not initialized!')

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize the input text and return the token ids

        Args:
            text: input text

        Returns:
            token ids of shape (1, seq_len)
        """
        return self.tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(self.model.device)

    def _tokenize_and_partition_context(self, ctx: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Split the input context into blocks. The last block is padded to keep each block the same size."""
        raise NotImplementedError

    def _process_blockwise_context(
        self, ctx_ids_blocks: Tuple[torch.Tensor, ...], position_ids_blocks: Tuple[torch.Tensor, ...]
    ):
        """Generate the KV cache for the context assigned to the current rank."""
        raise NotImplementedError

    def _generate_output(self, input_ids, position_ids, past_key_values):
        """Phase 2 of Star Attention: Process input tokens followed by autoregressive token generation."""
        output_seq = None
        for _ in range(self.max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    enable_star_attn=True, # This flag is specific to StarLlamaForCausalLM
                )  # type: ignore

            # Assign the new updated KV-cache to the last rank
            if self.rank == self.world_size - 1:
                past_key_values = outputs.past_key_values

            # Get the next token
            next_token_logits = outputs.logits
# [Continuing from Part 1]
            # ... (inside DistributedInferenceBaseModel._generate_output)

            # Get the next token
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            output_seq = next_tokens if output_seq is None else torch.cat([output_seq, next_tokens])

            # Update the input_ids and position_ids for the next iteration
            input_ids = next_tokens.unsqueeze(0)
            position_ids = torch.tensor([[position_ids[-1, -1] + 1]]).to(position_ids)

        return output_seq.unsqueeze(0)

    def _get_output_text(self, output, truncate_texts=[]):
        """Convert the generated token ids to text"""
        generated_text = self.tokenizer.decode(output[0].detach().cpu().numpy().tolist())

        # Remove the input from the generated text
        for t in truncate_texts:
            t = t.strip()
            if t and generated_text.startswith(t):
                generated_text = generated_text[len(t) :].strip()

        for s in self.stop_words:
            generated_text = generated_text.split(s)[0]

        return generated_text.strip()

    def __call__(self, prompt_context: str, prompt_query: str):
        raise NotImplementedError


# [ Original RingAttentionModel class is skipped as it is not being modified ]
# ...

# [ Original DenseAttentionModel class is skipped as it is not being modified ]
# ...


class StarAttentionModel(DistributedInferenceBaseModel):
    """
    MODIFIED Star Attention
    Implements sequential Tidal-Prefill (Phase 1)
    and distributed Star-Attention generation (Phase 2).
    
    This class inherits from DistributedInferenceBaseModel to get:
    - self._init_distributed()
    - self._generate_output() (for Phase 2)
    - self._get_output_text()
    """

    def __init__(
        self,
        path: str,
        max_new_tokens: int,
        stop_words: Optional[List[str]] = None,
        # New args for Tidal-Prefill
        top_k: int = 256,
        selection_layers: List[int] = [2, 14],
        num_blocks: int = 4,
        # Original Star-Attn args (block_size is now for prefill)
        block_size: int = -1,
        anchor_block_size: int = -1, # This is no longer used by prefill
    ):
        # --- THIS IS THE CRITICAL CHANGE ---
        # We DO NOT call super().__init__() because we are
        # loading and patching the model in a specific order.
        
        # 1. Init distributed environment
        self._init_distributed()

        # 2. Store params
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words if stop_words else []
        self.top_k = top_k
        self.selection_layers = selection_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        
        # 3. Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # 4. Load the BASE distributed model
        #    (from star_attention/modeling_llama.py)
        if StarLlamaForCausalLM is None:
             raise ImportError("star_attention.LlamaForCausalLM not found.")
             
        self.model = StarLlamaForCausalLM.from_pretrained(
            path,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            max_memory=self.max_memory,
            attn_implementation='flash_attention_2',
        )

        # 5. NOW, patch this model with the Tidal logic
        try:
            # Assuming 'model.py' is in the root, and 'src' is a folder
            from src.enable_tidal import enable_tidal
            # We also need the LlamaForCausalLM from src to check
            # if the 'set_tokenizer_for_decode' method exists.
            from src.models.llama_tidaldecoding import LlamaForCausalLM as TidalLlama
        except ImportError:
            raise ImportError(
                "Could not import from 'src' package."
                " Please ensure 'src' is in your PYTHONPATH."
            )

        print("Applying Tidal patches to StarLlama model...")
        enable_tidal(
            self.model,
            attn_type="tidal",
            top_k=self.top_k,
            selection_layers=self.selection_layers,
            sparse_layer_start=min(self.selection_layers) if self.selection_layers else 0,
            correction_layer=max(self.selection_layers) if self.selection_layers else 0,
        )
        
        # 6. Set tokenizer for history tracking
        #    The patching in enable_tidal *replaces* the model's forward
        #    but the model object itself is still a StarLlamaForCausalLM.
        #    We need to check if *that* model has the history-tracking methods
        #    from `llama_tidaldecoding.py`.
        #    Since it *doesn't*, we must add them.
        #    Let's borrow the methods from the TidalLlama class.
        if not hasattr(self.model, "set_tokenizer_for_decode"):
            print("Manually adding history-tracking methods to model...")
            self.model.set_tokenizer_for_decode = TidalLlama.set_tokenizer_for_decode.__get__(self.model)
            self.model.model.set_sequence_context = TidalLlama.model.set_sequence_context.__get__(self.model.model)
            self.model.model.get_top_tokens_history = TidalLlama.model.get_top_tokens_history.__get__(self.model.model)
            self.model.model.clear_top_tokens_history = TidalLlama.model.clear_top_tokens_history.__get__(self.model.model)
            self.model.model._generate_sequence_id = TidalLlama.model._generate_sequence_id.__get__(self.model.model)
            
            # Init the history dict on the inner model
            self.model.model.global_top_tokens_history = {}
            self.model.model.current_sequence_id = None
            self.model.model.current_input_text = None
            self.model.model._tokenizer_for_decode = None

        self.model.set_tokenizer_for_decode(self.tokenizer)


    def _tokenize_and_partition_context(self, ctx: str) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        """
        Tokenizes and splits the context into N blocks.
        If block_size > 0, it's used. Otherwise, num_blocks is used.
        """
        # Tokenize the context
        ctx_ids = self._tokenize(ctx)
        ctx_len = ctx_ids.shape[-1]
        
        if self.block_size > 0:
            num_blocks = (ctx_len + self.block_size - 1) // self.block_size
            effective_block_size = self.block_size
            print(f"Context length {ctx_len}, block size {effective_block_size}. Creating {num_blocks} blocks.")
        else:
            num_blocks = self.num_blocks
            effective_block_size = (ctx_len + num_blocks - 1) // num_blocks
            print(f"Context length {ctx_len}, num_blocks {num_blocks}. Creating {effective_block_size}-sized blocks.")
        
        self.num_blocks = num_blocks # Update num_blocks in case it was auto-calc'd

        # Pad the context to be a multiple of effective_block_size
        if ctx_ids.shape[-1] % effective_block_size != 0:
            padding = effective_block_size - (ctx_ids.shape[-1] % effective_block_size)
            # Use pad token ID, default to 0 if not set
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            pad_tensor = torch.full((1, padding), pad_id, dtype=torch.long, device=self.model.device)
            ctx_ids = torch.cat((ctx_ids, pad_tensor), dim=-1)

        position_ids = torch.arange(0, ctx_ids.shape[-1]).unsqueeze(0).to(self.model.device)

        # Split into blocks
        ctx_id_blocks = list(ctx_ids.split(effective_block_size, dim=-1))
        pos_id_blocks = list(position_ids.split(effective_block_size, dim=-1))

        return ctx_id_blocks, pos_id_blocks, ctx_len

# [Continuing from Part 2]
    # ... (inside StarAttentionModel class)

    @torch.no_grad()
    def _perform_sequential_tidal_prefill(
        self,
        ctx_id_blocks: List[torch.Tensor],
        pos_id_blocks: List[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Runs the new sequential prefill logic.
        Processes context block-by-block, sampling tokens at selection layers,
        and building a new sparse KV cache (anchor cache) from the union of
        all sampled tokens.
        
        This function is run *only* on rank 0.
        """
        print(f"[Rank 0] Starting Sequential Tidal Prefill for {self.num_blocks} blocks.")
        
        # These track the *sparse* set of tokens we want to keep
        anchor_token_ids: Optional[torch.Tensor] = None
        anchor_pos_ids: Optional[torch.Tensor] = None
        anchor_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

        full_input_ids: Optional[torch.Tensor] = None # Tracks all processed tokens
        full_pos_ids: Optional[torch.Tensor] = None   # Tracks all processed pos_ids
        
        master_anchor_pos_set: Set[int] = set() # Tracks unique pos_ids of anchors

        for i in range(self.num_blocks):
            print(f"[Rank 0] Processing block {i+1} / {self.num_blocks}...")
            block_ids = ctx_id_blocks[i]
            block_pos_ids = pos_id_blocks[i]
            
            # --- 1. Prepare Inputs for this step ---
            # The model processes the *new block* plus all *previous blocks*
            # This is standard prefill, but we will only *keep* the sparse anchors.
            if full_input_ids is None:
                current_input_ids = block_ids
                current_pos_ids = block_pos_ids
            else:
                current_input_ids = torch.cat([full_input_ids, block_ids], dim=-1)
                current_pos_ids = torch.cat([full_pos_ids, block_pos_ids], dim=-1)
            
            # Update the full history
            full_input_ids = current_input_ids
            full_pos_ids = current_pos_ids

            # We pass the *sparse* cache from the *previous* step
            # The model will recompute KVs for the new block and concat them
            # This is inefficient, but a pure-python implementation requires it.
            # A custom kernel would append to the cache.
            # For now, we re-compute the sparse anchor KVs + the new block KVs.
            # We must pass the *tokens* and *positions* for the anchors,
            # plus the *tokens* and *positions* for the new block.
            if anchor_token_ids is None:
                step_input_ids = block_ids
                step_pos_ids = block_pos_ids
            else:
                step_input_ids = torch.cat([anchor_token_ids, block_ids], dim=-1)
                step_pos_ids = torch.cat([anchor_pos_ids, block_pos_ids], dim=-1)
            
            # --- 2. Run Model Forward Pass ---
            # Clear history for this *block*
            # Note: We call clear on the *inner* model
            self.model.model.clear_top_tokens_history()
            
            outputs = self.model(
                input_ids=step_input_ids,
                position_ids=step_pos_ids,
                past_key_values=None, # We recompute KVs each time
                use_cache=True,
                # These are the custom args for our modified attention
                selection_layers=self.selection_layers,
                prefill_block_idx=i, # Use block index as the step key
            )
            
            # This cache contains KVs for (anchor_tokens + block_tokens)
            full_kv_cache_this_step = outputs.past_key_values
            
            # --- 3. Extract New Anchors ---
            # Get the sampled token indices (as position IDs) for this block
            history = self.model.model.get_top_tokens_for_step(i)
            if not history:
                print(f"[Rank 0] Warning: No tokens sampled for block {i}.")
                continue

            new_anchor_pos_set: Set[int] = set()
            for layer_idx in self.selection_layers:
                if layer_idx in history:
                    new_anchor_pos_set.update(history[layer_idx])
            
            # Filter this set to *only* include positions from the *current block*
            block_start_pos = block_pos_ids[0, 0].item()
            block_end_pos = block_pos_ids[0, -1].item()
            
            new_block_anchor_positions = sorted(
                [p for p in new_anchor_pos_set if block_start_pos <= p <= block_end_pos]
            )
            
            if not new_block_anchor_positions:
                print(f"[Rank 0] No *new* anchor tokens found in block {i}.")
                # We still need to rebuild the sparse cache from previous anchors
                if anchor_pos_ids is not None:
                    anchor_indices = anchor_pos_ids[0]
                    anchor_kv_cache = self._gather_kv_cache(
                        full_kv_cache_this_step, anchor_indices
                    )
                continue

            print(f"[Rank 0] Found {len(new_block_anchor_positions)} new anchor tokens in block {i}.")

            # Add to the master set of all anchor positions
            master_anchor_pos_set.update(new_block_anchor_positions)
            
            # --- 4. Update Sparse Anchor State for Next Loop ---
            
            # Get the token_ids for these new positions
            relative_indices = torch.tensor(
                [p - block_start_pos for p in new_block_anchor_positions],
                device=self.model.device, dtype=torch.long
            )
            new_anchor_tokens = torch.index_select(block_ids, dim=1, index=relative_indices)
            new_anchor_positions = torch.tensor(
                [new_block_anchor_positions], device=self.model.device, dtype=torch.long
            )

            # Update the master list of anchor tokens and positions
            if anchor_token_ids is None:
                anchor_token_ids = new_anchor_tokens
                anchor_pos_ids = new_anchor_positions
            else:
                anchor_token_ids = torch.cat([anchor_token_ids, new_anchor_tokens], dim=-1)
                anchor_pos_ids = torch.cat([anchor_pos_ids, new_anchor_positions], dim=-1)

            # --- 5. Re-build Sparse KV Cache ---
            # This is the most critical step. We must slice the
            # `full_kv_cache_this_step` to keep *only* the KVs for
            # tokens in our `master_anchor_pos_set`.
            
            # The `full_kv_cache_this_step` corresponds to `step_input_ids`.
            # We need the *relative indices* within that cache.
            all_anchor_pos_list = sorted(list(master_anchor_pos_set))
            
            # Find the indices of our anchors within the `step_pos_ids`
            # This is complex. Let's simplify.
            # `full_kv_cache_this_step` has shape (..., seq_len, ...)
            # where seq_len = anchor_pos_ids.shape[1] + block_pos_ids.shape[1]
            # The first N are anchors, the last M are the block.
            
            # Let's rebuild the cache from *all* anchors found so far
            master_anchor_indices = torch.tensor(
                sorted(list(master_anchor_pos_set)), 
                device=self.model.device, dtype=torch.long
            )
            
            # We must map these *global* positions to the *relative* indices
            # in the `full_kv_cache_this_step`.
            
            # `step_pos_ids` looks like [5, 12, 18, | 1024, 1025, ..., 2047]
            # `master_anchor_indices` looks like [5, 12, 18, | 1028, 1030, 1500]
            
            # Let's build a map from pos_id -> relative_index
            pos_to_idx_map = {pos_id.item(): idx for idx, pos_id in enumerate(step_pos_ids[0])}
            
            relative_indices_to_gather = [
                pos_to_idx_map[pos_id] for pos_id in all_anchor_pos_list 
                if pos_id in pos_to_idx_map
            ]
            
            if not relative_indices_to_gather:
                 print(f"[Rank 0] Warning: No anchor tokens to gather for block {i}.")
                 continue
            
            relative_indices_tensor = torch.tensor(
                relative_indices_to_gather,
                device=self.model.device, dtype=torch.long
            )

            anchor_kv_cache = self._gather_kv_cache(
                full_kv_cache_this_step, relative_indices_tensor
            )

        print(f"[Rank 0] Sequential prefill complete. Total anchor tokens: {len(master_anchor_pos_set)}")
        
        # After the loop, anchor_kv_cache holds the final sparse cache
        # And anchor_pos_ids holds the corresponding positions
        # We need to return the cache and the *original* context length
        
        # We also need to get the final, unpadded context length
        # `pos_id_blocks` includes padding. `ctx_len` is the original.
        # But Phase 2 needs the *padded* length to calc query pos_ids
        
        # The query position IDs should start *after* the *original* context
        self.final_context_len = pos_id_blocks[0].shape[-1] * self.num_blocks

        # Trim padding from the final cache
        if anchor_pos_ids is None:
             print("[Rank 0] Error: No anchor tokens found in any block.")
             return None, 0
             
        final_anchor_pos = anchor_pos_ids[0]
        unpadded_mask = final_anchor_pos < self.final_context_len
        unpadded_indices = torch.where(unpadded_mask)[0]
        
        final_sparse_cache = self._gather_kv_cache(anchor_kv_cache, unpadded_indices)
        final_sparse_pos_ids = torch.index_select(final_anchor_pos, dim=0, index=unpadded_indices)

        print(f"[Rank 0] Final sparse cache size: {final_sparse_pos_ids.shape[0]} tokens.")
        
        return final_sparse_cache


    def __call__(self, prompt_context: str, prompt_query: str) -> Dict[str, List[str]]:
        
        # --- PHASE 1: Sequential Tidal Prefill ---
        # Only Rank 0 does the sequential prefill
        
        # 1. Tokenize and partition context
        # This is done on all ranks to get ctx_len, but only rank 0 *uses* the blocks
        ctx_id_blocks, pos_id_blocks, ctx_len = self._tokenize_and_partition_context(ctx=prompt_context)
        self.final_context_len = ctx_len # Store original length
        
        final_sparse_cache = None
        if self.rank == 0:
            final_sparse_cache = self._perform_sequential_tidal_prefill(
                ctx_id_blocks, pos_id_blocks
            )
        
        # 2. Distribute the final sparse cache from Rank 0 to all other ranks
        sharded_kv_cache = self._shard_and_distribute_cache(final_sparse_cache)

        # Barrier to ensure all ranks have their cache
        dist.barrier()
        if self.rank == 0:
            print("Sequential prefill and cache distribution complete.")

        # --- PHASE 2: Distributed Star-Attention Generation ---
        qry_ids = self._tokenize(prompt_query)
        
        # Query positions start *after* the original context length
        qry_position_ids = torch.arange(
            self.final_context_len, 
            self.final_context_len + qry_ids.shape[-1]
        ).unsqueeze(0).to(self.model.device)
        
        # Use the *original* _generate_output from the base class
        # It's already built for distributed generation
        output = self._generate_output(qry_ids, qry_position_ids, sharded_kv_cache)

        # Get the generated text
        generated_text = self._get_output_text(output)
        return {'text': [generated_text]}
    
# [Continuing from Part 3]
    # ... (inside StarAttentionModel class)

    def _gather_kv_cache(
        self,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        indices: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Gathers (slices) the KV cache along the sequence dimension
        using the provided indices.
        """
        new_kv_cache = []
        for layer_kv in kv_cache:
            k_cache, v_cache = layer_kv
            # K/V cache shape: [bsz, num_heads, seq_len, head_dim]
            # We select along the seq_len dimension (dim=-2)
            new_k = torch.index_select(k_cache, dim=-2, index=indices)
            new_v = torch.index_select(v_cache, dim=-2, index=indices)
            new_kv_cache.append((new_k, new_v))
        return new_kv_cache

    def _shard_and_distribute_cache(
        self,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Shares the KV cache from rank 0 with all other ranks.
        Rank 0 splits the cache and sends chunks.
        Other ranks receive their chunk.
        """
        if self.world_size == 1:
            return kv_cache if kv_cache is not None else []

        if self.rank == 0:
            if kv_cache is None:
                # Should not happen if prefill is successful
                raise RuntimeError("Rank 0 has no KV cache to distribute.")
            
            # --- Rank 0: Split and Send ---
            print(f"[Rank 0] Sharding and distributing cache to {self.world_size} ranks.")
            sharded_cache_list = []
            
            # 1. Prepare metadata to send
            num_layers = len(kv_cache)
            k_shape = kv_cache[0][0].shape # [bsz, num_heads, seq_len, head_dim]
            v_shape = kv_cache[0][1].shape
            dtype = kv_cache[0][0].dtype
            
            # [bsz, num_heads, sparse_seq_len, head_dim]
            sparse_seq_len = k_shape[-2]
            
            # Calculate split size
            split_size = (sparse_seq_len + self.world_size - 1) // self.world_size
            
            # Send metadata
            metadata = {
                "num_layers": num_layers,
                "sparse_seq_len": sparse_seq_len,
                "split_size": split_size,
                "k_shape_tpl": (k_shape[0], k_shape[1], -1, k_shape[3]), # (bsz, num_h, S, head_d)
                "v_shape_tpl": (v_shape[0], v_shape[1], -1, v_shape[3]),
                "dtype": dtype,
            }
            metadata_tensor = torch.tensor(
                bytearray(str(metadata).encode('utf-8')),
                dtype=torch.uint8, device=self.model.device
            )
            
            for r in range(1, self.world_size):
                # Send size of metadata
                dist.send(torch.tensor(metadata_tensor.numel(), device=self.model.device), dst=r)
                # Send metadata
                dist.send(metadata_tensor, dst=r)

            # 2. Split and Send cache
            local_kv_cache = []
            for layer_kv in kv_cache:
                k_cache, v_cache = layer_kv
                k_chunks = list(k_cache.split(split_size, dim=-2))
                v_chunks = list(v_cache.split(split_size, dim=-2))
                
                # Pad the last chunk if necessary
                if k_chunks[-1].shape[-2] != split_size:
                    pad_size = split_size - k_chunks[-1].shape[-2]
                    k_pad = torch.zeros(
                        (*k_chunks[-1].shape[:-2], pad_size, k_chunks[-1].shape[-1]),
                        dtype=dtype, device=self.model.device
                    )
                    v_pad = torch.zeros(
                        (*v_chunks[-1].shape[:-2], pad_size, v_chunks[-1].shape[-1]),
                        dtype=dtype, device=self.model.device
                    )
                    k_chunks[-1] = torch.cat([k_chunks[-1], k_pad], dim=-2)
                    v_chunks[-1] = torch.cat([v_chunks[-1], v_pad], dim=-2)

                # Send chunks to other ranks
                for r in range(1, self.world_size):
                    dist.send(k_chunks[r].contiguous(), dst=r)
                    dist.send(v_chunks[r].contiguous(), dst=r)
                
                # Keep chunk 0 for rank 0
                local_kv_cache.append((k_chunks[0].contiguous(), v_chunks[0].contiguous()))
            
            return local_kv_cache

        else:
            # --- Other Ranks: Receive ---
            
            # 1. Receive metadata
            metadata_size_tensor = torch.empty(1, dtype=torch.long, device=self.model.device)
            dist.recv(metadata_size_tensor, src=0)
            metadata_tensor = torch.empty(metadata_size_tensor.item(), dtype=torch.uint8, device=self.model.device)
            dist.recv(metadata_tensor, src=0)
            
            metadata_str = metadata_tensor.cpu().numpy().tobytes().decode('utf-8')
            metadata = eval(metadata_str)
            
            num_layers = metadata['num_layers']
            split_size = metadata['split_size']
            k_shape = list(metadata['k_shape_tpl'])
            k_shape[2] = split_size # Set seq_len to split_size
            v_shape = list(metadata['v_shape_tpl'])
            v_shape[2] = split_size
            dtype = metadata['dtype']

            # 2. Receive cache chunks
            local_kv_cache = []
            for _ in range(num_layers):
                k_recv = torch.empty(k_shape, dtype=dtype, device=self.model.device)
                v_recv = torch.empty(v_shape, dtype=dtype, device=self.model.device)
                dist.recv(k_recv, src=0)
                dist.recv(v_recv, src=0)
                local_kv_cache.append((k_recv, v_recv))
            
            print(f"[Rank {self.rank}] Received sharded cache.")
            return local_kv_cache


class DenseAttentionModel:

    def __init__(self, path: str, max_new_tokens: int, stop_words):
        from transformers import AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
        )

        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words if stop_words else []

    def _generate_output(self, input_ids, position_ids):
        output_seq, past_key_values = None, None
        for _ in range(self.max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids, position_ids=position_ids, past_key_values=past_key_values, use_cache=True
                )  # type: ignore

            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            output_seq = next_tokens if output_seq is None else torch.cat([output_seq, next_tokens])

            # Update the input_ids and position_ids for the next iteration
            input_ids = next_tokens.unsqueeze(0)
            position_ids = torch.tensor([[position_ids[-1, -1] + 1]]).to(position_ids)

        return output_seq.unsqueeze(0)

    def _get_output_text(self, output, truncate_texts=[]):
        # Remove the input from the generated text
        generated_text = self.tokenizer.decode(output[0].detach().cpu().numpy().tolist())

        for t in truncate_texts:
            t = t.strip()
            if t and generated_text.startswith(t):
                generated_text = generated_text[len(t) :].strip()

        for s in self.stop_words:
            generated_text = generated_text.split(s)[0]

        return generated_text.strip()

    def __call__(self, prompt_context: str, prompt_query: str):
        prompt = prompt_context + prompt_query
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False).to(self.model.device)
        position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0).to(self.model.device)

        output = self._generate_output(input_ids, position_ids)

        return {'text': [self._get_output_text(output)]}

class CustomAccuracyModel:
    """
    Implements the custom two-pass mechanism for improved accuracy on a single GPU.
    FINAL VERSION: Replaces the broken .generate() method with a fully manual
    decoding loop to guarantee correct execution.
    """

    def __init__(self, path: str, max_new_tokens: int, block_size: int, k_summary_size: int, 
                 stop_words: Optional[List[str]] = None, summary_method: str = 'top_k'):
        if not torch.cuda.is_available():
            raise RuntimeError("This implementation requires a CUDA-enabled GPU.")
        
        self.device = "cuda"
        self.model_path = path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("\n[SETUP] Loading primary model with Flash Attention 2 in bfloat16...")
        self.model_flash = LlamaForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            attn_implementation='flash_attention_2',
        )
        self.model_flash.eval()

        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words if stop_words else []
        self.block_size = block_size
        self.k = k_summary_size
        self.summary_chunk_size = 512
        self.summary_method = summary_method
        print(f"[SETUP] Using summary_chunk_size of {self.summary_chunk_size} to manage memory.")
        print(f"[SETUP] Using summary method: {summary_method}")
        print("[SETUP] Initialization complete.")

    def _get_kmeans_summary(self, model_eager, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            chunks = list(input_ids.split(self.summary_chunk_size, dim=1))
            all_hidden_states = []
            
            # Get hidden states for each chunk
            for chunk in chunks:
                outputs = model_eager(chunk, output_hidden_states=True)
                # Use the last hidden state as token representations
                hidden_states = outputs.hidden_states[-1].squeeze(0)
                all_hidden_states.append(hidden_states)
                del outputs
                torch.cuda.empty_cache()
            
            # Combine hidden states
            combined_states = torch.cat(all_hidden_states, dim=0)
            
            # Convert to numpy for sklearn
            features = combined_states.cpu().numpy()
            
            # Apply k-means clustering
            n_clusters = min(self.k, features.shape[0])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features)
            
            # Find tokens closest to centroids
            centroids = kmeans.cluster_centers_
            selected_indices = []
            
            for i in range(n_clusters):
                cluster_points = features[clusters == i]
                if len(cluster_points) > 0:
                    # Find the point closest to the centroid
                    centroid = centroids[i]
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    closest_point_idx = np.argmin(distances)
                    # Get global index
                    global_idx = np.where(clusters == i)[0][closest_point_idx]
                    selected_indices.append(global_idx)
            
            # Sort indices to maintain sequence order
            selected_indices.sort()
            summary_ids = input_ids[:, selected_indices]
            
            return summary_ids

    def _get_top_k_summary(self, model_eager, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            chunks = list(input_ids.split(self.summary_chunk_size, dim=1))
            all_token_scores = []
            for chunk in chunks:
                outputs = model_eager(chunk, output_attentions=True)
                all_attentions = torch.stack(outputs.attentions)
                token_scores = all_attentions.sum(dim=(0, 1, 2, -1))
                all_token_scores.append(token_scores)
                del outputs, all_attentions, token_scores
                torch.cuda.empty_cache()
            
            combined_scores = torch.cat(all_token_scores, dim=0)
            _, top_k_indices_in_block = torch.topk(combined_scores, min(self.k, combined_scores.shape[0]))
            top_k_indices_sorted, _ = torch.sort(top_k_indices_in_block)
            summary_ids = input_ids[:, top_k_indices_sorted]
        return summary_ids

    def _get_summary(self, model_eager, input_ids: torch.Tensor) -> torch.Tensor:
        if self.summary_method == 'kmeans':
            return self._get_kmeans_summary(model_eager, input_ids)
        else:  # default to top_k
            return self._get_top_k_summary(model_eager, input_ids)

    def _get_output_text(self, full_output_ids: torch.Tensor, num_input_tokens: int) -> str:
        generated_ids = full_output_ids[0, num_input_tokens:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        for s in self.stop_words:
            generated_text = generated_text.split(s)[0]
        return generated_text.strip()

    def __call__(self, prompt_context: str, prompt_query: str) -> Dict[str, List[str]]:
        context_ids = self.tokenizer.encode(prompt_context, return_tensors='pt').to(self.device)
        context_blocks = list(context_ids.split(self.block_size, dim=1))
        
        # === PHASE 1: BUILD FULL CONTEXT KV CACHE ===
        with torch.no_grad():
            print("\n--- Phase 1, Pass 1: Generating summaries... ---")
            # ... (This section is correct and remains unchanged) ...
            model_eager = LlamaForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16, device_map='auto', attn_implementation='eager'
            ).eval()
            summaries = [self._get_summary(model_eager, block) for block in context_blocks]
            del model_eager
            torch.cuda.empty_cache()
            print("--- [Pass 1] All summaries generated. ---")

            print("\n--- Phase 1, Pass 2: Building final KV cache incrementally... ---")
            # ... (This section is correct and remains unchanged) ...
            full_kv_cache_tuple = None
            for i, block in enumerate(context_blocks):
                previous_summaries = summaries[:i]
                if previous_summaries:
                    augmented_input_ids = torch.cat(previous_summaries + [block], dim=1)
                    summary_len = sum(s.shape[1] for s in previous_summaries)
                else:
                    augmented_input_ids = block
                    summary_len = 0
                outputs = self.model_flash(augmented_input_ids, use_cache=True)
                current_kv_cache = outputs.past_key_values
                sliced_kv_cache = []
                for layer_cache in current_kv_cache:
                    key, value = layer_cache
                    sliced_key = key[:, :, summary_len:, :]
                    sliced_value = value[:, :, summary_len:, :]
                    sliced_kv_cache.append((sliced_key, sliced_value))
                if full_kv_cache_tuple is None:
                    full_kv_cache_tuple = tuple(sliced_kv_cache)
                else:
                    new_full_cache = []
                    for idx, layer_cache in enumerate(full_kv_cache_tuple):
                        full_key, full_value = layer_cache
                        slice_key, slice_value = sliced_kv_cache[idx]
                        combined_key = torch.cat([full_key, slice_key], dim=2)
                        combined_value = torch.cat([full_value, slice_value], dim=2)
                        new_full_cache.append((combined_key, combined_value))
                    full_kv_cache_tuple = tuple(new_full_cache)
            print("--- [Pass 2] Final KV cache for the full context is built. ---")

            # === PHASE 2: GENERATE RESPONSE ===
            print("\n--- Phase 2: Generating response... ---")
            
            messages = [{"role": "user", "content": prompt_query}]
            generation_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)

            # Step 1: Manually prefill the cache with the query.
            past_key_values_cache_object = DynamicCache.from_legacy_cache(full_kv_cache_tuple)
            print(f"  [Phase 2] Step 1: Manually prefilling cache with query. (Cache len: {past_key_values_cache_object.get_seq_length()}, Query len: {generation_ids.shape[1]})")
            outputs = self.model_flash(
                input_ids=generation_ids,
                past_key_values=past_key_values_cache_object,
                use_cache=True
            )
            
            # Step 2: Manually generate the first token.
            print("  [Phase 2] Step 2: Manually generating the first token...")
            current_cache = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            # Step 3: Manually implement the decoding loop to replace .generate().
            print("  [Phase 2] Step 3: Manually running decoding loop...")
            all_generated_ids = [next_token]
            for _ in range(self.max_new_tokens - 1):
                # The model's forward pass can correctly infer position_ids from the cache
                # when the input is a single token, so we don't need to pass it manually here.
                outputs = self.model_flash(
                    input_ids=next_token,
                    past_key_values=current_cache,
                    use_cache=True,
                )
                current_cache = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                all_generated_ids.append(next_token)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
            
            # Step 4: Combine and decode the final result.
            generated_sequence = torch.cat(all_generated_ids, dim=1)
            final_output_ids = torch.cat([generation_ids, generated_sequence], dim=1)
            generated_text = self._get_output_text(final_output_ids, num_input_tokens=generation_ids.shape[1])
            return {'text': [generated_text]}
