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
from transformers.cache_utils import DynamicCache

# --- NEW IMPORTS for k-means clustering ---
import numpy as np
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: sklearn not installed. K-means summarization will not be available.")
    print("Install with: pip install scikit-learn")
    KMeans = None
    cosine_similarity = None
    SKLEARN_AVAILABLE = False

# --- IMPORTS for Top-K Attention (from consolidated topk_attention module) ---
try:
    from topk_attention import (
        enable_topk_attention,
        load_topk_model,
        add_topk_history_tracking,
        add_topk_history_tracking_to_inner_model,
        TopKHistoryMixin,
        # Imports for block-wise accumulation
        start_block_accumulation,
        finish_block_accumulation,
        get_or_create_accumulator,
        AttentionScoreAccumulator,
    )
    TOPK_AVAILABLE = True
except ImportError:
    print("Warning: Could not import topk_attention module.")
    print("Please ensure 'topk_attention.py' is in your project root.")
    enable_topk_attention = None
    load_topk_model = None
    add_topk_history_tracking = None
    add_topk_history_tracking_to_inner_model = None
    TopKHistoryMixin = None
    start_block_accumulation = None
    finish_block_accumulation = None
    get_or_create_accumulator = None
    AttentionScoreAccumulator = None
    TOPK_AVAILABLE = False

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

    def __call__(self, prompt_context: str, prompt_query: str) -> Dict[str, List[str]]:
        context_ids = self.tokenizer.encode(prompt_context, return_tensors='pt').to(self.device)
        context_blocks = list(context_ids.split(self.block_size, dim=1))
        
        # === PHASE 1: BUILD FULL CONTEXT KV CACHE ===
        with torch.no_grad():
            print("\n--- Phase 1, Pass 1: Generating summaries... ---")
            model_eager = self.LlamaForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16, device_map='auto', attn_implementation='eager'
            ).eval()
            summaries = [self._get_summary(model_eager, block) for block in context_blocks]
            del model_eager
            torch.cuda.empty_cache()
            print("--- [Pass 1] All summaries generated. ---")


# [ Original RingAttentionModel class is skipped as it is not being modified ]
# ...

# [ Original DenseAttentionModel class is skipped as it is not being modified ]
# ...


class StarAttentionModel(DistributedInferenceBaseModel):
    """
    MODIFIED Star Attention
    Implements sequential Top-K Prefill (Phase 1)
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
        # Args for Top-K Prefill
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

        # 5. NOW, patch this model with the Top-K attention logic
        if not TOPK_AVAILABLE or enable_topk_attention is None:
            raise ImportError(
                "Could not import from 'topk_attention' module."
                " Please ensure 'topk_attention.py' is in your project root."
            )

        print("Applying Top-K attention patches to StarLlama model...")
        enable_topk_attention(
            self.model,
            attn_type="topk",
            top_k=self.top_k,
            selection_layers=self.selection_layers,
            sparse_layer_start=min(self.selection_layers) if self.selection_layers else 0,
            correction_layer=max(self.selection_layers) if self.selection_layers else 0,
        )
        
        # 6. Add history tracking methods to the model
        #    The patching in enable_topk_attention *replaces* the model's forward
        #    but the model object itself is still a StarLlamaForCausalLM.
        #    We add history tracking methods using the utility function.
        if not hasattr(self.model, "set_tokenizer_for_decode"):
            print("Adding history-tracking methods to model...")
            add_topk_history_tracking_to_inner_model(self.model)

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
    def _perform_sequential_topk_prefill(
        self,
        ctx_id_blocks: List[torch.Tensor],
        pos_id_blocks: List[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Runs the new sequential prefill logic with block-wise attention accumulation.
        
        For each block:
        1. Start accumulation for the block
        2. Run forward pass - attention scores are accumulated across ALL layers
        3. At the final layer, top-K selection is performed based on summed scores
        4. The accumulator is automatically cleaned up after each block
        
        This function is run *only* on rank 0.
        """
        print(f"[Rank 0] Starting Sequential Top-K Prefill for {self.num_blocks} blocks.")
        print(f"[Rank 0] Using block-wise attention accumulation across all layers.")
        
        # Get the accumulator for this model
        accumulator = get_or_create_accumulator(self.model)
        
        # These track the *sparse* set of tokens we want to keep
        anchor_token_ids: Optional[torch.Tensor] = None
        anchor_pos_ids: Optional[torch.Tensor] = None

        full_input_ids: Optional[torch.Tensor] = None
        full_pos_ids: Optional[torch.Tensor] = None
        
        # Master set of all anchor position IDs across all blocks
        master_anchor_pos_set: Set[int] = set()
        
        # Store selected tokens per block for building sparse cache
        all_selected_positions: List[int] = []

        for i in range(self.num_blocks):
            print(f"\n[Rank 0] Processing block {i+1} / {self.num_blocks}...")
            block_ids = ctx_id_blocks[i]
            block_pos_ids = pos_id_blocks[i]
            
            block_start_pos = block_pos_ids[0, 0].item()
            block_end_pos = block_pos_ids[0, -1].item()
            print(f"[Rank 0] Block positions: {block_start_pos} to {block_end_pos}")
            
            # --- 1. Start block accumulation ---
            # This tells the attention layers to start accumulating scores
            start_block_accumulation(self.model, block_pos_ids)
            
            # --- 2. Prepare inputs for this step ---
            if anchor_token_ids is None:
                step_input_ids = block_ids
                step_pos_ids = block_pos_ids
            else:
                step_input_ids = torch.cat([anchor_token_ids, block_ids], dim=-1)
                step_pos_ids = torch.cat([anchor_pos_ids, block_pos_ids], dim=-1)
            
            # --- 3. Run Model Forward Pass ---
            # Attention scores are accumulated across ALL layers
            # Selection happens automatically at the final layer
            outputs = self.model(
                input_ids=step_input_ids,
                position_ids=step_pos_ids,
                past_key_values=None,
                use_cache=True,
            )
            
            full_kv_cache_this_step = outputs.past_key_values
            
            # --- 4. Get selected tokens from the last layer's output ---
            # The chosen_tokens are returned as (indices, position_ids) from the final layer
            # We need to check if selection was made
            
            # Check if we got selected tokens (stored on the model after forward pass)
            if hasattr(self.model, '_last_block_selection'):
                selected_indices, selected_position_ids = self.model._last_block_selection
                del self.model._last_block_selection
            else:
                # Fallback: manually select if accumulator has data
                # This shouldn't happen if the forward pass worked correctly
                selected_indices, selected_position_ids = [], []
                print(f"[Rank 0] Warning: No selection result from forward pass for block {i}")
            
            if not selected_position_ids:
                print(f"[Rank 0] Warning: No tokens selected for block {i}.")
                continue
            
            # Filter to only include positions from the *current block*
            new_block_positions = [
                p for p in selected_position_ids 
                if block_start_pos <= p <= block_end_pos
            ]
            
            if not new_block_positions:
                print(f"[Rank 0] No new anchor tokens found in block {i}.")
                continue
                
            print(f"[Rank 0] Selected {len(new_block_positions)} anchor tokens from block {i}.")
            
            # --- 5. Update anchor state ---
            master_anchor_pos_set.update(new_block_positions)
            
            # Get token IDs for these positions
            relative_indices = torch.tensor(
                [p - block_start_pos for p in new_block_positions],
                device=self.model.device, dtype=torch.long
            )
            new_anchor_tokens = torch.index_select(block_ids, dim=1, index=relative_indices)
            new_anchor_positions = torch.tensor(
                [new_block_positions], device=self.model.device, dtype=torch.long
            )
            
            if anchor_token_ids is None:
                anchor_token_ids = new_anchor_tokens
                anchor_pos_ids = new_anchor_positions
            else:
                anchor_token_ids = torch.cat([anchor_token_ids, new_anchor_tokens], dim=-1)
                anchor_pos_ids = torch.cat([anchor_pos_ids, new_anchor_positions], dim=-1)
            
            # --- 6. Build position-to-index map for KV cache gathering ---
            all_anchor_pos_list = sorted(list(master_anchor_pos_set))
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
            
            # Note: The accumulator is automatically cleaned up in finish_block()
            # which is called at the final layer during forward pass

        print(f"\n[Rank 0] Sequential prefill complete. Total anchor tokens: {len(master_anchor_pos_set)}")
        
        # Final processing
        self.final_context_len = pos_id_blocks[0].shape[-1] * self.num_blocks

        if anchor_pos_ids is None:
            print("[Rank 0] Error: No anchor tokens found in any block.")
            return None
        
        # Trim padding from the final cache
        final_anchor_pos = anchor_pos_ids[0]
        unpadded_mask = final_anchor_pos < self.final_context_len
        unpadded_indices = torch.where(unpadded_mask)[0]
        
        final_sparse_cache = self._gather_kv_cache(anchor_kv_cache, unpadded_indices)
        final_sparse_pos_ids = torch.index_select(final_anchor_pos, dim=0, index=unpadded_indices)

        print(f"[Rank 0] Final sparse cache size: {final_sparse_pos_ids.shape[0]} tokens.")
        
        return final_sparse_cache


    def __call__(self, prompt_context: str, prompt_query: str) -> Dict[str, List[str]]:
        
        # --- PHASE 1: Sequential Top-K Prefill ---
        # Only Rank 0 does the sequential prefill
        
        # 1. Tokenize and partition context
        # This is done on all ranks to get ctx_len, but only rank 0 *uses* the blocks
        ctx_id_blocks, pos_id_blocks, ctx_len = self._tokenize_and_partition_context(ctx=prompt_context)
        self.final_context_len = ctx_len # Store original length
        
        final_sparse_cache = None
        if self.rank == 0:
            final_sparse_cache = self._perform_sequential_topk_prefill(
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
    Implements the SqueezedAttention-style two-pass mechanism:
    
    Phase 1 - Pass 1: For each block, cluster tokens and select representatives (summaries)
    Phase 1 - Pass 2: Build KV cache with context propagation:
        Block 1: Just Block1
        Block 2: Summary1 + Block2  
        Block 3: Summary1 + Summary2 + Block3
        Block 4: Summary1 + Summary2 + Summary3 + Block4
    Phase 2: Generate response using the full KV cache
    """

    def __init__(self, path: str, max_new_tokens: int, block_size: int, k_summary_size: int, 
                 stop_words: Optional[List[str]] = None, summary_method: str = 'kmeans',
                 pruning_percent: float = 75.0, 
                 # K-means parameters (SqueezedAttention style)
                 use_cosine_similarity: bool = True,
                 multi_layer_aggregation: bool = True,
                 layer_weights: Optional[List[float]] = None,
                 query_guided: bool = False,
                 query_weight: float = 0.3,
                 num_layers_for_clustering: int = 4):
        if not torch.cuda.is_available():
            raise RuntimeError("This implementation requires a CUDA-enabled GPU.")

        # Validate summary_method
        if summary_method == 'kmeans' and not SKLEARN_AVAILABLE:
            raise RuntimeError(
                "K-means summarization requires sklearn. "
                "Install with: pip install scikit-learn"
            )
        
        # Validate pruning_percent - should be between 0 and 100
        if not 0 < pruning_percent < 100:
            raise ValueError(
                f"pruning_percent must be between 0 and 100 (exclusive), got {pruning_percent}. "
                f"E.g., pruning_percent=90 means prune 90% of tokens (keep 10%)."
            )
        
        self.device = "cuda"
        self.model_path = path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("\n[SETUP] Loading primary model with Flash Attention 2 in bfloat16...")
        
        # Import AutoModelForCausalLM - this works for all model types
        from transformers import AutoModelForCausalLM
        
        self.model_flash = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            attn_implementation='flash_attention_2',
            trust_remote_code=True,
        )
        self.model_flash.eval()
        
        # Store the class for later use (for loading eager model)
        self._model_class = AutoModelForCausalLM

        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words if stop_words else []
        self.block_size = block_size
        self.k = k_summary_size
        self.summary_chunk_size = 512
        self.summary_method = summary_method
        self.pruning_percent = pruning_percent
        
        # K-means / SqueezedAttention parameters
        self.use_cosine_similarity = use_cosine_similarity
        self.multi_layer_aggregation = multi_layer_aggregation
        self.layer_weights = layer_weights
        self.query_guided = query_guided
        self.query_weight = query_weight
        self.num_layers_for_clustering = num_layers_for_clustering
        
        print(f"[SETUP] Block size: {block_size}")
        print(f"[SETUP] Summary method: {summary_method}")
        if summary_method == 'kmeans':
            print(f"[SETUP] Pruning: {pruning_percent}% (keeping {100 - pruning_percent:.1f}% of tokens per block)")
            print(f"[SETUP] Cosine similarity: {use_cosine_similarity}")
            print(f"[SETUP] Multi-layer aggregation: {multi_layer_aggregation} ({num_layers_for_clustering} layers)")
        print("[SETUP] Initialization complete.")

    def _extract_hidden_states_for_clustering(
        self, 
        model_eager, 
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract hidden states for clustering.
        
        SqueezedAttention approach: Use hidden states from later layers
        as they contain more semantic information.
        
        Returns:
            Hidden states tensor of shape (seq_len, hidden_dim)
        """
        with torch.no_grad():
            # Process the entire block at once to preserve context
            outputs = model_eager(input_ids, output_hidden_states=True)
            hidden_states_tuple = outputs.hidden_states  # (num_layers + 1,) tensors
            
            if self.multi_layer_aggregation:
                # Aggregate from last N layers (SqueezedAttention style)
                num_total_layers = len(hidden_states_tuple) - 1  # Exclude embedding
                layers_to_use = min(self.num_layers_for_clustering, num_total_layers)
                
                # Get last N layers
                selected_layers = hidden_states_tuple[-layers_to_use:]
                
                # Compute weights (later layers get more weight)
                if self.layer_weights is None:
                    weights = torch.tensor(
                        [2 ** i for i in range(layers_to_use)],
                        dtype=torch.float32,
                        device=self.device
                    )
                    weights = weights / weights.sum()
                else:
                    weights = torch.tensor(self.layer_weights[-layers_to_use:], device=self.device)
                    weights = weights / weights.sum()
                
                # Weighted sum: shape (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
                aggregated = torch.zeros_like(selected_layers[0].squeeze(0))
                for layer_states, weight in zip(selected_layers, weights):
                    aggregated += weight * layer_states.squeeze(0)
                
                result = aggregated
            else:
                # Use only last layer: shape (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
                result = outputs.hidden_states[-1].squeeze(0)
            
            del outputs, hidden_states_tuple
            torch.cuda.empty_cache()
            
            return result

    def _cluster_and_select_representatives(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        n_clusters: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        SqueezedAttention-style clustering and selection:
        1. Run k-means on hidden states
        2. For each cluster, select the token closest to centroid
        
        Args:
            hidden_states: (seq_len, hidden_dim) tensor
            input_ids: (1, seq_len) tensor
            n_clusters: Number of clusters (tokens to keep)
            
        Returns:
            summary_ids: (1, n_selected) token IDs
            selected_indices: List of selected token indices
        """
        if KMeans is None:
            raise RuntimeError("sklearn is required for k-means clustering")
        
        seq_len = hidden_states.shape[0]
        
        # Handle edge case: if n_clusters >= seq_len, return all tokens
        if n_clusters >= seq_len:
            selected_indices = list(range(seq_len))
            selected_indices_tensor = torch.tensor(selected_indices, device=input_ids.device, dtype=torch.long)
            return input_ids[:, selected_indices_tensor], selected_indices
        
        # Convert to numpy for sklearn
        features = hidden_states.float().cpu().numpy()
        
        # Identify zero-norm vectors before normalization
        norms = np.linalg.norm(features, axis=1)
        zero_norm_mask = norms < 1e-8
        num_zero_norm = zero_norm_mask.sum()
        
        if num_zero_norm > 0:
            print(f"    [Warning] Found {num_zero_norm} zero-norm vectors")
        
        # Normalize for cosine similarity-based clustering
        if self.use_cosine_similarity:
            # Add small epsilon to avoid division by zero
            norms_safe = np.where(norms < 1e-8, 1.0, norms)
            features = features / norms_safe[:, np.newaxis]
        
        # Ensure n_clusters doesn't exceed number of unique points
        # K-means can fail if there are fewer unique points than clusters
        n_clusters = min(n_clusters, seq_len)
        
        # Run k-means with error handling
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            centroids = kmeans.cluster_centers_
        except Exception as e:
            print(f"    [Warning] K-means failed: {e}. Using uniform sampling fallback.")
            # Fallback: uniformly sample n_clusters tokens
            indices = np.linspace(0, seq_len - 1, n_clusters, dtype=int)
            selected_indices = sorted(indices.tolist())
            selected_indices_tensor = torch.tensor(selected_indices, device=input_ids.device, dtype=torch.long)
            return input_ids[:, selected_indices_tensor], selected_indices
        
        # Select representative from each cluster (closest to centroid)
        selected_indices = []
        empty_clusters = 0
        
        for cluster_id in range(n_clusters):
            # Get indices of tokens in this cluster
            cluster_mask = (cluster_labels == cluster_id)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                empty_clusters += 1
                continue
            
            # Get features of tokens in this cluster
            cluster_features = features[cluster_mask]
            centroid = centroids[cluster_id]
            
            # Compute distance to centroid
            if self.use_cosine_similarity:
                # For normalized vectors: cosine_sim = dot product
                # Handle potential numerical issues
                similarities = np.clip(cluster_features @ centroid, -1.0, 1.0)
                distances = 1 - similarities
            else:
                distances = np.linalg.norm(cluster_features - centroid, axis=1)
            
            # Select token with minimum distance to centroid
            best_idx_in_cluster = np.argmin(distances)
            global_idx = cluster_indices[best_idx_in_cluster]
            selected_indices.append(int(global_idx))
        
        if empty_clusters > 0:
            print(f"    [Warning] {empty_clusters} empty clusters detected")
            
            # Compensate for empty clusters by adding more tokens
            # Select tokens that are furthest from any selected token
            if len(selected_indices) < n_clusters:
                remaining_needed = n_clusters - len(selected_indices)
                all_indices = set(range(seq_len))
                available_indices = list(all_indices - set(selected_indices))
                
                if available_indices and remaining_needed > 0:
                    # Add tokens uniformly from available indices
                    step = max(1, len(available_indices) // remaining_needed)
                    additional = available_indices[::step][:remaining_needed]
                    selected_indices.extend(additional)
                    print(f"    [Info] Added {len(additional)} tokens to compensate for empty clusters")
        
        # Sort to maintain original sequence order (important for positional encoding)
        selected_indices = sorted(set(selected_indices))  # Remove duplicates and sort
        
        # Extract selected token IDs
        selected_indices_tensor = torch.tensor(selected_indices, device=input_ids.device, dtype=torch.long)
        summary_ids = input_ids[:, selected_indices_tensor]
        
        return summary_ids, selected_indices

    def _get_block_summary(
        self,
        model_eager,
        block_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate summary for a single block using SqueezedAttention approach:
        1. Extract hidden states
        2. Cluster tokens
        3. Select representative tokens
        
        Args:
            model_eager: Model with eager attention
            block_ids: (1, block_len) token IDs
            
        Returns:
            summary_ids: (1, n_summary) selected token IDs
        """
        block_len = block_ids.shape[1]
        
        # Calculate number of tokens to keep
        keep_ratio = (100.0 - self.pruning_percent) / 100.0
        n_clusters = max(1, int(block_len * keep_ratio))
        
        print(f"    [Clustering] Block size: {block_len}, Keeping: {n_clusters} tokens ({100 - self.pruning_percent:.1f}%)")
        
        # Extract hidden states for clustering
        hidden_states = self._extract_hidden_states_for_clustering(model_eager, block_ids)
        
        # Cluster and select representatives
        summary_ids, selected_indices = self._cluster_and_select_representatives(
            hidden_states, block_ids, n_clusters
        )
        
        print(f"    [Clustering] Selected {len(selected_indices)} representative tokens")
        
        return summary_ids

    def _get_top_k_summary(
        self, 
        model_eager, 
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Original top-k attention-based summary extraction.
        """
        with torch.no_grad():
            chunks = list(input_ids.split(self.summary_chunk_size, dim=1))
            all_attention_scores = []
            
            for chunk in chunks:
                outputs = model_eager(chunk, output_attentions=True)
                attentions = outputs.attentions  # Tuple of (bsz, num_heads, seq_len, seq_len)
                
                # Average attention across layers and heads
                stacked_attentions = torch.stack(attentions, dim=0)  # (num_layers, bsz, heads, seq, seq)
                avg_attention = stacked_attentions.mean(dim=(0, 1, 2))  # (seq, seq)
                
                # Sum attention received by each token
                token_importance = avg_attention.sum(dim=0)  # (seq,)
                all_attention_scores.append(token_importance)
                
                del outputs, attentions, stacked_attentions
                torch.cuda.empty_cache()
            
            # Combine scores from all chunks
            combined_scores = torch.cat(all_attention_scores, dim=0)
            
            # Select top-k
            k = min(self.k, combined_scores.shape[0])
            _, top_indices = torch.topk(combined_scores, k)
            top_indices = top_indices.sort().values
            
            summary_ids = input_ids[:, top_indices]
            
            return summary_ids

    def _get_summary(
        self, 
        model_eager, 
        input_ids: torch.Tensor,
        query_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get summary tokens for a block.
        """
        if self.summary_method == 'kmeans':
            return self._get_block_summary(model_eager, input_ids)
        else:
            return self._get_top_k_summary(model_eager, input_ids)

    def _get_output_text(self, output_ids: torch.Tensor, num_input_tokens: int) -> str:
        """Convert generated token IDs to text."""
        generated_ids = output_ids[:, num_input_tokens:]
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        for s in self.stop_words:
            generated_text = generated_text.split(s)[0]
        
        return generated_text.strip()

    def __call__(self, prompt_context: str, prompt_query: str) -> Dict[str, List[str]]:
        """
        Main inference method implementing SqueezedAttention-style processing:
        
        Phase 1 - Pass 1: Generate summaries for each block independently
        Phase 1 - Pass 2: Build KV cache with context propagation:
            Block 1: Block1 → KV1
            Block 2: Summary1 + Block2 → KV2  
            Block 3: Summary1 + Summary2 + Block3 → KV3
            Block 4: Summary1 + Summary2 + Summary3 + Block4 → KV4
            Final KV = concat(KV1, KV2, KV3, KV4)
        Phase 2: Generate response
        """
        context_ids = self.tokenizer.encode(prompt_context, return_tensors='pt').to(self.device)
        context_blocks = list(context_ids.split(self.block_size, dim=1))
        num_blocks = len(context_blocks)
        
        print(f"\n[INFO] Context length: {context_ids.shape[1]}, Block size: {self.block_size}, Num blocks: {num_blocks}")
        
        # === PHASE 1, PASS 1: Generate summaries for each block ===
        with torch.no_grad():
            print("\n" + "="*60)
            print("PHASE 1 - PASS 1: Generating block summaries")
            print("="*60)
            
            # Load eager model using the stored class reference
            model_eager = self._model_class.from_pretrained(
                self.model_path, 
                torch_dtype=torch.bfloat16, 
                device_map='auto', 
                attn_implementation='eager',
                trust_remote_code=True,
            ).eval()
            
            summaries = []
            for i, block in enumerate(context_blocks):
                print(f"\n  Processing Block {i+1}/{num_blocks} (size: {block.shape[1]})")
                summary = self._get_summary(model_eager, block)
                summaries.append(summary)
                print(f"    Summary size: {summary.shape[1]}")
            
            del model_eager
            torch.cuda.empty_cache()
            print("\n[Pass 1 Complete] All summaries generated.")

            # === PHASE 1, PASS 2: Build KV cache with context propagation ===
            print("\n" + "="*60)
            print("PHASE 1 - PASS 2: Building KV cache with context propagation")
            print("="*60)
            print("Flow:")
            print("  Block 1: Block1")
            print("  Block 2: Summary1 + Block2")
            print("  Block 3: Summary1 + Summary2 + Block3")
            print("  ...")
            
            full_kv_cache_tuple = None
            
            for i, block in enumerate(context_blocks):
                # Get all summaries from PREVIOUS blocks
                previous_summaries = summaries[:i]
                
                if previous_summaries:
                    # Concatenate: [Summary1, Summary2, ..., Summary_{i-1}, Block_i]
                    augmented_input_ids = torch.cat(previous_summaries + [block], dim=1)
                    summary_len = sum(s.shape[1] for s in previous_summaries)
                    print(f"\n  Block {i+1}: {len(previous_summaries)} summaries ({summary_len} tokens) + Block ({block.shape[1]} tokens) = {augmented_input_ids.shape[1]} total")
                else:
                    # First block: just the block itself
                    augmented_input_ids = block
                    summary_len = 0
                    print(f"\n  Block {i+1}: Block only ({block.shape[1]} tokens)")
                
                # Forward pass to get KV cache
                outputs = self.model_flash(augmented_input_ids, use_cache=True)
                current_kv_cache = outputs.past_key_values
                
                # Slice out only the KV for the CURRENT BLOCK (not the summaries)
                # This is the key insight: we want KV[summary_len:] which is just the block's KV
                sliced_kv_cache = []
                for layer_cache in current_kv_cache:
                    key, value = layer_cache
                    # key/value shape: [bsz, num_heads, seq_len, head_dim]
                    sliced_key = key[:, :, summary_len:, :]
                    sliced_value = value[:, :, summary_len:, :]
                    sliced_kv_cache.append((sliced_key, sliced_value))
                
                print(f"    KV cache slice: positions [{summary_len}:{augmented_input_ids.shape[1]}] = {augmented_input_ids.shape[1] - summary_len} tokens")
                
                # Accumulate into full KV cache
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
                
                del outputs, current_kv_cache
                torch.cuda.empty_cache()
            
            # Report final KV cache size
            final_kv_len = full_kv_cache_tuple[0][0].shape[2]
            print(f"\n[Pass 2 Complete] Final KV cache length: {final_kv_len}")
            print(f"  Original context: {context_ids.shape[1]} tokens")
            print(f"  Compression ratio: {context_ids.shape[1] / final_kv_len:.2f}x")

            # === PHASE 2: Generate response ===
            print("\n" + "="*60)
            print("PHASE 2: Generating response")
            print("="*60)
            
            messages = [{"role": "user", "content": prompt_query}]
            generation_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)

            past_key_values_cache_object = DynamicCache.from_legacy_cache(full_kv_cache_tuple)
            print(f"  Cache length: {past_key_values_cache_object.get_seq_length()}")
            print(f"  Query length: {generation_ids.shape[1]}")
            
            # Prefill with query
            outputs = self.model_flash(
                input_ids=generation_ids,
                past_key_values=past_key_values_cache_object,
                use_cache=True
            )
            
            # Generate tokens
            current_cache = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            all_generated_ids = [next_token]
            for _ in range(self.max_new_tokens - 1):
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
            
            generated_sequence = torch.cat(all_generated_ids, dim=1)
            final_output_ids = torch.cat([generation_ids, generated_sequence], dim=1)
            generated_text = self._get_output_text(final_output_ids, num_input_tokens=generation_ids.shape[1])
            
            print(f"\n[Generation Complete]")
            print("="*60)
            
            return {'text': [generated_text]}
