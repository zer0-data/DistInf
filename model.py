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
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from transformers import AutoTokenizer,LlamaForCausalLM  
from transformers.cache_utils import DynamicCache # New Import
import warnings


class DistributedInferenceBaseModel:
    def __init__(
        self,
        path: str,
        max_new_tokens: int,
        stop_words: Optional[List[str]] = None,
        block_size: int = -1,
        anchor_block_size: int = -1,
    ):
        from star_attention import LlamaForCausalLM

        self._init_distributed()

        # Setup the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        # Define the model
        self.model = LlamaForCausalLM.from_pretrained(
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
        """Split the input context into blocks. The last block is padded to keep each block the same size.

        Args:
            ctx: input context

        Returns:
            token ids, position ids, context length (before padding)
        """
        raise NotImplementedError

    def _process_blockwise_context(
        self, ctx_ids_blocks: Tuple[torch.Tensor, ...], position_ids_blocks: Tuple[torch.Tensor, ...]
    ):
        """Generate the KV cache for the context assigned to the current rank.

        Args:
            ctx_ids_blocks: context blocks grouped by rank
            position_ids_blocks: position ids blocks grouped by rank

        Returns:
            KV cache for the context assigned to the current rank
        """
        raise NotImplementedError

    def _generate_output(self, input_ids, position_ids, past_key_values):
        """Phase 2 of Star Attention: Process input tokens followed by autoregressive token generation.

        Args:
            input_ids: input token ids
            position_ids: position ids of the input tokens
            past_key_values: KV cache

        Returns:
            generated token ids
        """
        output_seq = None
        for _ in range(self.max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    enable_star_attn=True,
                )  # type: ignore

            # Assign the new updated KV-cache to the last rank
            if self.rank == self.world_size - 1:
                past_key_values = outputs.past_key_values

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


class StarAttentionModel(DistributedInferenceBaseModel):
    """Star Attention - Phase 1 and Phase 2"""

    def _tokenize_and_partition_context(self, ctx):
        # Tokenize the context
        ctx_ids = self._tokenize(ctx)
        ctx_len = ctx_ids.shape[-1]

        # Split the context into chunks of size `block_size`
        if self.block_size is None:
            self.block_size = ctx_ids.shape[-1] // self.world_size

        # Pad the context to be a multiple of block_size
        if ctx_ids.shape[-1] % self.block_size != 0:
            padding = self.block_size - (ctx_ids.shape[-1] % self.block_size)
            ctx_ids = torch.cat((ctx_ids, torch.zeros_like(ctx_ids)[:, :padding]), dim=-1)

        position_ids = torch.arange(0, ctx_ids.shape[-1]).unsqueeze(0).to(self.model.device)

        return ctx_ids, position_ids, ctx_len

    def _process_blockwise_context(self, ctx_ids_blocks, position_ids_blocks):
        """Phase 1 of Star Attention: Blockwise Context Encoding with Anchor Blocks"""

        # If the anchor block size is not provided, use the entire first block
        if self.anchor_block_size is None:
            self.anchor_block_size = ctx_ids_blocks[0][0].shape[-1]

        kv_rank = []
        for idx in range(len(ctx_ids_blocks[self.rank])):
            # Select the current block
            ctx_block = ctx_ids_blocks[self.rank][idx]
            position_block = position_ids_blocks[self.rank][idx]

            # From 2nd block onwards, prepend the anchor block to the current block
            if self.rank != 0 or idx > 0:
                ctx_block = torch.cat((ctx_ids_blocks[0][0][:, : self.anchor_block_size], ctx_block), dim=-1)
                position_block = torch.cat(
                    (position_ids_blocks[0][0][:, : self.anchor_block_size], position_block), dim=-1
                )

            with torch.no_grad():
                kv_block = self.model(
                    ctx_block,
                    position_ids=position_block,
                    use_cache=True,
                    num_ring_steps=0,  # disable ring attention (local blockwise attention)
                    enable_star_attn=False,
                ).past_key_values  # type: ignore

            # Discard the anchor block KV cache
            if self.rank != 0 or idx > 0:
                kv_block = [
                    [x[0][:, :, self.anchor_block_size :], x[1][:, :, self.anchor_block_size :]] for x in kv_block
                ]

            kv_rank = (
                kv_block
                if not kv_rank
                else [
                    [torch.cat((kv_rank[i][j], kv_block[i][j]), dim=-2) for j in range(2)] for i in range(len(kv_rank))
                ]
            )

        return kv_rank

    def __call__(self, prompt_context: str, prompt_query: str) -> Dict[str, List[str]]:
        # Prepare the context
        ctx_ids, position_ids, ctx_len = self._tokenize_and_partition_context(prompt_context)

        # Split the context into blocks and divide the blocks among the ranks
        ctx_ids_blocks = torch.tensor_split(torch.stack(ctx_ids.split(self.block_size, dim=-1)), self.world_size)
        position_ids_blocks = torch.tensor_split(
            torch.stack(position_ids.split(self.block_size, dim=-1)), self.world_size
        )

        # Phase 1: Generate the KV cache for the local context
        kv_rank = self._process_blockwise_context(ctx_ids_blocks, position_ids_blocks)
        if self.rank == self.world_size - 1:  # discard padding from the last rank
            padding = ctx_ids.shape[-1] - ctx_len
            if padding > 0:
                kv_rank = [
                    [kv_rank[i][0][:, :, :-padding], kv_rank[i][1][:, :, :-padding]] for i in range(len(kv_rank))
                ]

        # Phase 2: Process query with global attention
        qry_ids = self._tokenize(prompt_query)
        qry_position_ids = torch.arange(ctx_len, ctx_len + qry_ids.shape[-1]).unsqueeze(0).to(self.model.device)
        output = self._generate_output(qry_ids, qry_position_ids, kv_rank)

        # Get the generated text
        generated_text = self._get_output_text(output)
        return {'text': [generated_text]}


class RingAttentionModel(DistributedInferenceBaseModel):
    """Ring Attention augmented with Phase 2 of Star Attention for Fast Token Generation"""

    def __init__(self, path, max_new_tokens, stop_words=None):
        super().__init__(path, max_new_tokens, stop_words=stop_words)

    def _tokenize_and_partition_context(self, ctx):
        # Tokenize the context
        ctx_ids = self._tokenize(ctx)
        ctx_len = ctx_ids.shape[-1]

        # Pad the context to be a multiple of world_size
        if ctx_ids.shape[-1] % self.world_size != 0:
            padding = self.world_size - (ctx_ids.shape[-1] % self.world_size)
            ctx_ids = torch.cat((ctx_ids, torch.zeros_like(ctx_ids)[:, :padding]), dim=-1)

        # Split the context into blocks
        self.block_size = ctx_ids.shape[-1] // self.world_size

        position_ids = torch.arange(0, ctx_ids.shape[-1]).unsqueeze(0).to(self.model.device)

        return ctx_ids, position_ids, ctx_len

    def _process_blockwise_context(self, ctx_ids_blocks, position_ids_blocks):
        assert len(ctx_ids_blocks[self.rank]) == 1, 'Ring Attention expects only one block per rank'

        ctx_block = ctx_ids_blocks[self.rank][0]
        position_block = position_ids_blocks[self.rank][0]
        with torch.no_grad():
            kv_rank = self.model(
                ctx_block,
                position_ids=position_block,
                use_cache=True,
                num_ring_steps=-1,  # enable ring attention
                enable_star_attn=False,
            ).past_key_values  # type: ignore

        return kv_rank

    def __call__(self, prompt_context: str, prompt_query: str) -> Dict[str, List[str]]:
        # Prepare the context
        ctx_ids, position_ids, ctx_len = self._tokenize_and_partition_context(prompt_context)

        # Divide the context blocks among the ranks
        ctx_ids_blocks = torch.tensor_split(torch.stack(ctx_ids.split(self.block_size, dim=-1)), self.world_size)
        position_ids_blocks = torch.tensor_split(
            torch.stack(position_ids.split(self.block_size, dim=-1)), self.world_size
        )

        # Generate the KV cache for the local context
        kv_rank = self._process_blockwise_context(ctx_ids_blocks, position_ids_blocks)
        if self.rank == self.world_size - 1:  # discard padding from the last rank
            padding = ctx_ids.shape[-1] - ctx_len
            if padding > 0:
                kv_rank = [
                    [kv_rank[i][0][:, :, :-padding], kv_rank[i][1][:, :, :-padding]] for i in range(len(kv_rank))
                ]

        # Phase 2 from Star Attention: Global attention with online softmax
        qry_ids = self._tokenize(prompt_query)
        qry_position_ids = torch.arange(ctx_len, ctx_len + qry_ids.shape[-1]).unsqueeze(0).to(self.model.device)
        output = self._generate_output(qry_ids, qry_position_ids, kv_rank)

        # Get the generated text
        generated_text = self._get_output_text(output)
        return {'text': [generated_text]}


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

# model.py

# ... (keep all existing code from the original file, including imports and all other classes)
# V V V V REPLACE THE PREVIOUS CustomAccuracyModel WITH THIS FINAL VERSION V V V V

# model.py

# ... (imports remain the same) ...

warnings.filterwarnings("ignore", category=UserWarning, message=".*where dtype is torch.float16.*")

import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from typing import Optional, List, Dict

class CustomAccuracyModel:
    """
    Implements the custom two-pass mechanism for improved accuracy on a single GPU.
    FINAL VERSION: Replaces the broken .generate() method with a fully manual
    decoding loop to guarantee correct execution.
    """

    def __init__(self, path: str, max_new_tokens: int, block_size: int, k_summary_size: int, stop_words: Optional[List[str]] = None):
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
        print(f"[SETUP] Using summary_chunk_size of {self.summary_chunk_size} to manage memory.")
        print("[SETUP] Initialization complete.")

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
            summaries = [self._get_top_k_summary(model_eager, block) for block in context_blocks]
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
