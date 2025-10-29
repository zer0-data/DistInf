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
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=UserWarning, message=".*where dtype is torch.float16.*")

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
