import math
import time  # Import time module
import types
from typing import Optional, Tuple, List
import torch.nn.functional as F
from torch import nn
import torch
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs


def llama_tidal_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    top_k: int = None,
    sparse_layer_start=2,
    correction_layer=9,
    attention_sink=0,
    lim_ratio_factor=1,
    **kwargs: Unpack[FlashAttentionKwargs],
):
    output_attentions = kwargs.get("output_attentions", False)

    # If output_attentions is requested, fall back to original implementation
    if output_attentions:
        # Note: original_forward may not return 3 values.
        # This path is for debugging/fallback and we assume we don't sample.
        attn_output, attn_weights = self.original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        return attn_output, attn_weights, None

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    
    # NEW: Get selection_layers from kwargs
    selection_layers: Optional[List[int]] = kwargs.get("selection_layers", None)
    chosen_tokens = None  # Initialize chosen_tokens

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    bsz, num_heads, q_len, head_dim = query_states.shape
    kv_seq_len = key_states.shape[-2]

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # --- NEW LOGIC ---
    is_prefill = q_len > 1 and (q_len == kv_seq_len) # More robust prefill check
    
    is_prefill_selection_layer = (
        is_prefill
        and selection_layers is not None
        and self.layer_idx in selection_layers
        and top_k is not None
    )
    
    # Use fast SDPA for:
    # 1. Non-sparse layers (e.g., layer 0, 1)
    # 2. Prefill, *unless* it's a designated selection layer
    if (self.layer_idx < sparse_layer_start or is_prefill) and not is_prefill_selection_layer:
        # non-sparse layers or non-selection prefilling
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        # Return 3-tuple to match the other path
        return attn_output, None, None
    
    else:
        # This block now handles:
        # 1. Decoding (is_prefill=False)
        # 2. Prefill Selection Layers (is_prefill_selection_layer=True)
        
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        )

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        last_dim_size = attn_weights.size(-1)  # total sequence length
        token_budget = min(last_dim_size, top_k)

        # Check if this is a decoding selection layer
        is_decoding_selection_layer = (
            not is_prefill and
            (self.layer_idx == sparse_layer_start or self.layer_idx == correction_layer)
        )

        if is_prefill_selection_layer:
            # --- NEW PREFILL SELECTION LOGIC ---
            # We want top_k from the entire kv_seq_len for all queries in the block
            # attn_weights shape: (bsz, num_heads, q_len, kv_seq_len)

            # Aggregate scores: max score for each key across all queries
            agg_scores = torch.max(attn_weights, dim=2)[0]  # (bsz, num_heads, kv_seq_len)
            
            # Get top-k indices per head
            _, top_k_indices = torch.topk(agg_scores, k=token_budget, dim=-1) # (bsz, num_heads, token_budget)
            
            # Union indices across all heads (for the first batch item)
            union_tensor = top_k_indices.transpose(0, 1).contiguous().view(bsz, -1) # (bsz, num_heads * token_budget)
            union_list = sorted(list(dict.fromkeys(union_tensor[0].tolist())))
            
            chosen_tokens = union_list # This is the list of KV indices
            
            # Create and store the mask for subsequent sparse layers
            # This mask is based on the *unioned* list
            top_k_indices_gathered = torch.tensor(union_list, device=attn_weights.device, dtype=torch.long)
            top_k_indices_gathered = top_k_indices_gathered.view(1, 1, 1, -1).expand(bsz, num_heads, q_len, len(union_list))
            
            top_k_mask = torch.zeros_like(attn_weights, dtype=torch.bool).scatter_(-1, top_k_indices_gathered, True)
            self.pos_mask = top_k_mask
            self.pos_index = top_k_indices_gathered
            
            # For prefill, we've identified tokens, but we must
            # compute the full attention output for this block.
            # We *don't* apply the mask to attn_weights for the output calculation.
            
        elif is_decoding_selection_layer:
            # --- ORIGINAL DECODING SELECTION LOGIC ---
            # q_len is 1 here
            middle_budget = int(token_budget * (1 - lim_ratio_factor))  # top-k
            most_recent_amount = token_budget - middle_budget  # window attention

            if most_recent_amount < attention_sink:
                attention_sink = 0
            else:
                most_recent_amount -= attention_sink

            assert middle_budget + attention_sink + most_recent_amount == token_budget

            # get sink token indices
            sink_indices = torch.arange(attention_sink, device=attn_weights.device)
            sink_indices = sink_indices.expand(attn_weights.shape[:-1] + (attention_sink,))

            # do top-k selection from the middle tokens
            recent_start = last_dim_size - most_recent_amount
            middle_scores = attn_weights[..., attention_sink:recent_start]
            _, middle_indices = torch.topk(middle_scores, k=middle_budget, dim=-1)
            middle_indices = middle_indices + attention_sink

            # Union operation capped by token_budget
            union_tensor = middle_indices.transpose(1, 3).contiguous().view(bsz, -1)
            union_list = list(dict.fromkeys(union_tensor[0].tolist()))
            if len(union_list) > middle_budget:
                union_list = union_list[:middle_budget]
            # Store chosen tokens for extraction
            chosen_tokens = union_list

            # Reshape back to proper dimensions
            middle_indices = torch.tensor(
                union_list, dtype=middle_indices.dtype, device=middle_indices.device
            )
            middle_indices = middle_indices.unsqueeze(0).unsqueeze(1).unsqueeze(2)
            middle_indices = middle_indices.expand(bsz, num_heads, q_len, -1)

            # get most recent tokens
            recent_indices = torch.arange(
                recent_start, last_dim_size, device=attn_weights.device
            )
            recent_indices = recent_indices.expand(
                attn_weights.shape[:-1] + (most_recent_amount,)
            )

            # combine indices
            top_k_indices = torch.cat(
                [sink_indices, middle_indices, recent_indices], dim=-1
            )

            top_k_mask = torch.zeros_like(attn_weights, dtype=torch.bool).scatter_(-1, top_k_indices, True)
            self.pos_mask = top_k_mask  # store top_k mask
            self.pos_index = top_k_indices
            
            # Apply mask for decoding selection layers
            min_value = torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights.masked_fill(self.pos_mask == 0, min_value)

        else:
            # --- ORIGINAL DECODING *NON*-SELECTION LOGIC ---
            if not hasattr(self, "pos_mask") or self.pos_mask is None:
                raise ValueError("pos mask should be set up in sparse attn layers")
            min_value = torch.finfo(attn_weights.dtype).min
            
            # self.pos_mask might be (bsz, num_heads, 1, k) but
            # attn_weights is (bsz, num_heads, 1, kv_seq_len)
            # The original .to(device) == 0 was problematic.
            # Let's use the mask as boolean
            mask = self.pos_mask.to(attn_weights.device)
            if mask.shape[-1] != attn_weights.shape[-1]:
                # This happens when pos_mask is from a previous step
                # and doesn't cover the full kv_seq_len.
                # Re-create the mask based on indices.
                mask = torch.zeros_like(attn_weights, dtype=torch.bool).scatter_(
                    -1, self.pos_index.to(attn_weights.device), True
                )
                self.pos_mask = mask

            attn_weights = attn_weights.masked_fill(mask == 0, min_value)

        # --- Common calculation for all paths in this `else` block ---
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = nn.functional.dropout(
            attn_weights,
            p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, num_heads, q_len, head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None, chosen_tokens


def enable_llama_tidal_attention(
    model,
    top_k,
    attn_type="tidal",
    sparse_layer_start=2,
    correction_layer=9,
    attn_sink=0,
    lim_ratio=1,
    **kwargs, # <-- NEW: Accept kwargs
):
    def wrap_forward(module):
        # Store the original forward method
        module.original_forward = module.forward

        def new_tidal_forward(
            hidden_states,
            position_embeddings,
            attention_mask=None,
            past_key_value=None,
            cache_position=None,
            **kwargs: Unpack[FlashAttentionKwargs],
        ):
            # selection_layers will be passed in via **kwargs
            return llama_tidal_attention_forward(
                module,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_value,
                cache_position,
                top_k=top_k,
                sparse_layer_start=sparse_layer_start,
                correction_layer=correction_layer,
                attention_sink=attn_sink,
                lim_ratio_factor=lim_ratio,
                **kwargs, # Pass all kwargs (including selection_layers)
            )

        # This check is for "lim" not "tidal"
        # but the logic is what we need.
        if attn_type == "lim" or attn_type == "tidal":
            module.forward = new_tidal_forward

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_tidal_attention(
                module,
                top_k,
                attn_type,
                sparse_layer_start,
                correction_layer,
                attn_sink,
                lim_ratio,
                **kwargs, # <-- NEW: Pass kwargs recursively
            )

        # Check if this module is a LlamaAttention module
        if "LlamaAttention" in module.__class__.__name__:
            print(f"Applying Tidal/LIM patch to layer {module.layer_idx}: {name}")
            print(f"  - top_k: {top_k}")
            print(f"  - sparse_layer_start (for decoding): {sparse_layer_start}")
            print(f"  - correction_layer (for decoding): {correction_layer}")

            wrap_forward(module)