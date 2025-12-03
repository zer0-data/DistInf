# tidal_attention.py
# Consolidated module for Tidal/LIM attention with top-K token sampling
# This replaces the src/ folder and its submodules

import math
import time
import types
import hashlib
import json
import os
from typing import Optional, Tuple, List, Dict, Set
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs


# =============================================================================
# GLOBAL ATTENTION SCORE ACCUMULATOR
# =============================================================================

class AttentionScoreAccumulator:
    """
    Accumulates attention scores across all layers during block-wise prefill.
    Performs top-K selection based on the sum of attention scores from all layers.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the accumulator for a new block."""
        self.accumulated_scores: Optional[torch.Tensor] = None
        self.layer_count: int = 0
        self.block_position_ids: Optional[torch.Tensor] = None
        self.is_active: bool = False
    
    def start_block(self, position_ids: torch.Tensor):
        """Start accumulating scores for a new block."""
        self.reset()
        self.block_position_ids = position_ids.clone()
        self.is_active = True
    
    def accumulate(self, attn_weights: torch.Tensor, layer_idx: int):
        """
        Add attention scores from a layer to the accumulator.
        
        Args:
            attn_weights: Attention weights of shape (bsz, num_heads, q_len, kv_seq_len)
            layer_idx: The layer index
        """
        if not self.is_active:
            return
        
        # Aggregate scores: sum across query positions and heads
        # attn_weights shape: (bsz, num_heads, q_len, kv_seq_len)
        # We want to get importance score for each key position
        
        # Sum across heads and query positions to get per-key-position score
        # Shape: (bsz, kv_seq_len)
        layer_scores = attn_weights.sum(dim=(1, 2))  # Sum over heads and queries
        
        if self.accumulated_scores is None:
            self.accumulated_scores = layer_scores.clone()
        else:
            # Handle case where kv_seq_len might differ (shouldn't happen in prefill)
            if self.accumulated_scores.shape[-1] == layer_scores.shape[-1]:
                self.accumulated_scores = self.accumulated_scores + layer_scores
            else:
                # Extend accumulated scores if needed
                min_len = min(self.accumulated_scores.shape[-1], layer_scores.shape[-1])
                self.accumulated_scores[..., :min_len] += layer_scores[..., :min_len]
        
        self.layer_count += 1
    
    def select_top_k(self, top_k: int, device: torch.device) -> Tuple[List[int], List[int]]:
        """
        Select top-K tokens based on accumulated attention scores.
        
        Args:
            top_k: Number of tokens to select
            device: Device for tensor operations
            
        Returns:
            Tuple of (selected_indices, selected_position_ids)
        """
        if self.accumulated_scores is None or self.block_position_ids is None:
            return [], []
        
        # Get the actual token budget (can't exceed sequence length)
        seq_len = self.accumulated_scores.shape[-1]
        token_budget = min(seq_len, top_k)
        
        # Select top-K indices based on accumulated scores
        # Shape of accumulated_scores: (bsz, kv_seq_len)
        _, top_k_indices = torch.topk(self.accumulated_scores[0], k=token_budget)
        
        # Sort indices to maintain sequence order
        top_k_indices_sorted, _ = torch.sort(top_k_indices)
        selected_indices = top_k_indices_sorted.tolist()
        
        # Get corresponding position IDs
        if self.block_position_ids.dim() == 2:
            pos_ids = self.block_position_ids[0]  # Remove batch dim
        else:
            pos_ids = self.block_position_ids
            
        selected_position_ids = [pos_ids[idx].item() for idx in selected_indices]
        
        return selected_indices, selected_position_ids
    
    def finish_block(self) -> None:
        """
        Clean up after finishing a block.
        Explicitly delete the accumulated scores tensor to free memory.
        """
        if self.accumulated_scores is not None:
            del self.accumulated_scores
        self.accumulated_scores = None
        self.block_position_ids = None
        self.layer_count = 0
        self.is_active = False
        # Force garbage collection for the tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# Global accumulator instance (will be attached to models)
_global_accumulator: Optional[AttentionScoreAccumulator] = None


def get_or_create_accumulator(model) -> AttentionScoreAccumulator:
    """Get or create an attention score accumulator for a model."""
    if not hasattr(model, '_tidal_accumulator'):
        model._tidal_accumulator = AttentionScoreAccumulator()
    return model._tidal_accumulator


# =============================================================================
# TIDAL ATTENTION FORWARD FUNCTION
# =============================================================================

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
    """
    Modified attention forward pass with top-K token sampling.
    
    This function replaces the standard attention forward pass with one that:
    1. Uses fast SDPA for non-sparse layers during decoding
    2. Accumulates attention scores across ALL layers during prefill
    3. Performs top-K token selection at the final layer based on summed scores
    4. Applies sparse attention during decoding based on previously selected tokens
    """
    output_attentions = kwargs.get("output_attentions", False)

    # If output_attentions is requested, fall back to original implementation
    if output_attentions:
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
    
    # Get accumulator and config from kwargs
    accumulator: Optional[AttentionScoreAccumulator] = kwargs.get("_tidal_accumulator", None)
    num_layers: int = kwargs.get("_num_layers", 32)  # Default for Llama
    final_selection_layer: int = kwargs.get("_final_selection_layer", num_layers - 1)
    
    chosen_tokens = None  # Will hold (indices, position_ids) tuple

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

    # Determine if this is a prefill pass
    is_prefill = q_len > 1 and (q_len == kv_seq_len)
    
    # Check if we should accumulate scores (during prefill when accumulator is active)
    should_accumulate = (
        is_prefill
        and accumulator is not None
        and accumulator.is_active
        and top_k is not None
    )
    
    # Check if this is the final layer where we perform selection
    is_final_selection_layer = (
        is_prefill
        and self.layer_idx == final_selection_layer
        and accumulator is not None
        and accumulator.is_active
    )
    
    # During prefill, we need to compute attention weights for accumulation
    if should_accumulate:
        # Compute attention weights for accumulation
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        )

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        # Apply softmax to get proper attention probabilities for accumulation
        attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        
        # Accumulate the attention scores
        accumulator.accumulate(attn_probs, self.layer_idx)
        
        # If this is the final layer, perform top-K selection
        if is_final_selection_layer:
            selected_indices, selected_position_ids = accumulator.select_top_k(
                top_k, attn_weights.device
            )
            chosen_tokens = (selected_indices, selected_position_ids)
            
            # Store the selection result on the root model for retrieval
            # We need to find the root model to store this
            root_model = kwargs.get("_root_model", None)
            if root_model is not None:
                root_model._last_block_selection = (selected_indices, selected_position_ids)
            
            # Clean up the accumulator for this block
            accumulator.finish_block()
        
        # Still compute output using standard attention (full attention during prefill)
        attn_probs_typed = attn_probs.to(query_states.dtype)
        attn_probs_typed = nn.functional.dropout(
            attn_probs_typed,
            p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        )
        attn_output = torch.matmul(attn_probs_typed, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, chosen_tokens
    
    # Non-prefill path or prefill without accumulation
    # Use fast SDPA for non-sparse layers during decoding
    if self.layer_idx < sparse_layer_start and not is_prefill:
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
        return attn_output, None, None
    
    # Standard prefill without accumulation (accumulator not active)
    if is_prefill and not should_accumulate:
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
        return attn_output, None, None
    
    # Decoding path with sparse attention
    attn_weights = (
        torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    )

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    last_dim_size = attn_weights.size(-1)
    token_budget = min(last_dim_size, top_k) if top_k else last_dim_size

    # Check if this is a decoding selection layer
    is_decoding_selection_layer = (
        not is_prefill and
        (self.layer_idx == sparse_layer_start or self.layer_idx == correction_layer)
    )

    if is_decoding_selection_layer:
        # --- DECODING SELECTION LOGIC ---
        middle_budget = int(token_budget * (1 - lim_ratio_factor))
        most_recent_amount = token_budget - middle_budget

        if most_recent_amount < attention_sink:
            attention_sink = 0
        else:
            most_recent_amount -= attention_sink

        assert middle_budget + attention_sink + most_recent_amount == token_budget

        sink_indices = torch.arange(attention_sink, device=attn_weights.device)
        sink_indices = sink_indices.expand(attn_weights.shape[:-1] + (attention_sink,))

        recent_start = last_dim_size - most_recent_amount
        middle_scores = attn_weights[..., attention_sink:recent_start]
        _, middle_indices = torch.topk(middle_scores, k=middle_budget, dim=-1)
        middle_indices = middle_indices + attention_sink

        union_tensor = middle_indices.transpose(1, 3).contiguous().view(bsz, -1)
        union_list = list(dict.fromkeys(union_tensor[0].tolist()))
        if len(union_list) > middle_budget:
            union_list = union_list[:middle_budget]
        chosen_tokens = union_list

        middle_indices = torch.tensor(
            union_list, dtype=middle_indices.dtype, device=middle_indices.device
        )
        middle_indices = middle_indices.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        middle_indices = middle_indices.expand(bsz, num_heads, q_len, -1)

        recent_indices = torch.arange(
            recent_start, last_dim_size, device=attn_weights.device
        )
        recent_indices = recent_indices.expand(
            attn_weights.shape[:-1] + (most_recent_amount,)
        )

        top_k_indices = torch.cat(
            [sink_indices, middle_indices, recent_indices], dim=-1
        )

        top_k_mask = torch.zeros_like(attn_weights, dtype=torch.bool).scatter_(-1, top_k_indices, True)
        self.pos_mask = top_k_mask
        self.pos_index = top_k_indices
        
        min_value = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(self.pos_mask == 0, min_value)

    elif not is_prefill:
        # --- DECODING NON-SELECTION LOGIC ---
        if not hasattr(self, "pos_mask") or self.pos_mask is None:
            raise ValueError("pos mask should be set up in sparse attn layers")
        min_value = torch.finfo(attn_weights.dtype).min
        
        mask = self.pos_mask.to(attn_weights.device)
        if mask.shape[-1] != attn_weights.shape[-1]:
            mask = torch.zeros_like(attn_weights, dtype=torch.bool).scatter_(
                -1, self.pos_index.to(attn_weights.device), True
            )
            self.pos_mask = mask

        attn_weights = attn_weights.masked_fill(mask == 0, min_value)

    # Common output calculation
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


# =============================================================================
# ENABLE TIDAL ATTENTION ON LLAMA MODELS
# =============================================================================

def enable_llama_tidal_attention(
    model,
    top_k,
    attn_type="tidal",
    sparse_layer_start=2,
    correction_layer=9,
    attn_sink=0,
    lim_ratio=1,
    num_layers=32,
    root_model=None,  # Track root model for storing selection results
    **kwargs,
):
    """
    Patches all LlamaAttention modules in the model to use Tidal attention.
    
    Args:
        model: The model to patch
        top_k: Number of tokens to select in sparse attention
        attn_type: Type of attention ("tidal" or "lim")
        sparse_layer_start: First layer to apply sparse attention
        correction_layer: Layer at which to re-select tokens during decoding
        attn_sink: Number of initial tokens to always attend to
        lim_ratio: Ratio for local/global attention split
        num_layers: Total number of layers in the model
        root_model: The root model (for storing selection results)
        **kwargs: Additional arguments
    """
    # Use the passed root_model, or default to model if not provided
    if root_model is None:
        root_model = model
        
    # Get or create the accumulator for this model
    accumulator = get_or_create_accumulator(root_model)
    final_selection_layer = num_layers - 1
    
    def wrap_forward(module, _root_model=root_model):
        # Store the original forward method
        module.original_forward = module.forward

        def new_tidal_forward(
            hidden_states,
            position_embeddings,
            attention_mask=None,
            past_key_value=None,
            cache_position=None,
            **fwd_kwargs: Unpack[FlashAttentionKwargs],
        ):
            # Pass accumulator, config, and root model via kwargs
            fwd_kwargs['_tidal_accumulator'] = accumulator
            fwd_kwargs['_num_layers'] = num_layers
            fwd_kwargs['_final_selection_layer'] = final_selection_layer
            fwd_kwargs['_root_model'] = _root_model
            
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
                **fwd_kwargs,
            )

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
                num_layers,
                root_model=root_model,  # Pass root_model down recursively
                **kwargs,
            )

        # Check if this module is a LlamaAttention module
        if "LlamaAttention" in module.__class__.__name__:
            print(f"Applying Tidal/LIM patch to layer {module.layer_idx}: {name}")
            print(f"  - top_k: {top_k}")
            print(f"  - sparse_layer_start (for decoding): {sparse_layer_start}")
            print(f"  - correction_layer (for decoding): {correction_layer}")
            print(f"  - final_selection_layer: {final_selection_layer}")
            wrap_forward(module)


# =============================================================================
# MAIN ENABLE TIDAL FUNCTION
# =============================================================================

def enable_tidal(
    model,
    attn_type="tidal",
    top_k=256,
    sparse_layer_start=2,
    correction_layer=13,
    attention_sink=0,
    lim_ratio=1,
    **kwargs,
):
    """
    Enable Tidal/LIM attention on a model.
    
    Args:
        model: The model to patch
        attn_type: Type of attention ("tidal" or "lim")
        top_k: Number of tokens to select in sparse attention
        sparse_layer_start: First layer to apply sparse attention
        correction_layer: Layer at which to re-select tokens during decoding
        attention_sink: Number of initial tokens to always attend to
        lim_ratio: Ratio for local/global attention split
        **kwargs: Additional arguments
    """
    if attn_type == "lim" or attn_type == "tidal":
        print(f"Tidal/LIM Enabled: attention_sink: {attention_sink}")
        print(f"token budget: {top_k}")
        print(f"sparse layer starts from: Layer {sparse_layer_start}")
        print(f"reselection layer: {correction_layer}")
        
        # Detect number of layers from model config
        num_layers = getattr(model.config, 'num_hidden_layers', 32)
        print(f"Number of layers detected: {num_layers}")
        print(f"Accumulating attention scores across ALL layers, selecting at final layer ({num_layers - 1})")
            
        model_type = model.config.model_type

        if "llama" in model_type:
            # Create the accumulator for this model
            accumulator = get_or_create_accumulator(model)
            
            enable_llama_tidal_attention(
                model,
                top_k,
                attn_type,
                sparse_layer_start,
                correction_layer,
                attention_sink,
                lim_ratio,
                num_layers=num_layers,
                root_model=model,  # Pass the root model for storing selection results
                **kwargs,
            )
        else:
            print(f"Warning: Model type '{model_type}' is not supported for Tidal attention.")
    return


# =============================================================================
# HELPER FUNCTIONS FOR BLOCK-WISE PREFILL
# =============================================================================

def start_block_accumulation(model, position_ids: torch.Tensor):
    """
    Start accumulating attention scores for a new block during prefill.
    Call this before processing each block.
    
    Args:
        model: The model with Tidal attention enabled
        position_ids: Position IDs for the current block
    """
    accumulator = get_or_create_accumulator(model)
    accumulator.start_block(position_ids)
    print(f"[Tidal] Started block accumulation for positions {position_ids[0, 0].item()} to {position_ids[0, -1].item()}")


def get_block_selected_tokens(model) -> Tuple[List[int], List[int]]:
    """
    Get the selected tokens from the last completed block.
    The accumulator automatically performs selection at the final layer.
    
    Args:
        model: The model with Tidal attention enabled
        
    Returns:
        Tuple of (selected_indices, selected_position_ids)
        Returns empty lists if no selection was made.
    """
    # The selection happens automatically in the attention forward pass
    # at the final layer. This function is for retrieving results if needed.
    accumulator = get_or_create_accumulator(model)
    # After finish_block() is called, the data is cleared
    # So we need a different approach - store the last results
    if hasattr(model, '_last_selected_tokens'):
        return model._last_selected_tokens
    return [], []


def finish_block_accumulation(model):
    """
    Explicitly finish block accumulation and clean up.
    Note: This is typically called automatically at the final layer.
    
    Args:
        model: The model with Tidal attention enabled
    """
    accumulator = get_or_create_accumulator(model)
    accumulator.finish_block()


# =============================================================================
# HISTORY TRACKING MIXIN FOR TOP-K TOKEN SELECTION
# =============================================================================

class TidalHistoryMixin:
    """
    Mixin class that provides history tracking methods for top-K token selection.
    These methods can be added to any model to track which tokens were selected
    during prefill/decoding.
    """
    
    def init_tidal_history(self):
        """Initialize the history tracking attributes."""
        self.global_top_tokens_history = {}
        self.current_sequence_id = None
        self.current_input_text = None
        self._tokenizer_for_decode = None

    def set_tokenizer_for_decode(self, tokenizer):
        """
        Set the tokenizer reference for automatic text decoding.

        Args:
            tokenizer: The tokenizer instance used for decoding input_ids to text
        """
        self._tokenizer_for_decode = tokenizer

    def get_top_tokens_history(self):
        """
        Get the complete history of top tokens for all layers across all generation steps.

        Returns:
            dict: {generation_step: {layer_idx: top_tokens}}
        """
        return dict(self.global_top_tokens_history)

    def get_top_tokens_for_step(self, generation_step):
        """
        Get top tokens for all layers at a specific generation step.

        Args:
            generation_step (int): The generation step to retrieve

        Returns:
            dict: {layer_idx: top_tokens} for the specified step, or None if step doesn't exist
        """
        return self.global_top_tokens_history.get(generation_step, None)

    def clear_top_tokens_history(self):
        """
        Clear the global top tokens history and reset sequence context.
        """
        self.global_top_tokens_history.clear()
        self.current_sequence_id = None
        self.current_input_text = None

    def get_generation_steps_count(self):
        """
        Get the number of generation steps recorded.

        Returns:
            int: Number of generation steps
        """
        return len(self.global_top_tokens_history)

    def _generate_sequence_id(self, input_ids, input_text=None):
        """
        Generate a unique identifier for the input sequence.

        Args:
            input_ids (torch.Tensor): The input token IDs
            input_text (str, optional): The original text (if available)

        Returns:
            str: Unique sequence identifier
        """
        # Convert input_ids to a consistent format for hashing
        if input_ids is not None:
            input_tokens = input_ids.detach().cpu().numpy().tolist()
            if isinstance(input_tokens[0], list):
                input_tokens = input_tokens[0]  # Handle batch dimension
        else:
            input_tokens = []

        # Create a deterministic hash from the input tokens
        hash_input = json.dumps(input_tokens, sort_keys=True)
        sequence_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]

        # Add timestamp for uniqueness across sessions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        sequence_id = f"seq_{sequence_hash}_{timestamp}"

        # Store the input text if provided, or try to decode from input_ids if not
        if input_text:
            self.current_input_text = input_text
        elif input_ids is not None and self._tokenizer_for_decode is not None:
            try:
                decoded_text = self._tokenizer_for_decode.decode(
                    input_ids[0] if len(input_ids.shape) > 1 else input_ids,
                    skip_special_tokens=True,
                )
                self.current_input_text = (
                    decoded_text[:500] + "..."
                    if len(decoded_text) > 500
                    else decoded_text
                )
            except Exception:
                self.current_input_text = f"<Decoded from {len(input_tokens)} tokens>"
        else:
            self.current_input_text = f"<Input with {len(input_tokens)} tokens>"

        return sequence_id

    def set_sequence_context(self, input_ids, input_text=None):
        """
        Set the sequence context for tracking top tokens.
        This should be called before generation starts.

        Args:
            input_ids (torch.Tensor): The input token IDs
            input_text (str, optional): The original input text
        """
        # Clear previous history AND context
        self.clear_top_tokens_history()
        # Now set the new context
        self.current_sequence_id = self._generate_sequence_id(input_ids, input_text)

    def save_top_tokens_history(self, output_dir="./top_tokens_logs", filename=None):
        """
        Save the complete top tokens history to a JSON file with sequence identification.

        Args:
            output_dir (str): Directory to save the history file
            filename (str, optional): Custom filename. If None, auto-generated.

        Returns:
            str: Path to the saved file
        """
        if not self.global_top_tokens_history:
            print("No top tokens history to save.")
            return None

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename if not provided
        if filename is None:
            if self.current_sequence_id:
                filename = f"top_tokens_{self.current_sequence_id}.json"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"top_tokens_unknown_seq_{timestamp}.json"

        filepath = os.path.join(output_dir, filename)

        # Prepare data structure for saving
        save_data = {
            "sequence_id": self.current_sequence_id,
            "input_text": self.current_input_text,
            "timestamp": datetime.now().isoformat(),
            "total_generation_steps": len(self.global_top_tokens_history),
            "num_layers": len(next(iter(self.global_top_tokens_history.values()), {})),
            "top_tokens_history": {},
        }

        # Convert tensor data to serializable format
        for gen_step, layer_data in self.global_top_tokens_history.items():
            save_data["top_tokens_history"][str(gen_step)] = {}
            for layer_idx, top_tokens in layer_data.items():
                if isinstance(top_tokens, torch.Tensor):
                    serializable_tokens = top_tokens.detach().cpu().tolist()
                elif isinstance(top_tokens, (list, tuple)):
                    serializable_tokens = list(top_tokens)
                else:
                    serializable_tokens = str(top_tokens)

                save_data["top_tokens_history"][str(gen_step)][
                    str(layer_idx)
                ] = serializable_tokens

        # Save to file
        try:
            with open(filepath, "w") as f:
                json.dump(save_data, f, indent=2)
            print(f"Top tokens history saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving top tokens history: {e}")
            return None

    @staticmethod
    def load_top_tokens_history(filepath):
        """
        Load top tokens history from a saved JSON file.

        Args:
            filepath (str): Path to the saved history file

        Returns:
            dict: Loaded top tokens history data
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            print(f"Top tokens history loaded from: {filepath}")
            print(f"Sequence ID: {data.get('sequence_id', 'Unknown')}")
            print(f"Generation steps: {data.get('total_generation_steps', 0)}")
            return data
        except Exception as e:
            print(f"Error loading top tokens history: {e}")
            return None


# =============================================================================
# UTILITY FUNCTION TO ADD HISTORY TRACKING TO ANY MODEL
# =============================================================================

def add_tidal_history_tracking(model):
    """
    Add Tidal history tracking methods to a model instance.
    
    This function dynamically adds the methods from TidalHistoryMixin to a model
    instance, allowing it to track top-K token selections during prefill/decoding.
    
    Args:
        model: The model instance to add history tracking to
    
    Returns:
        The model with history tracking methods added
    """
    mixin = TidalHistoryMixin()
    
    # Initialize history attributes on the model
    model.global_top_tokens_history = {}
    model.current_sequence_id = None
    model.current_input_text = None
    model._tokenizer_for_decode = None
    
    # Bind methods from the mixin to the model
    model.set_tokenizer_for_decode = types.MethodType(mixin.set_tokenizer_for_decode.__func__, model)
    model.get_top_tokens_history = types.MethodType(mixin.get_top_tokens_history.__func__, model)
    model.get_top_tokens_for_step = types.MethodType(mixin.get_top_tokens_for_step.__func__, model)
    model.clear_top_tokens_history = types.MethodType(mixin.clear_top_tokens_history.__func__, model)
    model.get_generation_steps_count = types.MethodType(mixin.get_generation_steps_count.__func__, model)
    model._generate_sequence_id = types.MethodType(mixin._generate_sequence_id.__func__, model)
    model.set_sequence_context = types.MethodType(mixin.set_sequence_context.__func__, model)
    model.save_top_tokens_history = types.MethodType(mixin.save_top_tokens_history.__func__, model)
    
    return model


def add_tidal_history_tracking_to_inner_model(model):
    """
    Add Tidal history tracking to the inner model (model.model) as well.
    This is needed for models that wrap another model internally.
    
    Args:
        model: The outer model (e.g., LlamaForCausalLM)
    
    Returns:
        The model with history tracking added to both outer and inner models
    """
    # Add to outer model
    add_tidal_history_tracking(model)
    
    # Add to inner model if it exists
    if hasattr(model, 'model'):
        add_tidal_history_tracking(model.model)
    
    return model


# =============================================================================
# LOAD FUNCTION FOR TIDAL-ENABLED MODELS
# =============================================================================

def load_tidal_model(model_name_or_path, attn_type, **kwargs):
    """
    Load a model with Tidal/LIM attention enabled.
    
    Args:
        model_name_or_path: Path to the model or model name
        attn_type: Type of attention ("tidal", "lim", or other for standard)
        **kwargs: Additional arguments including top_k, selection_layers, etc.
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model from {model_name_or_path} ...")
    print(f"attn_type: {attn_type}")

    if attn_type == "tidal" or attn_type == "lim":
        print("sparse attention (tidal or lim) enabled!")
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # Add history tracking
        add_tidal_history_tracking_to_inner_model(model)
        
        # Enable tidal attention
        enable_tidal(model, attn_type, **kwargs)
    else:
        # flash attention / standard
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("full-weight attention enabled")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    
    print(f"Loaded Model: {model.__class__.__name__}")
    
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer
