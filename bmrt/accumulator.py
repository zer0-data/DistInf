from typing import Optional, List
import torch
import torch.nn as nn

class AttentionScoreAccumulator:
    """
    Accumulates attention scores across all layers during block-wise prefill.
    Performs top-K selection based on the sum of attention scores from all layers.
    """
    
    def __init__(self):
        self.reset()
        self._hooks = []
    
    def reset(self):
        self.accumulated_scores: Optional[torch.Tensor] = None
        self.layer_count: int = 0
        self.is_active: bool = False
        self.block_token_count: int = 0
        self.total_seq_len: int = 0
        self.prefix_len: int = 0
        self.query_len: int = 0
        self.expected_q_len: int = 0
        self.prefix_in_kv_cache: int = 0
    
    def start_block_with_prefix(
        self, 
        total_seq_len: int, 
        prefix_len: int,
        block_len: int,
        query_len: int,
        prefix_in_kv_cache: int = 0,
        score_history: bool = False,
    ):
        self.reset()
        self.is_active = True
        self.total_seq_len = total_seq_len
        self.prefix_len = prefix_len
        self.block_token_count = block_len
        self.query_len = query_len
        self.prefix_in_kv_cache = prefix_in_kv_cache
        self.expected_q_len = total_seq_len - prefix_in_kv_cache
        self.score_history = score_history

    def accumulate(self, attn_weights: torch.Tensor, layer_idx: int):
        if not self.is_active:
            return
        
        if attn_weights.dim() != 4:
             return
        
        bsz, num_heads, q_len, kv_seq_len = attn_weights.shape
        if bsz != 1: return
        
        if q_len != self.expected_q_len: return
        
        expected_kv_len = self.prefix_in_kv_cache + self.block_token_count + self.query_len
        if kv_seq_len < expected_kv_len: return
        
        # Determine the Key range to score
        if self.score_history:
            # Score everything from 0 up to end of block
            block_start_in_kv = 0
            block_end_in_kv = self.prefix_in_kv_cache + self.block_token_count
        else:
            # Score only the current block (Standard)
            block_start_in_kv = self.prefix_in_kv_cache
            block_end_in_kv = self.prefix_in_kv_cache + self.block_token_count
        
        if self.query_len > 0:
            query_start_in_q = q_len - self.query_len
            layer_scores = attn_weights[:, :, query_start_in_q:, block_start_in_kv:block_end_in_kv].sum(dim=(1, 2))
        else:
            layer_scores = attn_weights[:, :, :, block_start_in_kv:block_end_in_kv].sum(dim=(1, 2))
        
        if self.accumulated_scores is None:
            self.accumulated_scores = layer_scores
        else:
            if self.accumulated_scores.shape[-1] != layer_scores.shape[-1]:
                self.accumulated_scores = layer_scores
            else:
                self.accumulated_scores.add_(layer_scores)
        
        self.layer_count += 1

    def finish_block(self) -> None:
        self.accumulated_scores = None
        self.layer_count = 0
        self.block_token_count = 0
        self.is_active = False

    def wrap_model(self, model):
        """Replaces attention modules with AttentionWrapper."""
        self.unwrap_model(model)
        
        attention_modules = {}
        for name, module in model.named_modules():
            if any(attn_name in name.lower() for attn_name in ['attention', 'attn']) and \
               not any(skip in name.lower() for skip in ['layernorm', 'ln', 'norm', 'dropout', 'wrapper']):
                if hasattr(module, 'forward') and 'Attention' in type(module).__name__:
                    attention_modules[name] = module
        
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
        
        self._hooks = [] 
        for layer_idx, (name, module) in enumerate(leaf_attention_modules):
            wrapper = AttentionWrapper(module, self, layer_idx)
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[1] if '.' in name else name
            
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                
            setattr(parent, child_name, wrapper)
            self._hooks.append((parent, child_name, module))

    def unwrap_model(self, model):
        if not hasattr(self, '_hooks') or not self._hooks:
            return
        for parent, child_name, original_module in self._hooks:
            setattr(parent, child_name, original_module)
        self._hooks = []


class AttentionWrapper(torch.nn.Module):
    def __init__(self, original_module, accumulator, layer_idx):
        super().__init__()
        self.original_module = original_module
        self.accumulator = accumulator
        self.layer_idx = layer_idx
        
    def forward(self, *args, **kwargs):
        kwargs['output_attentions'] = True
        outputs = self.original_module(*args, **kwargs)
        
        attn_weights = None
        if isinstance(outputs, tuple):
            for item in outputs:
                if isinstance(item, torch.Tensor) and item.dim() == 4:
                    attn_weights = item
                    break
        
        if attn_weights is not None:
            self.accumulator.accumulate(attn_weights, self.layer_idx)
            
        new_outputs = []
        if isinstance(outputs, tuple):
            for item in outputs:
                if isinstance(item, torch.Tensor) and item.dim() == 4:
                     new_outputs.append(None) 
                else:
                    new_outputs.append(item)
            return tuple(new_outputs)
        return outputs

def get_or_create_accumulator(model) -> AttentionScoreAccumulator:
    if not hasattr(model, '_topk_accumulator'):
        model._topk_accumulator = AttentionScoreAccumulator()
    return model._topk_accumulator
