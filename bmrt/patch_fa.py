import torch
import transformers.modeling_flash_attention_utils as fa_utils
from typing import Optional, Tuple
import contextlib

# Original function reference
_original_prepare_from_posids = fa_utils._prepare_from_posids

def _patched_prepare_from_posids(
    query, key, value, position_ids, *args, **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:
    """
    Patched version of _prepare_from_posids that forces single-sequence behavior
    when batch_size is 1, regardless of gaps in position_ids.
    """
    
    # Check if we have a single sequence (Batch Size == 1)
    # Check position_ids (4th arg)
    if isinstance(position_ids, torch.Tensor) and position_ids.dim() == 2 and position_ids.shape[0] == 1:
        seq_len = position_ids.shape[1]
        
        
        # We manually construct the cumulative sequence lengths for a SINGLE sequence.
        # Format: [0, seq_len]
        cu_seq_lens = torch.tensor([0, seq_len], device=position_ids.device, dtype=torch.int32)
        max_length = seq_len
        
        # Reshape (flatten) q, k, v for Flash Attention Varlen if they are 4D
        # shape: [1, seq, heads, dim] -> [seq, heads, dim]
        if query.dim() == 4:
            query = query.squeeze(0)
        if key.dim() == 4:
            key = key.squeeze(0)
        if value.dim() == 4:
            value = value.squeeze(0)
        
        # Return flattened q, k, v + new lengths
        return query, key, value, (cu_seq_lens, cu_seq_lens), (max_length, max_length)

    # Fallback to original
    return _original_prepare_from_posids(query, key, value, position_ids, *args, **kwargs)

@contextlib.contextmanager
def ApplyFlashAttentionPatch():
    """
    Context manager that temporarily patches transformers.modeling_flash_attention_utils
    to handle disjoint position_ids in single-batch inference.
    """
    try:
        # Apply Patch
        fa_utils._prepare_from_posids = _patched_prepare_from_posids
        yield
    finally:
        # Restore Original
        fa_utils._prepare_from_posids = _original_prepare_from_posids