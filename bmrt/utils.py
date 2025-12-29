import torch
from typing import List, Tuple, Optional

def compute_anchor_local_indices(
    block_len: int, 
    block_start_position: int,
    anchor_size: int,
    local_window_size: int
) -> Tuple[List[int], List[int]]:
    """Determine fixed indices for Anchor and Local tokens in a block."""
    anchor_indices = []
    
    # Anchor Logic
    if block_start_position < anchor_size:
        anchor_end_in_block = min(anchor_size - block_start_position, block_len)
        anchor_indices = list(range(anchor_end_in_block))
    
    # Local Window Logic
    # Indices are relative to the block start
    anchor_end = len(anchor_indices)
    local_start = max(anchor_end, block_len - local_window_size)
    local_indices = list(range(local_start, block_len))
    
    return anchor_indices, local_indices

def extract_kv_for_indices(
    past_key_values: Tuple,
    indices: List[int],
    offset: int = 0,
    device: str = "cuda"
) -> Tuple:
    """Extract KV cache entries for specific indices."""
    absolute_indices = [offset + idx for idx in indices]
    indices_tensor = torch.tensor(absolute_indices, device=device, dtype=torch.long)
    
    extracted_kv = []
    # Loop over layers
    for layer_kv in past_key_values:
        key, value = layer_kv
        # layer_kv is (K, V). K shape: [Batch, Heads, SeqLen, Dim]
        extracted_key = key.index_select(dim=2, index=indices_tensor)
        extracted_value = value.index_select(dim=2, index=indices_tensor)
        extracted_kv.append((extracted_key, extracted_value))
    
    return tuple(extracted_kv)

def merge_kv_caches(
    cache1: Optional[Tuple],
    cache2: Tuple,
) -> Tuple:
    """Concatenate two KV caches along the sequence dimension."""
    if cache1 is None:
        return cache2
    
    merged_kv = []
    for (k1, v1), (k2, v2) in zip(cache1, cache2):
        merged_key = torch.cat([k1, k2], dim=2)
        merged_value = torch.cat([v1, v2], dim=2)
        merged_kv.append((merged_key, merged_value))
    
    return tuple(merged_kv)
