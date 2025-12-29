from typing import List, Optional
import torch
from .base import BaseSelector

class HybridSelector(BaseSelector):
    """
    Combines two selection strategies (Primary and Secondary).
    
    Budget is split according to `primary_ratio`.
    Selections are mutually exclusive: items selected by Primary are removed 
    from the candidate pool for Secondary.
    """
    def __init__(
        self, 
        primary_selector: BaseSelector, 
        secondary_selector: BaseSelector, 
        primary_ratio: float = 0.5
    ):
        self.primary = primary_selector
        self.secondary = secondary_selector
        self.primary_ratio = primary_ratio
        
        if not (0.0 <= primary_ratio <= 1.0):
            raise ValueError(f"primary_ratio must be between 0.0 and 1.0, got {primary_ratio}")

    def setup(self, model):
        self.primary.setup(model)
        self.secondary.setup(model)

    def prepare_block(self, **kwargs):
        if hasattr(self.primary, 'prepare_block'):
            self.primary.prepare_block(**kwargs)
        if hasattr(self.secondary, 'prepare_block'):
            self.secondary.prepare_block(**kwargs)

    def select(
        self,
        query_ids: torch.Tensor,
        query_vectors: torch.Tensor,
        candidate_vectors: torch.Tensor,
        candidate_indices: List[int],
        budget: int,
        **kwargs
    ) -> List[int]:
        
        if budget <= 0:
            return []
            
        # 1. Budget Split
        primary_budget = int(budget * self.primary_ratio)
        secondary_budget = budget - primary_budget
        
        selected_primary = []
        
        # 2. Primary Selection
        if primary_budget > 0:
            selected_primary = self.primary.select(
                query_ids=query_ids,
                query_vectors=query_vectors,
                candidate_vectors=candidate_vectors,
                candidate_indices=candidate_indices,
                budget=primary_budget,
                **kwargs
            )
        
        # 3. Filter Candidates for Secondary
        # Identify remaining indices and exclude them from specific tensor operations
        selected_set = set(selected_primary)
        remaining_indices = [idx for idx in candidate_indices if idx not in selected_set]
        
        selected_secondary = []
        if secondary_budget > 0 and remaining_indices:
            # Re-align vectors
            # Assumption: candidate_vectors[i] corresponds to candidate_indices[i].
            keep_mask = [idx not in selected_set for idx in candidate_indices]
            keep_mask_tensor = torch.tensor(keep_mask, device=candidate_vectors.device)
            
            if keep_mask_tensor.any():
                remaining_vectors = candidate_vectors[keep_mask_tensor]
                
                selected_secondary = self.secondary.select(
                    query_ids=query_ids,
                    query_vectors=query_vectors,
                    candidate_vectors=remaining_vectors,
                    candidate_indices=remaining_indices,
                    budget=secondary_budget,
                    **kwargs
                )
        
        # 4. Merge
        return sorted(list(set(selected_primary) | set(selected_secondary)))

    def finish_block(self):
        if hasattr(self.primary, 'finish_block'):
            self.primary.finish_block()
        if hasattr(self.secondary, 'finish_block'):
            self.secondary.finish_block()

    def cleanup(self):
        self.primary.cleanup()
        self.secondary.cleanup()
