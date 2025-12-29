import torch
from typing import List, Optional
from .base import BaseSelector
from ..accumulator import AttentionScoreAccumulator

class ExactSelector(BaseSelector):
    def __init__(self):
        self.accumulator = AttentionScoreAccumulator()
        
    def setup(self, model):
        self.accumulator.wrap_model(model)
        
    def cleanup(self):
        self.accumulator.unwrap_model(None) 
        
    def prepare_block(self, total_seq_len, block_len, query_len, prefix_len, prefix_in_kv, score_history=False):
        self.accumulator.start_block_with_prefix(
            total_seq_len=total_seq_len,
            prefix_len=prefix_len,
            block_len=block_len,
            query_len=query_len,
            prefix_in_kv_cache=prefix_in_kv,
            score_history=score_history
        )
        
    def finish_block(self):
        self.accumulator.finish_block()

    def select(
        self,
        query_ids: torch.Tensor,
        query_vectors: torch.Tensor,
        candidate_vectors: torch.Tensor,
        candidate_indices: List[int],
        budget: int,
        **kwargs
    ) -> List[int]:
        
        if self.accumulator.accumulated_scores is None:
            return []
            
        full_scores = self.accumulator.accumulated_scores[0]
        
        # Filter scores for candidates
        cand_tensor = torch.tensor(candidate_indices, device=full_scores.device, dtype=torch.long)
        candidate_scores = full_scores[cand_tensor]
        
        if budget >= len(candidate_indices):
            return candidate_indices
            
        _, top_rel_indices = torch.topk(candidate_scores, k=budget)
        
        selected_indices = [candidate_indices[i] for i in top_rel_indices.cpu().tolist()]
        
        return sorted(selected_indices)
