import torch
from typing import List, Optional
from .base import BaseSelector
from ..accumulator import AttentionScoreAccumulator

class ExactSelector(BaseSelector):
    def __init__(self):
        self.accumulator = AttentionScoreAccumulator()
        self.block_start = 0
        self.block_len = 0
        self.score_history = False
        
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
        self.block_start = prefix_in_kv
        self.block_len = block_len
        self.score_history = score_history
        
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

        if self.score_history:
            filtered_candidates = candidate_indices
            relative_positions = candidate_indices
        else:
            block_start = self.block_start
            block_end = block_start + self.block_len
            relative_positions = []
            filtered_candidates = []
            for idx in candidate_indices:
                if block_start <= idx < block_end:
                    filtered_candidates.append(idx)
                    relative_positions.append(idx - block_start)
            if not relative_positions:
                return []

        cand_tensor = torch.tensor(relative_positions, device=full_scores.device, dtype=torch.long)
        candidate_scores = full_scores[cand_tensor]

        if budget >= len(filtered_candidates):
            return sorted(filtered_candidates)

        _, top_rel_indices = torch.topk(candidate_scores, k=budget)

        selected_indices = [filtered_candidates[i] for i in top_rel_indices.cpu().tolist()]

        return sorted(selected_indices)
