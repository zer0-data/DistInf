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
        self.prev_local_tail_len = 0
        
    def setup(self, model):
        self.accumulator.wrap_model(model)
        
    def cleanup(self):
        self.accumulator.unwrap_model(None) 
        
    def prepare_block(self, total_seq_len, block_len, query_len, prefix_len, prefix_in_kv, score_history=False, prev_local_tail_len=0):
        self.accumulator.start_block_with_prefix(
            total_seq_len=total_seq_len,
            prefix_len=prefix_len,
            block_len=block_len,
            query_len=query_len,
            prefix_in_kv_cache=prefix_in_kv,
            score_history=score_history,
            prev_local_tail_len=prev_local_tail_len
        )
        # In accumulate mode, block_start covers the previous tail so that
        # candidates from the tail (absolute indices starting at prefix_in_kv - prev_local_tail_len)
        # are included in the scorable range. block_len is extended to cover both tail + new block.
        self.block_start = prefix_in_kv - prev_local_tail_len
        self.block_len = block_len + prev_local_tail_len
        self.score_history = score_history
        self.prev_local_tail_len = prev_local_tail_len
        
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
            # In accumulate mode, block_start = prefix_cache_len (start of prev tail)
            # and block_end = prefix_cache_len + prev_local_tail_len + block_len
            # This ensures both previous tail tokens and current block tokens are scorable.
            block_start = self.block_start
            block_end = block_start + self.block_len
            relative_positions = []
            filtered_candidates = []
            for idx in candidate_indices:
                if block_start <= idx < block_end:
                    filtered_candidates.append(idx)
                    # Relative position in the scored slice:
                    # - Tail tokens: idx - block_start gives 0..prev_local_tail_len-1
                    # - Block tokens: idx - block_start gives prev_local_tail_len..prev_local_tail_len+block_len-1
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
