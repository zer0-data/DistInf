import torch
from typing import List, Optional
from .base import BaseSelector

class LSHSelector(BaseSelector):
    """
    Selects tokens using LSH techniques.
    Supports two LSH modes:
    1. 'frequency_rank' (Ours): Sort by collision count (Primary) + Tie-breaker (Secondary).
    2. 'magicpig_baseline': Probabilistic sampling from matching buckets.

    Additionally, a tie-breaking `mode` hyperparameter controls how ties in
    collision counts are resolved. The secondary tie-breaker can use L2 distance
    or no tie-breaking at all.
    """
    
    def __init__(self, head_dim: int, lsh_mode: str = 'frequency_rank', num_bits: int = 12, num_tables: int = 20, device: str = 'cuda', mode: str = 'l2'):
        self.head_dim = head_dim
        self.num_bits = num_bits
        self.num_tables = num_tables
        self.device = device
        self.lsh_mode = lsh_mode
        # Tie-breaking mode (hyperparameter)
        self.mode = mode
        if mode not in ['l2', 'none']:
            raise ValueError(f"Invalid tie-breaking mode: {mode}")
        if lsh_mode not in ['frequency_rank', 'magicpig_baseline']:
            raise ValueError(f"Invalid lsh_mode: {lsh_mode}")

        # Initialize LSH Projections
        # Matrix: [D, num_tables * num_bits]
        self.projection_matrix = torch.randn(
            head_dim, 
            num_tables * num_bits, 
            device=device, 
            dtype=torch.float32  # Will cast during matmul if needed
        )
        
        # Pre-compute powers of two for hashing
        # [1, 1, num_bits]
        self.powers_of_two = (2 ** torch.arange(num_bits, device=device)).view(1, 1, -1)

    def setup(self, model):
        pass
        
    def cleanup(self):
        pass

    def compute_bits(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute raw hash bits for multiple tables.
        
        Args:
            vectors: [N, D]
            
        Returns:
            bits: [N, L, K] (0/1 tensor)
        """
        # 1. Project: [N, D] @ [D, L*K] -> [N, L*K]
        projections = torch.matmul(vectors, self.projection_matrix.to(dtype=vectors.dtype))
        
        # 2. Binarize: [N, L*K] -> 0/1
        bits = (projections > 0).long()
        
        # 3. Reshape separate tables: [N, L, K]
        return bits.view(-1, self.num_tables, self.num_bits)

    def _compute_hashes(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute hash codes for multiple tables.
        
        Args:
            vectors: [N, D]
            
        Returns:
            hashes: [N, Tables] (Int32 hash codes)
        """
        # Reuse compute_bits
        bits = self.compute_bits(vectors) # [N, L, K]
        
        # 4. Pack bits into integer hashes: [N, L, K] * [1, 1, K] -> Sum -> [N, L]
        hashes = (bits * self.powers_of_two).sum(dim=-1)
        
        return hashes.int()

    def compute_hashes(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Public wrapper for computing hash codes. Preserves the original protected
        implementation while exposing a stable API for external callers.
        """
        return self._compute_hashes(vectors)

    def select(
        self,
        query_ids: torch.Tensor,
        query_vectors: torch.Tensor,
        candidate_vectors: torch.Tensor,
        candidate_indices: List[int],
        budget: int,
        **kwargs
    ) -> List[int]:
        
        if budget >= len(candidate_indices):
            return candidate_indices
        if budget <= 0:
            return []

        # 1. Centering & Hashing
        c_mean = candidate_vectors.mean(dim=0, keepdim=True)
        centered_query = query_vectors - c_mean
        centered_candidates = candidate_vectors - c_mean
        
        q_hashes = self.compute_hashes(centered_query)
        c_hashes = self.compute_hashes(centered_candidates)
        
        num_buckets = 2 ** self.num_bits
        
        # 2. Selection Logic
        if self.lsh_mode == 'frequency_rank':
            # Our Method: Collision Counting
            collision_counts = torch.zeros(len(candidate_indices), device=self.device, dtype=torch.int32)
            
            for l in range(self.num_tables):
                q_h = q_hashes[:, l]
                c_h = c_hashes[:, l]
                
                bucket_mask = torch.zeros(num_buckets, device=self.device, dtype=torch.bool)
                bucket_mask.index_fill_(0, q_h.long(), True)
                
                hits = bucket_mask[c_h.long()]
                collision_counts += hits.int()
            
            # --- TIE-BREAKING LOGIC (controlled via `self.mode`) ---
            
            # If mode is 'l2', use L2 (Euclidean) distance to the query
            # as a secondary key (ascending). Otherwise fall back to
            # sorting by collision count only.
            if self.mode == 'l2':
                # Compute L2 distances between the (mean) query vector and candidates.
                # Use the mean of query_vectors if multiple query vectors are present.
                # centered_query: [Q, D] -> take mean -> [D]
                q_vec = centered_query.mean(dim=0, keepdim=True)
                # centered_candidates: [C, D]
                dists = torch.norm(centered_candidates - q_vec, dim=1)

                # 1. Sort by secondary key (L2 distance) ASC
                sorted_by_dist_indices = torch.argsort(dists, descending=False)

                # 2. Re-order counts based on distance sort
                sorted_counts = collision_counts[sorted_by_dist_indices]

                # 3. Sort by primary key (Collision Counts) DESC using stable sort
                sorted_by_count_indices = torch.argsort(sorted_counts, descending=True, stable=True)

                # 4. Combine to get final indices
                final_prioritized_indices = sorted_by_dist_indices[sorted_by_count_indices]
                top_k_indices = final_prioritized_indices[:budget]
            else:
                # Default/fallback: Sort purely by collision count (stable)
                sorted_indices = torch.argsort(collision_counts, descending=True, stable=True)
                top_k_indices = sorted_indices[:budget]
        
        elif self.lsh_mode == 'magicpig_baseline':
            # 1. Compute Hash Bits (Integer form)
            q_buckets = self.compute_hashes(centered_query) 
            c_buckets = self.compute_hashes(centered_candidates)
            
            num_tables = self.num_tables
            num_bits = self.num_bits
            
            # 2. Efficient Bucket Retrieval
            # Offset buckets to make them unique across tables
            bucket_offset = torch.arange(num_tables, device=self.device) * (2 ** num_bits)
            q_global = q_buckets + bucket_offset 
            c_global = c_buckets + bucket_offset 
            
            # Identify active Query buckets
            query_active_buckets = torch.unique(q_global.view(-1))
            
            # Filter candidates: Check intersection
            c_flat = c_global.view(-1)
            mask_flat = torch.isin(c_flat, query_active_buckets)
            mask = mask_flat.view(c_global.shape)
            
            # Count collisions
            collision_counts = mask.sum(dim=1) 
            
            # 3. Filtering (Strict: >= 2 Collisions)
            candidates_mask = (collision_counts >= 2)
            candidate_indices_local = torch.nonzero(candidates_mask, as_tuple=True)[0]
            
            if len(candidate_indices_local) == 0:
                # Fallback: All temporal
                return sorted(candidate_indices[-budget:])

            # 4. Probability Scoring (on survivors only)
            c_subset_vecs = centered_candidates[candidate_indices_local]
            c_bits_subset = self.compute_bits(c_subset_vecs) 
            q_bits = self.compute_bits(centered_query)       
            
            # Calculate Hamming Similarity (p)
            matches = (q_bits.unsqueeze(1) == c_bits_subset.unsqueeze(0)) 
            total_bits = num_tables * num_bits
            hamming_sim = matches.float().sum(dim=(-1, -2)) / total_bits
            
            # MagicPIG Formula
            p_k = hamming_sim.pow(num_bits)
            one_minus_pk = 1 - p_k
            term2 = one_minus_pk.pow(num_tables)
            term3 = num_tables * p_k * (one_minus_pk.pow(num_tables - 1))
            
            u_scores = 1.0 - term2 - term3 
            
            final_scores, _ = u_scores.max(dim=0) 
            
            # 5. Selection
            final_local_indices = candidate_indices_local
            
            if len(final_scores) > budget:
                _, top_k_local = torch.topk(final_scores, k=budget)
                final_local_indices = candidate_indices_local[top_k_local]
            else:
                # Fallback: Fill remainder with most recent tokens
                needed = budget - len(final_local_indices)
                if needed > 0:
                     # Identify unselected indices
                     selected_set = set(final_local_indices.tolist())
                     # Simply take the last N candidates that aren't already selected
                     all_indices_range = range(len(candidate_indices))
                     fallback_pool = [i for i in reversed(all_indices_range) if i not in selected_set]
                     fallback_indices = fallback_pool[:needed]
                     
                     final_local_indices = torch.cat([
                         final_local_indices, 
                         torch.tensor(fallback_indices, device=self.device)
                     ])

            top_k_indices = final_local_indices

        # Map back
        selected_indices = [candidate_indices[i] for i in top_k_indices.cpu().tolist()]
        
        return sorted(selected_indices)
