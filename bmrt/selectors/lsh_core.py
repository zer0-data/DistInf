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
    
    def __init__(self, head_dim: int, lsh_mode: str = 'frequency_rank', num_bits: int = 12, num_tables: int = 20, device: str = 'cuda', mode: str = 'l2', dtype: torch.dtype = torch.bfloat16):
        self.head_dim = head_dim
        self.num_bits = num_bits
        self.num_tables = num_tables
        self.device = device
        self.lsh_mode = lsh_mode
        self.dtype = dtype
        # Tie-breaking mode (hyperparameter)
        self.mode = mode
        if mode not in ['l2', 'max_sim', 'mahalanobis', 'partitioned_centroid', 'none']:
            raise ValueError(f"Invalid tie-breaking mode: {mode}")
        if lsh_mode not in ['frequency_rank', 'magicpig_baseline']:
            raise ValueError(f"Invalid lsh_mode: {lsh_mode}")

        self.projection_matrix = torch.randn(
            head_dim, 
            num_tables * num_bits, 
            device=device, 
            dtype=dtype
        )
        
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
        projections = torch.matmul(vectors, self.projection_matrix)
        bits = (projections > 0).long()
        
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
        
        if self.lsh_mode == 'frequency_rank':
            # Vectorized Collision Counting
            # 1. Create global bucket indices by adding offsets to each table's hash codes
            #    This effectively flattens the tables into one large address space.
            offsets = torch.arange(self.num_tables, device=self.device) * num_buckets
            
            q_global = q_hashes + offsets.view(1, -1)
            c_global = c_hashes + offsets.view(1, -1)
            
            # 2. Mark active buckets in a single global mask
            total_buckets = self.num_tables * num_buckets
            # active_buckets: [Total_Buckets]
            active_buckets = torch.zeros(total_buckets, device=self.device, dtype=torch.bool)
            
            # Mark all buckets present in the query (flattened)
            active_buckets.index_fill_(0, q_global.view(-1).long(), True)
            
            # 3. Check collisions for candidates
            #    [N_c, Tables] -> Look up in [Total_Buckets] -> [N_c, Tables] bools
            hits = active_buckets[c_global.long()]
            
            # 4. Sum hits across tables to get total collisions per candidate
            collision_counts = hits.int().sum(dim=1)
            
            # --- TIE-BREAKING LOGIC (controlled via `self.mode`) ---
            
            # If mode is 'l2' or 'max_sim', use distance-based secondary sorting.
            # Otherwise fall back to sorting by collision count only.
            if self.mode == 'l2':
                q_vec = centered_query.mean(dim=0, keepdim=True)
                dists = torch.norm(centered_candidates - q_vec, dim=1)
                
                sorted_by_dist_indices = torch.argsort(dists, descending=False)
                sorted_counts = collision_counts[sorted_by_dist_indices]
                sorted_by_count_indices = torch.argsort(sorted_counts, descending=True, stable=True)

                final_prioritized_indices = sorted_by_dist_indices[sorted_by_count_indices]
                top_k_indices = final_prioritized_indices[:budget]
            
            elif self.mode == 'max_sim':
                # New approach: Pairwise L2 distances with max similarity (min distance)
                dist_matrix = torch.cdist(centered_query, centered_candidates, p=2.0)
                
                min_dists, _ = torch.min(dist_matrix, dim=0)  # [C]
                
                sorted_by_dist_indices = torch.argsort(min_dists, descending=False)
                sorted_counts = collision_counts[sorted_by_dist_indices]
                sorted_by_count_indices = torch.argsort(sorted_counts, descending=True, stable=True)
                
                final_prioritized_indices = sorted_by_dist_indices[sorted_by_count_indices]
                top_k_indices = final_prioritized_indices[:budget]
            
            elif self.mode == 'mahalanobis':
                # Mahalanobis distance using query variance as weighting
                q_mean = centered_query.mean(dim=0)  # [D]
                q_var = centered_query.var(dim=0)    # [D]
                
                epsilon = 1e-8
                
                diff = centered_candidates - q_mean.unsqueeze(0)  # [C, D]
                weighted_diff_sq = (diff ** 2) / (q_var.unsqueeze(0) + epsilon)  # [C, D]
                mahal_dists = torch.sqrt(weighted_diff_sq.sum(dim=1))  # [C]
                
                sorted_by_dist_indices = torch.argsort(mahal_dists, descending=False)
                sorted_counts = collision_counts[sorted_by_dist_indices]
                sorted_by_count_indices = torch.argsort(sorted_counts, descending=True, stable=True)
                
                final_prioritized_indices = sorted_by_dist_indices[sorted_by_count_indices]
                top_k_indices = final_prioritized_indices[:budget]
            
            elif self.mode == 'partitioned_centroid':
                # Partitioned centroid: Split query into k chunks, compute centroid per chunk,
                # then find minimum distance to any partition centroid
                
                Q = centered_query.shape[0]
                k = max(1, Q // 16)
                k = min(k, 8)
                
                query_chunks = torch.chunk(centered_query, chunks=k, dim=0)
                partition_centroids = torch.stack([chunk.mean(dim=0) for chunk in query_chunks], dim=0)  # [k, D]
                
                dist_matrix = torch.cdist(centered_candidates, partition_centroids, p=2.0)  # [C, k]
                
                min_partition_dists, _ = torch.min(dist_matrix, dim=1)  # [C]
                
                sorted_by_dist_indices = torch.argsort(min_partition_dists, descending=False)
                sorted_counts = collision_counts[sorted_by_dist_indices]
                sorted_by_count_indices = torch.argsort(sorted_counts, descending=True, stable=True)

                final_prioritized_indices = sorted_by_dist_indices[sorted_by_count_indices]
                top_k_indices = final_prioritized_indices[:budget]
            
            else:
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
                    selected_set = set(final_local_indices.tolist())
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
