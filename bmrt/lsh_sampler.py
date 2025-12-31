import torch
from typing import List, Tuple, Optional

class LSHSampler:
    """
    Implements Locality Sensitive Hashing (SimHash) with Multi-Table lookups 
    for approximate nearest neighbor search to select 'Global' tokens.
    
    This implementation is GPU-optimized for block-level compression (small N ~2048).
    It uses matrix operations instead of hash tables to avoid CPU-GPU synchronization overhead.
    """
    def __init__(self, head_dim: int, num_bits: int = 4, num_tables: int = 4, device: str = 'cuda'):
        """
        Args:
            head_dim: Dimension of the input vectors.
            num_bits (K): Number of bits per hash (defines number of buckets = 2^K).
            num_tables (L): Number of independent hash tables (improves recall).
            device: 'cuda' or 'cpu'.
        """
        self.head_dim = head_dim
        self.num_bits = num_bits
        self.num_tables = num_tables
        self.device = device
        
        # Projection Matrix: [Head_Dim, Tables * Bits]
        # We generate independent projections for each bit of each table.
        self.total_bits = num_tables * num_bits
        self.projection_matrix = torch.randn(head_dim, self.total_bits, device=device)
        
        # Pre-compute powers of two for bit-packing: [1, 1, Bits]
        # We pack 'num_bits' into a single integer for each table.
        self.powers_of_two = (2 ** torch.arange(num_bits, device=device)).view(1, 1, num_bits) # [1, 1, K]

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


    def select_global_tokens(
        self,
        query_vectors: torch.Tensor,
        candidate_vectors: torch.Tensor,
        candidate_indices: List[int],
        candidate_scores: torch.Tensor,
        budget: int,
    ) -> List[int]:
        """
        Select tokens using Multi-Table LSH.
        A candidate is selected if its hash matches the query's hash in ANY table (or sufficient tables).
        
        Refined Logic:
        - Count collisions across all tables.
        - Sort by (Collision Count DESC, Attention Score DESC).
        """
        if budget >= len(candidate_indices):
            return candidate_indices
            
        if budget <= 0:
            return []

        # 1. Centering
        q_mean = query_vectors.mean(dim=0, keepdim=True)
        c_mean = candidate_vectors.mean(dim=0, keepdim=True)
        
        centered_query = query_vectors - c_mean
        centered_candidates = candidate_vectors - c_mean
        
        # 2. Compute Hashes
        q_hashes = self._compute_hashes(centered_query)
        c_hashes = self._compute_hashes(centered_candidates)
        
        # 3. Compute Collision Function
        # Strategy: Bucket Lookups via boolean mask
        # Cost: O(Q*L + C*L + 2^K). Highly efficient for small K (4-8).
        
        num_buckets = 2 ** self.num_bits
        collision_counts = torch.zeros(len(candidate_indices), device=self.device, dtype=torch.int32)
        
        for l in range(self.num_tables):
            # Get Hashes for this table
            q_h = q_hashes[:, l] 
            c_h = c_hashes[:, l] 
            
            # Create bucket mask: Which buckets are 'touched' by query tokens?
            bucket_mask = torch.zeros(num_buckets, device=self.device, dtype=torch.bool)
            bucket_mask.index_fill_(0, q_h.long(), True)
            
            # Check collisions for candidates
            hits = bucket_mask[c_h.long()]
            collision_counts += hits.int()
            
        # 4. Selection (Lexicographical Sort)
        # Primary Key: Collision Count (Descending)
        # Secondary Key: Attention Score (Descending)
        
        # 1. Sort by secondary key (Score)
        sorted_by_score_indices = torch.argsort(candidate_scores, descending=True)
        
        # 2. Re-order counts based on score sort
        sorted_counts = collision_counts[sorted_by_score_indices]
        
        # 3. Sort by primary key (Counts) using stable sort
        sorted_by_count_indices = torch.argsort(sorted_counts, descending=True, stable=True)
        
        # 4. Combine
        final_prioritized_indices = sorted_by_score_indices[sorted_by_count_indices]
        
        # Select Top-K
        selected_subindices = final_prioritized_indices[:budget]
        
        # Map back to original indices
        selected_indices = [candidate_indices[i] for i in selected_subindices.cpu().tolist()]
        
        return sorted(selected_indices)
