from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import torch

class BaseSelector(ABC):
    """Abstract base class for token selection strategies."""
    
    @abstractmethod
    def setup(self, model):
        """Perform any model setup/wrapping needed."""
        pass
        
    @abstractmethod
    def cleanup(self):
        """Clean up hooks or wrappers."""
        pass

    @abstractmethod
    def select(
        self,
        query_ids: torch.Tensor,
        query_vectors: torch.Tensor,
        candidate_vectors: torch.Tensor,
        candidate_indices: List[int],
        budget: int,
        **kwargs
    ) -> List[int]:
        """
        Select tokens from candidates to keep in the cache.
        
        Args:
            query_ids: [1, Q_Len]
            query_vectors: Optional [Q_Len, D]
            candidate_vectors: Optional [N_Cand, D]
            candidate_indices: List of original indices for candidates
            budget: Number of tokens to select
        
        Returns:
            List of selected indices.
        """
        pass
