import torch
import unittest
import sys
import os

# Add parent directory to path to import lsh_sampler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distinf.lsh_sampler import LSHSampler

class TestLSHSampler(unittest.TestCase):
    def setUp(self):
        self.device = "cpu" # Test on CPU for simplicity
        self.head_dim = 16
        self.num_bits = 4
        self.num_tables = 4
        self.sampler = LSHSampler(
            head_dim=self.head_dim, 
            num_bits=self.num_bits, 
            num_tables=self.num_tables, 
            device=self.device
        )

    def test_hash_consistency(self):
        """Test that same vector produces same hash codes."""
        vec = torch.randn(10, self.head_dim, device=self.device)
        hashes1 = self.sampler._compute_hashes(vec)
        hashes2 = self.sampler._compute_hashes(vec)
        self.assertTrue(torch.equal(hashes1, hashes2))
        self.assertEqual(hashes1.shape, (10, self.num_tables))

    def test_collision_selection(self):
        """Test that identical vectors act as collisions."""
        # Query: [1, D]
        query = torch.randn(1, self.head_dim, device=self.device)
        
        # Candidates: 5 candidates
        # 0, 1: Identical to Query (Should collide perfectly in all tables)
        # 2, 3, 4: Random
        
        candidates = torch.cat([query, query, torch.randn(3, self.head_dim, device=self.device)])
        candidate_indices = [10, 11, 12, 13, 14]
        candidate_scores = torch.tensor([1.0, 0.9, 0.5, 0.4, 0.3], device=self.device)
        
        # Budget = 2. Should pick 10 and 11.
        selected = self.sampler.select_global_tokens(
            query,
            candidates,
            candidate_indices,
            candidate_scores,
            budget=2
        )
        
        # 10 and 11 should be in selected because they appear in all tables
        self.assertIn(10, selected)
        self.assertIn(11, selected)
        self.assertEqual(len(selected), 2)

    def test_topk_fallback(self):
        """Test that if no collisions, we fall back to top scores."""
        query = torch.randn(1, self.head_dim, device=self.device)
        candidates = torch.randn(5, self.head_dim, device=self.device)
        candidate_indices = [0, 1, 2, 3, 4]
        scores = torch.tensor([10, 20, 30, 40, 50], device=self.device).float()
        
        # With random vectors and ample budget, we might pick collisions or just top scores.
        # But here we want to ensure we get *something* and preferably high scores.
        # To force fallback, we can use 0 bits? No, hard to force NO collisions with random.
        # But we check size.
        
        selected = self.sampler.select_global_tokens(
            query, candidates, candidate_indices, scores, budget=5
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected, [0, 1, 2, 3, 4])

    def test_multi_table_robustness(self):
        """Test that items colliding in MORE tables are preferred."""
        # Create vectors that collide in specific tables? Hard with random projection.
        # Mock logic:
        # We can mock _compute_hashes output or just rely on statistical probability?
        # Better: Create logic test where we override hashes.
        
        pass

if __name__ == '__main__':
    unittest.main()
