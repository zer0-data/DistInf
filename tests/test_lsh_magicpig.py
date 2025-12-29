
import torch
import unittest
from unittest.mock import MagicMock

# Adjust imports
import sys
import os
# bmrt is in parent directory of tests/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bmrt.selectors.lsh_core import LSHSelector
# LSHSampler is in bmrt.lsh_sampler, imported in lsh_core
from bmrt.lsh_sampler import LSHSampler

class TestMagicPIG(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.head_dim = 16
        self.num_bits = 4
        self.num_tables = 4
        
        self.selector = LSHSelector(
            head_dim=self.head_dim, 
            lsh_mode='magicpig_baseline', 
            num_bits=self.num_bits, 
            num_tables=self.num_tables, 
            device=self.device
        )
        
    def test_probability_scoring(self):
        # Create vectors manually to control hash collisions
        # Sampler uses random projection. 
        # We can mock compute_bits to return specific patterns.
        
        # Mocking compute_bits on the selector's sampler
        # 1 Query token, 3 Candidates
        # bits shape: [N, L, K] = [N, 4, 4]
        
        # Q: [0000, 0000, 0000, 0000] (All zero bits)
        q_bits = torch.zeros(1, 4, 4, dtype=torch.long)
        
        # C1: Perfect Match (All zero) -> Should have score 1.0 (approx) and valid
        c1_bits = torch.zeros(1, 4, 4, dtype=torch.long)
        
        # C2: 2 Tables Match, 2 Tables Mismatch (All bits flip)
        # Hamming Dist > 0.
        # Collisions = 2. -> Should be valid, score < C1.
        c2_bits = torch.zeros(1, 4, 4, dtype=torch.long)
        c2_bits[0, 2:, :] = 1 
        
        # C3: 1 Table Match, 3 Tables Mismatch
        # Collisions = 1. -> Should be INVALID (Filter >= 2)
        c3_bits = torch.zeros(1, 4, 4, dtype=torch.long)
        c3_bits[0, 1:, :] = 1
        
        c_bits = torch.cat([c1_bits, c2_bits, c3_bits], dim=0) # [3, 4, 4]
        
        c_bits = torch.cat([c1_bits, c2_bits, c3_bits], dim=0) # [3, 4, 4]
        
        # Mock the sampler
        # Mock compute_bits to return bits based on input size
        def mock_compute_bits(vectors):
            # Q is usually 1x16
            if vectors.shape[0] == 1:
                return q_bits
            # C is 3x16 (Full candidate list)
            elif vectors.shape[0] == 3:
                return c_bits
            # C_subset could be anything. 
            # Based on loop 1, we expect C1 and C2 to survive (indices 0 and 1).
            # So subset will be size 2. We should return first 2 rows of c_bits.
            elif vectors.shape[0] == 2:
                return c_bits[:2]
            # Fallback for unexpected shapes
            return c_bits

        self.selector.lsh_sampler.compute_bits = MagicMock(side_effect=mock_compute_bits)
        
        # Dummy inputs
        query_vecs = torch.randn(1, 16)
        cand_vecs = torch.randn(3, 16)
        cand_indices = [10, 20, 30]
        budget = 2
        
        selected = self.selector.select(
            query_ids=None,
            query_vectors=query_vecs,
            candidate_vectors=cand_vecs,
            candidate_indices=cand_indices,
            budget=budget
        )
        
        print(f"Selected: {selected}")
        
        # C1 (10) should be first.
        # C2 (20) should be second (valid).
        # C3 (30) invalid, but budget=2. So we expect [10, 20].
        
        self.assertIn(10, selected)
        self.assertIn(20, selected)
        self.assertNotIn(30, selected)
        
        # Verify Fallback
        # If budget = 3, C3 should be picked via fallback (most recent)
        # Note: side_effect already has items for this call
        selected_fallback = self.selector.select(
            query_ids=None,
            query_vectors=query_vecs,
            candidate_vectors=cand_vecs,
            candidate_indices=cand_indices,
            budget=3
        )
        print(f"Selected Fallback: {selected_fallback}")
        self.assertEqual(len(selected_fallback), 3)
        self.assertIn(30, selected_fallback)

def main():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMagicPIG)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    main()
