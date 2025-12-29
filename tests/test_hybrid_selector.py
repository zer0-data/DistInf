
import unittest
from unittest.mock import MagicMock
import torch
import sys
import os

# bmrt is in parent directory of tests/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bmrt.selectors.hybrid import HybridSelector
from bmrt.selectors.base import BaseSelector

class MockSelector(BaseSelector):
    def __init__(self, name):
        self.name = name
    
    def setup(self, model):
        pass
        
    def select(self, candidate_indices, budget, **kwargs):
        # deterministically pick first 'budget' available indices
        return candidate_indices[:budget]

    def cleanup(self):
        pass

class TestHybridSelector(unittest.TestCase):
    def setUp(self):
        self.primary = MockSelector("primary")
        self.secondary = MockSelector("secondary")
        self.selector = HybridSelector(self.primary, self.secondary, primary_ratio=0.6)
        
    def test_select_split(self):
        # Total Budget = 10. Ratio 0.6.
        # Primary Budget = 6. Secondary Budget = 4.
        
        candidates = list(range(100)) # 0..99
        budget = 10
        
        # Primary should pick 0,1,2,3,4,5
        # Secondary should pick from remaining [6..99]. (0..5 are filtered out)
        # Secondary picks first 4: 6,7,8,9
        
        # We need to spy on calls or check result.
        # Since MockSelector just takes first N indices, we can predict output.
        
        selected = self.selector.select(
            query_ids=None, query_vectors=None, candidate_vectors=torch.zeros(100, 10),
            candidate_indices=candidates,
            budget=budget
        )
        
        expected = list(range(10)) # 0..9
        self.assertEqual(selected, expected)
        
        # Verify split
        # We can mock the select methods to return specific things to verify logic more strictly
        
    def test_overlap_filtering(self):
        # Force Primary to pick specific indices
        self.primary.select = MagicMock(return_value=[10, 20])
        self.selector.primary_ratio = 0.5
        budget = 4
        # Primary gets budget 2. Returns [10, 20].
        
        # Secondary gets budget 2.
        # Candidate pool passed to Secondary should NOT have 10, 20.
        
        candidates = [10, 20, 30, 40, 50]
        
        def secondary_side_effect(candidate_indices, budget, **kwargs):
            # Verify 10, 20 are not in candidate_indices
            assert 10 not in candidate_indices
            assert 20 not in candidate_indices
            return [30, 40]
            
        self.secondary.select = MagicMock(side_effect=secondary_side_effect)
        
        selected = self.selector.select(
            query_ids=None, query_vectors=None, candidate_vectors=torch.zeros(5, 10),
            candidate_indices=candidates,
            budget=budget
        )
        
        self.assertEqual(sorted(selected), [10, 20, 30, 40])
        self.primary.select.assert_called_once()
        self.secondary.select.assert_called_once()
        
    def test_vectors_filtering(self):
         # Verify that candidate_vectors are also sliced
         # Primary picks index 0.
         # Secondary should receive vector slice excluding index 0.
         
         candidates = [0, 1]
         vectors = torch.tensor([[0.0], [1.0]]) # 0 -> 0.0, 1 -> 1.0
         
         self.primary.select = MagicMock(return_value=[0])
         self.selector.primary_ratio = 0.5
         budget = 2
         
         def secondary_side_effect(candidate_indices, candidate_vectors, **kwargs):
             # Expect only index 1
             assert candidate_indices == [1]
             # Expect only vector [1.0]
             assert torch.equal(candidate_vectors, torch.tensor([[1.0]]))
             return [1]
             
         self.secondary.select = MagicMock(side_effect=secondary_side_effect)
         
         self.selector.select(
            query_ids=None, query_vectors=None, 
            candidate_vectors=vectors,
            candidate_indices=candidates,
            budget=budget
        )
        
         self.secondary.select.assert_called_once()

def main():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestHybridSelector)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    main()
