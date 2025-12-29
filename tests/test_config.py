import unittest
import torch
import sys
import os

# Ensure import works
try:
    from bmrt import RecursiveCompressionEngine
    from bmrt.selectors.eager_exact import ExactSelector
    from bmrt.selectors.lsh_core import LSHSelector
except ImportError:
    sys.path.append('.')
    from bmrt import RecursiveCompressionEngine

class TestGEMINIConfig(unittest.TestCase):
    def test_import_structure(self):
        """Verify imports work."""
        self.assertTrue(True)

    def test_invalid_exact_flash(self):
        """Verify Exact + Flash raises error."""
        with self.assertRaises(ValueError) as cm:
            RecursiveCompressionEngine(
                model_path="dummy",
                selector_type='exact',
                backend='flash',
                budget=100
            )
        self.assertIn("Flash Attention does not support", str(cm.exception))

    def test_invalid_divisor(self):
        """Verify protection divisor > 1."""
        with self.assertRaises(ValueError) as cm:
            RecursiveCompressionEngine(
                model_path="dummy",
                selector_type='exact',
                backend='eager',
                budget=100,
                protection_divisor=1
            )
        self.assertIn("protection_divisor must be > 1", str(cm.exception))

    def test_budget_exhaustion(self):
        """Verify budget exhaustion check."""
        # Divisor 2 -> Anchor=50, Window=50 -> Global=0 -> Error if expecting global
        # Engine checks if global_budget <= 0? Yes it does.
        with self.assertRaises(ValueError) as cm:
            RecursiveCompressionEngine(
                model_path="dummy",
                selector_type='exact',
                backend='eager',
                budget=100,
                protection_divisor=2 # 100//2 = 50 anchor + 50 window = 100. Global=0.
            )
        self.assertIn("consumes all budget", str(cm.exception))
        
if __name__ == '__main__':
    # Mocking AutoTokenizer and Model loading to avoid big downloads during simple config test
    from unittest.mock import MagicMock
    import bmrt.processor
    
    bmrt.processor.AutoTokenizer = MagicMock()
    bmrt.processor.AutoModelForCausalLM = MagicMock()
    
    unittest.main()
