
import torch
import unittest
from unittest.mock import MagicMock, patch

# Import the classes to test
# Adjust imports based on where this script is located relative to bmrt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bmrt.processor import RecursiveCompressionEngine
from bmrt.accumulator import AttentionScoreAccumulator

class MockTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "</s>"
        self.eos_token_id = 2

    def encode(self, text, **kwargs):
        # Return dummy tensor of length roughly proportional to text length
        # Just return 1 for each 'word'
        length = len(text.split())
        return torch.ones(1, length, dtype=torch.long)
    
    def apply_chat_template(self, messages, **kwargs):
        return torch.ones(1, 10, dtype=torch.long) # Dummy query

class TestRecursiveCompression(unittest.TestCase):
    
    @patch('bmrt.processor.AutoTokenizer.from_pretrained')
    @patch('bmrt.processor.AutoModelForCausalLM.from_pretrained')
    def test_numeric_counts(self, mock_model_cls, mock_tokenizer_cls):
        """
        Verify the token counts match the user's specific scenario:
        Budget=4096 (Anchor=1024, Local=1024, Global=2048).
        Block Size > Budget.
        """
        # Setup Mocks
        mock_tokenizer_cls.return_value = MockTokenizer()
        
        mock_model = MagicMock()
        mock_model.config.hidden_size = 4096
        mock_model.config.num_attention_heads = 32
        # Mock forward pass to return correct shapes
        def forward_mock(input_ids, **kwargs):
            bsz, seq_len = input_ids.shape
            # Mock KV cache: list of tuples
            # We need to return something that acts like DynamicCache or Legacy
            # Let's just return a list of tuples (K, V)
            # K, V shape: [1, 32, seq_len, 128]
            
            # The processor expects `past_key_values` from output.
            # If use_cache=True, it returns updated cache.
            # Updated cache should be Concatenation of Past + Input (if legacy-ish)
            
            past_kv = kwargs.get('past_key_values')
            start_len = 0
            if past_kv is not None:
                start_len = past_kv[0][0].shape[2]
                
            total_len = start_len + seq_len
            
            # Create dummy tensors on CPU
            k = torch.randn(1, 32, total_len, 128)
            v = torch.randn(1, 32, total_len, 128)
            new_kv = tuple([(k, v) for _ in range(2)]) # 2 layers
            
        class MockOutput:
            def __init__(self, past_key_values, logits):
                self.past_key_values = past_key_values
                self.logits = logits

        def forward_mock(input_ids, **kwargs):
            print("DEBUG: forward_mock called!")
            bsz, seq_len = input_ids.shape
            
            past_kv = kwargs.get('past_key_values')
            start_len = 0
            if past_kv is not None:
                # DynamicCache or Legacy?
                # engine passes DynamicCache if prefix exists.
                # DynamicCache has get_seq_length()
                if hasattr(past_kv, 'get_seq_length'):
                    start_len = past_kv.get_seq_length()
                elif isinstance(past_kv, tuple):
                     start_len = past_kv[0][0].shape[2]
            
            total_len = start_len + seq_len
            print(f"DEBUG: Mock Forward total_len={total_len}")
            
            k = torch.randn(1, 32, total_len, 128)
            v = torch.randn(1, 32, total_len, 128)
            new_kv = tuple([(k, v) for _ in range(2)]) 
            
            return MockOutput(new_kv, torch.randn(1, seq_len, 32000))

        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.side_effect = forward_mock # Accessing model() invokes side_effect if set on mock object?
        # Typically mock_obj() calls __call__.
        # Setting side_effect on the mock object itself makes it callable? 
        # No, mock_model is the object. 
        # mock_model() triggers __call__.
        mock_model.forward = forward_mock
        mock_model.device = "cpu" # Force CPU
        mock_model_cls.return_value = mock_model

        # Force Engine to CPU
        with patch('torch.cuda.is_available', return_value=False):
             # Initialize Engine with RECURSIVE mode
            engine = RecursiveCompressionEngine(
                model_path="dummy",
                selector_type='exact',
                compression_mode='recursive',
                budget=4096,
                protection_divisor=4, 
                block_size=8192
            )
        # Force engine device to cpu just in case
        engine.device = "cpu"
        
        # Check Allocation
        # Check Allocation
        # Anchor = 1024, Local = 1024, Global = 2048.
        self.assertEqual(engine.global_budget, 2048)
        
        # We need to mock the `ExactSelector.select` method because 
        # mocking the internal Accumulator and Model Attn weights is complex.
        # We assume `ExactSelector` works (tested separately) and just test the `RecursiveCompressionEngine` logic
        # regarding candidate pooling.
        
        # To do this, we can spy on `engine.selector.select`.
        original_select = engine.selector.select
        
        def mock_select(candidate_indices, budget, **kwargs):
            # Just select the first `budget` indices
            return candidate_indices[:budget]
            
        engine.selector.select = mock_select
        
        # --- TEST EXECUTION ---
        
        # Create Dummy Input: 2 Blocks of 8192 tokens.
        # Total 16384 tokens.
        # token_ids
        tokens = torch.arange(16384).unsqueeze(0)
        
        # Override `_split_into_blocks` to just return what we want
        engine._split_into_blocks = MagicMock(return_value=[
            tokens[:, :8192],
            tokens[:, 8192:]
        ])
        
        # Mock _process_block internals? No, we want to test _process_block logic.
        # But we need `_process_block` to call `select` with correct candidates.
        
        # Manually run loop step 1
        accumulated_kv = None
        block1 = tokens[:, :8192]
        query = torch.tensor([[1]])
        
        _, _, block1_kv = engine._process_block(
            prefix_kv_cache=None,
            block_ids=block1,
            query_ids=query,
            total_context_length=0,
            block_start_position=0
        )
        
        print(f"DEBUG: block1_kv type: {type(block1_kv)}")
        print(f"DEBUG: block1_kv len: {len(block1_kv)}")
        if len(block1_kv) > 0:
             print(f"DEBUG: block1_kv[0] type: {type(block1_kv[0])}")
        
        # Verify Block 1 Result
        # Size should be Anchor(1024) + Local(1024) + Global(2048) = 4096.
        # Because block_len (8192) > budget.
        
        kv_len_1 = block1_kv[0][0].shape[2]
        print(f"Block 1 KV Len: {kv_len_1}")
        self.assertEqual(kv_len_1, 4096)
        
        # Merge
        accumulated_kv = block1_kv
        
        # Manually run loop step 2
        block2 = tokens[:, 8192:]
        
        # Spy on `select` again to check inputs
        candidates_seen = []
        def spy_select(candidate_indices, budget, **kwargs):
            candidates_seen.append(candidate_indices)
            return candidate_indices[:budget]
        engine.selector.select = spy_select
        
        _, _, block2_kv = engine._process_block(
            prefix_kv_cache=accumulated_kv,
            block_ids=block2,
            query_ids=query,
            total_context_length=8192,
            block_start_position=8192
        )
        
        kv_len_2 = block2_kv[0][0].shape[2]
        print(f"Block 2 KV Len: {kv_len_2}")
        
        # Verify:
        # 1. Output size is consistent (bounded)
        # It should be Anchor(0 for new block?) + Local(1024) + Global(2048) = 3072 + Global Anchor (1024 from history) = 4096.
        # Wait, Step 2 logic:
        # Global Anchor (1024) is kept.
        # Current Local (1024) is kept.
        # Selection Budget (2048) is selected.
        # Total = 1024 + 1024 + 2048 = 4096.
        self.assertEqual(kv_len_2, 4096) 
        
        # 2. Verify Candidates passed to selector
        # Candidates should include:
        # - History candidates (from Block 1's non-anchor region)
        # - Current block candidates (Block 2's non-local region)
        
        passed_candidates = candidates_seen[0]
        
        # History Candidates:
        # Block 1 retained 4096 tokens.
        # Anchor=1024.
        # Candidates from History = [1024 ... 4096] (Indices in accumulated cache)
        # Count = 3072 tokens available from history.
        
        # Current Block Candidates:
        # Block 2 len = 8192.
        # Local = 1024.
        # Anchor = 0 (since block start > anchor size).
        # Candidates from Block 2 = [4096 ... 4096 + 8192 - 1024]
        # Range relative to start of new cache: [4096, 11264] ?
        # Count = 7168 tokens.
        
        # Total Pool = 3072 (History) + 7168 (Current) = 10240 candidates.
        
        self.assertEqual(len(passed_candidates), 3072 + 7168)
        
        print("Recursive Test Passed!")

def main():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestRecursiveCompression)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    main()
