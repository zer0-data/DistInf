
import torch
import torch.nn as nn
from bmrt.accumulator import AttentionScoreAccumulator, AttentionWrapper

class MockAttention(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        
    def forward(self, hidden_states, output_attentions=False, **kwargs):
        # Simulate attention output: (output, weights, past_key_values)
        bsz, seq_len, dim = hidden_states.shape
        # Fake weights: [bsz, heads, seq_len, seq_len]
        weights = torch.randn(bsz, 4, seq_len, seq_len) if output_attentions else None
        
        return (hidden_states, weights, None)

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({'self_attn': MockAttention(i)}) for i in range(2)
        ])
        
    def forward(self, input_ids, output_attentions=False, **kwargs):
        hidden_states = torch.randn(1, input_ids.shape[1], 32)
        all_attentions = []
        
        for layer in self.layers:
            # Typical HF structure
            out = layer.self_attn(hidden_states, output_attentions=output_attentions, **kwargs)
            if output_attentions:
                all_attentions.append(out[1])
        
        return hidden_states, tuple(all_attentions) if output_attentions else None

def test_wrapper():
    print("Initializing Accumulator and Mock Model...")
    acc = AttentionScoreAccumulator()
    model = MockModel()
    
    # 1. Test Wrapping
    print("Wrapping model...")
    acc.wrap_model(model)
    
    # Check if wrapped
    assert isinstance(model.layers[0].self_attn, AttentionWrapper)
    print("Model wrapped successfully.")
    
    # 2. Test Accumulation Logic
    # Configure accumulator for a block
    acc.start_block(total_seq_len=10, block_token_count=10)
    
    # Run forward pass WITHOUT output_attentions=True
    print("Running forward pass (output_attentions=False)...")
    input_ids = torch.zeros(1, 10)
    
    # This calls model(..., output_attentions=False) locally, but wrapper should force True internally
    output = model(input_ids, output_attentions=False)
    
    # 3. Verify Accumulator got data
    print(f"Accumulator layer count: {acc.layer_count}")
    assert acc.layer_count == 2
    assert acc.accumulated_scores is not None
    print("Accumulator successfully captured weights!")
    
    # 4. Verify Output does NOT contain weights (memory saving)
    # output is (hidden_states, None)
    # The second element should be None because we passed output_attentions=False
    # and the wrapper should clean up its leaks.
    # Wait, my MockModel returns (hidden, None) if output_attentions=False.
    # The wrapper's return value replaces the module's return value.
    # If wrapper works, it calls module with output_attentions=True.
    # Module returns (hidden, weights, kv).
    # Wrapper cleans it to (hidden, None, kv).
    # Parent model sees (hidden, None, kv).
    # Parent model logic for `all_attentions` might break if it expects a tensor but gets None?
    # In HF models, they usually do:
    # outputs = self.self_attn(...)
    # attn_weights = outputs[1]
    # if output_attentions: all_attentions = ...
    
    # Let's verify what the wrapper returns exactly
    print("Verifying wrapper return values...")
    # Manually call the wrapped module
    layer_out = model.layers[0].self_attn(torch.randn(1, 10, 32), output_attentions=False)
    # layer_out should be tuple
    print(f"Layer output preview: {layer_out}")
    assert layer_out[1] is None, "Wrapper failed to hide attention weights!"
    print("Wrapper correctly hid attention weights.")
    
    # 5. Test Unwrapping
    print("Unwrapping model...")
    acc.unwrap_model(model)
    assert isinstance(model.layers[0].self_attn, MockAttention)
    print("Model unwrapped successfully.")

if __name__ == "__main__":
    try:
        test_wrapper()
        print("\nALL TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
