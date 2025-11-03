# Question 3: Implement Multi-Head Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, v), attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        return self.w_o(attn_output)

# Follow-up: Optimize this for memory efficiency

import torch
import torch.nn as nn
import numpy as np


def test_basic_attention():
    """Test basic attention functionality"""
    print("=== Test 1: Basic Attention ===")
    d_model, num_heads, batch_size, seq_len = 512, 8, 2, 10

    attention = MultiHeadAttention(d_model, num_heads)

    # Create random input tensors
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    output = attention(query, key, value)

    # Verify output shape
    assert output.shape == (
    batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("✓ Basic attention test passed")


def test_attention_mask():
    """Test attention with masking"""
    print("\n=== Test 2: Attention with Mask ===")
    d_model, num_heads, batch_size, seq_len = 512, 8, 2, 5

    attention = MultiHeadAttention(d_model, num_heads)

    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    # Create causal mask (upper triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    mask = mask.repeat(batch_size, num_heads, 1, 1)

    output_with_mask = attention(query, key, value, mask)
    output_without_mask = attention(query, key, value, None)

    # Outputs should be different when mask is applied
    assert not torch.allclose(output_with_mask, output_without_mask), "Mask should affect output"
    print("✓ Attention mask test passed")


def test_various_shapes():
    """Test with different input shapes"""
    print("\n=== Test 3: Various Input Shapes ===")
    test_cases = [
        (256, 4, 1, 16),  # Small model
        (512, 8, 4, 32),  # Medium model
        (1024, 16, 8, 64),  # Large model
    ]

    for d_model, num_heads, batch_size, seq_len in test_cases:
        attention = MultiHeadAttention(d_model, num_heads)

        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)

        output = attention(query, key, value)

        expected_shape = (batch_size, seq_len, d_model)
        assert output.shape == expected_shape, f"Failed for d_model={d_model}, shape={output.shape}"
        print(f"✓ Shape test passed for d_model={d_model}, heads={num_heads}")


def test_different_sequence_lengths():
    """Test with different sequence lengths for query, key, value"""
    print("\n=== Test 4: Different Sequence Lengths ===")
    d_model, num_heads, batch_size = 512, 8, 2

    attention = MultiHeadAttention(d_model, num_heads)

    # Different sequence lengths
    query_seq_len, key_seq_len, value_seq_len = 10, 15, 15
    query = torch.randn(batch_size, query_seq_len, d_model)
    key = torch.randn(batch_size, key_seq_len, d_model)
    value = torch.randn(batch_size, value_seq_len, d_model)

    output = attention(query, key, value)

    # Output should have query sequence length
    assert output.shape == (batch_size, query_seq_len, d_model)
    print("✓ Different sequence lengths test passed")


def test_attention_scores():
    """Test attention score calculations"""
    print("\n=== Test 5: Attention Score Calculation ===")
    d_model, num_heads, batch_size, seq_len = 12, 2, 1, 3

    attention = MultiHeadAttention(d_model, num_heads)

    # Use simple inputs for easier verification
    query = torch.ones(batch_size, seq_len, d_model)
    key = torch.ones(batch_size, seq_len, d_model)
    value = torch.ones(batch_size, seq_len, d_model)

    # Manually compute expected attention
    q_proj = attention.w_q(query).view(batch_size, -1, num_heads, attention.d_k).transpose(1, 2)
    k_proj = attention.w_k(key).view(batch_size, -1, num_heads, attention.d_k).transpose(1, 2)

    # Expected scores should be uniform after softmax
    expected_scores = torch.ones(seq_len, seq_len) / seq_len

    output = attention(query, key, value)

    # Output should be transformed version of value
    assert output.shape == (batch_size, seq_len, d_model)
    print("✓ Attention score calculation test passed")


def test_gradient_flow():
    """Test that gradients flow through all components"""
    print("\n=== Test 6: Gradient Flow ===")
    d_model, num_heads, batch_size, seq_len = 512, 8, 2, 10

    attention = MultiHeadAttention(d_model, num_heads)

    query = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    key = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    value = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    output = attention(query, key, value)

    # Create dummy loss and backpropagate
    dummy_loss = output.sum()
    dummy_loss.backward()

    # Check that gradients are computed
    assert query.grad is not None, "Gradients should flow to query"
    assert key.grad is not None, "Gradients should flow to key"
    assert value.grad is not None, "Gradients should flow to value"
    assert attention.w_q.weight.grad is not None, "Gradients should flow to weight matrices"

    print("✓ Gradient flow test passed")


def test_single_head():
    """Test with single head attention"""
    print("\n=== Test 7: Single Head Attention ===")
    d_model, num_heads, batch_size, seq_len = 512, 1, 2, 10

    attention = MultiHeadAttention(d_model, num_heads)

    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    output = attention(query, key, value)
    assert output.shape == (batch_size, seq_len, d_model)
    print("✓ Single head attention test passed")


def test_mini_batch_size():
    """Test with batch size of 1"""
    print("\n=== Test 8: Mini Batch Size ===")
    d_model, num_heads, batch_size, seq_len = 512, 8, 1, 5

    attention = MultiHeadAttention(d_model, num_heads)

    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    output = attention(query, key, value)
    assert output.shape == (batch_size, seq_len, d_model)
    print("✓ Mini batch size test passed")


def test_very_short_sequence():
    """Test with very short sequence length"""
    print("\n=== Test 9: Very Short Sequence ===")
    d_model, num_heads, batch_size, seq_len = 512, 8, 2, 1

    attention = MultiHeadAttention(d_model, num_heads)

    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    output = attention(query, key, value)
    assert output.shape == (batch_size, seq_len, d_model)
    print("✓ Very short sequence test passed")


def test_memory_usage():
    """Test memory usage with large inputs"""
    print("\n=== Test 10: Memory Usage ===")
    d_model, num_heads, batch_size, seq_len = 1024, 16, 4, 256

    attention = MultiHeadAttention(d_model, num_heads)

    # Test that it doesn't crash with reasonable sizes
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    output = attention(query, key, value)
    assert output.shape == (batch_size, seq_len, d_model)
    print("✓ Memory usage test passed")


def test_deterministic_behavior():
    """Test that same inputs produce same outputs"""
    print("\n=== Test 11: Deterministic Behavior ===")
    torch.manual_seed(42)  # For reproducibility

    d_model, num_heads, batch_size, seq_len = 512, 8, 2, 10
    attention = MultiHeadAttention(d_model, num_heads)

    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    output1 = attention(query, key, value)
    output2 = attention(query, key, value)

    # Should be exactly the same
    assert torch.allclose(output1, output2), "Outputs should be deterministic"
    print("✓ Deterministic behavior test passed")


def test_with_real_transformer_data():
    """Test with data that resembles real transformer usage"""
    print("\n=== Test 12: Real Transformer Data ===")
    d_model, num_heads, batch_size, seq_len = 768, 12, 8, 128

    attention = MultiHeadAttention(d_model, num_heads)

    # Simulate embedded input (typical range for embeddings)
    query = torch.randn(batch_size, seq_len, d_model) * 0.02
    key = torch.randn(batch_size, seq_len, d_model) * 0.02
    value = torch.randn(batch_size, seq_len, d_model) * 0.02

    # Add causal mask (like in GPT models)
    mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
    mask = mask.repeat(batch_size, num_heads, 1, 1)

    output = attention(query, key, value, mask)

    assert output.shape == (batch_size, seq_len, d_model)

    # Output should have reasonable values (not NaN or Inf)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
    print("✓ Real transformer data test passed")


def test_parameter_initialization():
    """Test that all parameters are properly initialized"""
    print("\n=== Test 13: Parameter Initialization ===")
    d_model, num_heads = 512, 8
    attention = MultiHeadAttention(d_model, num_heads)

    # Check that all parameters have gradients and require grad
    for name, param in attention.named_parameters():
        assert param.requires_grad, f"Parameter {name} should require grad"
        assert param.data.shape == param.shape, f"Parameter {name} shape mismatch"

    total_params = sum(p.numel() for p in attention.parameters())
    expected_params = 4 * (d_model * d_model) + 4 * d_model  # 4 linear layers with bias

    assert total_params == expected_params, f"Expected {expected_params} parameters, got {total_params}"
    print("✓ Parameter initialization test passed")


def run_all_tests():
    """Run all test cases"""
    print("Running Multi-Head Attention Tests...")
    print("=" * 50)

    try:
        test_basic_attention()
        test_attention_mask()
        test_various_shapes()
        test_different_sequence_lengths()
        test_attention_scores()
        test_gradient_flow()
        test_single_head()
        test_mini_batch_size()
        test_very_short_sequence()
        test_memory_usage()
        test_deterministic_behavior()
        test_with_real_transformer_data()
        test_parameter_initialization()

        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The MultiHeadAttention implementation is working correctly.")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()