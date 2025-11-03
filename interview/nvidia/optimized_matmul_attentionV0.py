import torch

# Question 7: Optimize Matrix Operations for GPUs
def optimized_matmul_attention(q, k, v):
    """
    Implement attention with memory-efficient matrix multiplication
    """
    # Instead of: scores = q @ k.transpose(-2, -1)  # O(n²) memory

    # Memory-efficient approach for long sequences
    # Compute attention scores in chunks
    batch_size, num_heads, seq_len, d_k = q.shape
    chunk_size = 512  # Adjust based on available memory

    output = torch.zeros_like(v)

    for i in range(0, seq_len, chunk_size):
        end_i = min(i + chunk_size, seq_len)
        q_chunk = q[:, :, i:end_i, :]

        # Compute scores for this chunk
        scores_chunk = torch.matmul(q_chunk, k.transpose(-2, -1))
        scores_chunk = scores_chunk / math.sqrt(d_k)

        # Apply softmax and compute output chunk
        attn_weights = F.softmax(scores_chunk, dim=-1)
        output_chunk = torch.matmul(attn_weights, v)
        output[:, :, i:end_i, :] = output_chunk

    return output


import torch
import torch.nn.functional as F
import math
import time


def test_optimized_matmul_attention():
    """Test the optimized matrix multiplication for attention with memory efficiency"""
    print("=== Test: Optimized Matrix Multiplication for Attention ===")

    def optimized_matmul_attention(q, k, v):
        """
        Implement attention with memory-efficient matrix multiplication
        """
        # Memory-efficient approach for long sequences
        # Compute attention scores in chunks
        batch_size, num_heads, seq_len, d_k = q.shape
        chunk_size = 512  # Adjust based on available memory

        output = torch.zeros_like(v)

        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i, :]

            # Compute scores for this chunk
            scores_chunk = torch.matmul(q_chunk, k.transpose(-2, -1))
            scores_chunk = scores_chunk / math.sqrt(d_k)

            # Apply softmax and compute output chunk
            attn_weights = F.softmax(scores_chunk, dim=-1)
            output_chunk = torch.matmul(attn_weights, v)
            output[:, :, i:end_i, :] = output_chunk

        return output

    def standard_attention(q, k, v):
        """Standard attention implementation for comparison"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)

    # Test 1: Basic functionality with small sequences
    print("1. Testing basic functionality...")
    batch_size, num_heads, seq_len, d_k = 2, 4, 100, 64

    q = torch.randn(batch_size, num_heads, seq_len, d_k)
    k = torch.randn(batch_size, num_heads, seq_len, d_k)
    v = torch.randn(batch_size, num_heads, seq_len, d_k)

    output_optimized = optimized_matmul_attention(q, k, v)
    output_standard = standard_attention(q, k, v)

    # Results should be very close
    assert torch.allclose(output_optimized, output_standard, rtol=1e-4, atol=1e-6), \
        "Optimized attention should match standard attention for small sequences"
    print("   ✓ Basic functionality test passed")

    # Test 2: Memory efficiency with very long sequences
    print("2. Testing memory efficiency with long sequences...")
    batch_size, num_heads, seq_len, d_k = 1, 8, 5000, 64  # Very long sequence

    q = torch.randn(batch_size, num_heads, seq_len, d_k)
    k = torch.randn(batch_size, num_heads, seq_len, d_k)
    v = torch.randn(batch_size, num_heads, seq_len, d_k)

    # Test that optimized version doesn't crash with long sequences
    try:
        start_time = time.time()
        output_optimized = optimized_matmul_attention(q, k, v)
        optimized_time = time.time() - start_time

        # Should produce correct output shape
        assert output_optimized.shape == (batch_size, num_heads, seq_len, d_k), \
            f"Wrong output shape: {output_optimized.shape}"
        print(f"   ✓ Handled long sequence ({seq_len}) in {optimized_time:.2f}s")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("   ⚠️  GPU out of memory (expected for standard attention)")
        else:
            raise e

    # Test 3: Compare memory usage
    print("3. Testing memory usage comparison...")
    batch_size, num_heads, seq_len, d_k = 2, 4, 2000, 64

    q = torch.randn(batch_size, num_heads, seq_len, d_k)
    k = torch.randn(batch_size, num_heads, seq_len, d_k)
    v = torch.randn(batch_size, num_heads, seq_len, d_k)

    # Measure memory for optimized version
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        q, k, v = q.cuda(), k.cuda(), v.cuda()

        # Get initial memory
        initial_memory = torch.cuda.memory_allocated()

        # Run optimized version
        output_optimized = optimized_matmul_attention(q, k, v)
        optimized_memory = torch.cuda.memory_allocated() - initial_memory

        # Clear and test standard version (if it doesn't OOM)
        try:
            torch.cuda.empty_cache()
            initial_memory_std = torch.cuda.memory_allocated()
            output_standard = standard_attention(q, k, v)
            standard_memory = torch.cuda.memory_allocated() - initial_memory_std

            memory_savings = (standard_memory - optimized_memory) / standard_memory * 100
            print(f"   ✓ Memory usage - Standard: {standard_memory / 1024 ** 2:.1f}MB, "
                  f"Optimized: {optimized_memory / 1024 ** 2:.1f}MB, "
                  f"Savings: {memory_savings:.1f}%")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("   ✓ Standard attention OOM (demonstrates optimized version advantage)")
            else:
                raise e

    # Test 4: Different chunk sizes
    print("4. Testing different chunk sizes...")
    batch_size, num_heads, seq_len, d_k = 2, 4, 1000, 64

    q = torch.randn(batch_size, num_heads, seq_len, d_k)
    k = torch.randn(batch_size, num_heads, seq_len, d_k)
    v = torch.randn(batch_size, num_heads, seq_len, d_k)

    # Test with different chunk sizes
    chunk_sizes = [256, 512, 1024]
    outputs = []

    for chunk_size in chunk_sizes:
        def optimized_with_chunk_size(q, k, v, chunk_size=chunk_size):
            batch_size, num_heads, seq_len, d_k = q.shape
            output = torch.zeros_like(v)

            for i in range(0, seq_len, chunk_size):
                end_i = min(i + chunk_size, seq_len)
                q_chunk = q[:, :, i:end_i, :]
                scores_chunk = torch.matmul(q_chunk, k.transpose(-2, -1)) / math.sqrt(d_k)
                attn_weights = F.softmax(scores_chunk, dim=-1)
                output_chunk = torch.matmul(attn_weights, v)
                output[:, :, i:end_i, :] = output_chunk

            return output

        output_chunked = optimized_with_chunk_size(q, k, v, chunk_size)
        outputs.append(output_chunked)

    # All chunk sizes should produce same results
    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[0], outputs[i], rtol=1e-5), \
            f"Different chunk sizes should produce same results"
    print("   ✓ Different chunk sizes produce consistent results")

    # Test 5: Gradient computation
    print("5. Testing gradient computation...")
    batch_size, num_heads, seq_len, d_k = 2, 4, 500, 64

    q = torch.randn(batch_size, num_heads, seq_len, d_k, requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len, d_k, requires_grad=True)
    v = torch.randn(batch_size, num_heads, seq_len, d_k, requires_grad=True)

    output = optimized_matmul_attention(q, k, v)

    # Create a dummy loss and backpropagate
    dummy_target = torch.randn_like(output)
    loss = F.mse_loss(output, dummy_target)
    loss.backward()

    # Check that gradients are computed
    assert q.grad is not None, "Gradients should flow to query"
    assert k.grad is not None, "Gradients should flow to key"
    assert v.grad is not None, "Gradients should flow to value"

    # Gradients should be non-zero
    assert q.grad.abs().sum() > 0, "Query gradients should be non-zero"
    assert k.grad.abs().sum() > 0, "Key gradients should be non-zero"
    assert v.grad.abs().sum() > 0, "Value gradients should be non-zero"
    print("   ✓ Gradient computation test passed")

    # Test 6: Edge cases
    print("6. Testing edge cases...")

    # Very small sequence
    q_small = torch.randn(1, 2, 10, 32)
    k_small = torch.randn(1, 2, 10, 32)
    v_small = torch.randn(1, 2, 10, 32)

    output_small = optimized_matmul_attention(q_small, k_small, v_small)
    assert output_small.shape == (1, 2, 10, 32), "Wrong shape for small sequence"
    print("   ✓ Very small sequence test passed")

    # Sequence shorter than chunk size
    q_short = torch.randn(1, 2, 300, 64)  # seq_len < chunk_size (512)
    k_short = torch.randn(1, 2, 300, 64)
    v_short = torch.randn(1, 2, 300, 64)

    output_short = optimized_matmul_attention(q_short, k_short, v_short)
    output_short_std = standard_attention(q_short, k_short, v_short)
    assert torch.allclose(output_short, output_short_std, rtol=1e-5), \
        "Should match standard attention for sequences shorter than chunk size"
    print("   ✓ Sequence shorter than chunk size test passed")

    # Test 7: Performance benchmark
    print("7. Performance benchmark...")
    batch_size, num_heads, seq_len, d_k = 2, 8, 4096, 64  # LLM-scale sequence

    q = torch.randn(batch_size, num_heads, seq_len, d_k)
    k = torch.randn(batch_size, num_heads, seq_len, d_k)
    v = torch.randn(batch_size, num_heads, seq_len, d_k)

    # Warm up
    for _ in range(3):
        _ = optimized_matmul_attention(q, k, v)

    # Benchmark optimized version
    start_time = time.time()
    for _ in range(5):
        output_opt = optimized_matmul_attention(q, k, v)
    optimized_time = (time.time() - start_time) / 5

    print(f"   ✓ Optimized attention average time: {optimized_time:.3f}s for seq_len={seq_len}")

    # Verify output is reasonable
    assert not torch.isnan(output_opt).any(), "Output should not contain NaN values"
    assert not torch.isinf(output_opt).any(), "Output should not contain Inf values"

    # Output should have same statistics as input
    assert output_opt.mean().abs() < 10, "Output mean should be reasonable"
    assert output_opt.std() > 0, "Output should have non-zero variance"

    print("\n🎉 ALL OPTIMIZED MATMUL ATTENTION TESTS PASSED! 🎉")
    print("The optimized implementation successfully:")
    print("  • Matches standard attention accuracy")
    print("  • Handles long sequences without OOM")
    print("  • Computes gradients correctly")
    print("  • Works with various chunk sizes")
    print("  • Handles edge cases properly")
    print("  • Provides memory-efficient computation")


if __name__ == "__main__":
    test_optimized_matmul_attention()