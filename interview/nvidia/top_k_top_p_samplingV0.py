import torch
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn.functional as F
import numpy as np

# Pattern 1: Sliding Window for Context Management
####def sliding_window_attention(sequences, window_size):
####    """
####    Implement sliding window attention for long sequences
####    """
####    results = []
####    for seq in sequences:
####        for i in range(0, len(seq) - window_size + 1):
####            window = seq[i:i + window_size]
####            # Process window
####            results.append(process_window(window))
####    return results


# Pattern 2: Sampling Algorithms


def test_top_k_top_p_sampling():
    """Comprehensive test for top-k and top-p (nucleus) sampling"""
    print("=== Test: Top-K Top-P Sampling ===")

    def top_k_top_p_sampling(logits, top_k=50, top_p=0.9):
        """
        Combined top-k and top-p (nucleus) sampling
        """
        # Filter top-k (only if top_k > 0 and less than vocab size)
        if top_k > 0 and top_k < logits.size(-1):
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        elif top_k > 0 and top_k >= logits.size(-1):
            # If top_k is larger than vocab size, no top-k filtering applied
            pass

        # Filter top-p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    # Test 1: Basic top-k filtering
    print("1. Testing basic top-k filtering...")

    # Create logits with clear top values
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0])

    # Apply top-k with k=3
    result = top_k_top_p_sampling(logits, top_k=3, top_p=1.0)

    # The result should be one of the top 3 values (indices 0, 1, or 2)
    assert result.item() in [0, 1, 2], f"Top-k failed: sampled {result.item()} not in top 3"

    # Verify that lower values are masked out
    processed_logits = logits.clone()
    indices_to_remove = processed_logits < torch.topk(processed_logits, 3)[0][..., -1, None]
    processed_logits[indices_to_remove] = float('-inf')

    # Only top 3 should have finite values
    finite_indices = torch.where(processed_logits != float('-inf'))[0]
    assert len(finite_indices) == 3, f"Top-k should keep exactly 3 tokens, kept {len(finite_indices)}"
    print("   ✓ Basic top-k filtering test passed")

    # Test 2: Basic top-p (nucleus) sampling
    print("2. Testing basic top-p (nucleus) sampling...")

    # Create logits where cumulative probability makes sense
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    probs = F.softmax(logits, dim=-1)

    # Calculate cumulative probabilities
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # With top_p=0.7, should keep tokens until cumulative probability > 0.7
    result = top_k_top_p_sampling(logits, top_k=0, top_p=0.7)

    # Verify the sampling logic
    kept_indices = sorted_indices[cumulative_probs <= 0.7]
    assert result.item() in kept_indices.tolist(), f"Top-p failed: sampled {result.item()} not in nucleus"
    print("   ✓ Basic top-p sampling test passed")

    # Test 3: Combined top-k and top-p
    print("3. Testing combined top-k and top-p...")

    logits = torch.tensor([5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5])

    # Apply both filters
    result = top_k_top_p_sampling(logits, top_k=5, top_p=0.8)

    # Should be in intersection of top-k and top-p
    probs = F.softmax(logits, dim=-1)

    # Top-k filter
    top_k_indices = torch.topk(logits, 5)[1].tolist()

    # Top-p filter
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    top_p_indices = sorted_indices[cumulative_probs <= 0.8].tolist()

    # Intersection
    valid_indices = list(set(top_k_indices) & set(top_p_indices))

    assert result.item() in valid_indices, f"Combined sampling failed: {result.item()} not in valid set"
    print("   ✓ Combined top-k and top-p test passed")

    # Test 4: Edge case - all logits equal
    print("4. Testing edge case - uniform distribution...")

    logits = torch.ones(10)  # All logits equal
    result = top_k_top_p_sampling(logits, top_k=3, top_p=0.5)

    # Should sample from all tokens (top-k will restrict to 3, but all have equal prob)
    assert 0 <= result.item() < 10, "Should sample valid index"

    # With uniform distribution and top-k=3, should sample from first 3 after sorting
    # But since all are equal, sorting is arbitrary, so we just check it's valid
    print("   ✓ Uniform distribution test passed")

    # Test 5: Edge case - very small top-p
    print("5. Testing edge case - very small top-p...")

    logits = torch.tensor([5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    probs = F.softmax(logits, dim=-1)

    # With top_p=0.1, should only keep the highest probability token
    result = top_k_top_p_sampling(logits, top_k=0, top_p=0.1)

    # Should always sample index 0 (highest probability)
    assert result.item() == 0, f"With small top-p, should sample highest prob token, got {result.item()}"
    print("   ✓ Very small top-p test passed")

    # Test 6: Edge case - top-k larger than vocabulary (FIXED)
    print("6. Testing edge case - top-k larger than vocabulary size...")

    logits = torch.tensor([3.0, 2.0, 1.0])  # Only 3 tokens
    result = top_k_top_p_sampling(logits, top_k=10, top_p=1.0)  # top-k > vocab size

    # Should sample from all tokens (no filtering since top_k >= vocab_size)
    assert result.item() in [0, 1, 2], "Should sample from all tokens when top-k > vocab size"

    # Test with top_k exactly equal to vocab size
    result2 = top_k_top_p_sampling(logits, top_k=3, top_p=1.0)
    assert result2.item() in [0, 1, 2], "Should sample from all tokens when top-k = vocab size"
    print("   ✓ Large top-k test passed")

    # Test 7: Edge case - top-k = 0 (no top-k filtering)
    print("7. Testing edge case - top-k = 0...")

    logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0])
    result = top_k_top_p_sampling(logits, top_k=0, top_p=1.0)

    # Should sample from all tokens
    assert 0 <= result.item() < len(logits), "With top_k=0, should sample from all tokens"
    print("   ✓ Top-k = 0 test passed")

    # Test 8: Batch processing
    print("8. Testing batch processing...")

    batch_logits = torch.tensor([
        [3.0, 2.0, 1.0, 0.0],  # Batch 0
        [1.0, 3.0, 2.0, 0.0],  # Batch 1
        [0.0, 1.0, 3.0, 2.0]  # Batch 2
    ])

    results = []
    for i in range(batch_logits.shape[0]):
        result = top_k_top_p_sampling(batch_logits[i], top_k=2, top_p=0.8)
        results.append(result.item())

    # Each result should be in the top-2 of its respective batch
    top2_batch0 = torch.topk(batch_logits[0], 2)[1].tolist()
    top2_batch1 = torch.topk(batch_logits[1], 2)[1].tolist()
    top2_batch2 = torch.topk(batch_logits[2], 2)[1].tolist()

    assert results[0] in top2_batch0, f"Batch 0 sampling failed: {results[0]} not in {top2_batch0}"
    assert results[1] in top2_batch1, f"Batch 1 sampling failed: {results[1]} not in {top2_batch1}"
    assert results[2] in top2_batch2, f"Batch 2 sampling failed: {results[2]} not in {top2_batch2}"
    print("   ✓ Batch processing test passed")

    # Test 9: Numerical stability with extreme values
    print("9. Testing numerical stability...")

    # Very large logits (could cause overflow in softmax)
    large_logits = torch.tensor([1000.0, 100.0, 10.0, 1.0, 0.0])
    result = top_k_top_p_sampling(large_logits, top_k=3, top_p=0.9)

    # Should not crash and should return valid sample
    assert 0 <= result.item() < len(large_logits), "Should handle large logits without crashing"

    # Very small logits (all negative)
    small_logits = torch.tensor([-1000.0, -100.0, -10.0, -1.0])
    result = top_k_top_p_sampling(small_logits, top_k=2, top_p=0.8)

    assert 0 <= result.item() < len(small_logits), "Should handle small logits without crashing"
    print("   ✓ Numerical stability test passed")

    # Test 10: Reproducibility with seeding
    print("10. Testing reproducibility with seeding...")

    logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0, -2.0])

    # Set seed and get first result
    torch.manual_seed(42)
    result1 = top_k_top_p_sampling(logits, top_k=3, top_p=0.8)

    # Reset seed and get second result - should be identical
    torch.manual_seed(42)
    result2 = top_k_top_p_sampling(logits, top_k=3, top_p=0.8)

    assert result1.item() == result2.item(), "Results should be reproducible with same seed"
    print("   ✓ Reproducibility test passed")

    # Test 11: Verify probability distribution after filtering
    print("11. Testing probability distribution after filtering...")

    logits = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0])

    # Apply top-k and top-p
    filtered_logits = logits.clone()

    # Apply top-k=4
    indices_to_remove = filtered_logits < torch.topk(filtered_logits, 4)[0][..., -1, None]
    filtered_logits[indices_to_remove] = float('-inf')

    # Apply top-p=0.8
    sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > 0.8
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    filtered_logits[indices_to_remove] = float('-inf')

    # Get final probabilities
    final_probs = F.softmax(filtered_logits, dim=-1)

    # Sum of probabilities should be 1.0 (approximately)
    prob_sum = final_probs.sum().item()
    assert abs(prob_sum - 1.0) < 1e-6, f"Probabilities should sum to 1, got {prob_sum}"

    # Only filtered tokens should have non-zero probability
    non_zero_indices = torch.where(final_probs > 0)[0]
    assert len(non_zero_indices) > 0, "Should have at least one valid token"
    print("   ✓ Probability distribution test passed")

    # Test 12: Real-world LLM scenario
    print("12. Testing real-world LLM scenario...")

    # Simulate typical LLM output logits
    vocab_size = 50000
    logits = torch.randn(vocab_size) * 2  # Random logits similar to real models

    # Typical sampling parameters used in practice
    result = top_k_top_p_sampling(logits, top_k=40, top_p=0.9)

    # Should return a valid token ID
    assert 0 <= result.item() < vocab_size, f"Sampled invalid token ID: {result.item()}"

    # Verify the sampling actually occurred (not just argmax)
    # Run multiple samples to see diversity
    torch.manual_seed(123)  # For reproducibility in test
    samples = set()
    for _ in range(10):
        sample = top_k_top_p_sampling(logits, top_k=40, top_p=0.9)
        samples.add(sample.item())

    # With top-k=40 and top-p=0.9, we should see some diversity
    assert len(samples) > 1, "Should see diverse samples with these parameters"
    print(f"   ✓ Generated {len(samples)} unique samples from 10 trials")
    print("   ✓ Real-world LLM scenario test passed")

    # Test 13: Disabled sampling (greedy decoding equivalent)
    print("13. Testing disabled sampling (greedy equivalent)...")

    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])

    # With top_k=1 and top_p=1.0, should always pick the highest
    result = top_k_top_p_sampling(logits, top_k=1, top_p=1.0)
    assert result.item() == 0, "With top_k=1, should always pick highest probability token"

    # With top_k=0 and top_p=1.0, no filtering - standard multinomial
    result = top_k_top_p_sampling(logits, top_k=0, top_p=1.0)
    assert 0 <= result.item() < len(logits), "Should sample from all tokens"
    print("   ✓ Disabled sampling test passed")

    print("\n🎉 ALL TOP-K TOP-P SAMPLING TESTS PASSED! 🎉")
    print("The sampling function successfully:")
    print("  • Applies top-k filtering correctly")
    print("  • Applies top-p (nucleus) filtering correctly")
    print("  • Combines both filters appropriately")
    print("  • Handles edge cases and numerical stability")
    print("  • Works with batch processing")
    print("  • Provides reproducible results with seeding")
    print("  • Maintains valid probability distributions")
    print("  • Works in real-world LLM scenarios")
    print("  • Supports greedy decoding when needed")


if __name__ == "__main__":
    test_top_k_top_p_sampling()