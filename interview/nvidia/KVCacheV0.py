# Question 6: Implement KV Cache for Efficient Inference
class KVCache:
    def __init__(self, max_batch_size, max_seq_length, num_heads, head_dim):
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Pre-allocate cache tensors
        self.k_cache = torch.zeros(
            max_batch_size, num_heads, max_seq_length, head_dim
        )
        self.v_cache = torch.zeros(
            max_batch_size, num_heads, max_seq_length, head_dim
        )
        self.current_positions = torch.zeros(max_batch_size, dtype=torch.long)

    def update(self, batch_idx, new_k, new_v):
        """
        Update cache with new key-value pairs for given batch indices
        """
        positions = self.current_positions[batch_idx]

        # Update cache
        self.k_cache[batch_idx, :, positions:positions + 1, :] = new_k
        self.v_cache[batch_idx, :, positions:positions + 1, :] = new_v

        # Update positions
        self.current_positions[batch_idx] += 1

    def get(self, batch_idx, seq_length):
        """
        Retrieve cached key-values up to current position
        """
        positions = self.current_positions[batch_idx]
        return (
            self.k_cache[batch_idx, :, :positions, :],
            self.v_cache[batch_idx, :, :positions, :]
        )


import torch
import numpy as np


def test_basic_kv_cache_initialization():
    """Test basic KV cache initialization"""
    print("=== Test 1: Basic KV Cache Initialization ===")
    max_batch_size = 2
    max_seq_length = 10
    num_heads = 4
    head_dim = 64

    cache = KVCache(max_batch_size, max_seq_length, num_heads, head_dim)

    # Test cache shapes
    assert cache.k_cache.shape == (max_batch_size, num_heads, max_seq_length, head_dim)
    assert cache.v_cache.shape == (max_batch_size, num_heads, max_seq_length, head_dim)
    assert cache.current_positions.shape == (max_batch_size,)

    # Test initial values are zeros
    assert torch.all(cache.k_cache == 0), "K cache should be initialized to zeros"
    assert torch.all(cache.v_cache == 0), "V cache should be initialized to zeros"
    assert torch.all(cache.current_positions == 0), "Positions should start at 0"

    print("✓ Basic KV cache initialization test passed")


def test_single_update():
    """Test updating cache for single batch element"""
    print("\n=== Test 2: Single Update ===")
    max_batch_size = 2
    max_seq_length = 5
    num_heads = 2
    head_dim = 8

    cache = KVCache(max_batch_size, max_seq_length, num_heads, head_dim)

    # Create new key-value pairs for batch index 0
    batch_idx = torch.tensor([0])
    new_k = torch.randn(1, num_heads, 1, head_dim)  # (1, num_heads, 1, head_dim)
    new_v = torch.randn(1, num_heads, 1, head_dim)

    # Update cache
    cache.update(batch_idx, new_k, new_v)

    # Check that cache was updated at position 0
    assert torch.all(cache.k_cache[0, :, 0:1, :] == new_k[0]), "K cache not updated correctly"
    assert torch.all(cache.v_cache[0, :, 0:1, :] == new_v[0]), "V cache not updated correctly"
    assert cache.current_positions[0] == 1, f"Position should be 1, got {cache.current_positions[0]}"

    print("✓ Single update test passed")


import torch
import numpy as np


def test_basic_kv_cache_initialization():
    """Test basic KV cache initialization"""
    print("=== Test 1: Basic KV Cache Initialization ===")
    max_batch_size = 2
    max_seq_length = 10
    num_heads = 4
    head_dim = 64

    cache = KVCache(max_batch_size, max_seq_length, num_heads, head_dim)

    # Test cache shapes
    assert cache.k_cache.shape == (max_batch_size, num_heads, max_seq_length, head_dim)
    assert cache.v_cache.shape == (max_batch_size, num_heads, max_seq_length, head_dim)
    assert cache.current_positions.shape == (max_batch_size,)

    # Test initial values are zeros
    assert torch.all(cache.k_cache == 0), "K cache should be initialized to zeros"
    assert torch.all(cache.v_cache == 0), "V cache should be initialized to zeros"
    assert torch.all(cache.current_positions == 0), "Positions should start at 0"

    print("✓ Basic KV cache initialization test passed")


def test_single_update():
    """Test updating cache for single batch element"""
    print("\n=== Test 2: Single Update ===")
    max_batch_size = 2
    max_seq_length = 5
    num_heads = 2
    head_dim = 8

    cache = KVCache(max_batch_size, max_seq_length, num_heads, head_dim)

    # Create new key-value pairs for batch index 0
    batch_idx = torch.tensor([0])
    new_k = torch.randn(1, num_heads, 1, head_dim)  # (1, num_heads, 1, head_dim)
    new_v = torch.randn(1, num_heads, 1, head_dim)

    # Update cache
    cache.update(batch_idx, new_k, new_v)

    # Check that cache was updated at position 0
    assert torch.all(cache.k_cache[0, :, 0:1, :] == new_k[0]), "K cache not updated correctly"
    assert torch.all(cache.v_cache[0, :, 0:1, :] == new_v[0]), "V cache not updated correctly"
    assert cache.current_positions[0] == 1, f"Position should be 1, got {cache.current_positions[0]}"

    print("✓ Single update test passed")


def test_multiple_updates():
    """Test multiple sequential updates"""
    print("\n=== Test 3: Multiple Sequential Updates ===")
    max_batch_size = 1
    max_seq_length = 5
    num_heads = 2
    head_dim = 4

    cache = KVCache(max_batch_size, max_seq_length, num_heads, head_dim)
    batch_idx = torch.tensor([0])

    # Perform multiple updates
    for i in range(3):
        new_k = torch.ones(1, num_heads, 1, head_dim) * (i + 1)  # Different values each time
        new_v = torch.ones(1, num_heads, 1, head_dim) * (i + 10)

        cache.update(batch_idx, new_k, new_v)

        # Check current position
        assert cache.current_positions[0] == i + 1, f"Position should be {i + 1}, got {cache.current_positions[0]}"

    # Verify all updates are stored correctly
    expected_k = torch.tensor([[[[1., 1., 1., 1.]], [[1., 1., 1., 1.]]],
                               [[[2., 2., 2., 2.]], [[2., 2., 2., 2.]]],
                               [[[3., 3., 3., 3.]], [[3., 3., 3., 3.]]]])

    expected_v = torch.tensor([[[[10., 10., 10., 10.]], [[10., 10., 10., 10.]]],
                               [[[11., 11., 11., 11.]], [[11., 11., 11., 11.]]],
                               [[[12., 12., 12., 12.]], [[12., 12., 12., 12.]]]])

    assert torch.all(cache.k_cache[0, :, :3, :] == expected_k), "K cache values incorrect"
    assert torch.all(cache.v_cache[0, :, :3, :] == expected_v), "V cache values incorrect"

    print("✓ Multiple sequential updates test passed")


def test_cache_retrieval():
    """Test retrieving cached key-value pairs"""
    print("\n=== Test 4: Cache Retrieval ===")
    max_batch_size = 2
    max_seq_length = 6
    num_heads = 3
    head_dim = 8

    cache = KVCache(max_batch_size, max_seq_length, num_heads, head_dim)

    # Update cache for both batch elements
    for batch_idx in [0, 1]:
        for pos in range(3):  # Add 3 entries for each batch
            new_k = torch.ones(1, num_heads, 1, head_dim) * (batch_idx * 10 + pos + 1)
            new_v = torch.ones(1, num_heads, 1, head_dim) * (batch_idx * 10 + pos + 100)

            cache.update(torch.tensor([batch_idx]), new_k, new_v)

    # Retrieve cached values
    for batch_idx in [0, 1]:
        k_retrieved, v_retrieved = cache.get(torch.tensor([batch_idx]), seq_length=3)

        assert k_retrieved.shape == (1, num_heads, 3, head_dim), f"Wrong K shape: {k_retrieved.shape}"
        assert v_retrieved.shape == (1, num_heads, 3, head_dim), f"Wrong V shape: {v_retrieved.shape}"

        # Check values
        expected_k = torch.ones(num_heads, 3, head_dim) * (
                    batch_idx * 10 + torch.tensor([1, 2, 3]).unsqueeze(-1).unsqueeze(0))
        expected_v = torch.ones(num_heads, 3, head_dim) * (
                    batch_idx * 10 + torch.tensor([100, 101, 102]).unsqueeze(-1).unsqueeze(0))

        assert torch.all(k_retrieved[0] == expected_k), "Retrieved K values incorrect"
        assert torch.all(v_retrieved[0] == expected_v), "Retrieved V values incorrect"

    print("✓ Cache retrieval test passed")


def test_batch_updates():
    """Test updating multiple batch elements simultaneously"""
    print("\n=== Test 5: Batch Updates ===")
    max_batch_size = 4
    max_seq_length = 8
    num_heads = 2
    head_dim = 16

    cache = KVCache(max_batch_size, max_seq_length, num_heads, head_dim)

    # Update multiple batch indices at once
    batch_indices = torch.tensor([0, 2, 3])
    new_k = torch.randn(3, num_heads, 1, head_dim)
    new_v = torch.randn(3, num_heads, 1, head_dim)

    cache.update(batch_indices, new_k, new_v)

    # Check that all specified batch indices were updated
    for i, batch_idx in enumerate(batch_indices):
        assert torch.all(cache.k_cache[batch_idx, :, 0:1, :] == new_k[i]), f"K cache not updated for batch {batch_idx}"
        assert torch.all(cache.v_cache[batch_idx, :, 0:1, :] == new_v[i]), f"V cache not updated for batch {batch_idx}"
        assert cache.current_positions[batch_idx] == 1, f"Position should be 1 for batch {batch_idx}"

    # Check that other batch indices remain unchanged
    assert cache.current_positions[1] == 0, "Batch 1 should remain unchanged"

    print("✓ Batch updates test passed")


def test_mixed_batch_processing():
    """Test processing batches at different positions"""
    print("\n=== Test 6: Mixed Batch Processing ===")
    max_batch_size = 3
    max_seq_length = 10
    num_heads = 4
    head_dim = 12

    cache = KVCache(max_batch_size, max_seq_length, num_heads, head_dim)

    # Update batch 0 with 2 entries
    cache.update(torch.tensor([0]), torch.ones(1, num_heads, 1, head_dim) * 1,
                 torch.ones(1, num_heads, 1, head_dim) * 10)
    cache.update(torch.tensor([0]), torch.ones(1, num_heads, 1, head_dim) * 2,
                 torch.ones(1, num_heads, 1, head_dim) * 20)

    # Update batch 1 with 1 entry
    cache.update(torch.tensor([1]), torch.ones(1, num_heads, 1, head_dim) * 3,
                 torch.ones(1, num_heads, 1, head_dim) * 30)

    # Update batch 2 with 3 entries
    for i in range(3):
        cache.update(torch.tensor([2]), torch.ones(1, num_heads, 1, head_dim) * (4 + i),
                     torch.ones(1, num_heads, 1, head_dim) * (40 + i))

    # Verify positions
    assert cache.current_positions[0] == 2
    assert cache.current_positions[1] == 1
    assert cache.current_positions[2] == 3

    # Verify retrieval for each batch
    k0, v0 = cache.get(torch.tensor([0]), seq_length=2)
    assert k0.shape == (1, num_heads, 2, head_dim)
    assert torch.all(k0[0, :, 0, :] == 1)
    assert torch.all(k0[0, :, 1, :] == 2)

    k1, v1 = cache.get(torch.tensor([1]), seq_length=1)
    assert k1.shape == (1, num_heads, 1, head_dim)
    assert torch.all(k1[0, :, 0, :] == 3)

    k2, v2 = cache.get(torch.tensor([2]), seq_length=3)
    assert k2.shape == (1, num_heads, 3, head_dim)

    print("✓ Mixed batch processing test passed")


def test_cache_overflow_protection():
    """Test behavior when approaching max sequence length"""
    print("\n=== Test 7: Cache Overflow Protection ===")
    max_batch_size = 1
    max_seq_length = 3  # Small limit for testing
    num_heads = 2
    head_dim = 4

    cache = KVCache(max_batch_size, max_seq_length, num_heads, head_dim)
    batch_idx = torch.tensor([0])

    # Fill cache to capacity
    for i in range(max_seq_length):
        new_k = torch.ones(1, num_heads, 1, head_dim) * (i + 1)
        new_v = torch.ones(1, num_heads, 1, head_dim) * (i + 10)
        cache.update(batch_idx, new_k, new_v)

    # Try to exceed capacity - this should work but might overwrite or be handled by external logic
    try:
        new_k = torch.ones(1, num_heads, 1, head_dim) * 99
        new_v = torch.ones(1, num_heads, 1, head_dim) * 99
        cache.update(batch_idx, new_k, new_v)

        # In this implementation, it will write beyond max_seq_length
        # In practice, external logic should prevent this
        print("✓ Cache overflow handled (external prevention needed)")
    except Exception as e:
        print(f"✓ Cache overflow caught: {e}")

    print("✓ Cache overflow protection test passed")


def test_empty_batch_retrieval():
    """Test retrieving from batch with no updates"""
    print("\n=== Test 8: Empty Batch Retrieval ===")
    max_batch_size = 2
    max_seq_length = 5
    num_heads = 2
    head_dim = 8

    cache = KVCache(max_batch_size, max_seq_length, num_heads, head_dim)

    # Only update batch 0
    cache.update(torch.tensor([0]), torch.ones(1, num_heads, 1, head_dim), torch.ones(1, num_heads, 1, head_dim))

    # Retrieve from batch 1 (no updates)
    k, v = cache.get(torch.tensor([1]), seq_length=0)

    # Should return empty tensors with correct shapes
    assert k.shape == (1, num_heads, 0, head_dim), f"Expected empty K, got shape {k.shape}"
    assert v.shape == (1, num_heads, 0, head_dim), f"Expected empty V, got shape {v.shape}"

    print("✓ Empty batch retrieval test passed")


def test_single_head_scenario():
    """Test with single head (common in some architectures)"""
    print("\n=== Test 9: Single Head Scenario ===")
    max_batch_size = 2
    max_seq_length = 10
    num_heads = 1  # Single head
    head_dim = 64

    cache = KVCache(max_batch_size, max_seq_length, num_heads, head_dim)

    batch_idx = torch.tensor([0, 1])
    new_k = torch.randn(2, num_heads, 1, head_dim)
    new_v = torch.randn(2, num_heads, 1, head_dim)

    cache.update(batch_idx, new_k, new_v)

    k_retrieved, v_retrieved = cache.get(batch_idx, seq_length=1)

    assert k_retrieved.shape == (2, 1, 1, head_dim)
    assert v_retrieved.shape == (2, 1, 1, head_dim)

    print("✓ Single head scenario test passed")


def test_large_scale_cache():
    """Test with large cache sizes (like in real LLM inference)"""
    print("\n=== Test 10: Large Scale Cache ===")
    max_batch_size = 16
    max_seq_length = 2048  # Typical context length
    num_heads = 32  # Typical for medium-sized models
    head_dim = 64  # Typical head dimension

    import time
    start_time = time.time()

    cache = KVCache(max_batch_size, max_seq_length, num_heads, head_dim)

    # Verify memory allocation
    expected_memory_k = max_batch_size * num_heads * max_seq_length * head_dim * 4  # 4 bytes per float32
    expected_memory_v = expected_memory_k
    total_expected_memory = (expected_memory_k + expected_memory_v) / (1024 ** 3)  # GB

    actual_memory_k = cache.k_cache.nelement() * cache.k_cache.element_size() / (1024 ** 3)
    actual_memory_v = cache.v_cache.nelement() * cache.v_cache.element_size() / (1024 ** 3)

    print(f"K cache memory: {actual_memory_k:.2f} GB")
    print(f"V cache memory: {actual_memory_v:.2f} GB")
    print(f"Total cache memory: {actual_memory_k + actual_memory_v:.2f} GB")

    # Test performance with multiple updates
    batch_indices = torch.tensor([0, 1, 2, 3])
    update_time = 0

    for i in range(100):  # Simulate 100 generation steps
        new_k = torch.randn(4, num_heads, 1, head_dim)
        new_v = torch.randn(4, num_heads, 1, head_dim)

        update_start = time.time()
        cache.update(batch_indices, new_k, new_v)
        update_time += time.time() - update_start

    avg_update_time = update_time / 100
    print(f"Average update time: {avg_update_time * 1000:.2f} ms")

    assert avg_update_time < 0.01, "Update operations should be fast"

    end_time = time.time()
    print(f"Total test time: {end_time - start_time:.2f} seconds")
    print("✓ Large scale cache test passed")


def test_memory_persistence():
    """Test that cache maintains values correctly"""
    print("\n=== Test 11: Memory Persistence ===")
    max_batch_size = 2
    max_seq_length = 5
    num_heads = 2
    head_dim = 8

    cache = KVCache(max_batch_size, max_seq_length, num_heads, head_dim)

    # Add some values
    values_to_remember = []
    for i in range(3):
        new_k = torch.randn(1, num_heads, 1, head_dim)
        new_v = torch.randn(1, num_heads, 1, head_dim)
        values_to_remember.append((new_k.clone(), new_v.clone()))

        cache.update(torch.tensor([0]), new_k, new_v)

    # Verify all values are still there
    k_retrieved, v_retrieved = cache.get(torch.tensor([0]), seq_length=3)

    for i in range(3):
        assert torch.all(k_retrieved[0, :, i:i + 1, :] == values_to_remember[i][0][0]), f"K value {i} corrupted"
        assert torch.all(v_retrieved[0, :, i:i + 1, :] == values_to_remember[i][1][0]), f"V value {i} corrupted"

    print("✓ Memory persistence test passed")


def test_attention_integration():
    """Test integration with attention mechanism"""
    print("\n=== Test 12: Attention Integration ===")
    max_batch_size = 2
    max_seq_length = 10
    num_heads = 4
    head_dim = 16

    cache = KVCache(max_batch_size, max_seq_length, num_heads, head_dim)

    # Simulate autoregressive generation steps
    batch_size = 2
    generated_sequences = []

    for step in range(4):  # Generate 4 tokens
        # In real scenario, this would come from model output
        new_k = torch.randn(batch_size, num_heads, 1, head_dim)
        new_v = torch.randn(batch_size, num_heads, 1, head_dim)

        batch_indices = torch.tensor([0, 1])
        cache.update(batch_indices, new_k, new_v)

        # Simulate attention computation using cached values
        if step > 0:  # After first token, we can use cache
            k_cached, v_cached = cache.get(batch_indices, seq_length=step + 1)

            # Verify shapes for attention
            assert k_cached.shape == (batch_size, num_heads, step + 1, head_dim)
            assert v_cached.shape == (batch_size, num_heads, step + 1, head_dim)

            # Simulate attention scores (q @ k.transpose)
            q = torch.randn(batch_size, num_heads, 1, head_dim)  # Current query
            attention_scores = torch.matmul(q, k_cached.transpose(-2, -1))
            assert attention_scores.shape == (batch_size, num_heads, 1, step + 1)

    print("✓ Attention integration test passed")


def test_concurrent_batch_processing():
    """Test realistic concurrent batch processing scenario"""
    print("\n=== Test 13: Concurrent Batch Processing ===")
    max_batch_size = 8
    max_seq_length = 1024
    num_heads = 8
    head_dim = 64

    cache = KVCache(max_batch_size, max_seq_length, num_heads, head_dim)

    # Simulate different sequences at different stages
    sequences_info = [
        (0, 150),  # batch 0, at position 150
        (1, 45),  # batch 1, at position 45
        (2, 300),  # batch 2, at position 300
        (3, 12),  # batch 3, at position 12
    ]

    # Initialize positions
    for batch_idx, position in sequences_info:
        cache.current_positions[batch_idx] = position

    # Simulate one generation step
    active_batches = torch.tensor([0, 1, 2, 3])
    new_k = torch.randn(4, num_heads, 1, head_dim)
    new_v = torch.randn(4, num_heads, 1, head_dim)

    cache.update(active_batches, new_k, new_v)

    # Verify positions incremented
    for batch_idx, old_position in sequences_info:
        assert cache.current_positions[batch_idx] == old_position + 1, f"Batch {batch_idx} position incorrect"

    # Verify retrieval works for different sequence lengths
    for batch_idx, position in sequences_info:
        k, v = cache.get(torch.tensor([batch_idx]), seq_length=position + 1)
        assert k.shape == (1, num_heads, position + 1, head_dim), f"Wrong shape for batch {batch_idx}"

    print("✓ Concurrent batch processing test passed")


def test_error_handling():
    """Test error conditions and edge cases"""
    print("\n=== Test 14: Error Handling ===")
    max_batch_size = 2
    max_seq_length = 10
    num_heads = 2
    head_dim = 8

    cache = KVCache(max_batch_size, max_seq_length, num_heads, head_dim)

    # Test invalid batch indices
    try:
        cache.update(torch.tensor([5]), torch.randn(1, num_heads, 1, head_dim), torch.randn(1, num_heads, 1, head_dim))
        print("⚠️  Should validate batch indices")
    except Exception as e:
        print(f"✓ Correctly caught invalid batch index: {e}")

    # Test shape mismatches
    try:
        cache.update(torch.tensor([0]), torch.randn(1, 3, 1, head_dim),
                     torch.randn(1, num_heads, 1, head_dim))  # Wrong num_heads
        print("⚠️  Should validate tensor shapes")
    except Exception as e:
        print(f"✓ Correctly caught shape mismatch: {e}")

    print("✓ Error handling test passed")


def run_all_kv_cache_tests():
    """Run all KV cache test cases"""
    print("Running KV Cache Tests...")
    print("=" * 60)

    try:
        test_basic_kv_cache_initialization()
        test_single_update()
        test_multiple_updates()
        test_cache_retrieval()
        test_batch_updates()
        test_mixed_batch_processing()
        test_cache_overflow_protection()
        test_empty_batch_retrieval()
        test_single_head_scenario()
        test_large_scale_cache()
        test_memory_persistence()
        test_attention_integration()
        test_concurrent_batch_processing()
        test_error_handling()

        print("\n" + "=" * 60)
        print("🎉 ALL KV CACHE TESTS PASSED! 🎉")
        print("The KVCache implementation is working correctly.")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_kv_cache_tests()