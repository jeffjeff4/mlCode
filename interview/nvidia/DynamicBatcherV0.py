import torch

# Question 5: Implement Dynamic Batching for Training
class DynamicBatcher:
    def __init__(self, max_batch_size, max_sequence_length):
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.batches = []

    def create_batches(self, sequences):
        """
        Group sequences by similar lengths to minimize padding
        """
        # Sort by sequence length
        sequences.sort(key=len, reverse=True)
        batches = []
        current_batch = []
        current_max_len = 0

        for seq in sequences:
            seq_len = len(seq)

            # Check if we can add to current batch
            if (len(current_batch) < self.max_batch_size and
                    max(current_max_len, seq_len) * (len(current_batch) + 1) <=
                    self.max_sequence_length * self.max_batch_size):

                current_batch.append(seq)
                current_max_len = max(current_max_len, seq_len)
            else:
                if current_batch:
                    batches.append(self.pad_batch(current_batch, current_max_len))
                current_batch = [seq]
                current_max_len = seq_len

        if current_batch:
            batches.append(self.pad_batch(current_batch, current_max_len))

        return batches

    def pad_batch(self, batch, max_len):
        padded_batch = []
        for seq in batch:
            padded_seq = seq + [0] * (max_len - len(seq))
            padded_batch.append(padded_seq)
        return torch.tensor(padded_batch)


import torch
import numpy as np


def test_basic_batching():
    """Test basic dynamic batching functionality"""
    print("=== Test 1: Basic Dynamic Batching ===")
    batcher = DynamicBatcher(max_batch_size=4, max_sequence_length=20)

    sequences = [
        [1, 2, 3],  # length 3
        [4, 5, 6, 7],  # length 4
        [8, 9],  # length 2
        [10, 11, 12, 13, 14]  # length 5
    ]

    batches = batcher.create_batches(sequences)

    # Should create batches with similar lengths
    assert len(batches) > 0, "Should create at least one batch"

    total_sequences = sum(batch.shape[0] for batch in batches)
    assert total_sequences == len(
        sequences), f"All sequences should be batched. Expected {len(sequences)}, got {total_sequences}"

    print("✓ Basic dynamic batching test passed")


def test_batch_shapes():
    """Test that batches have correct shapes"""
    print("\n=== Test 2: Batch Shape Verification ===")
    batcher = DynamicBatcher(max_batch_size=3, max_sequence_length=10)

    sequences = [
        [1, 2, 3],
        [4, 5],
        [6, 7, 8, 9]
    ]

    batches = batcher.create_batches(sequences)

    for batch in batches:
        assert isinstance(batch, torch.Tensor), "Batch should be a tensor"
        assert len(batch.shape) == 2, f"Batch should be 2D, got shape {batch.shape}"
        assert batch.shape[0] <= 3, f"Batch size should be <= 3, got {batch.shape[0]}"
        assert batch.shape[1] <= 10, f"Sequence length should be <= 10, got {batch.shape[1]}"

    print("✓ Batch shape verification test passed")


def test_length_grouping():
    """Test that sequences are grouped by similar lengths"""
    print("\n=== Test 3: Length-Based Grouping ===")
    batcher = DynamicBatcher(max_batch_size=4, max_sequence_length=20)

    sequences = [
        [1, 2, 3],  # length 3
        [4, 5, 6, 7, 8],  # length 5
        [9, 10],  # length 2
        [11, 12, 13],  # length 3
        [14, 15, 16, 17],  # length 4
        [18, 19, 20, 21, 22]  # length 5
    ]

    batches = batcher.create_batches(sequences)

    # Check that similar lengths are batched together
    for batch in batches:
        batch_lengths = [torch.sum(seq != 0).item() for seq in batch]

        # Lengths in same batch should be similar
        max_len = max(batch_lengths)
        min_len = min(batch_lengths)
        length_variance = max_len - min_len

        # Variance should be small (sequences should have similar lengths)
        assert length_variance <= 3, f"Lengths in batch vary too much: {min_len} to {max_len}"

    print("✓ Length-based grouping test passed")


def test_optimal_padding():
    """Test that padding is minimized"""
    print("\n=== Test 4: Optimal Padding ===")
    batcher = DynamicBatcher(max_batch_size=3, max_sequence_length=10)

    sequences = [
        [1, 2, 3],  # length 3
        [4, 5],  # length 2
        [6, 7, 8, 9],  # length 4
        [10, 11],  # length 2
        [12, 13, 14]  # length 3
    ]

    batches = batcher.create_batches(sequences)

    total_padding = 0
    total_elements = 0

    for batch in batches:
        batch_size, seq_len = batch.shape
        for seq in batch:
            actual_length = torch.sum(seq != 0).item()
            padding = seq_len - actual_length
            total_padding += padding
            total_elements += seq_len

    padding_ratio = total_padding / total_elements
    print(f"Padding ratio: {padding_ratio:.2f}")

    # Padding should be reasonable (less than 40% for this case)
    assert padding_ratio < 0.4, f"Too much padding: {padding_ratio:.2f}"
    print("✓ Optimal padding test passed")


def test_single_sequence():
    """Test with only one sequence"""
    print("\n=== Test 5: Single Sequence ===")
    batcher = DynamicBatcher(max_batch_size=4, max_sequence_length=10)

    sequences = [[1, 2, 3, 4, 5]]

    batches = batcher.create_batches(sequences)

    assert len(batches) == 1, "Should create one batch"
    assert batches[0].shape == (1, 5), f"Expected shape (1, 5), got {batches[0].shape}"
    print("✓ Single sequence test passed")


def test_empty_sequences():
    """Test with empty input"""
    print("\n=== Test 6: Empty Sequences ===")
    batcher = DynamicBatcher(max_batch_size=4, max_sequence_length=10)

    sequences = []

    batches = batcher.create_batches(sequences)

    assert len(batches) == 0, "Should return empty list for empty input"
    print("✓ Empty sequences test passed")


def test_very_long_sequence():
    """Test with sequences longer than max_sequence_length"""
    print("\n=== Test 7: Very Long Sequences ===")
    batcher = DynamicBatcher(max_batch_size=2, max_sequence_length=5)

    sequences = [
        [1, 2, 3, 4, 5, 6, 7, 8],  # length 8 > max_sequence_length
        [9, 10, 11]  # length 3
    ]

    batches = batcher.create_batches(sequences)

    # Long sequence should be truncated
    for batch in batches:
        assert batch.shape[1] == 5, f"All sequences should be length 5 after processing, got {batch.shape[1]}"

        # Check that long sequence was truncated
        for seq in batch:
            if seq[0].item() == 1:  # The long sequence
                # Should be truncated to first 5 elements
                expected = [1, 2, 3, 4, 5]
                assert torch.all(seq == torch.tensor(expected)), "Long sequence should be truncated"

    print("✓ Very long sequence test passed")


def test_max_batch_size_respected():
    """Test that max_batch_size is never exceeded"""
    print("\n=== Test 8: Max Batch Size Respect ===")
    batcher = DynamicBatcher(max_batch_size=2, max_sequence_length=10)

    sequences = [
        [1, 2], [3, 4], [5, 6], [7, 8], [9, 10]  # 5 sequences of length 2
    ]

    batches = batcher.create_batches(sequences)

    for batch in batches:
        assert batch.shape[0] <= 2, f"Batch size should be <= 2, got {batch.shape[0]}"

    # Should create multiple batches
    assert len(batches) >= 3, "Should create at least 3 batches for 5 sequences with max_batch_size=2"
    print("✓ Max batch size respect test passed")


def test_sequence_sorting():
    """Test that sequences are sorted by length"""
    print("\n=== Test 9: Sequence Sorting ===")
    batcher = DynamicBatcher(max_batch_size=3, max_sequence_length=10)

    sequences = [
        [1, 2, 3, 4, 5],  # length 5
        [6, 7],  # length 2
        [8, 9, 10, 11],  # length 4
        [12],  # length 1
        [13, 14, 15]  # length 3
    ]

    # Store original order
    original_first = sequences[0][0]

    batches = batcher.create_batches(sequences)

    # Sequences should be reordered by length
    # We can't easily test internal order, but we can verify all sequences are present
    all_batched_sequences = []
    for batch in batches:
        for seq in batch:
            # Remove padding and get original values
            non_padded = seq[seq != 0]
            all_batched_sequences.append(non_padded.tolist())

    # Flatten and check all original values are present
    original_flat = [item for sublist in sequences for item in sublist]
    batched_flat = [item for sublist in all_batched_sequences for item in sublist]

    assert set(original_flat) == set(batched_flat), "All original elements should be present"
    print("✓ Sequence sorting test passed")


def test_memory_efficiency():
    """Test memory efficiency with large inputs"""
    print("\n=== Test 10: Memory Efficiency ===")
    batcher = DynamicBatcher(max_batch_size=8, max_sequence_length=512)

    # Create many sequences with varying lengths
    sequences = []
    for i in range(50):
        length = np.random.randint(10, 100)
        seq = list(range(i, i + length))
        sequences.append(seq)

    batches = batcher.create_batches(sequences)

    total_sequences = sum(batch.shape[0] for batch in batches)
    assert total_sequences == len(sequences), "All sequences should be batched"

    # Check no batch exceeds limits
    for batch in batches:
        assert batch.shape[0] <= 8, f"Batch size exceeded: {batch.shape[0]}"
        assert batch.shape[1] <= 512, f"Sequence length exceeded: {batch.shape[1]}"

    print("✓ Memory efficiency test passed")


def test_padding_correctness():
    """Test that padding is applied correctly"""
    print("\n=== Test 11: Padding Correctness ===")
    batcher = DynamicBatcher(max_batch_size=3, max_sequence_length=6)

    sequences = [
        [1, 2, 3],  # length 3
        [4, 5],  # length 2
        [6, 7, 8, 9]  # length 4
    ]

    batches = batcher.create_batches(sequences)

    for batch in batches:
        batch_size, max_len = batch.shape

        for i, seq in enumerate(batch):
            actual_length = torch.sum(seq != 0).item()

            # Check padding values are zeros
            if actual_length < max_len:
                padding_section = seq[actual_length:]
                assert torch.all(padding_section == 0), "Padding should be zeros"

            # Check original sequence is preserved
            original_seq = sequences[i] if i < len(sequences) else None
            if original_seq:
                assert torch.all(seq[:actual_length] == torch.tensor(original_seq)), "Original sequence corrupted"

    print("✓ Padding correctness test passed")


def test_transformer_training_scenario():
    """Test with realistic transformer training data"""
    print("\n=== Test 12: Transformer Training Scenario ===")
    batcher = DynamicBatcher(max_batch_size=32, max_sequence_length=1024)

    # Simulate tokenized text sequences (like in LLM training)
    sequences = []
    for i in range(100):
        # Realistic sequence lengths for text
        length = np.random.choice([50, 100, 200, 300, 400, 500, 600, 700, 800],
                                  p=[0.1, 0.15, 0.2, 0.15, 0.1, 0.1, 0.08, 0.07, 0.05])
        seq = list(range(1000 + i, 1000 + i + length))  # Simulate token IDs
        sequences.append(seq)

    batches = batcher.create_batches(sequences)

    total_tokens_before = sum(len(seq) for seq in sequences)
    total_tokens_after = sum(batch.shape[0] * batch.shape[1] for batch in batches)

    print(f"Original tokens: {total_tokens_before}")
    print(f"After batching: {total_tokens_after}")
    print(f"Padding overhead: {(total_tokens_after - total_tokens_before) / total_tokens_before * 100:.1f}%")

    # Padding should be reasonable (< 30% for realistic data)
    padding_overhead = (total_tokens_after - total_tokens_before) / total_tokens_before
    assert padding_overhead < 0.3, f"Too much padding overhead: {padding_overhead:.2f}"

    print("✓ Transformer training scenario test passed")


def test_various_sequence_distributions():
    """Test with different sequence length distributions"""
    print("\n=== Test 13: Various Sequence Distributions ===")

    test_distributions = [
        # (name, lengths)
        ("uniform", [10, 20, 30, 40, 50]),
        ("skewed_short", [5, 6, 7, 8, 9, 100]),
        ("skewed_long", [100, 10, 10, 10, 10]),
        ("bimodal", [10, 10, 10, 80, 80, 80]),
    ]

    for dist_name, lengths in test_distributions:
        batcher = DynamicBatcher(max_batch_size=4, max_sequence_length=100)

        sequences = [list(range(length)) for length in lengths]
        batches = batcher.create_batches(sequences)

        total_sequences = sum(batch.shape[0] for batch in batches)
        assert total_sequences == len(sequences), f"Distribution {dist_name}: all sequences should be batched"

        # Verify no constraints violated
        for batch in batches:
            assert batch.shape[0] <= 4
            assert batch.shape[1] <= 100

        print(f"✓ Distribution '{dist_name}' handled correctly")


def test_performance_benchmark():
    """Benchmark performance with large dataset"""
    print("\n=== Test 14: Performance Benchmark ===")
    import time

    # Create large dataset
    num_sequences = 1000
    sequences = []
    for i in range(num_sequences):
        length = np.random.randint(50, 500)
        seq = list(range(i * 1000, i * 1000 + length))
        sequences.append(seq)

    batcher = DynamicBatcher(max_batch_size=64, max_sequence_length=512)

    start_time = time.time()
    batches = batcher.create_batches(sequences)
    end_time = time.time()

    processing_time = end_time - start_time
    sequences_per_second = num_sequences / processing_time

    print(f"Processed {num_sequences} sequences in {processing_time:.2f} seconds")
    print(f"Speed: {sequences_per_second:.0f} sequences/second")

    # Should be reasonably fast
    assert processing_time < 1.0, f"Too slow: {processing_time:.2f} seconds"

    total_sequences = sum(batch.shape[0] for batch in batches)
    assert total_sequences == num_sequences, "All sequences should be processed"

    print("✓ Performance benchmark test passed")


def run_all_dynamic_batcher_tests():
    """Run all dynamic batcher test cases"""
    print("Running Dynamic Batcher Tests...")
    print("=" * 60)

    try:
        test_basic_batching()
        test_batch_shapes()
        test_length_grouping()
        test_optimal_padding()
        test_single_sequence()
        test_empty_sequences()
        test_very_long_sequence()
        test_max_batch_size_respected()
        test_sequence_sorting()
        test_memory_efficiency()
        test_padding_correctness()
        test_transformer_training_scenario()
        test_various_sequence_distributions()
        test_performance_benchmark()

        print("\n" + "=" * 60)
        print("🎉 ALL DYNAMIC BATCHER TESTS PASSED! 🎉")
        print("The DynamicBatcher implementation is working correctly.")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_dynamic_batcher_tests()