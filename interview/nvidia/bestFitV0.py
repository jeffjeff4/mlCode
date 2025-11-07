####Question:
####"""
####In LLM supervised finetuning, one way to reduce the number of pad tokens during training
####is to pack (concatenate) several sequences into a single sequence.
####
##### This is an example of a batch. 0 means padding, 1-8 mean a non-pad token
####[[1, 0, 0, 0, 0, 0, 0, 0, 0],   # 1
#### [2, 2, 2, 2, 2, 0, 0, 0, 0],   # 5
#### [3, 3, 0, 0, 0, 0, 0, 0, 0],   # 2
#### [4, 4, 4, 0, 0, 0, 0, 0, 0],   # 3
#### [5, 5, 5, 5, 5, 5, 5, 5, 5],   # 9
#### [6, 6, 6, 0, 0, 0, 0, 0, 0],   # 3
#### [7, 7, 7, 7, 7, 7, 7, 7, 7],   # 9
#### [8, 0, 0, 0, 0, 0, 0, 0, 0]]   # 1
####
####We can pack this batch of 8 elements into a batch of 4.
####[[1, 2, 2, 2, 2, 2, 3, 3, 8],   # [1, 5, 2, 1]
#### [4, 4, 4, 6, 6, 6, 0, 0, 0],   # [3, 3]
#### [5, 5, 5, 5, 5, 5, 5, 5, 5],   # [9]
#### [7, 7, 7, 7, 7, 7, 7, 7, 7]]   # [9]
####
####
####Your task: write a function that performs efficient sequence packing.
####==========================================================
####Args:
####  seq_lens: Iterator[int]: An iterator of sequence lengths to be packed. You're
####           not allowed to convert this back to a list.
####  pack_size: int: The maximum allowed size in each pack. It's possible that
####           pack_size >> max(seq_lens)
####
####Return:
####    A list of list of sequence lengths representing the packed sequences.
####    The sum of sequences in each pack must be less than or equal to the pack_size.
####
####Example:
####  seq_lens = iter([1, 5, 2, 3, 9, 3, 9, 1]), pack_size = 9
####  These are the possible outputs. You only need to return one output.
####  [[1, 5, 2, 1], [3, 3], [9], [9]] (shown above)
####  [[1, 5, 2], [3, 3, 1], [9], [9]]
####  [[1, 5, 3], [2, 3, 1], [9], [9]]
####  (... and many more)
####
####  This is a valid output, but a worse packing strategy than the above (5 packs vs 4)
####  [[1, 5], [2, 3], [9], [3, 1], [9]]
####
####We want packing to be as tight as possible to minimize the number of pad tokens.
####In other words, we want to minimize the number of packs.
####The optimal solution is NP-hard. We're looking for an approximate
####algorithm based on simple heuristics.
####"""
from typing import Iterator, List


def pack_sequence(seq_lens: Iterator, pack_size: int) -> List[List[int]]:
    import sys
    from typing import Iterator, List

    def pack_sequence(seq_lens: Iterator[int], pack_size: int) -> List[List[int]]:
        """
        Packs sequences from an iterator into 'packs' of a maximum size.

        This function implements a 'Best Fit' greedy heuristic to minimize
        the number of packs used. It processes sequences one by one from the
        iterator without converting them to a list.

        Args:
            seq_lens: An iterator of sequence lengths.
            pack_size: The maximum allowed sum of lengths in a single pack.

        Returns:
            A list of lists, where each inner list represents a pack
            containing one or more sequence lengths.

        Raises:
            ValueError: If pack_size is not positive, or if a sequence
                        length is found that is larger than the pack_size.
        """
        if pack_size <= 0:
            raise ValueError(f"pack_size must be positive, but got {pack_size}")

        # This will store the final lists of sequence lengths
        packs: List[List[int]] = []

        # This will store the current sum of each pack to quickly check capacity
        pack_sums: List[int] = []

        # Process each sequence length from the iterator
        for seq_len in seq_lens:

            # --- Handle Edge Cases ---

            # A sequence of length 0 takes no space and can be ignored
            if seq_len == 0:
                continue

            # A sequence larger than the pack size is an invalid input
            if seq_len > pack_size:
                raise ValueError(
                    f"Encountered sequence of length {seq_len}, "
                    f"which is greater than pack_size {pack_size}."
                )

            # --- Find Best Fit ---
            best_pack_index = -1

            # We're looking for the *tightest* fit.
            # This is the pack that will have the minimum remaining capacity
            # *after* the sequence is added.
            # We initialize this to a value larger than any possible capacity.
            min_remaining_capacity = pack_size + 1

            # Check all *existing* packs for the best fit
            for i in range(len(pack_sums)):
                current_sum = pack_sums[i]
                remaining_capacity = pack_size - current_sum

                # Check if the sequence fits in this pack
                if seq_len <= remaining_capacity:
                    # It fits. Calculate the remaining capacity *after* adding.
                    capacity_after_add = remaining_capacity - seq_len

                    # Is this fit better (tighter) than the best one we've found?
                    if capacity_after_add < min_remaining_capacity:
                        min_remaining_capacity = capacity_after_add
                        best_pack_index = i

            # --- Place the Sequence ---
            if best_pack_index != -1:
                # We found a suitable pack. Add the sequence to it.
                packs[best_pack_index].append(seq_len)
                pack_sums[best_pack_index] += seq_len
            else:
                # No existing pack could fit this sequence.
                # We must open a new pack.
                packs.append([seq_len])
                pack_sums.append(seq_len)

        return packs

    # --- Analysis ---

    """
    ## ⚙️ Complexity Analysis

    Let $N$ be the total number of sequences in the `seq_lens` iterator.
    Let $P$ be the final number of packs created.

    In the worst case, $P$ can be equal to $N$ (e.g., if `pack_size` is 10 and 
    all sequences have a length of 6, no two sequences can be combined).

    * **Time Complexity: $O(N \cdot P)$ or $O(N^2)$**
        * The outer loop iterates $N$ times (once for each sequence).
        * Inside the loop, we iterate through all $P$ currently existing packs to find the "best fit".
        * Since $P$ can grow up to $N$, the total time complexity is $1 + 2 + 3 + ... + N$, which is $O(N^2)$ in the worst case.
        * In the best case (e.g., all sequences fit in the first pack), the inner loop is $O(1)$, and the total time is $O(N)$.

    * **Space Complexity: $O(N + P)$ or $O(N)$**
        * The `packs` list stores $N$ integers in total, distributed across $P$ sub-lists. The storage for the integers is $O(N)$. The storage for the sub-list objects is $O(P)$.
        * The `pack_sums` list stores $P$ integers.
        * The total space is $O(N + P)$.
        * Since $P \le N$, the worst-case space complexity is $O(N + N)$, which simplifies to $O(N)$.
    """

    # --- Evaluation (Test Cases) ---

    def run_test_case(name: str, seq_lens: List[int], pack_size: int):
        """Helper function to run and print test cases."""
        print(f"--- {name} ---")
        print(f"Input Seqs: {seq_lens}")
        print(f"Pack Size:  {pack_size}")

        # We must pass an iterator, not a list
        iterator = iter(seq_lens)

        try:
            packed_seqs = pack_sequence(iterator, pack_size)
            print(f"Output Packs: {packed_seqs}")

            # Verification
            num_packs = len(packed_seqs)
            total_items = sum(len(p) for p in packed_seqs)
            print(f"Stats: {num_packs} packs used for {total_items} items.")

            # Check constraints
            for i, pack in enumerate(packed_seqs):
                pack_sum = sum(pack)
                if pack_sum > pack_size:
                    print(f"!! ERROR: Pack {i} sum {pack_sum} > {pack_size}")

        except ValueError as e:
            print(f"Caught Expected Error: {e}")
        except Exception as e:
            print(f"!! UNEXPECTED ERROR: {e}")
        print("-" * (len(name) + 8) + "\n")

    if __name__ == "__main__":
        print("### Running Sequence Packing Test Cases ###\n")

        # Test 1: The example from the prompt
        run_test_case(
            "Test 1: Provided Example",
            [1, 5, 2, 3, 9, 3, 9, 1],
            9
        )

        # Test 2: All sequences fit in one pack
        run_test_case(
            "Test 2: All Fit in One",
            [1, 2, 3, 2, 1],
            10
        )

        # Test 3: No two sequences fit together
        run_test_case(
            "Test 3: No Two Fit",
            [6, 6, 6, 6, 6],
            10
        )

        # Test 4: "Best Fit" vs "First Fit" difference
        # A "First Fit" (put in first available) might do: [[6, 2], [7, 2], [1]] (3 packs)
        # Our "Best Fit" should do: [[6, 2], [7, 1], [2]] (3 packs)
        # Let's trace...
        # [6] (sum 6)
        # [7] -> [6], [7] (sum 6, 7)
        # [1] -> best fit is [7] -> [6], [7, 1] (sum 6, 8)
        # [2] -> best fit is [6] -> [6, 2], [7, 1] (sum 8, 8)
        # This is a good result.
        run_test_case(
            "Test 4: Best Fit Example",
            [6, 7, 1, 2],
            8
        )

        # Test 5: Empty input
        run_test_case(
            "Test 5: Empty Input",
            [],
            9
        )

        # Test 6: Input with zero-length sequences
        run_test_case(
            "Test 6: With Zero-Length Seqs",
            [1, 5, 0, 2, 0, 0, 3],
            9
        )

        # Test 7: Edge case - oversized sequence
        run_test_case(
            "Test 7: Oversized Sequence",
            [1, 5, 10, 2],
            9
        )

        # Test 8: Edge case - invalid pack size
        run_test_case(
            "Test 8: Invalid Pack Size",
            [1, 2, 3],
            0
        )

        # Test 9: Sequences exactly equal to pack size
        run_test_case(
            "Test 9: Exact Size Seqs",
            [9, 9, 9],
            9
        )

