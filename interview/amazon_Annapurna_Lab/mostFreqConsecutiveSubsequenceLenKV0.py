from collections import Counter
from typing import List, Any, Tuple, Optional


def find_most_frequent_subsequence(sequence: List[Any], length: int = 3) -> Optional[Tuple[Tuple[Any, ...], int]]:
    """
    Finds the most frequent consecutive sub-sequence of a given length.

    Args:
        sequence: The list to search through. Can contain any hashable elements.
        length: The length of the sub-sequence to find. Defaults to 3.

    Returns:
        A tuple containing the most frequent sub-sequence and its count,
        or None if the sequence is too short.
    """
    if len(sequence) < length:
        print(f"Error: Sequence length must be at least {length}.")
        return None

    # Use a Counter for efficient counting of sub-sequences
    subsequence_counts = Counter()

    # Iterate through the sequence to get all consecutive sub-sequences
    for i in range(len(sequence) - length + 1):
        # Slice the sub-sequence and convert it to a tuple to make it hashable
        subsequence = tuple(sequence[i:i + length])
        subsequence_counts[subsequence] += 1

    if not subsequence_counts:
        return None

    # Find the most common sub-sequence and its count
    most_common_sequence, count = subsequence_counts.most_common(1)[0]

    return most_common_sequence, count


if __name__ == "__main__":
    # Example 1: A list of integers
    my_sequence_int = [1, 2, 3, 4, 2, 3, 5, 2, 3, 4, 6, 7, 2, 3, 4]

    result_int = find_most_frequent_subsequence(my_sequence_int, length=3)
    if result_int:
        sequence, count = result_int
        print(f"Original sequence: {my_sequence_int}")
        print(f"The most frequent consecutive sub-sequence of length 3 is {sequence} with a count of {count}.")

    print("-" * 40)

    # Example 2: A list of strings
    my_sequence_str = ["A", "B", "C", "A", "B", "D", "A", "B", "C", "B", "C", "D"]

    result_str = find_most_frequent_subsequence(my_sequence_str, length=3)
    if result_str:
        sequence, count = result_str
        print(f"Original sequence: {my_sequence_str}")
        print(f"The most frequent consecutive sub-sequence of length 3 is {sequence} with a count of {count}.")
