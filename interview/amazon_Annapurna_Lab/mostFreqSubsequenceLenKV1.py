####python code, with test case
####1. MUST be CORRECT.
####2. consider all cases, including edge cases
####3. make it runnable in this chat session
####4. provide time complexity and space complexity analysis
####5. code clearance, class design needs to be expandable, code organization, make it more like production level code
####
####Question:
####What is the most frequent sub-sequence of length 3? what if we replace 3 with k?



import collections
from typing import List


class SubsequenceFinder:
    """
    A class to find the most frequent subsequence of a given length in a string.
    This implementation uses a dynamic programming approach to solve the problem
    efficiently for a generalized subsequence length 'k'.
    """

    def find_most_frequent(self, text: str, k: int) -> str:
        """
        Finds the most frequent subsequence of length 'k' in the given text.

        The algorithm uses dynamic programming. It iterates through the string,
        building up counts for subsequences of length 1, 2, ..., up to k.
        `dp[i]` stores a counter for all subsequences of length `i`.

        In case of a tie in frequency, the lexicographically smallest subsequence
        is returned.

        Args:
            text: The input string to search within.
            k: The desired length of the subsequence.

        Returns:
            The most frequent subsequence of length k. Returns an empty string
            if no such subsequence can be formed (e.g., if k is non-positive or
            the input string is shorter than k).

        Raises:
            TypeError: If the input `text` is not a string or `k` is not an integer.
        """
        # --- Input Validation and Edge Cases ---
        if not isinstance(text, str):
            raise TypeError("Input 'text' must be a string.")
        if not isinstance(k, int):
            raise TypeError("Input 'k' must be an integer.")

        if k <= 0 or len(text) < k:
            return ""

        # --- Dynamic Programming Initialization ---
        # dp[i] will be a Counter mapping subsequences of length 'i' to their frequency.
        # We need k+1 counters, for lengths 0 through k. dp[0] is a dummy.
        dp: List[collections.Counter] = [collections.Counter() for _ in range(k + 1)]

        # --- Main DP Loop ---
        for char in text:
            # Iterate downwards from k to 1. This is crucial to ensure that when
            # we calculate dp[i], we are using the counts from dp[i-1] *before*
            # the current character is processed.
            for i in range(k, 0, -1):
                # For every subsequence of length i-1 we've seen so far,
                # we can form new subsequences of length i by appending the current 'char'.
                for sub, count in dp[i - 1].items():
                    dp[i][sub + char] += count

            # The current character itself forms a subsequence of length 1.
            dp[1][char] += 1

        # --- Result Finalization ---
        final_counts = dp[k]
        if not final_counts:
            return ""

        # Find the maximum frequency among all subsequences of length k.
        max_frequency = 0
        for freq in final_counts.values():
            if freq > max_frequency:
                max_frequency = freq

        # Filter for all subsequences that have the maximum frequency.
        candidates = [
            sub for sub, freq in final_counts.items()
            if freq == max_frequency
        ]

        print("    max_frequency = ", max_frequency)
        # Return the lexicographically smallest candidate as the tie-breaker.
        return min(candidates)


# --- Test Infrastructure ---

def run_tests():
    """
    Executes a suite of test cases against the SubsequenceFinder class.
    """
    finder = SubsequenceFinder()
    test_cases = [
        # --- Tests for k=3 (from previous problem) ---
        {"input": ("abacaba", 3), "expected": "aba", "name": "k=3 Symmetric"},
        {"input": ("topcoderopen", 3), "expected": "opn", "name": "k=3 Standard"},
        {"input": ("abccba", 3), "expected": "abc", "name": "k=3 Tie-breaking"},

        # --- Tests for general k ---
        {"input": ("banana", 2), "expected": "an", "name": "k=2, banana"},
        {"input": ("abracadabra", 4), "expected": "abra", "name": "k=4, abracadabra"},
        {"input": ("mississippi", 4), "expected": "issi", "name": "k=4, mississippi"},

        # --- Edge Cases ---
        {"input": ("abc", 4), "expected": "", "name": "k > length of text"},
        {"input": ("abcde", 5), "expected": "abcde", "name": "k = length of text"},
        {"input": ("zzzyyxxx", 1), "expected": "x", "name": "k=1, Tie-breaking"},
        {"input": ("abc", 0), "expected": "", "name": "k=0"},
        {"input": ("abc", -1), "expected": "", "name": "k=-1"},
        {"input": ("", 2), "expected": "", "name": "Empty String"},

        # --- Repetitive Character Cases ---
        {"input": ("aaaaa", 3), "expected": "aaa", "name": "k=3, All Same Characters"},
        {"input": ("babab", 3), "expected": "bab", "name": "k=3, Alternating"},
    ]

    print("Running test cases...\n")
    passed_all = True
    for i, test in enumerate(test_cases):
        text, k = test["input"]
        try:
            actual = finder.find_most_frequent(text, k)

            print(f"--- Test Case {i + 1}: {test['name']} ---")
            print(f"Input: ('{text}', {k})")
            print(f"Expected Output: '{test['expected']}'")
            print(f"Actual Output:   '{actual}'")

            if actual == test["expected"]:
                print("Result: PASSED ✅")
            else:
                print("Result: FAILED ❌")
                passed_all = False
        except Exception as e:
            print(f"--- Test Case {i + 1}: {test['name']} ---")
            print(f"Input: ('{text}', {k})")
            print(f"An exception occurred: {e}")
            print("Result: FAILED ❌")
            passed_all = False
        finally:
            print("-" * 30 + "\n")

    # Test for invalid input types
    try:
        finder.find_most_frequent(12345, 3)
    except TypeError:
        print("--- Test Case: Invalid Text Input --- PASSED ✅")
    else:
        print("--- Test Case: Invalid Text Input --- FAILED ❌")
        passed_all = False

    try:
        finder.find_most_frequent("abc", "3")
    except TypeError:
        print("--- Test Case: Invalid k Input --- PASSED ✅")
    else:
        print("--- Test Case: Invalid k Input --- FAILED ❌")
        passed_all = False

    print("-" * 30 + "\n")
    if passed_all:
        print("🎉 All test cases passed! 🎉")
    else:
        print("🔥 Some test cases failed. 🔥")


if __name__ == "__main__":
    run_tests()

    # --- Complexity Analysis ---
    #
    # N = Length of the input string
    # k = The desired length of the subsequence
    # A = Size of the character alphabet (e.g., 26 for lowercase English)
    #
    # Time Complexity: O(N * k * A^(k-1))
    # The main logic is a loop over the N characters of the string. Inside it,
    # another loop runs k times. The innermost loop iterates through the keys of
    # a DP counter. The number of unique subsequences of length `i` is at most A^i.
    # The most expensive inner step is for `i=k`, where we iterate over `dp[k-1]`,
    # which has at most A^(k-1) keys. This gives a complexity of O(N * k * A^(k-1)).
    # Since 'k' and 'A' are typically small constants, the complexity is
    # effectively linear with respect to N, i.e., O(N).
    #
    # Space Complexity: O(k * A^k)
    # The space is dominated by the `dp` table. We store `k` counters. The counter
    # for subsequences of length `i` can store up to A^i keys. The total space
    # is the sum of the sizes of these counters, which is bounded by O(k * A^k).
    # Like time complexity, this is constant if 'k' and 'A' are considered constants.
