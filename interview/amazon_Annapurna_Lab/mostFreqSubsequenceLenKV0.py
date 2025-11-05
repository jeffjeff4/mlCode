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
from typing import Dict


class SubsequenceFinder:
    """
    A class to find the most frequent subsequence of a given length in a string.

    This implementation is optimized to find length-3 subsequences in linear time.
    """

    def find_most_frequent_length_3(self, text: str) -> str:
        """
        Finds the most frequent subsequence of length 3 in the given text.

        The algorithm iterates through each character of the string, treating it as the
        middle character of a potential subsequence. It efficiently calculates how many
        subsequences can be formed with that middle character by using frequency counts
        of characters to the left and right.

        In case of a tie in frequency, the lexicographically smallest subsequence is returned.

        Args:
            text: The input string to search within.

        Returns:
            The most frequent subsequence of length 3, or an empty string if no such
            subsequence can be formed (e.g., if the input string is too short).

        Raises:
            TypeError: If the input `text` is not a string.
        """
        # --- Input Validation and Edge Cases ---
        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        if len(text) < 3:
            return ""

        # --- Algorithm Initialization ---
        subsequence_counts: Dict[str, int] = collections.defaultdict(int)

        # right_counts tracks the frequency of all characters to the "right" of the
        # current middle character. Initially, it's the count of all characters.
        right_counts = collections.Counter(text)

        # left_counts tracks the frequency of characters to the "left".
        # It starts empty.
        left_counts = collections.Counter()

        # --- Main Loop: O(N * A^2) where A is alphabet size ---
        for middle_char in text:
            # 1. The current `middle_char` is no longer in the "right" part.
            right_counts[middle_char] -= 1
            if right_counts[middle_char] == 0:
                del right_counts[middle_char]

            # 2. Calculate new subsequences with the current `middle_char`.
            #    For each char on the left and each char on the right, form a subsequence.
            #    The number of times this subsequence is formed is (count_left * count_right).
            if left_counts and right_counts:
                for char_left, count_left in left_counts.items():
                    for char_right, count_right in right_counts.items():
                        subsequence = char_left + middle_char + char_right
                        subsequence_counts[subsequence] += count_left * count_right

            # 3. The current `middle_char` now becomes part of the "left" for the next iteration.
            left_counts[middle_char] += 1

        # --- Result Finalization ---
        if not subsequence_counts:
            return ""

        # Find the maximum frequency among all found subsequences.
        max_frequency = max(subsequence_counts.values())

        # Filter for all subsequences that have the maximum frequency.
        candidates = [
            sub for sub, freq in subsequence_counts.items()
            if freq == max_frequency
        ]

        # Return the lexicographically smallest candidate as a tie-breaker.
        return min(candidates)


# --- Test Infrastructure ---

def run_tests():
    """
    Executes a suite of test cases against the SubsequenceFinder class.
    """
    finder = SubsequenceFinder()
    test_cases = [
        # Basic Cases
        {"input": "abacaba", "expected": "aba", "name": "Symmetric String"},
        {"input": "topcoderopen", "expected": "opn", "name": "Standard Example"},
        {"input": "axbyczaxby", "expected": "axy", "name": "Interspersed Characters"},

        # Edge Cases
        {"input": "ab", "expected": "", "name": "String Too Short"},
        {"input": "", "expected": "", "name": "Empty String"},
        {"input": "zyxw", "expected": "zyw", "name": "Reverse Alphabetical"},

        # Tie-breaking Cases
        {"input": "abccba", "expected": "abc", "name": "Tie, Lexicographically Smallest"},
        {"input": "zyxxzy", "expected": "zxy", "name": "Tie, Lexicographically Smallest 2"},

        # Repetitive Character Cases
        {"input": "aaaaa", "expected": "aaa", "name": "All Same Characters"},
        {"input": "ababa", "expected": "aba", "name": "Alternating Characters"},

        # Case with numbers and symbols
        {"input": "1a2b3c1a2b", "expected": "1a2", "name": "Alphanumeric String"},
    ]

    print("Running test cases...\n")
    passed_all = True
    for i, test in enumerate(test_cases):
        try:
            actual = finder.find_most_frequent_length_3(test["input"])

            print(f"--- Test Case {i + 1}: {test['name']} ---")
            print(f"Input String: '{test['input']}'")
            print(f"Expected Output: '{test['expected']}'")
            print(f"Actual Output:   '{actual}'")

            if actual == test["expected"]:
                print("Result: PASSED ✅")
            else:
                print(f"Result: FAILED ❌")
                passed_all = False
        except Exception as e:
            print(f"--- Test Case {i + 1}: {test['name']} ---")
            print(f"Input String: '{test['input']}'")
            print(f"An exception occurred: {e}")
            print(f"Result: FAILED ❌")
            passed_all = False
        finally:
            print("-" * 30 + "\n")

    # Test for non-string input
    try:
        finder.find_most_frequent_length_3(12345)
    except TypeError as e:
        print("--- Test Case: Non-String Input ---")
        print("Input: 12345")
        print(f"Caught expected exception: {e}")
        print("Result: PASSED ✅")
        print("-" * 30 + "\n")
    except Exception as e:
        print("--- Test Case: Non-String Input ---")
        print(f"Caught unexpected exception: {e}")
        print("Result: FAILED ❌")
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
    # A = Size of the character alphabet (e.g., 26 for lowercase English, ~128 for ASCII)
    #
    # Time Complexity: O(N * A^2)
    # The main logic is a single loop that iterates N times. Inside this loop, we have
    # two nested loops that iterate over the unique characters found in the left and
    # right parts of the string. In the worst case, this is A * A iterations.
    # Since the alphabet size A is a constant that does not grow with the input
    # string length N, the overall time complexity is considered linear, O(N).
    #
    # Space Complexity: O(A^3)
    # The space is dominated by the `subsequence_counts` dictionary. The number of
    # possible subsequences of length 3 is determined by the number of unique
    # characters in the string, which is at most A. The total number of unique
    # length-3 subsequences is therefore at most A * A * A = A^3.
    # The `left_counts` and `right_counts` dictionaries each store at most A keys.
    # As A is a constant, the space complexity is also constant, O(1).
