import collections


class SynonymFinder:
    """
    Manages synonym groups using a Union-Find (Disjoint Set Union) data structure.

    This class efficiently processes synonym pairs and can quickly determine if
    two words belong to the same synonym group, correctly handling transitive
    relationships (e.g., if A=B and B=C, then A=C).

    The implementation uses path compression and union-by-rank for
    near-constant time (amortized) complexity for `find` and `union` operations.
    """

    def __init__(self, synonym_pairs: list[list[str]]):
        """
        Initializes the Union-Find structure by processing all synonym pairs.

        Args:
            synonym_pairs: A list of [word1, word2] pairs.
        """
        self.parent = {}
        self.rank = {}

        for w1, w2 in synonym_pairs:
            self.union(w1, w2)

    def find(self, word: str) -> str:
        """
        Finds the representative (root) of the set containing 'word'.

        This method uses path compression for optimization. If the word has not
        been seen, it is added to the structure as its own set.

        Args:
            word: The word to find the set representative for.

        Returns:
            The representative (root) word of the synonym group.
        """
        # Lazily add the word if it's not in our map (make_set)
        if word not in self.parent:
            self.parent[word] = word
            self.rank[word] = 0
            return word

        # Path compression:
        # If this word is not its own parent, recursively find the root
        # and set this word's parent directly to the root.
        if self.parent[word] != word:
            self.parent[word] = self.find(self.parent[word])

        return self.parent[word]

    def union(self, w1: str, w2: str) -> bool:
        """
        Merges the synonym groups containing w1 and w2.

        This method uses union-by-rank to keep the "trees" flat,
        ensuring high efficiency.

        Args:
            w1: The first word.
            w2: The second word.

        Returns:
            True if a merge occurred, False if w1 and w2 were already
            in the same group.
        """
        root1 = self.find(w1)
        root2 = self.find(w2)

        if root1 != root2:
            # Union-by-rank:
            # Attach the smaller-rank tree to the root of the higher-rank tree
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
                self.rank[root1] += 1
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
                self.rank[root2] += 1
            else:
                # Ranks are equal, pick one (e.g., root1) as the new parent
                # and increment its rank.
                self.parent[root2] = root1
                self.rank[root1] += 1
            return True

        return False  # They were already in the same set

    def are_synonymous(self, w1: str, w2: str) -> bool:
        """
        Checks if two words are in the same synonym group (or are the same word).

        Args:
            w1: The first word.
            w2: The second word.

        Returns:
            True if w1 and w2 are synonymous, False otherwise.
        """
        # Two words are synonymous if they share the same root
        # in the Union-Find structure. This check correctly handles
        # reflexivity (e.g., "word" == "word"), symmetry, and transitivity.
        return self.find(w1) == self.find(w2)


class ReviewComparator:
    """
    Compares product reviews for similarity based on word-by-word synonymy.

    This class is designed to be expandable. Future methods could be added
    to compare reviews using other metrics (e.g., semantic similarity).
    """

    def __init__(self, synonyms: list[list[str]]):
        """
        Initializes the comparator by preprocessing the synonym pairs.

        Args:
            synonyms: A list of pairs [w1, w2] indicating that w1 and w2
                      are synonyms.
        """
        if not isinstance(synonyms, list):
            raise TypeError("Synonyms must be a list of lists.")

        # The comparator holds an instance of the SynonymFinder.
        # This encapsulates the complex logic of synonym-grouping.
        self.synonym_finder = SynonymFinder(synonyms)

    def are_reviews_similar(self, review1: list[str], review2: list[str]) -> bool:
        """
        Checks if two reviews are similar.

        Similarity is defined as:
        1. Both reviews must have the same number of words.
        2. For each position 'i', review1[i] and review2[i] must be synonymous
           (or the exact same word).

        Args:
            review1: The first review, as a list of words.
            review2: The second review, as a list of words.

        Returns:
            True if the reviews are similar, False otherwise.
        """

        # Edge Case: Handle non-list inputs gracefully
        if not isinstance(review1, list) or not isinstance(review2, list):
            # Depending on requirements, could raise TypeError or log warning
            return False

            # Edge Case: Different lengths
        # By definition, reviews of different lengths cannot be similar.
        if len(review1) != len(review2):
            return False

        # Edge Case: Empty reviews
        # If both are empty, len(review1) == len(review2) == 0.
        # The loop will not run, and it will correctly return True.

        # Compare each word pair
        # zip() efficiently pairs corresponding words and stops at the
        # end of the shorter list (but we've already ensures they are
        # the same length).
        for w1, w2 in zip(review1, review2):
            # We rely on our robust SynonymFinder to check for synonymy.
            if not self.synonym_finder.are_synonymous(w1, w2):
                # If any pair is not synonymous, the reviews are not similar.
                return False

        # If the loop completes, all word pairs were synonymous.
        return True


# --- Analysis ---

### Time Complexity

####Let:
####* $N$ = the
####number
####of
####synonym
####pairs in the
####`synonyms`
####list.
####* $W$ = the
####number
####of
####unique
####words * in the
####`synonyms`
####list *.
####* $L$ = the
####length
####of
####the
####reviews(assuming
####`len(review1) == len(review2)`).
####* $\alpha(n)$ = the ** Inverse
####Ackermann
####function **.This
####function
####grows * extremely * slowly(e.g., $\alpha(
####    n) < 5$ for any practical value of $n$).It is considered * near-constant * or $O(1)$ for all practical purposes.
####
####1. ** `ReviewComparator`
####Initialization(`__init__`) **:
####*This
####involves
####creating
####the
####`SynonymFinder`.
####*The
####`SynonymFinder`
####'s `__init__` iterates $N$ times.
####*Each
####iteration
####calls
####`union`, which in turn
####calls
####`find`
####twice.
####*With
####path
####compression and union - by - rank, both
####`find` and `union`
####operations
####have
####an ** amortized
####time
####complexity
####of $O(\alpha(W))$ **.
####* ** Total
####Initialization
####Time: ** $O(N \cdot \alpha(W))$, which is practically $O(N)$.
####
####2. ** `are_reviews_similar`(Query
####Time) **:
####*Checking
####lengths: $O(1)$.
####*The
####`zip`
####loop
####runs $L$ times.
####*Inside
####the
####loop, `are_synonymous` is called, which
####calls
####`find`
####twice.
####*The
####`find`
####operation(on
####words
####from the reviews, which
####
####may
####be
####new
####to
####the
####`SynonymFinder`) also
####has
####an
####amortized
####complexity
####of $O(\alpha(U))$, where $U$ is the
####total
####number
####of
####unique
####words
####seen
####so
####far(
####from synonyms
####
####+ reviews).
####* ** Total
####Query
####Time: ** $O(L \cdot \alpha(U))$, which is practically $O(L)$.
####
####** Overall
####Time: ** $O(N)$ preprocessing and $O(L)$ query
####time.This is highly
####efficient.
####
####---
####
####### Space Complexity
####
####1. ** `ReviewComparator`(`__init__`) **:
####*The
####`SynonymFinder`
####stores
####two
####dictionaries: `parent` and `rank`.
####*In
####the
####worst
####case, all
####words in the $N$ pairs
####are
####unique($W = 2
####N$).
####*The
####space
####required is proportional
####to
####the
####number
####of
####unique
####words
####seen
####during
####initialization.
####* ** Total
####Initialization
####Space: ** $O(W)$ (or $O(N)$).
####
####2. ** `are_reviews_similar`(Query
####Space) **:
####*The
####comparison
####loop
####uses $O(1)$ auxiliary
####space.
####*The
####`find`
####operation
####'s recursion stack is bounded by $O(\log U)$ in the worst case (or $O(\alpha(U))$ amortized), but path compression keeps it very shallow.
####*The
####`parent` and `rank`
####maps
####might
####grow if the
####reviews
####contain
####words not in the
####original
####`synonyms`
####list.The
####total
####space
####complexity
####of
####the * object * can
####grow
####to $O(U)$ over
####time, where $U$ is the
####total
####number
####of
####unique
####words
####ever
####processed.
####* ** Auxiliary
####Query
####Space: ** $O(1)$ (not counting the growth of the object's state).
####
####** Overall Space: ** $O(U)$, where $U$ is the
####total
####number
####of
####unique
####words
####across
####all
####`synonyms` and all
####`reviews`
####processed.
####
####- --

### Runnable Test Cases

####```python
if __name__ == "__main__":
# --- Test Case 1: Provided Example ---
    synonyms1 = [["great", "amazing"], ["product", "device"]]
r1_1 = ["great", "product"]
r1_2 = ["amazing", "device"]

comparator1 = ReviewComparator(synonyms1)
result1 = comparator1.are_reviews_similar(r1_1, r1_2)
print(f"Test 1 (Example): {result1}")  # Expected: True

# --- Test Case 2: Transitive Property ---
# a=b, b=c  =>  a=c
synonyms2 = [["a", "b"], ["b", "c"], ["x", "y"]]
r2_1 = ["a", "x"]
r2_2 = ["c", "y"]

comparator2 = ReviewComparator(synonyms2)
result2 = comparator2.are_reviews_similar(r2_1, r2_2)
print(f"Test 2 (Transitive): {result2}")  # Expected: True

# --- Test Case 3: Transitive (False) ---
# a=b, c=d  =>  a != d
synonyms3 = [["a", "b"], ["c", "d"]]
r3_1 = ["a"]
r3_2 = ["d"]

comparator3 = ReviewComparator(synonyms3)
result3 = comparator3.are_reviews_similar(r3_1, r3_2)
print(f"Test 3 (Transitive False): {result3}")  # Expected: False

# --- Test Case 4: Reflexive Property (Words are identical) ---
synonyms4 = []  # No synonyms given
r4_1 = ["fast", "shipping"]
r4_2 = ["fast", "shipping"]

comparator4 = ReviewComparator(synonyms4)
result4 = comparator4.are_reviews_similar(r4_1, r4_2)
print(f"Test 4 (Reflexive/Identical): {result4}")  # Expected: True

# --- Test Case 5: Partial Match (False) ---
synonyms5 = [["great", "amazing"], ["product", "device"]]
r5_1 = ["great", "product"]
r5_2 = ["amazing", "item"]  # "item" is not synonymous with "product"

comparator5 = ReviewComparator(synonyms5)
result5 = comparator5.are_reviews_similar(r5_1, r5_2)
print(f"Test 5 (Partial Match): {result5}")  # Expected: False

# --- Test Case 6: Words not in Synonym List (False) ---
synonyms6 = [["fast", "quick"]]
r6_1 = ["fast", "shipping"]
r6_2 = ["quick", "delivery"]  # "shipping" and "delivery" are not synonymous

comparator6 = ReviewComparator(synonyms6)
result6 = comparator6.are_reviews_similar(r6_1, r6_2)
print(f"Test 6 (Words not in list, False): {result6}")  # Expected: False

# --- Test Case 7: Words not in Synonym List (True) ---
synonyms7 = [["fast", "quick"]]
r7_1 = ["fast", "shipping"]
r7_2 = ["quick", "shipping"]  # "shipping" is synonymous with itself

comparator7 = ReviewComparator(synonyms7)
result7 = comparator7.are_reviews_similar(r7_1, r7_2)
print(f"Test 7 (Words not in list, True): {result7}")  # Expected: True

# --- Test Case 8: Edge Case (Different Lengths) ---
synonyms8 = []
r8_1 = ["good"]
r8_2 = ["good", "product"]

comparator8 = ReviewComparator(synonyms8)
result8 = comparator8.are_reviews_similar(r8_1, r8_2)
print(f"Test 8 (Different Lengths): {result8}")  # Expected: False

# --- Test Case 9: Edge Case (Both Empty) ---
synonyms9 = []
r9_1 = []
r9_2 = []

comparator9 = ReviewComparator(synonyms9)
result9 = comparator9.are_reviews_similar(r9_1, r9_2)
print(f"Test 9 (Both Empty): {result9}")  # Expected: True

# --- Test Case 10: Edge Case (One Empty) ---
synonyms10 = []
r10_1 = []
r10_2 = ["great"]

comparator10 = ReviewComparator(synonyms10)
result10 = comparator10.are_reviews_similar(r10_1, r10_2)
print(f"Test 10 (One Empty): {result10}")  # Expected: False

# --- Test Case 11: Complex Synonym Group ---
synonyms11 = [
    ["terrible", "bad"],
    ["awful", "horrible"],
    ["bad", "awful"]  # Connects the two pairs
]
r11_1 = ["terrible", "product"]
r11_2 = ["horrible", "product"]

comparator11 = ReviewComparator(synonyms11)
result11 = comparator11.are_reviews_similar(r11_1, r11_2)
print(f"Test 11 (Complex Group): {result11}")  # Expected: True