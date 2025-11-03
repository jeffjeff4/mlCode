# Question 2: Implement Byte-Pair Encoding (BPE) trie
class BPETrie:
    def __init__(self):
        self.children = {}
        self.frequency = 0
        self.is_merge = False

    def add_token(self, token, freq):
        node = self
        for char in token:
            if char not in node.children:
                node.children[char] = BPETrie()
            node = node.children[char]
        node.frequency += freq

    def find_most_frequent_pair(self):
        """
        Find the most frequent character pair in the vocabulary
        """
        max_freq = 0
        best_pair = None

        def dfs(node, path):
            nonlocal max_freq, best_pair
            if len(path) >= 2 and node.frequency > max_freq:
                max_freq = node.frequency
                best_pair = path[-2:]  # Last two characters

            for char, child in node.children.items():
                dfs(child, path + [char])

        dfs(self, [])
        return ''.join(best_pair), max_freq


# Test case
trie = BPETrie()
tokens = {"ab": 10, "bc": 15, "abc": 5, "a": 20}
for token, freq in tokens.items():
    trie.add_token(token, freq)
print(trie.find_most_frequent_pair())  # Should find most frequent pair

