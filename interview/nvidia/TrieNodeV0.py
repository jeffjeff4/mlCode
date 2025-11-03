class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search_prefix(self, prefix):
        # Traverse to the prefix node
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]

        # DFS to collect all words starting from this prefix node
        result = []

        def dfs(curr, path):
            if curr.is_end:
                result.append(prefix + path)
            for ch, nxt in curr.children.items():
                dfs(nxt, path + ch)

        dfs(node, "")
        return sorted(result)  # sorted for deterministic order


# ✅ Test
t = Trie()
t.insert("apple")
t.insert("app")
assert t.search_prefix("ap") == ["app", "apple"]
print("All tests passed ✅")
