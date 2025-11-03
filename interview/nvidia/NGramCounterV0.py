from collections import Counter, deque

class NGramCounter:
    def __init__(self, n: int, window: int):
        """
        n: length of n-gram (e.g., 2 = bigram)
        window: max number of recent tokens to consider
        """
        self.n = n
        self.window = window
        self.tokens = deque()
        self.ngrams = deque()
        self.counter = Counter()

    def insert(self, token: str):
        # Add token to window
        self.tokens.append(token)
        if len(self.tokens) > self.window:
            # Remove oldest token
            old_token = self.tokens.popleft()
            # If enough tokens before removal, remove its contribution
            if len(self.tokens) >= self.n - 1:
                # Remove oldest n-gram
                old_ngram = tuple(self.tokens[i] for i in range(self.n - 1))
                if self.ngrams:
                    old_ngram = self.ngrams.popleft()
                    self.counter[old_ngram] -= 1
                    if self.counter[old_ngram] == 0:
                        del self.counter[old_ngram]

        # Add new n-gram if possible
        if len(self.tokens) >= self.n:
            new_ngram = tuple(self.tokens)[-self.n:]
            self.ngrams.append(new_ngram)
            self.counter[new_ngram] += 1

    def top_k(self, k: int):
        # Return top-k ngrams sorted by frequency
        return self.counter.most_common(k)


# ✅ Test
ng = NGramCounter(2, 5)
for t in ["a", "b", "a", "c", "a", "b"]:
    ng.insert(t)
print(ng.top_k(2))
