# Question 1: Efficient Token Frequency Counting
def token_frequency_optimized(text_stream, top_k=10):
    """
    Process a stream of text and return top-k most frequent tokens
    Optimize for memory and speed for large datasets
    """
    # Expected: Use heap for O(n log k) instead of O(n log n)
    import heapq
    from collections import defaultdict

    freq = defaultdict(int)
    heap = []

    for token in text_stream:
        freq[token] += 1

        # Maintain min-heap of size k
        if len(heap) < top_k:
            heapq.heappush(heap, (freq[token], token))
        else:
            if freq[token] > heap[0][0]:
                heapq.heappushpop(heap, (freq[token], token))

    return [(token, count) for count, token in sorted(heap, reverse=True)]

# Follow-up: How would you make this memory-efficient for 1TB of data?

