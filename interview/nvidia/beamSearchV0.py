import heapq
import numpy as np


def beam_search_step(logits: np.ndarray, beam_size: int):
    """
    Return top `beam_size` indices and their scores from logits (per batch).

    Args:
        logits: np.ndarray of shape (batch_size, vocab_size)
        beam_size: int, number of top candidates to keep
    Returns:
        List of tuples [(score, index), ...] per batch entry.
        Shape: List[List[(float, int)]]
    """
    batch_size, vocab_size = logits.shape
    top_results = []

    for i in range(batch_size):
        # Get (score, index) pairs for this sample
        heap = []
        for j, score in enumerate(logits[i]):
            if len(heap) < beam_size:
                heapq.heappush(heap, (score, j))
            else:
                # maintain top-k highest scores
                heapq.heappushpop(heap, (score, j))
        # Sort descending by score
        top_results.append(sorted(heap, key=lambda x: -x[0]))

    return top_results


# ✅ Test
scores = np.array([[0.1, 0.4, 0.5]])
top = beam_search_step(scores, 2)
print(top)  # Expected: [[(0.5, 2), (0.4, 1)]]
