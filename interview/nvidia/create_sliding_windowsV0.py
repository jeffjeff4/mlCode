####1. What This Solves (The LLM Concept)
####
####This technique is the standard solution for processing documents that are longer than a model's context window (e.g., 4096 tokens for BERT or 32k for GPT-4).
####
####You can't just feed the first 4096 tokens (truncation) because you lose all information from the rest of the document. Instead, you "slide" a window across the document, creating multiple, overlapping chunks.
####
####The overlap is the most important part. By setting step_size < window_size, you ensure that each new chunk contains some of the end of the previous chunk.
####
####Why overlap? This gives the model "local context." When the model processes Chunk 2, it has already "seen" the start of its content (which was the end of Chunk 1). This helps the model maintain a coherent understanding and link ideas across the chunk boundaries.
####
####2. The Code Explained (Python Proficiency)
####
####The implementation is surprisingly simple and "Pythonic" by leveraging range() and list slicing.
####
####for start_index in range(0, total_tokens, step_size):
####    end_index = start_index + window_size
####    chunk = tokens[start_index:end_index]
####    chunks.append(chunk)
####
####
####range(0, total_tokens, step_size): This is the core of the logic. It generates the start_index for each chunk: 0, 2048, 4096, 6144, 8192. It correctly stops before it hits total_tokens (10000).
####
####chunk = tokens[start_index:end_index]: This is the magic of Python list slicing.
####
####For the last chunk, start_index is 8192.
####
####end_index becomes 8192 + 4096 = 12288.
####
####Even though 12288 is way past the end of the list (9999), Python's slicer doesn't crash. It simply says, "give me the tokens from index 8192 to the end."
####
####This gracefully handles the final, shorter chunk (len=1808) without any special if statements.
####
####3. Handling the Final Chunk and Padding
####
####You're right to note the ambiguity in the prompt about padding.
####
####This Function's Job: This function create_sliding_windows should only be responsible for chunking. It correctly returns [5, 6], the short, final chunk.
####
####The collate_fn's Job: These chunks ([[1, 2, 3], [3, 4, 5], [5, 6]]) would then be passed to the collate_fn from the previous problem.
####
####The collate_fn would see this list of lists, find the max length (which is 3), and it would be responsible for adding the padding:
####
####[[1, 2, 3],
#### [3, 4, 5],
#### [5, 6, 0]]  <-- Padding added by collate_fn
####
####
####This creates a clean "separation of concerns" where each function does one job well.



from typing import List, Any


def create_sliding_windows(
        tokens: List[Any],
        window_size: int,
        step_size: int
) -> List[List[Any]]:
    """
    Creates overlapping sliding windows (chunks) from a long list of tokens.

    This function is used to break a document that is longer than the
    model's context window into smaller, overlapping pieces.

    Example:
        tokens = [1, 2, 3, 4, 5, 6]
        window_size = 3
        step_size = 2
        Returns: [[1, 2, 3], [3, 4, 5], [5, 6]]

    Args:
        tokens (List[Any]): The full list of tokens (e.g., `input_ids`).
        window_size (int): The maximum size of each chunk (e.g., 4096).
        step_size (int): The "stride" or number of tokens to "slide"
                         the window forward. `step_size < window_size`
                         creates overlap.

    Returns:
        List[List[Any]]: A list of token chunks.
    """
    chunks = []
    total_tokens = len(tokens)

    # We iterate using a for loop with a step size.
    # range(start, stop, step)
    for start_index in range(0, total_tokens, step_size):
        # The end_index is simply the start_index plus the window size.
        # Python's list slicing handles the edge case where
        # `end_index` goes past the end of the list.
        end_index = start_index + window_size

        # Slice the tokens
        chunk = tokens[start_index:end_index]

        # We add the chunk, even if it's empty (though range()
        # should prevent this if total_tokens > 0)
        # If start_index is already past the end, slicing returns []
        # but the range() loop condition `start_index < total_tokens`
        # naturally handles this.
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

    return chunks


# --- Example Usage ---
if __name__ == "__main__":
    # 1. The simple example from the prompt
    simple_tokens = [1, 2, 3, 4, 5, 6]
    window = 3
    step = 2

    print(f"--- Simple Example ---")
    print(f"Tokens: {simple_tokens}")
    print(f"Window: {window}, Step: {step}")
    chunks = create_sliding_windows(simple_tokens, window, step)
    print(f"Result: {chunks}")
    # Expected: [[1, 2, 3], [3, 4, 5], [5, 6]]
    assert chunks == [[1, 2, 3], [3, 4, 5], [5, 6]]

    # 2. The large-scale example
    long_doc_tokens = list(range(10000))  # A document with 10,000 tokens
    context_window = 4096
    overlap = 2048  # This is a common overlap strategy

    print(f"\n--- Large Document Example ---")
    print(f"Total Tokens: {len(long_doc_tokens)}")
    print(f"Window Size: {context_window}")
    print(f"Step Size (Overlap): {overlap}")

    long_chunks = create_sliding_windows(long_doc_tokens, context_window, overlap)

    print(f"\nTotal chunks created: {len(long_chunks)}")

    # Let's inspect the chunks to verify
    # Total chunks:
    # Chunk 1: start 0
    # Chunk 2: start 2048
    # Chunk 3: start 4096
    # Chunk 4: start 6144
    # Chunk 5: start 8192
    # Next start would be 10240, which is > 10000, so loop stops.
    # Total should be 5 chunks.
    assert len(long_chunks) == 5

    # Check first chunk
    print(f"Chunk 1 length: {len(long_chunks[0])}")
    print(f"Chunk 1 starts with: {long_chunks[0][:5]}...")
    print(f"Chunk 1 ends with: ...{long_chunks[0][-5:]}")
    assert len(long_chunks[0]) == 4096
    assert long_chunks[0][0] == 0
    assert long_chunks[0][-1] == 4095

    # Check last chunk
    # It should start at token 8192 and go to the end (9999)
    # Length should be 10000 - 8192 = 1808
    print(f"\nLast chunk length: {len(long_chunks[-1])}")
    print(f"Last chunk starts with: {long_chunks[-1][:5]}...")
    print(f"Last chunk ends with: ...{long_chunks[-1][-5:]}")
    assert len(long_chunks[-1]) == 1808
    assert long_chunks[-1][0] == 8192
    assert long_chunks[-1][-1] == 9999

    print("\nAll assertions passed. Logic is correct.")
