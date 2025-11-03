def tokenize_bpe(text, merges):
    """
    Perform simple BPE tokenization based on given merges.

    Args:
        text (str): input string, tokens separated by spaces
        merges (dict): dictionary of merge rules, e.g., {('a','b'): 'ab'}

    Returns:
        List[str]: tokenized output
    """
    tokens = text.split()

    # Keep applying merges until no more merge possible
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(tokens) - 1:
            pair = (tokens[i], tokens[i + 1])
            if pair in merges:
                tokens[i] = merges[pair]  # merge the pair
                del tokens[i + 1]  # remove second token
                changed = True
            else:
                i += 1
    return tokens


# ✅ Test
text = "a b c a b"
merges = {("a", "b"): "ab"}
print(tokenize_bpe(text, merges))  # Output: ['ab', 'c', 'ab']
