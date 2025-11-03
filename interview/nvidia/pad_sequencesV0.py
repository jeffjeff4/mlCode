import numpy as np


def pad_sequences(sequences, pad_value=0):
    # Find the maximum sequence length
    max_len = max(len(seq) for seq in sequences)

    # Create padded array
    padded = np.full((len(sequences), max_len), pad_value, dtype=int)

    # Copy each sequence into the padded array
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq

    return padded


# ✅ Test
seqs = [[1, 2, 3], [4, 5]]
print(pad_sequences(seqs))
