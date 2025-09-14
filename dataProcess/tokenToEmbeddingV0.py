import torch
from collections import defaultdict

# Sample text
text = "PyTorch makes deep learning easy and fun"

# Create vocabulary
word_to_idx = defaultdict(lambda: len(word_to_idx))
word_to_idx['<unk>'] = 0  # Unknown token
word_to_idx['<pad>'] = 1  # Padding token

# Tokenize
tokens = text.lower().split()
indices = [word_to_idx[word] for word in tokens]

print("Vocabulary:", dict(word_to_idx))
print("Token indices:", indices)

# Convert to PyTorch tensor
token_tensor = torch.tensor(indices)
print("PyTorch tensor:", token_tensor)