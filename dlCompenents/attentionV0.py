import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        # 1. Calculate the dot product between Q and K
        # Q: (batch_size, n_heads, query_len, d_k)
        # K: (batch_size, n_heads, key_len, d_k)
        # scores: (batch_size, n_heads, query_len, key_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # 2. Scale the scores
        scores = scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # 3. Apply optional mask to the scores
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Apply softmax to get the attention weights
        # attention_weights: (batch_size, n_heads, query_len, key_len)
        attention_weights = F.softmax(scores, dim=-1)

        # 5. Multiply weights by V to get the output
        # output: (batch_size, n_heads, query_len, d_v)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


# Example Tensors
batch_size, query_len, key_len, d_k = 2, 3, 3, 4

# Q, K, V tensors with shape (batch_size, seq_len, embed_dim)
Q = torch.randn(batch_size, query_len, d_k)  # (2, 3, 4)
K = torch.randn(batch_size, key_len, d_k)    # (2, 3, 4)
V = torch.randn(batch_size, key_len, d_k)    # (2, 3, 4)

# Create an instance of the attention module
attn = ScaledDotProductAttention(d_k=d_k)

# Forward pass
output, attention_weights = attn(Q, K, V)

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)
print("-" * 20)
print("Output shape:", output.shape)
print("Attention weights shape:", attention_weights.shape)
