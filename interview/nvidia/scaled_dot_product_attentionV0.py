####Here are the answers to the "why" and "how" of your follow-up questions.
####
####1. Why the scale factor $\sqrt{d_k}$?
####
####Short Answer: To prevent the softmax function from saturating, which causes vanishing gradients and stops the model from learning.
####
####Detailed Explanation:
####
####The Problem: The attention scores are calculated by torch.matmul(q, k.transpose(-2, -1)). Let's assume the components of q and k are independent random variables with a mean of 0 and a variance of 1.
####
####The Math: A dot product of two such vectors of length $d_k$ will result in a new random variable that has a mean of 0 but a variance of $d_k$.
####
####The Saturation: This means that as the key/query dimension $d_k$ gets larger (e.g., 64, 128, 512), the variance of the scores also gets larger. This pushes many of the scores to extreme values (e.g., very large positive or very large negative numbers).
####
####The Softmax Gradient: The softmax function, $e^x / \sum(e^x)$, is "saturating." If it receives inputs like [1, 2, 80], it will output [0.0, 0.0, 1.0]. This is a "hard" max, not a "soft" one. The gradient for the 80 is almost zero, because changing it slightly (e.g., to 80.1) doesn't change the output.
####
####Vanishing Gradients: When gradients are zero, no information flows backward during backpropagation. The model stops learning.
####
####The Solution: By dividing the scores by $\sqrt{d_k}$ (which is the standard deviation), we normalize the variance back to 1. This keeps the inputs to the softmax function in a "healthy" range, where it can produce meaningful distributions (e.g., [0.1, 0.3, 0.6]) and, most importantly, has non-zero gradients that allow the model to learn.
####
####2. How would you modify this for a decoder (Causal Mask)?
####
####Short Answer: You apply a "look-ahead" or "causal" mask to the scores before the softmax step.
####
####Detailed Explanation:
####
####The Purpose: In a decoder (like in GPT), the model is generating a sequence token by token. When generating the 5th token, it should only be allowed to see (attend to) tokens 1, 2, 3, 4, and itself (token 5). It must not "look ahead" at tokens 6, 7, 8, etc., as this would be "cheating" and it wouldn't learn to predict the next word.
####
####The Mechanism: We create a mask that prevents attention to all "future" positions. For a sequence of length $S$, this mask is an $S \times S$ matrix where:
####
####mask[i, j] = 1 (allow) if j <= i
####
####mask[i, j] = 0 (block) if j > i
####This creates a lower-triangular matrix of 1s.
####
####The Implementation:
####
####You create this mask. In PyTorch, torch.tril(torch.ones(S_q, S_k)) is a common way.
####
####You pass this mask to the mask argument of the scaled_dot_product_attention function I wrote.
####
####The line scaled_scores = scaled_scores.masked_fill(mask == 0, -1e9) sets all "future" positions (where the mask is 0) to negative infinity.
####
####When you apply softmax, $e^{-\infty}$ becomes 0. This ensures that no attention "probability" is assigned to any future token.
####
####3. How would you implement Multi-Head Attention (MHA)?
####
####Short Answer: You create nn.Linear layers to project Q, K, and V for each head, reshape/transpose the tensors to separate the heads, call your scaled_dot_product_attention function, and then reshape/transpose back and pass through a final linear layer.
####
####Detailed Explanation:
####
####The "multi-head" part is about running the same attention mechanism $h$ times in parallel, each on a different "projection" (or subspace) of the Q, K, and V matrices.
####
####Here is the
####process as a
####PyTorch
####nn.Module:
####
####import torch
####import torch.nn as nn
####
####
##### Assume scaled_dot_product_attention is imported from the other file
####
####class MultiHeadAttention(nn.Module):
####    def __init__(self, d_model: int, num_heads: int):
####        super().__init__()
####        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
####
####        self.d_model = d_model
####        self.num_heads = num_heads
####        self.d_head = d_model // num_heads
####
####        # 1. Create linear layers for Q, K, V projections (all in one go)
####        # We create one big layer for each, which is more efficient
####        self.wq = nn.Linear(d_model, d_model)
####        self.wk = nn.Linear(d_model, d_model)
####        self.wv = nn.Linear(d_model, d_model)
####
####        # 8. Final output linear layer
####        self.fc_out = nn.Linear(d_model, d_model)
####
####    def forward(self, q, k, v, mask=None):
####        # Input shapes: (batch_size, seq_len, d_model)
####        batch_size = q.shape[0]
####
####        # 2. Project Q, K, V
####        # (B, S, D_model) -> (B, S, D_model)
####        Q_proj = self.wq(q)
####        K_proj = self.wk(k)
####        V_proj = self.wv(v)
####
####        # 3. Split into heads
####        # (B, S, D_model) -> (B, S, h, D_head) -> (B, h, S, D_head)
####        # We transpose to (B, h, S, D_head) because our attention
####        # function is batch-aware and will treat 'h' as just
####        # another batch dimension.
####        Q = Q_proj.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
####        K = K_proj.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
####        V = V_proj.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
####
####        # 4. Apply attention
####        # The 'mask' will be broadcasted across the 'h' dimension.
####        # output shape: (B, h, S_q, D_head)
####        # weights shape: (B, h, S_q, S_k)
####        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
####
####        # 5. Concatenate heads
####        # We need to put the 'h' dimension back next to 'D_head'
####        # (B, h, S_q, D_head) -> (B, S_q, h, D_head)
####        # Then .contiguous() to ensure memory layout is correct
####        # before .view()
####        output = output.transpose(1, 2).contiguous()
####
####        # 6. Reshape to (B, S_q, D_model)
####        output = output.view(batch_size, -1, self.d_model)
####
####        # 7. Pass through final linear layer
####        output = self.fc_out(output)
####
####        return output, attention_weights


import torch
import torch.nn.functional as F
import math


def scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
) -> (torch.Tensor, torch.Tensor):
    """
    Implements the Scaled Dot-Product Attention mechanism.

    Args:
        q (torch.Tensor): Query tensor.
                          Shape: (batch_size, ..., seq_len_q, d_k)
        k (torch.Tensor): Key tensor.
                          Shape: (batch_size, ..., seq_len_k, d_k)
        v (torch.Tensor): Value tensor.
                          Shape: (batch_size, ..., seq_len_v, d_v)
                          Note: seq_len_k and seq_len_v must be the same.
        mask (torch.Tensor, optional): Mask to apply to the scores.
                                       Shape must be broadcastable to
                                       (batch_size, ..., seq_len_q, seq_len_k).
                                       A `0` in the mask means the
                                       position will be masked (set to -inf).
                                       Defaults to None.

    Returns:
        (torch.Tensor, torch.Tensor): A tuple containing:
            - output: The context tensor after attention.
                      Shape: (batch_size, ..., seq_len_q, d_v)
            - attention_weights: The attention weights.
                                 Shape: (batch_size, ..., seq_len_q, seq_len_k)
    """

    # 1. & 2. Calculate scores and scale
    # d_k is the dimension of the keys (and queries)
    d_k = k.size(-1)

    # Q @ K^T
    # (..., seq_len_q, d_k) @ (..., d_k, seq_len_k) -> (..., seq_len_q, seq_len_k)
    scores = torch.matmul(q, k.transpose(-2, -1))

    # Scale by sqrt(d_k)
    scaled_scores = scores / math.sqrt(d_k)

    # 3. Apply mask (if provided)
    if mask is not None:
        # Fills elements with -1e9 (a very small number) where mask is 0
        scaled_scores = scaled_scores.masked_fill(mask == 0, -1e9)

    # 4. Apply softmax
    # Softmax is applied on the last dimension (seq_len_k) so that
    # all weights for a single query sum to 1.
    attention_weights = F.softmax(scaled_scores, dim=-1)

    # 5. Multiply by Value
    # (..., seq_len_q, seq_len_k) @ (..., seq_len_v, d_v) -> (..., seq_len_q, d_v)
    # Note: seq_len_k == seq_len_v
    output = torch.matmul(attention_weights, v)

    return output, attention_weights


# --- Example Usage ---
if __name__ == "__main__":
    # Setup: B=batch, S_q=query_len, S_k=key_len, D_k=key_dim, D_v=value_dim
    B, S_q, S_k, D_k, D_v = 4, 10, 12, 64, 128

    # Create random tensors
    q = torch.randn(B, S_q, D_k)
    k = torch.randn(B, S_k, D_k)
    v = torch.randn(B, S_k, D_v)  # S_k must match S_k of k

    print(f"--- 1. Standard Attention ---")
    print(f"Q shape: {q.shape}")
    print(f"K shape: {k.shape}")
    print(f"V shape: {v.shape}")

    output, weights = scaled_dot_product_attention(q, k, v)

    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

    # Verify shapes
    assert output.shape == (B, S_q, D_v)
    assert weights.shape == (B, S_q, S_k)

    # --- 2. Masked Attention (e.g., for padding) ---
    print(f"\n--- 2. Masked Attention (Padding) ---")

    # Create a simple padding mask.
    # Let's say in our 4 batches, the key/value sequences have
    # "real" lengths of [12, 10, 8, 12]
    # We want to mask out the padding.
    # Mask shape: (B, 1, S_k) -> broadcoasts to (B, S_q, S_k)
    key_lengths = torch.tensor([12, 10, 8, 12])
    # Create a (B, S_k) mask
    padding_mask = torch.arange(S_k)[None, :] < key_lengths[:, None]
    # Add a dim for the query sequence to broadcast
    padding_mask = padding_mask.unsqueeze(1).to(q.dtype)  # (B, 1, S_k)

    print(f"Padding mask shape: {padding_mask.shape}")
    print("Example mask (first batch, no padding):")
    print(padding_mask[0])
    print("Example mask (third batch, 4 padding tokens):")
    print(padding_mask[2])

    output_masked, weights_masked = scaled_dot_product_attention(q, k, v, mask=padding_mask)

    print(f"\nMasked output shape: {output_masked.shape}")
    print(f"Masked weights shape: {weights_masked.shape}")

    # Check if the masked weights are 0
    print("\nWeights for 3rd batch, 1st query (should be 0 for last 4 tokens):")
    print(weights_masked[2, 0, :])

    # The sum of weights for the last 4 tokens should be 0
    masked_sum = weights_masked[2, 0, 8:].sum()
    print(f"Sum of masked weights: {masked_sum.item()}")
    assert masked_sum < 1e-6  # Should be effectively zero
