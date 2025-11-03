import torch
import torch.nn as nn
import math


def scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
) -> (torch.Tensor, torch.Tensor):
    """
    This is the core attention function.
    It's implemented here as a dependency for the MHA module.

    Args:
        q: Queries. Shape: (..., seq_len_q, d_k)
        k: Keys. Shape: (..., seq_len_k, d_k)
        v: Values. Shape: (..., seq_len_v, d_v)
           (where seq_len_k == seq_len_v)
        mask: Optional mask. Shape: (..., seq_len_q, seq_len_k)
              The mask should be a boolean tensor (or 0/1).
              Positions with 0 (or False) will be masked (set to -inf).

    Returns:
        (output, attention_weights)
        output: Shape: (..., seq_len_q, d_v)
        attention_weights: Shape: (..., seq_len_q, seq_len_k)
    """
    d_k = q.size(-1)  # Get the dimension of the keys

    # 1. Calculate scores: (..., S_q, d_k) @ (..., d_k, S_k) -> (..., S_q, S_k)
    scores = torch.matmul(q, k.transpose(-2, -1))

    # 2. Scale scores
    scaled_scores = scores / math.sqrt(d_k)

    # 3. Apply mask (if provided)
    if mask is not None:
        # Fills elements with -1e9 where mask is 0 (or False)
        scaled_scores = scaled_scores.masked_fill(mask == 0, -1e9)

    # 4. Apply softmax to get weights
    # Softmax is applied on the last dimension (S_k)
    attention_weights = torch.softmax(scaled_scores, dim=-1)

    # 5. Multiply weights by values
    # (..., S_q, S_k) @ (..., S_v, d_v) -> (..., S_q, d_v)
    # (S_k must be == S_v)
    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadSelfAttention(nn.Module):
    """
    Implements Multi-Head Self-Attention as described in
    "Attention is All You Need".

    Q, K, and V are all derived from the same input tensor 'x'.
    """

    def __init__(self, d_model: int, num_heads: int):
        """
        Args:
            d_model (int): The total dimensionality of the input 'x'
                           and the output. (e.g., 512)
            num_heads (int): The number of parallel attention heads.
                             'd_model' must be divisible by 'num_heads'. (e.g., 8)
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        # d_head is the dimension for each head's Q, K, V
        self.d_head = d_model // num_heads

        # 1. Create the linear projection layers for Q, K, V
        # In Self-Attention, Wq, Wk, and Wv are applied to the *same* input 'x'
        self.wq = nn.Linear(d_model, d_model)  # W_q
        self.wk = nn.Linear(d_model, d_model)  # W_k
        self.wv = nn.Linear(d_model, d_model)  # W_v

        # 6. The final output linear layer (W_o)
        self.fc_out = nn.Linear(d_model, d_model)  # W_o

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            x (torch.Tensor): The input tensor.
                              Shape: (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): A mask to be applied.
                                          Shape: (batch_size, 1, 1, seq_len) or
                                                 (batch_size, 1, seq_len, seq_len)
                                          (The 1s will broadcast to 'num_heads')

        Returns:
            (output, attention_weights)
            output: The final processed tensor.
                    Shape: (batch_size, seq_len, d_model)
            attention_weights: The attention weights from the core function.
                               Shape: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]  # seq_len_q = seq_len_k = seq_len

        # For self-attention, q_input = k_input = v_input = x
        q_input, k_input, v_input = x, x, x

        # 1. Project Q, K, V
        # (B, S, D_model) -> (B, S, D_model)
        Q = self.wq(q_input)
        K = self.wk(k_input)
        V = self.wv(v_input)

        # 2. Split into heads
        # (B, S, D_model) -> (B, S, h, D_head) -> (B, h, S, D_head)
        # We transpose to (B, h, S, D_head) so that all batch dimensions
        # (B and h) are up front for the attention function.
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        # 3. Apply scaled dot-product attention
        # The mask will be broadcasted across the 'h' dimension.
        # attention_output shape: (B, h, S, D_head)
        # attention_weights shape: (B, h, S, S)
        attention_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        # 4. Concatenate heads
        # First, transpose back: (B, h, S, D_head) -> (B, S, h, D_head)
        # .contiguous() is needed to fix the memory layout before .view()
        attention_output = attention_output.transpose(1, 2).contiguous()

        # 5. Reshape to (B, S, D_model)
        # This effectively concatenates all the D_head outputs
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)

        # 6. Pass through final linear layer
        # (B, S, D_model) -> (B, S, D_model)
        output = self.fc_out(attention_output)

        return output, attention_weights


# --- Example Usage ---
if __name__ == "__main__":
    # Parameters
    d_model = 512  # Dimension of the model (e.g., embeddings)
    num_heads = 8  # Number of attention heads
    batch_size = 4  # Number of sequences in a batch
    seq_len = 10  # Length of each sequence

    # 1. Create the MHA module
    mha = MultiHeadSelfAttention(d_model, num_heads)

    # 2. Create a dummy input tensor
    # (batch_size, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"Input shape: {x.shape}")

    # 3. Test with no mask (Encoder Self-Attention)
    print("\n--- Testing with no mask (Encoder-style) ---")
    output_no_mask, weights_no_mask = mha(x, mask=None)

    print(f"Output shape: {output_no_mask.shape}")
    print(f"Weights shape: {weights_no_mask.shape}")

    # 4. Test with a causal (look-ahead) mask (Decoder Self-Attention)
    print("\n--- Testing with causal mask (Decoder-style) ---")

    # Create a lower-triangular mask
    # torch.tril creates a matrix with 1s on/below the diagonal, 0s above
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))

    # The mask needs to be broadcastable to (B, h, S, S)
    # So we add the B and h dimensions: (1, 1, S, S)
    # The mask == 0 (where we mask) will be the upper triangle.
    causal_mask = causal_mask.view(1, 1, seq_len, seq_len)

    output_causal, weights_causal = mha(x, mask=causal_mask)

    print(f"Output shape: {output_causal.shape}")
    print(f"Weights shape: {weights_causal.shape}")

    # Check if the causal mask worked
    # The weights for the first token should only be on the first token
    print(f"\nWeights for 1st token (causal): \n{weights_causal[0, 0, 0, :]}")
    # The weights for the last token can be on all tokens
    print(f"\nWeights for last token (causal): \n{weights_causal[0, 0, -1, :]}")

    # The upper triangle of the weights matrix should be all zeros
    # (or very close, due to softmax(-1e9))
    assert torch.allclose(weights_causal[0, 0].triu(diagonal=1), torch.tensor(0.0), atol=1e-6)
    print("\nCausal mask successfully applied: upper triangle of weights is zero.")
