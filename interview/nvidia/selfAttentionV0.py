import numpy as np


def self_attention(X: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.

    Args:
        X: np.ndarray of shape (seq_len, d_model)
        Wq, Wk, Wv: np.ndarray of shape (d_model, d_k)
    Returns:
        np.ndarray of shape (seq_len, d_k) — attention output
    """
    # 1. Linear projections
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # 2. Compute scaled dot-product attention scores
    d_k = K.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)

    # 3. Apply softmax along last axis for attention weights
    # for numerical stability
    scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

    # 4. Weighted sum of values
    output = attn_weights @ V

    return output


# ✅ Test
np.random.seed(42)
X = np.random.rand(2, 4)
Wq = Wk = Wv = np.random.rand(4, 4)
out = self_attention(X, Wq, Wk, Wv)
print("Output shape:", out.shape)
print(out)
