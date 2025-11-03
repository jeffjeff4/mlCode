import numpy as np


def layer_norm(x, eps=1e-5):
    """
    Applies Layer Normalization over the last dimension of x.

    Args:
        x (np.ndarray): Input array of shape (..., D)
        eps (float): Small constant to avoid division by zero
    Returns:
        np.ndarray: Normalized array with same shape as x
    """
    # Compute mean and variance along last dimension
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    # Normalize
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm


# ✅ Test
x = np.array([[1, 2, 3],
              [4, 5, 6]])
out = layer_norm(x)
print(out)
