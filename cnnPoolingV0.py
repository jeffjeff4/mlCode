import numpy as np


def max_pool2d(X, pool_size=2, stride=2):
    """
    2D Max Pooling

    Args:
        X: Input array of shape (H, W) or (C, H, W)
        pool_size: Size of pooling window
        stride: Step size for sliding window

    Returns:
        Pooled array
    """
    # Handle both single-channel and multi-channel
    if X.ndim == 2:
        X = X[np.newaxis, ...]  # Add channel dimension

    C, H, W = X.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    pooled = np.zeros((C, out_h, out_w))

    for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                h_end = h_start + pool_size
                w_start = j * stride
                w_end = w_start + pool_size
                pooled[c, i, j] = np.max(X[c, h_start:h_end, w_start:w_end])

    return pooled.squeeze()  # remove extra channel dim if single-channel


def avg_pool2d(X, pool_size=2, stride=2):
    """
    2D Average Pooling
    """
    if X.ndim == 2:
        X = X[np.newaxis, ...]

    #H_out = floor[(H_in + 2⋅P_H - K_H) / s_h] + 1
    #𝑊_𝑜𝑢𝑡 = floor[(𝑊_𝑖𝑛 + 2⋅𝑃_𝑊 − 𝐾_𝑊) / 𝑆_𝑊] + 1
    C, H, W = X.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    pooled = np.zeros((C, out_h, out_w))

    for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                h_end = h_start + pool_size
                w_start = j * stride
                w_end = w_start + pool_size
                pooled[c, i, j] = np.mean(X[c, h_start:h_end, w_start:w_end])

    return pooled.squeeze()


X = np.array([
    [1, 3, 2, 4],
    [5, 6, 7, 8],
    [2, 4, 6, 8],
    [1, 2, 3, 4]
])

print("Max Pooling:\n", max_pool2d(X, pool_size=2, stride=2))
print("Average Pooling:\n", avg_pool2d(X, pool_size=2, stride=2))
