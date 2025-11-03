import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute stable softmax for a batch of inputs.
    Args:
        x: np.ndarray of shape (batch_size, num_classes)
    Returns:
        np.ndarray of same shape, softmax probabilities
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute cross-entropy loss for classification.
    Args:
        pred: np.ndarray of shape (batch_size, num_classes), probabilities
        target: np.ndarray of shape (batch_size,), true class indices
    Returns:
        scalar float — average cross-entropy loss
    """
    # Avoid log(0)
    eps = 1e-12
    pred = np.clip(pred, eps, 1.0 - eps)

    # Select probabilities corresponding to the correct class
    #correct_log_probs = -np.log(pred[np.arange(len(target)), target])
    tmp0 = np.arange(len(target))
    tmp1 = pred[tmp0, target]
    correct_log_probs = -np.log(tmp1)

    return np.mean(correct_log_probs)


# ✅ Test
x = np.array([[2.0, 1.0, 0.1]])
target = np.array([0])
assert np.allclose(cross_entropy(softmax(x), target), 0.417, atol=1e-3)

print("✅ Test passed. Cross entropy =", cross_entropy(softmax(x), target))


