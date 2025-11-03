import numpy as np

def average_gradients(grad_list):
    """
    Average a list of gradient vectors.

    Args:
        grad_list (List[np.ndarray]): List of gradient arrays (same shape)
    Returns:
        np.ndarray: Element-wise average of gradients
    """
    if not grad_list:
        raise ValueError("grad_list cannot be empty")

    # Stack and average element-wise
    return np.mean(np.stack(grad_list, axis=0), axis=0)


# ✅ Test
grads = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
assert np.allclose(average_gradients(grads), np.array([3, 4]))
print("All tests passed ✅")
