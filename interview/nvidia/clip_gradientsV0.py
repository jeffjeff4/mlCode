import numpy as np


def clip_gradients(grads, max_norm):
    """
    Clip gradient vectors to have a maximum global norm of `max_norm`.

    Args:
        grads (list[np.ndarray]): list of gradient arrays
        max_norm (float): maximum allowed L2 norm for all gradients combined
    """
    # Compute global L2 norm across all gradients
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))

    # Compute scaling factor
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        for i in range(len(grads)):
            grads[i] = grads[i] * scale
    return grads


# ✅ Test
grads = [np.array([3, 4], dtype=float), np.array([6, 8], dtype=float)]
clip_gradients(grads, 5)

for g in grads:
    print(g)
