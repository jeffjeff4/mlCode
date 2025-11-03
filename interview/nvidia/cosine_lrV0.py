import math


def cosine_lr(epoch, total_epochs, base_lr):
    """
    Cosine annealing learning rate schedule.

    Args:
        epoch (int): Current epoch (0-based)
        total_epochs (int): Total number of epochs
        base_lr (float): Initial learning rate

    Returns:
        float: Adjusted learning rate
    """
    if epoch < 0 or epoch > total_epochs:
        raise ValueError("epoch must be in [0, total_epochs]")

    # Cosine decay: lr = base_lr * 0.5 * (1 + cos(pi * epoch / total_epochs))
    lr = base_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    return lr


# ✅ Tests
assert round(cosine_lr(0, 100, 0.1), 5) == 0.1  # start = base_lr
assert round(cosine_lr(50, 100, 0.1), 5) == 0.05  # midpoint
assert round(cosine_lr(100, 100, 0.1), 5) == 0.0  # end = 0
print("All tests passed ✅")
