import numpy as np


def grad_check(f, x, analytical_grad, eps=1e-5):
    """Compare analytical gradient vs numerical gradient using finite differences."""
    num_grad = np.zeros_like(x)

    for i in range(len(x)):
        x_pos = x.copy()
        x_neg = x.copy()
        x_pos[i] += eps
        x_neg[i] -= eps

        # Central difference approximation
        num_grad[i] = (f(x_pos) - f(x_neg)) / (2 * eps)

    # Compute relative error
    numerator = np.linalg.norm(num_grad - analytical_grad)
    denominator = np.linalg.norm(num_grad) + np.linalg.norm(analytical_grad)
    rel_error = numerator / (denominator + 1e-12)

    print(f"Numerical grad: {num_grad}")
    print(f"Analytical grad: {analytical_grad}")
    print(f"Relative error: {rel_error:.6e}")

    # Simple assertion for check
    if rel_error < 1e-6:
        print("✅ Gradient check PASSED")
    else:
        print("❌ Gradient check FAILED")


# ✅ Test
f = lambda x: x ** 2
x = np.array([2.0])
grad_check(f, x, np.array([4.0]))
