def matmul(A, B):
    """
    Perform matrix multiplication A x B.
    A: list of lists, shape (m x n)
    B: list of lists, shape (n x p)
    Returns: list of lists, shape (m x p)
    """
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    assert n == n2, "Matrix dimension mismatch"

    # Initialize result matrix with zeros
    result = [[0 for _ in range(p)] for _ in range(m)]

    # Compute matrix multiplication
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result


# ✅ Test
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
assert matmul(A, B) == [[19, 22], [43, 50]]
print("✅ Test passed")
