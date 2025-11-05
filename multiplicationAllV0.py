import torch

def print_op(title, op_str, result):
    """Helper function to print operations nicely."""
    print("-" * 60)
    print(f"{title}")
    print(f"Operation: {op_str}")
    print(f"Result Tensor:\n{result}")
    print(f"Result Shape: {result.shape}")
    print("-" * 60, "\n")

# --- 1. Element-wise Multiplication (* or torch.mul) ---
print("=" * 70)
print(" 1. ELEMENT-WISE MULTIPLICATION (Hadamard Product) ")
print("=" * 70)

a_elem = torch.tensor([[1, 2], [3, 4]])
b_elem = torch.tensor([[10, 20], [30, 40]])
print(f"Input A:\n{a_elem}")
print(f"Input B:\n{b_elem}\n")

# Using the '*' operator
result_elem_op = a_elem * b_elem
print_op("Element-wise (*)", "a_elem * b_elem", result_elem_op)

# Using the torch.mul function
result_elem_func = torch.mul(a_elem, b_elem)
print_op("Element-wise (torch.mul)", "torch.mul(a_elem, b_elem)", result_elem_func)

# With Broadcasting using *
c_broadcast = torch.tensor([10, 20])
print(f"Input A:\n{a_elem}")
print(f"Input C (for broadcasting) using *:\n{c_broadcast}\n")
result_broadcastV0 = a_elem * c_broadcast
print_op("Element-wise (Broadcasting)", "a_elem * c_broadcast", result_broadcastV0)

# With Broadcasting using torch.mul
c_broadcast = torch.tensor([10, 20])
print(f"Input A:\n{a_elem}")
print(f"Input C (for broadcasting) using torch.mul:\n{c_broadcast}\n")
result_broadcastV1 = torch.mul(a_elem, c_broadcast)
print_op("Element-wise (Broadcasting)", "torch.mul(a_elem, c_broadcast)", result_broadcastV1)

# --- 2. Matrix Multiplication (@ or torch.matmul) ---
print("=" * 70)
print(" 2. MATRIX MULTIPLICATION (Dot Product) ")
print("=" * 70)

# Case 2a: 1D x 1D (Vector Dot Product)
v1 = torch.tensor([1, 2, 3])
v2 = torch.tensor([4, 5, 6])
print(f"Vector v1:\n{v1}")
print(f"Vector v2:\n{v2}\n")

result_dot_at = v1 @ v2
print_op("1D x 1D (@)", "v1 @ v2", result_dot_at)

result_dot_matmul = torch.matmul(v1, v2)
print_op("1D x 1D (torch.matmul)", "torch.matmul(v1, v2)", result_dot_matmul)


# Case 2b: 2D x 2D (Matrix Multiply)
m1 = torch.tensor([[1, 2], [3, 4]])
m2 = torch.tensor([[5, 6], [7, 8]])
print(f"Matrix m1 (2x2):\n{m1}")
print(f"Matrix m2 (2x2):\n{m2}\n")

result_mm_at = m1 @ m2
print_op("2D x 2D (@)", "m1 @ m2", result_mm_at)

result_mm_matmul = torch.matmul(m1, m2)
print_op("2D x 2D (torch.matmul)", "torch.matmul(m1, m2)", result_mm_matmul)


# Case 2c: 3D x 3D (Batch Matrix Multiply)
b1 = torch.randn(2, 3, 4) # Batch of 2, 3x4 matrices
b2 = torch.randn(2, 4, 2) # Batch of 2, 4x2 matrices
print(f"Batch b1 (2x3x4):\n{b1.shape}")
print(f"Batch b2 (2x4x2):\n{b2.shape}\n")

result_bmm_at = b1 @ b2
print_op("3D x 3D (@)", "b1 @ b2", result_bmm_at)

result_bmm_matmul = torch.matmul(b1, b2)
print_op("3D x 3D (torch.matmul)", "torch.matmul(b1, b2)", result_bmm_matmul)


# --- 3. Specialized (Strict) Multiplications ---
print("=" * 70)
print(" 3. SPECIALIZED (STRICT) MULTIPLICATIONS ")
print("=" * 70)

# torch.dot (Strictly 1D)
print(f"Vector v1:\n{v1}")
print(f"Vector v2:\n{v2}\n")
result_dot_strict = torch.dot(v1, v2)
print_op("torch.dot (1D only)", "torch.dot(v1, v2)", result_dot_strict)
# torch.dot(m1, m2) # This would FAIL!

# torch.mm (Strictly 2D)
print(f"Matrix m1 (2x2):\n{m1}")
print(f"Matrix m2 (2x2):\n{m2}\n")
result_mm_strict = torch.mm(m1, m2)
print_op("torch.mm (2D only)", "torch.mm(m1, m2)", result_mm_strict)
# torch.mm(b1, b2) # This would FAIL!

# torch.bmm (Strictly 3D)
print(f"Batch b1 (2x3x4):\n{b1.shape}")
print(f"Batch b2 (2x4x2):\n{b2.shape}\n")
result_bmm_strict = torch.bmm(b1, b2)
print_op("torch.bmm (3D only)", "torch.bmm(b1, b2)", result_bmm_strict)
# torch.bmm(m1, m2) # This would FAIL!


# --- 4. Einstein Summation (torch.einsum) ---
print("=" * 70)
print(" 4. EINSTEIN SUMMATION (torch.einsum) ")
print("=" * 70)

print(f"Matrix m1 (2x2):\n{m1}")
print(f"Matrix m2 (2x2):\n{m2}\n")

# Re-doing Matrix Multiply (2D) with einsum
result_einsum_mm = torch.einsum("ik, kj -> ij", m1, m2)
print_op("einsum (Matrix Multiply)", 'torch.einsum("ik, kj -> ij", m1, m2)', result_einsum_mm)

# Re-doing Element-wise with einsum
result_einsum_elem = torch.einsum("ij, ij -> ij", m1, m2)
print_op("einsum (Element-wise)", 'torch.einsum("ij, ij -> ij", m1, m2)', result_einsum_elem)

# Re-doing Dot Product (1D) with einsum
print(f"Vector v1:\n{v1}")
print(f"Vector v2:\n{v2}\n")
result_einsum_dot = torch.einsum("i, i ->", v1, v2)
print_op("einsum (Dot Product)", 'torch.einsum("i, i ->", v1, v2)', result_einsum_dot)
