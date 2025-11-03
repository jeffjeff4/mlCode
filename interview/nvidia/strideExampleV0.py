import numpy as np

# 1D Vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 2D Matrices
X = np.array([[1, 2, 3],
              [4, 5, 6]]) # shape (2, 3)

Y = np.array([[7, 8],
              [9, 10],
              [11, 12]]) # shape (3, 2)

# 'ij': input is shape (i, j)
# '-> ji': output is shape (j, i)
result = np.einsum('ij -> ji', X)
print(f"Original shape: {X.shape}")
print(f"Transposed shape: {result.shape}")
print(result)
print()


result1 = np.einsum('ij -> ij', X)
print("-----------result1---------")
print(f"Original shape: {X.shape}")
print(f"result1 shape: {result1.shape}")
print(result1)
print()

result2 = np.einsum('ij,ij -> ij', X, Y)
print("-----------result2---------")
print(f"Original shape: {X.shape}")
print(f"result2 shape: {result2.shape}")
print(result2)