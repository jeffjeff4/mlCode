import torch
import numpy as np

# 输入激活张量
A = torch.tensor([
    [0.1, 0.2, 10.0, 0.3],
    [0.2, 0.1, 9.8, 0.4],
    [0.3, 0.3, 10.2, 0.2]
])

# 构造旋转矩阵（针对第3列和第4列，theta=pi/4）
theta = np.pi / 4
cos_theta, sin_theta = np.cos(theta), np.sin(theta)
R = torch.eye(4)
R[2, 2], R[2, 3] = cos_theta, -sin_theta
R[3, 2], R[3, 3] = sin_theta, cos_theta

# 应用旋转
A_prime = A @ R

print("Before Rotation:\n", A)
print("Rotation Matrix:\n", R)
print("After Rotation:\n", A_prime)