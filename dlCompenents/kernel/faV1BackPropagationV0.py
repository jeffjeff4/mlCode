import torch


def flash_attention_backward(Q, K, V, P, dA, Sr=4):
    B, S, D = Q.shape
    H = 2  # 简化，2 个头
    Dh = D // H
    dQ, dK, dV = torch.zeros_like(Q), torch.zeros_like(K), torch.zeros_like(V)

    # 分块循环
    for i in range(0, S, Sr):
        Qi = Q[:, i:i + Sr, :]  # B×Sr×D
        dAi = dA[:, i:i + Sr, :]  # B×Sr×D
        for j in range(0, S, Sr):
            Kj, Vj = K[:, j:j + Sr, :], V[:, j:j + Sr, :]
            Pij = P[:, i:i + Sr, j:j + Sr]  # 前向保存的 P
            # 梯度计算
            dPij = dAi @ Vj.transpose(-2, -1)  # ∂L/∂P = ∂L/∂A V^T
            dSij = Pij * (dPij - (Pij * dPij).sum(dim=-1, keepdim=True))  # ∂L/∂S
            dV[:, j:j + Sr, :] += Pij.transpose(-2, -1) @ dAi  # ∂L/∂V
            dQ[:, i:i + Sr, :] += dSij @ Kj  # ∂L/∂Q
            dK[:, j:j + Sr, :] += dSij.transpose(-2, -1) @ Qi  # ∂L/∂K

    # 参数梯度
    dWq = X.transpose(-2, -1) @ dQ
    dWk = X.transpose(-2, -1) @ dK
    dWv = X.transpose(-2, -1) @ dV
    return dWq, dWk, dWv


# 示例数据
X = torch.tensor([[[1, 0, 0, 1],
                   [0, 1, 1, 0],
                   [1, 0, 0, 1],
                   [0, 1, 1, 0],
                   [1, 0, 0, 1],
                   [0, 1, 1, 0],
                   [1, 0, 0, 1],
                   [0, 1, 1, 0]]], dtype=torch.float32)
Q, K, V = X, X, X  # 简化
P = torch.ones(1, 8, 8) * 0.25  # 假设前向 Softmax 输出
dA = torch.ones(1, 8, 4)  # 输入梯度
dWq, dWk, dWv = flash_attention_backward(Q, K, V, P, dA, Sr=4)