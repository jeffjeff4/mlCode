import torch
import torch.nn.functional as F
import math

def flash_attention_v3_forward(Q, K, V, block_size=1):
    """
    Flash Attention v3 前向传播，模拟分块计算和异步操作。
    Args:
        Q, K, V: 查询、键、值张量，形状 (batch_size, seq_len, d_model)
        block_size: 分块大小，控制内存使用
    Returns:
        O: 注意力输出，形状 (batch_size, seq_len, d_model)
        softmax_stats: (m_i, l_i) 用于重计算的 Softmax 统计信息
    """
    batch_size, seq_len, d_model = Q.shape
    scale = 1.0 / math.sqrt(d_model)
    # Initialize O, m, l in FP16 to match computation dtype
    O = torch.zeros(batch_size, seq_len, d_model, dtype=torch.float16, device=Q.device)
    m = torch.full((batch_size, seq_len, 1), -float('inf'), dtype=torch.float16, device=Q.device)
    l = torch.zeros((batch_size, seq_len, 1), dtype=torch.float16, device=Q.device)

    # 模拟 FP8 量化（使用 FP16）
    Q = Q.to(torch.float16)
    K = K.to(torch.float16)
    V = V.to(torch.float16)

    # 分块计算，模拟异步 GEMM 和 Softmax
    for i in range(0, seq_len, block_size):
        Q_block = Q[:, i:i+block_size, :]
        for j in range(0, seq_len, block_size):
            K_block = K[:, j:j+block_size, :]
            V_block = V[:, j:j+block_size, :]

            # 计算 QK^T / sqrt(d_k)
            S_block = torch.bmm(Q_block, K_block.transpose(1, 2)) * scale

            # Softmax（增量式更新，模拟异步）
            m_block = torch.max(S_block, dim=-1, keepdim=True)[0]
            m_new = torch.max(m[:, i:i+block_size, :], m_block)
            exp_m_diff = torch.exp(m[:, i:i+block_size, :] - m_new)
            P_block = torch.exp(S_block - m_new)
            l_block = l[:, i:i+block_size, :] * exp_m_diff + P_block.sum(dim=-1, keepdim=True)
            O[:, i:i+block_size, :] = O[:, i:i+block_size, :] * exp_m_diff + torch.bmm(P_block, V_block)
            m[:, i:i+block_size, :] = m_new
            l[:, i:i+block_size, :] = l_block

    # 归一化输出
    O = O / (l + 1e-6)
    return O.to(torch.float32), (m.to(torch.float32), l.to(torch.float32))

def flash_attention_v3_backward(Q, K, V, dO, softmax_stats, block_size=1):
    """
    Flash Attention v3 反向传播，计算 dQ, dK, dV。
    Args:
        Q, K, V: 前向传播的输入
        dO: 输出梯度，形状 (batch_size, seq_len, d_model)
        softmax_stats: (m_i, l_i) Softmax 统计信息
        block_size: 分块大小
    Returns:
        dQ, dK, dV: 输入梯度
    """
    batch_size, seq_len, d_model = Q.shape
    scale = 1.0 / math.sqrt(d_model)
    # Initialize gradients in FP16
    dQ = torch.zeros(batch_size, seq_len, d_model, dtype=torch.float16, device=Q.device)
    dK = torch.zeros(batch_size, seq_len, d_model, dtype=torch.float16, device=Q.device)
    dV = torch.zeros(batch_size, seq_len, d_model, dtype=torch.float16, device=Q.device)
    m, l = softmax_stats

    # 确保输入和梯度为 FP16
    Q = Q.to(torch.float16)
    K = K.to(torch.float16)
    V = V.to(torch.float16)
    dO = dO.to(torch.float16)
    m = m.to(torch.float16)
    l = l.to(torch.float16)

    # 分块反向传播，重计算 S 和 A
    for i in range(0, seq_len, block_size):
        Q_block = Q[:, i:i+block_size, :]
        dO_block = dO[:, i:i+block_size, :]
        for j in range(0, seq_len, block_size):
            K_block = K[:, j:j+block_size, :]
            V_block = V[:, j:j+block_size, :]

            # 重计算 S_block 和 P_block
            S_block = torch.bmm(Q_block, K_block.transpose(1, 2)) * scale
            P_block = torch.exp(S_block - m[:, i:i+block_size, :]) / (l[:, i:i+block_size, :] + 1e-6)

            # 计算 dV
            dV_block = torch.bmm(P_block.transpose(1, 2), dO_block)
            dV[:, j:j+block_size, :] += dV_block

            # 计算 dP
            dP_block = torch.bmm(dO_block, V_block.transpose(1, 2))

            # Softmax 梯度
            dS_block = P_block * (dP_block - torch.sum(dP_block * P_block, dim=-1, keepdim=True)) * scale

            # 计算 dQ 和 dK
            dQ_block = torch.bmm(dS_block, K_block)
            dK_block = torch.bmm(dS_block.transpose(1, 2), Q_block)
            dQ[:, i:i+block_size, :] += dQ_block
            dK[:, j:j+block_size, :] += dK_block

    return dQ.to(torch.float32), dK.to(torch.float32), dV.to(torch.float32)

# 数据准备
batch_size, seq_len, d_model = 1, 3, 4
X = torch.tensor([[[1., 0., 0., 1.],
                   [0., 1., 0., 1.],
                   [0., 0., 1., 1.]]], requires_grad=True)
Y_target = torch.tensor([[[1., 0., 0., 0.],
                          [0., 1., 0., 0.],
                          [0., 0., 1., 0.]]])

# 权重矩阵
W_Q = torch.eye(d_model, requires_grad=True)
W_K = torch.eye(d_model, requires_grad=True)
W_V = torch.eye(d_model, requires_grad=True)

# 前向传播
Q = torch.bmm(X, W_Q.unsqueeze(0))
K = torch.bmm(X, W_K.unsqueeze(0))
V = torch.bmm(X, W_V.unsqueeze(0))
O, softmax_stats = flash_attention_v3_forward(Q, K, V, block_size=1)

# 损失函数
loss = F.mse_loss(O, Y_target)
print(f"Loss: {loss.item()}")

# 反向传播
dO = torch.autograd.grad(loss, O, retain_graph=True)[0]
dQ, dK, dV = flash_attention_v3_backward(Q, K, V, dO, softmax_stats, block_size=1)

# 计算权重梯度
dW_Q = torch.bmm(X.transpose(1, 2), dQ).squeeze(0)
dW_K = torch.bmm(X.transpose(1, 2), dK).squeeze(0)
dW_V = torch.bmm(X.transpose(1, 2), dV).squeeze(0)

# 更新权重
lr = 0.1
with torch.no_grad():
    W_Q -= lr * dW_Q
    W_K -= lr * dW_K
    W_V -= lr * dW_V

print("Gradient of W_Q:\n", dW_Q)
print("Updated W_Q:\n", W_Q)