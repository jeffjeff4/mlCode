import torch
import torch.nn.functional as F
import math


def flash_attention_v2_forward(Q, K, V, block_size=1):
    """
    Flash Attention v2 前向传播（简化版），用于生成中间结果供反向传播使用。
    Args:
        Q, K, V: 查询、键、值张量，形状 (batch_size, seq_len, d_model)
        block_size: 分块大小
    Returns:
        O: 注意力输出
        softmax_sum: Softmax 分母（用于反向传播）
        S_blocks: 存储分块的注意力分数（用于重计算）
    """
    batch_size, seq_len, d_model = Q.shape
    scale = 1.0 / math.sqrt(d_model)
    O = torch.zeros_like(Q)
    softmax_sum = torch.zeros(batch_size, seq_len, 1, device=Q.device)
    S_blocks = []  # 存储 S 分块用于反向传播

    for i in range(0, seq_len, block_size):
        Q_block = Q[:, i:i + block_size, :]
        S_row = []
        for j in range(0, seq_len, block_size):
            K_block = K[:, j:j + block_size, :]
            V_block = V[:, j:j + block_size, :]

            # 计算分块注意力分数
            S_block = torch.bmm(Q_block, K_block.transpose(1, 2)) * scale
            S_row.append(S_block)

            # Softmax 前向
            S_block_exp = torch.exp(S_block)
            O[:, i:i + block_size, :] += torch.bmm(S_block_exp, V_block)
            softmax_sum[:, i:i + block_size, :] += S_block_exp.sum(dim=-1, keepdim=True)

        S_blocks.append(S_row)

    O = O / (softmax_sum + 1e-6)
    return O, softmax_sum, S_blocks


def flash_attention_v2_backward(Q, K, V, dO, softmax_sum, S_blocks, block_size=1):
    """
    Flash Attention v2 反向传播，计算 dQ, dK, dV。
    Args:
        Q, K, V: 前向传播的输入
        dO: 输出梯度，形状 (batch_size, seq_len, d_model)
        softmax_sum: 前向传播的 Softmax 分母
        S_blocks: 前向传播的注意力分数分块
        block_size: 分块大小
    Returns:
        dQ, dK, dV: 输入的梯度
    """
    batch_size, seq_len, d_model = Q.shape
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    scale = 1.0 / math.sqrt(d_model)

    for i in range(0, seq_len, block_size):
        Q_block = Q[:, i:i + block_size, :]
        dO_block = dO[:, i:i + block_size, :]
        softmax_sum_block = softmax_sum[:, i:i + block_size, :]

        for j in range(0, seq_len, block_size):
            K_block = K[:, j:j + block_size, :]
            V_block = V[:, j:j + block_size, :]
            S_block = S_blocks[i // block_size][j // block_size]

            # 重新计算 Softmax 输出
            A_block = torch.exp(S_block) / (softmax_sum_block + 1e-6)

            # 计算 dV
            dV_block = torch.bmm(A_block.transpose(1, 2), dO_block)
            dV[:, j:j + block_size, :] += dV_block

            # 计算 dA（Softmax 的梯度）
            dA_block = torch.bmm(dO_block, V_block.transpose(1, 2))

            # Softmax 梯度
            A_block_grad = A_block * (dA_block - torch.sum(dA_block * A_block, dim=-1, keepdim=True))

            # 计算 dS
            dS_block = A_block_grad * scale

            # 计算 dQ 和 dK
            dQ_block = torch.bmm(dS_block, K_block)
            dK_block = torch.bmm(dS_block.transpose(1, 2), Q_block)
            dQ[:, i:i + block_size, :] += dQ_block
            dK[:, j:j + block_size, :] += dK_block

    return dQ, dK, dV


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

# 前向传播（仅为生成中间结果）
Q = torch.bmm(X, W_Q.unsqueeze(0))
K = torch.bmm(X, W_K.unsqueeze(0))
V = torch.bmm(X, W_V.unsqueeze(0))
O, softmax_sum, S_blocks = flash_attention_v2_forward(Q, K, V, block_size=1)

# 损失函数
loss = F.mse_loss(O, Y_target)
print(f"Loss: {loss.item()}")

# 计算输出梯度
dO = torch.autograd.grad(loss, O, retain_graph=True)[0]

# 反向传播
dQ, dK, dV = flash_attention_v2_backward(Q, K, V, dO, softmax_sum, S_blocks, block_size=1)

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