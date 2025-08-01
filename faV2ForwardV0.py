import torch


def flash_attention_v2_forward(Q, K, V, Sr=4):
    B, S, D = Q.shape
    H = 2  # 简化，2 个头
    Dh = D // H
    A = torch.zeros(B, S, D)  # 输出
    l = torch.zeros(B, S)  # Softmax 归一化因子
    m = torch.full((B, S), -float('inf'))  # 最大值

    # 序列长度并行：每个线程块处理 Q_i
    for i in range(0, S, Sr):
        Qi = Q[:, i:i + Sr, :]  # B×Sr×D
        Ai = torch.zeros(B, Sr, D)  # 块输出
        li = torch.zeros(B, Sr)  # 块归一化因子
        mi = torch.full((B, Sr), -float('inf'))  # 块最大值

        # 内循环：K, V 块，共享内存
        for j in range(0, S, Sr):
            Kj, Vj = K[:, j:j + Sr, :], V[:, j:j + Sr, :]
            # 4 warps 分片 Q_i 的行
            for k in range(0, Sr):  # 模拟 warp 分片
                Qik = Qi[:, k:k + 1, :]  # B×1×D
                Sijk = torch.matmul(Qik, Kj.transpose(-2, -1)) / (Dh ** 0.5)  # B×1×Sr
                mijk = torch.max(Sijk, dim=-1)[0]  # B×1
                Pijk = torch.exp(Sijk - mijk.unsqueeze(-1))  # B×1×Sr
                lijk = torch.sum(Pijk, dim=-1)  # B×1
                Ai_new = torch.matmul(Pijk, Vj)  # B×1×D
                mi_new = torch.max(mi[:, k], mijk)
                li_new = li[:, k] * torch.exp(mi[:, k] - mi_new) + lijk * torch.exp(mijk - mi_new)
                Ai[:, k:k + 1, :] = li[:, k:k + 1].unsqueeze(-1) * torch.exp(mi[:, k:k + 1] - mi_new).unsqueeze(-1) * \
                                    Ai[:, k:k + 1, :] + torch.exp(mijk - mi_new).unsqueeze(-1) * Ai_new
                li[:, k], mi[:, k] = li_new, mi_new
        A[:, i:i + Sr, :] = Ai / li.unsqueeze(-1)
        l[:, i:i + Sr], m[:, i:i + Sr] = li, mi

    return A, l, m


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
A, l, m = flash_attention_v2_forward(Q, K, V, Sr=4)
print("A = ", A)
print("l = ", l)
print("m = ", m)
