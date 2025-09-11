import torch


def flash_attention_v1_forward_conceptual(q, k, v, B, H, N, D):
    """
    概念性地模拟 FlashAttention v1 的前向传播。
    这段代码仅用于理解核心原理，并不代表实际的CUDA实现。

    参数:
    q, k, v: 输入的查询、键、值张量。形状分别为 (B, H, N, D)。
    B: 批次大小 (Batch size)
    H: 注意力头数量 (Number of heads)
    N: 序列长度 (Sequence length)
    D: 向量维度 (Head dimension)
    """

    # --- FlashAttention的核心概念 ---
    # 1. 分块 (Tiling): 将Q, K, V沿序列长度N维度切分成小块。
    # 2. 内存优化: 在小块上进行计算，避免将中间结果（注意力矩阵）写入高带宽内存（HBM）。
    # ----------------------------------

    # 假设我们设置的块大小（tile size）为 T。
    # 块大小是一个超参数，需要根据GPU的SRAM大小来选择。
    T = 64

    # 初始化最终输出张量，并初始化为0
    o = torch.zeros_like(q, device=q.device)

    # 初始化 Softmax 的归一化因子（l_i）和最大值（m_i）
    # m_i 存储每个输出行当前的最大注意力得分，l_i 存储Softmax分母的累加和
    m_i = torch.full((B, H, N), float('-inf'), device=q.device)
    l_i = torch.zeros((B, H, N), device=q.device)

    # 循环遍历Q矩阵的块（i表示第i个块）
    for i in range(0, N, T):
        # 从高带宽内存（HBM）加载第i个Q块到片上SRAM
        q_tile = q[:, :, i:i + T, :]

        # 在内部循环中，遍历K和V矩阵的块（j表示第j个块）
        for j in range(0, N, T):
            # 从高带宽内存（HBM）加载第j个K和V块到片上SRAM
            k_tile = k[:, :, j:j + T, :]
            v_tile = v[:, :, j:j + T, :]

            # 在片上SRAM中计算Q和K的点积（S_ij）
            s_ij = torch.einsum('bhid,bhjd->bhij', q_tile, k_tile) / D ** 0.5

            # --- Softmax的在线计算和归一化 ---
            # 这是一个概念性的步骤，用于更新Softmax归一化因子。
            # 这就是FlashAttention避免将整个N x N注意力矩阵写入HBM的关键。

            # 1. 找到当前块s_ij的最大值
            m_ij_new, _ = torch.max(s_ij, dim=-1)

            # 2. 合并当前块最大值和历史最大值m_i
            m_i_prev = m_i[:, :, i:i + T].clone()
            m_i_new = torch.max(m_i_prev, m_ij_new)

            # 3. 更新Softmax归一化分母（l_i）
            # 这是在线更新分母的巧妙之处。
            l_i_new = torch.exp(m_i_prev - m_i_new) * l_i[:, :, i:i + T] + torch.exp(m_ij_new - m_i_new)

            # 4. 根据新的归一化因子更新Softmax结果
            p_ij = torch.exp(s_ij - m_i_new.unsqueeze(-1)) / l_i_new.unsqueeze(-1)

            # --- 累加计算输出O ---
            # 将新的P_ij结果与v_tile相乘，并加到输出O中
            o_tile = torch.einsum('bhij,bhjd->bhid', p_ij, v_tile)
            o[:, :, i:i + T, :] = (l_i[:, :, i:i + T].unsqueeze(-1) * torch.exp(m_i_prev - m_i_new).unsqueeze(-1)) * o[
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     i:i + T,
                                                                                                                     :] + o_tile

            # 更新全局的m_i和l_i
            m_i[:, :, i:i + T] = m_i_new
            l_i[:, :, i:i + T] = l_i_new

    return o


# --- 验证代码（使用标准的PyTorch注意力作为参考）---
B, H, N, D = 2, 8, 256, 64  # 批次大小、头数、序列长度、维度
#q = torch.randn(B, H, N, D, device='cuda', requires_grad=True)
#k = torch.randn(B, H, N, D, device='cuda', requires_grad=True)
#v = torch.randn(B, H, N, D, device='cuda', requires_grad=True)

q = torch.randn(B, H, N, D, device='cpu', requires_grad=True)
k = torch.randn(B, H, N, D, device='cpu', requires_grad=True)
v = torch.randn(B, H, N, D, device='cpu', requires_grad=True)

# 1. 标准PyTorch注意力
attn_matrix = torch.einsum('bhid,bhjd->bhij', q, k) / D ** 0.5
attn_weights = torch.softmax(attn_matrix, dim=-1)
standard_o = torch.einsum('bhij,bhjd->bhid', attn_weights, v)

# 2. FlashAttention概念性代码
flash_o = flash_attention_v1_forward_conceptual(q.detach(), k.detach(), v.detach(), B, H, N, D)

# 3. 比较结果
print(f"标准注意力输出的形状: {standard_o.shape}")
print(f"FlashAttention概念性代码输出的形状: {flash_o.shape}")
# 通常会因为浮点数精度问题有一点微小的误差，但结果应该非常接近
#print(f"两个结果的最大差异: {torch.max(torch.abs(standard_o - flash_o.cuda()))}")
print(f"两个结果的最大差异: {torch.max(torch.abs(standard_o - flash_o.cpu()))}")