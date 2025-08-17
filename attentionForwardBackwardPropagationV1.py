import torch
from torch.autograd import Function

class AttentionFunction(Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        # 保存中间结果供 backward 使用
        ctx.save_for_backward(Q, K, V, attn)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, attn = ctx.saved_tensors
        d_k = Q.size(-1)

        # grad_output: dL/dOut
        # Out = attn @ V
        dAttn = torch.matmul(grad_output, V.transpose(-2, -1))   # dL/dAttn
        dV = torch.matmul(attn.transpose(-2, -1), grad_output)   # dL/dV

        # attn = softmax(scores)
        # softmax 梯度: dAttn -> dScores
        dScores = dAttn * attn - attn * (dAttn * attn).sum(dim=-1, keepdim=True)

        # scores = QK^T / sqrt(d_k)
        dQ = torch.matmul(dScores, K) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype))
        dK = torch.matmul(dScores.transpose(-2, -1), Q) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype))

        return dQ, dK, dV


class MyAttention(torch.nn.Module):
    def forward(self, Q, K, V):
        return AttentionFunction.apply(Q, K, V)


# 假设 batch=1, 序列长度=3, d_k=4, d_v=4
torch.manual_seed(0)

Q = torch.randn(1, 3, 4, requires_grad=True)
K = torch.randn(1, 3, 4, requires_grad=True)
V = torch.randn(1, 3, 4, requires_grad=True)

attn = MyAttention()
out = attn(Q, K, V)

print("Forward Output:\n", out)

# 定义损失 = 输出和全 1 的 MSE
loss = ((out - 1.0)**2).mean()
loss.backward()

print("\nLoss:", loss.item())
print("Grad Q:\n", Q.grad)
print("Grad K:\n", K.grad)
print("Grad V:\n", V.grad)
