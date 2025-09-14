import torch
import torch.nn as nn
from torch.distributions.constraints import lower_triangular
from torchtyping import TensorType
import numpy as np

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        k = self.key_gen(embedded)
        q = self.query_gen(embedded)
        v = self.value_gen(embedded)

        k_transposed = torch.transpose(k, 1, 2)
        scores = torch.matmul(q, k_transposed)
        batch_size = k.shape[0]
        context_length = k.shape[1]
        attention_dim = k.shape[2]

        scores = scores / (attention_dim ** 0.5)

        lower_triangle = torch.tril(torch.ones(context_length, context_length))
        mask = lower_triangle == 0
        scores = scores.masked_fill(mask, float("-Infinity"))
        scores = nn.functional.softmax(scores, dim=2)

        rst = scores @ v
        rst = torch.round(rst, decimals=4)
        return rst


dim0 = 2
dim1 = 3
dim2 = 4

embedded = np.random.rand(dim0, dim1, dim2).astype(np.float32)
sol = SingleHeadAttention(dim2, dim1)

embedded_tensor = torch.tensor(embedded)
rst = sol.forward(embedded_tensor)
print("rst = ", rst)
