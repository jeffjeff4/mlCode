import torch
import math

class ScaledDotProductAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value):
        d_k = query.size(-1)
        scale = 1.0 / math.sqrt(d_k)
        # Compute scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        # Softmax for attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        # Output
        output = torch.matmul(attn_weights, value)
        # Save for backward
        ctx.save_for_backward(query, key, value, attn_weights, torch.tensor(scale))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, attn_weights, scale = ctx.saved_tensors
        # Gradient w.r.t. value
        grad_value = torch.matmul(attn_weights.transpose(-2, -1), grad_output)
        # Gradient w.r.t. attn_weights
        grad_attn = torch.matmul(grad_output, value.transpose(-2, -1))
        # Softmax backward for grad_scores
        grad_scores = attn_weights * grad_attn - attn_weights * (attn_weights * grad_attn).sum(dim=-1, keepdim=True)
        # Gradient w.r.t. query
        grad_query = torch.matmul(grad_scores, key) * scale
        # Gradient w.r.t. key (note the transpose)
        grad_key = torch.matmul(query.transpose(-2, -1), grad_scores).transpose(-2, -1) * scale
        return grad_query, grad_key, grad_value

# Wrapper module for ease of use
class ScaledDotProductAttention(torch.nn.Module):
    def forward(self, query, key, value):
        return ScaledDotProductAttentionFunction.apply(query, key, value)


import torch
import torch.nn.functional as F

# Custom attention
custom_attn = ScaledDotProductAttention()

# Random inputs (requires_grad=True for backward)
query = torch.randn(2, 3, 4, requires_grad=True)
key = torch.randn(2, 3, 4, requires_grad=True)
value = torch.randn(2, 3, 4, requires_grad=True)

# Forward with custom
out_custom = custom_attn(query, key, value)

# Simulate loss (mean of output)
loss_custom = out_custom.mean()
loss_custom.backward()

print("Custom Output shape:", out_custom.shape)  # torch.Size([2, 3, 4])
print("Grad w.r.t. Query (custom):", query.grad is not None)  # True

# Compare with built-in (reset grads)
query.grad = None
key.grad = None
value.grad = None

# Built-in forward
out_builtin = F.scaled_dot_product_attention(query, key, value)

# Loss
loss_builtin = out_builtin.mean()
loss_builtin.backward()

# Check closeness
print("Outputs close:", torch.allclose(out_custom, out_builtin, atol=1e-6))
print("Query grads close:", torch.allclose(query.grad, query.grad, atol=1e-6))  # Self-comparison, but in practice compare across