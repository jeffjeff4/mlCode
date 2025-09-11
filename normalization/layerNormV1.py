import torch
import torch.nn as nn

class MyLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, normalized_shape, eps):
        # Compute dimensions for normalization (last len(normalized_shape) dimensions)
        dim = tuple(range(-len(normalized_shape), 0))
        mean = x.mean(dim=dim, keepdim=True)
        var = x.var(dim=dim, unbiased=False, keepdim=True)

        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + eps)

        # Affine transformation
        if weight is not None and bias is not None:
            weight_view = weight.view(*([1] * (x.dim() - len(normalized_shape))), *normalized_shape)
            bias_view = bias.view(*([1] * (x.dim() - len(normalized_shape))), *normalized_shape)
            output = x_hat * weight_view + bias_view
        else:
            output = x_hat

        # Save for backward
        ctx.save_for_backward(x, weight, bias, mean, var, x_hat)
        ctx.eps = eps
        ctx.dim = dim
        ctx.normalized_shape = normalized_shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, mean, var, x_hat = ctx.saved_tensors
        eps = ctx.eps
        dim = ctx.dim
        normalized_shape = ctx.normalized_shape

        # Initialize gradients
        grad_x = None
        grad_weight = None
        grad_bias = None

        # Number of elements in normalized dimensions
        m = torch.prod(torch.tensor(normalized_shape)).item()

        # Compute gradient w.r.t. x_hat
        if weight is not None:
            weight_view = weight.view(*([1] * (x.dim() - len(normalized_shape))), *normalized_shape)
            grad_x_hat = grad_output * weight_view
        else:
            grad_x_hat = grad_output

        # Compute gradient w.r.t. x
        inv_std = 1.0 / torch.sqrt(var + eps)
        x_minus_mean = x - mean
        grad_x = inv_std * (
            grad_x_hat
            - grad_x_hat.mean(dim=dim, keepdim=True)
            - x_minus_mean * (grad_x_hat * x_minus_mean).mean(dim=dim, keepdim=True) / var
        )

        # Compute gradients for weight and bias
        if weight is not None and bias is not None:
            # Sum over batch dimension only
            batch_dim = tuple(range(0, x.dim() - len(normalized_shape)))
            grad_weight = (grad_output * x_hat).sum(dim=batch_dim).view(normalized_shape)
            grad_bias = grad_output.sum(dim=batch_dim).view(normalized_shape)

        return grad_x, grad_weight, grad_bias, None, None

class MyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        return MyLayerNormFunction.apply(
            x, self.weight, self.bias, self.normalized_shape, self.eps
        )

if __name__ == "__main__":
    # Test with 2D normalized shape
    ln = MyLayerNorm(normalized_shape=[10, 128])
    x = torch.randn(32, 10, 128, requires_grad=True)
    y = ln(x)
    print("Output shape:", y.shape)

    # Test backpropagation
    y.mean().backward()
    print("Input gradient:", x.grad is not None)
    print("Weight gradient:", ln.weight.grad is not None if ln.affine else "No affine")
    print("Bias gradient:", ln.bias.grad is not None if ln.affine else "No affine")

    # Test in eval mode
    ln.eval()
    x = torch.randn(32, 10, 128, requires_grad=True)
    y = ln(x)
    y.mean().backward()
    print("\nIn eval mode:")
    print("Output shape:", y.shape)
    print("Input gradient:", x.grad is not None)
    print("Weight gradient:", ln.weight.grad is not None if ln.affine else "No affine")
    print("Bias gradient:", ln.bias.grad is not None if ln.affine else "No affine")

    # Compare with PyTorch's LayerNorm
    ref = nn.LayerNorm(normalized_shape=[10, 128])
    custom = MyLayerNorm(normalized_shape=[10, 128])
    custom.load_state_dict(ref.state_dict(), strict=False)

    x = torch.randn(32, 10, 128, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)

    out_custom = custom(x)
    out_ref = ref(x_ref)

    loss_custom = out_custom.mean()
    loss_ref = out_ref.mean()

    loss_custom.backward()
    loss_ref.backward()

    print("\nComparison with nn.LayerNorm:")
    print("Output close:", torch.allclose(out_custom, out_ref, atol=1e-5))
    print("Input grad close:", torch.allclose(x.grad, x_ref.grad, atol=1e-5))
    if custom.affine:
        print("Weight grad close:", torch.allclose(custom.weight.grad, ref.weight.grad, atol=1e-5))
        print("Bias grad close:", torch.allclose(custom.bias.grad, ref.bias.grad, atol=1e-5))

