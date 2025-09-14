import torch
import torch.nn as nn

class MyBatchNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, eps, momentum, training, track_running_stats):
        # Compute dimensions for mean and variance (excluding channel dim)
        dims = (0,) + tuple(range(2, x.dim()))

        if training:
            # Training mode: compute batch statistics
            batch_mean = x.mean(dim=dims, keepdim=True)
            batch_var = x.var(dim=dims, unbiased=False, keepdim=True)

            if track_running_stats:
                with torch.no_grad():
                    running_mean.mul_(1 - momentum).add_(momentum * batch_mean.squeeze())
                    running_var.mul_(1 - momentum).add_(momentum * batch_var.squeeze())

            mean = batch_mean
            var = batch_var
        else:
            # Evaluation mode: use running statistics
            mean = running_mean.view(1, -1, *([1] * (x.dim() - 2)))
            var = running_var.view(1, -1, *([1] * (x.dim() - 2)))

        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + eps)

        # Affine transformation
        if weight is not None and bias is not None:
            w = weight.view(1, -1, *([1] * (x.dim() - 2)))
            b = bias.view(1, -1, *([1] * (x.dim() - 2)))
            output = w * x_hat + b
        else:
            output = x_hat

        # Save tensors for backward
        ctx.save_for_backward(x, weight, bias, mean, var, x_hat)
        ctx.eps = eps
        ctx.dims = dims
        ctx.training = training
        ctx.track_running_stats = track_running_stats
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, mean, var, x_hat = ctx.saved_tensors
        eps = ctx.eps
        dims = ctx.dims
        training = ctx.training

        # Initialize gradients
        grad_x = None
        grad_weight = None
        grad_bias = None

        if training:
            # Number of elements per channel (e.g., N * H * W)
            m = x.numel() // x.shape[1]

            # Compute gradient w.r.t. x_hat
            if weight is not None and bias is not None:
                w = weight.view(1, -1, *([1] * (x.dim() - 2)))
                grad_x_hat = grad_output * w
            else:
                grad_x_hat = grad_output

            # Compute gradients for x
            inv_std = 1.0 / torch.sqrt(var + eps)
            x_minus_mean = x - mean
            grad_x = inv_std * (
                grad_x_hat
                - grad_x_hat.mean(dim=dims, keepdim=True)
                - x_minus_mean * (grad_x_hat * x_minus_mean).mean(dim=dims, keepdim=True) / var
            )

            # Compute gradients for weight and bias
            if weight is not None and bias is not None:
                grad_weight = (grad_output * x_hat).sum(dim=dims).squeeze()
                grad_bias = grad_output.sum(dim=dims).squeeze()
        else:
            # In evaluation mode, only compute gradient w.r.t. input and parameters
            if weight is not None and bias is not None:
                w = weight.view(1, -1, *([1] * (x.dim() - 2)))
                grad_x_hat = grad_output * w
            else:
                grad_x_hat = grad_output

            grad_x = grad_x_hat / torch.sqrt(var + eps)

            if weight is not None and bias is not None:
                grad_weight = (grad_output * x_hat).sum(dim=dims).squeeze()
                grad_bias = grad_output.sum(dim=dims).squeeze()

        return grad_x, grad_weight, grad_bias, None, None, None, None, None, None

class MyBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x):
        return MyBatchNormFunction.apply(
            x, self.weight, self.bias, self.running_mean, self.running_var,
            self.eps, self.momentum, self.training, self.track_running_stats
        )

# Test the implementation
if __name__ == "__main__":
    bn = MyBatchNorm(num_features=64)
    x = torch.randn(16, 64, 32, 32, requires_grad=True)
    out = bn(x)
    out.mean().backward()  # Simulate a loss function

    print("Input gradient:", x.grad is not None)
    print("Weight gradient:", bn.weight.grad is not None if bn.affine else "No affine")
    print("Bias gradient:", bn.bias.grad is not None if bn.affine else "No affine")

    # Test in eval mode
    bn.eval()
    x = torch.randn(16, 64, 32, 32, requires_grad=True)
    out = bn(x)
    out.mean().backward()

    print("\nIn eval mode:")
    print("Input gradient:", x.grad is not None)
    print("Weight gradient:", bn.weight.grad is not None if bn.affine else "No affine")
    print("Bias gradient:", bn.bias.grad is not None if bn.affine else "No affine")

    # Compare with PyTorch's BatchNorm2d
    ref = nn.BatchNorm2d(64)
    custom = MyBatchNorm(64)
    custom.load_state_dict(ref.state_dict(), strict=False)

    x = torch.randn(16, 64, 32, 32, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)

    out_custom = custom(x)
    out_ref = ref(x_ref)

    loss_custom = out_custom.mean()
    loss_ref = out_ref.mean()

    loss_custom.backward()
    loss_ref.backward()

    print("\nComparison with nn.BatchNorm2d:")
    print("Output close:", torch.allclose(out_custom, out_ref, atol=1e-5))
    print("Input grad close:", torch.allclose(x.grad, x_ref.grad, atol=1e-5))
    if custom.affine:
        print("Weight grad close:", torch.allclose(custom.weight.grad, ref.weight.grad, atol=1e-5))
        print("Bias grad close:", torch.allclose(custom.bias.grad, ref.bias.grad, atol=1e-5))