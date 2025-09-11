import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn


class MyBatchNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, training, momentum, eps):
        if not training:
            # 推理模式，直接使用 running stats
            mean_view = running_mean.view(1, -1, *([1] * (x.dim() - 2)))
            var_view = running_var.view(1, -1, *([1] * (x.dim() - 2)))
            x_hat = (x - mean_view) / torch.sqrt(var_view + eps)
            if weight is not None:
                x_hat = weight.view(1, -1, *([1] * (x.dim() - 2))) * x_hat + \
                        bias.view(1, -1, *([1] * (x.dim() - 2)))
            return x_hat

        # 训练模式，计算当前 batch 的统计量
        dims = (0,) + tuple(range(2, x.dim()))
        num_elements = x.numel() // x.size(1)

        batch_mean = x.mean(dim=dims, keepdim=True)
        batch_var = x.var(dim=dims, unbiased=False, keepdim=True)

        # 保存必要的张量供 backward 使用
        ctx.save_for_backward(x, batch_mean, batch_var, weight)
        ctx.num_elements = num_elements
        ctx.eps = eps
        ctx.affine = weight is not None

        # 归一化
        x_hat = (x - batch_mean) / torch.sqrt(batch_var + eps)

        # 仿射变换
        if weight is not None:
            output = weight.view(1, -1, *([1] * (x.dim() - 2))) * x_hat + \
                     bias.view(1, -1, *([1] * (x.dim() - 2)))
        else:
            output = x_hat

        # 更新 running stats
        with torch.no_grad():
            if running_mean is not None:
                running_mean.mul_(1 - momentum).add_(batch_mean.squeeze(), alpha=momentum)
            if running_var is not None:
                running_var.mul_(1 - momentum).add_(batch_var.squeeze(), alpha=momentum)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, batch_mean, batch_var, weight = ctx.saved_tensors
        num_elements = ctx.num_elements
        eps = ctx.eps
        affine = ctx.affine

        # 根据链式法则重新计算所有梯度

        # 1. 对 x_hat 的梯度
        grad_x_hat = grad_output.clone()
        if affine:
            grad_x_hat = grad_x_hat * weight.view(1, -1, *([1] * (x.dim() - 2)))

        # 2. 对 batch_var 的梯度
        grad_var = torch.sum(grad_x_hat * (x - batch_mean),
                             dim=(0,) + tuple(range(2, x.dim())), keepdim=True) * \
                   -0.5 * torch.pow(batch_var + eps, -1.5)

        # 3. 对 batch_mean 的梯度
        grad_mean = torch.sum(grad_x_hat * -1.0 / torch.sqrt(batch_var + eps),
                              dim=(0,) + tuple(range(2, x.dim())), keepdim=True)

        # 4. 对输入的梯度 grad_x
        grad_x = grad_x_hat / torch.sqrt(batch_var + eps) + \
                 grad_var * 2 * (x - batch_mean) / num_elements + \
                 grad_mean / num_elements

        # 5. 对 weight 和 bias 的梯度
        grad_weight = None
        grad_bias = None
        if affine:
            # 在这里重新计算 x_hat，以确保数值准确
            x_hat = (x - batch_mean) / torch.sqrt(batch_var + eps)
            grad_weight = (grad_output * x_hat).sum(dim=(0,) + tuple(range(2, x.dim())))
            grad_bias = grad_output.sum(dim=(0,) + tuple(range(2, x.dim())))

        return grad_x, grad_weight, grad_bias, None, None, None, None, None


####class MyBatchNormFunction(torch.autograd.Function):
####    @staticmethod
####    def forward(ctx, x, weight, bias, running_mean, running_var, training, momentum, eps):
####        if not training:
####            # 推理模式，直接使用 running stats
####            x_hat = (x - running_mean.view(1, -1, *([1] * (x.dim() - 2)))) / \
####                    torch.sqrt(running_var.view(1, -1, *([1] * (x.dim() - 2))) + eps)
####            if weight is not None:
####                x_hat = weight.view(1, -1, *([1] * (x.dim() - 2))) * x_hat + \
####                        bias.view(1, -1, *([1] * (x.dim() - 2)))
####            return x_hat
####
####        # 训练模式，计算当前 batch 的统计量
####        dims = (0,) + tuple(range(2, x.dim()))
####        num_elements = x.numel() // x.size(1)
####
####        batch_mean = x.mean(dim=dims, keepdim=True)
####        batch_var = x.var(dim=dims, unbiased=False, keepdim=True)
####
####        # 保存中间变量，供 backward 使用
####        ctx.save_for_backward(x, batch_mean, batch_var, weight, bias)
####        ctx.num_elements = num_elements
####        ctx.eps = eps
####
####        # 归一化
####        x_hat = (x - batch_mean) / torch.sqrt(batch_var + eps)
####
####        # 仿射变换
####        if weight is not None:
####            output = weight.view(1, -1, *([1] * (x.dim() - 2))) * x_hat + \
####                     bias.view(1, -1, *([1] * (x.dim() - 2)))
####        else:
####            output = x_hat
####
####        # 更新 running stats
####        with torch.no_grad():
####            running_mean.mul_(1 - momentum).add_(batch_mean.squeeze(), alpha=momentum)
####            running_var.mul_(1 - momentum).add_(batch_var.squeeze(), alpha=momentum)
####
####        return output
####
####    @staticmethod
####    def backward(ctx, grad_output):
####        x, batch_mean, batch_var, weight, bias = ctx.saved_tensors
####        num_elements = ctx.num_elements
####        eps = ctx.eps
####
####        # 1. 重新计算 x_hat，以确保其与 forward 中的完全一致
####        inv_sqrt_var = 1.0 / torch.sqrt(batch_var + eps)
####        x_hat = (x - batch_mean) * inv_sqrt_var
####
####        # 2. 仿射变换的反向传播
####        grad_x_hat = grad_output.clone()
####        if weight is not None:
####            # 计算对 weight 和 bias 的梯度
####            grad_weight = (grad_x_hat * x_hat).sum(dim=(0,) + tuple(range(2, x.dim())))
####            grad_bias = grad_x_hat.sum(dim=(0,) + tuple(range(2, x.dim())))
####            # 乘以 weight 的梯度
####            grad_x_hat = grad_x_hat * weight.view(1, -1, *([1] * (x.dim() - 2)))
####        else:
####            grad_weight = None
####            grad_bias = None
####
####        # 3. 标准化层的反向传播
####        # 计算对方差的梯度
####        grad_var = torch.sum(grad_x_hat * (x - batch_mean), dim=(0,) + tuple(range(2, x.dim())), keepdim=True) * \
####                   -0.5 * torch.pow(batch_var + eps, -1.5)
####
####        # 计算对均值的梯度
####        grad_mean = torch.sum(grad_x_hat * -inv_sqrt_var, dim=(0,) + tuple(range(2, x.dim())), keepdim=True)
####
####        # 4. 计算对输入的梯度
####        grad_x = grad_x_hat * inv_sqrt_var + \
####                 grad_var * 2 * (x - batch_mean) / num_elements + \
####                 grad_mean / num_elements
####
####        return grad_x, grad_weight, grad_bias, None, None, None, None, None

####    @staticmethod
####    def backward(ctx, grad_output):
####        x, batch_mean, batch_var, weight, bias = ctx.saved_tensors
####        num_elements = ctx.num_elements
####        eps = ctx.eps
####
####        # 仿射变换的反向传播
####        if weight is not None:
####            # 计算对 weight 和 bias 的梯度
####            x_hat = (x - batch_mean) / torch.sqrt(batch_var + eps)
####            grad_weight = (grad_output * x_hat).sum(dim=(0,) + tuple(range(2, x.dim())))
####            grad_bias = grad_output.sum(dim=(0,) + tuple(range(2, x.dim())))
####            grad_x_hat = grad_output * weight.view(1, -1, *([1] * (x.dim() - 2)))
####        else:
####            grad_weight = None
####            grad_bias = None
####            grad_x_hat = grad_output
####
####        # 标准化层的反向传播
####        inv_sqrt_var = 1.0 / torch.sqrt(batch_var + eps)
####        grad_var = torch.sum(grad_x_hat * (x - batch_mean) * -0.5 * torch.pow(batch_var + eps, -1.5),
####                             dim=(0,) + tuple(range(2, x.dim())), keepdim=True)
####        grad_mean = torch.sum(grad_x_hat * -inv_sqrt_var, dim=(0,) + tuple(range(2, x.dim())), keepdim=True)
####
####        grad_x = grad_x_hat * inv_sqrt_var + \
####                 grad_var * 2 * (x - batch_mean) / num_elements + \
####                 grad_mean / num_elements
####
####        return grad_x, grad_weight, grad_bias, None, None, None, None, None


class MyBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
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
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

    def forward(self, x):
        return MyBatchNormFunction.apply(x, self.weight, self.bias, self.running_mean,
                                         self.running_var, self.training, self.momentum, self.eps)


import torch
import torch.nn as nn
#from MyBatchNorm import MyBatchNorm  # 假设你将上述代码保存在 MyBatchNorm.py 文件中


def test_batchnorm():
    # 1. 初始化
    print("--- 1. 初始化测试 ---")
    batch_size = 16
    channels = 64
    height = 32
    width = 32
    input_data = torch.randn(batch_size, channels, height, width, requires_grad=True)

    # 创建自定义的 BatchNorm 模块
    custom_bn = MyBatchNorm(num_features=channels)

    # 创建 PyTorch 内置的 BatchNorm 模块作为参考
    ref_bn = nn.BatchNorm2d(num_features=channels)

    # 将内置模块的参数加载到自定义模块中，以确保参数一致
    custom_bn.load_state_dict(ref_bn.state_dict(), strict=False)

    print("初始化完成。")

    # 2. 训练模式测试
    print("\n--- 2. 训练模式前向/后向传播测试 ---")

    # 设置为训练模式
    custom_bn.train()
    ref_bn.train()

    # 前向传播
    custom_output = custom_bn(input_data)
    ref_output = ref_bn(input_data)

    # 验证输出是否接近
    assert torch.allclose(custom_output, ref_output, rtol=1e-4, atol=1e-6), "训练模式下前向传播输出不匹配"
    print("训练模式下前向传播输出匹配！")

    # 反向传播并验证梯度
    loss = custom_output.sum()
    loss.backward()
    custom_grad_x = input_data.grad.clone()
    custom_grad_weight = custom_bn.weight.grad.clone()
    custom_grad_bias = custom_bn.bias.grad.clone()

    input_data.grad.zero_()
    ref_bn.zero_grad()

    ref_output.sum().backward()
    ref_grad_x = input_data.grad.clone()
    ref_grad_weight = ref_bn.weight.grad.clone()
    ref_grad_bias = ref_bn.bias.grad.clone()

    assert torch.allclose(custom_grad_x, ref_grad_x, rtol=1e-4, atol=1e-6), "训练模式下输入梯度不匹配"
    assert torch.allclose(custom_grad_weight, ref_grad_weight, rtol=1e-4, atol=1e-6), "训练模式下权重梯度不匹配"
    assert torch.allclose(custom_grad_bias, ref_grad_bias, rtol=1e-4, atol=1e-6), "训练模式下偏置梯度不匹配"
    print("训练模式下反向传播梯度匹配！")

    # 3. 评估模式测试
    print("\n--- 3. 评估模式前向传播测试 ---")

    # 设置为评估模式
    custom_bn.eval()
    ref_bn.eval()

    # 评估模式下的输入应该与训练模式不同，以确保使用的不是当前 batch 的统计量
    eval_input = torch.randn(batch_size, channels, height, width)

    # 前向传播
    custom_eval_output = custom_bn(eval_input)
    ref_eval_output = ref_bn(eval_input)

    # 验证输出是否接近
    assert torch.allclose(custom_eval_output, ref_eval_output, rtol=1e-4, atol=1e-6), "评估模式下前向传播输出不匹配"
    print("评估模式下前向传播输出匹配！")

    # 4. 其他参数组合测试
    print("\n--- 4. 参数组合测试 (affine=False) ---")

    custom_bn_no_affine = MyBatchNorm(num_features=channels, affine=False)
    ref_bn_no_affine = nn.BatchNorm2d(num_features=channels, affine=False)

    # 训练模式测试
    custom_bn_no_affine.train()
    ref_bn_no_affine.train()
    custom_output_na = custom_bn_no_affine(input_data)
    ref_output_na = ref_bn_no_affine(input_data)

    assert torch.allclose(custom_output_na, ref_output_na, rtol=1e-4, atol=1e-6), "affine=False 训练模式不匹配"
    print("affine=False 训练模式测试通过！")

    print("\n所有测试通过！自定义的 BatchNorm 实现与 PyTorch 内置的 nn.BatchNorm2d 行为一致。")


if __name__ == '__main__':
    test_batchnorm()