import torch
import torch.nn as nn

class MyBatchNorm(nn.Module):
    def __init__(self, num_features,
                 eps:float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.tracking_running_statcks = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.tracking_running_statcks:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

    ##this is the correct code
    def forward(self, x):
        # 计算 mean/var 维度，跳过 channel 维
        dims = (0,) + tuple(range(2, x.dim()))

        if self.training:
            # 训练模式：用当前 batch 的统计量
            batch_mean = x.mean(dim=dims, keepdim=True)
            batch_var = x.var(dim=dims, unbiased=False, keepdim=True)

            if self.tracking_running_statcks:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean \
                                        + self.momentum * batch_mean.squeeze()
                    self.running_var = (1 - self.momentum) * self.running_var \
                                       + self.momentum * batch_var.squeeze()
                    self.num_batches_tracked += 1

            mean = batch_mean
            var = batch_var

        else:
            # 推理模式：用 running_mean / running_var（固定的）
            mean = self.running_mean.view(1, -1, *([1] * (x.dim() - 2)))
            var = self.running_var.view(1, -1, *([1] * (x.dim() - 2)))

        # 归一化
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        # 仿射变换
        if self.affine:
            w = self.weight.view(1, -1, *([1] * (x.dim() - 2)))
            b = self.bias.view(1, -1, *([1] * (x.dim() - 2)))
            x_hat = w * x_hat + b

        return x_hat

    ######this is a wrong code
    ####def forward(self, x:torch.Tensor) -> torch.Tensor:
    ####    # x shape: (N, C, *), we mormalize across N and spatial dims
    ####    if x.dim() < 2:
    ####        raise ValueError('Input must have at least 2 dims (N, C, ...)')
    ####
    ####    #??? wrong
    ####    #if self.training or not self.tracking_running_statcks:
    ####    if self.training or self.tracking_running_statcks:
    ####        # leave channel dim(1) out
    ####        dims = (0,) + tuple(range(2, x.dim()))
    ####        batch_mean = x.mean(dim=dims, keepdim=True)
    ####        batch_var = x.var(dim=dims, unbiased=False, keepdim=True)
    ####
    ####        if self.tracking_running_statcks:
    ####            with torch.no_grad():
    ####                self.running_mean = (1 - self.momentum) * self.running_mean \
    ####                                    + self.momentum * batch_mean.squeeze()
    ####                self.running_var = (1 - self.momentum) * self.running_var \
    ####                                    + self.momentum * batch_var.squeeze()
    ####                self.num_batches_tracked += 1
    ####
    ####            mean = batch_mean
    ####            var = batch_var
    ####        else:
    ####            #in eval() use running estimates
    ####            mean = self.running_mean.view(1, -1, *([1] * (x.dim() - 2)))
    ####            var = self.running_var.view(1, -1, *([1] * (x.dim() - 2)))
    ####
    ####        x_hat = (x-mean) / torch.sqrt(var + self.eps)
    ####
    ####        if self.affine:
    ####            w = self.weight.view(1, -1, *([1] * (x.dim() - 2)))
    ####            b = self.bias.view(1, -1, *([1] * (x.dim() - 2)))
    ####            x_hat = w * x_hat +b
    ####        return x_hat


bn = MyBatchNorm(num_features=64)
x = torch.randn(16, 64, 32, 32)
out = bn(x)
#print("out = ", out)

bn.eval()
out_eval = bn(x)
#print("out_eval = ", out_eval)

ref = nn.BatchNorm2d(64)
custom = MyBatchNorm(64)
custom.load_state_dict(ref.state_dict(), strict=False)
#print("custom.load_state_dict(ref.state_dict(), strict=False) = ",
# custom.load_state_dict(ref.state_dict(), strict=False))

print("torch.isclose(out, out_eval, 1e-5, 1e-5) = ", torch.isclose(out, out_eval, 1e-5, 1e-5))
print("(out != out_eval).sum() = ", (out != out_eval).sum())
print("(out - out_eval).abs().max() = ", (out - out_eval).abs().max())


#########################################################################
# flow
#########################################################################
####1. training
####[Input x]
####    │
####┌───┴────────────┐
####│ training = True│
####│(model.train()) │
####└──────┬─────────┘
####       │
####dims = (0,) + range(2, x.dim())
####       │
####batch_mean = mean(x, dims)
####batch_var = var(x, dims)
####       │
####Update
####running_mean / running_var
####       │
####mean ← batch_mean
####var  ← batch_var
####       │
####x_hat = (x - mean) / sqrt(var + eps)
####       │
####Affine
####transform( if affine)
####       │
####[Output]
####
####
####2. eval
####[Input x]
####    │
####┌───┴────────────┐
####│ training = False │
####│ (model.eval()) │
####└──────┬─────────┘
####       │
####mean ← running_mean(固定值)
####var  ← running_var(固定值)
####       │
####x_hat = (x - mean) / sqrt(var + eps)
####       │
####Affine
####transform( if affine)
####       │
####[Output]

####修正后的行为对比
####模式	        均值来源	        方差来源	        是否更新 running stats	    输出依赖输入分布
####训练	        batch_mean	    batch_var	        ✅ 是	                    ✅ 是
####推理 (eval)	running_mean	running_var	        ❌ 否	                    ❌ 否

#########################################################################
# affine
#########################################################################
####这是干啥用？
####            if self.affine:
####                w = self.weight.view(1, -1, *([1] * (x.dim() - 2)))
####                b = self.bias.view(1, -1, *([1] * (x.dim() - 2)))
####                x_hat = w * x_hat +b
####
####这段代码是 BatchNorm 的仿射变换（Affine transform） 部分，也就是让归一化后的数据能通过**可学习的缩放（scale）和平移（shift）**来恢复或调整分布。
####作用
####BatchNorm 的核心有两步：
####1. 归一化
####    ^
####    x   =  (x-u) / sqrt(sigma^2 + epsilon)
####这一步把每个通道的数据变成均值 0、方差 1 的分布。
####
####2. 仿射变换（Affine transform）
####                ^
####    y = gamma * x  +  beta
####γ (gamma, not y)（对应 self.weight）是可学习的缩放参数
####β（对应 self.bias）是可学习的偏移参数
####这样做的原因是：虽然归一化能稳定训练，但完全归一化可能限制模型表达能力，所以再加上可学习的线性变换，让模型自己决定是否需要恢复原分布或调整到新的分布。
####
####3. 为什么 .view(1, -1, *([1] * (x.dim() - 2)))？
####假设输入 x 形状为 (N, C, H, W)：
####    A)    1, -1, 1, 1 代表：
####                1：batch 维，不参与缩放
####                -1：通道维，对应 C 个参数
####                1, 1：空间维度，保持广播机制
####    B)    这样 w 和 b 会变成 (1, C, 1, 1)，广播到整个 batch 的每个像素。
####
####4. 最终效果
####x_hat = (x - mean) / sqrt(var + eps)   # 标准化
####y = w * x_hat + b                      # 缩放 + 平移
####
####如果 self.affine=False，这步会跳过，输出就是完全归一化的结果。
####如果 self.affine=True（默认），模型可以学到
