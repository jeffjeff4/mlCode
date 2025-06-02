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
            self.register_parameter('running_mean', torch.zeros(num_features))
            self.register_parameter('running_var', torch.ones(num_features))
            self.register_parameter('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x shape: (N, C, *), we mormalize across N and spatial dims
        if x.dim() < 2:
            raise ValueError('Input must have at least 2 dims (N, C, ...)')

        #??? wrong
        #if self.training or not self.tracking_running_statcks:
        if self.training or self.tracking_running_statcks:
            # leave channel dim(1) out
            dims = (0,) + tuple(range(2, x.dim()))
            batch_mean = x.mean(dim=dims, keepdim=True)
            batch_var = x.var(dim=dims, unbiased=False, keepdim=True)

            if self.tracking_running_statcks:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean \
                                        + self.momentum * batch_mean.squueze()
                    self.running_var = (1 - self.momentum) * self.running_var \
                                        + self.momentum * batch_var.squueze()
                    self.num_batches_tracked += 1

                mean = batch_mean
                var = batch_var
            else:
                #in eval() use running estimates
                mean = self.running_mean.view(1, -1, *([1] * (x.dim() - 2)))
                var = self.running_var.view(1, -1, *([1] * (x.dim() - 2)))

            x_hat = (x-mean) / torch.sqrt(var + self.eps)

            if self.affine:
                w = self.weight.view(1, -1, *([1] * (x.dim() - 2)))
                b = self.bias.view(1, -1, *([1] * (x.dim() - 2)))
                x_hat = w * x_hat +b
            return x_hat


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



