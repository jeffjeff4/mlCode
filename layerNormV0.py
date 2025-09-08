import torch
import torch.nn as nn

class MyLayerNorm(nn.Module):
    def __init__(self, normalizd_shape, eps=1e-5, affine=True):
        super().__init__()
        if isinstance(normalizd_shape, int):
            normalizd_shape = (normalizd_shape,)
        self.normalized_shape = tuple(normalizd_shape)
        self.eps = eps
        self.affine = affine

        if self.affine:
            # gamma, beta 's shape is the same as normalized_shape
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias   = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x could be any shape, will normalize len(normalized_shape) at last
        # for example, normalized_shape = (D,), will normalize the last dimension
        dim = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dim, keepdim=True)
        var = x.var(dim=dim, unbiased=False, keepdim=True)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x_hat = x_hat * self.weight.view(*([1] * (x.dim() - len(self.normalized_shape))),
                                             *self.normalized_shape) \
                          + self.bias.view(*([1] * (x.dim() - len(self.normalized_shape))),
                                             *self.normalized_shape)
        return x_hat

if __name__ == '__main__':
    # assume input is (batch_size, seq_len, feature_dim), then normalize at feature_dim
    #ln = MyLayerNorm(normalizd_shape=128)
    ln = MyLayerNorm(normalizd_shape=[10, 128])
    x = torch.randn(32, 10, 128)
    y = ln(x)
    print(y.shape)