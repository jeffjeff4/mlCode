import torch
import torch.nn as nn

class MyBatchNorm(nn.Module):
    def __init__(self, num_features,
                 eps:float = 1e-5,
                 momemtum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        super().__init__()
        self.eps = eps
