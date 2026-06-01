import torch
import torch.nn as nn
import torch.nn.functional as F

class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(MyConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Kernel size and stride
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Learnable weights: [out_channels, in_channels, kH, kW]
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        # x shape: [B, C_in, H, W]
        B, C, H, W = x.shape

        # Pad input
        x_padded = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))

        # Use unfold to extract sliding local blocks
        x_unfold = F.unfold(x_padded, kernel_size=self.kernel_size, stride=self.stride)  # [B, C*kH*kW, L]
        weight_flat = self.weight.view(self.out_channels, -1)                             # [C_out, C_in*kH*kW]

        # Perform matrix multiplication + bias
        out_unfold = weight_flat @ x_unfold + self.bias.unsqueeze(1)                     # [B, C_out, L]
        out = out_unfold.view(B, self.out_channels,
                              (H + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1,
                              (W + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1)
        return out


# Example input: batch of 4 RGB images of size 8x8
x = torch.randn(4, 3, 8, 8)  # [B, C, H, W]

# Our custom conv2d layer: from 3 input channels to 6 output channels
conv = MyConv2D(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)

# Apply it
y = conv(x)

print("Input shape: ", x.shape)   # [4, 3, 8, 8]
print("Output shape:", y.shape)  # [4, 6, 8, 8] — same due to padding=1


