import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Conv Layer 1: input [B, 3, 32, 32] → output [B, 16, 30, 30]
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=0)

        # Conv Layer 2: output [B, 32, 14, 14]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)

        # Pooling: 2x2 max pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer: input [B, 32*6*6] → output [B, num_classes]
        self.fc = nn.Linear(32 * 6 * 6, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [B, 16, 30, 30]
        x = self.pool(x)  # [B, 16, 15, 15]

        x = F.relu(self.conv2(x))  # [B, 32, 13, 13]
        x = self.pool(x)  # [B, 32, 6, 6]

        x = x.view(x.size(0), -1)  # Flatten: [B, 32*6*6]
        x = self.fc(x)  # [B, num_classes]
        return x


model = SimpleCNN(in_channels=3, num_classes=10)

# Dummy input: batch of 8 images, each 3x32x32
x = torch.randn(8, 3, 32, 32)

# Forward pass
output = model(x)

print("Output shape:", output.shape)  # [8, 10]

import torch
import torch.nn as nn
import torch.nn.functional as F


# Attention 模块：Squeeze-and-Excitation
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, h, w = x.size()
        y = x.view(b, c, -1).mean(dim=2)  # Squeeze: [B, C]
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))  # [B, C]
        y = y.view(b, c, 1, 1)  # [B, C, 1, 1]
        return x * y  # Excitation: scale input


# 残差模块（含 Conv2d + BatchNorm + SE 注意力）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels)

        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class CustomResAttentionCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, downsample=True)
        self.layer3 = ResidualBlock(128, 256, downsample=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)  # [B, 64, H, W]
        x = self.layer1(x)  # [B, 64, H, W]
        x = self.layer2(x)  # [B, 128, H/2, W/2]
        x = self.layer3(x)  # [B, 256, H/4, W/4]

        x = self.pool(x)  # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 256]
        x = self.fc(x)  # [B, num_classes]
        return x


model = CustomResAttentionCNN(in_channels=3, num_classes=10)
x = torch.randn(8, 3, 32, 32)  # 8张图片
output = model(x)

print(output.shape)  # [8, 10]
