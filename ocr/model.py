import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Residual Block ----
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.05):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += identity
        return self.relu(out)

# ---- HFC Layer ----
class HFCLayer(nn.Module):
    def __init__(self, num_classes, D_b):
        super().__init__()
        self.V = nn.Parameter(torch.randn(num_classes, D_b))
        self.bn = nn.BatchNorm1d(num_classes * D_b)

    def forward(self, x):
        U_b = x.sum(dim=1)
        T_b = U_b.unsqueeze(1) * self.V.unsqueeze(0)
        T_b = self.bn(T_b.view(T_b.size(0), -1))
        T_b = F.relu(T_b.view(T_b.size(0), -1, self.V.size(1)))
        return T_b.sum(dim=2)

# ---- Merging ----
class MergingLayer(nn.Module):
    def __init__(self, num_branches=3):
        super().__init__()
        self.w = nn.Parameter(torch.ones(num_branches) / num_branches)

    def forward(self, inputs):
        weights = F.softmax(self.w, dim=0)
        return sum(w * x for w, x in zip(weights, inputs))

# ---- Full Model ----
class EnhancedBMCNNwHFCs(nn.Module):
    def __init__(self, num_classes=58, dropout_rate=0.05):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            ResidualBlock(1, 128, dropout_rate=dropout_rate),
            ResidualBlock(128, 128, dropout_rate=dropout_rate),
            ResidualBlock(128, 128, dropout_rate=dropout_rate),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv_block2 = nn.Sequential(
            ResidualBlock(128, 256, dropout_rate=dropout_rate),
            ResidualBlock(256, 256, dropout_rate=dropout_rate),
            ResidualBlock(256, 256, dropout_rate=dropout_rate),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv_block3 = nn.Sequential(
            ResidualBlock(256, 512, dropout_rate=dropout_rate),
            ResidualBlock(512, 512, dropout_rate=dropout_rate),
            ResidualBlock(512, 512, dropout_rate=dropout_rate),
        )

        self.hfc1 = HFCLayer(num_classes=58, D_b=1024)
        self.hfc2 = HFCLayer(num_classes=58, D_b=256)
        self.hfc3 = HFCLayer(num_classes=58, D_b=64)

        self.merging = MergingLayer(3)

    def forward(self, x):
        x1 = self.conv_block1(x)
        x = self.pool1(x1)
        x2 = self.conv_block2(x)
        x = self.pool2(x2)
        x3 = self.conv_block3(x)
        l1 = self.hfc1(x1.view(x1.size(0), x1.size(1), -1))
        l2 = self.hfc2(x2.view(x2.size(0), x2.size(1), -1))
        l3 = self.hfc3(x3.view(x3.size(0), x3.size(1), -1))

        return self.merging((l1, l2, l3))