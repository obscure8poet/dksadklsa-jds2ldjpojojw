import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class ECA_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, in_channels, esa_channels):
        super(ESA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, esa_channels, kernel_size=1)
        self.conv_f = nn.Conv2d(esa_channels, esa_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(esa_channels, esa_channels, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(esa_channels, esa_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(esa_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m
