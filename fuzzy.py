import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import torch.nn.functional as F


class FuzzyConv(nn.Module):
    """
    Fuzzy Dilated Convolution Layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, sigma=0.1, *args, **kwargs):
        super(FuzzyConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = _pair(dilation)
        self.sigma = sigma
        if self.dilation[0] != self.dilation[1]:  # all pairs are considered to be of form (n, n)
            raise NotImplementedError(f"Unequal dilation not supported. Got dilation rates: {self.dilation}")
        if self.dilation[0] > 1:
            fuzzy_points = np.linspace(self.sigma, 1, int(self.dilation[0] / 2) + 1)
            fuzzy_points /= np.sum(fuzzy_points)  # for normalizing. Otherwise, exploding gradients
            self.fuzzy_points = nn.Parameter(torch.Tensor(fuzzy_points), requires_grad=True)
            self.create_fuzzy_mask()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=self.dilation, *args, **kwargs)

    def create_fuzzy_mask(self):
        fh, fw = self.dilation if self.dilation[0] % 2 else (self.dilation[0] + 1, self.dilation[1] + 1)
        fuzzy_mask = torch.zeros((fh, fw), device=self.fuzzy_points.device)
        for e, v in enumerate(self.fuzzy_points):
            for _ in range(int(self.dilation[0] / 2)):
                fuzzy_mask[e : fh - e, e : fw - e] = v

        self.fuzzy_mask = nn.Parameter(torch.broadcast_to(fuzzy_mask, (self.in_channels, 1, fh, fw)), requires_grad=False)

    def forward(self, x):
        if self.dilation[0] > 1:
            if self.training:
                self.create_fuzzy_mask()
            x = F.conv2d(x, self.fuzzy_mask, padding="same", groups=self.in_channels)
        x = self.conv(x)
        return x


class SpatialAttention(nn.Module):
    """
    Spatial Attention in different rates
    """

    def __init__(self, in_channels, out_size, kernel_size, downsample_ratio, dilation, sigma=0.1, upsample_mode="bilinear"):
        super(SpatialAttention, self).__init__()
        self.downsample_ratio = downsample_ratio
        if downsample_ratio == 1:
            self.fuzzy_conv = FuzzyConv(in_channels, in_channels, kernel_size, dilation, bias=True, padding=dilation, stride=downsample_ratio, sigma=sigma)
        else:
            self.fuzzy_conv = FuzzyConv(in_channels, in_channels, kernel_size, dilation, bias=True, padding=dilation, stride=downsample_ratio, sigma=sigma)
            self.upsample = nn.Upsample(size=out_size, mode=upsample_mode)

        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, padding="same")

    def forward(self, x):
        if self.downsample_ratio == 1:
            y = self.fuzzy_conv(x)
        else:
            y = self.fuzzy_conv(x)
            y = self.upsample(y)

        y = self.conv2d(y)
        return torch.add(x, y)


class GlobalLearnablePool(nn.Module):
    """
    Learnable Pooling Layer.
    Takes (n, ci, h, w) input and returns (n, co, 1, 1) output
    """

    def __init__(self, in_channels, in_height, in_width):
        super(GlobalLearnablePool, self).__init__()
        self.in_channels = in_channels
        self.pool_mask = nn.Parameter(torch.ones((in_channels, 1, in_height, in_width)) / (in_height * in_width), requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, self.pool_mask, groups=self.in_channels)
        return x


class ChannelAttention(nn.Module):
    """
    Channel Attention module using the learnable pooling layer
    """

    def __init__(self, in_channels, in_height, in_width, se_hidden_channels, dropout_rate=0.1):
        super(ChannelAttention, self).__init__()
        self.pool = GlobalLearnablePool(in_channels, in_height, in_width)
        self.linear1 = nn.Linear(in_channels, se_hidden_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(se_hidden_channels, in_channels)

    def forward(self, x):
        out = self.pool(x)
        out = out.reshape(x.shape[0], -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = out[:, :, None, None]  # reshape - add height and width axes
        out = x * torch.sigmoid(out)
        return out


class ChannelSpatialAttention(nn.Module):
    def __init__(self, in_channels, out_size, kernel_size, downsample_ratio_list, dilation_rates_list, sigma=0.1, upsample_mode="bilinear"):
        super().__init__()
        if isinstance(downsample_ratio_list, int):
            downsample_ratio_list = [downsample_ratio_list]
        if isinstance(dilation_rates_list, int):
            dilation_rates_list = [dilation_rates_list]
        if isinstance(kernel_size, int):
            kernel_size_list = [kernel_size] * len(dilation_rates_list)
        else:
            kernel_size_list = kernel_size
        assert len(kernel_size_list) == len(downsample_ratio_list) == len(dilation_rates_list), "Downsample list and Dilation list have to be equal length"
        fuzzy_convs = []
        for kernel_size, stride, dilation in zip(kernel_size_list, downsample_ratio_list, dilation_rates_list):
            fuzzy_convs.append(
                nn.Sequential(
                    ChannelAttention(in_channels, 32, 32, 512, 0.2),
                    SpatialAttention(in_channels, out_size, kernel_size, stride, dilation, sigma),
                )
            )
        self.fuzzy_convs = nn.ModuleList(fuzzy_convs)
        self.channel_conv = nn.Conv2d(in_channels * len(downsample_ratio_list), in_channels, (1, 1))

    def forward(self, x):
        out = []
        for fuzzy_conv in self.fuzzy_convs:
            out.append(fuzzy_conv(x))

        out = torch.concat(out, dim=1)
        out = self.channel_conv(out)
        return out
