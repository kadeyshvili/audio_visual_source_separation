import torch
from torch import nn
from torch.nn import functional as F


class GlobalLayerNorm(nn.Module):
    def __init__(self, num_channels) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, num_channels)

    def forward(self, x):
        return self.norm(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, num_groups=1, dilation=1, padding=0, 
                 norm_type=GlobalLayerNorm, activation_type=nn.PReLU) -> None:
        super().__init__()
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            groups=num_groups,
        )
        if norm_type is not None:
            norm = norm_type(out_channels)
        else:
            norm = nn.Identity()
        if activation_type is not None:
            activation = activation_type()
        else:
            activation = nn.Identity()
        self.block = nn.Sequential(conv, norm, activation)

    def forward(self, x):
        return self.block(x)
    

