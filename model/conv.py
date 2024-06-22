import torch
from torch import nn
from torch.nn import functional as F


def downsample(x, f1, f2):
    idx1 = 64 - f1
    idx2 = 64 + f1
    sliced = x[..., idx1:idx2, idx1:idx2]
    return F.avg_pool3d(sliced, (1, f2, f2))


# This is a minimal proof of concept taken from a different branch
# Depending on your final architecture, you may chose to either:
# - to apply a positional/modality embedding along axes 3, 4 and 5
# (5 being the last one) and feed it to a Perceiver (or other
# attention based model)
# - To merge dim 5 with dim 1 (channels) and feed it through a group 
# conv3d network first
def multi_resolution_downsample(x):
    """Downsample the input tensor at multiple resolutions.
    Args:
        x: input tensor of shape (B, C, T, 128, 128)
    Returns:
        x: downsampled tensor of shape (B, C, T, 16, 16, 4)
    """
    xs = []
    for f1, f2 in [(64, 8), (32, 4), (16, 2), (8, 1)]:
        xi = downsample(x, f1, f2)
        xs.append(xi)
    # somehow torch.compile manages to handle this abomination
    x = torch.stack(xs, dim=-1)
    return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int | tuple[int, int, int] = 1,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self._reset_parameters()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.silu(out)
        return out

    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shrink=True):
        super(ResidualBlock, self).__init__()
        mid_size = in_channels // 2
        self.shrink = shrink
        self.conv1 = ConvBlock(in_channels, mid_size)
        self.conv2 = ConvBlock(
            mid_size,
            mid_size,
            kernel_size=(2, 3, 3) if shrink else 2,
            stride=(1, 2, 2) if shrink else 1,
        )
        self.conv3 = ConvBlock(mid_size, out_channels)
        self.conv4 = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        out = self.conv1(x)
        padding = (1, 1, 1, 1, 1, 0) if self.shrink else (0, 0, 0, 0, 1, 0)
        out = F.pad(out, padding)
        out = self.conv2(out)
        out = self.conv3(out)
        x = self.conv4(x)
        out += (
            F.avg_pool3d(x, (1, 2, 2))
            if self.shrink
            else F.adaptive_avg_pool3d(x, (None, 3, 3))
        )
        return out


class ConvNet(nn.Module):
    def __init__(self, embed_size=512):
        super(ConvNet, self).__init__()
        self.conv1 = ConvBlock(
            1, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(24, 48),
            ResidualBlock(48, 96),
            ResidualBlock(96, 192),
            ResidualBlock(192, 384),
            ResidualBlock(384, embed_size, False),
        )
        self.pos_h = nn.Parameter(torch.randn(1, embed_size, 1, 3, 1))
        self.pos_w = nn.Parameter(torch.randn(1, embed_size, 1, 1, 3))
        self._reset_parameters()

    def forward(self, x):
        out = self.conv1(x)
        out = self.res_blocks(out)
        return out + self.pos_h + self.pos_w

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.pos_h, std=0.02)
        nn.init.trunc_normal_(self.pos_w, std=0.02)
