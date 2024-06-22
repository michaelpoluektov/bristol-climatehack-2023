import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model.conv import ConvNet
from model.weight_init import init_weights
from model.blocks import SelfGQA_Block, CrossGQA_Block


class ConstModel(nn.Module):
    def __init__(self, embed_size=512):
        super(ConstModel, self).__init__()
        self.linear1 = nn.Linear(6, 128)
        self.linear2 = nn.Linear(128, embed_size)
        self.pos_enc = nn.Parameter(torch.zeros(1, 1, embed_size))
        self._reset_parameters()

    def forward(self, x):
        x = F.silu(self.linear1(x))
        x = F.silu(self.linear2(x))
        return x + self.pos_enc

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.pos_enc, std=0.02)
        self.linear1.apply(init_weights)
        self.linear2.apply(init_weights)


class TimeModel(nn.Module):
    def __init__(self, embed_size=512):
        super(TimeModel, self).__init__()
        self.linear1 = nn.Linear(8, 128)
        self.linear2 = nn.Linear(128, embed_size)
        self._reset_parameters()

    def forward(self, x):
        x = F.silu(self.linear1(x))
        x = F.silu(self.linear2(x))
        return x

    def _reset_parameters(self):
        self.linear1.apply(init_weights)
        self.linear2.apply(init_weights)


class PowerModel(nn.Module):
    def __init__(self, embed_size=512):
        super(PowerModel, self).__init__()
        self.linear = nn.Linear(1, embed_size)
        self._reset_parameters()

    def forward(self, x):
        return self.linear(x)

    def _reset_parameters(self):
        self.linear.apply(init_weights)


class FancyActivation(nn.Module):
    """Fancy activation function for transformer output, where the output
    has the following properties:
    - The output is in the range of (0, 1)
    - Each scalar in the output can be approximated reasonably well by a
    weighed average of some input.
    """
    def __init__(self, dim_in, dim_out, eps=0.005):
        super(FancyActivation, self).__init__()
        self.eps = eps
        self.skip_parameter = nn.Parameter(torch.zeros(dim_in, dim_out))

    def forward(self, x, y):
        """Args:
        x: torch.Tensor, (batch_size, dim_in) where dim_in = 48, representing the last 4 hours of PV generation.
        y: torch.Tensor, (batch_size, dim_out) where dim_out = 12, the output of the transformer model.
        """
        # x is in the range of (0, 1)
        # skip_parameter is a matrix representing the effect of each input on the output
        # can be computed using least squares, but we chose to make it trainable because
        # that's simpler
        x = x @ F.softmax(self.skip_parameter, dim=0)
        x = x.clamp(self.eps, 1 - self.eps)
        n = -x / (x - 1)
        return n / (n + torch.exp(-y))


class Transformer(nn.Module):
    def __init__(
        self,
        embed_size=512,
        num_heads=32,
        num_groups=4,
        num_layers_enc=1,
        num_layers_dec=1,
        forward_expansion=4,
        dropout=0.2,
    ):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        enc_gamma = (
            np.log(num_layers_dec * 3.0) * np.log(num_layers_enc * 2.0) / 3.0
        ) ** 0.5
        dec_gamma = np.log(num_layers_dec * 3.0) ** 0.5
        params = (embed_size, num_heads, num_groups, forward_expansion, dropout)
        self.encoder_layers = nn.ModuleList(
            [SelfGQA_Block(*params, enc_gamma) for _ in range(num_layers_enc)]
        )
        self.decoder_layers = nn.ModuleList(
            [CrossGQA_Block(*params, dec_gamma) for _ in range(num_layers_dec)]
        )
        self.hrv_model = ConvNet(embed_size)
        self.time_model = TimeModel(embed_size)
        self.time_enc_emb = nn.Parameter(torch.randn(1, 1, embed_size))
        self.cst_model = ConstModel(embed_size)
        self.power_model = PowerModel(embed_size)
        self.enc_pos = nn.Parameter(torch.randn(1, 12, embed_size))
        self.dec_pos = nn.Parameter(torch.randn(1, 48, embed_size))
        self.linear1 = nn.Linear(embed_size, 128)
        self.linear2 = nn.Linear(128, 1, bias=False)
        self.fancy_activation = FancyActivation(12, 48)
        self._reset_parameters()

    def _reset_parameters(self):
        # trunc normal for positional embeddings
        nn.init.trunc_normal_(self.time_enc_emb, std=0.02)
        nn.init.trunc_normal_(self.enc_pos, std=0.02)
        nn.init.trunc_normal_(self.dec_pos, std=0.02)
        self.linear1.apply(init_weights)
        self.linear2.apply(init_weights)

    def forward(self, hrv, power, time_ftrs, cst_ftrs):
        # shape of hrv: (batch_size, 1, 12, 128, 128)
        # shape of power: (batch_size, 12, 1)
        # shape of time_ftrs: (batch_size, 60, 8)
        # shape of cst_ftrs: (batch_size, 1, 6)
        hrv = self.hrv_model(hrv)
        hrv = hrv + self.enc_pos.permute(0, 2, 1)[..., None, None]
        hrv = hrv.flatten(2).permute(0, 2, 1)
        new_pow = self.power_model(power) + self.enc_pos
        time_ftrs = self.time_model(time_ftrs)
        enc_time_ftrs = time_ftrs[:, :12] + self.time_enc_emb + self.enc_pos

        cst_ftrs = self.cst_model(cst_ftrs)
        enc = torch.cat([hrv, new_pow, enc_time_ftrs, cst_ftrs], dim=1)
        for l in self.encoder_layers:
            enc = l(enc)
        out = time_ftrs[:, 12:] + self.dec_pos
        for l in self.decoder_layers:
            out = l(out, enc)

        out = F.silu(self.linear1(out))
        out = self.linear2(out)
        y = power.squeeze(-1)
        return self.fancy_activation(y, out)


if __name__ == "__main__":
    model = Transformer()
    print(model)
    hrv = torch.randn(1, 1, 12, 128, 128)
    power = torch.randn(1, 12, 1)
    time_ftrs = torch.randn(1, 60, 8)
    cst_ftrs = torch.randn(1, 1, 6)
    out = model(hrv, power, time_ftrs, cst_ftrs)
