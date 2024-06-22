from torch import nn
import torch.nn.functional as F

from model.attention import MultiheadGQA
from model.weight_init import init_weights


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class GQA_Block(nn.Module):
    def __init__(self, embed_size, forward_expansion, dropout):
        super(GQA_Block, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, 2 * forward_expansion * embed_size),
            SwiGLU(),
            nn.LayerNorm(forward_expansion * embed_size),
            nn.Linear(forward_expansion * embed_size, embed_size),
            nn.Dropout(dropout),
        )
        self.feed_forward.apply(init_weights)


class SelfGQA_Block(GQA_Block):
    def __init__(self, embed_size, heads, groups, forward_expansion, dropout, gamma):
        super(SelfGQA_Block, self).__init__(embed_size, forward_expansion, dropout)
        self.GQA = MultiheadGQA(
            embed_size, heads, heads // groups, dropout=dropout, gamma_init=gamma
        )

    def forward(self, x):
        # shape of x: (batch_size, t, h, w, embed_size)
        x = self.norm1(x)
        x = self.GQA(x, x, x) + x
        x = self.feed_forward(x) + x
        return x


class CrossGQA_Block(GQA_Block):
    def __init__(
        self,
        embed_size,
        heads,
        groups,
        forward_expansion,
        dropout,
        gamma,
    ):
        super(CrossGQA_Block, self).__init__(embed_size, forward_expansion, dropout)
        self.norm2 = nn.LayerNorm(embed_size)
        self.GQA1 = MultiheadGQA(
            embed_size,
            heads,
            heads // groups,
            dropout=dropout,
            gamma_init=gamma,
        )
        self.GQA2 = MultiheadGQA(
            embed_size,
            heads,
            heads // groups,
            is_self_attn=False,
            dropout=dropout,
            gamma_init=gamma,
        )

    def forward(self, x, y):
        # shape of x: (batch_size, t, h, w, embed_size)
        x = self.norm1(x)
        x = self.GQA1(x, x, x) + x
        x = self.norm2(x)
        x = self.GQA2(x, y, y) + x
        x = self.feed_forward(x) + x
        return x
