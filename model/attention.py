# credit to https://github.com/fkodom/grouped-query-attention-pytorch

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor, nn


def scaled_dot_product_gqa(query, key, value, dropout: float = 0.0):
    """Scaled dot product attention with support for grouped queries.

    Einstein notation:
    - b: batch size
    - n / s: sequence length
    - h: number of heads
    - g: number of groups
    - d: dimension of query/key/value

    Args:
        query: Query tensor of shape (b, n, h, d)
        key: Key tensor of shape (b, s, h, d)
        value: Value tensor of shape (b, s, h, d)
        dropout: Dropout probability (default: 0.0)
        scale: Scale factor for query (default: d_query ** 0.5)

    Returns:
        - Attention output with shape (b, n, h, d)
    """
    if not query.ndim == key.ndim == value.ndim == 4:
        print(query.ndim, key.ndim, value.ndim)
        raise ValueError

    # Move sequence length dimension to axis 2.
    # This makes the attention operations below *much* faster.
    query = rearrange(query, "b n h d -> b h n d")
    key = rearrange(key, "b s h d -> b h s d")
    value = rearrange(value, "b s h d -> b h s d")

    bq, hq, _, dq = query.shape
    bk, hk, nk, dk = key.shape
    bv, hv, nv, dv = value.shape
    if not (bq == bk == bv and dq == dk == dv):
        print(bq, bk, bv, dq, dk, dv)
        raise ValueError
    elif (hk != hv) or (nk != nv):
        print(hk, hv, nk, nv)
        raise ValueError
    elif hq % hk != 0:
        print(hq, hk)
        raise ValueError

    scale = query.size(-1) ** 0.5
    query = query / scale

    num_head_groups = hq // hk
    if num_head_groups > 1:
        query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
        similarity = einsum(query, key, "b g h n d, b h s d -> b h n s")
    else:
        similarity = einsum(query, key, "b h n d, b h s d -> b h n s")

    attention = F.softmax(similarity / scale, dim=-1)
    if dropout > 0.0:
        attention = F.dropout(attention, p=dropout)

    # Apply attention matrix to the value Tensor.
    out = einsum(attention, value, "b h n s, b h s d -> b h n d")
    # Move head dimension back to axis 2
    out = rearrange(out, "b h n d -> b n h d")
    return out


class MultiheadGQA(nn.Module):
    """Multi-head grouped query attention (GQA) layer.

    Reference:
        "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
        https://arxiv.org/pdf/2305.13245v1.pdf

    GQA is a variant of multihead attention (MHA) that uses fewer write heads
    (key / value) than query heads.  GQA can be viewed as a generalization of
    multi-query attention (MQA), which uses a single write head. GQA and MQA give
    significant speedups over standard MHA in decoder layers, with minimal loss in
    accuracy. In the paper, GQA is shown to be more accurate than MQA, while still
    having a significant speedup over MHA.

    NOTE: The original authors only benchmark GQA by adapting the T5 (XL or XXL) model
    from MHA to GQA.  As a result, they do not mention parameter initialization or
    layer normalization strategies.  I follow the best practices laid out in the
    MAGNETO paper, which improves Transformer performance through better parameter
    initialization and layer norm placement.  See:
        https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
    """

    def __init__(
        self,
        embed_dim: int,
        query_heads: int,
        kv_heads: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        is_self_attn: bool = True,
    ):
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.dropout = dropout
        self.gamma_init = gamma_init

        if self.query_heads % self.kv_heads != 0:
            print(self.query_heads, self.kv_heads)
            raise ValueError
        elif (embed_dim % self.query_heads != 0) or (embed_dim % self.kv_heads != 0):
            print(embed_dim, self.query_heads, self.kv_heads)
            raise ValueError

        head_dim = embed_dim // query_heads
        if not head_dim % 8 == 0:
            print(head_dim)
            raise ValueError
        if not head_dim <= 128:
            print(head_dim)
            raise ValueError

        # Query projection layer is the same as in vanilla MHA.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Key/value projection layers have a smaller output dimension, so that
        # the we have fewer key/value attention heads after reshaping.
        kv_embed_dim = embed_dim // query_heads * kv_heads
        self.k_proj = nn.Linear(embed_dim, kv_embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, kv_embed_dim, bias=False)
        self.is_self_attn = is_self_attn
        if is_self_attn:
            self.norm = nn.LayerNorm(kv_embed_dim, eps=layer_norm_eps)

        # Grouped attention output will have the same embedding dimension as the
        # key/value Tensors.  So the output projection layer needs to accept the
        # same dimension (kv_embed_dim).
        self.out_proj = nn.Linear(kv_embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)

        # TODO: PROVIDE SELF.GAMMA
        nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        # Notation:
        #   b - batch size
        #   n - sequence length
        #   h - number of heads
        #   d - embedding dimension

        q: Tensor = self.q_proj(query)
        k: Tensor = self.k_proj(key)
        v: Tensor = self.v_proj(value)

        # Unfold 'd' dimension into 'h' separate attention heads.
        q = rearrange(q, "b n (h d) -> b n h d", h=self.query_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.kv_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.kv_heads)

        # Apply attention, then fold 'h' attention heads back into 'd'.
        x = scaled_dot_product_gqa(q, k, v)
        x = rearrange(x, "b n h d -> b n (h d)")
        if self.is_self_attn:
            x = self.norm(x)

        # Linear projection on attention outputs.
        x = self.out_proj(x)
        return x


if __name__ == "__main__":
    ins = torch.randn(2, 40, 512).to("cuda")
    mhgqa = MultiheadGQA(512, 4, 1)
    out = mhgqa(ins, ins, ins)
    print(out.shape)
