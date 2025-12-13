from typing import final, override

import torch.nn.functional as F
from torch import Tensor, nn

from avp_vit.rope import rope_apply_with_prefix


@final
class CrossAttention(nn.Module):
    """Cross-attention with RoPE. Q attends to KV."""

    num_heads: int
    head_dim: int
    q_proj: nn.Linear
    kv_proj: nn.Linear
    out_proj: nn.Linear

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

    @override
    def forward(
        self,
        q_tokens: Tensor,
        kv_tokens: Tensor,
        q_rope: tuple[Tensor, Tensor] | None = None,
        kv_rope: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        B, N_q, D = q_tokens.shape
        N_kv = kv_tokens.shape[1]

        q = self.q_proj(q_tokens).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(kv_tokens).view(B, N_kv, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if q_rope is not None:
            q = rope_apply_with_prefix(q, q_rope)
        if kv_rope is not None:
            k = rope_apply_with_prefix(k, kv_rope)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N_q, D)
        return self.out_proj(out)
