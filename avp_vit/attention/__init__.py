from typing import final

import torch.nn.functional as F
from torch import Tensor, nn

from avp_vit.rope import rope_apply_with_prefix


class RoPECrossAttention(nn.Module):
    """Cross-attention with RoPE. Subclasses configure Q/K/V/O transforms."""

    num_heads: int
    head_dim: int
    norm_q: nn.LayerNorm
    norm_kv: nn.LayerNorm
    q_transform: nn.Module
    k_transform: nn.Module
    v_transform: nn.Module
    out_transform: nn.Module

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def _to_heads(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _from_heads(self, x: Tensor) -> Tensor:
        B, _, N, _ = x.shape
        return x.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)

    def forward(
        self,
        q_in: Tensor,
        kv_in: Tensor,
        q_rope: tuple[Tensor, Tensor],
        kv_rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        q = self._to_heads(self.q_transform(self.norm_q(q_in)))
        k = self._to_heads(self.k_transform(self.norm_kv(kv_in)))
        v = self._to_heads(self.v_transform(self.norm_kv(kv_in)))

        q = rope_apply_with_prefix(q, q_rope)
        k = rope_apply_with_prefix(k, kv_rope)

        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_transform(self._from_heads(out))


@final
class RoPEReadCrossAttention(RoPECrossAttention):
    """For reading: Q and O projected, K and V unprojected."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__(dim, num_heads)
        self.q_transform = nn.Linear(dim, dim)
        self.k_transform = nn.Identity()
        self.v_transform = nn.Identity()
        self.out_transform = nn.Linear(dim, dim)


@final
class RoPEWriteCrossAttention(RoPECrossAttention):
    """For writing: K and V projected, Q and O unprojected."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__(dim, num_heads)
        self.q_transform = nn.Identity()
        self.k_transform = nn.Linear(dim, dim)
        self.v_transform = nn.Linear(dim, dim)
        self.out_transform = nn.Identity()
