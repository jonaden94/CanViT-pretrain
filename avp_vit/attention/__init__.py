from dataclasses import dataclass
from typing import final

import torch.nn.functional as F
from torch import Tensor, nn
from ytch.attention.mh import from_multihead, to_multihead
from ytch.nn.elementwise_affine import ElementwiseAffine

from avp_vit.rope import rope_apply_with_prefix


@final
@dataclass
class AttentionConfig:
    """Ablation flags for cross-attention modules."""

    use_ewa_transforms: bool = True  # EWA instead of Identity for unprojected streams
    use_post_rope_ewa: bool = False  # EWA after RoPE on Q/K
    identity_init_v: bool = False  # Identity-init V projection (write attn only)
    write_v_expansion: int | None = None  # None = Linear, int = MLP with SiLU and given expansion


class RoPECrossAttention(nn.Module):
    """Cross-attention with RoPE. Subclasses configure Q/K/V/O transforms."""

    dim: int
    num_heads: int
    cfg: AttentionConfig
    post_rope_q: nn.Module
    post_rope_k: nn.Module
    q_transform: nn.Module
    k_transform: nn.Module
    v_transform: nn.Module
    out_transform: nn.Module

    def __init__(self, dim: int, num_heads: int, cfg: AttentionConfig) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.cfg = cfg

        # Post-RoPE affines (operate on head_dim)
        head_dim = dim // num_heads
        post = ElementwiseAffine if cfg.use_post_rope_ewa else nn.Identity
        self.post_rope_q = post(head_dim)
        self.post_rope_k = post(head_dim)

    def forward(
        self,
        q_in: Tensor,
        kv_in: Tensor,
        q_rope: tuple[Tensor, Tensor],
        kv_rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        q_normed = F.layer_norm(q_in, (self.dim,))
        kv_normed = F.layer_norm(kv_in, (self.dim,))

        q = to_multihead(self.q_transform(q_normed), self.num_heads)
        k = to_multihead(self.k_transform(kv_normed), self.num_heads)
        v = to_multihead(self.v_transform(kv_normed), self.num_heads)

        q = self.post_rope_q(rope_apply_with_prefix(q, q_rope))
        k = self.post_rope_k(rope_apply_with_prefix(k, kv_rope))

        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_transform(from_multihead(out))

    def flops(self, n_q: int, n_kv: int) -> int:
        """FLOPs for one forward pass. SDPA + Linear projections (ignores cheap EWA)."""
        D = self.dim
        f = 4 * n_q * n_kv * D  # SDPA: Q @ Kᵀ + softmax @ V
        if isinstance(self.q_transform, nn.Linear):
            f += 2 * n_q * D * D
        if isinstance(self.k_transform, nn.Linear):
            f += 2 * n_kv * D * D
        if isinstance(self.v_transform, nn.Linear):
            f += 2 * n_kv * D * D
        if isinstance(self.out_transform, nn.Linear):
            f += 2 * n_q * D * D
        return f


def _ewa_or_identity(dim: int, use_ewa: bool) -> nn.Module:
    return ElementwiseAffine(dim) if use_ewa else nn.Identity()


@final
class RoPEReadCrossAttention(RoPECrossAttention):
    """For reading: Q and O projected, K and V use EWA (or Identity)."""

    def __init__(self, dim: int, num_heads: int, cfg: AttentionConfig) -> None:
        super().__init__(dim, num_heads, cfg)
        self.q_transform = nn.Linear(dim, dim)
        self.k_transform = _ewa_or_identity(dim, cfg.use_ewa_transforms)
        self.v_transform = _ewa_or_identity(dim, cfg.use_ewa_transforms)
        self.out_transform = nn.Linear(dim, dim)


@final
class RoPEWriteCrossAttention(RoPECrossAttention):
    """For writing: K and V projected, Q and O use EWA (or Identity)."""

    def __init__(self, dim: int, num_heads: int, cfg: AttentionConfig) -> None:
        super().__init__(dim, num_heads, cfg)
        self.q_transform = _ewa_or_identity(dim, cfg.use_ewa_transforms)
        self.k_transform = nn.Linear(dim, dim)
        self.v_transform = self._make_v_proj(dim, cfg.identity_init_v, cfg.write_v_expansion)
        self.out_transform = _ewa_or_identity(dim, cfg.use_ewa_transforms)

    @staticmethod
    def _make_v_proj(dim: int, identity_init: bool, expansion: int | None) -> nn.Module:
        if expansion is None:
            proj = nn.Linear(dim, dim)
            if identity_init:
                nn.init.eye_(proj.weight)
                nn.init.zeros_(proj.bias)
            return proj
        return nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Linear(dim * expansion, dim),
        )


