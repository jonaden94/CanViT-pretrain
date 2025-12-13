"""DINOv3 backbone wrapper for AVP."""

from typing import TYPE_CHECKING, cast, override

import torch
from torch import Tensor, nn

from .. import ViTBackbone

if TYPE_CHECKING:
    from dinov3.models.vision_transformer import DinoVisionTransformer


class DINOv3Backbone(ViTBackbone, nn.Module):
    """Wraps a DINOv3 ViT for use with AVP."""

    _backbone: "DinoVisionTransformer"
    _embed_dim: int
    _num_heads: int
    _n_prefix_tokens: int
    _n_blocks: int
    _rope_periods: Tensor
    _rope_dtype: torch.dtype

    def __init__(self, backbone: "DinoVisionTransformer") -> None:
        nn.Module.__init__(self)
        self._backbone = backbone

        # Cache properties with explicit types (avoids Any from nn.Module getattr)
        self._embed_dim = backbone.embed_dim
        self._num_heads = backbone.num_heads
        self._n_prefix_tokens = 1 + backbone.n_storage_tokens
        self._n_blocks = len(backbone.blocks)

        rope_embed = backbone.rope_embed
        periods = rope_embed.periods
        dtype = rope_embed.dtype
        assert isinstance(periods, Tensor)
        assert dtype is not None
        self.register_buffer("_rope_periods", periods)
        self._rope_dtype = dtype

    @property
    @override
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    @override
    def num_heads(self) -> int:
        return self._num_heads

    @property
    @override
    def n_prefix_tokens(self) -> int:
        return self._n_prefix_tokens

    @property
    @override
    def n_blocks(self) -> int:
        return self._n_blocks

    @property
    @override
    def rope_periods(self) -> Tensor:
        return self._rope_periods

    @property
    @override
    def rope_dtype(self) -> torch.dtype:
        return self._rope_dtype

    @override
    def forward_block(self, idx: int, x: Tensor, rope: tuple[Tensor, Tensor] | None) -> Tensor:
        out: Tensor = self._backbone.blocks[idx](x, rope)
        return out

    @override
    def prepare_tokens(self, images: Tensor) -> tuple[Tensor, int, int]:
        result = self._backbone.prepare_tokens_with_masks(images, masks=None)
        x: Tensor = result[0]
        # dinov3's stub says Tuple[int] but it's actually (H, W)
        grid_size = cast(tuple[int, int], result[1])
        H, W = grid_size
        return x, H, W
