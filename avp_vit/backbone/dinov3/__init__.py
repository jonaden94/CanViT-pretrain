"""DINOv3 backbone wrapper for AVP."""

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from .. import ViTBackbone

if TYPE_CHECKING:
    from dinov3.models.vision_transformer import DinoVisionTransformer


class DINOv3Backbone(ViTBackbone, nn.Module):
    """Wraps a DINOv3 ViT for use with AVP."""

    _backbone: "DinoVisionTransformer"

    def __init__(self, backbone: "DinoVisionTransformer") -> None:
        nn.Module.__init__(self)
        self._backbone = backbone

    @property
    @override
    def embed_dim(self) -> int:
        return self._backbone.embed_dim

    @property
    @override
    def num_heads(self) -> int:
        return self._backbone.num_heads

    @property
    @override
    def n_prefix_tokens(self) -> int:
        return 1 + self._backbone.n_storage_tokens

    @property
    @override
    def n_blocks(self) -> int:
        return len(self._backbone.blocks)

    @property
    @override
    def rope_periods(self) -> Tensor:
        return self._backbone.rope_embed.periods

    @property
    @override
    def rope_dtype(self) -> torch.dtype:
        return self._backbone.rope_embed.dtype

    @override
    def forward_block(self, idx: int, x: Tensor, rope: tuple[Tensor, Tensor] | None) -> Tensor:
        return self._backbone.blocks[idx](x, rope)

    @override
    def prepare_tokens(self, images: Tensor) -> tuple[Tensor, int, int]:
        x, (H, W) = self._backbone.prepare_tokens_with_masks(images, masks=None)
        return x, H, W
