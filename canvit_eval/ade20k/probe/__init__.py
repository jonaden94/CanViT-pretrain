"""Segmentation probe head for ADE20K."""

import torch.nn as nn
from torch import Tensor

from canvit_eval.ade20k.dataset import NUM_CLASSES


class ProbeHead(nn.Module):
    """Linear probe: LayerNorm + Linear for dense prediction."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, H, W, D] -> [B, C, H, W]"""
        assert x.ndim == 4
        return self.linear(self.ln(x)).permute(0, 3, 1, 2)
