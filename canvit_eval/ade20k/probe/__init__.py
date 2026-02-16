"""Segmentation probe head for ADE20K.

Architecture: LN -> BN -> Dropout -> Conv1x1

Shape flow for 512x512 input with patch_size=16:
  Input image: [B, 3, 512, 512]
  Backbone patches: [B, 32, 32, D]
  Probe output: [B, 150, 32, 32]
  After rescale: [B, 150, H, W]
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from canvit_eval.ade20k.dataset import NUM_CLASSES
from canvit_eval.metrics import IoUAccumulator


class ProbeHead(nn.Module):
    """Linear probe: optional LN + DINOv3-style head (BN + Dropout + Conv1x1)."""

    def __init__(self, embed_dim: int, dropout: float, use_ln: bool) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ln = nn.LayerNorm(embed_dim) if use_ln else nn.Identity()
        self.bn = nn.BatchNorm2d(embed_dim)
        self.dropout = nn.Dropout2d(dropout)
        self.conv = nn.Conv2d(embed_dim, NUM_CLASSES, kernel_size=1)
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """[B, H_patches, W_patches, D] -> [B, NUM_CLASSES, H_patches, W_patches]."""
        B, Hp, Wp, D = x.shape
        assert D == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {D}"

        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).clone()  # clone for MPS BN bug
        x = self.dropout(x)
        x = self.bn(x)
        x = self.conv(x)

        assert x.shape == (B, NUM_CLASSES, Hp, Wp)
        return x

    def predict(self, x: Tensor, rescale_to: tuple[int, int]) -> Tensor:
        out = self(x)
        return F.interpolate(out, size=rescale_to, mode="bilinear", align_corners=False)


def _upsample_preds(preds: Tensor, H: int, W: int) -> Tensor:
    if preds.shape[1:] == (H, W):
        return preds
    return F.interpolate(preds.unsqueeze(1).float(), (H, W), mode="nearest").squeeze(1).long()


def eval_probe_on_batch(
    probe: ProbeHead,
    features: Tensor,
    masks: Tensor,
    iou: IoUAccumulator,
) -> None:
    """Forward probe, upsample predictions, update IoU accumulator."""
    logits = probe(features.float())
    preds_up = _upsample_preds(logits.argmax(1), masks.shape[1], masks.shape[2])
    iou.update(preds_up, masks)
