"""Segmentation probe utilities for ADE20K evaluation.

The probe class (SegmentationProbe) lives in canvit_utils.probes.
This module provides eval-specific helpers that depend on IoUAccumulator.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from canvit_eval.metrics import IoUAccumulator


def _upsample_preds(preds: Tensor, H: int, W: int) -> Tensor:
    if preds.shape[1:] == (H, W):
        return preds
    return F.interpolate(preds.unsqueeze(1).float(), (H, W), mode="nearest").squeeze(1).long()


def eval_probe_on_batch(
    probe: nn.Module,
    features: Tensor,
    masks: Tensor,
    iou: IoUAccumulator,
) -> None:
    """Forward probe, upsample predictions, update IoU accumulator."""
    logits = probe(features.float())
    preds_up = _upsample_preds(logits.argmax(1), masks.shape[1], masks.shape[2])
    iou.update(preds_up, masks)
