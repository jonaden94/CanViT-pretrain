"""Segmentation metrics using DINOv3's histc-based approach.

Works on MPS/CUDA/CPU without sync issues in hot path.
"""

import torch
from dinov3.eval.segmentation.metrics import calculate_intersect_and_union
from torch import Tensor


def per_image_miou(preds: Tensor, targets: Tensor, num_classes: int, ignore_index: int) -> Tensor:
    """Compute mIoU for each image in batch. Returns [B] tensor."""
    B = preds.shape[0]
    mious = torch.empty(B, device=preds.device)
    for i in range(B):
        inter, union, _, _ = calculate_intersect_and_union(
            preds[i], targets[i], num_classes, ignore_index
        )
        valid = union > 0
        mious[i] = (inter[valid] / (union[valid] + 1e-8)).mean() if valid.any() else 0.0
    return mious


class IoUAccumulator:
    """Accumulates intersection/union for mIoU computation.

    Uses torch.histc (DINOv3 approach) - no GPU sync until compute().
    """

    def __init__(self, num_classes: int, ignore_index: int, device: torch.device) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.intersection = torch.zeros(num_classes, device=device)
        self.union = torch.zeros(num_classes, device=device)

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Accumulate stats. Handles batch or single image. No GPU sync."""
        if preds.dim() == 3:  # (B, H, W)
            for i in range(preds.shape[0]):
                self._update_single(preds[i], targets[i])
        else:  # (H, W)
            self._update_single(preds, targets)

    def _update_single(self, preds: Tensor, targets: Tensor) -> None:
        areas = calculate_intersect_and_union(
            preds, targets, self.num_classes, self.ignore_index
        )
        self.intersection += areas[0]
        self.union += areas[1]

    def compute(self) -> float:
        """Compute mIoU. GPU sync happens here."""
        iou_per_class = self.intersection / (self.union + 1e-8)
        valid = self.union > 0
        return iou_per_class[valid].mean().item()

    def reset(self) -> None:
        self.intersection.zero_()
        self.union.zero_()

    def to(self, device: torch.device) -> "IoUAccumulator":
        """Move to device (for compatibility with torchmetrics pattern)."""
        self.device = device
        self.intersection = self.intersection.to(device)
        self.union = self.union.to(device)
        return self
