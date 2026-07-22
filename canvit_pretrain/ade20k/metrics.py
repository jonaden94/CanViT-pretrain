"""Segmentation loss, mIoU accumulation, and probe training state.

Port of canvit_specialize's loss.py / metrics.py / state.py / eval_utils.py
(unchanged math; the P2 gate compares numbers against specialize runs)."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from canvit_pytorch.probes import SegmentationProbe
from torch import Tensor
from torch.optim import AdamW

from .data import IGNORE_LABEL


def ce_loss(logits: Tensor, masks: Tensor) -> Tensor:
    """Cross-entropy for semantic segmentation (masks nearest-resized to logits)."""
    if masks.shape[1:] != logits.shape[2:]:
        masks = F.interpolate(masks.unsqueeze(1).float(), logits.shape[2:], mode="nearest").squeeze(1).long()
    return F.cross_entropy(logits, masks, ignore_index=IGNORE_LABEL)


def upsample_preds(preds: Tensor, H: int, W: int) -> Tensor:
    if preds.shape[1:] == (H, W):
        return preds
    return F.interpolate(preds.unsqueeze(1).float(), (H, W), mode="nearest").squeeze(1).long()


class mIoUAccumulator:
    """Global mIoU: sum intersection/union across all images, then average over
    classes. No GPU sync until compute()."""

    def __init__(self, num_classes: int, ignore_index: int, device: torch.device) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.intersection = torch.zeros(num_classes, device=device)
        self.union = torch.zeros(num_classes, device=device)

    def update(self, preds: Tensor, targets: Tensor) -> None:
        assert preds.ndim == 3, f"Expected [B, H, W], got shape {preds.shape}"
        assert preds.shape == targets.shape, f"Shape mismatch: {preds.shape} vs {targets.shape}"
        n = self.num_classes
        p = preds.flatten().long()
        t = targets.flatten().long()
        valid = t != self.ignore_index
        p, t = p[valid], t[valid]
        cm = torch.bincount(t * n + p, minlength=n * n).view(n, n)
        diag = cm.diag()
        self.intersection += diag
        self.union += cm.sum(dim=1) + cm.sum(dim=0) - diag

    def reset(self) -> None:
        self.intersection.zero_()
        self.union.zero_()

    def compute(self) -> float:
        iou = self.intersection / (self.union + 1e-8)
        valid = self.union > 0
        return iou[valid].mean().item() if valid.any() else 0.0


def eval_probe_on_batch(probe: nn.Module, features: Tensor, masks: Tensor, iou: mIoUAccumulator) -> None:
    """Forward probe, upsample predictions, update IoU accumulator."""
    logits = probe(features.float())
    preds_up = upsample_preds(logits.argmax(1), masks.shape[1], masks.shape[2])
    iou.update(preds_up, masks)


@dataclass
class ProbeState:
    """Training state for one probe head."""

    name: str
    head: SegmentationProbe
    optimizer: AdamW
    scheduler: Any  # WarmupOneCycleLR or other LR scheduler
    n_timesteps: int = 0
    _best_mious: list[float] | None = None
    _loss_sum: Tensor | None = None
    _grad_norm_sum: Tensor | None = None
    _count: int = 0

    def init_best_mious(self, n_timesteps: int) -> None:
        self.n_timesteps = n_timesteps
        self._best_mious = [0.0] * n_timesteps

    @property
    def best_mious(self) -> list[float]:
        assert self._best_mious is not None, "call init_best_mious first"
        return self._best_mious

    @property
    def best_last_miou(self) -> float:
        return self.best_mious[-1]

    def update_best(self, mious: list[float]) -> bool:
        """Update per-timestep bests. Returns True if last timestep improved."""
        assert len(mious) == self.n_timesteps
        old_last = self.best_last_miou
        for t, v in enumerate(mious):
            if v > self.best_mious[t]:
                self.best_mious[t] = v
        return self.best_last_miou > old_last

    def accumulate(self, loss: Tensor, grad_norm: Tensor) -> None:
        """Accumulate loss/grad_norm. NO GPU sync."""
        if self._loss_sum is None:
            self._loss_sum = loss.detach().clone()
            self._grad_norm_sum = grad_norm.detach().clone()
        else:
            self._loss_sum += loss.detach()
            assert self._grad_norm_sum is not None
            self._grad_norm_sum += grad_norm.detach()
        self._count += 1

    def get_and_reset(self) -> tuple[float, float]:
        """Get averaged stats and reset. SYNCS here."""
        assert self._loss_sum is not None and self._grad_norm_sum is not None
        avg_loss = (self._loss_sum / self._count).item()
        avg_grad = (self._grad_norm_sum / self._count).item()
        self._loss_sum = self._grad_norm_sum = None
        self._count = 0
        return avg_loss, avg_grad
