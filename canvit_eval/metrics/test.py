"""Tests for IoU metrics."""

import torch
from torchmetrics.classification import MulticlassJaccardIndex

from . import IoUAccumulator


def test_matches_torchmetrics() -> None:
    """Verify IoUAccumulator matches torchmetrics on real-ish data."""
    NUM_CLASSES = 150
    IGNORE = 255

    torch.manual_seed(42)
    preds = torch.randint(0, NUM_CLASSES, (4, 64, 64))
    targets = torch.randint(0, NUM_CLASSES, (4, 64, 64))
    targets[0, :10, :10] = IGNORE  # some ignored pixels

    # torchmetrics
    tm = MulticlassJaccardIndex(NUM_CLASSES, ignore_index=IGNORE, average="macro")
    tm.update(preds, targets)
    tm_miou = tm.compute().item()

    # Ours
    acc = IoUAccumulator(NUM_CLASSES, IGNORE, torch.device("cpu"))
    acc.update(preds, targets)
    our_miou = acc.compute()

    assert abs(tm_miou - our_miou) < 1e-6, f"Mismatch: {tm_miou} vs {our_miou}"


def test_reset() -> None:
    """Test reset clears state."""
    acc = IoUAccumulator(10, 255, torch.device("cpu"))
    acc.update(torch.zeros(8, 8, dtype=torch.long), torch.zeros(8, 8, dtype=torch.long))
    assert acc.intersection.sum() > 0
    acc.reset()
    assert acc.intersection.sum() == 0
    assert acc.union.sum() == 0


def test_single_image() -> None:
    """Test with single image (H, W) instead of batch."""
    acc = IoUAccumulator(10, 255, torch.device("cpu"))
    preds = torch.randint(0, 10, (32, 32))
    targets = torch.randint(0, 10, (32, 32))
    acc.update(preds, targets)
    miou = acc.compute()
    assert 0 <= miou <= 1
