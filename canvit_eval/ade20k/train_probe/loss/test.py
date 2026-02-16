"""Tests for loss functions."""

import torch

from . import ce_loss, upsample_preds


def test_ce_loss_shape():
    B, C, H, W = 2, 150, 32, 32
    logits = torch.randn(B, C, H, W)
    masks = torch.randint(0, C, (B, H, W))
    loss = ce_loss(logits, masks)
    assert loss.shape == ()
    assert loss.item() > 0


def test_ce_loss_downsamples_masks():
    B, C = 2, 150
    logits = torch.randn(B, C, 16, 16)
    masks = torch.randint(0, C, (B, 32, 32))
    loss = ce_loss(logits, masks)
    assert loss.shape == ()


def test_upsample_preds_noop():
    preds = torch.randint(0, 150, (2, 32, 32))
    out = upsample_preds(preds, 32, 32)
    assert torch.equal(out, preds)


def test_upsample_preds_upscale():
    preds = torch.randint(0, 150, (2, 16, 16))
    out = upsample_preds(preds, 32, 32)
    assert out.shape == (2, 32, 32)
    assert out.dtype == torch.int64
