"""The mIoU reduction must upsample logits BEFORE argmax (the paper protocol).

This repo had the order backwards until 2026-07-29: argmax at the 64x64 probe grid,
then nearest-upsample. That cost a measured 0.19 mIoU at t4 on ADE20K val (2 seeds).
"""
import torch
import torch.nn.functional as F

from .metrics import preds_from_logits, upsample_preds


def test_matches_canvit_eval_and_the_rl_repo():
    """Both references do `F.interpolate(logits, ..., bilinear).argmax(1)`:
    canvit_eval/tasks/ade20k_seg.py and canvit_pytorch_rl/scoring.py."""
    torch.manual_seed(0)
    logits = torch.randn(2, 150, 64, 64)
    reference = F.interpolate(logits, size=(512, 512), mode="bilinear",
                              align_corners=False).argmax(dim=1)
    assert torch.equal(preds_from_logits(logits, 512, 512), reference)


def test_differs_from_the_old_argmax_first_order():
    """Guards the regression: if these ever agree, the fix has been undone."""
    torch.manual_seed(0)
    logits = torch.randn(2, 150, 64, 64)
    old = upsample_preds(logits.argmax(1), 512, 512)
    assert not torch.equal(preds_from_logits(logits, 512, 512), old)


def test_is_a_noop_when_already_at_mask_resolution():
    torch.manual_seed(0)
    logits = torch.randn(2, 150, 512, 512)
    assert torch.equal(preds_from_logits(logits, 512, 512), logits.argmax(1))


def test_probe_eval_uses_the_paper_order():
    """`eval_probe_on_batch` is THE ade20k probe/harness metric path — it must not
    silently drift back to argmax-first."""
    import inspect

    from .metrics import eval_probe_on_batch

    src = inspect.getsource(eval_probe_on_batch)
    assert "preds_from_logits" in src
    assert "upsample_preds" not in src
