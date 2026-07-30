"""The policy reward comes from ONE implementation, at the reference's resolution.

`rl_train.ce_from_logits` and `BoundAde20kTask.per_image_loss` both compute the reward.
They used to differ: rl_train scored at score_res=128 (bilinear-upsample logits,
stride-subsample masks) while the harness scored at the probe's native 64 with
nearest-downsampled masks. That was doc 15 §A gap #5 — the last known config difference
from CanViT-PyTorch-RL. Both now call `reward_ce`.
"""
import torch

from canvit_pretrain.ade20k.data import IGNORE_LABEL
from canvit_pretrain.ade20k.metrics import _warn_score_res, reward_ce
from canvit_pretrain.ade20k.rl_train import ce_from_logits

B, C, G, S = 2, 6, 8, 32


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, C, G, G, generator=g)
    masks = torch.randint(0, C, (B, S, S), generator=g)
    masks[0, 0, :4] = IGNORE_LABEL  # exercise the ignore path
    return logits, masks


def test_rl_train_delegates_bit_identically():
    """rl_train is the frozen reference: delegating must not change its numbers."""
    logits, masks = _batch()
    for res in (None, 16, S):
        assert torch.equal(ce_from_logits(logits, masks, score_res=res),
                           reward_ce(logits, masks, score_res=res)), f"score_res={res}"


def test_the_harness_task_uses_the_same_function_at_128_by_default():
    import inspect

    from canvit_pretrain.ade20k.config import Ade20kConfig
    from canvit_pretrain.tasks.ade20k.task import BoundAde20kTask

    assert Ade20kConfig().reward_score_res == 128, "must match rl_train's score_res"
    src = inspect.getsource(BoundAde20kTask.per_image_loss)
    assert "reward_ce" in src
    assert "nearest" not in src, "the old native-grid/nearest-downsample path must be gone"


def test_score_res_actually_changes_the_reward():
    """Guards against the knob being silently ignored — the whole failure mode of gap #5."""
    logits, masks = _batch()
    assert not torch.allclose(reward_ce(logits, masks, score_res=16),
                              reward_ce(logits, masks, score_res=S))


def test_none_means_full_mask_resolution():
    logits, masks = _batch()
    assert torch.equal(reward_ce(logits, masks, score_res=None),
                       reward_ce(logits, masks, score_res=S))


def test_indivisible_score_res_falls_back_to_full_res_and_warns(caplog):
    """A stride needs divisibility. Rather than assert -- 128 divides the reference's 512
    but not e.g. a 224 scene -- fall back to FULL resolution, which is what score_res
    approximates for speed, so the reward can only get slower, never wrong. Loudly."""
    import logging

    logits, masks = _batch()
    _warn_score_res.cache_clear()
    with caplog.at_level(logging.WARNING):
        got = reward_ce(logits, masks, score_res=7)   # 7 does not divide S=32
    assert torch.equal(got, reward_ce(logits, masks, score_res=None))
    assert "does not divide" in caplog.text


def test_the_fallback_warning_fires_once_not_per_glimpse():
    """It sits in the per-glimpse reward path; an unthrottled warning would flood."""
    import logging

    logits, masks = _batch()
    _warn_score_res.cache_clear()
    records = []
    h = logging.Handler()
    h.emit = records.append
    lg = logging.getLogger("canvit_pretrain.ade20k.metrics")
    lg.addHandler(h)
    try:
        for _ in range(5):
            reward_ce(logits, masks, score_res=7)
    finally:
        lg.removeHandler(h)
    assert len(records) == 1, f"expected 1 warning, got {len(records)}"
