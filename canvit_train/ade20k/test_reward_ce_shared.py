"""The policy reward matches CanViT-PyTorch-RL's, at the reference's resolution.

The harness used to score the reward at the probe's native 64 with nearest-downsampled
masks, while the reference scored at score_res=128 (bilinear-upsample logits,
stride-subsample masks). That was doc 15 §A gap #5 — the last known config difference
from CanViT-PyTorch-RL. `reward_ce` is now the single implementation.

The reference arm used to be imported from `ade20k/rl_train.py`, but that file was a
PORT, and by the end its `ce_from_logits` was a one-line delegation to `reward_ce` — so
the test was asserting that a wrapper equals what it wraps. Since the consolidation
deleted the port, the arm below is transcribed from the TRUE upstream instead
(`CanViT-PyTorch-RL/src/canvit_pytorch_rl/canvas_ops.py::ce_from_logits` +
`scoring.py::per_image_ce`), which is a strictly stronger guard: it can catch a
mis-transcription that comparing against our own port never could.
"""
import torch
import torch.nn.functional as F

from canvit_train.ade20k.data import IGNORE_LABEL
from canvit_train.ade20k.metrics import _warn_score_res, reward_ce

B, C, G, S = 2, 6, 8, 32


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, C, G, G, generator=g)
    masks = torch.randint(0, C, (B, S, S), generator=g)
    masks[0, 0, :4] = IGNORE_LABEL  # exercise the ignore path
    return logits, masks


def _upstream_ce_from_logits(logits, masks, *, score_res=None):
    """Verbatim transcription of canvit_pytorch_rl's reward, kept independent of ours.

    Do NOT refactor this to share code with `reward_ce` — being a separate expression is
    the entire point. The one deliberate difference is the indivisible-score_res case:
    upstream asserts, we warn and fall back to full res (covered separately below), so the
    comparison here only uses divisors of S. `.float()` on the logits is a no-op for the
    fp32 inputs used here.
    """
    full = masks.shape[-1]
    res = score_res or full
    assert full % res == 0
    m = masks if res == full else masks[:, :: full // res, :: full // res].contiguous()
    up = F.interpolate(logits, size=(res, res), mode="bilinear", align_corners=False)
    ce = F.cross_entropy(up, m, ignore_index=IGNORE_LABEL, reduction="none")
    valid = (m != IGNORE_LABEL).sum(dim=(1, 2)).clamp(min=1)
    return (ce.sum(dim=(1, 2)) / valid).float()


def test_matches_upstream_bit_identically():
    """The reward IS the objective, so it must equal the reference's to the bit."""
    logits, masks = _batch()
    for res in (None, 16, S):
        assert torch.equal(_upstream_ce_from_logits(logits, masks, score_res=res),
                           reward_ce(logits, masks, score_res=res)), f"score_res={res}"


def test_the_harness_task_uses_the_same_function_at_128_by_default():
    import inspect

    from canvit_train.ade20k.config import Ade20kConfig
    from canvit_train.ade20k.task import BoundAde20kTask

    assert Ade20kConfig().reward_score_res == 128, "must match the reference's score_res"
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
    lg = logging.getLogger("canvit_train.ade20k.metrics")
    lg.addHandler(h)
    try:
        for _ in range(5):
            reward_ce(logits, masks, score_res=7)
    finally:
        lg.removeHandler(h)
    assert len(records) == 1, f"expected 1 warning, got {len(records)}"
