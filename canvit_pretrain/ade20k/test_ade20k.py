"""CPU smoke tests for the unified ADE20K task (P2).

Covers both patcher routings of rollout_canvas_hidden on tiny models:
uniform -> pre-crop via sample_at_viewpoint; foveated -> full image (the NEW
capability specialize never had). Asserts the probe head trains while the
frozen backbone stays untouched."""

import pytest
import torch
from canvit_pytorch import CanViTForSemanticSegmentation
from canvit_pytorch.patcher import FoveatedPatcherConfig

from .data import IGNORE_LABEL, NUM_CLASSES
from .metrics import ce_loss
from .rollout import consumes_full_image, make_random_viewpoints, rollout_canvas_hidden

_B, _G, _T, _IMG = 2, 8, 2, 224
_DEVICE = torch.device("cpu")


def _tiny_seg(model_config: dict) -> CanViTForSemanticSegmentation:
    torch.manual_seed(0)
    seg = CanViTForSemanticSegmentation(
        backbone_name="vits16", model_config=model_config, num_classes=NUM_CLASSES
    ).to(_DEVICE)
    seg.canvit.requires_grad_(False)
    seg.canvit.eval()
    return seg


def _run_two_step(seg: CanViTForSemanticSegmentation) -> None:
    torch.manual_seed(1)
    images = torch.randn(_B, 3, _IMG, _IMG, device=_DEVICE)
    masks = torch.randint(0, NUM_CLASSES, (_B, _IMG, _IMG), device=_DEVICE)
    masks[:, :4] = IGNORE_LABEL
    vps = make_random_viewpoints(
        _B, _DEVICE, _T, min_scale=0.3, max_scale=1.0, start_with_full_scene=True
    )
    hidden = rollout_canvas_hidden(
        seg=seg, images=images, viewpoints=vps, canvas_grid=_G, glimpse_px=None
    )
    assert len(hidden) == _T
    assert hidden[0].shape[:3] == (_B, _G, _G)
    assert torch.isfinite(hidden[0]).all()

    logits = [seg.head(h.float()) for h in hidden]
    assert logits[0].shape == (_B, NUM_CLASSES, _G, _G)
    loss = torch.stack([ce_loss(lg, masks) for lg in logits]).mean()
    assert torch.isfinite(loss)
    loss.backward()

    head_grads = [p.grad for p in seg.head.parameters() if p.grad is not None]
    assert head_grads and any(g.abs().sum() > 0 for g in head_grads)
    assert all(p.grad is None for p in seg.canvit.parameters())


def test_uniform_rollout_trains_probe() -> None:
    seg = _tiny_seg({})
    assert not consumes_full_image(seg)
    _run_two_step(seg)


def test_foveated_rollout_trains_probe() -> None:
    seg = _tiny_seg({"patcher_name": "foveated", "foveated_patcher": FoveatedPatcherConfig()})
    assert consumes_full_image(seg)  # full-image routing — the new capability
    _run_two_step(seg)


def test_glimpse_px_token_guard() -> None:
    """A glimpse_px that yields the wrong token count must fail loudly."""
    from .rollout import derive_glimpse_px

    seg = _tiny_seg({})
    seg.canvit.glimpse_grid_size = 8  # 8 tokens/side @ patch 16 -> 128 px
    assert derive_glimpse_px(seg, None) == 128
    with pytest.raises(AssertionError):
        derive_glimpse_px(seg, 129)  # not stride-aligned
    with pytest.raises(AssertionError):
        derive_glimpse_px(seg, 160)  # aligned but 10 tokens != grid 8


def test_foveated_viewpoints_use_pretraining_scale() -> None:
    """Regression (job 15025338): the probe rollout must sample the FOVEATED scale
    law, not the uniform safe-box one. exp22-fovi pretrained at fixed_scale=2.0;
    feeding it safe-box scales (<=1) made mIoU DROP with each glimpse (0.217 ->
    0.198) because every glimpse was out of distribution."""
    from ..train.config import FoveatedScaleConfig

    fs = FoveatedScaleConfig(mode="fixed", fixed_scale=2.0)
    vps = make_random_viewpoints(
        _B, _DEVICE, 4, min_scale=0.05, max_scale=1.0, start_with_full_scene=True,
        is_foveated=True, foveated_scale=fs,
    )
    # mode='fixed' => EVERY glimpse (including the t0 anchor) at the training scale
    for vp in vps:
        assert torch.allclose(vp.scales, torch.full((_B,), 2.0)), f"{vp.scales} != 2.0"

    # uniform path untouched: safe-box law, scales in (0, 1], t0 full scene
    uni = make_random_viewpoints(
        _B, _DEVICE, 4, min_scale=0.05, max_scale=1.0, start_with_full_scene=True,
    )
    assert torch.allclose(uni[0].scales, torch.ones(_B))
    for vp in uni[1:]:
        assert (vp.scales > 0).all() and (vp.scales <= 1.0).all()
