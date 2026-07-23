"""Engine-behavior tests for the new axes beyond distill parity (design §6):

  * ``sample_n_glimpses`` — fixed horizon vs stochastic TBPTT length.
  * grad regime ``none`` (frozen backbone forward under no_grad) vs ``full`` — the
    ade20k/in1k pattern where the head is applied OUTSIDE the backbone forward, so
    a frozen backbone still trains its head. Validates that the engine's
    ``backbone_no_grad`` wiring routes gradients to exactly the right params.
"""

import random
from types import SimpleNamespace

import torch
from canvit_pytorch import create_backbone
from torch import nn

from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig
from canvit_pretrain.harness.rollout import GlimpseOut, run_rollout, sample_n_glimpses
from canvit_pretrain.harness.spec import BpttSpec
from canvit_pretrain.train.config import FoveatedScaleConfig
from canvit_pretrain.train.selector import RandomSelector
from canvit_pretrain.train.viewpoint import ViewpointType

_B, _G, _D = 2, 8, 384
_DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- #
# sample_n_glimpses (pure — no model).
# --------------------------------------------------------------------------- #
def test_fixed_horizon():
    assert sample_n_glimpses(BpttSpec(mode="full", horizon=7)) == 7
    assert sample_n_glimpses(BpttSpec(mode="none", horizon=1)) == 1


def test_stochastic_length_is_multiple_of_chunk_size():
    bptt = BpttSpec(mode="chunked", chunk_size=2, continue_prob=0.5)
    for seed in range(20):
        n = sample_n_glimpses(bptt, rng=random.Random(seed))
        assert n >= 2 and n % 2 == 0


def test_stochastic_length_deterministic_given_rng():
    bptt = BpttSpec(mode="chunked", chunk_size=3, continue_prob=0.7)
    a = sample_n_glimpses(bptt, rng=random.Random(123))
    b = sample_n_glimpses(bptt, rng=random.Random(123))
    assert a == b and a % 3 == 0


# --------------------------------------------------------------------------- #
# Grad regime none vs full — external-head grad routing.
# --------------------------------------------------------------------------- #
class _HeadTask:
    """Minimal task with a head applied OUTSIDE the backbone forward (ade20k/in1k
    pattern): loss = MSE of a linear readout of the recurrent CLS vs a target."""

    def __init__(self, head: nn.Module):
        self.head = head

    def forward_glimpse(self, *, model, images, state, viewpoint, backbone_no_grad):
        ctx = torch.no_grad() if backbone_no_grad else torch.enable_grad()
        with ctx:
            out = model(image=images, state=state, viewpoint=viewpoint)
        return GlimpseOut(readout=out.state, state=out.state, vpe=out.vpe)

    def step_loss(self, readout):
        cls = readout.recurrent_cls[:, 0].float()
        pred = self.head(cls)
        return SimpleNamespace(combined=(pred**2).mean())

    def per_image_loss(self, readout):
        cls = readout.recurrent_cls[:, 0].float()
        return (self.head(cls) ** 2).mean(dim=1)


def _model() -> CanViTForPretraining:
    torch.manual_seed(0)
    backbone = create_backbone("vits16").to(_DEVICE)
    return CanViTForPretraining(
        backbone=backbone, cfg=CanViTForPretrainingConfig(teacher_dim=_D),
        glimpse_size_px=128, backbone_name="vits16", canvas_patch_grid_sizes=[_G],
    ).to(_DEVICE)


def _run(mode: str, model, head):
    torch.manual_seed(7)
    selector = RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.1)
    for p in model.parameters():
        p.grad = None
    for p in head.parameters():
        p.grad = None
    run_rollout(
        model=model, images=torch.randn(_B, 3, 224, 224), task=_HeadTask(head),
        selector=selector, bptt=BpttSpec(mode=mode, horizon=3),
        branches=[ViewpointType.RANDOM], canvas_grid_size=_G,
        amp_ctx=torch.enable_grad(),
    )


def test_regime_none_freezes_backbone_trains_head():
    model, head = _model(), nn.Linear(384, 384)
    _run("none", model, head)
    # Backbone forward ran under no_grad -> no backbone gradients.
    assert all(p.grad is None for p in model.backbone.parameters())
    # Head applied outside -> it still learns.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def test_regime_full_trains_backbone():
    model, head = _model(), nn.Linear(384, 384)
    _run("full", model, head)
    # Backbone carries grad in 'full' -> at least some backbone params get gradients.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())
    assert any(p.grad is not None for p in head.parameters())


# --------------------------------------------------------------------------- #
# task_weight scaling (design §4) — parity path is task_weight=1.0 (guarded).
# --------------------------------------------------------------------------- #
def _run_tw(model, head, tw: float):
    """Deterministic single-branch 'full' rollout; return the head's flattened grad."""
    torch.manual_seed(7)  # same viewpoints for every tw (fixed horizon => no random draw)
    for p in list(model.parameters()) + list(head.parameters()):
        p.grad = None
    run_rollout(
        model=model, images=torch.randn(_B, 3, 224, 224), task=_HeadTask(head),
        selector=RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.1),
        bptt=BpttSpec(mode="full", horizon=3), branches=[ViewpointType.RANDOM],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), task_weight=tw,
    )
    return torch.cat([p.grad.reshape(-1) for p in head.parameters() if p.grad is not None])


def test_task_weight_scales_task_grad_linearly():
    # The loss is linear in task_weight, so head grads scale by exactly task_weight.
    model, head = _model(), nn.Linear(384, 384)
    g_full = _run_tw(model, head, 1.0)
    g_half = _run_tw(model, head, 0.5)
    assert g_full.abs().sum() > 0
    assert torch.allclose(g_half, 0.5 * g_full, atol=1e-6, rtol=1e-4)


def test_task_weight_zero_kills_task_grad():
    model, head = _model(), nn.Linear(384, 384)
    g = _run_tw(model, head, 0.0)
    assert g.abs().sum() == 0  # task loss contributes nothing
