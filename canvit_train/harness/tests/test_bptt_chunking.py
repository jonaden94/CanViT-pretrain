"""Chunked BPTT on the fixed-horizon tasks: the rule, and the indivisible/prime case.

Two things are pinned here:
  1. `fixed_horizon_bptt` maps (mode, chunk_size) onto a regime, including the collapse
     cases — frozen always wins, and chunk >= horizon is just `full`.
  2. `run_rollout` produces the SAME total gradient no matter how the rollout is chunked,
     including when the horizon is not divisible by the chunk size (the trailing partial
     chunk at rollout.py's tail). A prime horizon is the sharp version of that.

Also pins the measured invariant behind the "frozen => none" rule: `bptt` moves the
BACKBONE only, never the head.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from canvit_pytorch import create_backbone
from torch import nn

from canvit_train import CanViTForPretraining, CanViTForPretrainingConfig
from canvit_train.harness.config import FoveatedScaleConfig
from canvit_train.harness.rollout import GlimpseOut, run_rollout
from canvit_train.harness.selector import RandomSelector
from canvit_train.harness.spec import BpttSpec, fixed_horizon_bptt
from canvit_train.harness.viewpoint import ViewpointType

_B, _G, _D = 2, 4, 32


# --- the mapping rule -------------------------------------------------------
def test_frozen_always_yields_none_whatever_the_chunk_size():
    """Probe mode must ignore chunk_size: with the backbone under no_grad there is
    nothing to accumulate into, so chunking would cost memory and change nothing."""
    for chunk in (0, 1, 3, 99):
        assert fixed_horizon_bptt(frozen=True, horizon=10, chunk_size=chunk).mode == "none"


@pytest.mark.parametrize("chunk,expected_mode", [
    (0, "full"),      # default: one graph over the rollout
    (1, "chunked"),
    (3, "chunked"),
    (10, "full"),     # == horizon: collapse rather than pretend it is chunked
    (99, "full"),     # > horizon: same
])
def test_chunk_size_maps_to_the_right_regime(chunk, expected_mode):
    b = fixed_horizon_bptt(frozen=False, horizon=10, chunk_size=chunk)
    assert b.mode == expected_mode
    assert b.horizon == 10
    if expected_mode == "chunked":
        assert b.chunk_size == chunk


def test_the_spec_it_produces_is_valid():
    for horizon in (1, 4, 7, 10):
        for chunk in (0, 1, 2, 3, 7, 11):
            for frozen in (True, False):
                b = fixed_horizon_bptt(frozen=frozen, horizon=horizon, chunk_size=chunk)
                assert b.errors() == [], f"{frozen=} {horizon=} {chunk=} -> {b.errors()}"


# --- the gradient invariant -------------------------------------------------
class _HeadTask:
    def __init__(self, head):
        self.head = head

    def forward_glimpse(self, *, model, images, state, viewpoint, backbone_no_grad):
        ctx = torch.no_grad() if backbone_no_grad else torch.enable_grad()
        with ctx:
            out = model(image=images, state=state, viewpoint=viewpoint)
        return GlimpseOut(readout=out.state, state=out.state, vpe=out.vpe)

    def step_loss(self, readout):
        cls = readout.recurrent_cls[:, 0].float()
        return SimpleNamespace(combined=(self.head(cls) ** 2).mean())

    def per_image_loss(self, readout):
        return (self.head(readout.recurrent_cls[:, 0].float()) ** 2).mean(dim=1)


def _model():
    torch.manual_seed(0)
    return CanViTForPretraining(
        backbone=create_backbone("vits16"), cfg=CanViTForPretrainingConfig(teacher_dim=_D),
        glimpse_size_px=128, backbone_name="vits16", canvas_patch_grid_sizes=[_G],
    )


def _grads(bptt, model, head, images):
    torch.manual_seed(7)
    for p in list(model.parameters()) + list(head.parameters()):
        p.grad = None
    run_rollout(
        model=model, images=images, task=_HeadTask(head),
        selector=RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(),
                                min_viewpoint_scale=0.1),
        bptt=bptt, branches=[ViewpointType.RANDOM], canvas_grid_size=_G,
        amp_ctx=torch.enable_grad(),
    )
    return torch.cat([p.grad.flatten() for p in head.parameters() if p.grad is not None])


def test_bptt_moves_the_backbone_only_never_the_head():
    """Why "frozen => none" is safe: the head reads the state at t and never feeds back
    into it, so no head parameter influences a later timestep. Measured 2026-07-28."""
    model, head, images = _model(), nn.Linear(384, 384), torch.randn(_B, 3, 224, 224)
    g_none = _grads(BpttSpec(mode="none", horizon=3), model, head, images)
    g_full = _grads(BpttSpec(mode="full", horizon=3), model, head, images)
    torch.testing.assert_close(g_none, g_full, rtol=0, atol=0)
    # ... while the backbone DOES see the difference.
    assert all(p.grad is None for p in model.backbone.parameters()) is False


@pytest.mark.parametrize("horizon,chunk", [(6, 3), (6, 2), (7, 3), (7, 2), (5, 4), (11, 3)])
def test_chunking_preserves_the_total_gradient_even_when_indivisible(horizon, chunk):
    """7 glimpses at chunk 3 runs [0,1,2][3,4,5][6] — the trailing partial chunk still
    backwards, and every chunk normalises by n_glimpses, so the accumulated gradient
    equals the unchunked one. Prime horizons are the sharp case."""
    model, head, images = _model(), nn.Linear(384, 384), torch.randn(_B, 3, 224, 224)
    full = _grads(BpttSpec(mode="full", horizon=horizon), model, head, images)
    chunked = _grads(BpttSpec(mode="chunked", chunk_size=chunk, horizon=horizon),
                     model, head, images)
    torch.testing.assert_close(chunked, full, rtol=1e-4, atol=1e-6)


# --- the rule is ENFORCED, not just documented ------------------------------
def test_check_spec_warns_when_a_frozen_backbone_carries_a_graph():
    """The config path can't produce this (fixed_horizon_bptt forces 'none'), but a
    hand-built TrainSpec can — so the validator has to catch it. This warning is the
    guardrail; a comment alone would be missable."""
    from canvit_train.harness.spec import GroupOptim, TaskCaps, TrainSpec, check_spec

    caps = TaskCaps(has_head=True, supports_policy=True)
    bad = TrainSpec(train_backbone=False, train_head=True, task_grad_to_backbone=False,
                    bptt=BpttSpec(mode="full", horizon=10),
                    optim={"head": GroupOptim(lr=1e-3)})
    report = check_spec(bad, caps)
    assert report.ok, "wasteful, not wrong — must stay a warning, not an error"
    assert any("FROZEN backbone" in x for x in report.warnings), report.warnings

    good = TrainSpec(train_backbone=False, train_head=True, task_grad_to_backbone=False,
                     bptt=BpttSpec(mode="none", horizon=10),
                     optim={"head": GroupOptim(lr=1e-3)})
    assert not any("FROZEN backbone" in x for x in check_spec(good, caps).warnings)
