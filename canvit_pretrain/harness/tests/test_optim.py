"""CPU unit tests for the per-group optimizer/scheduler builder (design D-E)."""

import math

import pytest
import torch
from torch import nn

from canvit_pretrain.harness.optim import build_optimizer_and_scheduler
from canvit_pretrain.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec


def _spec(**optim) -> TrainSpec:
    # a finetune-shaped spec (backbone + head trainable), optim groups injected
    return TrainSpec(
        train_backbone=True, train_head=True, task_grad_to_backbone=True,
        bptt=BpttSpec(mode="full", horizon=4), optim=optim,
    )


def _params(n=2):
    return [nn.Parameter(torch.zeros(3)) for _ in range(n)]


def test_per_group_lrs_and_wd():
    spec = _spec(
        backbone=GroupOptim(lr=1e-4, weight_decay=0.05),
        head=GroupOptim(lr=1e-3, weight_decay=0.0),
    )
    opt, _ = build_optimizer_and_scheduler(spec, {"backbone": _params(), "head": _params()})
    assert len(opt.param_groups) == 2
    lrs = {g["lr"] for g in opt.param_groups}
    assert lrs == {1e-4, 1e-3}
    wds = sorted(g["weight_decay"] for g in opt.param_groups)
    assert wds == [0.0, 0.05]


def test_different_schedule_shapes_per_group():
    # backbone: warmup->cosine decay; head: warmup->constant. Distinct shapes, one optimizer.
    spec = _spec(
        backbone=GroupOptim(lr=1.0, schedule=ScheduleSpec(
            kind="warmup_cosine", warmup_steps=2, total_steps=10, warmup_lr_ratio=0.5)),
        head=GroupOptim(lr=1.0, schedule=ScheduleSpec(
            kind="warmup_constant", warmup_steps=2, warmup_lr_ratio=0.5)),
    )
    opt, sched = build_optimizer_and_scheduler(spec, {"backbone": _params(), "head": _params()})
    bb, hd = 0, 1  # group order == trainable_modules() order (backbone, head)

    # step 0: both in warmup at the start factor (0.5 * base 1.0).
    assert math.isclose(opt.param_groups[bb]["lr"], 0.5, rel_tol=1e-6)
    assert math.isclose(opt.param_groups[hd]["lr"], 0.5, rel_tol=1e-6)

    lrs_bb, lrs_hd = [], []
    for _ in range(10):
        sched.step()
        lrs_bb.append(opt.param_groups[bb]["lr"])
        lrs_hd.append(opt.param_groups[hd]["lr"])

    # head holds at peak after warmup; backbone decays below peak (cosine).
    assert math.isclose(lrs_hd[-1], 1.0, rel_tol=1e-6)
    assert lrs_bb[-1] < 0.5  # cosine has annealed well below peak by the end
    assert lrs_bb[-1] < lrs_hd[-1]


def test_missing_optim_group_raises():
    spec = _spec(head=GroupOptim(lr=1e-3))  # backbone trainable but no optim entry
    with pytest.raises(ValueError, match="missing group 'backbone'"):
        build_optimizer_and_scheduler(spec, {"backbone": _params(), "head": _params()})


def test_missing_params_raises():
    spec = _spec(backbone=GroupOptim(lr=1e-4), head=GroupOptim(lr=1e-3))
    with pytest.raises(ValueError, match="param_groups missing 'head'"):
        build_optimizer_and_scheduler(spec, {"backbone": _params()})


def test_onecycle_not_yet_supported():
    # onecycle with no warmup evaluates the (unsupported) anneal branch at step 0,
    # so it raises when LambdaLR is constructed inside the builder.
    spec = _spec(
        backbone=GroupOptim(lr=1e-4),
        head=GroupOptim(lr=1e-3, schedule=ScheduleSpec(kind="warmup_onecycle", total_steps=10)),
    )
    with pytest.raises(NotImplementedError, match="warmup_onecycle"):
        build_optimizer_and_scheduler(spec, {"backbone": _params(), "head": _params()})
