"""CPU unit tests for the per-group optimizer/scheduler builder (design D-E)."""

import math

import pytest
import torch
from torch import nn

from canvit_train.harness.optim import build_optimizer_and_scheduler
from canvit_train.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec


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


# --- schedule fidelity: compare against the REAL schedulers the standalone
# entry points build, not against a hand-derived expectation. A fixture that
# encodes my own belief about the shape cannot catch a drifted port.

def _harness_lrs(sched_spec: ScheduleSpec, lr: float, n: int) -> list[float]:
    spec = _spec(head=GroupOptim(lr=lr, schedule=sched_spec))
    spec = TrainSpec(train_head=True, task_grad_to_backbone=False,
                     bptt=BpttSpec(mode="none", horizon=4), optim=spec.optim)
    opt, sched = build_optimizer_and_scheduler(spec, {"head": _params()})
    out = []
    for _ in range(n):
        out.append(opt.param_groups[0]["lr"])
        sched.step()
    return out


def _reference_lrs(make, lr: float, n: int) -> list[float]:
    opt = torch.optim.AdamW(_params(), lr=lr)
    sched = make(opt)
    out = []
    for _ in range(n):
        out.append(opt.param_groups[0]["lr"])
        sched.step()
    return out


@pytest.mark.parametrize("warmup_steps,warmup_lr_ratio", [(15, 1e-6), (15, 0.0), (0, 1e-6)])
def test_onecycle_matches_ade20k_reference_scheduler(warmup_steps, warmup_lr_ratio):
    """``warmup_onecycle`` must equal ADE20K's AdamW + WarmupOneCycleLR step for step."""
    from canvit_train.ade20k.data import make_optimizer_and_scheduler

    lr, total = 3e-4, 400
    got = _harness_lrs(
        ScheduleSpec(kind="warmup_onecycle", warmup_steps=warmup_steps, total_steps=total,
                     warmup_lr_ratio=warmup_lr_ratio or None), lr, total)
    # make_optimizer_and_scheduler builds its own AdamW, so drive that one directly
    # rather than through _reference_lrs.
    ref_opt, ref_sched = make_optimizer_and_scheduler(
        _params(), lr=lr, weight_decay=0.0, max_steps=total,
        warmup_steps=warmup_steps, warmup_lr_ratio=warmup_lr_ratio)
    want = []
    for _ in range(total):
        want.append(ref_opt.param_groups[0]["lr"])
        ref_sched.step()

    assert len(got) == len(want) == total
    for step, (g, w) in enumerate(zip(got, want, strict=True)):
        assert math.isclose(g, w, rel_tol=1e-9, abs_tol=1e-15), f"step {step}: {g} != {w}"


def test_warmup_cosine_matches_in1k_reference_scheduler():
    """``warmup_cosine`` must equal in1k's AdamW + warmup_cosine_scheduler step for step."""
    from canvit_train.harness.scheduler import warmup_cosine_scheduler

    lr, total, warm = 3e-4, 400, 20
    start_lr = lr * 1e-6
    got = _harness_lrs(
        ScheduleSpec(kind="warmup_cosine", warmup_steps=warm, total_steps=total,
                     start_lr=start_lr), lr, total)
    want = _reference_lrs(
        lambda opt: warmup_cosine_scheduler(opt, warm, total, lr, start_lr=start_lr), lr, total)
    for step, (g, w) in enumerate(zip(got, want, strict=True)):
        assert math.isclose(g, w, rel_tol=1e-9, abs_tol=1e-15), f"step {step}: {g} != {w}"
