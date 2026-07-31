"""Per-group optimizer + scheduler builder for the unified harness (design §7 D-E).

Each trainable module group (backbone / head / policy) gets its OWN lr, weight
decay, and LR-schedule shape — the generality the new joint configs need (e.g. a
low-lr backbone + high-lr head + policy@2e-4 in one run). One AdamW with one param
group per trainable module; one ``LambdaLR`` whose per-group ``lr_lambda`` realizes
that group's :class:`ScheduleSpec`.

Not parity-gated: the distill parity probe runs at constant LR (no scheduler), so
the schedule math here is validated by its own unit tests + the GPU gate, not the
digest. Those unit tests compare step for step against the REAL schedulers the
standalone entry points build (ADE20K's ``WarmupOneCycleLR``, in1k's
``warmup_cosine_scheduler``) rather than against hard-coded expected values.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from canvit_train.harness.spec import Module, ScheduleSpec, TrainSpec


def _onecycle_factor(sched: ScheduleSpec, step: int) -> float:
    """ADE20K's ``WarmupOneCycleLR`` shape, as a multiplicative factor on the peak lr.

    Ported from ``dinov3.eval.segmentation.schedulers.WarmupOneCycleLR`` under exactly
    the parametrization ``canvit_train.ade20k.data.make_optimizer_and_scheduler``
    fixes: ``anneal_strategy='cos'`` and ``final_div_factor=inf`` (so ``min_lr == 0``,
    which also zeroes the original's ``max_lr / final_div_factor`` warmup offset).
    Momentum is not touched (``use_beta1=False, update_momentum=False``), so the whole
    scheduler reduces to this one lr factor. Checked step-for-step against the real
    scheduler in ``tests/test_optim.py`` rather than by eye.

    Two quirks of the original are reproduced deliberately:

    * warmup is the plain linear ramp ``ratio -> 1`` scaled by an extra
      ``(1 - step/total_steps)`` decay — the original writes it as ``1 - k`` with
      ``k = (1 - step/warm)(1 - ratio)``, which expands to ``ratio + (1-ratio)*step/warm``.
    * the anneal's progress is ``(step + 1) / total_steps`` measured from step 0, NOT
      from the end of warmup, so the lr never quite reaches the peak.
    """
    total = sched.total_steps
    assert total is not None  # ScheduleSpec.errors() rejects onecycle without it
    warm = sched.warmup_steps
    ratio = sched.warmup_lr_ratio or 0.0  # the original tests `if self.warmup_ratio:`
    if step < warm:
        if ratio:
            return (ratio + (1.0 - ratio) * step / warm) * (1.0 - step / total)
        return (1.0 + math.cos(math.pi * step / total)) / 2.0 * (step / warm)
    return (1.0 + math.cos(math.pi * (step + 1) / total)) / 2.0


def _lr_lambda(sched: ScheduleSpec, base_lr: float) -> Callable[[int], float]:
    """Multiplicative factor on ``base_lr`` at a given step, per the schedule shape.
    Warmup ramps from a start factor to 1.0 over ``warmup_steps`` (relative
    ``warmup_lr_ratio`` wins over an absolute ``start_lr``); then constant or cosine.
    ``warmup_onecycle`` has its own warmup shape and is handled wholesale."""
    warm = sched.warmup_steps
    if sched.warmup_lr_ratio is not None:
        start_factor = sched.warmup_lr_ratio
    elif sched.start_lr is not None:
        start_factor = sched.start_lr / base_lr
    else:
        start_factor = 1.0 / max(warm, 1)

    def fn(step: int) -> float:
        if sched.kind == "warmup_onecycle":
            return _onecycle_factor(sched, step)
        if warm > 0 and step < warm:
            return start_factor + (1.0 - start_factor) * (step / warm)
        if sched.kind == "warmup_constant":
            return 1.0
        if sched.kind == "warmup_cosine":
            assert sched.total_steps is not None
            prog = min(1.0, (step - warm) / max(1, sched.total_steps - warm))
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        raise ValueError(f"unknown schedule kind: {sched.kind!r}")

    return fn


def build_optimizer_and_scheduler(
    spec: TrainSpec, param_groups: dict[Module, Sequence[nn.Parameter]],
) -> tuple[Optimizer, LRScheduler]:
    """Build the AdamW + LambdaLR for exactly the trainable module groups in ``spec``.

    ``param_groups`` maps each trainable module name to its parameters (the caller —
    the loop — collects them from the model/head/scorer). Every trainable module must
    have both a ``spec.optim[...]`` entry (validated) and a ``param_groups[...]`` entry.
    """
    modules = spec.trainable_modules()
    groups = []
    lambdas: list[Callable[[int], float]] = []
    for m in modules:
        go = spec.optim.get(m)
        if go is None:
            raise ValueError(f"spec.optim missing group {m!r} (trainable module without optimizer settings)")
        if m not in param_groups:
            raise ValueError(f"param_groups missing {m!r} (trainable module with no parameters supplied)")
        params = list(param_groups[m])
        groups.append({"params": params, "lr": go.lr, "weight_decay": go.weight_decay,
                       "betas": go.betas})
        lambdas.append(_lr_lambda(go.schedule, go.lr))
    if not groups:
        raise ValueError("no trainable groups to optimize")
    opt = AdamW(groups)
    sched = LambdaLR(opt, lr_lambda=lambdas)
    return opt, sched


__all__ = ["build_optimizer_and_scheduler"]
