"""Per-group optimizer + scheduler builder for the unified harness (design §7 D-E).

Each trainable module group (backbone / head / policy) gets its OWN lr, weight
decay, and LR-schedule shape — the generality the new joint configs need (e.g. a
low-lr backbone + high-lr head + policy@2e-4 in one run). One AdamW with one param
group per trainable module; one ``LambdaLR`` whose per-group ``lr_lambda`` realizes
that group's :class:`ScheduleSpec`.

Not parity-gated: the distill parity probe runs at constant LR (no scheduler), so
the schedule math here is validated by its own unit tests + the GPU gate, not the
digest.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from canvit_pretrain.harness.spec import Module, ScheduleSpec, TrainSpec


def _lr_lambda(sched: ScheduleSpec, base_lr: float) -> Callable[[int], float]:
    """Multiplicative factor on ``base_lr`` at a given step, per the schedule shape.
    Warmup ramps from a start factor to 1.0 over ``warmup_steps`` (relative
    ``warmup_lr_ratio`` wins over an absolute ``start_lr``); then constant or cosine."""
    warm = sched.warmup_steps
    if sched.warmup_lr_ratio is not None:
        start_factor = sched.warmup_lr_ratio
    elif sched.start_lr is not None:
        start_factor = sched.start_lr / base_lr
    else:
        start_factor = 1.0 / max(warm, 1)

    def fn(step: int) -> float:
        if warm > 0 and step < warm:
            return start_factor + (1.0 - start_factor) * (step / warm)
        if sched.kind == "warmup_constant":
            return 1.0
        if sched.kind == "warmup_cosine":
            assert sched.total_steps is not None
            prog = min(1.0, (step - warm) / max(1, sched.total_steps - warm))
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        if sched.kind == "warmup_onecycle":
            raise NotImplementedError(
                "warmup_onecycle is not yet wired into the harness builder; use warmup_cosine, "
                "or port ADE20K's WarmupOneCycleLR when its exact anneal shape is needed."
            )
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
        groups.append({"params": params, "lr": go.lr, "weight_decay": go.weight_decay})
        lambdas.append(_lr_lambda(go.schedule, go.lr))
    if not groups:
        raise ValueError("no trainable groups to optimize")
    opt = AdamW(groups)
    sched = LambdaLR(opt, lr_lambda=lambdas)
    return opt, sched


__all__ = ["build_optimizer_and_scheduler"]
