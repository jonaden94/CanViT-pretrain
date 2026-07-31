"""Optimizer, LR schedule, and weight averaging.

- ``build.py``     — ``build_optimizer_and_scheduler``: per-group optimizers from the spec
- ``scheduler.py`` — the warmup/constant/cosine/onecycle LR schedules
- ``ema.py``       — exponential moving average of weights

``build``'s public API is re-exported here, so ``from canvit_train.harness.optim import
build_optimizer_and_scheduler`` works exactly as it did when this was a single module.
"""

from canvit_train.harness.optim.build import build_optimizer_and_scheduler

__all__ = ["build_optimizer_and_scheduler"]
