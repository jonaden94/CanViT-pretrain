"""The glimpse rollout: how a batch becomes a sequence of viewpoints and losses.

- ``engine.py``          — ``run_rollout``, the task-agnostic rollout loop (BPTT modes,
                           branch handling, per-glimpse loss accumulation)
- ``viewpoint.py``       — viewpoint types, pixel-box geometry, eval viewpoint grids
- ``selector.py``        — which viewpoint comes next (random / policy / mixture)
- ``eval_viewpoints.py`` — viewpoint schedules used at validation and deploy time

``engine``'s public API is re-exported here, so ``from canvit_train.harness.rollout
import run_rollout`` works exactly as it did when this was a single module.
"""

from canvit_train.harness.rollout.engine import (
    BranchResult,
    GlimpseLoss,
    GlimpseOut,
    RolloutResult,
    RolloutTask,
    RolloutViz,
    TaskLoss,
    run_rollout,
    sample_n_glimpses,
)

__all__ = [
    "BranchResult",
    "GlimpseLoss",
    "GlimpseOut",
    "RolloutResult",
    "RolloutTask",
    "RolloutViz",
    "TaskLoss",
    "run_rollout",
    "sample_n_glimpses",
]
