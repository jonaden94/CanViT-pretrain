"""Task-neutral training harness: the orchestrator AND everything shared between tasks.

Design: ``unification_docs/07-unified-harness-design.md``. The three tasks
(``distill`` / ``ade20k`` / ``in1k``) are equal peers that plug in via the task seam;
nothing in this package knows about DINOv3, segmentation, or classification.

**Orchestration** — run.py (entry point), cli.py (tyro CLI + preset→spec), loop.py,
rollout.py (the glimpse rollout engine), spec.py (TrainSpec/BpttSpec + validation),
optim.py, checkpoint.py, ddp.py, policy.py, eval_viewpoints.py.

**Shared primitives** — used by two or more tasks, so they live here rather than in any
one task's folder: config.py (``FoveatedScaleConfig``, ``JointPolicyConfig``),
viewpoint.py, selector.py, rl.py, joint.py, scheduler.py, ema.py, dist.py, tracker.py,
schedule.py (shard schedule; distill + in1k), utils.py, viz/ (the task-agnostic PCA /
figure-I/O / metric leaves).

The rule: **shared lives here, task-specific lives in that task's folder.** Until
2026-07-31 the shared primitives sat in a folder called ``train/`` alongside distill's
own code, because distill was once the whole repo — see
``unification_docs/18-package-restructure.md``.
"""

from canvit_train.harness.spec import (
    BpttSpec,
    GroupOptim,
    ScheduleSpec,
    SpecReport,
    TaskCaps,
    TrainSpec,
    check_spec,
)

__all__ = [
    "BpttSpec",
    "GroupOptim",
    "ScheduleSpec",
    "SpecReport",
    "TaskCaps",
    "TrainSpec",
    "check_spec",
]
