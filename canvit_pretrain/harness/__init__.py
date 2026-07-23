"""Task-neutral training harness for the unified CanViT trainer.

Design: ``unification_docs/07-unified-harness-design.md``. One harness provides
the shared, task-agnostic services (rollout engine, grad routing, optimizer/
scheduler construction, checkpoint I/O, DDP sync, validation cadence); the three
tasks (distill / ade20k / in1k) are equal peers that plug in via the ``Task``
seam. Nothing in this package knows about DINOv3, segmentation, or classification.

Stage status (see the design doc):
  * spec.py   — TrainSpec / BpttSpec / GroupOptim + validation  [stage 1/2]
  * rollout.py, loop.py, optim.py, ddp.py, checkpoint.py        [in progress]
"""

from canvit_pretrain.harness.spec import (
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
