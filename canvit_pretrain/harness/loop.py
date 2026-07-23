"""The task-neutral step-based training driver (design §1, §Loop).

ONE loop for all three tasks. It owns the task-agnostic mechanics — per-step
rollout → grad-clip → optimizer/scheduler step, plus logging / checkpoint / eval
cadence — and delegates everything task-specific to the ``Task`` seam (data via the
caller's iterator, per-batch targets via ``task.bind``, metrics via ``on_eval``).
There is only a step loop (owner decision: step-based only; IN1k epochs are derived
from its ``with_epoch`` shard count upstream).

This is deliberately thin: the heavy, task-specific machinery (distill's teacher /
normalizer / webdataset-resume, ade20k's mIoU eval, in1k's top-k) lives in each
``Task``, never here (design §1 "keep the outer loop thin"). DDP grad-sync for
in-rollout modules (backbone/scorer) is the manual-AllReduce path (design §9); it is
wired here for the scorer and completed in the DDP stage.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from canvit_pretrain.harness.checkpoint import save_checkpoint
from canvit_pretrain.harness.rollout import run_rollout
from canvit_pretrain.harness.spec import TrainSpec
from canvit_pretrain.train.viewpoint import ViewpointType

log = logging.getLogger(__name__)


def apply_requires_grad(*, model: Any, head: Any, joint: Any, spec: TrainSpec) -> None:
    """Set ``requires_grad`` from the spec — the harness, not the task, decides what
    trains (design §3.1). ``model`` owns the backbone (+ head as a submodule); ``head``
    is the same head module (or None); ``joint`` owns the scorer."""
    core = getattr(model, "module", model)
    # Freeze/unfreeze the CanViT trunk. For task wrappers the trunk is core.canvit;
    # for the distill core model it is core itself (its heads are submodules).
    trunk = getattr(core, "canvit", core)
    trunk.requires_grad_(spec.train_backbone)
    if not spec.train_backbone:
        trunk.eval()
    if head is not None:
        head.requires_grad_(spec.train_head)
    if joint is not None:
        joint.scorer.requires_grad_(spec.train_policy)


def run_training_loop(
    *,
    task: Any,                       # Task: batch_images(batch, device), bind(batch, device, model=, head=)
    model: Any,
    head: Any,
    optimizer: Any,
    scheduler: Any,
    selector: Any,
    spec: TrainSpec,
    branches: list[ViewpointType],
    canvas_grid: int,
    device: torch.device,
    train_batches: Iterator,
    n_steps: int,
    start_step: int = 0,
    joint: Any | None = None,
    amp_ctx: Any | None = None,
    grad_clip: float = 1.0,
    is_dist: bool = False,
    task_name: str = "task",
    model_config: dict | None = None,
    metadata: dict | None = None,
    log_every: int = 20,
    ckpt_dir: Path | None = None,
    ckpt_every: int = 0,             # 0 => only the end-of-run checkpoint
    eval_every: int = 0,             # 0 => no periodic eval
    on_log: Callable[[int, dict], None] | None = None,
    on_eval: Callable[[int], dict] | None = None,
) -> dict:
    """Run ``n_steps`` of training and return the last step's metrics. ``train_batches``
    yields task-native batches; ``task.bind`` turns each into the per-glimpse
    ``RolloutTask`` the engine runs."""
    amp_ctx = amp_ctx or nullcontext()
    trainable = [p for g in optimizer.param_groups for p in g["params"]]
    last: dict = {}

    def _save(step: int) -> None:
        if ckpt_dir is not None:
            save_checkpoint(
                ckpt_dir / f"step-{step}.pt", model=model, optimizer=optimizer, scheduler=scheduler,
                step=step, task_name=task_name, spec=spec, model_config=model_config,
                metadata=metadata, joint=joint,
            )

    for step in range(start_step, start_step + n_steps):
        batch = next(train_batches)
        images = task.batch_images(batch, device)
        bound = task.bind(batch, device, model=model, head=head)

        optimizer.zero_grad()
        result = run_rollout(
            model=model, images=images, task=bound, selector=selector, bptt=spec.bptt,
            branches=branches, canvas_grid_size=canvas_grid, amp_ctx=amp_ctx,
            task_weight=spec.task_weight, joint=joint,
        )
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        if joint is not None and is_dist:  # scorer is not DDP-wrapped (design §9)
            joint.allreduce_grads()
        optimizer.step()
        scheduler.step()

        last = {"step": step, "total_loss": float(result.total_loss), "n_glimpses": result.n_glimpses}
        if result.policy_metrics is not None:
            last["reward_frac"] = float(result.policy_metrics["reward_frac"])
            last["policy_loss"] = float(result.policy_metrics["policy_loss"])
        if on_log is not None and step % log_every == 0:
            on_log(step, last)
        if on_eval is not None and eval_every and step > 0 and step % eval_every == 0:
            last["eval"] = on_eval(step)
        if ckpt_every and step > start_step and step % ckpt_every == 0:
            _save(step)

    _save(start_step + n_steps)
    return last


__all__ = ["apply_requires_grad", "run_training_loop"]
