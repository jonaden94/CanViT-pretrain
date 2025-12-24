"""Training step with gradient accumulation over multiple trajectories."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from avp_vit import ActiveCanViT, GlimpseOutput, gram_mse
from canvit import Viewpoint

from .viewpoint import Viewpoint as TrainViewpoint, random_viewpoint


class TrajectoryMetrics(NamedTuple):
    """Metrics from a single trajectory type."""

    loss: Tensor
    scene_cos: Tensor
    cls_cos: Tensor


class StepMetrics(NamedTuple):
    """Metrics from a full training step (all trajectories)."""

    total_loss: Tensor
    full_full: TrajectoryMetrics
    full_rand: TrajectoryMetrics
    random: TrajectoryMetrics  # averaged over n_random_trajectories


@dataclass
class TrajectoryAcc:
    """Accumulated state during trajectory.

    Stores accumulated losses and last predictions (for metrics without recomputation).
    """

    scene_loss: Tensor
    cls_loss: Tensor
    gram_loss: Tensor | None
    last_scene_pred: Tensor
    last_cls_pred: Tensor


def make_full_trajectory(batch_size: int, length: int, device: torch.device) -> list[TrainViewpoint]:
    """Full → Full trajectory."""
    full = TrainViewpoint.full_scene(batch_size=batch_size, device=device)
    return [full] * length


def make_full_random_trajectory(
    batch_size: int, length: int, device: torch.device, min_scale: float
) -> list[TrainViewpoint]:
    """Full → Random trajectory."""
    full = TrainViewpoint.full_scene(batch_size=batch_size, device=device)
    rest = [random_viewpoint(batch_size, device, min_scale=min_scale) for _ in range(length - 1)]
    return [full] + rest


def make_random_trajectory(
    batch_size: int, length: int, device: torch.device, min_scale: float
) -> list[TrainViewpoint]:
    """Random trajectory."""
    return [random_viewpoint(batch_size, device, min_scale=min_scale) for _ in range(length)]


def training_step(
    *,
    model: ActiveCanViT,
    images: Tensor,
    scene_target: Tensor,
    cls_target: Tensor,
    glimpse_size_px: int,
    canvas_grid_size: int,
    trajectory_length: int,
    n_random_trajectories: int,
    min_viewpoint_scale: float,
    compute_gram: bool,
    gram_loss_weight: float,
    amp_ctx: AbstractContextManager,
) -> StepMetrics:
    """Run training step with gradient accumulation over trajectories.

    Runs 2 canonical trajectories (full→full, full→random) plus n_random_trajectories.
    Accumulates gradients with proper scaling. Does NOT call optimizer.step().

    Returns metrics for logging.
    """
    device = images.device
    batch_size = images.shape[0]
    n_trajectories = 2 + n_random_trajectories

    def run_trajectory(viewpoints: list[TrainViewpoint]) -> TrajectoryMetrics:
        """Run one trajectory, backward with accumulation, return metrics."""
        # Convert TrainViewpoint to canvit Viewpoint
        canvit_viewpoints: list[Viewpoint] = [
            Viewpoint(centers=vp.centers, scales=vp.scales) for vp in viewpoints
        ]

        canvas = model.init_canvas(batch_size=batch_size, canvas_grid_size=canvas_grid_size)

        with amp_ctx:
            zero = scene_target.new_zeros(())

            def init_fn(_canvas: Tensor, _cls: Tensor) -> TrajectoryAcc:
                # Predictions are placeholders - overwritten by first step_fn call
                return TrajectoryAcc(
                    scene_loss=zero,
                    cls_loss=zero,
                    gram_loss=None,
                    last_scene_pred=scene_target.new_empty(0),
                    last_cls_pred=cls_target.new_empty(0),
                )

            def step_fn(acc: TrajectoryAcc, out: GlimpseOutput, _vp: Viewpoint) -> TrajectoryAcc:
                scene_pred = model.predict_teacher_scene(out.canvas)
                cls_pred = model.predict_teacher_cls(out.cls, out.canvas)
                scene_loss = F.mse_loss(scene_pred, scene_target)
                cls_loss = F.mse_loss(cls_pred, cls_target)
                step_gram = gram_mse(scene_pred, scene_target) if compute_gram else None

                new_gram: Tensor | None
                if acc.gram_loss is None:
                    new_gram = step_gram
                elif step_gram is not None:
                    new_gram = acc.gram_loss + step_gram
                else:
                    new_gram = acc.gram_loss

                return TrajectoryAcc(
                    scene_loss=acc.scene_loss + scene_loss,
                    cls_loss=acc.cls_loss + cls_loss,
                    gram_loss=new_gram,
                    last_scene_pred=scene_pred,
                    last_cls_pred=cls_pred,
                )

            acc, _final_canvas, _final_cls = model.forward_reduce(
                image=images,
                viewpoints=canvit_viewpoints,  # type: ignore[arg-type]
                glimpse_size_px=glimpse_size_px,
                canvas_grid_size=canvas_grid_size,
                init_fn=init_fn,
                step_fn=step_fn,
                canvas=canvas,
            )

            # Compute total loss (averaged over trajectory steps)
            n_steps = len(viewpoints)
            traj_loss = acc.scene_loss / n_steps + acc.cls_loss / n_steps
            if acc.gram_loss is not None:
                traj_loss = traj_loss + gram_loss_weight * acc.gram_loss / n_steps

            assert torch.isfinite(traj_loss), "NaN/Inf loss"
            (traj_loss / n_trajectories).backward()

        # Metrics from last predictions (already computed, no recomputation!)
        with torch.no_grad():
            scene_cos = F.cosine_similarity(acc.last_scene_pred, scene_target, dim=-1).mean()
            cls_cos = F.cosine_similarity(acc.last_cls_pred, cls_target, dim=-1).mean()

        return TrajectoryMetrics(loss=traj_loss.detach(), scene_cos=scene_cos, cls_cos=cls_cos)

    # Full → Full
    full_full = run_trajectory(make_full_trajectory(batch_size, trajectory_length, device))

    # Full → Random
    full_rand = run_trajectory(
        make_full_random_trajectory(batch_size, trajectory_length, device, min_viewpoint_scale)
    )

    # Random trajectories (averaged)
    random_loss_sum = torch.zeros((), device=device)
    random_scene_cos_sum = torch.zeros((), device=device)
    random_cls_cos_sum = torch.zeros((), device=device)

    for _ in range(n_random_trajectories):
        m = run_trajectory(
            make_random_trajectory(batch_size, trajectory_length, device, min_viewpoint_scale)
        )
        random_loss_sum = random_loss_sum + m.loss
        random_scene_cos_sum = random_scene_cos_sum + m.scene_cos
        random_cls_cos_sum = random_cls_cos_sum + m.cls_cos

    random_metrics = TrajectoryMetrics(
        loss=random_loss_sum / n_random_trajectories,
        scene_cos=random_scene_cos_sum / n_random_trajectories,
        cls_cos=random_cls_cos_sum / n_random_trajectories,
    )

    total_loss = (full_full.loss + full_rand.loss + random_loss_sum) / n_trajectories

    return StepMetrics(
        total_loss=total_loss,
        full_full=full_full,
        full_rand=full_rand,
        random=random_metrics,
    )
