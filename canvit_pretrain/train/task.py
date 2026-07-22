"""Task seam for the unified harness (unification master plan §4.2).

A Task owns the per-glimpse objective: what the model output is scored against
and which loss terms are active. P1 ships DistillTask only — a byte-for-byte
extraction of training_step's historical ``compute_loss`` closure. The ADE20K
and IN1k tasks (with their own targets/heads/data) arrive in P2/P5; the full
interface (build_data/build_targets/reward) grows with them.
"""

from dataclasses import dataclass
from typing import NamedTuple, Protocol

import torch
import torch.nn.functional as F
from canvit_pytorch import CanViTOutput
from torch import Tensor


class LossOutput(NamedTuple):
    """Output from Task.step_loss - individual losses + combined mean."""

    scene_patches_loss: Tensor
    scene_cls_loss: Tensor
    combined: Tensor  # sum of active losses
    scene_pred: Tensor  # for cosine similarity metrics
    cls_pred: Tensor


class Task(Protocol):
    def step_loss(self, out: CanViTOutput) -> LossOutput: ...


@dataclass
class DistillTask:
    """DINOv3 feature-regression: MSE of the model's scene/cls predictions
    against the (standardized) teacher targets. Historical pretrain objective."""

    scene_target: Tensor
    cls_target: Tensor
    enable_scene_patches_loss: bool
    enable_scene_cls_loss: bool

    def step_loss(self, out: CanViTOutput) -> LossOutput:
        # `out` is a CanViTForPretrainingOutput coming through the DDP-wrapped
        # forward — scene_pred / cls_pred were produced INSIDE that forward,
        # so head-param gradients are part of the autograd graph DDP's
        # Reducer instruments. Calling `core_model.predict_*` here would
        # bypass the DDP wrapper and skip AllReduce on the head gradients
        # (manifests as √N grad-norm scaling on the heads in N-GPU runs).
        device = self.scene_target.device
        scene_pred = out.scene_pred  # type: ignore[attr-defined]
        cls_pred = out.cls_pred  # type: ignore[attr-defined]

        scene_patches_loss = torch.zeros((), device=device)
        scene_cls_loss = torch.zeros((), device=device)

        if self.enable_scene_patches_loss:
            scene_patches_loss = F.mse_loss(scene_pred, self.scene_target)
        if self.enable_scene_cls_loss:
            scene_cls_loss = F.mse_loss(cls_pred, self.cls_target)

        active: list[Tensor] = []
        if self.enable_scene_patches_loss:
            active.append(scene_patches_loss)
        if self.enable_scene_cls_loss:
            active.append(scene_cls_loss)
        assert len(active) > 0, "At least one loss must be enabled"
        combined = torch.stack(active).sum()

        return LossOutput(
            scene_patches_loss=scene_patches_loss,
            scene_cls_loss=scene_cls_loss,
            combined=combined,
            scene_pred=scene_pred,
            cls_pred=cls_pred,
        )

    def per_image_loss(self, out: CanViTOutput) -> Tensor:
        """Per-image combined distill MSE [B] — the policy reward's raw material
        (master plan §3: ``r_t = (L_t - L_{t+1}) / L_t``). Same active terms as
        ``step_loss`` but reduced per image instead of to a scalar. Caller detaches."""
        scene_pred = out.scene_pred  # type: ignore[attr-defined]  # [B, G^2, teacher_dim]
        cls_pred = out.cls_pred  # type: ignore[attr-defined]  # [B, teacher_dim]
        parts: list[Tensor] = []
        if self.enable_scene_patches_loss:
            parts.append((scene_pred - self.scene_target).pow(2).mean(dim=(1, 2)))
        if self.enable_scene_cls_loss:
            parts.append((cls_pred - self.cls_target).pow(2).mean(dim=1))
        assert parts, "At least one loss must be enabled"
        return torch.stack(parts).sum(dim=0)
