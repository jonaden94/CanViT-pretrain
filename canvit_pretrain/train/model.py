"""Model creation and loading utilities."""

import logging
from typing import NamedTuple

import torch
from canvit_pytorch import create_backbone
from canvit_pytorch.backbone.vit import ViTBackbone
from canvit_pytorch.teacher import DINOv3Teacher
from canvit_pytorch.teacher import load_teacher as _load_teacher

from canvit_pretrain import CanViTForPretraining

from .config import Config

log = logging.getLogger(__name__)


class ModelBundle(NamedTuple):
    """Model with derived runtime parameters."""

    model: CanViTForPretraining
    glimpse_size_px: int


def load_teacher(cfg: Config) -> DINOv3Teacher:
    """Load frozen DINOv3 teacher from HuggingFace Hub."""
    return _load_teacher(cfg.teacher_repo_id, cfg.device)


def load_student_backbone(cfg: Config) -> ViTBackbone:
    """Load student backbone (random init; pretrained weights loaded via checkpoint system)."""
    backbone = create_backbone(cfg.backbone_name)
    log.info(f"Student backbone: {cfg.backbone_name} (random init)")
    return backbone.to(cfg.device)


def create_model(
    student_backbone: ViTBackbone,
    teacher_dim: int,
    cfg: Config,
) -> ModelBundle:
    """Create CanViTForPretraining wrapping student backbone.

    glimpse_size_px is the side length of the pixel crop fed to ``model.forward``.
    - uniform mode: glimpse_grid_size * student_backbone.patch_size_px
    - foveated mode: cfg.model.foveated_patcher.fixation_size (fovi's fixation window)
    """
    cfg.model.teacher_dim = teacher_dim

    model = CanViTForPretraining(
        backbone=student_backbone,
        cfg=cfg.model,
        backbone_name=cfg.backbone_name,
        canvas_patch_grid_sizes=[cfg.canvas_patch_grid_size],
    ).to(cfg.device)

    if cfg.model.patcher_name == "foveated":
        glimpse_size_px = cfg.model.foveated_patcher.fixation_size
        log.info(
            f"Model created (foveated): canvas={cfg.canvas_patch_grid_size}x{cfg.canvas_patch_grid_size}, "
            f"glimpse={glimpse_size_px}px, n_patches={model.patcher.n_patches}, "
            f"student_dim={student_backbone.embed_dim} -> teacher_dim={teacher_dim}, "
        )
    else:
        glimpse_size_px = cfg.glimpse_grid_size * student_backbone.patch_size_px
        log.info(
            f"Model created: canvas={cfg.canvas_patch_grid_size}x{cfg.canvas_patch_grid_size}, "
            f"glimpse={cfg.glimpse_grid_size}x{cfg.glimpse_grid_size} ({glimpse_size_px}px), "
            f"student_dim={student_backbone.embed_dim} -> teacher_dim={teacher_dim}, "
        )
    return ModelBundle(model, glimpse_size_px)


def compile_teacher(teacher: DINOv3Teacher) -> None:
    """Compile teacher HF model blocks in-place."""
    teacher.model = torch.compile(teacher.model)  # type: ignore[assignment]


def compile_model(model: CanViTForPretraining) -> None:
    """Compile CanViTForPretraining in-place."""
    model.compile()
