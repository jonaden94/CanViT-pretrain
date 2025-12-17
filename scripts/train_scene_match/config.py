"""Configuration for scene matching training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from ytch.device import get_sensible_device

from avp_vit import AVPConfig


@dataclass
class Config:
    # Model architectures (required - no defaults to force explicit choice)
    teacher_model: str = "dinov3_vits16"  # e.g., "dinov3_vits16", "dinov3_vitb16"
    student_model: str = "dinov3_vits16"  # e.g., "dinov3_vits16"
    # Checkpoints
    teacher_ckpt: Path = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    student_ckpt: Path | None = None  # None = random init with student_model template
    # Paths
    train_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/train")
    val_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/val")
    ckpt_dir: Path = Path("checkpoints")
    # Model (uses AVPConfig defaults, only override scene_grid_size)
    avp: AVPConfig = field(default_factory=lambda: AVPConfig(scene_grid_size=64))
    freeze_inner_backbone: bool = False
    # Grid sizes (randomly sampled each step)
    grid_sizes: tuple[int, ...] = (16, 32, 64)
    # Training
    n_viewpoints_per_step: int = (
        2  # Inner loop viewpoints (>=2 for length generalization)
    )
    n_steps: int = 200000
    batch_size: int = 16  # Max batch size (at max grid size)
    num_workers: int = 8
    ref_lr: float = 1e-5
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    crop_scale_min: float = 0.4
    loss: Literal["l1", "mse"] = "mse"
    # Logging
    log_every: int = 20
    val_every: int = 50
    curve_every: int = 1000  # Curves less often than val (Comet limit: 1000/experiment)
    ckpt_every: int = 500
    log_spatial_stats: bool = True  # Log target/pred spatial mean/std
    # Compilation
    compile: bool = True
    # Optuna
    n_trials: int = 100
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)

    @property
    def max_grid_size(self) -> int:
        return max(self.grid_sizes)
