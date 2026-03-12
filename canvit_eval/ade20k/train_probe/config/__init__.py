"""ADE20K canvas probe training configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _default_probe_ckpt_dir() -> Path:
    base = os.environ.get("CHECKPOINTS_DIR", "checkpoints")
    return Path(base) / "canvit-ade20k-probes"


def _default_ade20k_root() -> Path:
    if root := os.environ.get("ADE20K_ROOT"):
        return Path(root)
    if tmpdir := os.environ.get("SLURM_TMPDIR"):
        return Path(tmpdir) / "ADEChallengeData2016"
    raise ValueError("ADE20K_ROOT env var not set and not running under SLURM")


# --- Canvas feature registry ---

CanvasFeatureType = Literal["canvas_hidden", "recon_normalized"]


@dataclass(frozen=True)
class FeatureSpec:
    needs_ln: bool
    dim_source: Literal["canvas", "teacher"]


CANVAS_FEATURES: dict[CanvasFeatureType, FeatureSpec] = {
    "canvas_hidden": FeatureSpec(needs_ln=True, dim_source="canvas"),
    "recon_normalized": FeatureSpec(needs_ln=False, dim_source="teacher"),
}


def get_feature_dims(canvas_dim: int, teacher_dim: int) -> dict[CanvasFeatureType, int]:
    dim_map: dict[str, int] = {"canvas": canvas_dim, "teacher": teacher_dim}
    return {k: dim_map[v.dim_source] for k, v in CANVAS_FEATURES.items()}


# --- Shared base config ---

@dataclass
class ProbeTrainBase:
    """Shared training hyperparameters for all ADE20K probe types."""

    ade20k_root: Path = field(default_factory=_default_ade20k_root)
    scene_size: int = 512

    # Training (defaults match DINOv3 linear probing, Appendix D.1)
    batch_size: int = 16
    eval_batch_size: int = 32
    num_workers: int = 4
    peak_lr: float = 3e-4
    weight_decay: float = 1e-3
    warmup_steps: int = 1500
    warmup_lr_ratio: float = 1e-6
    max_steps: int = 40000
    grad_clip: float = float("inf")

    # Probe head
    dropout: float = 0.1

    # Data augmentation (DINOv3 defaults)
    aug_scale_range: tuple[float, float] = (0.5, 2.0)
    aug_flip_prob: float = 0.5

    # Logging
    log_every: int = 20
    val_every: int = 500
    viz_every: int = 500
    viz_samples: int = 4
    comet_project: str = "canvit-ade20k-probes"
    comet_workspace: str = "m2b3-ava"
    device: str = "cuda"
    amp: bool = True
    probe_ckpt_dir: Path | None = field(default_factory=_default_probe_ckpt_dir)


@dataclass
class Config(ProbeTrainBase):
    """Canvas probe training configuration."""

    model_repo: str = "canvit/canvit-vitb16-pretrain-512px-in21k"
    teacher_repo: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    features: list[CanvasFeatureType] = field(default_factory=lambda: ["canvas_hidden"])
    n_timesteps: int = 10
    glimpse_px: int = 128
    canvas_grid: int | None = None

    # Viewpoint policy for TRAINING: pure IID random by default
    min_vp_scale: float = 0.05
    max_vp_scale: float = 1.0
    train_start_full: bool = False
