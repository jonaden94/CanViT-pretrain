"""ADE20K probe-training configuration (port of canvit_specialize's, trimmed per the
unification master plan: canvas_hidden only — recon_normalized dropped (D3); the
legacy full-finetune branch is not ported (full-FT arrives via the harness, P4)."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from canvit_pytorch import resolve_canvit_repo
from canvit_pytorch.data.ade20k import ResizeMode  # noqa: F401  (re-exported for this repo's consumers)

from ..train.config import FoveatedScaleConfig


def _default_wandb_project() -> str | None:
    return os.environ.get("WANDB_PROJECT")


def _default_wandb_entity() -> str | None:
    return os.environ.get("WANDB_ENTITY") or None


def _default_wandb_dir() -> Path | None:
    if d := os.environ.get("WANDB_DIR"):
        return Path(d)
    return None


def _default_probe_ckpt_dir() -> Path:
    base = os.environ.get("CHECKPOINTS_DIR", "checkpoints")
    return Path(base) / "canvit-ade20k-probes"


def _default_ade20k_root() -> Path:
    if root := os.environ.get("ADE20K_ROOT"):
        return Path(root)
    if tmpdir := os.environ.get("SLURM_TMPDIR"):
        return Path(tmpdir) / "ADEChallengeData2016"
    return Path("/datasets/ADE20k/ADEChallengeData2016")


@dataclass
class Ade20kConfig:
    """Frozen-backbone ADE20K probe training."""

    model_repo: str = resolve_canvit_repo("canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02")
    ade20k_root: Path = field(default_factory=_default_ade20k_root)
    scene_size: int = 512

    # Rollout
    n_timesteps: int = 10
    glimpse_px: int | None = None
    """Uniform-patcher glimpse crop size in px. None = derive from the model's
    glimpse_grid_size × patch size/stride (the canvit_eval rule). Ignored for
    foveated/square models (they consume the full image)."""
    canvas_grid: int | None = None
    """None = scene_size // patch_size."""

    # Viewpoint policy for TRAINING: pure IID random by default
    min_vp_scale: float = 0.05
    max_vp_scale: float = 1.0
    train_start_full: bool = False
    foveated_scale: FoveatedScaleConfig = field(default_factory=FoveatedScaleConfig)
    """Foveated/square patcher only: the view-scale law for the probe rollout,
    which MUST match how the backbone was pretrained — the foveated patcher
    derives its fixation window as ``fix_size = scale * H``, so glimpses at an
    unseen scale are out of distribution and actively degrade the canvas (this
    is not a soft mismatch: it shows up as mIoU *falling* with more glimpses).
    Pass the pretrain run's value, e.g. ``--foveated-scale.fixed-scale 2.0`` for
    exp22-fovi. Ignored for uniform models, which use ``min/max_vp_scale``."""

    # Training
    batch_size: int = 16
    eval_batch_size: int = 32
    num_workers: int = 4
    peak_lr: float = 3e-4
    weight_decay: float = 1e-3
    warmup_steps: int = 1500
    warmup_lr_ratio: float = 1e-6
    max_steps: int = 40000
    grad_clip: float = float("inf")
    dropout: float = 0.1

    # Data augmentation
    aug_scale_range: tuple[float, float] = (0.5, 2.0)
    aug_flip_prob: float = 0.5

    # Validation resize
    resize_mode: ResizeMode = "center_crop"
    """How val images/masks are fitted to ``scene_size``. ``center_crop``
    (default): resize short side then crop the central square — aspect-ratio
    PRESERVING, so it matches how the backbone was pretrained and how the probe
    trained (both aspect-preserving); the cost is that long-side margins are
    discarded, so mIoU is over the central crop. ``squish``: resize to
    (size, size), distorting aspect ratio — required to reproduce the
    CanViT-PyTorch-RL documented numbers (qband band / EG-C2F) and the
    specialize reference, which were all measured under squish. Foveated/square
    models MUST use an aspect-preserving mode (they model biological vision;
    squish is off-distribution) — keep the default for them."""

    # Logging / checkpoints
    log_every: int = 20
    val_every: int = 500
    viz_every: int = 500
    """Render the segmentation overlay figure every N steps (0 = off), for the training
    batch and the first val batch. Specialize's default, restored — but the figures go to
    ``{run_dir}/visualization/seg_{train,val}/`` on disk instead of the wandb Media tab.
    Under the harness this needs ``--opts.run-dir`` set (ade20k has no run_group of its
    own to derive one from); without it there is nowhere to write and viz stays off."""
    viz_samples: int = 4
    """Images per figure (one row each)."""
    device: str = "cuda"
    amp: bool = True
    probe_ckpt_dir: Path | None = field(default_factory=_default_probe_ckpt_dir)
    tracker: Literal["comet", "wandb", "none"] = "wandb"
    wandb_project: str | None = field(default_factory=_default_wandb_project)
    wandb_entity: str | None = field(default_factory=_default_wandb_entity)
    wandb_dir: Path | None = field(default_factory=_default_wandb_dir)
