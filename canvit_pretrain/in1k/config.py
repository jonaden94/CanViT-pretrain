"""ImageNet-1k classification config (unification P5). Fresh CUDA task (D2),
mirroring ade20k/config.py, and step-based like it (max_steps / warmup_steps /
val_every) — the train stream is an infinite resampled WebDataset, so epochs were
only ever a derived batch count."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from canvit_pytorch import resolve_canvit_repo

from ..ade20k.config import (
    ResizeMode,
    _default_wandb_dir,
    _default_wandb_entity,
    _default_wandb_project,
)
from ..train.config import FoveatedScaleConfig

NUM_CLASSES = 1000


def _default_in1k_train_dir() -> Path:
    if root := os.environ.get("IN1K_TRAIN_DIR"):
        return Path(root)
    # WebDataset shards (jpg + json label), pre-resized to 512²; see p5-notes.md.
    return Path("/user/henrich1/u25995/jonathan/datasets/webdataset-imagenet-1k-no-features/train-shuffled")


def _default_in1k_val_dir() -> Path:
    if root := os.environ.get("IN1K_VAL_DIR"):
        return Path(root)
    # Synset-folder ImageFolder (n01440764/, …) — the same val + ordering canvit_eval
    # uses, so ImageFolder's alphabetical class_idx matches the webdataset's int labels.
    return Path("/user/henrich1/u25995/jonathan/datasets/imagenet1k-val")


def _default_clf_ckpt_dir() -> Path:
    base = os.environ.get("CHECKPOINTS_DIR", "checkpoints")
    return Path(base) / "canvit-in1k-clf"


@dataclass
class In1kConfig:
    """CanViT ImageNet-1k classification: frozen-backbone linear probe (default)
    or full finetune, over a glimpse rollout."""

    model_repo: str = resolve_canvit_repo("canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02")
    train_dir: Path = field(default_factory=_default_in1k_train_dir)
    val_dir: Path = field(default_factory=_default_in1k_val_dir)
    scene_size: int = 512

    mode: Literal["frozen", "finetune"] = "frozen"
    """``frozen`` (default, the P5 acceptance target): freeze the CanViT backbone,
    train only the LN+Linear head — the direct analogue of canvit_eval's frozen
    linear-clf-probe baseline. ``finetune``: train the whole classifier end to end
    (the ``...-finetune-...-in1k`` flagship)."""

    # Rollout (how the CLS token that feeds the head is produced)
    n_timesteps: int = 10
    glimpse_px: int | None = None
    """Uniform-patcher glimpse crop px. None = derive from the model's
    glimpse_grid_size × patch size/stride (the canvit_eval rule). Ignored for
    foveated/square models (they consume the full image)."""
    canvas_grid: int | None = None
    """None = scene_size // patch_size."""

    # TRAINING viewpoint policy: IID random by default (matches the ade20k probe)
    min_vp_scale: float = 0.05
    max_vp_scale: float = 1.0
    train_start_full: bool = False
    foveated_scale: FoveatedScaleConfig = field(default_factory=FoveatedScaleConfig)
    """Foveated/square only: the view-scale law for the rollout, which MUST match
    how the backbone was pretrained (``fix_size = scale * H``; off-scale glimpses
    are out of distribution). Ignored for uniform models. See ade20k/config.py."""

    # EVAL viewpoint policy: coarse-to-fine quadtree (the canvit_eval deploy default)
    eval_policy: Literal["coarse_to_fine", "random", "full"] = "coarse_to_fine"

    # Training (step-based, like Ade20kConfig — the train stream is an infinite
    # `resampled=True` WebDataset, so "epoch" was only ever a derived batch count)
    max_steps: int = 200_000
    """Total optimizer steps. The old epoch-based default was 10 epochs, which at
    batch_size=64 over IN1k's 1,281,167 images is 10 * 20018 ~= 200k steps."""
    batch_size: int = 64
    eval_batch_size: int = 64
    num_workers: int = 8
    peak_lr: float = 3e-4
    weight_decay: float = 1e-3
    warmup_steps: int = 10_000
    """LR warmup length in steps (was 0.5 epoch ~= 10k steps at batch_size=64)."""
    warmup_lr_ratio: float = 1e-6
    grad_clip: float = float("inf")
    label_smoothing: float = 0.0

    # Data augmentation (train): RandomResizedCrop + flip (canonical IN1k probe recipe)
    aug_min_scale: float = 0.35
    aug_flip_prob: float = 0.5
    resize_mode: ResizeMode = "center_crop"
    """Val resize (aspect-preserving center_crop matches canvit_eval's canonical
    IN1k preprocessing; foveated/square models MUST stay aspect-preserving)."""

    # Debug / smoke: cap batches per eval (None = full). Train length is `max_steps`.
    limit_val_batches: int | None = None

    # Logging / checkpoints
    log_every: int = 50
    val_every: int = 20_000
    """Validate every N steps (was eval_every_epochs=1, i.e. ~20k steps at batch_size=64)."""
    device: str = "cuda"
    amp: bool = True
    seed: int = 0
    clf_ckpt_dir: Path | None = field(default_factory=_default_clf_ckpt_dir)
    run_name: str = "in1k-clf"
    tracker: Literal["comet", "wandb", "none"] = "wandb"
    wandb_project: str | None = field(default_factory=_default_wandb_project)
    wandb_entity: str | None = field(default_factory=_default_wandb_entity)
    wandb_dir: Path | None = field(default_factory=_default_wandb_dir)
