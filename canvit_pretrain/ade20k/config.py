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

    mode: Literal["frozen", "finetune"] = "frozen"
    """``frozen`` (default, the historical probe regime): freeze the CanViT backbone and
    train only the segmentation head. ``finetune``: train the whole model end to end.
    Mirrors ``In1kConfig.mode`` so both downstream tasks name the same thing the same way
    — before this, ade20k finetune was reachable ONLY via the generic ``--preset
    finetune``, which is a different entry point with different LR-schedule behaviour.
    Honored by the harness (``tasks/ade20k/task.py::default_spec``); the standalone
    entry point trains the probe only and ignores it."""

    probe_repo: str | None = None
    """FINETUNE only: a published segmentation probe to initialise the head from
    (core ``from_pretrained_with_probe``) instead of a fresh random one — the ADE20K
    analogue of ``In1kConfig.probe_repo`` and of specialize's ``init_probe_repo``.
    Starting a finetune from a RANDOM head at the small finetune LR trains far too
    slowly; that exact bug cost the unified in1k finetune a whole run (fixed in
    8f780ba). ``None`` => fresh head. Ignored in ``frozen`` mode."""

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
    """How val images/masks are fitted to the square ``scene_size``. Both modes are
    supported for every patcher; neither is "the correct one" — each trades away something
    different, so pick per experiment and say which one a number came from.

    ``center_crop`` (default): resize the short side, crop the central square. Preserves
    geometry (a circle stays a circle), matching how the backbone was pretrained and how
    the probe trains; the cost is that long-side content is thrown away, so the mIoU is over
    the central crop rather than the whole image.

    ``squish``: resize to (size, size), distorting aspect ratio. Keeps the whole field of
    view, and it is what the CanViT-PyTorch-RL numbers (qband band / EG-C2F) and the
    specialize reference were measured under — so it is the mode to use for comparability
    with those.

    Which to prefer for a FOVEATED/square model: for comparison against **human** viewing,
    aspect-preserving is the better match, since the cortical-magnification geometry is
    defined on the undistorted scene — a human looking at the same picture does not see it
    anisotropically stretched. For comparability with the **original CanViT** results,
    squish is fine, and the distortion is a mild domain shift, not a broken input (unlike a
    view-scale mismatch, which really does degrade the canvas — see ``foveated_scale``).
    NB the truest "what a human sees" option would be padding/letterbox (preserve aspect
    AND the full frame), which is not implemented: ``ResizeMode`` is center_crop | squish."""

    # Debug / smoke: cap batches per eval (None = full), mirroring In1kConfig. Honored by
    # both entry points (the standalone val loop and the harness task's `evaluate`).
    limit_val_batches: int | None = None

    # Run identity — the same trio Config/In1kConfig carry, so every task names its runs
    # the same way. The harness derives BOTH the tracker run name and
    # `logs_dir/run_group/run_name/` (checkpoints + visualization) from these. The
    # standalone entry point honors `run_name` (tracker name + its checkpoint subdir) and
    # ignores `run_group`/`logs_dir`: its artifact root is `probe_ckpt_dir`.
    run_group: str | None = None
    run_name: str | None = None
    """None => auto: the descriptive `ade20k_{model}_{T}t_s{scene}_c{grid}_{ts}` name in
    the standalone, `ade20k_{timestamp}` in the harness."""
    logs_dir: Path = Path("logs")

    # Logging / checkpoints
    log_every: int = 20
    val_every: int = 500
    viz_every: int = 500
    """Render the segmentation overlay figure every N steps (0 = off), for the training
    batch and the first val batch. Specialize's default, restored — but the figures go to
    ``{run_dir}/visualization/seg_{train,val}/`` on disk instead of the wandb Media tab.
    Harness only (the standalone renders no figures), and it needs a run dir: set
    ``run_group`` (or ``--opts.run-dir``), else there is nowhere to write and viz is off."""
    viz_samples: int = 4
    """Images per figure (one row each)."""
    device: str = "cuda"
    amp: bool = True
    seed: int = 0
    """`torch.manual_seed(seed + rank)`. Historically the probe had NO seed at all (both
    entry points), which made A/B gates against it impossible — same config, different
    curve. Both honor it now; 0 keeps the value the harness was already passing."""
    probe_ckpt_dir: Path | None = field(default_factory=_default_probe_ckpt_dir)
    tracker: Literal["comet", "wandb", "none"] = "wandb"
    wandb_project: str | None = field(default_factory=_default_wandb_project)
    wandb_entity: str | None = field(default_factory=_default_wandb_entity)
    wandb_dir: Path | None = field(default_factory=_default_wandb_dir)
