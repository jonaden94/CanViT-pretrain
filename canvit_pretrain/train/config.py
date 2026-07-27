"""Configuration for CanViT pretraining."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from canvit_pretrain import CanViTForPretrainingConfig
from canvit_pretrain.train.utils import get_sensible_device

# Default HF repo for the teacher model
TEACHER_REPO_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
# Short name used for shard paths and probe lookup (matches precomputed feature directories)
TEACHER_NAME = "dinov3_vitb16"


@dataclass
class FoveatedScaleConfig:
    """How the per-glimpse view scale is sampled for the foveated/square patchers.

    The foveation window is ``fix_size = scale * H``. The FULL start glimpse is
    always centered; its scale is ``fixed_scale`` in ``mode='fixed'`` (so it
    matches every other glimpse) and ``1`` in ``per_rollout``/``per_glimpse`` (a
    full-image anchor). RANDOM glimpses draw their scale per ``mode``: a single
    ``fixed_scale`` (``fixed``), one scale frozen across the rollout
    (``per_rollout``), or a fresh scale each glimpse (``per_glimpse``). The
    uniform patcher path is unaffected by this config.
    """

    mode: Literal["fixed", "per_rollout", "per_glimpse"] = "fixed"
    """``fixed``: one constant scale everywhere (= current full-image training).
    ``per_rollout``: one scale per rollout (per image), held across its glimpses.
    ``per_glimpse``: a fresh scale every glimpse."""
    distribution: Literal["uniform", "safebox"] = "uniform"
    """Sampled-scale distribution (ignored when ``mode='fixed'``). ``uniform``:
    ``scale ~ U(min_scale, max_scale)`` with centers uniform over ``[-1,1]^2``
    (``max_scale > 1`` allows zoom-out). ``safebox``: reuse the uniform-patcher
    safe-box joint sampler (center coupled to scale, no overshoot, ``scale<=1``)."""
    fixed_scale: float = 1.0
    """Scale used when ``mode='fixed'`` (1.0 = full-image foveation)."""
    min_scale: float = 0.5
    max_scale: float = 1.0
    """Sampled-scale range (scale units = fraction of the image side)."""


@dataclass
class JointPolicyConfig:
    """Joint task+policy training (unification P4b): learn a ViewpointScorer *while*
    the distill task trains, driving the glimpses from the policy's candidate grid.

    OFF by default (``use_rl=False``) — the training loop then builds the historical
    RandomSelector and behaves byte-for-byte like pre-P4b pretraining (the parity
    gate). When ON, the exploratory glimpses (t>=1) of the policy branches come from
    the scorer's discrete candidate grid via ε-greedy (QReg) or on-policy sampling
    (PG); the per-glimpse reward is the fractional reduction in per-image distill MSE
    (master plan §3), standardized per depth. The action space is the fixation grid
    for foveated/square models and the safe-box grid for uniform.
    """

    use_rl: bool = False
    """Master off-switch. False => no policy, no behavior change (parity gate)."""
    rl_weight: float = 1.0
    """Scale on the policy loss added to the per-glimpse distill loss."""
    feats_detached: bool = True
    """Detach the canvas state feeding the scorer, so the policy gradient reaches
    ONLY the scorer (backbone/head shaped purely by the distill loss). False couples
    them: the policy loss also reshapes the backbone ('glimpse-plannable' features),
    more ambitious and more memory (backbone activations enter the policy graph)."""
    keep_random_branch: bool = False
    """False (default): every branch is a policy branch (all t>=1 glimpses are the
    policy's grid picks; the distill loss trains on exactly those states). True:
    the FULL-start branches become policy branches while the RANDOM-start branches
    stay pure continuous-random (distill-only, no policy loss) for broad view
    coverage — needs n_full_start_branches>=1 and n_random_start_branches>=1."""

    # Objective
    objective: Literal["qreg", "pg"] = "qreg"
    prime_on_policy: float = 0.5
    """QReg ε-greedy: fraction of glimpses taken by the net's argmax (rest = a random
    candidate). Ramped 0 -> this over ``policy_warmup_steps`` (the ε-curriculum)."""
    dueling: bool = True
    entropy_bonus: float = 0.01
    entropy_target: float | None = 1.0
    alpha_lr: float = 0.05
    policy_warmup_steps: int = 0
    """Steps to ramp prime_on_policy 0 -> its target (0 = constant target from step 0)."""

    # Action space / scorer net
    centers_per_axis: int = 16
    scales: tuple[float, ...] = (0.5, 0.25)
    """Uniform-patcher candidate scales; ignored for foveated/square (fixation grid)."""
    width: int = 128
    block_layers: int = 3
    feature_groups: tuple[str, ...] = ("cos_prev", "cos_init", "ln_feat", "feat_delta")
    """= policy.features.INTRINSIC_GROUPS — the probe-free groups (distill has no probe)."""

    # Target standardizer + policy optimizer
    target_momentum: float = 0.997
    policy_lr: float = 2e-4
    policy_weight_decay: float = 1e-2


@dataclass
class Config:
    # Teacher
    teacher_repo_id: str = TEACHER_REPO_ID
    teacher_name: str = TEACHER_NAME
    # Student
    backbone_name: str = "vitb16"
    # Model config (PretrainingConfig via alias)
    # teacher_dim placeholder - overridden by create_model based on actual teacher
    model: CanViTForPretrainingConfig = field(
        default_factory=lambda: CanViTForPretrainingConfig(teacher_dim=768)
    )
    # Glimpse/canvas sizes (runtime, not in model config)
    glimpse_grid_size: int = 8  # tokens per glimpse side
    patch_stride: int | None = None  # uniform patcher: patch-embed conv stride;
    # None = patch_size (non-overlapping, default). Set < patch_size for overlapping
    # patches; glimpse_size_px is then (grid-1)*stride + patch_size.
    canvas_patch_grid_size: int = 32  # canvas spatial grid side length in tokens
    # Training
    batch_size_per_gpu: int = 64
    warmup_steps: int = 100_000
    start_lr: float | None = 1e-7  # None = peak_lr / warmup_steps
    peak_lr: float = 4e-4
    cosine_total_steps: int | None = None  # None = constant after warmup; set to enable cosine decay
    weight_decay: float = 1e-4
    min_viewpoint_scale: float = 0.05  # Minimum scale for random viewpoints
    n_full_start_branches: int = 1  # branches starting with FULL viewpoint at t0
    n_random_start_branches: int = 1  # branches starting with RANDOM viewpoint at t0
    foveated_scale: FoveatedScaleConfig = field(default_factory=FoveatedScaleConfig)
    """Per-glimpse view-scale sampling for foveated/square patchers (RANDOM
    glimpses). Default: fixed scale 1.0 = current full-image foveation."""
    rl: JointPolicyConfig = field(default_factory=JointPolicyConfig)
    """Joint task+policy training (P4b). OFF by default => byte-identical to
    pre-P4b pretraining. See :class:`JointPolicyConfig`."""
    chunk_size: int = 2  # BPTT chunk size (glimpses per chunk, gradient flows within)
    continue_prob: float = 0.5  # prob of adding another chunk to trajectory
    enable_scene_patches_loss: bool = True  # Scene (canvas) patch reconstruction loss
    enable_scene_cls_loss: bool = True  # Scene (global) CLS reconstruction loss
    ema_alpha: float = 0.1  # EMA smoothing for metrics
    grad_clip: float = 1.0
    steps_per_job: int = 4_992  # Steps this job does before exiting (for SLURM arrays)
    # Data
    train_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/train")
    val_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/val")
    train_index_dir: Path | None = None  # Required for raw image training
    val_index_dir: Path | None = None  # Required for validation
    # Precomputed features (skips teacher inference on train images)
    # If feature_base_dir is set, shards path is auto-constructed:
    #   {feature_base_dir}/{teacher_name}/{scene_resolution}/shards/
    feature_base_dir: Path | None = None
    feature_image_root: Path | None = None  # Required with feature_base_dir
    tar_dir: Path | None = None  # images read directly from mmap'd tars
    # WebDataset path (alternative to feature_base_dir). When set, training
    # reads pre-shuffled WebDataset tar shards under {webdataset_dir}/train-shuffled
    # and validates against {webdataset_dir}/val.
    webdataset_dir: Path | None = None
    seed: int = 0  # for reproducibility (shard schedule permutations)
    # Run identification and checkpointing
    run_group: str | None = None
    """Run category (e.g. 'foveated', 'crop'). Combined with run_name to form
    `logs_dir / run_group / run_name /` as the root of all per-run artifacts."""
    run_name: str | None = None
    """Run name. Auto-generated from SLURM_ARRAY_JOB_ID or timestamp if None."""
    logs_dir: Path = Path("logs")
    """Root for run artifacts. Per-run files go under
    `logs_dir / run_group / run_name / {checkpoints,log}/`."""
    seed_ckpt: Path | None = None
    """Seed model weights from external checkpoint (.pt in CheckpointData format).
    Starts fresh (new experiment, step=0). Only used if no checkpoint exists in run_dir."""
    hf_seed_ckpt: str | None = None
    """Seed model weights from HF Hub repo (e.g. '<org>/canvitb16-add-vpe-...'). Downloads
    config.json + model.safetensors, overrides cfg.model with the checkpoint's config.
    Mutually exclusive with seed_ckpt."""
    init_backbone_from_teacher: bool = False
    """Initialize the student ViT backbone from the (already-loaded) DINOv3 teacher's
    weights instead of random init. The two share the same ViT-B/16 design (12 layers,
    12 heads, head_dim 64, RoPE theta 100, LayerScale, 4x MLP), so the full transformer
    trunk transfers 1:1 (teacher q/k/v fused into the student's qkv, K-bias zero-filled);
    patch_embed transfers only when the student patch size matches the teacher's (16 ->
    vitb16 yes, vitb8 no -> left random). Backbone only; the rest of CanViT (canvas, VPE,
    heads) stays random. Applied on a fresh start; resume/seed_ckpt take precedence."""
    reset_normalizer: bool = False
    """Re-warmup normalizer stats when loading any checkpoint."""
    normalizer_max_samples: int = 0
    """Max samples from shard for normalizer stats. 0 = use all samples."""
    # Training
    num_workers: int = 4
    scene_resolution: int = 512
    dataset: str = "in21k"
    # Logging
    log_every: int = 20
    val_every: int = 1000
    n_eval_viewpoints: int = 10  # Number of viewpoints in validation (quadtree)
    n_val_samples: int = 256
    """Number of validation samples evaluated per validation, independent of
    ``batch_size_per_gpu`` and of world size. A fixed, seeded random subset of the
    val set (the SAME images every validation — no cycling), clamped to the val set
    size. Evaluated on rank 0 only, in chunks of ``batch_size_per_gpu`` whose
    per-timestep metrics are aggregated (chunk size does not affect the result)."""
    val_seed: int = 0
    """Seed selecting the fixed validation subset. Same seed -> same images across
    runs and across world sizes."""
    viz_every_n_vals: int = 5  # Log viz every N validation runs
    curve_every_n_vals: int = 5  # Log curves every N validation runs
    log_spatial_stats: bool = True
    log_patcher_grad_detail: bool = True
    """Break the patcher's per-validation grad-norm logs into sub-components:
    ``patcher.kpe``, ``patcher.embed_head``, ``patcher.conditioner.*`` (FiLM MLP /
    learned per-patch code / etc.). When False, the patcher is reported as a
    single ``patcher`` group like every other top-level module. All other modules
    (backbone, scene_cls_head, …) are unaffected either way."""
    # Experiment tracker
    tracker: Literal["comet", "wandb", "none"] = "wandb"
    """Backend for parameter/metric/figure logging."""
    wandb_project: str | None = field(default_factory=lambda: os.environ.get("WANDB_PROJECT"))
    """W&B project name. Required when tracker='wandb'. Defaults to $WANDB_PROJECT
    (.envrc.grete sets it) — the same default Ade20kConfig/In1kConfig already had, which
    distill alone was ignoring. Every launcher passes it explicitly, so this only changes
    what a hand-run job does: land in the default project instead of asserting."""
    wandb_entity: str | None = field(default_factory=lambda: os.environ.get("WANDB_ENTITY") or None)
    """W&B entity (team or user). Falls back to $WANDB_ENTITY, then your default account."""
    wandb_dir: Path | None = Path("/mnt/vast-nhr/projects/nib00021/jonathan")
    """Directory wandb writes its run files into. None = wandb's own default (./wandb)."""
    # Compilation and precision
    compile: bool = True
    combo_kernels: bool = False  # torch._inductor.config.combo_kernels (experimental)
    amp: bool = True
    non_blocking_transfer: bool = True  # Ablation: async CPU→GPU transfers
    # Optuna
    n_trials: int = 1
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)
