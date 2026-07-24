"""The single run orchestration for the unified harness (design §2 ``run.py``, D-B).

``run(task, spec, settings)`` is the one path all three peer tasks take:

    build_model → build_policy (joint) → apply_requires_grad → build_optimizer
    → build_loaders → build_selector → run_training_loop (+ eval/log/ckpt hooks)

Everything task-specific lives behind the ``RunTask`` seam (a run-level ``Task``:
``tasks/{distill,ade20k,in1k}/task.py``); everything task-neutral is here or in the
sibling harness modules (rollout / optim / checkpoint / loop / policy). This subsumes
the three legacy entry points (``train`` loop, ``ade20k.train``, ``in1k.train``) as one
``TrainSpec``-driven call — the big-bang cutover (a separate, owner-gated step) only
repoints ``python -m canvit_pretrain.train`` at this and deletes the old loops.

Config is **composed** (D-B), not one mega-dataclass: the task holds its own per-task
config; :class:`RunSettings` carries the harness-level, cross-cutting knobs; ``TrainSpec``
carries what-trains-under-which-loss. ``run`` writes checkpoints LOCALLY only (D-G); HF
publishing stays the manual ``python -m canvit_pretrain.checkpoint.to_hf`` step.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import torch

from canvit_pretrain.harness.checkpoint import find_latest, load_checkpoint, restore_into
from canvit_pretrain.harness.loop import (
    apply_requires_grad,
    cancel_slurm_array,
    install_sigusr1_handler,
    run_training_loop,
)
from canvit_pretrain.harness.optim import build_optimizer_and_scheduler
from canvit_pretrain.harness.spec import TaskCaps, TrainSpec, check_spec
from canvit_pretrain.train.viewpoint import ViewpointType

log = logging.getLogger(__name__)


@dataclass
class RunSettings:
    """Harness-level, task-neutral run knobs (design D-B composed config). Task-specific
    settings (data paths, model repo, augmentation, batch size, horizon) stay in the
    task's own config; ``TrainSpec`` owns what-trains-under-which-loss."""

    n_steps: int = 100
    start_step: int = 0
    grad_clip: float = 1.0
    amp: bool = True
    amp_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    device: str = "cuda"
    seed: int = 0
    # DDP (single-GPU default; the loop AllReduces the scorer when world_size>1)
    world_size: int = 1
    rank: int = 0
    # cadence
    log_every: int = 20
    ckpt_every: int = 0          # 0 => only the end-of-run checkpoint
    eval_every: int = 0          # 0 => no periodic eval
    ckpt_dir: Path | None = None
    # resume + operational (task-neutral; parity with train/loop.py)
    resume: bool = True          # resume from find_latest(ckpt_dir) if a checkpoint exists
    signal_checkpoint: bool = True   # SIGUSR1 -> checkpoint after the current step
    use_failed_marker: bool = False  # SLURM crash-loop guard: FAILED file + scancel array
    ema_alpha: float = 0.1       # EMA-smoothed total_loss in the logs (0 => off)
    log_grad_norms: bool = True  # per-module grad L2 norms on log steps
    grad_norm_deep_prefixes: tuple[str, ...] = ()
    seed_ckpt: Path | None = None
    """SEED mode: load WEIGHTS ONLY from this local checkpoint and start a fresh run at
    step 0 (fresh optimizer/scheduler) — how production runs start from a pretrained
    model. Distinct from ``resume`` (full state, continues the step count), which takes
    priority: resume > seed_ckpt > fresh. Task-level HF seeding (distill's
    ``cfg.hf_seed_ckpt``, ade20k/in1k's ``cfg.model_repo``) happens in ``build_model``."""
    compile: bool = False        # torch.compile the model after build (perf)
    run_dir: Path | None = None
    """Root for this run's artifacts (`{run_dir}/checkpoints`, `{run_dir}/visualization`),
    the `logs_dir/run_group/run_name` convention. When set and ``ckpt_dir`` is not,
    checkpoints go to ``run_dir/checkpoints``."""
    log_timing: bool = True      # log cumulative data% vs gpu% (bottleneck analysis)
    viz_every: int = 0
    """Render the task's training-batch visualization every N steps (0 = off). Figures
    are written LOCALLY under ``run_dir/visualization/`` — never uploaded (D-G). Only
    tasks that implement ``render_viz`` (currently distill's multistep PCA) produce any."""
    # experiment tracker (off by default so run() is import-safe / offline)
    tracker: Literal["wandb", "none"] = "none"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_dir: Path | None = None
    run_name: str | None = None


@runtime_checkable
class RunTask(Protocol):
    """What ``run`` needs from a task (the run-level seam; the per-glimpse ``RolloutTask``
    seam in ``rollout.py`` is what ``bind`` returns). Concrete impls: the ``*RunTask``
    classes in ``tasks/<name>/task.py``."""

    name: str

    def caps(self) -> TaskCaps: ...
    def default_spec(self) -> TrainSpec: ...
    # prior_model_config: the resume checkpoint's model_config, which must win over
    # this run's config defaults (None when not resuming).
    def build_model(self, device: torch.device,
                    prior_model_config: dict | None = None) -> tuple[Any, Any]: ...  # (model, head|None)
    def canvas_grid(self, model: Any) -> int: ...
    def is_foveated(self, model: Any) -> bool: ...
    def branches(self) -> list[ViewpointType]: ...
    def build_loaders(self, *, world_size: int, rank: int) -> tuple[Any, Any]: ...  # (train_iter, val)
    def build_selector(self, *, device: torch.device, canvas_grid: int, is_foveated: bool) -> Any: ...
    def build_policy(self, model: Any, *, device: torch.device, canvas_grid: int,
                     generator: torch.Generator) -> Any: ...
    def trainable_param_groups(self, *, model: Any, head: Any, joint: Any,
                               spec: TrainSpec) -> dict[str, list]: ...
    def resume_start_step(self, payload: dict, scheduler: Any) -> int: ...  # default: scheduler.last_epoch
    def resume_state(self) -> dict: ...  # extra state this task needs to resume its data schedule
    def batch_images(self, batch: Any, device: torch.device) -> torch.Tensor: ...
    def bind(self, batch: Any, device: torch.device, *, model: Any, head: Any) -> Any: ...
    def evaluate(self, *, model: Any, head: Any, val_loader: Any, device: torch.device,
                 step: int, tracker: Any = None, run_dir: Path | None = None) -> dict: ...
    def model_config(self, model: Any) -> dict: ...
    def checkpoint_metadata(self, model: Any) -> dict: ...


def _as_dict(cfg: Any) -> dict:
    """A task config as a plain dict (dataclass or not); {} when there is none."""
    from dataclasses import asdict, is_dataclass
    if cfg is None:
        return {}
    return asdict(cfg) if is_dataclass(cfg) else dict(getattr(cfg, "__dict__", {}))


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """Nested dict -> dotted keys, non-scalars stringified (train/loop.py flatten_dict)."""
    flat: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten(v, f"{key}."))
        else:
            flat[key] = v if isinstance(v, (int, float, bool, str, type(None))) else str(v)
    return flat


def _infinite(loader: Any):
    """Yield batches forever. Map-style loaders (ade20k) are re-iterated at exhaustion;
    webdataset/streaming loaders (distill/in1k) are effectively infinite already but this
    is harmless."""
    if hasattr(loader, "next"):        # distill WebDatasetTrainLoader
        while True:
            yield loader.next()
    while True:
        for batch in loader:
            yield batch


def run(*, task: RunTask, spec: TrainSpec, settings: RunSettings) -> dict:
    """Train ``task`` under ``spec`` for ``settings.n_steps`` and return the last
    step's metrics. Task-neutral; every task-specific decision is delegated to ``task``.
    """
    use_cuda = settings.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(settings.device if use_cuda else "cpu")
    torch.manual_seed(settings.seed + settings.rank)
    is_dist = settings.world_size > 1

    caps = task.caps()
    report = check_spec(spec, caps, is_dist=is_dist)
    if not report.ok:
        raise ValueError("invalid TrainSpec:\n  - " + "\n  - ".join(report.errors))
    for w in report.warnings:
        log.warning("TrainSpec warning: %s", w)

    log.info("=" * 60)
    log.info("Unified harness run: task=%s device=%s n_steps=%d", task.name, device, settings.n_steps)
    log.info("spec: train(bb=%s head=%s policy=%s) task_w=%.3g policy_w=%.3g task->bb=%s pol->bb=%s bptt=%s",
             spec.train_backbone, spec.train_head, spec.train_policy, spec.task_weight,
             spec.policy_weight, spec.task_grad_to_backbone, spec.policy_grad_to_backbone, spec.bptt)
    log.info("=" * 60)

    if settings.signal_checkpoint:
        install_sigusr1_handler()

    # Run-dir convention (logs_dir/run_group/run_name): checkpoints + visualizations
    # live under it. An explicit ckpt_dir still wins.
    run_dir = settings.run_dir
    ckpt_dir = settings.ckpt_dir or ((run_dir / "checkpoints") if run_dir else None)

    # SLURM crash-loop guard (opt-in): a FAILED file left by a prior crash stops the
    # array from re-crashing forever (see the failed-token-blocks-resume incident).
    failed_marker = ((run_dir or ckpt_dir) / "FAILED") if (run_dir or ckpt_dir) else None
    if settings.use_failed_marker and failed_marker is not None and failed_marker.exists():
        log.error("FAILED marker exists (%s) — a previous job crashed; delete it to retry.", failed_marker)
        if settings.rank == 0:
            cancel_slurm_array()
        raise RuntimeError(f"Refusing to start: {failed_marker} exists")

    # --- resume checkpoint is resolved BEFORE the model is built, because its
    # model_config must WIN over CLI defaults: building from the defaults and then
    # strict-loading saved weights of a different arch fails on missing/unexpected keys
    # (train/loop.py 254-261). ---------------------------------------------
    latest = find_latest(ckpt_dir) if (settings.resume and ckpt_dir is not None) else None
    prior_ckpt: dict | None = load_checkpoint(latest, device) if latest is not None else None

    # --- model + (optional) joint policy -----------------------------------
    model, head = task.build_model(
        device, prior_model_config=(prior_ckpt or {}).get("model_config") or None)
    canvas_grid = task.canvas_grid(model)
    is_foveated = task.is_foveated(model)

    if settings.compile:
        log.info("Compiling model (torch.compile)")
        compiled = getattr(model, "compile", None)
        if callable(compiled):
            compiled()          # CanViT wrappers expose their own .compile()
        else:
            model = torch.compile(model)  # type: ignore[assignment]

    joint = None
    if spec.train_policy or spec.policy_loss_active:
        gen = torch.Generator(device=device).manual_seed(settings.seed + settings.rank)
        joint = task.build_policy(model, device=device, canvas_grid=canvas_grid, generator=gen)

    # The harness — not the task — decides what trains (design §3.1).
    apply_requires_grad(model=model, head=head, joint=joint, spec=spec)

    # --- optimizer + scheduler (per trainable group, design D-E) -----------
    param_groups = task.trainable_param_groups(model=model, head=head, joint=joint, spec=spec)
    optimizer, scheduler = build_optimizer_and_scheduler(spec, param_groups)

    # --- RESUME / SEED / FRESH (task-neutral), BEFORE loaders so distill's
    # model-owned normalizer arrives already-initialized from the checkpoint --------
    # Priority mirrors train/loop.py: resume (full state, continues the step count)
    # > seed_ckpt (WEIGHTS ONLY, fresh opt/sched at step 0) > fresh.
    start_step = settings.start_step
    if prior_ckpt is not None:
        log.info("RESUME mode: continuing from %s", latest)
        restore_into(prior_ckpt, model=model, optimizer=optimizer, scheduler=scheduler,
                     joint=joint, path=latest, device=device)
        start_step = task.resume_start_step(prior_ckpt, scheduler)
        log.info("RESUME: start_step=%d", start_step)
    elif settings.seed_ckpt is not None:
        log.info("SEED mode: loading weights from %s (fresh opt/sched, step 0)", settings.seed_ckpt)
        seed_payload = load_checkpoint(settings.seed_ckpt, device)
        # weights only: no optimizer/scheduler => the run starts a fresh schedule.
        restore_into(seed_payload, model=model, joint=joint, path=settings.seed_ckpt, device=device)
        start_step = 0
    else:
        log.info("FRESH mode: no checkpoint, starting from scratch")

    # --- data + selector ---------------------------------------------------
    train_loader, val_loader = task.build_loaders(world_size=settings.world_size, rank=settings.rank)
    train_batches = _infinite(train_loader)
    selector = task.build_selector(device=device, canvas_grid=canvas_grid, is_foveated=is_foveated)
    branches = task.branches()

    amp_ctx: Any = nullcontext()
    if settings.amp and use_cuda:
        amp_ctx = torch.autocast("cuda", dtype=getattr(torch, settings.amp_dtype))

    # --- tracker (optional) ------------------------------------------------
    tracker = None
    if settings.tracker == "wandb":
        from canvit_pretrain.train.tracker import make_tracker
        tracker = make_tracker(
            tracker="wandb", is_main=(settings.rank == 0), is_seeding=False,
            run_name=settings.run_name or f"{task.name}-unified",
            wandb_project=settings.wandb_project, wandb_entity=settings.wandb_entity,
            wandb_dir=settings.wandb_dir, prev_comet_id=None, prev_wandb_id=None,
        )
        # Hyperparameters (train/loop.py 279-282, 423): the task's flattened config,
        # the resolved TrainSpec, the SLURM job id and the trainable/total param split.
        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params: dict[str, Any] = {
            **_flatten(_as_dict(getattr(task, "cfg", None))),
            "train_spec": str(spec), "task": task.name, "run_name": settings.run_name or task.name,
            "trainable_params": n_trainable, "total_params": n_total,
        }
        if slurm_job_id := os.environ.get("SLURM_JOB_ID"):
            params["slurm_job_id"] = slurm_job_id
        tracker.log_parameters(params)

    def on_log(step: int, m: dict) -> None:
        extra = ""
        if "reward_frac" in m:
            extra = f"  reward_frac={m['reward_frac']:+.4f}  policy_loss={m['policy_loss']:.4f}"
        log.info("step %d  loss=%.5f  n_glimpses=%.1f  lr=%.2e%s",
                 step, m["total_loss"], m["n_glimpses"], scheduler.get_last_lr()[0], extra)
        if tracker is not None:
            # train/loop.py's namespacing: everything under `train/` except the
            # per-module gradient norms, which have their own `grad_norm/` namespace.
            payload = {(k if k.startswith("grad_norm/") else f"train/{k}"): v
                       for k, v in m.items() if k != "step"}
            payload["train/lr"] = scheduler.get_last_lr()[0]
            tracker.log_metrics(payload, step=step)

    def on_eval(step: int) -> dict:
        # tracker/run_dir go through so a task's validation can log its OWN rich series
        # (distill's per-timestep val curves + probe accuracy) and write its figures to
        # the run dir — train/loop.py passed `exp` and `run_dir` straight into validate().
        metrics = task.evaluate(model=model, head=head, val_loader=val_loader, device=device,
                                step=step, tracker=tracker, run_dir=run_dir)
        log.info("step %d  eval: %s", step, metrics)
        if tracker is not None:
            tracker.log_metrics({f"eval/{k}": v for k, v in metrics.items()}, step=step)
        return metrics

    # Checkpoint metadata: ACCUMULATE config + provenance history across resumes
    # (train/loop.py semantics — `to_hf` reads training_config_history to recover
    # `pretrain_view_scale`, the foveated footgun; a single snapshot would lose the
    # earlier legs of a multi-job run).
    from datetime import UTC, datetime

    from canvit_pretrain.checkpoint import current_provenance

    task_meta = task.checkpoint_metadata(model)
    prior_meta = (prior_ckpt or {}).get("metadata", {}) if prior_ckpt else {}
    cfg_history = dict(prior_meta.get("training_config_history") or {})
    prov_history = dict(prior_meta.get("provenance_history") or {})
    _now = datetime.now(UTC).isoformat()
    cfg_history[_now] = {**task_meta, "train_spec": str(spec)}
    prov_history[_now] = current_provenance()
    metadata = {
        **task_meta,
        # What the NEXT job needs to resume this task's DATA schedule (distill's
        # WebDataset job_index + the invariants it is only valid under; {} for the
        # map-style tasks). Built after build_loaders, which is where it becomes known.
        "resume_state": task.resume_state(),
        "training_config_history": cfg_history,
        "provenance_history": prov_history,
    }

    try:
        last = run_training_loop(
            task=task, model=model, head=head, optimizer=optimizer, scheduler=scheduler,
            selector=selector, spec=spec, branches=branches, canvas_grid=canvas_grid, device=device,
            train_batches=train_batches, n_steps=settings.n_steps, start_step=start_step,
            joint=joint, amp_ctx=amp_ctx, grad_clip=settings.grad_clip, is_dist=is_dist,
            task_name=task.name, model_config=task.model_config(model),
            metadata=metadata, log_every=settings.log_every,
            ckpt_dir=ckpt_dir, ckpt_every=settings.ckpt_every, eval_every=settings.eval_every,
            ema_alpha=settings.ema_alpha, log_grad_norms=settings.log_grad_norms,
            log_timing=settings.log_timing,
            viz_every=settings.viz_every, run_dir=run_dir,
            grad_norm_deep_prefixes=settings.grad_norm_deep_prefixes,
            signal_checkpoint=settings.signal_checkpoint,
            on_log=on_log, on_eval=(on_eval if settings.eval_every else None),
        )
    except Exception:
        # SLURM crash-loop guard: leave a FAILED marker + stop the array so the next
        # task doesn't re-crash on the same fault (opt-in; matches train/loop.py).
        if settings.use_failed_marker and failed_marker is not None and settings.rank == 0:
            failed_marker.parent.mkdir(parents=True, exist_ok=True)
            failed_marker.write_text("crashed\n")
            cancel_slurm_array()
        raise
    if tracker is not None:
        tracker.end()
    log.info("run complete: %s", last)
    return last


def _build_task(task_name: str, overrides: dict):
    """Construct a task's default per-task config (env-driven defaults) with a curated
    set of CLI overrides applied, and wrap it in its ``*RunTask``."""
    from dataclasses import replace

    if task_name == "ade20k":
        from canvit_pretrain.ade20k.config import Ade20kConfig
        from canvit_pretrain.tasks.ade20k.task import Ade20kRunTask
        cfg = Ade20kConfig()
        for k in ("model_repo", "batch_size", "n_timesteps", "scene_size", "ade20k_root"):
            if overrides.get(k) is not None:
                cfg = replace(cfg, **{k: overrides[k]})
        if overrides.get("lr") is not None:
            cfg = replace(cfg, peak_lr=overrides["lr"])
        return Ade20kRunTask(cfg)
    if task_name == "in1k":
        from canvit_pretrain.in1k.config import In1kConfig
        from canvit_pretrain.tasks.in1k.task import In1kRunTask
        cfg = In1kConfig()
        for k in ("model_repo", "batch_size", "n_timesteps", "scene_size", "train_dir", "val_dir", "mode"):
            if overrides.get(k) is not None:
                cfg = replace(cfg, **{k: overrides[k]})
        if overrides.get("lr") is not None:
            cfg = replace(cfg, peak_lr=overrides["lr"])
        return In1kRunTask(cfg)
    if task_name == "distill":
        from canvit_pretrain.train.config import Config
        from canvit_pretrain.tasks.distill.task import DistillRunTask
        cfg = Config()
        wds = overrides.get("webdataset_dir") or os.environ.get("WEBDATASET_DIR")
        if wds is not None:
            cfg = replace(cfg, webdataset_dir=Path(wds))
        for k in ("batch_size_per_gpu", "canvas_patch_grid_size"):
            if overrides.get(k) is not None:
                cfg = replace(cfg, **{k: overrides[k]})
        if overrides.get("lr") is not None:
            cfg = replace(cfg, peak_lr=overrides["lr"])
        return DistillRunTask(cfg)
    raise ValueError(f"unknown task {task_name!r}")


def _resolve_spec(task, preset: str, lr: float | None, wd: float | None) -> TrainSpec:
    """Pick the spec from ``--preset`` (``default`` = the task's own default_spec) and
    ensure every trainable module has an optimizer group (presets ship empty ``optim``)."""
    from dataclasses import replace

    from canvit_pretrain.harness.spec import BpttSpec, GroupOptim
    from canvit_pretrain.train.config import JointPolicyConfig

    if preset == "default":
        return task.default_spec()
    horizon = getattr(task.cfg, "n_timesteps", 10)
    bptt_none = BpttSpec(mode="none", horizon=horizon)
    bptt_full = BpttSpec(mode="full", horizon=horizon)
    if preset == "probe":
        spec = TrainSpec.probe(bptt=bptt_none)
    elif preset == "finetune":
        spec = TrainSpec.finetune(bptt=bptt_full)
    elif preset == "policy_only":
        spec = TrainSpec.policy_only(bptt=bptt_none)
    elif preset == "joint":
        spec = TrainSpec.joint()
    else:
        raise ValueError(f"unknown preset {preset!r}")
    # Headless tasks (distill: heads live inside the forward, caps.has_head=False) can't
    # train a separate head — drop train_head so the head-bearing presets still apply
    # (distill 'finetune' => task-only backbone; distill 'joint' => backbone + policy).
    if not task.caps().has_head and spec.train_head:
        spec = replace(spec, train_head=False)
    # fill optim for trainable modules that lack a group
    _lr = lr if lr is not None else 3e-4
    _wd = wd if wd is not None else 1e-3
    pol = JointPolicyConfig()
    optim = dict(spec.optim)
    for m in spec.trainable_modules():
        if m in optim:
            continue
        if m == "policy":
            optim[m] = GroupOptim(lr=pol.policy_lr, weight_decay=pol.policy_weight_decay)
        else:
            optim[m] = GroupOptim(lr=_lr, weight_decay=_wd)
    return replace(spec, optim=optim)


def main(argv: list[str] | None = None) -> None:
    """Additive unified CLI: ``python -m canvit_pretrain.harness.run --task {distill,ade20k,in1k}``.

    Deliberately does NOT replace the live ``python -m canvit_pretrain.train`` (distill)
    entry — the big-bang cutover (owner-gated) repoints that name here and rewrites the
    SLURM launchers. This is a v1 curated surface (preset + common knobs); full per-task
    config CLI parity lands at the cutover."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser("canvit_pretrain.harness.run")
    p.add_argument("--task", required=True, choices=["distill", "ade20k", "in1k"])
    p.add_argument("--preset", default="default",
                   choices=["default", "probe", "finetune", "policy_only", "joint"])
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--start-step", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--tracker", default="none", choices=["none", "wandb"])
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--run-name", default=None)
    # curated per-task overrides (paths otherwise come from env defaults)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--wd", type=float, default=None)
    p.add_argument("--model-repo", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-timesteps", type=int, default=None)
    p.add_argument("--scene-size", type=int, default=None)
    p.add_argument("--mode", default=None, choices=[None, "frozen", "finetune"])
    p.add_argument("--webdataset-dir", default=None)
    p.add_argument("--ade20k-root", type=Path, default=None)
    p.add_argument("--in1k-train-dir", type=Path, default=None)
    p.add_argument("--in1k-val-dir", type=Path, default=None)
    a = p.parse_args(argv)

    overrides = {
        "model_repo": a.model_repo, "batch_size": a.batch_size, "batch_size_per_gpu": a.batch_size,
        "n_timesteps": a.n_timesteps, "scene_size": a.scene_size, "mode": a.mode,
        "webdataset_dir": a.webdataset_dir, "ade20k_root": a.ade20k_root,
        "train_dir": a.in1k_train_dir, "val_dir": a.in1k_val_dir, "lr": a.lr,
    }
    task = _build_task(a.task, overrides)
    spec = _resolve_spec(task, a.preset, a.lr, a.wd)
    settings = RunSettings(
        n_steps=a.n_steps, start_step=a.start_step, grad_clip=a.grad_clip, amp=not a.no_amp,
        device=a.device, seed=a.seed, log_every=a.log_every, ckpt_every=a.ckpt_every,
        eval_every=a.eval_every, ckpt_dir=a.ckpt_dir, tracker=a.tracker,
        wandb_project=a.wandb_project, wandb_entity=a.wandb_entity, run_name=a.run_name,
    )
    run(task=task, spec=spec, settings=settings)


__all__ = ["RunSettings", "RunTask", "run", "main"]


if __name__ == "__main__":
    main()
