"""Full-fidelity CLI for the unified harness (``python -m canvit_pretrain.harness.run``).

Built on ``tyro`` over each task's OWN config dataclass — the same idiom the three
standalone entry points already use (``train/__main__.py``, ``ade20k/__main__.py``,
``in1k/__main__.py``). Every config field is therefore reachable from the command
line, including the nested trees a hand-rolled argparse could not express:

    # foveated pretraining (exp22-style)
    python -m canvit_pretrain.harness.run distill \
        --cfg.model.patcher-name foveated --cfg.foveated-scale.mode per_rollout \
        --cfg.run-group foveated --cfg.webdataset-dir /path/to/wds

    # ADE20K linear probe, and joint probe+policy
    python -m canvit_pretrain.harness.run ade20k --preset probe
    python -m canvit_pretrain.harness.run ade20k --preset joint --rl.use-rl True

**The task config is the source of truth.** :class:`~canvit_pretrain.harness.run.RunSettings`
is DERIVED from it — ``cfg.steps_per_job`` sets ``n_steps``, ``cfg.val_every`` sets
``eval_every``, and ``compile`` / ``amp`` / ``grad_clip`` / ``tracker`` / ``wandb_*`` /
``seed_ckpt`` carry over — so a config that reproduced a run under the old entry point
reproduces it here, with no second place to set the same thing. The handful of knobs
with no task-config counterpart live in :class:`HarnessOpts` (``--opts.*``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import tyro

from canvit_pretrain.ade20k.config import Ade20kConfig
from canvit_pretrain.harness.run import RunSettings, run
from canvit_pretrain.harness.spec import BpttSpec, GroupOptim, TrainSpec
from canvit_pretrain.in1k.config import In1kConfig
from canvit_pretrain.train.config import Config, JointPolicyConfig

log = logging.getLogger(__name__)

PresetName = Literal["default", "probe", "finetune", "policy_only", "joint"]


@dataclass
class HarnessOpts:
    """Harness-level knobs with no task-config counterpart (everything else is read
    off the task config, which stays the single source of truth)."""

    n_steps: int | None = None
    """Steps this job runs. None => the task's natural job length (distill:
    ``cfg.steps_per_job``, ade20k/in1k: ``cfg.max_steps``). Set it explicitly for
    short smoke runs."""
    eval_every: int | None = None
    """Validate every N steps. None => the task's own cadence (``cfg.val_every``)."""
    seed: int | None = None
    """RNG seed (``torch.manual_seed(seed + rank)``). None => the task config's own seed
    (distill/in1k ``cfg.seed``; ade20k has no seed field so None => 0). Set it to compare
    seed-to-seed variability — e.g. against the UNSEEDED standalone ade20k probe."""
    start_step: int = 0
    ckpt_every: int = 0
    """Periodic checkpoints every N steps (0 => only the end-of-run checkpoint)."""
    viz_every: int = 0
    """Render the task's training-batch visualization every N steps (distill only)."""
    ckpt_dir: Path | None = None
    """Explicit checkpoint dir. Overrides the ``logs_dir/run_group/run_name`` convention."""
    resume: bool | None = None
    """Resume from ``find_latest(ckpt_dir)`` if a checkpoint exists. None => the task
    default: True for distill (array jobs must continue across tasks), False for the
    ade20k/in1k probes (single-job launchers mirroring the no-resume standalone, so
    re-running into a populated dir starts fresh instead of silently continuing the
    old run). Pass ``--opts.resume True`` to opt a probe into array-style resume."""
    signal_checkpoint: bool = True
    use_failed_marker: bool = False
    """SLURM crash-loop guard: write a FAILED marker and scancel the array on crash."""
    amp_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    log_grad_norms: bool = True
    log_timing: bool = True


def _resolve_run_dir(logs_dir: Path, run_group: str | None,
                     run_name: str | None) -> tuple[Path | None, str]:
    """The ``logs_dir/run_group/run_name`` convention (train/loop.py 157-161): an
    auto-generated timestamp name when unset, and no run dir at all without a group."""
    name = run_name or datetime.now().strftime("%Y-%m-%d_%H-%M")
    if run_group is None:
        return None, name
    return logs_dir / run_group / name, name


def _tracker(kind: str) -> str:
    if kind == "comet":
        raise NotImplementedError(
            "the harness supports tracker='wandb' or 'none' (comet was not ported); "
            "pass --cfg.tracker wandb")
    return kind


def _common(opts: HarnessOpts) -> dict[str, Any]:
    """HarnessOpts fields that map straight onto RunSettings."""
    return {
        "start_step": opts.start_step, "ckpt_every": opts.ckpt_every,
        "viz_every": opts.viz_every, "ckpt_dir": opts.ckpt_dir,
        "signal_checkpoint": opts.signal_checkpoint,
        "use_failed_marker": opts.use_failed_marker, "amp_dtype": opts.amp_dtype,
        "log_grad_norms": opts.log_grad_norms, "log_timing": opts.log_timing,
    }


@dataclass
class DistillCmd:
    """Pretraining: passive -> active dense latent distillation from DINOv3."""

    cfg: Config = field(default_factory=Config)
    preset: PresetName = "default"
    opts: HarnessOpts = field(default_factory=HarnessOpts)

    def build(self) -> tuple[Any, RunSettings]:
        from canvit_pretrain.tasks.distill.task import DistillRunTask

        run_dir, run_name = _resolve_run_dir(self.cfg.logs_dir, self.cfg.run_group,
                                             self.cfg.run_name)
        settings = RunSettings(
            # The shard-schedule window IS the job length for distill: a job trains
            # exactly steps_per_job steps and the next array task resumes at the next
            # shard block. Decoupling them corrupts the WebDataset resume.
            n_steps=self.opts.n_steps if self.opts.n_steps is not None else self.cfg.steps_per_job,
            eval_every=self.opts.eval_every if self.opts.eval_every is not None else self.cfg.val_every,
            log_every=self.cfg.log_every, grad_clip=self.cfg.grad_clip, amp=self.cfg.amp,
            seed=self.opts.seed if self.opts.seed is not None else self.cfg.seed,
            device=str(self.cfg.device), compile=self.cfg.compile,
            ema_alpha=self.cfg.ema_alpha, seed_ckpt=self.cfg.seed_ckpt,
            tracker=_tracker(self.cfg.tracker), wandb_project=self.cfg.wandb_project,
            wandb_entity=self.cfg.wandb_entity, wandb_dir=self.cfg.wandb_dir,
            run_name=run_name, run_dir=run_dir,
            resume=self.opts.resume if self.opts.resume is not None else True,
            **_common(self.opts),
        )
        return DistillRunTask(self.cfg), settings

    def lr_wd(self) -> tuple[float, float]:
        return self.cfg.peak_lr, self.cfg.weight_decay


@dataclass
class Ade20kCmd:
    """ADE20K semantic segmentation: frozen probe, finetune, or joint probe+policy."""

    cfg: Ade20kConfig = field(default_factory=Ade20kConfig)
    preset: PresetName = "default"
    rl: JointPolicyConfig = field(default_factory=JointPolicyConfig)
    """Viewpoint-policy config; only consulted for policy/joint presets."""
    opts: HarnessOpts = field(default_factory=HarnessOpts)

    def build(self) -> tuple[Any, RunSettings]:
        from canvit_pretrain.tasks.ade20k.task import Ade20kRunTask

        settings = RunSettings(
            n_steps=self.opts.n_steps if self.opts.n_steps is not None else self.cfg.max_steps,
            eval_every=self.opts.eval_every if self.opts.eval_every is not None else self.cfg.val_every,
            log_every=self.cfg.log_every, grad_clip=self.cfg.grad_clip, amp=self.cfg.amp,
            seed=self.opts.seed if self.opts.seed is not None else 0,
            device=self.cfg.device, tracker=_tracker(self.cfg.tracker),
            wandb_project=self.cfg.wandb_project, wandb_entity=self.cfg.wandb_entity,
            wandb_dir=self.cfg.wandb_dir, run_name="ade20k",
            resume=self.opts.resume if self.opts.resume is not None else False,
            **{**_common(self.opts),
               "ckpt_dir": self.opts.ckpt_dir or self.cfg.probe_ckpt_dir},
        )
        return Ade20kRunTask(self.cfg, rl=self.rl), settings

    def lr_wd(self) -> tuple[float, float]:
        return self.cfg.peak_lr, self.cfg.weight_decay


@dataclass
class In1kCmd:
    """ImageNet-1k classification: frozen linear probe, finetune, or joint clf+policy."""

    cfg: In1kConfig = field(default_factory=In1kConfig)
    preset: PresetName = "default"
    rl: JointPolicyConfig = field(default_factory=JointPolicyConfig)
    opts: HarnessOpts = field(default_factory=HarnessOpts)

    def build(self) -> tuple[Any, RunSettings]:
        from canvit_pretrain.tasks.in1k.task import In1kRunTask

        n_steps = self.opts.n_steps if self.opts.n_steps is not None else self.cfg.max_steps
        settings = RunSettings(
            n_steps=n_steps,
            eval_every=self.opts.eval_every if self.opts.eval_every is not None else self.cfg.val_every,
            log_every=self.cfg.log_every, grad_clip=self.cfg.grad_clip, amp=self.cfg.amp,
            seed=self.opts.seed if self.opts.seed is not None else self.cfg.seed,
            device=self.cfg.device, tracker=_tracker(self.cfg.tracker),
            wandb_project=self.cfg.wandb_project, wandb_entity=self.cfg.wandb_entity,
            wandb_dir=self.cfg.wandb_dir, run_name=self.cfg.run_name,
            resume=self.opts.resume if self.opts.resume is not None else False,
            **{**_common(self.opts),
               "ckpt_dir": self.opts.ckpt_dir or self.cfg.clf_ckpt_dir},
        )
        return In1kRunTask(self.cfg, rl=self.rl, total_steps=n_steps), settings

    def lr_wd(self) -> tuple[float, float]:
        return self.cfg.peak_lr, self.cfg.weight_decay


def resolve_spec(task: Any, preset: str, lr: float, wd: float) -> TrainSpec:
    """Pick the spec from ``preset`` (``default`` = the task's own ``default_spec``) and
    give every trainable module an optimizer group (the presets ship an empty ``optim``,
    filled here from the task config's peak_lr / weight_decay)."""
    from dataclasses import replace

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
    # ade20k/in1k carry the policy config on the task (passed in); distill keeps it
    # inside its own config as `cfg.rl`.
    pol = getattr(task, "rl", None) or getattr(task.cfg, "rl", None) or JointPolicyConfig()
    optim = dict(spec.optim)
    for m in spec.trainable_modules():
        if m in optim:
            continue
        optim[m] = (GroupOptim(lr=pol.policy_lr, weight_decay=pol.policy_weight_decay)
                    if m == "policy" else GroupOptim(lr=lr, weight_decay=wd))
    return replace(spec, optim=optim)


Command = (
    Annotated[DistillCmd, tyro.conf.subcommand(name="distill")]
    | Annotated[Ade20kCmd, tyro.conf.subcommand(name="ade20k")]
    | Annotated[In1kCmd, tyro.conf.subcommand(name="in1k")]
)


def main(argv: list[str] | None = None) -> dict:
    import torch

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # TF32 matmuls for the fp32 paths — train/__main__.py sets this before building
    # anything, and it changes both speed and numerics.
    torch.set_float32_matmul_precision("high")

    cmd: Any = tyro.cli(Command, args=argv)
    task, settings = cmd.build()
    spec = resolve_spec(task, cmd.preset, *cmd.lr_wd())
    log.info("task=%s preset=%s n_steps=%d eval_every=%d run_dir=%s",
             task.name, cmd.preset, settings.n_steps, settings.eval_every, settings.run_dir)
    return run(task=task, spec=spec, settings=settings)


__all__ = ["Ade20kCmd", "DistillCmd", "HarnessOpts", "In1kCmd", "main", "resolve_spec"]
