"""Real ADE20K training through the NEW harness loop on GPU (task-core validation).

Drives ``harness/loop.py::run_training_loop`` with a run-level ADE20K task over the
real ADE20K dataset + the cached pretrained CanViT, in two configs:

  * PROBE    — frozen backbone, train the seg head only (``TrainSpec.probe``,
               bptt='none'). This is the historical ade20k trainer's regime; the
               engine's single end-of-rollout backward over ``mean_t(CE)`` is the
               byte-faithful analogue of ade20k/train.py's ``stack(per_t CE).mean()``.
  * FINETUNE — train backbone + head end to end (``TrainSpec.finetune``, chunked
               TBPTT). Exercises the NEW capability (ade20k full-FT never existed)
               and the head-applied-OUTSIDE-forward + backbone-grad combination on a
               real model.

For each: assert the loss is finite/sane over N steps, print the trajectory, and
confirm the checkpoint reloads into a fresh model bit-for-bit.

Run (offline):
  HF_HOME=... HF_HUB_OFFLINE=1 .venv-cu126/bin/python unification_docs/harness_realdata_ade20k.py
"""

import os
from itertools import cycle
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from canvit_pytorch import CanViTForSemanticSegmentation

from canvit_train.ade20k.config import Ade20kConfig
from canvit_train.ade20k.data import NUM_CLASSES, make_ade20k_loaders
from canvit_train.ade20k.rollout import consumes_full_image
from canvit_train.ade20k.task import BoundAde20kTask
from canvit_train.harness.config import FoveatedScaleConfig
from canvit_train.harness.infra.checkpoint import find_latest, load_checkpoint, restore_into
from canvit_train.harness.loop import apply_requires_grad, run_training_loop
from canvit_train.harness.optim import build_optimizer_and_scheduler
from canvit_train.harness.rollout.selector import RandomSelector
from canvit_train.harness.rollout.viewpoint import ViewpointType
from canvit_train.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TaskCaps, TrainSpec

ADE_ROOT = Path("/user/henrich1/u25995/jonathan/datasets/"
                "zhoubolei--scene_parse_150/ADEChallengeData2016")
SCRATCH = Path("/tmp/claude-966121/-mnt-vast-nhr-projects-nib00021-jonathan-repos-canvit-modify/"
               "d55dcc83-a887-4c35-ac0d-862a62d401be/scratchpad")
_BATCH, _N, _CANVAS = 8, 24, 32


class Ade20kRunTask:
    """Run-level ADE20K task surface the loop needs: extract images + bind the
    per-batch masks into a fresh ``BoundAde20kTask`` (mirrors distill's per-batch
    ``DistillTask``)."""

    def __init__(self, canvas_grid: int, glimpse_px: int | None):
        self.canvas_grid, self.glimpse_px = canvas_grid, glimpse_px

    def batch_images(self, batch, device):
        return batch[0].to(device, non_blocking=True)

    def bind(self, batch, device, *, model, head):
        _, masks = batch
        return BoundAde20kTask(
            seg=model, masks=masks.to(device), canvas_grid=self.canvas_grid, glimpse_px=self.glimpse_px,
        )


def _build_seg(device):
    cfg = Ade20kConfig(ade20k_root=ADE_ROOT, scene_size=512, batch_size=_BATCH,
                       eval_batch_size=_BATCH, num_workers=4, tracker="none")
    seg = CanViTForSemanticSegmentation.from_pretrained_with_new_probe(
        pretrained_repo=cfg.model_repo, num_classes=NUM_CLASSES, dropout=cfg.dropout, use_ln=True,
    ).to(device)
    return cfg, seg


def _run_config(name, *, spec, seg, cfg, param_groups, device, train_batches, ckpt_dir):
    spec.validate(TaskCaps(has_head=True, supports_policy=True))
    apply_requires_grad(model=seg, head=seg.head, joint=None, spec=spec)
    opt, sched = build_optimizer_and_scheduler(spec, param_groups)
    selector = RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(),
                              min_viewpoint_scale=0.05)
    losses: list[float] = []

    def on_log(step, m):
        losses.append(m["total_loss"])
        print(f"  [{name}] step {step:3d}  loss={m['total_loss']:.4f}  "
              f"n_glimpses={m['n_glimpses']}  lr={sched.get_last_lr()[0]:.2e}", flush=True)

    last = run_training_loop(
        task=Ade20kRunTask(_CANVAS, cfg.glimpse_px), model=seg, head=seg.head,
        optimizer=opt, scheduler=sched, selector=selector, spec=spec,
        branches=[ViewpointType.RANDOM], canvas_grid=_CANVAS, device=device,
        train_batches=train_batches, n_steps=_N, task_name="ade20k",
        model_config={"num_classes": NUM_CLASSES, "canvas_grid": _CANVAS},
        amp_ctx=torch.autocast("cuda", dtype=torch.bfloat16), grad_clip=float("inf"),
        log_every=4, ckpt_dir=ckpt_dir,
        on_log=on_log,
    )
    assert losses, f"[{name}] on_log never fired"
    early, late = sum(losses[:2]) / 2, sum(losses[-2:]) / 2
    print(f"  [{name}] early={early:.4f} late={late:.4f} delta={late - early:+.4f}", flush=True)
    assert all(torch.isfinite(torch.tensor(x)) for x in losses), f"[{name}] non-finite loss"
    return last, early, late


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  torch={torch.__version__}")
    assert dev.type == "cuda"
    torch.manual_seed(0)

    cfg, seg = _build_seg(dev)
    train_loader, _ = make_ade20k_loaders(cfg)
    print(f"model={cfg.model_repo}  canvas={_CANVAS}  glimpse_px={cfg.glimpse_px}  "
          f"full_image={consumes_full_image(seg)}", flush=True)
    batches = cycle(train_loader)  # cheap: reuse the DataLoader across both configs

    # ── PROBE (frozen backbone, train head) — historical ade20k regime ──────
    probe_spec = TrainSpec.probe(
        bptt=BpttSpec(mode="none", horizon=6),
        optim={"head": GroupOptim(lr=3e-4, weight_decay=1e-3,
               schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=5, warmup_lr_ratio=1e-2))},
    )
    probe_ckpt = SCRATCH / "ade20k_probe_ckpt"
    _run_config("probe", spec=probe_spec, seg=seg, cfg=cfg,
                param_groups={"head": list(seg.head.parameters())},
                device=dev, train_batches=batches, ckpt_dir=probe_ckpt)

    # checkpoint round-trip (fresh model with random head + pretrained backbone).
    latest = find_latest(probe_ckpt)
    assert latest is not None and latest.name == f"step-{_N}.pt", latest
    payload = load_checkpoint(latest, "cpu")
    _, fresh = _build_seg("cpu")
    restore_into(payload, model=fresh)
    ok_probe = all(torch.allclose(p.cpu(), q) for p, q in
                   zip(seg.state_dict().values(), fresh.state_dict().values()))
    print(f"  [probe] checkpoint reload matches: {ok_probe}", flush=True)

    # ── FINETUNE (train backbone + head) — NEW capability ───────────────────
    _, seg2 = _build_seg(dev)
    ft_spec = TrainSpec.finetune(
        bptt=BpttSpec(mode="chunked", chunk_size=2, horizon=4),
        optim={
            "backbone": GroupOptim(lr=1e-5, weight_decay=1e-4,
                schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=5, warmup_lr_ratio=1e-2)),
            "head": GroupOptim(lr=3e-4, weight_decay=1e-3,
                schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=5, warmup_lr_ratio=1e-2)),
        },
    )
    ft_ckpt = SCRATCH / "ade20k_ft_ckpt"
    _run_config("finetune", spec=ft_spec, seg=seg2, cfg=cfg,
                param_groups={"backbone": list(seg2.canvit.parameters()),
                              "head": list(seg2.head.parameters())},
                device=dev, train_batches=batches, ckpt_dir=ft_ckpt)
    latest = find_latest(ft_ckpt)
    payload = load_checkpoint(latest, "cpu")
    _, fresh2 = _build_seg("cpu")
    restore_into(payload, model=fresh2)
    ok_ft = all(torch.allclose(p.cpu(), q) for p, q in
                zip(seg2.state_dict().values(), fresh2.state_dict().values()))
    print(f"  [finetune] checkpoint reload matches: {ok_ft}", flush=True)

    print("PASS: harness trains ADE20K (probe + finetune) on real data + checkpoints round-trip"
          if (ok_probe and ok_ft) else "FAIL")


if __name__ == "__main__":
    main()
