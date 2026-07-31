"""Real IN1k classification training through the NEW harness loop on GPU.

Drives ``harness/loop.py::run_training_loop`` with a run-level IN1k task over the
real IN1k WebDataset shards + the cached pretrained CanViT, in two configs:

  * FROZEN   — freeze the backbone, train LN+head only (``TrainSpec.probe``,
               bptt='none'); the direct analogue of the historical in1k frozen
               linear-clf-probe. The engine's single end-of-rollout backward over
               ``mean_t(CE)`` matches in1k/train.py's ``stack(per_t CE).mean()``.
  * FINETUNE — train backbone + LN + head end to end (``TrainSpec.finetune``,
               chunked TBPTT), the ``...-finetune-...-in1k`` flagship regime.

NOTE on the IN1k "head": the trainable head is LN(``clf.norm``) + Linear(``clf.head``).
``from_pretrained_with_new_head`` leaves ``clf.norm`` at requires_grad=True and
``apply_requires_grad`` freezes only the trunk (``clf.canvit``), so norm stays
trainable in both configs; the optimizer's "head" group is norm+head params. The
run-level Task wrapper will formalize this (head = norm+head).

Run (offline):
  HF_HOME=... HF_HUB_OFFLINE=1 .venv-cu126/bin/python unification_docs/harness_realdata_in1k.py
"""

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from canvit_pytorch import CanViTForImageClassification

from canvit_train.harness.config import FoveatedScaleConfig
from canvit_train.harness.infra.checkpoint import find_latest, load_checkpoint, restore_into
from canvit_train.harness.loop import apply_requires_grad, run_training_loop
from canvit_train.harness.optim import build_optimizer_and_scheduler
from canvit_train.harness.rollout.selector import RandomSelector
from canvit_train.harness.rollout.viewpoint import ViewpointType
from canvit_train.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TaskCaps, TrainSpec
from canvit_train.in1k.config import NUM_CLASSES, In1kConfig
from canvit_train.in1k.data import make_train_loader
from canvit_train.in1k.rollout import consumes_full_image
from canvit_train.in1k.task import BoundIn1kTask

TRAIN_DIR = Path("/user/henrich1/u25995/jonathan/datasets/"
                 "webdataset-imagenet-1k-no-features/train-shuffled")
# Checkpoints go on disk-backed vast-nhr, NOT the 4GB /tmp tmpfs (full-backbone +
# AdamW optimizer state is >1GB per finetune checkpoint).
SCRATCH = Path("/mnt/vast-nhr/projects/nib00021/jonathan/_harness_smoke_ckpts")
_BATCH, _N, _CANVAS = 8, 24, 32


class In1kRunTask:
    def __init__(self, canvas_grid: int, glimpse_px: int | None, label_smoothing: float):
        self.canvas_grid, self.glimpse_px, self.ls = canvas_grid, glimpse_px, label_smoothing

    def batch_images(self, batch, device):
        return batch[0].to(device, non_blocking=True)

    def bind(self, batch, device, *, model, head):
        _, labels = batch
        return BoundIn1kTask(
            clf=model, targets=torch.as_tensor(labels, dtype=torch.long, device=device),
            canvas_grid=self.canvas_grid, glimpse_px=self.glimpse_px, label_smoothing=self.ls,
        )


def _build_clf(device):
    cfg = In1kConfig(train_dir=TRAIN_DIR, scene_size=512, batch_size=_BATCH,
                     num_workers=4, tracker="none")
    clf = CanViTForImageClassification.from_pretrained_with_new_head(
        pretrained_repo=cfg.model_repo, n_classes=NUM_CLASSES,
    ).to(device)
    return cfg, clf


def _head_params(clf):
    return list(clf.norm.parameters()) + list(clf.head.parameters())


def _run_config(name, *, spec, clf, cfg, param_groups, device, train_batches, ckpt_dir):
    spec.validate(TaskCaps(has_head=True, supports_policy=True))
    apply_requires_grad(model=clf, head=clf.head, joint=None, spec=spec)
    opt, sched = build_optimizer_and_scheduler(spec, param_groups)
    selector = RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(),
                              min_viewpoint_scale=0.05)
    losses: list[float] = []

    def on_log(step, m):
        losses.append(m["total_loss"])
        print(f"  [{name}] step {step:3d}  loss={m['total_loss']:.4f}  "
              f"n_glimpses={m['n_glimpses']}  lr={sched.get_last_lr()[0]:.2e}", flush=True)

    run_training_loop(
        task=In1kRunTask(_CANVAS, cfg.glimpse_px, cfg.label_smoothing), model=clf, head=clf.head,
        optimizer=opt, scheduler=sched, selector=selector, spec=spec,
        branches=[ViewpointType.RANDOM], canvas_grid=_CANVAS, device=device,
        train_batches=train_batches, n_steps=_N, task_name="in1k",
        model_config={"n_classes": NUM_CLASSES, "canvas_grid": _CANVAS},
        amp_ctx=torch.autocast("cuda", dtype=torch.bfloat16), grad_clip=float("inf"),
        log_every=4, ckpt_dir=ckpt_dir, on_log=on_log,
    )
    assert losses, f"[{name}] on_log never fired"
    early, late = sum(losses[:2]) / 2, sum(losses[-2:]) / 2
    # A fresh 1000-class head starts near ln(1000)≈6.9; assert it's finite + started high.
    print(f"  [{name}] early={early:.4f} late={late:.4f} delta={late - early:+.4f}", flush=True)
    assert all(torch.isfinite(torch.tensor(x)) for x in losses), f"[{name}] non-finite loss"
    return early, late


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  torch={torch.__version__}")
    assert dev.type == "cuda"
    torch.manual_seed(0)

    cfg, clf = _build_clf(dev)
    loader, batches_per_epoch = make_train_loader(cfg, world_size=1, rank=0)
    print(f"model={cfg.model_repo}  canvas={_CANVAS}  glimpse_px={cfg.glimpse_px}  "
          f"full_image={consumes_full_image(clf)}  batches/epoch={batches_per_epoch}", flush=True)
    it = iter(loader)  # shared across configs (webdataset resampled → effectively infinite)

    # ── FROZEN (probe) ─────────────────────────────────────────────────────
    frozen_spec = TrainSpec.probe(
        bptt=BpttSpec(mode="none", horizon=6),
        optim={"head": GroupOptim(lr=3e-4, weight_decay=1e-3,
               schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=5, warmup_lr_ratio=1e-2))},
    )
    frozen_ckpt = SCRATCH / "in1k_frozen_ckpt"
    _run_config("frozen", spec=frozen_spec, clf=clf, cfg=cfg,
                param_groups={"head": _head_params(clf)},
                device=dev, train_batches=it, ckpt_dir=frozen_ckpt)
    latest = find_latest(frozen_ckpt)
    assert latest is not None and latest.name == f"step-{_N}.pt", latest
    payload = load_checkpoint(latest, "cpu")
    _, fresh = _build_clf("cpu")
    restore_into(payload, model=fresh)
    ok_frozen = all(torch.allclose(p.cpu(), q) for p, q in
                    zip(clf.state_dict().values(), fresh.state_dict().values()))
    print(f"  [frozen] checkpoint reload matches: {ok_frozen}", flush=True)

    # ── FINETUNE (backbone + LN + head) ─────────────────────────────────────
    _, clf2 = _build_clf(dev)
    ft_spec = TrainSpec.finetune(
        bptt=BpttSpec(mode="chunked", chunk_size=2, horizon=4),
        optim={
            "backbone": GroupOptim(lr=1e-5, weight_decay=1e-4,
                schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=5, warmup_lr_ratio=1e-2)),
            "head": GroupOptim(lr=3e-4, weight_decay=1e-3,
                schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=5, warmup_lr_ratio=1e-2)),
        },
    )
    ft_ckpt = SCRATCH / "in1k_ft_ckpt"
    _run_config("finetune", spec=ft_spec, clf=clf2, cfg=cfg,
                param_groups={"backbone": list(clf2.canvit.parameters()), "head": _head_params(clf2)},
                device=dev, train_batches=it, ckpt_dir=ft_ckpt)
    latest = find_latest(ft_ckpt)
    payload = load_checkpoint(latest, "cpu")
    _, fresh2 = _build_clf("cpu")
    restore_into(payload, model=fresh2)
    ok_ft = all(torch.allclose(p.cpu(), q) for p, q in
                zip(clf2.state_dict().values(), fresh2.state_dict().values()))
    print(f"  [finetune] checkpoint reload matches: {ok_ft}", flush=True)

    print("PASS: harness trains IN1k (frozen + finetune) on real data + checkpoints round-trip"
          if (ok_frozen and ok_ft) else "FAIL")


if __name__ == "__main__":
    main()
