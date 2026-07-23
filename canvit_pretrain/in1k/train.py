"""IN1k classification training (unification P5): frozen-backbone linear probe
(default) or full finetune of CanViTForImageClassification over a glimpse rollout,
DDP epochs on the on-cluster WebDataset train shards.

The rollout steps ``clf.canvit`` directly (per ade20k/rollout.py + episode.py), so
a DDP wrapper around the classifier would not instrument that forward. As in P4b's
scorer, DDP is handled by hand instead: broadcast the trainable params once so all
ranks start identical, AllReduce their grads each step. Frozen mode trains only
LN+head (backbone under no_grad); finetune trains the whole classifier.

Validated on CPU for construction/rollout/data (test_in1k.py); the epoch loop's
real gate is a SLURM run vs canvit_eval's frozen-probe baseline (user's call).
"""

import logging
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as tdist
import torch.nn as nn
from canvit_pytorch import CanViTForImageClassification
from tqdm import tqdm

from ..train import dist as ddp
from ..train.scheduler import warmup_cosine_scheduler
from ..train.tracker import make_tracker
from .config import NUM_CLASSES, In1kConfig
from .data import make_train_loader, make_val_loader
from .metrics import TopKAccuracy, ce_loss
from .rollout import consumes_full_image, eval_viewpoints, make_random_viewpoints, rollout_cls_tokens

log = logging.getLogger(__name__)


def _trainable_params(clf: CanViTForImageClassification, mode: str) -> list[nn.Parameter]:
    if mode == "frozen":
        return list(clf.norm.parameters()) + list(clf.head.parameters())
    return [p for p in clf.parameters() if p.requires_grad]


def _broadcast(clf: CanViTForImageClassification) -> None:
    """Sync all params + buffers from rank 0 (the classifier is not DDP-wrapped)."""
    for p in clf.parameters():
        tdist.broadcast(p.data, src=0)
    ddp.broadcast_module_buffers(clf, src=0)


def _allreduce_grads(params: list[nn.Parameter]) -> None:
    for p in params:
        if p.grad is not None:
            tdist.all_reduce(p.grad, op=tdist.ReduceOp.AVG)


@torch.no_grad()
def evaluate(clf, cfg: In1kConfig, val_loader, *, device, canvas_grid, amp_ctx, is_foveated) -> dict[int, float]:
    """Deploy (argmax over the eval policy's viewpoints); global top-1/5 at the final
    timestep, aggregated across ranks."""
    clf.eval()
    acc = TopKAccuracy(ks=(1, 5))
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        vps = eval_viewpoints(
            cfg.eval_policy, images.shape[0], device, cfg.n_timesteps,
            is_foveated=is_foveated, foveated_scale=cfg.foveated_scale,
        )
        with amp_ctx:
            cls_tokens = rollout_cls_tokens(
                clf=clf, images=images, viewpoints=vps, canvas_grid=canvas_grid,
                glimpse_px=cfg.glimpse_px, freeze_backbone=True,
            )
        logits = clf.head(clf.norm(cls_tokens[-1]))
        acc.update(logits, labels)
        if cfg.limit_val_batches is not None and acc.total >= cfg.limit_val_batches * cfg.eval_batch_size:
            break
    # aggregate correct/total across ranks
    stats = torch.tensor([acc.correct[1], acc.correct[5], acc.total], dtype=torch.float64, device=device)
    if ddp.is_dist():
        tdist.all_reduce(stats, op=tdist.ReduceOp.SUM)
    top1, top5, total = stats.tolist()
    return {1: top1 / max(total, 1), 5: top5 / max(total, 1)}


def train(cfg: In1kConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_float32_matmul_precision("high")
    ddp.init_dist()
    device = ddp.device()
    torch.manual_seed(cfg.seed + ddp.rank())

    if ddp.is_main():
        log.info("=" * 60)
        log.info(f"IN1k classification ({cfg.mode}) — {cfg.model_repo}")
        log.info(f"epochs={cfg.epochs} scene={cfg.scene_size} T={cfg.n_timesteps} "
                 f"eval_policy={cfg.eval_policy} world_size={ddp.world_size()}")

    # Pretrained backbone + fresh head (probe training path).
    clf = CanViTForImageClassification.from_pretrained_with_new_head(
        pretrained_repo=cfg.model_repo, n_classes=NUM_CLASSES
    ).to(device)
    if cfg.mode == "frozen":
        clf.canvit.requires_grad_(False)
        clf.canvit.eval()
    else:
        clf.train()
    if ddp.is_dist():
        _broadcast(clf)  # identical start across ranks (no DDP wrapper)

    is_foveated = consumes_full_image(clf)
    patch_size = clf.canvit.backbone.patch_size_px
    canvas_grid = cfg.canvas_grid if cfg.canvas_grid is not None else cfg.scene_size // patch_size
    params = _trainable_params(clf, cfg.mode)
    if ddp.is_main():
        n_train = sum(p.numel() for p in params)
        log.info(f"  patcher={'foveated/square (full-image)' if is_foveated else 'uniform (pre-crop)'}, "
                 f"canvas={canvas_grid}x{canvas_grid}, trainable={n_train:,} ({cfg.mode})")

    train_loader, batches_per_epoch = make_train_loader(cfg, world_size=ddp.world_size(), rank=ddp.rank())
    if cfg.limit_train_batches is not None:
        batches_per_epoch = min(batches_per_epoch, cfg.limit_train_batches)
    have_val = cfg.val_dir.is_dir()
    val_loader = make_val_loader(cfg, world_size=ddp.world_size(), rank=ddp.rank()) if have_val else None
    if not have_val and ddp.is_main():
        log.warning(f"No val dir at {cfg.val_dir} (set IN1K_VAL_DIR) — training WITHOUT eval; "
                    f"the P5 acceptance gate needs it.")

    total_steps = cfg.epochs * batches_per_epoch
    warmup_steps = max(1, int(cfg.warmup_epochs * batches_per_epoch))
    opt = torch.optim.AdamW(params, lr=cfg.peak_lr, weight_decay=cfg.weight_decay)
    scheduler = warmup_cosine_scheduler(opt, warmup_steps, total_steps, cfg.peak_lr,
                                        start_lr=cfg.peak_lr * cfg.warmup_lr_ratio)

    amp_dtype = torch.bfloat16 if cfg.amp else torch.float32
    amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.amp)

    exp = None
    run_dir: Path | None = None
    if ddp.is_main():
        ts = time.strftime("%Y-%m-%d-%H%M%S-%Z")
        exp_name = f"in1k_{cfg.mode}_{cfg.model_repo.split('/')[-1]}_{cfg.n_timesteps}t_s{cfg.scene_size}_{ts}"
        exp = make_tracker(
            tracker=cfg.tracker, is_main=True, is_seeding=False, run_name=exp_name,
            wandb_project=cfg.wandb_project, wandb_entity=cfg.wandb_entity, wandb_dir=cfg.wandb_dir,
            prev_comet_id=None, prev_wandb_id=None,
        )
        exp.log_parameters({k: str(v) for k, v in asdict(cfg).items()})
        if cfg.clf_ckpt_dir:
            run_dir = cfg.clf_ckpt_dir / exp_name
            run_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Checkpoints: {run_dir}")

    freeze = cfg.mode == "frozen"
    step = 0
    best_top1 = 0.0
    pbar = tqdm(total=total_steps, desc=f"in1k-{cfg.mode}", disable=not ddp.is_main())
    for epoch in range(cfg.epochs):
        # Reset train mode each epoch: evaluate() flips the whole model to eval(),
        # so finetune must re-enable the backbone (else BN/dropout stay frozen);
        # frozen keeps the backbone in eval and trains only LN+head.
        if freeze:
            clf.canvit.eval()
            clf.head.train()
            clf.norm.train()
        else:
            clf.train()
        for batch_i, (images, labels) in enumerate(train_loader):
            if batch_i >= batches_per_epoch:  # honor limit_train_batches (loader may yield more)
                break
            images = images.to(device, non_blocking=True)
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)

            with amp_ctx:
                cls_tokens = rollout_cls_tokens(
                    clf=clf, images=images,
                    viewpoints=make_random_viewpoints(
                        images.shape[0], device, cfg.n_timesteps,
                        min_scale=cfg.min_vp_scale, max_scale=cfg.max_vp_scale,
                        start_with_full_scene=cfg.train_start_full,
                        is_foveated=is_foveated, foveated_scale=cfg.foveated_scale,
                    ),
                    canvas_grid=canvas_grid, glimpse_px=cfg.glimpse_px, freeze_backbone=freeze,
                )
                logits = [clf.head(clf.norm(c)) for c in cls_tokens]
                loss = torch.stack([ce_loss(lg, labels, label_smoothing=cfg.label_smoothing) for lg in logits]).mean()

            opt.zero_grad()
            loss.backward()
            if ddp.is_dist():
                _allreduce_grads(params)
            grad_norm = nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            scheduler.step()

            step += 1
            pbar.update(1)
            if ddp.is_main() and step % cfg.log_every == 0 and exp is not None:
                exp.log_metrics(
                    {"loss": loss.item(), "grad_norm": float(grad_norm), "lr": scheduler.get_last_lr()[0],
                     "epoch": epoch}, step=step,
                )

        if val_loader is not None and (epoch + 1) % cfg.eval_every_epochs == 0:
            accs = evaluate(clf, cfg, val_loader, device=device, canvas_grid=canvas_grid,
                            amp_ctx=amp_ctx, is_foveated=is_foveated)
            if ddp.is_main():
                log.info(f"Epoch {epoch}: top1={accs[1]:.4f} top5={accs[5]:.4f}")
                if exp is not None:
                    exp.log_metrics({f"val_top{k}": v for k, v in accs.items()}, step=step)
                if accs[1] > best_top1 and run_dir is not None:
                    best_top1 = accs[1]
                    clf.save_pretrained(run_dir / "best-hf")
                    torch.save({"epoch": epoch, "top1": accs[1], "head": clf.head.state_dict(),
                                "norm": clf.norm.state_dict()}, run_dir / "best.pt")

    pbar.close()
    if ddp.is_main():
        log.info(f"Done. best val top-1 = {best_top1:.4f}")
        if exp is not None:
            exp.end()
    ddp.barrier()


if __name__ == "__main__":
    import tyro

    train(tyro.cli(In1kConfig))
