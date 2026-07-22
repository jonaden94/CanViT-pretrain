"""ADE20K frozen-backbone probe training on the stable wrapper.

Port of canvit_specialize/training/ade20k/train_canvit.py with the unification
deltas (master plan P2): loads via
CanViTForSemanticSegmentation.from_pretrained_with_new_probe (D3 — no raw
pretraining model, no teacher), patcher-aware rollout (foveated/square now
supported), canvas_hidden only, pretrain's Tracker, per-t scalar logging (no
figure/image uploads). The optimization recipe (AdamW + WarmupOneCycleLR,
per-timestep CE mean, head-only clip) is unchanged from specialize.
"""

import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from canvit_pytorch import CanViTForSemanticSegmentation
from torch import Tensor
from tqdm import tqdm

from canvit_pretrain.train.tracker import make_tracker

from .config import Ade20kConfig
from .data import IGNORE_LABEL, NUM_CLASSES, make_ade20k_loaders, make_amp_ctx, make_optimizer_and_scheduler
from .metrics import ProbeState, ce_loss, eval_probe_on_batch, mIoUAccumulator, upsample_preds
from .rollout import consumes_full_image, make_random_viewpoints, rollout_canvas_hidden

log = logging.getLogger(__name__)


def _save_probe_checkpoint(
    run_dir: Path, probe: ProbeState, step: int, cfg: Ade20kConfig, *, is_best: bool
) -> Path:
    """Save the probe head (frozen-backbone mode: never the CanViT weights).
    'best' saves keep at most one file."""
    t_last = cfg.n_timesteps - 1
    miou = probe.best_last_miou
    prefix = (
        f"canvas_hidden_best_t{t_last}_miou{miou:.4f}_step{step}" if is_best
        else f"canvas_hidden_final_step{step}"
    )
    filename = f"{prefix}.pt"
    path = run_dir / filename
    tmp_path = run_dir / f".{filename}.tmp"
    data = {
        "step": step,
        "feat_type": "canvas_hidden",
        "probe_state_dict": probe.head.state_dict(),
        "best_mious_per_t": probe.best_mious,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    if is_best:
        for old in run_dir.glob("canvas_hidden_best_*.pt"):
            old.unlink()
    torch.save(data, tmp_path)
    tmp_path.rename(path)
    log.info(f"Saved checkpoint: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def train(cfg: Ade20kConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.device)

    log.info("=" * 60)
    log.info("ADE20K Canvas Probe Training (unified repo)")
    log.info("=" * 60)
    log.info(f"Model: {cfg.model_repo}")
    log.info(f"Timesteps: {cfg.n_timesteps}")
    log.info(f"Viewpoint: scale=[{cfg.min_vp_scale}, {cfg.max_vp_scale}], train_start_full={cfg.train_start_full}")
    log.info(f"Training: BS={cfg.batch_size}, steps={cfg.max_steps}, LR={cfg.peak_lr}, WD={cfg.weight_decay}")

    # Model: pretrained backbone + fresh probe on the stable wrapper (D3)
    log.info("Loading model...")
    seg = CanViTForSemanticSegmentation.from_pretrained_with_new_probe(
        pretrained_repo=cfg.model_repo, num_classes=NUM_CLASSES, dropout=cfg.dropout, use_ln=True
    ).to(device)
    seg.canvit.requires_grad_(False)
    seg.canvit.eval()
    # Same predicate names two things: how the glimpse is fed (full image vs
    # pre-crop) and which random-viewpoint law applies (foveated vs safe-box).
    full_image = is_foveated = consumes_full_image(seg)
    log.info(
        f"  backbone FROZEN ({sum(p.numel() for p in seg.canvit.parameters()) / 1e6:.1f}M params), "
        f"patcher={'foveated/square (full-image)' if full_image else 'uniform (pre-crop)'}"
    )
    if is_foveated:
        fs = cfg.foveated_scale
        detail = f"fixed_scale={fs.fixed_scale}" if fs.mode == "fixed" else (
            f"{fs.distribution} in [{fs.min_scale}, {fs.max_scale}]"
        )
        log.warning(
            f"  foveated view scale: mode={fs.mode}, {detail} — this MUST match the "
            f"backbone's pretraining scale or every glimpse is out of distribution "
            f"(symptom: mIoU falls as glimpses accumulate)."
        )

    patch_size = seg.canvit.backbone.patch_size_px
    canvas_grid = cfg.canvas_grid if cfg.canvas_grid is not None else cfg.scene_size // patch_size
    log.info(f"  scene: {cfg.scene_size}px, canvas: {canvas_grid}x{canvas_grid}, glimpse_px: {cfg.glimpse_px}")

    opt, scheduler = make_optimizer_and_scheduler(
        list(seg.head.parameters()), lr=cfg.peak_lr, weight_decay=cfg.weight_decay,
        max_steps=cfg.max_steps, warmup_steps=cfg.warmup_steps, warmup_lr_ratio=cfg.warmup_lr_ratio,
    )
    probe = ProbeState("canvas_hidden", seg.head, opt, scheduler)
    probe.init_best_mious(cfg.n_timesteps)
    log.info(f"  probe: dim={seg.canvas_dim}, head_params={sum(p.numel() for p in seg.head.parameters()):,}")

    val_iou = [mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in range(cfg.n_timesteps)]
    train_iou = [mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in range(cfg.n_timesteps)]

    train_loader, val_loader = make_ade20k_loaders(cfg)

    model_slug = cfg.model_repo.split("/")[-1]
    ts = time.strftime("%Y-%m-%d-%H%M%S-%Z")
    exp_name = f"ade20k_{model_slug}_{cfg.n_timesteps}t_s{cfg.scene_size}_c{canvas_grid}_{ts}"
    exp = make_tracker(
        tracker=cfg.tracker, is_main=True, is_seeding=False, run_name=exp_name,
        wandb_project=cfg.wandb_project, wandb_entity=cfg.wandb_entity, wandb_dir=cfg.wandb_dir,
        prev_comet_id=None, prev_wandb_id=None,
    )
    exp.log_parameters({k: str(v) for k, v in asdict(cfg).items()})
    log.info(f"Tracker: {cfg.tracker} (run={exp_name})")

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.probe_ckpt_dir / f"{model_slug}_{timestamp}_{job_id}" if cfg.probe_ckpt_dir else None
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Checkpoints: {run_dir}")

    amp_ctx = make_amp_ctx(cfg.amp, device)

    log.info("Starting training...")
    step = 0
    train_iter = iter(train_loader)
    pbar = tqdm(total=cfg.max_steps, desc="ade20k-probe")

    while step < cfg.max_steps:
        try:
            images, masks = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, masks = next(train_iter)
        images, masks = images.to(device), masks.to(device)

        # ── validation ────────────────────────────────────────────────
        if step % cfg.val_every == 0:
            val_start = time.perf_counter()
            seg.head.eval()
            for m in val_iou:
                m.reset()
            with torch.no_grad():
                for vi, vm in val_loader:
                    vi, vm = vi.to(device), vm.to(device)
                    vps = make_random_viewpoints(
                        vi.shape[0], device, cfg.n_timesteps,
                        min_scale=cfg.min_vp_scale, max_scale=cfg.max_vp_scale,
                        start_with_full_scene=True,
                        is_foveated=is_foveated, foveated_scale=cfg.foveated_scale,
                    )
                    with amp_ctx:
                        hidden = rollout_canvas_hidden(
                            seg=seg, images=vi, viewpoints=vps,
                            canvas_grid=canvas_grid, glimpse_px=cfg.glimpse_px,
                        )
                    for t in range(cfg.n_timesteps):
                        eval_probe_on_batch(seg.head, hidden[t], vm, val_iou[t])

            mious = [val_iou[t].compute() for t in range(cfg.n_timesteps)]
            improved = probe.update_best(mious)
            for t, miou in enumerate(mious):
                exp.log_metric(f"val_miou_t{t}", miou, step=step)
                exp.log_metric(f"best_val_miou_t{t}", probe.best_mious[t], step=step)
            if improved and run_dir:
                _save_probe_checkpoint(run_dir, probe, step, cfg, is_best=True)
            val_time = time.perf_counter() - val_start
            log.info(f"Step {step}: val mIoU t{cfg.n_timesteps - 1}={mious[-1]:.4f} ({val_time:.1f}s)")
            exp.log_metric("timing/val_seconds", val_time, step=step)

        # ── train step ────────────────────────────────────────────────
        seg.head.train()
        B = images.shape[0]
        vps = make_random_viewpoints(
            B, device, cfg.n_timesteps,
            min_scale=cfg.min_vp_scale, max_scale=cfg.max_vp_scale,
            start_with_full_scene=cfg.train_start_full,
            is_foveated=is_foveated, foveated_scale=cfg.foveated_scale,
        )
        with amp_ctx:
            hidden = rollout_canvas_hidden(
                seg=seg, images=images, viewpoints=vps,
                canvas_grid=canvas_grid, glimpse_px=cfg.glimpse_px,
            )

        probe.optimizer.zero_grad()
        logits_list = [seg.head(hidden[t].float()) for t in range(cfg.n_timesteps)]
        losses = [ce_loss(logits, masks) for logits in logits_list]
        loss = torch.stack(losses).mean()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(seg.head.parameters(), cfg.grad_clip)
        probe.optimizer.step()
        probe.scheduler.step()
        probe.accumulate(loss, grad_norm)

        with torch.no_grad():
            for t, logits in enumerate(logits_list):
                preds_up = upsample_preds(logits.detach().argmax(1), masks.shape[1], masks.shape[2])
                train_iou[t].update(preds_up, masks)

        step += 1
        pbar.update(1)

        if step % cfg.log_every == 0:
            avg_loss, avg_grad = probe.get_and_reset()
            mious = [m.compute() for m in train_iou]
            exp.log_metrics(
                {
                    "lr": float(probe.scheduler.get_last_lr()[0]),
                    "loss": avg_loss,
                    "grad_norm": avg_grad,
                    "train_miou_mean": sum(mious) / len(mious),
                },
                step=step,
            )
            for m in train_iou:
                m.reset()

    pbar.close()
    if run_dir:
        _save_probe_checkpoint(run_dir, probe, step, cfg, is_best=False)

    log.info("Training complete. Best val mIoU per timestep:")
    for t, v in enumerate(probe.best_mious):
        exp.log_metric(f"best/canvas_hidden_t{t}", v)
    log.info(f"  t0={probe.best_mious[0]:.4f} ... t{cfg.n_timesteps - 1}={probe.best_last_miou:.4f}")
    exp.end()
