#!/usr/bin/env python3
"""Train ADE20K segmentation probes across multiple timesteps.

Recurrent probes (per-timestep): hidden, predicted_norm
Static probes (single, baseline): teacher_full, teacher_glimpse

Logs mIoU curves vs timestep to see feature quality evolution.
"""

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import albumentations as A
import comet_ml
import dacite
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from canvit.backbone.dinov3 import DINOv3Backbone
from canvit.hub import create_backbone
from canvit.viewpoint import Viewpoint
from PIL import Image
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm

from avp_vit import ActiveCanViT, ActiveCanViTConfig
from avp_vit.checkpoint import load as load_ckpt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NUM_CLASSES = 150
IGNORE_LABEL = 255
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

# Recurrent features change across timesteps, static features don't
RecurrentFeature = Literal["hidden", "predicted_norm"]
StaticFeature = Literal["teacher_full", "teacher_glimpse"]
FeatureType = Literal["hidden", "predicted_norm", "teacher_full", "teacher_glimpse"]

RECURRENT_FEATURES: set[FeatureType] = {"hidden", "predicted_norm"}
STATIC_FEATURES: set[FeatureType] = {"teacher_full", "teacher_glimpse"}


@dataclass
class Config:
    avp_ckpt: Path
    ade20k_root: Path = Path("/datasets/ADE20k/ADEChallengeData2016")
    teacher_ckpt: Path | None = None

    features: list[FeatureType] = field(default_factory=lambda: ["hidden", "predicted_norm", "teacher_full"])
    n_timesteps: int = 5  # t=0 full, t=1..4 random

    image_size: int = 512
    batch_size: int = 64
    eval_batch_size: int = 32
    num_workers: int = 4

    peak_lr: float = 1e-4
    min_lr: float = 1e-7
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    max_steps: int = 5000
    grad_clip: float = 1.0

    log_every: int = 20
    val_every: int = 500
    viz_every: int = 1000

    comet_project: str = "avp-ade20k-probe"
    comet_workspace: str = "m2b3-ava"
    device: str | None = None
    amp: bool = True


def focal_loss(logits: Tensor, masks: Tensor, scale: int, gamma: float = 2.0) -> Tensor:
    B, C, h, w = logits.shape
    log_probs = F.log_softmax(logits, dim=1).permute(0, 2, 3, 1).reshape(-1, C)
    probs = log_probs.exp()
    mask_patches = masks.reshape(B, h, scale, w, scale).permute(0, 1, 3, 2, 4).reshape(-1, scale * scale)
    valid = mask_patches != IGNORE_LABEL
    targets = mask_patches.clamp(0, C - 1).long()
    log_p = log_probs.gather(1, targets)
    p = probs.gather(1, targets)
    return -((1 - p) ** gamma * log_p * valid).sum() / valid.sum().clamp(min=1)


class ProbeHead(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.ln(x)).permute(0, 3, 1, 2)


class ADE20kDataset(Dataset):
    def __init__(self, root: Path, split: str, size: int, augment: bool = False) -> None:
        self.size = size
        self.load_size = size * 2 if augment else size
        self.transform = A.Compose([A.HorizontalFlip(p=0.5), A.RandomCrop(size, size)]) if augment else None
        img_dir, ann_dir = root / "images" / split, root / "annotations" / split
        self.imgs = sorted(img_dir.glob("*.jpg"))
        self.anns = [ann_dir / (p.stem + ".png") for p in self.imgs]
        log.info(f"ADE20k {split}: {len(self)} images")

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        img = np.array(Image.open(self.imgs[i]).convert("RGB").resize((self.load_size, self.load_size), Image.Resampling.BILINEAR))
        mask = np.array(Image.open(self.anns[i]).resize((self.load_size, self.load_size), Image.Resampling.NEAREST))
        if self.transform:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img_t = (img_t - IMAGENET_MEAN.view(3, 1, 1)) / IMAGENET_STD.view(3, 1, 1)
        mask_t = torch.from_numpy(mask.astype(np.int64))
        valid = (mask_t >= 1) & (mask_t <= 150)
        return img_t, torch.where(valid, mask_t - 1, IGNORE_LABEL)


@dataclass
class Probe:
    name: str
    head: ProbeHead
    optimizer: AdamW
    scheduler: SequentialLR
    loss_sum: float = 0.0
    loss_count: int = 0
    best_miou: float = 0.0

    def accumulate(self, loss: Tensor) -> None:
        self.loss_sum += loss.item()
        self.loss_count += 1

    def reset(self) -> float:
        avg = self.loss_sum / max(self.loss_count, 1)
        self.loss_sum, self.loss_count = 0.0, 0
        return avg


@dataclass
class Features:
    """Extracted features. Recurrent have list per timestep, static have single tensor."""
    hidden: list[Tensor]  # [t0, t1, ...]
    predicted_norm: list[Tensor]
    teacher_full: Tensor  # single
    teacher_glimpse: Tensor  # single


def extract_features(
    model: ActiveCanViT,
    teacher: DINOv3Backbone,
    images: Tensor,
    n_timesteps: int,
    canvas_grid: int,
    glimpse_grid: int,
    glimpse_px: int,
    device: torch.device,
) -> Features:
    """Extract features. Recurrent features vary by timestep, static don't."""
    B = images.shape[0]
    hidden_list: list[Tensor] = []
    predicted_list: list[Tensor] = []

    state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)

    for t in range(n_timesteps):
        if t == 0:
            vp = Viewpoint(torch.zeros(B, 2, device=device), torch.ones(B, device=device))
        else:
            vp = Viewpoint(torch.rand(B, 2, device=device) * 2 - 1, torch.rand(B, device=device) * 0.4 + 0.1)

        out = model.forward_step(image=images, state=state, viewpoint=vp, glimpse_size_px=glimpse_px)
        state = out.state

        hidden_list.append(model.get_spatial(state.canvas).view(B, canvas_grid, canvas_grid, -1))
        predicted_list.append(model.predict_teacher_scene(state.canvas).view(B, canvas_grid, canvas_grid, -1))

    # Static: teacher on full image and glimpse-sized image (baselines)
    teacher_full = teacher.forward_norm_features(images).patches.view(B, canvas_grid, canvas_grid, -1)
    sz = glimpse_grid * teacher.patch_size_px
    small = F.interpolate(images, (sz, sz), mode="bilinear", align_corners=False)
    teacher_glimpse = teacher.forward_norm_features(small).patches.view(B, glimpse_grid, glimpse_grid, -1)

    return Features(hidden_list, predicted_list, teacher_full, teacher_glimpse)


def log_viz(
    exp: comet_ml.Experiment,
    step: int,
    probes: dict[str, Probe],
    feats: Features,
    images: Tensor,
    masks: Tensor,
    cfg: Config,
    n_samples: int = 4,
) -> None:
    """Log visualization of predictions to Comet."""
    n = min(n_samples, images.shape[0])
    H_mask = masks.shape[1]

    palette = np.random.RandomState(42).randint(0, 255, (NUM_CLASSES + 1, 3), dtype=np.uint8)
    palette[NUM_CLASSES] = 0

    def colorize(m: np.ndarray) -> np.ndarray:
        return palette[np.where(m == IGNORE_LABEL, NUM_CLASSES, m)]

    def denorm(t: Tensor) -> np.ndarray:
        t = t * IMAGENET_STD.view(3, 1, 1).to(t.device) + IMAGENET_MEAN.view(3, 1, 1).to(t.device)
        return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # Collect predictions for t=0 and last timestep
    preds_t0: dict[str, np.ndarray] = {}
    preds_tlast: dict[str, np.ndarray] = {}

    for feat_type in cfg.features:
        if feat_type in RECURRENT_FEATURES:
            feat_list = feats.hidden if feat_type == "hidden" else feats.predicted_norm
            for t, feat in [(0, feat_list[0]), (cfg.n_timesteps - 1, feat_list[-1])]:
                name = f"{feat_type}/t{t}"
                if name in probes:
                    scale = H_mask // feat.shape[1]
                    logits = probes[name].head(feat.float())
                    pred = logits[:n].argmax(1).repeat_interleave(scale, 1).repeat_interleave(scale, 2).cpu().numpy()
                    if t == 0:
                        preds_t0[feat_type] = pred
                    else:
                        preds_tlast[feat_type] = pred
        else:
            feat = feats.teacher_full if feat_type == "teacher_full" else feats.teacher_glimpse
            if feat_type in probes:
                scale = H_mask // feat.shape[1]
                logits = probes[feat_type].head(feat.float())
                pred = logits[:n].argmax(1).repeat_interleave(scale, 1).repeat_interleave(scale, 2).cpu().numpy()
                preds_t0[feat_type] = pred

    # Plot: image, GT, then predictions
    cols = 2 + len(preds_t0) + len(preds_tlast)
    fig, axes = plt.subplots(n, cols, figsize=(2.5 * cols, 2.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        col = 0
        axes[i, col].imshow(denorm(images[i]))
        axes[i, col].set_title("Image")
        col += 1

        axes[i, col].imshow(colorize(masks[i].cpu().numpy()))
        axes[i, col].set_title("GT")
        col += 1

        for name, pred in preds_t0.items():
            axes[i, col].imshow(colorize(pred[i]))
            axes[i, col].set_title(f"{name[:6]}/t0")
            col += 1

        for name, pred in preds_tlast.items():
            axes[i, col].imshow(colorize(pred[i]))
            axes[i, col].set_title(f"{name[:6]}/t{cfg.n_timesteps-1}")
            col += 1

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    exp.log_figure(figure_name=f"predictions_{step}", figure=fig, step=step)
    plt.close(fig)


def main(cfg: Config) -> None:
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info(f"Device: {device}, AMP: {cfg.amp}")

    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if cfg.amp else torch.autocast(device_type=device.type, enabled=False)

    # Load AVP model
    ckpt = load_ckpt(cfg.avp_ckpt, device)
    model_cfg = dacite.from_dict(ActiveCanViTConfig, {**ckpt["model_config"], "teacher_dim": ckpt["teacher_dim"]})
    bb = create_backbone(ckpt["backbone"], pretrained=False)
    model = ActiveCanViT(backbone=bb, cfg=model_cfg, policy=None)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Teacher
    weights = str(cfg.teacher_ckpt) if cfg.teacher_ckpt else None
    teacher = create_backbone(ckpt["backbone"], pretrained=weights is None, weights=weights)
    assert isinstance(teacher, DINOv3Backbone)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Dimensions
    canvas_grid = cfg.image_size // model.backbone.patch_size_px
    glimpse_grid = ckpt["model_config"].get("glimpse_grid_size", 8)
    glimpse_px = glimpse_grid * model.backbone.patch_size_px
    dims: dict[FeatureType, int] = {
        "hidden": model.canvas_dim,
        "predicted_norm": ckpt["teacher_dim"],
        "teacher_full": teacher.embed_dim,
        "teacher_glimpse": teacher.embed_dim,
    }

    # Create probes
    probes: dict[str, Probe] = {}
    warmup_steps = int(cfg.warmup_ratio * cfg.max_steps)

    def make_probe(name: str, dim: int) -> Probe:
        head = ProbeHead(dim).to(device)
        opt = AdamW(head.parameters(), lr=cfg.peak_lr, weight_decay=cfg.weight_decay)
        warmup = LinearLR(opt, cfg.min_lr / cfg.peak_lr, 1.0, max(1, warmup_steps))
        cosine = CosineAnnealingLR(opt, cfg.max_steps - warmup_steps, eta_min=cfg.min_lr)
        return Probe(name, head, opt, SequentialLR(opt, [warmup, cosine], [warmup_steps]))

    for feat in cfg.features:
        if feat in RECURRENT_FEATURES:
            # One probe per timestep
            for t in range(cfg.n_timesteps):
                name = f"{feat}/t{t}"
                probes[name] = make_probe(name, dims[feat])
        else:
            # Single probe (static baseline)
            probes[feat] = make_probe(feat, dims[feat])

    log.info(f"Probes: {list(probes.keys())}")

    # Data
    train_ds = ADE20kDataset(cfg.ade20k_root, "training", cfg.image_size, augment=True)
    val_ds = ADE20kDataset(cfg.ade20k_root, "validation", cfg.image_size)
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, cfg.eval_batch_size, num_workers=cfg.num_workers, pin_memory=True)

    # Comet
    exp = comet_ml.Experiment(project_name=cfg.comet_project, workspace=cfg.comet_workspace)
    exp.log_parameters(asdict(cfg))

    # Training
    step, train_iter = 0, iter(train_loader)
    pbar = tqdm(total=cfg.max_steps, desc="Training")

    while step < cfg.max_steps:
        try:
            images, masks = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, masks = next(train_iter)
        images, masks = images.to(device), masks.to(device)
        H_mask = masks.shape[1]

        # Validation
        if step % cfg.val_every == 0:
            for p in probes.values():
                p.head.eval()
            ious = {n: MulticlassJaccardIndex(NUM_CLASSES, ignore_index=IGNORE_LABEL, average="macro").to(device) for n in probes}

            with torch.no_grad():
                for vi, vm in val_loader:
                    vi, vm = vi.to(device), vm.to(device)
                    with amp_ctx:
                        feats = extract_features(model, teacher, vi, cfg.n_timesteps, canvas_grid, glimpse_grid, glimpse_px, device)

                    # Recurrent features: evaluate each timestep
                    for feat_type in cfg.features:
                        if feat_type in RECURRENT_FEATURES:
                            feat_list = feats.hidden if feat_type == "hidden" else feats.predicted_norm
                            for t, feat in enumerate(feat_list):
                                name = f"{feat_type}/t{t}"
                                scale = vm.shape[1] // feat.shape[1]
                                preds = probes[name].head(feat.float()).argmax(1).repeat_interleave(scale, 1).repeat_interleave(scale, 2)
                                ious[name].update(preds, vm)
                        elif feat_type in probes:
                            feat = feats.teacher_full if feat_type == "teacher_full" else feats.teacher_glimpse
                            scale = vm.shape[1] // feat.shape[1]
                            preds = probes[feat_type].head(feat.float()).argmax(1).repeat_interleave(scale, 1).repeat_interleave(scale, 2)
                            ious[feat_type].update(preds, vm)

            # Log metrics
            for name, iou in ious.items():
                miou = iou.compute().item()
                exp.log_metric(f"{name}/val_miou", miou, step=step)
                if miou > probes[name].best_miou:
                    probes[name].best_miou = miou

            # Log curves for recurrent features
            for feat_type in cfg.features:
                if feat_type in RECURRENT_FEATURES:
                    curve_y = [ious[f"{feat_type}/t{t}"].compute().item() for t in range(cfg.n_timesteps)]
                    exp.log_curve(f"{feat_type}/miou_vs_t", x=list(range(cfg.n_timesteps)), y=curve_y, step=step)

            # Postfix: show t0 mIoU for recurrent, single mIoU for static
            postfix = {}
            for feat in cfg.features:
                if feat in RECURRENT_FEATURES:
                    postfix[feat[:3]] = f"{ious[f'{feat}/t0'].compute().item():.3f}"
                elif feat in probes:
                    postfix[feat[:3]] = f"{ious[feat].compute().item():.3f}"
            pbar.set_postfix(postfix)

        # Visualization
        if step % cfg.viz_every == 0:
            with torch.no_grad(), amp_ctx:
                viz_feats = extract_features(model, teacher, images, cfg.n_timesteps, canvas_grid, glimpse_grid, glimpse_px, device)
            log_viz(exp, step, probes, viz_feats, images, masks, cfg)

        # Train
        with amp_ctx:
            feats = extract_features(model, teacher, images, cfg.n_timesteps, canvas_grid, glimpse_grid, glimpse_px, device)

        for feat_type in cfg.features:
            if feat_type in RECURRENT_FEATURES:
                feat_list = feats.hidden if feat_type == "hidden" else feats.predicted_norm
                for t, feat in enumerate(feat_list):
                    name = f"{feat_type}/t{t}"
                    p = probes[name]
                    p.head.train()
                    p.optimizer.zero_grad()
                    scale = H_mask // feat.shape[1]
                    loss = focal_loss(p.head(feat.detach().float()), masks, scale)
                    loss.backward()
                    nn.utils.clip_grad_norm_(p.head.parameters(), cfg.grad_clip)
                    p.optimizer.step()
                    p.scheduler.step()
                    p.accumulate(loss)
            elif feat_type in probes:
                feat = feats.teacher_full if feat_type == "teacher_full" else feats.teacher_glimpse
                p = probes[feat_type]
                p.head.train()
                p.optimizer.zero_grad()
                scale = H_mask // feat.shape[1]
                loss = focal_loss(p.head(feat.detach().float()), masks, scale)
                loss.backward()
                nn.utils.clip_grad_norm_(p.head.parameters(), cfg.grad_clip)
                p.optimizer.step()
                p.scheduler.step()
                p.accumulate(loss)

        step += 1
        pbar.update(1)

        if step % cfg.log_every == 0:
            log_dict = {"lr": list(probes.values())[0].scheduler.get_last_lr()[0]}
            for name, p in probes.items():
                log_dict[f"{name}/loss"] = p.reset()
            exp.log_metrics(log_dict, step=step)

    pbar.close()
    log.info("Best mIoU:")
    for name, p in probes.items():
        log.info(f"  {name}: {p.best_miou:.4f}")
        exp.log_metric(f"best/{name}", p.best_miou)


if __name__ == "__main__":
    main(tyro.cli(Config))
