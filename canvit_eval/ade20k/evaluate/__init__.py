"""ADE20K probe evaluation with configurable policies and transforms.

Runs trained probes on ADE20K validation set with:
- Different viewpoint policies (coarse_to_fine, random, full_then_random)
- Different transforms (center_crop, squish)
- Efficient batch processing (no GPU syncs in hot path)
"""

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from canvit import CanViTForPretrainingHFHub
from canvit_utils.teacher import load_teacher
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision import transforms as T
from tqdm import tqdm

from canvit_eval.ade20k.dataset import IGNORE_LABEL, NUM_CLASSES
from canvit_eval.ade20k.probe import ProbeHead
from canvit_eval.ade20k.train_probe.config import STATIC_FEATURES, FeatureType
from canvit_eval.ade20k.train_probe.features import extract_features
from canvit_eval.ade20k.train_probe.loss import upsample_preds
from canvit_eval.utils import PolicyName, collect_metadata, make_viewpoints

log = logging.getLogger(__name__)

TransformMode = Literal["center_crop", "squish"]


# === Config ===


def _default_output() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    return Path(f"outputs/ade20k_eval_{job_id}.pt")


@dataclass
class EvalConfig:
    """ADE20K probe evaluation configuration."""

    probe_ckpt: Path
    ade20k_root: Path
    output: Path = field(default_factory=_default_output)

    model_repo: str = "canvit/canvit-vitb16-pretrain-512px-in21k"
    policy: PolicyName = "coarse_to_fine"
    transform: TransformMode = "center_crop"
    n_timesteps: int = 10
    image_size: int = 512
    glimpse_px: int = 128

    # Random policy params
    min_scale: float = 0.05
    max_scale: float = 1.0

    batch_size: int = 32
    num_workers: int = 8
    device: str = "cuda"
    amp: bool = True


# === Dataset with transform mode ===


class ADE20kValDataset(Dataset[tuple[Tensor, Tensor]]):
    """ADE20K validation dataset with configurable transform."""

    def __init__(self, root: Path, size: int, transform_mode: TransformMode) -> None:
        self.size = size
        self._transform_mode = transform_mode

        if transform_mode == "center_crop":
            self._img_transform = T.Compose([
                T.Resize(size),
                T.CenterCrop(size),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ])
            self._mask_transform = T.Compose([
                T.Resize(size, T.InterpolationMode.NEAREST),
                T.CenterCrop(size),
            ])
        else:  # squish
            self._img_transform = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ])
            self._mask_transform = T.Resize((size, size), T.InterpolationMode.NEAREST)

        img_dir = root / "images" / "validation"
        ann_dir = root / "annotations" / "validation"
        self.imgs = sorted(img_dir.glob("*.jpg"))
        self.anns = [ann_dir / (p.stem + ".png") for p in self.imgs]

        assert len(self.imgs) > 0, f"No images in {img_dir}"
        log.info(f"ADE20k val: {len(self.imgs)} images, size={size}, transform={transform_mode}")

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        img = Image.open(self.imgs[i]).convert("RGB")
        mask = Image.open(self.anns[i])

        img_t = self._img_transform(img)
        mask_t = torch.from_numpy(np.array(self._mask_transform(mask))).long()
        # reduce_zero_label: 0→255, 1-150→0-149
        mask_t = torch.where((mask_t >= 1) & (mask_t <= 150), mask_t - 1, IGNORE_LABEL)

        return img_t, mask_t


# === Checkpoint loading ===


def load_probes(
    ckpt_path: Path,
    device: torch.device,
    canvas_dim: int,
    teacher_dim: int,
) -> dict[FeatureType, ProbeHead]:
    """Load trained probes from checkpoint."""
    log.info(f"Loading probes from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    dims: dict[FeatureType, int] = {
        "hidden": canvas_dim,
        "predicted_norm": teacher_dim,
        "teacher_glimpse": teacher_dim,
        "teacher_full": teacher_dim,
    }
    needs_ln: dict[FeatureType, bool] = {
        "hidden": True, "predicted_norm": False, "teacher_glimpse": False, "teacher_full": False,
    }

    probes: dict[FeatureType, ProbeHead] = {}
    for name, state_dict in ckpt["probe_state_dicts"].items():
        probe = ProbeHead(dims[name], use_ln=needs_ln[name]).to(device)
        probe.load_state_dict(state_dict)
        probe.eval()
        probes[name] = probe
        log.info(f"  {name}: loaded (best mIoU={ckpt['best_mean_mious'].get(name, 'N/A')})")

    return probes


# === Main evaluation ===


@torch.inference_mode()
def evaluate(cfg: EvalConfig) -> Path:
    """Run evaluation, save results, return output path."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_float32_matmul_precision("high")

    device = torch.device(cfg.device)
    T = cfg.n_timesteps

    log.info("=" * 70)
    log.info("ADE20K Probe Evaluation")
    log.info("=" * 70)
    for k, v in asdict(cfg).items():
        log.info(f"  {k}: {v}")

    # Load model
    log.info(f"Loading model: {cfg.model_repo}")
    model = CanViTForPretrainingHFHub.from_pretrained(cfg.model_repo).to(device).eval()
    teacher = load_teacher(model.backbone_name, device)
    log.info(f"  canvas_dim={model.canvas_dim}, teacher_dim={teacher.embed_dim}")

    patch_size = model.backbone.patch_size_px
    canvas_grid = cfg.image_size // patch_size

    # Load probes
    probes = load_probes(cfg.probe_ckpt, device, model.canvas_dim, teacher.embed_dim)
    feature_types = list(probes.keys())
    need_canvit = any(f not in STATIC_FEATURES for f in feature_types)
    compute_teacher_full = "teacher_full" in feature_types

    log.info(f"Features: {feature_types}, need_canvit={need_canvit}")

    # Dataset
    dataset = ADE20kValDataset(cfg.ade20k_root, cfg.image_size, cfg.transform)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # IoU metrics per feature per timestep (on GPU, no sync until end)
    iou_metrics: dict[FeatureType, list[MulticlassJaccardIndex]] = {
        feat: [MulticlassJaccardIndex(NUM_CLASSES, ignore_index=IGNORE_LABEL, average="macro").to(device)
               for _ in range(T)]
        for feat in feature_types
    }

    amp_dtype = torch.bfloat16 if cfg.amp else torch.float32
    amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.amp)

    # Run evaluation
    log.info(f"Evaluating with policy={cfg.policy}, {T} timesteps...")
    pbar = tqdm(loader, desc="Evaluating", unit="batch")

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        B = images.shape[0]

        # Generate viewpoints for this batch
        viewpoints = make_viewpoints(
            cfg.policy, B, device, T,
            min_scale=cfg.min_scale, max_scale=cfg.max_scale,
            start_with_full_scene=True,
        ) if need_canvit else None

        with amp_ctx:
            feats = extract_features(
                model=model, teacher=teacher, images=images,
                canvas_grid=canvas_grid, glimpse_px=cfg.glimpse_px,
                viewpoints=viewpoints, compute_teacher_full=compute_teacher_full,
            )

        # Update metrics (no sync here - torchmetrics accumulates on GPU)
        for feat_type in feature_types:
            t_range = [0] if feat_type in STATIC_FEATURES else range(T)
            for t in t_range:
                feat_t = feats.get(feat_type, t)
                assert feat_t is not None
                logits = probes[feat_type](feat_t.float())
                preds = logits.argmax(1)
                preds_up = upsample_preds(preds, masks.shape[1], masks.shape[2])
                iou_metrics[feat_type][t].update(preds_up, masks)

    # Compute final mIoUs (sync happens here)
    log.info("")
    log.info("=" * 70)
    log.info("RESULTS")
    log.info("=" * 70)

    results: dict[str, dict] = {"mious": {}, "metadata": collect_metadata(cfg)}

    for feat_type in feature_types:
        if feat_type in STATIC_FEATURES:
            miou = iou_metrics[feat_type][0].compute().item()
            results["mious"][feat_type] = {"t0": miou, "mean": miou}
            log.info(f"  {feat_type}: mIoU={100*miou:.2f}%")
        else:
            mious = [iou_metrics[feat_type][t].compute().item() for t in range(T)]
            mean_miou = sum(mious) / len(mious)
            results["mious"][feat_type] = {
                **{f"t{t}": m for t, m in enumerate(mious)},
                "mean": mean_miou,
            }
            log.info(f"  {feat_type}:")
            for t, m in enumerate(mious):
                log.info(f"    t{t}: {100*m:.2f}%")
            log.info(f"    mean: {100*mean_miou:.2f}%")

    # Save
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, cfg.output)
    log.info(f"Saved to {cfg.output}")

    return cfg.output


def main() -> None:
    import tyro
    cfg = tyro.cli(EvalConfig)
    evaluate(cfg)
