"""Evaluate reconstruction quality on held-out images.

Computes per-timestep cosine similarity between CanViT canvas reconstruction
and DINOv3 teacher features. No probes, no labels — just forward passes.

Usage:
    uv run python -m canvit_eval.reconstruction \
        --ckpt path/to/checkpoint.pt \
        --image-dir /datasets/ADE20k/ADEChallengeData2016/images/validation \
        --output results/abl_baseline.pt
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from canvit import CanViTOutput, RecurrentState, Viewpoint, sample_at_viewpoint
from canvit_utils.teacher import DINOv3Teacher, load_teacher
from canvit_utils.transforms import preprocess
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from canvit_pretrain import CanViTForPretraining
from canvit_pretrain.checkpoint import load_model

from canvit_eval.utils import collect_metadata, make_viewpoints

log = logging.getLogger(__name__)

TEACHER_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class FlatImageDataset(Dataset):
    """Load images from a flat directory (no subdirectories required)."""

    def __init__(self, root: Path, transform: object = None) -> None:
        self.paths = sorted(
            p for p in root.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        assert len(self.paths) > 0, f"No images found in {root}"
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


@dataclass
class ReconstructionEvalConfig:
    ckpt: Path
    image_dir: Path
    output: Path

    scene_size: int = 512
    canvas_grid: int = 32
    glimpse_px: int = 128
    n_timesteps: int = 10
    batch_size: int = 16
    num_workers: int = 4
    device: str = "cuda"
    teacher_cache: Path | None = None


@dataclass
class TimestepMetrics:
    """Accumulated cosine similarities for one timestep."""
    scene_cos_sum: float = 0.0
    cls_cos_sum: float = 0.0
    n_images: int = 0

    def update(self, scene_cos: float, cls_cos: float, batch_size: int) -> None:
        self.scene_cos_sum += scene_cos * batch_size
        self.cls_cos_sum += cls_cos * batch_size
        self.n_images += batch_size

    @property
    def scene_cos_mean(self) -> float:
        return self.scene_cos_sum / self.n_images

    @property
    def cls_cos_mean(self) -> float:
        return self.cls_cos_sum / self.n_images


def _cache_teacher_features(
    teacher: DINOv3Teacher,
    loader: DataLoader,
    scene_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Extract teacher features for entire dataset. Returns (patches, cls)."""
    all_patches: list[Tensor] = []
    all_cls: list[Tensor] = []
    log.info("Caching teacher features for %d batches...", len(loader))
    with torch.inference_mode():
        for images in tqdm(loader, desc="Teacher features"):
            images = images.to(device)
            feats = teacher.forward_norm_features(images)
            all_patches.append(feats.patches.cpu().half())
            all_cls.append(feats.cls.cpu().half())
    patches = torch.cat(all_patches)
    cls = torch.cat(all_cls)
    log.info("Cached: patches %s, cls %s (%.2f GB)",
             patches.shape, cls.shape,
             (patches.nelement() + cls.nelement()) * 2 / 1e9)
    return patches, cls


def evaluate(cfg: ReconstructionEvalConfig) -> dict:
    device = torch.device(cfg.device)

    # Load model from training checkpoint
    log.info("Loading model from %s", cfg.ckpt)
    model, ckpt_data = load_model(cfg.ckpt, device=device)
    model.eval()

    canvas_grid = cfg.canvas_grid
    has_cls = model.scene_cls_head is not None

    # Dataset — just images, no labels needed
    transform = preprocess(cfg.scene_size)
    dataset = FlatImageDataset(cfg.image_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    log.info("Dataset: %d images from %s", len(dataset), cfg.image_dir)

    # Teacher features — cache or compute
    teacher = load_teacher(TEACHER_REPO, device)
    if cfg.teacher_cache is not None and cfg.teacher_cache.exists():
        log.info("Loading cached teacher features from %s", cfg.teacher_cache)
        cached = torch.load(cfg.teacher_cache, map_location="cpu", weights_only=True)
        teacher_patches = cached["patches"]
        teacher_cls = cached["cls"]
    else:
        teacher_patches, teacher_cls = _cache_teacher_features(
            teacher, loader, cfg.scene_size, device,
        )
        if cfg.teacher_cache is not None:
            cfg.teacher_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"patches": teacher_patches, "cls": teacher_cls}, cfg.teacher_cache)
            log.info("Saved teacher cache to %s", cfg.teacher_cache)

    del teacher
    torch.cuda.empty_cache()

    # Evaluate reconstruction quality
    T = cfg.n_timesteps
    metrics = [TimestepMetrics() for _ in range(T)]

    start_time = time.perf_counter()
    img_idx = 0

    with torch.inference_mode():
        for images in tqdm(loader, desc="Reconstruction eval"):
            B = images.shape[0]
            images = images.to(device)

            # Slice pre-cached teacher features for this batch
            target_scene = teacher_patches[img_idx:img_idx + B].to(device).float()
            target_cls = teacher_cls[img_idx:img_idx + B].to(device).float()
            img_idx += B

            viewpoints = make_viewpoints(
                "coarse_to_fine", B, device, T,
            )

            # Step-by-step recurrent forward pass
            state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)
            for t, vp in enumerate(viewpoints):
                glimpse = sample_at_viewpoint(
                    spatial=images, viewpoint=vp, glimpse_size_px=cfg.glimpse_px,
                )
                out = model.forward(glimpse=glimpse, state=state, viewpoint=vp)
                state = out.state

                pred_scene = model.predict_teacher_scene(state.canvas)
                scene_cos = F.cosine_similarity(
                    pred_scene, target_scene, dim=-1,
                ).mean().item()

                cls_cos = 0.0
                if has_cls:
                    pred_cls = model.predict_scene_teacher_cls(state.recurrent_cls)
                    cls_cos = F.cosine_similarity(
                        pred_cls, target_cls, dim=-1,
                    ).mean().item()

                metrics[t].update(scene_cos, cls_cos, B)

    elapsed = time.perf_counter() - start_time
    log.info("Evaluation done: %d images in %.1fs (%.1f img/s)",
             img_idx, elapsed, img_idx / elapsed)

    # Build results
    per_timestep = [
        {
            "t": t,
            "scene_cos": round(m.scene_cos_mean, 6),
            "cls_cos": round(m.cls_cos_mean, 6),
        }
        for t, m in enumerate(metrics)
    ]

    result = {
        "per_timestep": per_timestep,
        "n_images": img_idx,
        "elapsed_s": round(elapsed, 1),
        "metadata": {
            "ckpt_path": str(cfg.ckpt),
            "ckpt_step": ckpt_data.get("step"),
            "ckpt_backbone": ckpt_data["backbone_name"],
            "scene_size": cfg.scene_size,
            "canvas_grid": cfg.canvas_grid,
            "glimpse_px": cfg.glimpse_px,
            "n_timesteps": T,
            "dataset": str(cfg.image_dir),
            **collect_metadata(cfg),
        },
    }

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, cfg.output)
    log.info("Saved to %s", cfg.output)

    # Print summary
    for p in per_timestep:
        log.info("  t=%d  scene_cos=%.4f  cls_cos=%.4f", p["t"], p["scene_cos"], p["cls_cos"])

    return result
