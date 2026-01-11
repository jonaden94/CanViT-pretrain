"""Export teacher features for IN21k.

Precomputes DINOv3 features for all images, stores as sharded .pt files.

Shard schema (v1):
    # Data
    patches: [N, n_patches, embed_dim] STORAGE_DTYPE  - patch features (L2-normalized)
    cls: [N, embed_dim] STORAGE_DTYPE                 - CLS token (L2-normalized)
    paths: list[str]                                  - relative paths within image_root
    class_idxs: [N] int32                             - class indices from parquet
    failed_indices: list[int]                         - indices with load errors (NaN features)

    # Position (row indices into parquet table)
    shard_id: int                                     - which shard (0-indexed)
    start_idx: int                                    - first parquet row (inclusive)
    end_idx: int                                      - last parquet row (exclusive)

    # Compatibility (must match across all shards)
    parquet_path: str                                 - path to index file
    parquet_sha256: str                               - 16-char hash of parquet
    teacher_model: str                                - e.g. "dinov3_vitb16"
    teacher_ckpt: str                                 - path to weights file
    image_size: int                                   - input resolution (e.g. 512)
    shard_size: int                                   - max images per shard (e.g. 4096)
    dtype: str                                        - e.g. "torch.bfloat16"
    embed_dim: int                                    - feature dimension (e.g. 768)
    n_patches: int                                    - patches per image (e.g. 1024)

    # Provenance
    created_at: str                                   - ISO 8601 UTC timestamp
    schema_version: int                               - 1
"""

import gc
import hashlib
import logging
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path

import pyarrow.parquet as pq
import torch
import tyro
from canvit.hub import create_backbone
from PIL import Image, ImageFile
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

STORAGE_DTYPE = torch.bfloat16
STORAGE_BYTES = torch.tensor([], dtype=STORAGE_DTYPE).element_size()
SCHEMA_VERSION = 1

ImageFile.LOAD_TRUNCATED_IMAGES = False
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


@dataclass
class Config:
    # Shard selection (mutually exclusive)
    shard: int | None = None
    start_shard: int | None = None
    end_shard: int | None = None

    # Paths (defaults from env vars)
    parquet: Path | None = None
    image_root: Path | None = None
    out_dir: Path | None = None
    teacher_ckpt: Path | None = None

    # Export settings
    teacher_model: str = "dinov3_vitb16"
    shard_size: int = 4096
    batch_size: int = 64
    num_workers: int = 8
    image_size: int = 512


# -----------------------------------------------------------------------------
# Preflight
# -----------------------------------------------------------------------------


def preflight_checks(
    parquet_path: Path,
    image_root: Path,
    teacher_ckpt: Path,
    cfg: Config,
    n_images: int,
    n_shards: int,
    start: int,
    end: int,
) -> None:
    """Fail fast on config/path issues. No GPU, no heavy I/O."""
    assert parquet_path.exists(), f"Parquet not found: {parquet_path}"
    assert image_root.is_dir(), f"Image root not a directory: {image_root}"
    assert teacher_ckpt.exists(), f"Teacher ckpt not found: {teacher_ckpt}"

    schema = pq.read_schema(parquet_path)
    required = {"path", "class_idx"}
    missing = required - set(schema.names)
    assert not missing, f"Parquet missing columns: {missing}"

    assert 0 <= start < end <= n_shards, (
        f"Invalid range [{start}, {end}) for {n_shards} shards"
    )
    assert cfg.image_size > 0
    assert cfg.shard_size > 0
    assert cfg.batch_size > 0
    assert cfg.num_workers >= 0


def estimate_bytes(n_images: int, n_patches: int, embed_dim: int) -> int:
    """Estimate shard size in bytes. patches + cls, both STORAGE_DTYPE."""
    return n_images * (n_patches + 1) * embed_dim * STORAGE_BYTES


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def log_gpu(label: str) -> None:
    log.info(f"GPU [{label}]: {torch.cuda.memory_allocated() / 1e9:.2f}GB")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def resolve_paths(cfg: Config) -> tuple[Path, Path, Path, Path]:
    image_root = cfg.image_root or Path(os.environ["AVP_TRAIN_DIR"])
    parquet = (
        cfg.parquet or Path(os.environ["AVP_INDEX_DIR"]) / f"{image_root.name}.parquet"
    )
    out_dir = cfg.out_dir or Path(os.environ["AVP_FEATURES_DIR"])
    teacher_ckpt = cfg.teacher_ckpt or Path(
        os.path.expanduser(os.environ["AVP_TEACHER_CKPT"])
    )
    return parquet, image_root, out_dir, teacher_ckpt


def get_shard_range(cfg: Config, n_shards: int) -> tuple[int, int]:
    if cfg.shard is not None:
        return cfg.shard, cfg.shard + 1
    if cfg.start_shard is not None and cfg.end_shard is not None:
        return cfg.start_shard, min(cfg.end_shard, n_shards)
    raise ValueError("Specify --shard or --start-shard/--end-shard")


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class ImageDataset(Dataset):
    def __init__(self, root: Path, paths: list[str], size: int):
        self.root = root
        self.paths = paths
        self.size = size
        self.transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, bool]:
        path = self.root / self.paths[idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                img = Image.open(path).convert("RGB")
                img.load()
            return self.transform(img), idx, True
        except Exception as e:
            log.warning(f"Bad image {path}: {e}")
            return torch.full((3, self.size, self.size), float("nan")), idx, False


# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------


def export_shard(
    *,
    shard_id: int,
    paths: list[str],
    class_idxs: list[int],
    start_idx: int,
    image_root: Path,
    shards_dir: Path,
    teacher,
    device: torch.device,
    cfg: Config,
    n_patches: int,
    embed_dim: int,
    parquet_path: Path,
    parquet_hash: str,
    teacher_ckpt: Path,
    pbar_global: tqdm,
) -> tuple[int, int, int]:
    """Export one shard. Returns (n_images, n_failed, shard_bytes)."""
    n = len(paths)
    shard_path = shards_dir / f"{shard_id:05d}.pt"

    # Preallocate GPU buffers
    patches_buf = torch.empty(
        n, n_patches, embed_dim, dtype=STORAGE_DTYPE, device=device
    )
    cls_buf = torch.empty(n, embed_dim, dtype=STORAGE_DTYPE, device=device)
    failed: list[int] = []

    loader = DataLoader(
        ImageDataset(image_root, paths, cfg.image_size),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    t0 = time.perf_counter()
    write_idx = 0

    with torch.no_grad(), torch.autocast("cuda", dtype=STORAGE_DTYPE):
        for imgs, indices, ok in tqdm(loader, desc=f"Shard {shard_id}", leave=False):
            for i, success in zip(indices.tolist(), ok.tolist()):
                if not success:
                    failed.append(i)

            imgs = imgs.to(device)
            feats = teacher.forward_norm_features(imgs)

            bs = imgs.shape[0]
            patches_buf[write_idx : write_idx + bs] = feats.patches.to(STORAGE_DTYPE)
            cls_buf[write_idx : write_idx + bs] = feats.cls.to(STORAGE_DTYPE)
            write_idx += bs

    assert write_idx == n, f"Expected {n}, wrote {write_idx}"

    # Save atomically (torch.save implicitly syncs GPU when reading tensors)
    tmp = shard_path.with_suffix(".tmp")
    torch.save(
        {
            "patches": patches_buf,
            "cls": cls_buf,
            "paths": paths,
            "class_idxs": torch.tensor(class_idxs, dtype=torch.int32),
            "failed_indices": failed,
            "shard_id": shard_id,
            "start_idx": start_idx,
            "end_idx": start_idx + n,
            "parquet_path": str(parquet_path),
            "parquet_sha256": parquet_hash,
            "teacher_model": cfg.teacher_model,
            "teacher_ckpt": str(teacher_ckpt),
            "image_size": cfg.image_size,
            "shard_size": cfg.shard_size,
            "dtype": str(STORAGE_DTYPE),
            "embed_dim": embed_dim,
            "n_patches": n_patches,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": SCHEMA_VERSION,
        },
        tmp,
    )
    tmp.rename(shard_path)

    elapsed = time.perf_counter() - t0
    shard_bytes = shard_path.stat().st_size

    # Update global progress
    pbar_global.update(n)
    pbar_global.set_postfix(
        {
            "img/s": f"{n / elapsed:.0f}",
            "MB/s": f"{shard_bytes / elapsed / 1e6:.0f}",
            "fail": len(failed),
        }
    )

    # Cleanup
    del patches_buf, cls_buf, loader
    gc.collect()
    torch.cuda.empty_cache()

    return n, len(failed), shard_bytes


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(cfg: Config) -> None:
    device = torch.device("cuda")

    # Resolve paths
    parquet_path, image_root, out_dir, teacher_ckpt = resolve_paths(cfg)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"out_dir: {out_dir}")
    log.info(f"dtype: {STORAGE_DTYPE} ({STORAGE_BYTES} bytes)")

    # Parquet metadata
    n_images = pq.read_metadata(parquet_path).num_rows
    n_shards = ceil(n_images / cfg.shard_size)
    log.info(f"Parquet: {n_images:,} images → {n_shards} shards")

    # Shard range
    start, end = get_shard_range(cfg, n_shards)

    # Preflight (no GPU yet)
    preflight_checks(
        parquet_path, image_root, teacher_ckpt, cfg, n_images, n_shards, start, end
    )
    log.info("Preflight OK")

    # Parquet hash
    parquet_hash = sha256_file(parquet_path)
    log.info(f"Parquet hash: {parquet_hash}")

    # Load teacher
    teacher = (
        create_backbone(cfg.teacher_model, weights=str(teacher_ckpt)).to(device).eval()
    )
    for p in teacher.parameters():
        p.requires_grad = False

    patch_size = teacher.patch_size_px
    embed_dim = teacher.embed_dim
    n_patches = (cfg.image_size // patch_size) ** 2
    assert cfg.image_size % patch_size == 0, (
        f"{cfg.image_size} not divisible by {patch_size}"
    )
    log.info(f"Teacher: {cfg.teacher_model}, {embed_dim}d, {n_patches} patches")
    log_gpu("after teacher")

    # Warmup
    with torch.no_grad(), torch.autocast("cuda", dtype=STORAGE_DTYPE):
        teacher.forward_norm_features(
            torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device)
        )
    log_gpu("after warmup")

    # Determine work
    shards_todo = []
    images_todo = 0
    for sid in range(start, end):
        if (shards_dir / f"{sid:05d}.pt").exists():
            continue
        shard_start = sid * cfg.shard_size
        shard_end = min(shard_start + cfg.shard_size, n_images)
        shards_todo.append(sid)
        images_todo += shard_end - shard_start

    if not shards_todo:
        log.info("All shards exist")
        return

    est_bytes = estimate_bytes(cfg.shard_size, n_patches, embed_dim)
    est_total_gb = len(shards_todo) * est_bytes / 1e9
    log.info(
        f"Exporting: {len(shards_todo)} shards, {images_todo:,} images, ~{est_total_gb:.1f} GB"
    )

    # Load full parquet for slicing
    table = pq.read_table(parquet_path)

    # Export
    t0_total = time.perf_counter()
    total_bytes = 0
    total_failed = 0

    pbar = tqdm(total=images_todo, unit="img", desc="Export")

    for shard_id in shards_todo:
        start_idx = shard_id * cfg.shard_size
        end_idx = min(start_idx + cfg.shard_size, n_images)
        n = end_idx - start_idx

        slice_table = table.slice(start_idx, n)
        paths = slice_table.column("path").to_pylist()
        class_idxs = slice_table.column("class_idx").to_pylist()

        _, n_failed, shard_bytes = export_shard(
            shard_id=shard_id,
            paths=paths,
            class_idxs=class_idxs,
            start_idx=start_idx,
            image_root=image_root,
            shards_dir=shards_dir,
            teacher=teacher,
            device=device,
            cfg=cfg,
            n_patches=n_patches,
            embed_dim=embed_dim,
            parquet_path=parquet_path,
            parquet_hash=parquet_hash,
            teacher_ckpt=teacher_ckpt,
            pbar_global=pbar,
        )
        total_bytes += shard_bytes
        total_failed += n_failed

    pbar.close()
    elapsed = time.perf_counter() - t0_total

    log.info(
        f"Done: {images_todo:,} images, {total_bytes / 1e9:.2f} GB in {elapsed:.1f}s"
    )
    log.info(
        f"  {images_todo / elapsed:.0f} img/s, {total_bytes / elapsed / 1e6:.0f} MB/s"
    )
    if total_failed:
        log.warning(f"  {total_failed} failed images")
    log_gpu("final")


if __name__ == "__main__":
    main(tyro.cli(Config))
