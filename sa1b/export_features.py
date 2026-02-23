"""Export DINOv3 features for one SA-1B tar.

Designed for SLURM array jobs: 1 task = 1 tar = 1 shard.

Each invocation:
  1. Extracts one tar to a temp dir (SLURM_TMPDIR or specified)
  2. Loads frozen DINOv3 teacher from HuggingFace Hub
  3. Runs batched inference on all JPEGs
  4. Saves one .pt shard (named after tar: sa_NNNNNN.pt)

Shard format matches training loader expectations (shards.py):
  patches:        [N, n_patches, embed_dim] float16
  cls:            [N, embed_dim] float16
  paths:          list[str] — filenames relative to image dir
  class_idxs:     [N] int32 (all 0 for SA-1B)
  failed_indices: list[int]
  image_hashes:   list[str] — xxh64 of decoded pixels

Usage:
  # Single tar (interactive)
  uv run python sa1b/export_features.py \
      --tar /path/to/sa_000020.tar \
      --out-dir /path/to/shards \
      --extract-dir $SLURM_TMPDIR/sa1b_images

  # SLURM array job (see sa1b/export_features.sh)
  sbatch --array=0-999 sa1b/export_features.sh
"""

import logging
import subprocess
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
import tyro
import xxhash
from canvit_utils.teacher import load_teacher
from PIL import Image, ImageFile
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from canvit_pretrain.train.transforms import val_transform

STORAGE_DTYPE = torch.float16
ImageFile.LOAD_TRUNCATED_IMAGES = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


@dataclass
class Config:
    tar: Path
    out_dir: Path
    extract_dir: Path
    image_size: int = 1024
    teacher_repo_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    batch_size: int = 32
    num_workers: int = 8


class ImageDataset(Dataset[tuple[Tensor, int, bool, str]]):
    def __init__(self, paths: list[Path], size: int) -> None:
        self.paths = paths
        self.transform = val_transform(size)
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, bool, str]:
        path = self.paths[idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with Image.open(path) as f:
                    img = f.convert("RGB")
                    img.load()
            img_hash = xxhash.xxh64(img.tobytes()).hexdigest()
            tensor = self.transform(img)
            assert isinstance(tensor, Tensor)
            return tensor, idx, True, img_hash
        except Exception as e:
            log.warning(f"Bad image {path}: {e}")
            return torch.full((3, self.size, self.size), float("nan")), idx, False, ""


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def extract_tar(tar: Path, dest: Path) -> int:
    """Extract tar to dest, return number of JPEGs extracted."""
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    subprocess.run(["tar", "xf", str(tar), "-C", str(dest)], check=True)
    elapsed = time.perf_counter() - t0
    jpgs = sorted(dest.glob("*.jpg"))
    log.info(f"Extracted {tar.name}: {len(jpgs)} JPEGs in {elapsed:.1f}s")
    return len(jpgs)


def main(cfg: Config) -> None:
    device = torch.device("cuda")
    tar_stem = cfg.tar.stem  # e.g. "sa_000020"
    shard_path = cfg.out_dir / f"{tar_stem}.pt"

    # Skip if already exported
    if shard_path.exists():
        log.info(f"Shard already exists: {shard_path}")
        return

    assert cfg.tar.exists(), f"Tar not found: {cfg.tar}"

    log.info(f"tar: {cfg.tar}")
    log.info(f"out_dir: {cfg.out_dir}")
    log.info(f"extract_dir: {cfg.extract_dir}")
    log.info(f"image_size: {cfg.image_size}")
    log.info(f"teacher_repo_id: {cfg.teacher_repo_id}")

    # Step 1: Extract
    extract_tar(cfg.tar, cfg.extract_dir)
    jpg_paths = sorted(cfg.extract_dir.glob("*.jpg"))
    n = len(jpg_paths)
    assert n > 0, f"No JPEGs found in {cfg.extract_dir}"
    log.info(f"{n} images to process")

    # Step 2: Load teacher
    teacher = load_teacher(cfg.teacher_repo_id, device)
    patch_size = teacher.model.config.patch_size
    embed_dim = teacher.embed_dim
    n_patches = (cfg.image_size // patch_size) ** 2
    assert cfg.image_size % patch_size == 0
    log.info(f"Teacher: {embed_dim}d, {n_patches} patches, patch_size={patch_size}")
    log.info(f"GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Step 3: Inference
    patches_buf = torch.empty(n, n_patches, embed_dim, dtype=STORAGE_DTYPE, device=device)
    cls_buf = torch.empty(n, embed_dim, dtype=STORAGE_DTYPE, device=device)
    hashes: list[str] = [""] * n
    failed: list[int] = []

    loader = DataLoader(
        ImageDataset(jpg_paths, cfg.image_size),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    t0 = time.perf_counter()
    write_idx = 0

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for imgs, indices, ok, batch_hashes in tqdm(loader, desc=tar_stem):
            for i, success, h in zip(indices.tolist(), ok.tolist(), batch_hashes):
                hashes[i] = h
                if not success:
                    failed.append(i)

            imgs = imgs.to(device)
            feats = teacher.forward_norm_features(imgs)
            bs = imgs.shape[0]
            patches_buf[write_idx : write_idx + bs] = feats.patches.to(STORAGE_DTYPE)
            cls_buf[write_idx : write_idx + bs] = feats.cls.to(STORAGE_DTYPE)
            write_idx += bs

    assert write_idx == n, f"Expected {n}, wrote {write_idx}"
    elapsed = time.perf_counter() - t0
    log.info(f"Inference: {n} images in {elapsed:.1f}s ({n / elapsed:.0f} img/s)")

    if failed:
        log.warning(f"{len(failed)} failed images")

    # Step 4: Save shard
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    filenames = [p.name for p in jpg_paths]

    tmp = shard_path.with_suffix(".tmp")
    torch.save(
        {
            "patches": patches_buf.cpu(),
            "cls": cls_buf.cpu(),
            "paths": filenames,
            "class_idxs": torch.zeros(n, dtype=torch.int32),
            "image_hashes": hashes,
            "failed_indices": failed,
            # Metadata
            "tar_name": cfg.tar.name,
            "image_size": cfg.image_size,
            "teacher_repo_id": cfg.teacher_repo_id,
            "dtype": str(STORAGE_DTYPE),
            "embed_dim": embed_dim,
            "n_patches": n_patches,
            "n_images": n,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": get_git_commit(),
        },
        tmp,
    )
    tmp.rename(shard_path)

    shard_mb = shard_path.stat().st_size / 1e6
    log.info(f"Saved {shard_path} ({shard_mb:.0f} MB, {n} images)")


if __name__ == "__main__":
    main(tyro.cli(Config))
