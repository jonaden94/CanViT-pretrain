"""Benchmark SA-1B dataloader components: tar reads, JPEG decode, transforms, shard load.

CPU-only — no GPU needed. Measures throughput and per-worker RSS at different
image sizes and worker counts.

Usage:
  uv run python sa1b/bench_dataloader.py --tar /path/to/sa_000020.tar --shard /path/to/sa_000020.pt
  uv run python sa1b/bench_dataloader.py --tar ... --shard ... --image-sizes 1024 1500 --workers 0 1 2 4 8
"""

import logging
import os
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from canvit_pretrain.train.data.tar_images import TarImageReader
from canvit_utils.transforms import preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


@dataclass
class Config:
    tar: Path
    shard: Path
    image_sizes: list[int] = field(default_factory=lambda: [1024, 1500])
    workers: list[int] = field(default_factory=lambda: [0, 1, 2, 4])
    n_images: int = 200  # images to benchmark per config
    batch_size: int = 8  # small batch — we're measuring data speed, not GPU


def bench_components(reader: TarImageReader, names: list[str], size: int, n: int) -> None:
    """Time individual components: mmap read, PIL decode, transform."""
    transform = preprocess(size)
    names = names[:n]

    # 1) Raw mmap read (bytes from tar)
    t0 = time.perf_counter()
    raw_buffers = []
    for name in names:
        offset, length = reader.index[name]
        raw_buffers.append(reader._mmap[offset : offset + length])
    t_mmap = time.perf_counter() - t0

    # 2) PIL decode (bytes → Image)
    from io import BytesIO
    t0 = time.perf_counter()
    images = []
    for buf in raw_buffers:
        img = Image.open(BytesIO(buf)).convert("RGB")
        img.load()  # force decode
        images.append(img)
    t_decode = time.perf_counter() - t0

    # 3) Transform (Image → Tensor at target size)
    t0 = time.perf_counter()
    for img in images:
        transform(img)
    t_transform = time.perf_counter() - t0

    total = t_mmap + t_decode + t_transform
    log.info(f"  Components ({n} images @ {size}px):")
    log.info(f"    mmap read:  {t_mmap:.3f}s ({t_mmap/n*1000:.1f}ms/img, {t_mmap/total*100:.0f}%)")
    log.info(f"    PIL decode: {t_decode:.3f}s ({t_decode/n*1000:.1f}ms/img, {t_decode/total*100:.0f}%)")
    log.info(f"    transform:  {t_transform:.3f}s ({t_transform/n*1000:.1f}ms/img, {t_transform/total*100:.0f}%)")
    log.info(f"    total:      {total:.3f}s ({n/total:.1f} img/s)")


class BenchDataset(IterableDataset):
    """Mimics AllShardsDataset but for a single shard, limited to n_images."""

    def __init__(self, shard_path: Path, tar_path: Path, size: int, n_images: int) -> None:
        self.shard_path = shard_path
        self.tar_path = tar_path
        self.size = size
        self.n_images = n_images
        self.transform = preprocess(size)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        shard = torch.load(self.shard_path, map_location="cpu", weights_only=False, mmap=True)
        reader = TarImageReader(self.tar_path)
        paths = shard["paths"]
        n = min(self.n_images, len(paths))

        for i in range(worker_id, n, num_workers):
            img = reader.read_image(paths[i])
            tensor = self.transform(img)
            yield tensor, shard["patches"][i].clone(), shard["cls"][i].clone()

        reader.close()


def bench_dataloader(cfg: Config, size: int, num_workers: int) -> None:
    """Measure end-to-end DataLoader throughput and worker RSS."""
    dataset = BenchDataset(cfg.shard, cfg.tar, size, cfg.n_images)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    n_batches = 0
    n_samples = 0
    t0 = time.perf_counter()
    for batch in loader:
        n_batches += 1
        n_samples += batch[0].shape[0]
        if n_samples >= cfg.n_images:
            break
    elapsed = time.perf_counter() - t0

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB → MB
    log.info(f"  DataLoader: {num_workers}w, {size}px, {n_samples} imgs in {elapsed:.2f}s → {n_samples/elapsed:.1f} img/s, RSS={rss_mb:.0f}MB")


def bench_shard_load(shard_path: Path) -> None:
    """Time shard loading with mmap."""
    # Cold: first load
    t0 = time.perf_counter()
    shard = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
    n = len(shard["paths"])
    t_load = time.perf_counter() - t0
    log.info(f"  Shard load (mmap): {t_load:.2f}s, {n} samples")

    # Access patches[0] to measure page fault cost
    t0 = time.perf_counter()
    _ = shard["patches"][0].clone()
    t_access = time.perf_counter() - t0
    log.info(f"  First patch access: {t_access*1000:.1f}ms")
    del shard


def main(cfg: Config) -> None:
    log.info(f"=== Dataloader Benchmark ===")
    log.info(f"tar: {cfg.tar}")
    log.info(f"shard: {cfg.shard}")
    log.info(f"image_sizes: {cfg.image_sizes}")
    log.info(f"workers: {cfg.workers}")
    log.info(f"n_images: {cfg.n_images}")
    log.info(f"PID: {os.getpid()}, CPUs available: {os.cpu_count()}")

    # 1) Shard load timing
    log.info("\n--- Shard Load ---")
    bench_shard_load(cfg.shard)

    # 2) Tar index timing
    log.info("\n--- Tar Index ---")
    t0 = time.perf_counter()
    reader = TarImageReader(cfg.tar)
    t_index = time.perf_counter() - t0
    names = list(reader.index.keys())
    log.info(f"  Index: {len(names)} JPEGs in {t_index:.1f}s")

    # 3) Component breakdown per image size
    for size in cfg.image_sizes:
        log.info(f"\n--- Components @ {size}px ---")
        bench_components(reader, names, size, cfg.n_images)

    reader.close()

    # 4) DataLoader throughput: size × workers grid
    log.info(f"\n--- DataLoader Throughput ---")
    for size in cfg.image_sizes:
        for nw in cfg.workers:
            bench_dataloader(cfg, size, nw)


if __name__ == "__main__":
    main(tyro.cli(Config))
