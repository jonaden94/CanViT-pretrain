#!/usr/bin/env python
"""Benchmark data loader throughput.

Interactive use:
    uv run ipython -i scripts/bench_loaders.py -- --train-dir /path/to/train --index-dir /path/to/index --num-workers 8

Or import and call:
    from scripts.bench_loaders import bench_image_loader, bench_feature_loader
"""

import argparse
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from drac_imagenet import IndexedImageFolder

from avp_vit.train.data import train_transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ============================================================================
# CONFIG - tweak these
# ============================================================================
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 8
DEFAULT_IMAGE_SIZE = 512
DEFAULT_NUM_BATCHES = 100  # how many batches to time


def bench_image_loader(
    train_dir: Path,
    index_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_batches: int = DEFAULT_NUM_BATCHES,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> float:
    """Benchmark raw image loading throughput.

    Returns images/sec.
    """
    log.info("=== Image Loader Benchmark ===")
    log.info(f"  train_dir: {train_dir}")
    log.info(f"  index_dir: {index_dir}")
    log.info(f"  batch_size: {batch_size}")
    log.info(f"  num_workers: {num_workers}")
    log.info(f"  image_size: {image_size}")
    log.info(f"  num_batches: {num_batches}")
    log.info(f"  pin_memory: {pin_memory}")
    log.info(f"  persistent_workers: {persistent_workers}")

    tf = train_transform(image_size, (0.8, 1.0))
    ds = IndexedImageFolder(train_dir, index_dir, tf)
    log.info(f"  dataset size: {len(ds):,} images")

    persistent = persistent_workers and num_workers > 0
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent,
    )

    log.info("Warming up (1 batch)...")
    it = iter(loader)
    _ = next(it)

    log.info(f"Timing {num_batches} batches...")
    t0 = time.perf_counter()
    for i, (images, labels) in enumerate(tqdm(it, total=num_batches, desc="Loading")):
        if i >= num_batches - 1:  # -1 because we already did 1 warmup
            break
    elapsed = time.perf_counter() - t0

    total_images = num_batches * batch_size
    imgs_per_sec = total_images / elapsed
    log.info(f"  elapsed: {elapsed:.2f}s")
    log.info(f"  throughput: {imgs_per_sec:.1f} img/sec")
    log.info(f"  per batch: {elapsed / num_batches * 1000:.1f}ms")
    return imgs_per_sec


def bench_feature_loader(
    shards_dir: Path,
    image_root: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    num_batches: int = DEFAULT_NUM_BATCHES,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> float:
    """Benchmark feature loading throughput (images + precomputed features).

    Returns images/sec.
    """
    from avp_vit.train.feature_dataset import FeatureIterableDataset

    log.info("=== Feature Loader Benchmark ===")
    log.info(f"  shards_dir: {shards_dir}")
    log.info(f"  image_root: {image_root}")
    log.info(f"  batch_size: {batch_size}")
    log.info(f"  num_workers: {num_workers}")
    log.info(f"  num_batches: {num_batches}")
    log.info(f"  pin_memory: {pin_memory}")
    log.info(f"  persistent_workers: {persistent_workers}")

    ds = FeatureIterableDataset(shards_dir, image_root)
    log.info(f"  shards: {len(ds.shard_files)}")

    persistent = persistent_workers and num_workers > 0
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent,
    )

    log.info("Warming up (1 batch)...")
    it = iter(loader)
    batch = next(it)
    images, patches, cls_tokens, labels = batch
    log.info(f"  batch shapes: images={images.shape}, patches={patches.shape}, cls={cls_tokens.shape}")
    log.info(f"  dtypes: images={images.dtype}, patches={patches.dtype}, cls={cls_tokens.dtype}")

    log.info(f"Timing {num_batches} batches...")
    t0 = time.perf_counter()
    for i, batch in enumerate(tqdm(it, total=num_batches, desc="Loading")):
        if i >= num_batches - 1:
            break
    elapsed = time.perf_counter() - t0

    total_images = num_batches * batch_size
    imgs_per_sec = total_images / elapsed
    log.info(f"  elapsed: {elapsed:.2f}s")
    log.info(f"  throughput: {imgs_per_sec:.1f} img/sec")
    log.info(f"  per batch: {elapsed / num_batches * 1000:.1f}ms")
    return imgs_per_sec


def bench_feature_loader_features_only(
    shards_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    num_batches: int = DEFAULT_NUM_BATCHES,
) -> float:
    """Benchmark loading just features (no images) - to isolate image loading cost.

    Uses a minimal IterableDataset that skips image loading.
    """
    from torch.utils.data import IterableDataset, get_worker_info

    log.info("=== Features-Only Benchmark (no image loading) ===")
    log.info(f"  shards_dir: {shards_dir}")
    log.info(f"  batch_size: {batch_size}")
    log.info(f"  num_workers: {num_workers}")
    log.info(f"  num_batches: {num_batches}")

    shard_files = sorted(Path(shards_dir).glob("*.pt"))
    log.info(f"  shards: {len(shard_files)}")

    class FeaturesOnlyDataset(IterableDataset):
        def __iter__(self):
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info else 0
            n_workers = worker_info.num_workers if worker_info else 1
            my_shards = shard_files[worker_id::n_workers]

            for shard_path in my_shards:
                shard = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
                for i in range(len(shard["paths"])):
                    yield shard["patches"][i], shard["cls"][i], shard["class_idxs"][i]

    ds = FeaturesOnlyDataset()
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, drop_last=True)

    log.info("Warming up (1 batch)...")
    it = iter(loader)
    patches, cls_tokens, labels = next(it)
    log.info(f"  batch shapes: patches={patches.shape}, cls={cls_tokens.shape}")

    log.info(f"Timing {num_batches} batches...")
    t0 = time.perf_counter()
    for i, batch in enumerate(tqdm(it, total=num_batches, desc="Loading")):
        if i >= num_batches - 1:
            break
    elapsed = time.perf_counter() - t0

    total_images = num_batches * batch_size
    imgs_per_sec = total_images / elapsed
    log.info(f"  elapsed: {elapsed:.2f}s")
    log.info(f"  throughput: {imgs_per_sec:.1f} img/sec (features only)")
    log.info(f"  per batch: {elapsed / num_batches * 1000:.1f}ms")
    return imgs_per_sec


def sweep_workers(
    bench_fn,
    worker_counts: list[int] = [0, 1, 2, 4, 8, 12, 16],
    **kwargs,
) -> dict[int, float]:
    """Sweep num_workers and return throughput for each."""
    results: dict[int, float] = {}
    for nw in worker_counts:
        log.info(f"\n{'='*60}")
        log.info(f"num_workers = {nw}")
        log.info(f"{'='*60}")
        results[nw] = bench_fn(num_workers=nw, **kwargs)

    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    for nw, throughput in results.items():
        log.info(f"  {nw} workers: {throughput:.1f} img/sec")

    best_nw = max(results, key=lambda k: results[k])
    log.info(f"Best: {best_nw} workers @ {results[best_nw]:.1f} img/sec")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark data loaders")
    parser.add_argument("--train-dir", type=Path, help="Path to training images")
    parser.add_argument("--index-dir", type=Path, help="Path to index (required with --train-dir)")
    parser.add_argument("--shards-dir", type=Path, help="Path to feature shards")
    parser.add_argument("--image-root", type=Path, help="Path to images for feature loader")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--num-batches", type=int, default=DEFAULT_NUM_BATCHES)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--sweep", action="store_true", help="Sweep worker counts")
    args = parser.parse_args()

    log.info(f"PyTorch {torch.__version__}")
    log.info(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

    if args.train_dir:
        assert args.index_dir, "--index-dir required with --train-dir"
        if args.sweep:
            sweep_workers(
                bench_image_loader,
                train_dir=args.train_dir,
                index_dir=args.index_dir,
                batch_size=args.batch_size,
                image_size=args.image_size,
                num_batches=args.num_batches,
            )
        else:
            bench_image_loader(
                args.train_dir,
                args.index_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                image_size=args.image_size,
                num_batches=args.num_batches,
            )

    if args.shards_dir and args.image_root:
        if args.sweep:
            sweep_workers(
                bench_feature_loader,
                shards_dir=args.shards_dir,
                image_root=args.image_root,
                batch_size=args.batch_size,
                num_batches=args.num_batches,
            )
        else:
            bench_feature_loader(
                args.shards_dir,
                args.image_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                num_batches=args.num_batches,
            )

    if args.shards_dir and not args.image_root:
        # Features only benchmark
        if args.sweep:
            sweep_workers(
                bench_feature_loader_features_only,
                shards_dir=args.shards_dir,
                batch_size=args.batch_size,
                num_batches=args.num_batches,
            )
        else:
            bench_feature_loader_features_only(
                args.shards_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                num_batches=args.num_batches,
            )

    log.info("\nDone. For interactive use:")
    log.info("  from scripts.bench_loaders import bench_image_loader, bench_feature_loader")
    log.info("  bench_image_loader(Path('/path/to/train'), Path('/path/to/index'), num_workers=4)")
