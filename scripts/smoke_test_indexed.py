#!/usr/bin/env python3
"""Smoke test for IndexedImageFolder on Compute Canada.

Usage:
    uv run python scripts/smoke_test_indexed.py \
        --root /datashare/imagenet/winter21_whole \
        --index-dir ./indices \
        --num-batches 5

This will:
1. Generate index files if missing (~15-30 min for 13M images)
2. Load train and val IndexedImageFolder (~0.5s each)
3. Create DataLoaders and iterate a few batches
4. Print timing, shapes, sample paths
"""

import argparse
import logging
import time
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test IndexedImageFolder")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root (e.g., winter21_whole)")
    parser.add_argument("--index-dir", type=Path, default=Path("indices"), help="Where to store/load indices")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for smoke test")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--num-batches", type=int, default=5, help="Number of batches to iterate")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("IndexedImageFolder Smoke Test")
    log.info("=" * 60)
    log.info(f"root: {args.root}")
    log.info(f"index_dir: {args.index_dir}")
    log.info(f"val_ratio: {args.val_ratio}, seed: {args.seed}")
    log.info(f"batch_size: {args.batch_size}, num_workers: {args.num_workers}")
    log.info("=" * 60)

    # Import here so logging is configured first
    from avp_vit.train.data.indexed import IndexedImageFolder, ensure_split_index

    # 1. Ensure indices exist (generates if missing)
    log.info("Step 1: Ensuring index files exist...")
    t0 = time.perf_counter()
    index = ensure_split_index(args.root, args.index_dir, args.val_ratio, args.seed)
    log.info(f"  Index ready in {time.perf_counter() - t0:.2f}s")
    log.info(f"  train: {index.train}")
    log.info(f"  val: {index.val}")

    # 2. Create datasets
    log.info("Step 2: Creating IndexedImageFolder datasets...")
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])

    t0 = time.perf_counter()
    train_ds = IndexedImageFolder(str(args.root), index.train, transform=transform)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    val_ds = IndexedImageFolder(str(args.root), index.val, transform=transform)
    val_time = time.perf_counter() - t0

    log.info(f"  Train dataset: {len(train_ds):,} samples, loaded in {train_time:.2f}s")
    log.info(f"  Val dataset: {len(val_ds):,} samples, loaded in {val_time:.2f}s")
    log.info(f"  Classes: {len(train_ds.classes):,}")

    # 3. Create DataLoaders
    log.info("Step 3: Creating DataLoaders...")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 4. Iterate batches
    log.info(f"Step 4: Iterating {args.num_batches} batches...")

    def iterate_batches(loader: DataLoader, name: str, n: int) -> None:
        times = []
        for i, (images, labels) in enumerate(loader):
            t0 = time.perf_counter()
            if i >= n:
                break
            times.append(time.perf_counter() - t0)
            if i == 0:
                log.info(f"  {name} batch 0: images={images.shape}, labels={labels.shape}")
                log.info(f"    dtype={images.dtype}, min={images.min():.3f}, max={images.max():.3f}")
        if times:
            avg = sum(times) / len(times)
            log.info(f"  {name}: {len(times)} batches, avg {avg*1000:.1f}ms/batch")

    iterate_batches(train_loader, "Train", args.num_batches)
    iterate_batches(val_loader, "Val", args.num_batches)

    # 5. Sample paths check
    log.info("Step 5: Checking sample paths...")
    log.info(f"  First train sample: {train_ds.samples[0][0]}")
    log.info(f"  First val sample: {val_ds.samples[0][0]}")

    # Verify files exist (spot check)
    first_train_path = Path(train_ds.samples[0][0])
    first_val_path = Path(val_ds.samples[0][0])
    log.info(f"  First train path exists: {first_train_path.exists()}")
    log.info(f"  First val path exists: {first_val_path.exists()}")

    log.info("=" * 60)
    log.info("Smoke test complete!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
