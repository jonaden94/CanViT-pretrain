"""Smoke test for FeatureIterableDataset.

Usage:
    source slurm/env.sh
    uv run python scripts/test_feature_dataset.py \
        --shards-dir $FEATURES_DIR/in21k/dinov3_vitb16/512/shards \
        --image-root $IN21K_DIR
"""

import argparse
import logging
import time
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-batches", type=int, default=50)
    args = parser.parse_args()

    log.info("Importing FeatureIterableDataset...")
    t0 = time.perf_counter()
    from avp_vit.train.feature_dataset import FeatureIterableDataset
    log.info(f"  Import took {time.perf_counter() - t0:.2f}s")

    log.info("Creating dataset...")
    log.info(f"  shards_dir: {args.shards_dir}")
    log.info(f"  image_root: {args.image_root}")
    t0 = time.perf_counter()
    ds = FeatureIterableDataset(args.shards_dir, args.image_root)
    log.info(f"  Dataset created in {time.perf_counter() - t0:.2f}s")
    log.info(f"  n_shards: {len(ds.shard_files)}")

    log.info("\n=== Throughput test ===")
    log.info(f"  batch_size={args.batch_size}, num_workers={args.num_workers}, num_batches={args.num_batches}")

    log.info("Creating DataLoader...")
    t0 = time.perf_counter()
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)
    log.info(f"  DataLoader created in {time.perf_counter() - t0:.2f}s")

    log.info("Starting iteration...")
    n_images = 0
    t0 = time.perf_counter()
    for i, batch in enumerate(tqdm(loader, total=args.num_batches, desc="Batches")):
        if i >= args.num_batches:
            break
        images, patches, cls, class_idx = batch
        n_images += images.shape[0]
        if i == 0:
            log.info(f"  First batch shapes: images={images.shape}, patches={patches.shape}, cls={cls.shape}")
    elapsed = time.perf_counter() - t0

    log.info("\n=== Results ===")
    log.info(f"  {n_images:,} images in {elapsed:.2f}s")
    log.info(f"  Throughput: {n_images/elapsed:.0f} images/sec")
    log.info(f"  Batches/sec: {args.num_batches/elapsed:.1f}")

    log.info("\n✓ Test passed")


if __name__ == "__main__":
    main()
