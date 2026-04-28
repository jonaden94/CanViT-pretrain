"""Data loading for CanViT pretraining: shard loaders + batch types."""

import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from torch import Tensor

if TYPE_CHECKING:
    from ..config import Config
from canvit_pytorch.preprocess import preprocess
from canvit_pretrain.datasets import IndexedImageFolder
from torch.utils.data import DataLoader, Dataset

from .schedule import SCHEDULE_FILENAME
from .shards import ShardedFeatureLoader
from .webdataset import WebDatasetTrainLoader, WebDatasetValLoader

log = logging.getLogger(__name__)

type Batch = tuple[Tensor, ...]  # Generic batch (images, labels, ...)


# IN21k contains corrupt images that cause DataLoader workers to fail.
# Observed PIL errors: "Corrupt EXIF data", "Truncated File Read", UnidentifiedImageError.
# See bad_images.txt for the full list. Skip failed batches up to this limit.
MAX_CONSECUTIVE_FAILURES = 10


class InfiniteLoader:
    """Infinite iterator over a DataLoader with retry on worker errors.

    Note: We use explicit iterator management instead of a generator because
    when an exception propagates out of a Python generator, the generator is
    finalized (gi_frame=None) and subsequent next() calls raise StopIteration.

    Used only for the val loader (map-style IndexedImageFolder dataset).
    """

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self._iter: Iterator[Batch] | None = None

    def _next_with_retry(self) -> Batch:
        failures = 0
        while True:
            if self._iter is None:
                self._iter = iter(self._loader)
            try:
                return next(self._iter)
            except StopIteration:
                # End of epoch - start new one
                self._iter = iter(self._loader)
            except Exception as e:
                failures += 1
                log.warning(f"Batch failed ({failures}/{MAX_CONSECUTIVE_FAILURES}): {e}")
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(f"{MAX_CONSECUTIVE_FAILURES} consecutive batch failures") from e
                # Worker error corrupts iterator state - reset it
                self._iter = None

    def next(self) -> Batch:
        """Get next batch (raw tuple from DataLoader)."""
        return self._next_with_retry()

    def next_batch(self) -> Tensor:
        """Get images only (first element of batch)."""
        images, *_ = self._next_with_retry()
        return images

    def next_batch_with_labels(self) -> tuple[Tensor, Tensor]:
        """Get (images, labels) - for raw image loaders."""
        batch = self._next_with_retry()
        return batch[0], batch[1]


class Loaders(NamedTuple):
    """Train and validation data loaders."""

    train: ShardedFeatureLoader | WebDatasetTrainLoader
    val: InfiniteLoader | WebDatasetValLoader


def scene_size_px(grid_size: int, patch_size: int) -> int:
    return grid_size * patch_size


def create_loaders(
    cfg: "Config",
    start_step: int,
    *,
    job_index: int = 0,
    world_size: int = 1,
    rank: int = 0,
) -> Loaders:
    """Train + val loaders. Dispatches on cfg.webdataset_dir:
      - set: WebDataset path (rank-aware, job_index-driven shard schedule).
      - unset: existing precomputed-features path (raw images for val).
    start_step positions the shard cursor on resume (sharded path); job_index
    plays the analogous role for the WebDataset path.
    """
    from ..config import Config
    assert isinstance(cfg, Config)
    log.info(
        f"=== CREATE_LOADERS: start_step={start_step}, job_index={job_index}, "
        f"rank={rank}/{world_size} ==="
    )

    if cfg.webdataset_dir is not None:
        return _create_webdataset_loaders(cfg, job_index=job_index, world_size=world_size, rank=rank)

    val_dir = cfg.val_dir
    assert val_dir.is_dir(), f"val_dir not found: {val_dir}"

    sz = cfg.scene_resolution
    persistent = cfg.num_workers > 0

    # Train loader: precomputed features (required)
    assert cfg.feature_base_dir is not None, "feature_base_dir required"
    assert (cfg.feature_image_root is None) != (cfg.tar_dir is None), \
        "Exactly one of feature_image_root or tar_dir required"
    log.info("Train: using PRECOMPUTED FEATURES (ShardedFeatureLoader)")
    shards_dir = cfg.feature_base_dir / cfg.teacher_name / str(sz) / "shards"
    log.info(f"  feature_base_dir: {cfg.feature_base_dir}")
    log.info(f"  teacher_name: {cfg.teacher_name}")
    log.info(f"  resolution: {sz}")
    log.info(f"  → shards_dir: {shards_dir}")
    if cfg.tar_dir is not None:
        log.info(f"  tar_dir: {cfg.tar_dir} (images read directly from tar)")
    else:
        log.info(f"  image_root: {cfg.feature_image_root}")
    assert shards_dir.is_dir(), f"shards_dir not found: {shards_dir}"
    train_loader = ShardedFeatureLoader(
        shards_dir=shards_dir,
        image_size=sz,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        start_step=start_step,
        image_root=cfg.feature_image_root,
        tar_dir=cfg.tar_dir,
        steps_per_job=cfg.steps_per_job,
    )
    log.info(f"  {len(train_loader.shard_files)} shards, start_shard={train_loader.start_shard}")

    # Val loader
    val_tf = preprocess(sz)
    if cfg.val_index_dir is not None:
        val_index_dir = cfg.val_index_dir
        log.info(f"Val: using provided val_index_dir={val_index_dir}")
    elif cfg.train_index_dir is not None:
        val_index_dir = cfg.train_index_dir
        log.info(f"Val: val_index_dir not set, using train_index_dir={val_index_dir}")
    else:
        val_index_dir = Path(tempfile.mkdtemp(prefix="avp_val_index_"))
        log.info(f"Val: no index_dir available, using temp dir: {val_index_dir}")
    val_ds: Dataset[tuple] = IndexedImageFolder(val_dir, val_index_dir, val_tf)
    assert len(val_ds) > 0, "val dataset empty"
    log.info(f"Val dataset: {len(val_ds):,} images, resolution: {sz}px")
    # CRITICAL: shuffle=True required! Without it, batches are sequential
    # (all tench, then all goldfish, etc.) which gives misleading metrics.
    val_loader = InfiniteLoader(DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True, persistent_workers=persistent,
    ))

    return Loaders(train=train_loader, val=val_loader)


def _create_webdataset_loaders(
    cfg: "Config",
    *,
    job_index: int,
    world_size: int,
    rank: int,
) -> Loaders:
    """Build WebDataset train + val loaders for the rank-aware path."""
    assert cfg.webdataset_dir is not None
    train_dir = cfg.webdataset_dir / "train-shuffled"
    val_dir_wds = cfg.webdataset_dir / "val"
    assert train_dir.is_dir(), f"train dir not found: {train_dir}"
    assert val_dir_wds.is_dir(), f"val dir not found: {val_dir_wds}"

    schedule_path = cfg.shard_schedule_path or (train_dir / SCHEDULE_FILENAME)

    log.info(f"WebDataset path: {cfg.webdataset_dir}")
    log.info(f"  train: {train_dir}")
    log.info(f"  val: {val_dir_wds}")
    log.info(f"  schedule: {schedule_path}")

    train_loader = WebDatasetTrainLoader(
        train_dir=train_dir,
        schedule_path=schedule_path,
        job_index=job_index,
        batch_size_per_gpu=cfg.batch_size,
        steps_per_job=cfg.steps_per_job,
        image_size=cfg.scene_resolution,
        world_size=world_size,
        rank=rank,
    )
    val_loader = WebDatasetValLoader(
        val_dir=val_dir_wds,
        batch_size_per_gpu=cfg.batch_size,
        image_size=cfg.scene_resolution,
        world_size=world_size,
        rank=rank,
    )
    return Loaders(train=train_loader, val=val_loader)
