"""Data loading for CanViT pretraining: the WebDataset train loader, the ImageFolder val
loader, and the batch types.
"""

import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from torch import Tensor

if TYPE_CHECKING:
    from ..config import Config
import torch
from canvit_pytorch.preprocess import preprocess
from torch.utils.data import DataLoader, Dataset, Subset

from .indexed_image_folder import IndexedImageFolder
from .webdataset import WebDatasetTrainLoader

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


class FixedValLoader:
    """Deterministic, finite loader over a fixed N-sample subset of the val set.

    Always yields the SAME images in the SAME order on every pass (re-iterable via
    ``batches()``), independent of world size — the subset is chosen once by a fixed
    seed from the full val set. Used for rank-0 validation: no cycling, no
    per-rank/per-call sample drift.
    """

    def __init__(self, loader: DataLoader, n_samples: int) -> None:
        self._loader = loader
        self.n_samples = n_samples

    def batches(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield ``(images, labels)`` for the fixed subset, one chunk per step."""
        for batch in self._loader:
            images, labels = batch[0], batch[1]
            if not isinstance(labels, Tensor):
                labels = torch.as_tensor(labels, dtype=torch.long)
            yield images, labels


class Loaders(NamedTuple):
    """Train and validation data loaders."""

    train: WebDatasetTrainLoader
    val: FixedValLoader


def scene_size_px(grid_size: int, patch_size: int) -> int:
    return grid_size * patch_size


def create_loaders(
    cfg: "Config",
    *,
    job_index: int = 0,
    world_size: int = 1,
    rank: int = 0,
) -> Loaders:
    """Train + val loaders.

    Training reads WebDataset shards from ``cfg.webdataset_dir`` (rank-aware,
    ``job_index``-driven shard schedule, resumable). Shards may carry precomputed teacher
    features or be raw jpg+json, in which case the teacher runs on the fly — both go
    through ``WebDatasetTrainLoader``.

    Validation ALWAYS reads the raw ImageNet-1k val ImageFolder at ``cfg.val_dir``
    (synset-named class subfolders), independent of the training data source — teacher
    targets are computed live during validation.
    """
    from ..config import Config
    assert isinstance(cfg, Config)
    assert cfg.webdataset_dir is not None, (
        "cfg.webdataset_dir is required: it is the only training data source. Pass "
        "--cfg.webdataset-dir (launchers pass $WEBDATASET_DIR from .envrc.grete). The "
        "precomputed-feature tar path that used to serve cfg.webdataset_dir=None was "
        "removed 2026-07-31."
    )
    log.info(f"=== CREATE_LOADERS: job_index={job_index}, rank={rank}/{world_size} ===")

    val_loader = _create_imagefolder_val_loader(cfg)
    train_loader = _create_webdataset_train_loader(
        cfg, job_index=job_index, world_size=world_size, rank=rank
    )
    return Loaders(train=train_loader, val=val_loader)


def _create_imagefolder_val_loader(cfg: "Config") -> FixedValLoader:
    """Validation loader over a fixed N-sample subset of the raw ImageNet-1k val
    ImageFolder (``cfg.val_dir``).

    Synset-named class subfolders; labels are the canonical sorted-synset 0-999
    indices (probe-compatible). The parquet index is cached under
    ``cfg.val_index_dir`` so the directory scan happens once across runs.

    The subset is ``min(cfg.n_val_samples, len(val_set))`` images, drawn once by a
    seeded permutation (``cfg.val_seed``) over the full val set — so the SAME images
    are used in every validation, independent of ``batch_size_per_gpu`` and of world
    size. Iterated with ``shuffle=False`` in fixed order; metrics over the subset are
    identical regardless of the chunk (batch) size.
    """
    val_dir = cfg.val_dir
    assert val_dir.is_dir(), f"val_dir not found: {val_dir}"

    sz = cfg.scene_resolution
    persistent = cfg.num_workers > 0
    val_tf = preprocess(sz)

    if cfg.val_index_dir is not None:
        val_index_dir = cfg.val_index_dir
        log.info(f"Val: using val_index_dir={val_index_dir}")
    elif cfg.train_index_dir is not None:
        val_index_dir = cfg.train_index_dir
        log.info(f"Val: val_index_dir not set, using train_index_dir={val_index_dir}")
    else:
        val_index_dir = Path(tempfile.mkdtemp(prefix="avp_val_index_"))
        log.info(f"Val: no index_dir available, using temp dir: {val_index_dir}")

    val_ds: Dataset[tuple] = IndexedImageFolder(val_dir, val_index_dir, val_tf)
    n_total = len(val_ds)
    assert n_total > 0, "val dataset empty"

    n = min(cfg.n_val_samples, n_total)
    gen = torch.Generator().manual_seed(cfg.val_seed)
    indices = torch.randperm(n_total, generator=gen)[:n].tolist()
    subset = Subset(val_ds, indices)
    log.info(
        f"Val: raw ImageFolder {val_dir} — fixed subset {n}/{n_total} images "
        f"(seed={cfg.val_seed}), resolution: {sz}px, chunk={cfg.batch_size_per_gpu}"
    )
    loader = DataLoader(
        subset, batch_size=cfg.batch_size_per_gpu, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False, persistent_workers=persistent,
    )
    return FixedValLoader(loader, n_samples=n)


def _create_webdataset_train_loader(
    cfg: "Config",
    *,
    job_index: int,
    world_size: int,
    rank: int,
) -> WebDatasetTrainLoader:
    """Build the WebDataset training loader for the rank-aware path."""
    assert cfg.webdataset_dir is not None
    train_dir = cfg.webdataset_dir / "train-shuffled"
    assert train_dir.is_dir(), f"train dir not found: {train_dir}"

    log.info(f"WebDataset path: {cfg.webdataset_dir}")
    log.info(f"  train: {train_dir}")

    return WebDatasetTrainLoader(
        train_dir=train_dir,
        seed=cfg.seed,
        job_index=job_index,
        batch_size_per_gpu=cfg.batch_size_per_gpu,
        steps_per_job=cfg.steps_per_job,
        image_size=cfg.scene_resolution,
        world_size=world_size,
        rank=rank,
        num_workers=cfg.num_workers,
    )
