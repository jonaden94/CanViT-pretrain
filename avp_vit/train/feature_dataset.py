"""Shard-based feature loading with deterministic resume.

Design:
- Single IterableDataset that iterates over ALL shards sequentially
- Workers split SAMPLES within each shard (not shards across workers)
- Deterministic order: shard 0, 1, 2, ..., n-1, 0, 1, ... (no shuffling)
- Progress tracked via `shards_completed` for checkpoint/resume
- Persistent workers work naturally (dataset controls iteration)
"""

import logging
import time
from pathlib import Path
from typing import Iterator, TypedDict

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .transforms import val_transform

log = logging.getLogger(__name__)


class LoaderState(TypedDict):
    """Checkpoint state for ShardedFeatureLoader."""
    shards_completed: int


class AllShardsDataset(IterableDataset[tuple[Tensor, Tensor, Tensor, int]]):
    """IterableDataset that iterates over all shards sequentially, forever.

    Workers split samples within each shard via `sample_idx % num_workers == worker_id`.
    Tracks `shards_completed` for resume support.
    """

    def __init__(
        self,
        shard_files: list[Path],
        image_root: Path,
        image_size: int,
        start_shard: int = 0,
    ) -> None:
        self.shard_files = shard_files
        self.image_root = Path(image_root)
        self.image_size = image_size
        self.start_shard = start_shard
        self.transform = val_transform(image_size)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor, int]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        n_shards = len(self.shard_files)
        shard_counter = self.start_shard

        while True:
            shard_idx = shard_counter % n_shards
            shard_path = self.shard_files[shard_idx]

            if worker_id == 0:
                log.info(f"Worker 0: starting shard {shard_idx}/{n_shards} ({shard_path.name})")

            t0 = time.perf_counter()
            shard = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
            t_load = time.perf_counter() - t0

            t1 = time.perf_counter()
            n_samples = len(shard["paths"])
            failed_indices = set(shard.get("failed_indices", []))
            t_parse = time.perf_counter() - t1

            if worker_id == 0:
                log.info(f"  torch.load: {t_load:.3f}s, parse: {t_parse:.3f}s, {n_samples} samples")
                if failed_indices:
                    log.warning(f"  {len(failed_indices)} pre-marked failures")

            yielded = 0
            skipped = 0

            for i in range(worker_id, n_samples, num_workers):
                if i in failed_indices:
                    skipped += 1
                    continue

                rel_path = shard["paths"][i]
                try:
                    with Image.open(self.image_root / rel_path) as f:
                        img = f.convert("RGB")
                    img_tensor = self.transform(img)
                except Exception as e:
                    log.warning(f"Worker {worker_id}: RUNTIME FAILURE {rel_path}: {e}")
                    skipped += 1
                    continue

                assert isinstance(img_tensor, Tensor)
                yield (
                    img_tensor,
                    shard["patches"][i].clone(),
                    shard["cls"][i].clone(),
                    int(shard["class_idxs"][i]),
                )
                yielded += 1

            if worker_id == 0:
                log.debug(f"  Worker 0: shard done, yielded={yielded}, skipped={skipped}")

            shard_counter += 1


class ShardedFeatureLoader:
    """Infinite loader over shards with checkpoint/resume support.

    Wraps AllShardsDataset + DataLoader. Tracks shards_completed for checkpointing.
    """

    def __init__(
        self,
        shards_dir: Path,
        image_root: Path,
        image_size: int,
        batch_size: int,
        num_workers: int,
    ) -> None:
        self.shards_dir = Path(shards_dir)
        self.image_root = Path(image_root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        log.info(f"Globbing shards in {self.shards_dir}...")
        t0 = time.perf_counter()
        self.shard_files = sorted(self.shards_dir.glob("*.pt"))
        log.info(f"  Found {len(self.shard_files)} shards in {time.perf_counter() - t0:.2f}s")
        assert self.shard_files, f"No shards found in {shards_dir}"

        # Read first shard to get samples_per_shard (all shards same size)
        first_shard = torch.load(self.shard_files[0], map_location="cpu", weights_only=False)
        samples_per_shard = len(first_shard["paths"])
        del first_shard
        log.info(f"  {samples_per_shard} samples/shard")

        self.shards_completed = 0
        self.batches_per_shard = samples_per_shard // batch_size

        # Will be created lazily on first iteration
        self.loader: DataLoader | None = None
        self.loader_iter: Iterator | None = None
        self.batch_in_shard = 0

    def state_dict(self) -> LoaderState:
        """Return checkpoint state."""
        return {"shards_completed": self.shards_completed}

    def load_state_dict(self, state: LoaderState) -> None:
        """Restore from checkpoint."""
        self.shards_completed = state["shards_completed"]
        log.info(f"Resumed ShardedFeatureLoader at shard {self.shards_completed}")

    def _create_loader(self) -> DataLoader:
        """Create DataLoader with dataset starting at current shard."""
        dataset = AllShardsDataset(
            shard_files=self.shard_files,
            image_root=self.image_root,
            image_size=self.image_size,
            start_shard=self.shards_completed,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def next(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get next batch. Tracks shard completion for checkpointing."""
        if self.loader is None:
            log.info(f"Creating DataLoader with {self.num_workers} workers, persistent={self.num_workers > 0}")
            self.loader = self._create_loader()
            self.loader_iter = iter(self.loader)
            self.batch_in_shard = 0

        assert self.loader_iter is not None
        batch = next(self.loader_iter)

        self.batch_in_shard += 1
        if self.batch_in_shard >= self.batches_per_shard:
            self.shards_completed += 1
            self.batch_in_shard = 0
            log.info(f"Shard boundary: shards_completed={self.shards_completed}")

        return batch
