"""WebDataset-based train/val loaders.

Each rank receives a deterministically computed list of shard paths (via
`schedule.compute_schedule_slice`). DataLoader workers then split those shards
via `wds.split_by_worker` — one shard per worker (`num_workers = shards_per_gpu`).

Loader interface matches the existing `ShardedFeatureLoader` and
`InfiniteLoader` so `loop.py` can call `train_loader.next()` and
`val_loader.next_batch_with_labels()` without modification.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import webdataset as wds
from canvit_pytorch import CLSStandardizer, PatchStandardizer
from canvit_pytorch.preprocess import preprocess
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from .schedule import compute_schedule_slice, compute_shards_per_gpu

if TYPE_CHECKING:
    from ..config import Config

log = logging.getLogger(__name__)


def _decode_jpg(data: bytes, image_size: int) -> Tensor:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    tensor = preprocess(image_size)(img)
    assert isinstance(tensor, Tensor)
    return tensor


def _decode_label(data: bytes) -> int:
    return int(json.loads(data.decode("utf-8"))["label"])


def _decode_npy_fp16(data: bytes) -> Tensor:
    arr = np.load(io.BytesIO(data))
    return torch.from_numpy(arr.copy())


def _read_info(dir_: Path) -> dict:
    info_path = dir_ / "info.json"
    assert info_path.exists(), f"info.json not found at {info_path}"
    with open(info_path) as f:
        return json.load(f)


def _build_pipeline(
    shards: list[str],
    *,
    image_size: int,
    batch_size: int,
    use_worker_split: bool,
) -> wds.WebDataset:
    """Build a WebDataset pipeline that yields (image, label, cls, ptch) batches.

    wds.WebDataset already applies split_by_worker internally (via its default
    workersplitter parameter) at the shard-URL level — one shard per worker.
    We must NOT add split_by_worker again via .compose(), as that would apply
    it a second time at the decoded-sample level, keeping only every Nth sample
    per worker and reducing throughput by num_workers. See claude_docs/webdataset.md.

    nodesplitter=None: we pre-slice shards per rank before constructing the
    dataset, so no node-level splitting inside WebDataset is needed (the default
    single_node_only would raise ValueError under DDP).
    """
    # shardshuffle=False — the schedule already provides global shuffle.
    # empty_check=False — single-shard val/init pipelines should not warn.
    workersplitter = wds.split_by_worker if use_worker_split else None
    ds = wds.WebDataset(
        shards,
        shardshuffle=False,
        empty_check=False,
        nodesplitter=None,
        workersplitter=workersplitter,
    )
    return (
        ds.to_tuple("jpg", "json", "cls.npy", "ptch.npy")
        .map_tuple(
            lambda d: _decode_jpg(d, image_size),
            _decode_label,
            _decode_npy_fp16,
            _decode_npy_fp16,
        )
        .batched(batch_size, partial=False)
    )


class WebDatasetTrainLoader:
    """Streams training samples from one rank's slice of the shard schedule.

    Yields `(images, raw_patches, raw_cls, labels)` per call to `next()`,
    matching the existing `ShardedFeatureLoader.next()` contract.
    """

    def __init__(
        self,
        *,
        train_dir: Path,
        seed: int,
        job_index: int,
        batch_size_per_gpu: int,
        steps_per_job: int,
        image_size: int,
        world_size: int,
        rank: int,
    ) -> None:
        info = _read_info(train_dir)
        assert "cls.npy" in info["keys"], (
            "WebDatasetTrainLoader currently requires precomputed features "
            f"(info.json keys = {info['keys']})."
        )
        self.samples_per_shard: int = int(info["images_per_shard"])
        assert self.samples_per_shard % batch_size_per_gpu == 0, (
            f"samples_per_shard ({self.samples_per_shard}) must be divisible by "
            f"batch_size_per_gpu ({batch_size_per_gpu}) so each worker yields a "
            f"clean number of batches."
        )

        self.shards_per_gpu = compute_shards_per_gpu(
            steps_per_job, batch_size_per_gpu, self.samples_per_shard
        )
        self.shard_files: list[Path] = compute_schedule_slice(
            seed=seed,
            train_dir=train_dir,
            job_index=job_index,
            shards_per_gpu=self.shards_per_gpu,
            world_size=world_size,
            rank=rank,
        )
        self.train_dir = train_dir
        self.batch_size = batch_size_per_gpu
        self.image_size = image_size
        self.num_workers = self.shards_per_gpu

        total_samples = len(self.shard_files) * self.samples_per_shard
        assert total_samples == steps_per_job * batch_size_per_gpu, (
            f"Sample count mismatch: {len(self.shard_files)} shards × {self.samples_per_shard} "
            f"= {total_samples} samples, but steps_per_job × batch_size = "
            f"{steps_per_job} × {batch_size_per_gpu} = {steps_per_job * batch_size_per_gpu}"
        )

        log.info(
            f"WebDatasetTrainLoader: rank={rank}/{world_size}, job_index={job_index}, "
            f"shards_per_gpu={self.shards_per_gpu}, num_workers={self.num_workers}, "
            f"samples_per_shard={self.samples_per_shard}, batch_size={batch_size_per_gpu}, "
            f"steps_per_job={steps_per_job}, total_samples={total_samples} ✓"
        )

        self._iter: Iterator | None = None
        self._loader: DataLoader | None = None

    def first_shard_path(self) -> Path:
        return self.shard_files[0]

    def _ensure_iter(self) -> None:
        if self._iter is not None:
            return
        ds = _build_pipeline(
            [str(p) for p in self.shard_files],
            image_size=self.image_size,
            batch_size=self.batch_size,
            use_worker_split=True,
        )
        self._loader = DataLoader(
            ds,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
        self._iter = iter(self._loader)

    def next(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns (images, raw_patches, raw_cls, labels)."""
        self._ensure_iter()
        assert self._iter is not None
        images, labels, raw_cls, raw_patches = next(self._iter)
        # webdataset's .batched() returns labels as a list[int] — convert.
        if not isinstance(labels, Tensor):
            labels = torch.as_tensor(labels, dtype=torch.long)
        return images, raw_patches, raw_cls, labels


class WebDatasetValLoader:
    """Streams validation samples from one rank's slice of the val shards.

    Cycles forever (matches `InfiniteLoader` semantics). Yields one batch per
    `next_batch_with_labels()` call. Each rank reads a non-overlapping slice
    of val shards (round-robin partition).
    """

    def __init__(
        self,
        *,
        val_dir: Path,
        batch_size_per_gpu: int,
        image_size: int,
        world_size: int,
        rank: int,
    ) -> None:
        info = _read_info(val_dir)
        all_shards = sorted(val_dir.glob("shard-*.tar"))
        assert all_shards, f"No shards in {val_dir}"
        # Round-robin partition across ranks. Last shard is partial — included
        # because validation must touch every sample.
        rank_shards = [s for i, s in enumerate(all_shards) if i % world_size == rank]
        assert rank_shards, (
            f"Rank {rank} got no val shards (world_size={world_size}, "
            f"n_shards={len(all_shards)}). Need world_size <= n_val_shards."
        )

        self.shard_files = rank_shards
        self.batch_size = batch_size_per_gpu
        self.image_size = image_size
        self.num_workers = min(len(rank_shards), 4)

        log.info(
            f"WebDatasetValLoader: rank={rank}/{world_size}, "
            f"shards={len(rank_shards)}/{len(all_shards)} (n_val_images={info.get('n_images', '?')}), "
            f"num_workers={self.num_workers}"
        )

        self._iter: Iterator | None = None
        self._loader: DataLoader | None = None

    def _build_loader(self) -> DataLoader:
        ds = _build_pipeline(
            [str(p) for p in self.shard_files],
            image_size=self.image_size,
            batch_size=self.batch_size,
            use_worker_split=True,
        )
        return DataLoader(
            ds,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def _next_with_cycle(self) -> tuple[Tensor, ...]:
        if self._iter is None:
            self._loader = self._build_loader()
            self._iter = iter(self._loader)
        try:
            return next(self._iter)
        except StopIteration:
            self._loader = self._build_loader()
            self._iter = iter(self._loader)
            return next(self._iter)

    def next_batch_with_labels(self) -> tuple[Tensor, Tensor]:
        images, labels, _raw_cls, _raw_patches = self._next_with_cycle()
        if not isinstance(labels, Tensor):
            labels = torch.as_tensor(labels, dtype=torch.long)
        return images, labels


def init_normalizer_stats_from_tar(
    shard_path: Path,
    scene_norm: PatchStandardizer,
    cls_norm: CLSStandardizer,
    device: torch.device,
    max_samples: int,
) -> None:
    """Initialise standardizer stats from a single WebDataset tar shard.

    Streams `cls.npy` and `ptch.npy` entries directly from the tar via the
    stdlib `tarfile` module, accumulates up to `max_samples` samples, then
    calls `set_stats` exactly like `init_normalizer_stats_from_shard`.
    """
    log.info(f"Computing normalizer stats from tar: {shard_path.name}")

    cls_buf: list[np.ndarray] = []
    ptch_buf: list[np.ndarray] = []
    keys_seen: dict[str, dict[str, np.ndarray]] = {}

    with tarfile.open(shard_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            # entries are <key>.cls.npy and <key>.ptch.npy
            if name.endswith(".cls.npy"):
                key, kind = name[: -len(".cls.npy")], "cls"
            elif name.endswith(".ptch.npy"):
                key, kind = name[: -len(".ptch.npy")], "ptch"
            else:
                continue
            f = tf.extractfile(member)
            assert f is not None
            arr = np.load(io.BytesIO(f.read()))
            keys_seen.setdefault(key, {})[kind] = arr

            entry = keys_seen[key]
            if "cls" in entry and "ptch" in entry:
                cls_buf.append(entry["cls"])
                ptch_buf.append(entry["ptch"])
                del keys_seen[key]
                if max_samples > 0 and len(cls_buf) >= max_samples:
                    break

    n = len(cls_buf)
    assert n > 0, f"No cls/ptch pairs found in {shard_path}"
    log.info(f"  Collected {n} samples from {shard_path.name}")

    cls = torch.from_numpy(np.stack(cls_buf)).float().to(device)  # [n, D]
    patches = torch.from_numpy(np.stack(ptch_buf)).float().to(device)  # [n, T, D]

    scene_norm.set_stats(patches)
    cls_norm.set_stats(cls.unsqueeze(1))
    log.info(f"  Scene/CLS stats from {n} samples")
    del cls, patches
    torch.cuda.empty_cache()
