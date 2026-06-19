"""WebDataset-based training loader.

Used for training only; validation always reads the raw ImageNet-1k val
ImageFolder (see `create_loaders` / `IndexedImageFolder`), independent of the
training data source.

Each rank receives a deterministically computed list of shard paths (via
`schedule.compute_schedule_slice`). DataLoader workers then split those shards
via `wds.split_by_worker` — each worker streams one or more shards sequentially.
The actual `num_workers` is derived from `cfg.num_workers`, capped at
`shards_per_gpu`, and rounded down to a divisor of `shards_per_gpu` so every
worker gets the same number of shards.

Loader interface matches the existing `ShardedFeatureLoader` so `loop.py` can
call `train_loader.next()` without modification.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
from collections.abc import Callable, Iterator
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
    has_features: bool = True,
) -> wds.WebDataset:
    """Build a WebDataset pipeline.

    With ``has_features=True`` (precomputed-feature shards) it yields
    ``(image, label, cls, ptch)`` batches. With ``has_features=False`` (raw
    shards carrying only ``jpg``/``json``) it yields ``(image, label)`` batches —
    teacher features are then computed on the fly in the training loop.

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
    if has_features:
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
    return (
        ds.to_tuple("jpg", "json")
        .map_tuple(
            lambda d: _decode_jpg(d, image_size),
            _decode_label,
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
        num_workers: int,
    ) -> None:
        info = _read_info(train_dir)
        # Precomputed-feature shards carry cls.npy/ptch.npy. "Raw" shards have
        # only jpg+json; in that case teacher features are computed on the fly
        # in the training loop (load_train_batch -> compute_raw_targets).
        self.has_features: bool = "cls.npy" in info["keys"]
        log.info(
            "WebDatasetTrainLoader: %s (info.json keys = %s)",
            "PRECOMPUTED features" if self.has_features
            else "RAW images (teacher features computed on the fly)",
            info["keys"],
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

        # Resolve num_workers: cap at shards_per_gpu (extra workers would get
        # zero shards), then round down to a divisor of shards_per_gpu so every
        # worker streams the same number of shards (keeps the per-worker batch
        # count uniform under .batched(partial=False)).
        requested = max(1, num_workers)
        capped = min(requested, self.shards_per_gpu)
        nw = capped
        while self.shards_per_gpu % nw != 0:
            nw -= 1
        self.num_workers = nw
        shards_per_worker = self.shards_per_gpu // self.num_workers
        if requested != self.num_workers:
            log.info(
                f"WebDatasetTrainLoader: requested num_workers={requested}, "
                f"using {self.num_workers} "
                f"({'capped to shards_per_gpu' if requested > self.shards_per_gpu else 'rounded down for divisibility'}); "
                f"each worker streams {shards_per_worker} shard(s)"
            )
        else:
            log.info(
                f"WebDatasetTrainLoader: num_workers={self.num_workers}; "
                f"each worker streams {shards_per_worker} shard(s)"
            )

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
            has_features=self.has_features,
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

    def next(self) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        """Returns (images, raw_patches, raw_cls, labels).

        For raw (no-feature) shards, ``raw_patches`` and ``raw_cls`` are None —
        the training loop computes teacher features on the fly from ``images``.
        """
        self._ensure_iter()
        assert self._iter is not None
        raw_cls: Tensor | None
        raw_patches: Tensor | None
        if self.has_features:
            images, labels, raw_cls, raw_patches = next(self._iter)
        else:
            images, labels = next(self._iter)
            raw_cls = raw_patches = None
        # webdataset's .batched() returns labels as a list[int] — convert.
        if not isinstance(labels, Tensor):
            labels = torch.as_tensor(labels, dtype=torch.long)
        return images, raw_patches, raw_cls, labels


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


def init_normalizer_stats_from_tar_raw(
    shard_path: Path,
    scene_norm: PatchStandardizer,
    cls_norm: CLSStandardizer,
    *,
    image_size: int,
    compute_features: Callable[[Tensor], object],
    device: torch.device,
    max_samples: int,
    sub_batch: int = 64,
) -> None:
    """Initialise standardizer stats from a RAW (no-feature) WebDataset shard.

    Streams ``jpg`` images from the tar, decodes them to ``image_size``, and
    computes teacher features on the fly via ``compute_features`` (which returns
    an object exposing ``.patches`` [B, T, D] and ``.cls`` [B, D]). Teacher
    forwards run in sub-batches of ``sub_batch``; features are accumulated on the
    CPU to bound GPU memory, then ``set_stats`` is called exactly like the
    precomputed-feature path.

    Only reached for a fresh (step-0) run on raw shards — resumed runs load
    standardizer stats from the checkpoint and skip init entirely.
    """
    # max_samples<=0 means "use the whole shard"; computing teacher features for
    # a full 4096-image shard is many forwards — cap to keep init quick/bounded.
    cap = max_samples if max_samples > 0 else 2048
    log.info(
        f"Computing normalizer stats from RAW tar (teacher on the fly): "
        f"{shard_path.name}, up to {cap} samples"
    )

    imgs: list[Tensor] = []
    with tarfile.open(shard_path, "r") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".jpg"):
                continue
            f = tf.extractfile(member)
            assert f is not None
            imgs.append(_decode_jpg(f.read(), image_size))
            if len(imgs) >= cap:
                break

    n = len(imgs)
    assert n > 0, f"No jpg images found in {shard_path}"

    cls_buf: list[Tensor] = []
    ptch_buf: list[Tensor] = []
    for i in range(0, n, sub_batch):
        batch = torch.stack(imgs[i : i + sub_batch]).to(device, non_blocking=True)
        feats = compute_features(batch)
        cls_buf.append(feats.cls.detach().float().cpu())       # type: ignore[attr-defined]
        ptch_buf.append(feats.patches.detach().float().cpu())  # type: ignore[attr-defined]

    cls = torch.cat(cls_buf).to(device)       # [n, D]
    patches = torch.cat(ptch_buf).to(device)  # [n, T, D]
    scene_norm.set_stats(patches)
    cls_norm.set_stats(cls.unsqueeze(1))
    log.info(f"  Scene/CLS stats from {n} samples (computed live)")
    del cls, patches
    torch.cuda.empty_cache()
