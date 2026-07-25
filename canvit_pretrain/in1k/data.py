"""IN1k data pipeline (unification P5).

TRAIN: WebDataset shards of ``jpg`` + ``json`` (``{"label": int}``) — the
CanViT-pretrain repo's IN1k-no-features set (images pre-resized to scene_size).
Decoded with train augmentation (RandomResizedCrop + flip). Uses the SAME resumable,
shard-aligned schedule as distill pretraining (``canvit_pretrain.train.data.schedule``):
a seeded global shard permutation, each SLURM-array job consuming a contiguous block so
the next job resumes at the next shard slice. A seeded within-stream shuffle buffer adds
cross-shard mixing (the shards are already globally pre-shuffled at creation). This
replaced the earlier ``resampled=True`` stream, which could not resume across array jobs.

VAL: the IN1k validation ImageFolder with canvit_eval's canonical preprocessing
(Resize short side + CenterCrop, aspect-preserving), since the no-features set
ships no val shards. Point cfg.val_dir at it (IN1K_VAL_DIR / the eval IMAGENET_VAL).
"""

import io
import json
import logging
from pathlib import Path

import webdataset as wds
from canvit_pytorch.preprocess import preprocess
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from ..train.data.schedule import compute_schedule_slice, compute_shards_per_gpu
from .config import In1kConfig

log = logging.getLogger(__name__)


def make_train_transform(scene_size: int, *, min_scale: float, flip_prob: float) -> T.Compose:
    """RandomResizedCrop + flip — the canonical IN1k classifier train aug. The
    shards are already scene_size²; RandomResizedCrop still gives scale/translation
    jitter (scale in [min_scale, 1.0]) then resizes back to scene_size."""
    return T.Compose([
        T.RandomResizedCrop(scene_size, scale=(min_scale, 1.0), antialias=True),
        T.RandomHorizontalFlip(flip_prob),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


def _decode_label(data: bytes) -> int:
    return int(json.loads(data.decode("utf-8"))["label"])


def _decode_jpg(data: bytes, transform: T.Compose) -> Tensor:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    out = transform(img)
    assert isinstance(out, Tensor)
    return out


def _read_info(shard_dir: Path) -> dict:
    info = shard_dir / "info.json"
    assert info.exists(), f"info.json not found at {info}"
    with open(info) as f:
        return json.load(f)


def build_train_pipeline(
    shard_files: list[Path], *, transform: T.Compose, batch_size: int,
    num_workers: int, shuffle_buffer: int, shuffle_seed: int,
) -> wds.WebDataset:
    """Deterministic WebDataset over ONE job's per-rank shard slice, yielding
    (images [B,3,H,W], labels[list]). Shards are pre-sliced per rank by the schedule;
    ``split_by_worker`` then hands each DataLoader worker a disjoint subset so no shard is
    read twice. A seeded shuffle buffer adds within-stream mixing (0 = off)."""
    urls = [str(p) for p in shard_files]
    assert urls, "empty shard slice"
    ds = wds.WebDataset(
        urls, shardshuffle=False, empty_check=False, nodesplitter=None,
        workersplitter=wds.split_by_worker if num_workers > 0 else None,
    )
    if shuffle_buffer > 0:
        ds = ds.shuffle(shuffle_buffer, seed=shuffle_seed)
    return (
        ds.to_tuple("jpg", "json")
        .map_tuple(lambda d: _decode_jpg(d, transform), _decode_label)
        .batched(batch_size, partial=False)
    )


def _resolve_num_workers(requested: int, shards_per_gpu: int) -> int:
    """Cap at shards_per_gpu (extra workers would get zero shards), then round down to a
    divisor so every worker streams the same whole number of shards (mirrors distill)."""
    nw = min(max(requested, 0), shards_per_gpu)
    while nw > 1 and shards_per_gpu % nw != 0:
        nw -= 1
    return nw


def make_train_loader(
    cfg: In1kConfig, *, world_size: int, rank: int, job_index: int, steps_per_job: int,
) -> tuple[DataLoader, int]:
    """(loader, steps_per_job). Resumable shard schedule: job ``job_index`` consumes a
    contiguous block of the seeded global shard permutation, so the next job continues at
    the next slice. ``steps_per_job * batch_size`` must be a multiple of images_per_shard
    (enforced by ``compute_shards_per_gpu``) so a job ends shard-aligned. The returned
    loader carries ``samples_per_shard`` for the task's resume_state / invariant checks."""
    info = _read_info(cfg.train_dir)
    samples_per_shard = int(info["images_per_shard"])
    assert samples_per_shard % cfg.batch_size == 0, (
        f"images_per_shard ({samples_per_shard}) must be divisible by batch_size "
        f"({cfg.batch_size}) so batched(partial=False) drops nothing")
    shards_per_gpu = compute_shards_per_gpu(steps_per_job, cfg.batch_size, samples_per_shard)
    shard_files = compute_schedule_slice(
        seed=cfg.seed, train_dir=cfg.train_dir, job_index=job_index,
        shards_per_gpu=shards_per_gpu, world_size=world_size, rank=rank,
    )
    nw = _resolve_num_workers(cfg.num_workers, shards_per_gpu)
    transform = make_train_transform(cfg.scene_size, min_scale=cfg.aug_min_scale, flip_prob=cfg.aug_flip_prob)
    ds = build_train_pipeline(
        shard_files, transform=transform, batch_size=cfg.batch_size, num_workers=nw,
        shuffle_buffer=cfg.shuffle_buffer, shuffle_seed=cfg.seed + job_index,
    )
    loader = DataLoader(
        ds, batch_size=None, num_workers=nw, pin_memory=True,
        prefetch_factor=2 if nw > 0 else None,
    )
    loader.samples_per_shard = samples_per_shard  # for In1kRunTask.resume_state / invariants
    log.info(f"IN1k train: shard schedule job_index={job_index}, rank={rank}/{world_size}, "
             f"shards_per_gpu={shards_per_gpu}, num_workers={nw}, "
             f"samples_per_shard={samples_per_shard}, batch={cfg.batch_size}, steps_per_job={steps_per_job}")
    return loader, steps_per_job


def make_val_loader(cfg: In1kConfig, *, world_size: int, rank: int) -> DataLoader:
    """IN1k val ImageFolder with canonical (aspect-preserving) preprocessing.
    DistributedSampler shards it across ranks (drop nothing; rank 0 aggregates)."""
    assert cfg.val_dir.is_dir(), (
        f"IN1k val dir not found: {cfg.val_dir}. Set IN1K_VAL_DIR (the ImageFolder val, "
        f"e.g. .../ILSVRC/Data/CLS-LOC/val) — the no-features webdataset ships no val split."
    )
    ds = ImageFolder(str(cfg.val_dir), transform=preprocess(cfg.scene_size))
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    return DataLoader(
        ds, batch_size=cfg.eval_batch_size, sampler=sampler, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
