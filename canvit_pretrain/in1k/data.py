"""IN1k data pipeline (unification P5).

TRAIN: WebDataset shards of ``jpg`` + ``json`` (``{"label": int}``) — the
CanViT-pretrain repo's IN1k-no-features set (images pre-resized to scene_size).
Decoded with train augmentation (RandomResizedCrop + flip). Epochs are made
DDP-safe with ``resampled=True`` + ``.with_epoch(N)``: every rank independently
samples shards and yields the SAME fixed batch count, so no rank stalls at an
uneven shard boundary (the classic webdataset+DDP hang). This trades exact
once-per-image for statistical coverage — standard for large-scale wds training.

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
    shard_dir: Path, *, transform: T.Compose, batch_size: int, batches_per_epoch: int, seed: int,
) -> wds.WebDataset:
    """Resampled, fixed-length-epoch WebDataset yielding (images [B,3,H,W], labels[list])."""
    shards = sorted(str(p) for p in shard_dir.glob("shard-*.tar"))
    assert shards, f"no shard-*.tar in {shard_dir}"
    ds = (
        # resampled => shards sampled with replacement per rank (shardshuffle unused)
        wds.WebDataset(shards, resampled=True, shardshuffle=False, empty_check=False, seed=seed)
        .shuffle(2000)
        .to_tuple("jpg", "json")
        .map_tuple(lambda d: _decode_jpg(d, transform), _decode_label)
        .batched(batch_size, partial=False)
    )
    return ds.with_epoch(batches_per_epoch)


def make_train_loader(cfg: In1kConfig, *, world_size: int, rank: int) -> tuple[DataLoader, int]:
    """(loader, batches_per_epoch). Each rank samples independently (resampled) and
    yields batches_per_epoch = n_images // (world_size * batch_size) batches."""
    n_images = int(_read_info(cfg.train_dir)["n_images"])
    batches_per_epoch = n_images // (world_size * cfg.batch_size)
    assert batches_per_epoch > 0, f"n_images={n_images} too small for world_size×batch_size"
    transform = make_train_transform(cfg.scene_size, min_scale=cfg.aug_min_scale, flip_prob=cfg.aug_flip_prob)
    ds = build_train_pipeline(
        cfg.train_dir, transform=transform, batch_size=cfg.batch_size,
        batches_per_epoch=batches_per_epoch, seed=cfg.seed + rank,
    )
    loader = DataLoader(
        ds, batch_size=None, num_workers=cfg.num_workers, pin_memory=True,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    log.info(f"IN1k train: {n_images} imgs, {batches_per_epoch} batches/epoch/rank "
             f"(world_size={world_size}, batch={cfg.batch_size})")
    return loader, batches_per_epoch


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
