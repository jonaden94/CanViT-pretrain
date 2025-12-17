"""Data loading with multi-resolution support."""

import logging
from typing import Any

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from avp_vit.train import InfiniteLoader, train_transform, val_transform
from avp_vit.train.curriculum import CurriculumStage, create_curriculum_stage

from .config import Config

log = logging.getLogger(__name__)


class SingleImageDataset(Dataset[tuple[Tensor, int]]):
    """Wraps a dataset, always returns item at index 0."""

    def __init__(self, dataset: Dataset[Any]) -> None:
        self._item: tuple[Tensor, int] = dataset[0]
        log.info(f"DEBUG: Cached single image from dataset (index 0, label={self._item[1]})")

    def __len__(self) -> int:
        return 1_000_000  # Large enough for any training run

    def __getitem__(self, _idx: int) -> tuple[Tensor, int]:
        return self._item


def create_curriculum_stages(cfg: Config, patch_size: int) -> dict[int, CurriculumStage]:
    """Create curriculum stages for each grid size."""
    stages: dict[int, CurriculumStage] = {}
    for G in cfg.grid_sizes:
        stages[G] = create_curriculum_stage(
            scene_grid_size=G,
            glimpse_grid_size=cfg.avp.glimpse_grid_size,
            patch_size=patch_size,
            max_grid_size=cfg.max_grid_size,
            max_batch_size=cfg.batch_size,
            n_viewpoints_per_step=cfg.n_viewpoints_per_step,
        )
    return stages


def create_loaders_for_curriculum(
    cfg: Config, stages: dict[int, CurriculumStage]
) -> tuple[dict[int, InfiniteLoader], dict[int, InfiniteLoader]]:
    """Create train/val loaders for each resolution stage.

    Returns: (train_loaders, val_loaders) dicts keyed by grid size.
    """
    if cfg.debug_train_on_single_image:
        log.warning("=" * 60)
        log.warning("DEBUG MODE: Training on single repeated image (index 0)")
        log.warning("=" * 60)

    train_loaders: dict[int, InfiniteLoader] = {}
    val_loaders: dict[int, InfiniteLoader] = {}

    for G, stage in stages.items():
        scene_size_px = stage.scene_size_px
        fresh_count = stage.fresh_count

        log.info(
            f"Creating loaders for G={G}: scene_size={scene_size_px}px, "
            f"batch={stage.batch_size}, fresh={fresh_count}"
        )

        train_dataset: Dataset[Any] = ImageFolder(
            str(cfg.train_dir), train_transform(scene_size_px, (cfg.crop_scale_min, 1.0))
        )
        val_dataset: Dataset[Any] = ImageFolder(
            str(cfg.val_dir), val_transform(scene_size_px)
        )

        if cfg.debug_train_on_single_image:
            train_dataset = SingleImageDataset(train_dataset)
            val_dataset = SingleImageDataset(val_dataset)

        train_loader: DataLoader[Any] = DataLoader(
            train_dataset,
            batch_size=fresh_count,
            shuffle=not cfg.debug_train_on_single_image,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader: DataLoader[Any] = DataLoader(
            val_dataset,
            batch_size=fresh_count,
            shuffle=not cfg.debug_train_on_single_image,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        train_loaders[G] = InfiniteLoader(train_loader)
        val_loaders[G] = InfiniteLoader(val_loader)

    return train_loaders, val_loaders
