"""Data loading with curriculum support."""

import logging

from avp_vit.train import InfiniteLoader, make_loader, train_transform, val_transform
from avp_vit.train.curriculum import CurriculumStage, create_curriculum_stage

from .config import Config

log = logging.getLogger(__name__)


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
    """Create train/val loaders for each curriculum stage.

    Returns: (train_loaders, val_loaders) dicts keyed by grid size.
    """
    train_loaders: dict[int, InfiniteLoader] = {}
    val_loaders: dict[int, InfiniteLoader] = {}

    for G, stage in stages.items():
        scene_size_px = stage.scene_size_px
        fresh_count = stage.fresh_count

        log.info(
            f"Creating loaders for G={G}: scene_size={scene_size_px}px, "
            f"batch={stage.batch_size}, fresh={fresh_count}"
        )

        train_loaders[G] = InfiniteLoader(
            make_loader(
                cfg.train_dir,
                train_transform(scene_size_px, (cfg.crop_scale_min, 1.0)),
                fresh_count,
                cfg.num_workers,
                shuffle=True,
            )
        )
        val_loaders[G] = InfiniteLoader(
            make_loader(
                cfg.val_dir,
                val_transform(scene_size_px),
                fresh_count,
                cfg.num_workers,
                shuffle=True,
            )
        )

    return train_loaders, val_loaders
