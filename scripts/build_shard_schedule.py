"""Build the shard schedule for WebDataset training.

Generates a flat list of `n_epochs` permutations of the training shards
(excluding the partial last shard) and saves it next to the dataset as
`shard_schedule.npz`. Each job slices its own block from this list at
training time.

Usage:
    uv run python scripts/build_shard_schedule.py \
        --webdataset-dir $WEBDATASET_DIR \
        --seed 0
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from canvit_pretrain.train.data.schedule import SCHEDULE_FILENAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class Config:
    webdataset_dir: Path
    """Parent directory containing train-shuffled/ and val/."""
    seed: int = 0
    """Seed for permutations. Determines the schedule order across all jobs."""
    n_epochs: int = 1000
    """Number of full-dataset permutations to tile. 1000 covers any practical run."""
    force: bool = False
    """Overwrite an existing schedule file if present."""


def main(cfg: Config) -> None:
    train_dir = cfg.webdataset_dir / "train-shuffled"
    info_path = train_dir / "info.json"
    assert info_path.exists(), f"info.json not found at {info_path}"

    with open(info_path) as f:
        info = json.load(f)
    log.info(f"Dataset: {info.get('n_images', '?')} images, {info.get('n_shards', '?')} shards")

    all_shards = sorted(train_dir.glob("shard-*.tar"))
    assert all_shards, f"No shards in {train_dir}"
    n_shards_total = len(all_shards)
    expected = int(info.get("n_shards", n_shards_total))
    assert n_shards_total == expected, (
        f"Shard count mismatch: info.json says {expected}, found {n_shards_total} on disk"
    )

    # Drop the last (partial) shard. Spec: training schedule must keep clean
    # shard boundaries; the partial shard breaks the arithmetic.
    train_shards = all_shards[:-1]
    log.info(f"Excluding last shard: {all_shards[-1].name}")
    log.info(f"Training shards: {len(train_shards)} (= {expected} - 1)")

    # Store paths relative to the train dir so the schedule travels if the
    # dataset is moved.
    rel_paths = [s.relative_to(train_dir).as_posix() for s in train_shards]

    rng = np.random.default_rng(seed=cfg.seed)
    rel_arr = np.asarray(rel_paths)
    epochs = [rng.permutation(rel_arr).tolist() for _ in range(cfg.n_epochs)]
    schedule = [s for ep in epochs for s in ep]
    log.info(f"Built schedule: {cfg.n_epochs} epochs × {len(rel_paths)} shards = {len(schedule):,} entries")

    # Resolve to absolute paths (loader joins with train_dir if relative).
    abs_schedule = [str(train_dir / s) for s in schedule]

    schedule_path = train_dir / SCHEDULE_FILENAME
    if schedule_path.exists() and not cfg.force:
        log.error(
            f"Schedule already exists at {schedule_path}. "
            f"Pass --force to overwrite (training will desynchronise from "
            f"any in-flight runs that depend on the existing schedule)."
        )
        return

    metadata_keys = ["seed", "n_epochs", "n_shards_train", "info_n_images", "train_dir"]
    metadata_vals = [
        str(cfg.seed),
        str(cfg.n_epochs),
        str(len(rel_paths)),
        str(info.get("n_images", "?")),
        str(train_dir),
    ]
    np.savez(
        schedule_path,
        shards=np.asarray(abs_schedule),
        meta_keys=np.asarray(metadata_keys),
        meta_vals=np.asarray(metadata_vals),
    )
    size_mb = schedule_path.stat().st_size / (1024 * 1024)
    log.info(f"Wrote {schedule_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main(tyro.cli(Config))
