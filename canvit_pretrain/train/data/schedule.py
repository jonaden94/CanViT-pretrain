"""Shard schedule helpers for WebDataset training under SLURM job arrays.

The schedule is a flat list of shard paths produced by tiling per-epoch
permutations of the training shards (excluding the last partial shard).
It is built once by `scripts/build_shard_schedule.py` and saved alongside
the dataset. Each job reads its slice from the schedule via job_index.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SCHEDULE_FILENAME = "shard_schedule.npz"


def compute_shards_per_gpu(
    steps_per_job: int, batch_size_per_gpu: int, samples_per_shard: int
) -> int:
    """Number of shards consumed per GPU per job. Must divide evenly.

    Constraint:  steps_per_job * batch_size_per_gpu % samples_per_shard == 0
    """
    samples_per_gpu = steps_per_job * batch_size_per_gpu
    assert samples_per_gpu % samples_per_shard == 0, (
        f"Bad divisibility: steps_per_job ({steps_per_job}) * batch_size_per_gpu "
        f"({batch_size_per_gpu}) = {samples_per_gpu} samples per GPU, but "
        f"samples_per_shard = {samples_per_shard}. "
        f"Pick steps_per_job and batch_size such that their product is a "
        f"multiple of {samples_per_shard}."
    )
    return samples_per_gpu // samples_per_shard


def load_schedule(path: Path) -> tuple[list[str], dict[str, str]]:
    """Load the shard schedule. Returns (shard_paths, metadata).

    The schedule is stored as a numpy .npz with two arrays:
      shards     — np.array of shard paths (str), length = n_epochs * n_shards
      metadata   — single-row np.array of bytes describing seed, dataset hash,
                   n_shards, n_epochs (for audit only, not enforcement).
    """
    assert path.exists(), f"Schedule not found at {path}. Build it with scripts/build_shard_schedule.py"
    data = np.load(path, allow_pickle=False)
    shards = data["shards"].tolist()
    meta_keys = data["meta_keys"].tolist()
    meta_vals = data["meta_vals"].tolist()
    metadata = {k: v for k, v in zip(meta_keys, meta_vals)}
    log.info(f"Loaded schedule: {len(shards):,} entries from {path}")
    log.info(f"  metadata: {metadata}")
    return shards, metadata


def slice_for_job(
    schedule: list[str],
    *,
    job_index: int,
    shards_per_gpu: int,
    world_size: int,
    rank: int,
) -> list[Path]:
    """Slice the schedule for one (job, rank) tuple.

    Layout: each job consumes `shards_per_job = shards_per_gpu * world_size`
    contiguous shards from the schedule. Within a job, ranks split that block
    evenly: rank r gets the r-th `shards_per_gpu`-sized chunk.

    This deterministic layout makes it possible to use `wds.split_by_node`
    inside WebDataset itself if we choose to (it would split a flat list across
    ranks); we instead pre-slice per rank here so the loader sees only its own
    shards. Either way the result is the same partition.
    """
    shards_per_job = shards_per_gpu * world_size
    start = job_index * shards_per_job
    end = start + shards_per_job
    assert end <= len(schedule), (
        f"Schedule exhausted: job_index={job_index} requires shards "
        f"[{start}:{end}] but schedule has {len(schedule)}. "
        f"Rebuild with more epochs in scripts/build_shard_schedule.py."
    )
    job_block = schedule[start:end]
    rank_start = rank * shards_per_gpu
    rank_end = rank_start + shards_per_gpu
    rank_shards = job_block[rank_start:rank_end]
    log.info(
        f"Job slice: job_index={job_index}, rank={rank}/{world_size}, "
        f"shards_per_gpu={shards_per_gpu}, schedule[{start + rank_start}:{start + rank_end}]"
    )
    return [Path(s) for s in rank_shards]
