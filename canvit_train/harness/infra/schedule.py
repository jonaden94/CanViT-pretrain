"""Shard schedule helpers for WebDataset training under SLURM job arrays."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


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


def compute_schedule_slice(
    *,
    seed: int,
    train_dir: Path,
    job_index: int,
    shards_per_gpu: int,
    world_size: int,
    rank: int,
) -> list[Path]:
    """Deterministically compute this rank's shard list for one job, no file IO.

    The schedule is an infinite flat sequence built by repeating per-epoch
    permutations of the training shards (last partial shard excluded). Job j
    consumes shards_per_job = shards_per_gpu * world_size contiguous entries.
    Within that block, rank r gets the r-th shards_per_gpu-sized chunk.

    Epochs are generated lazily so the sequence is infinite.
    """
    all_shards = sorted(train_dir.glob("shard-*.tar"))
    assert all_shards, f"No shards in {train_dir}"
    train_shards = all_shards[:-1]  # exclude partial last shard
    n = len(train_shards)
    assert n > 0, f"Need ≥2 shards in {train_dir} (last is excluded as partial)"

    shards_per_job = shards_per_gpu * world_size
    flat_start = job_index * shards_per_job
    flat_end = flat_start + shards_per_job

    rng = np.random.default_rng(seed=seed)
    pos = 0
    job_block: list[Path] = []

    while pos < flat_end:
        epoch_perm = rng.permutation(n)
        epoch_paths = [train_shards[int(i)] for i in epoch_perm]
        epoch_end = pos + n

        if epoch_end > flat_start:
            lo = max(0, flat_start - pos)
            hi = min(n, flat_end - pos)
            job_block.extend(epoch_paths[lo:hi])

        pos = epoch_end

    assert len(job_block) == shards_per_job, (
        f"Expected {shards_per_job} shards in job block, got {len(job_block)}"
    )

    rank_start = rank * shards_per_gpu
    rank_end = rank_start + shards_per_gpu
    rank_shards = job_block[rank_start:rank_end]

    log.info(
        f"Schedule: seed={seed}, job_index={job_index}, rank={rank}/{world_size}, "
        f"shards_per_gpu={shards_per_gpu}, n_train_shards={n}"
    )
    return rank_shards
