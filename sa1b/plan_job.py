"""Determine which tars to extract for the current SA-1B training job.

Reads checkpoint (if any) to determine start_step, computes shard range,
outputs tar indices (one per line, zero-padded to 6 digits).

Usage:
    uv run python sa1b/plan_job.py RUN_DIR SHARDS_DIR BATCH_SIZE SHARDS_PER_JOB STEPS_PER_JOB
"""

import math
import re
import sys
from pathlib import Path

import torch


def main() -> None:
    run_dir = Path(sys.argv[1])
    shards_dir = Path(sys.argv[2])
    batch_size = int(sys.argv[3])
    shards_per_job = int(sys.argv[4])
    steps_per_job = int(sys.argv[5])

    # Determine start_step from checkpoint
    latest = run_dir / "latest.pt"
    if latest.is_symlink() and latest.resolve().exists():
        ckpt = torch.load(latest, weights_only=False, map_location="cpu", mmap=True)
        sched = ckpt["scheduler_state"]
        assert sched is not None, "Checkpoint has no scheduler_state"
        start_step = sched["last_epoch"]
    else:
        start_step = 0

    # List all shards (sorted = deterministic order matching ShardedFeatureLoader)
    shard_files = sorted(shards_dir.glob("*.pt"))
    assert shard_files, f"No shards in {shards_dir}"

    # Compute batches_per_shard from first shard (must match ShardedFeatureLoader logic)
    first = torch.load(shard_files[0], map_location="cpu", weights_only=False, mmap=True)
    samples_per_shard = len(first["paths"])
    del first
    batches_per_shard = samples_per_shard // batch_size

    # Cap shards to what's actually needed for steps_per_job
    shards_needed = math.ceil(steps_per_job / batches_per_shard)
    n_shards = min(shards_per_job, shards_needed)

    start_shard = start_step // batches_per_shard
    end_shard = min(start_shard + n_shards, len(shard_files))

    # Print diagnostic info to stderr
    print(f"start_step={start_step}, batches_per_shard={batches_per_shard}, "
          f"shards_needed={shards_needed}, n_shards={n_shards}, "
          f"start_shard={start_shard}, end_shard={end_shard}, "
          f"total_shards={len(shard_files)}", file=sys.stderr)

    # Extract tar indices from shard filenames → stdout (one per line)
    for shard_path in shard_files[start_shard:end_shard]:
        m = re.match(r"sa_(\d+)\.pt", shard_path.name)
        assert m, f"Unexpected shard name: {shard_path.name}"
        print(m.group(1))


if __name__ == "__main__":
    main()
