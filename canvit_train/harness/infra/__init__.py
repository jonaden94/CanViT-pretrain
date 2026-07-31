"""Plumbing: process-level and I/O concerns that carry no training logic.

- ``checkpoint.py`` — save/load run state
- ``tracker.py``    — wandb / metric mirroring
- ``ddp.py``        — distributed wrap-up and gradient sync
- ``dist.py``       — SLURM rendezvous and process-group setup
- ``schedule.py``   — resumable shard schedule (shared by distill and in1k loaders)
- ``utils.py``      — device pick, shape assertions
"""
