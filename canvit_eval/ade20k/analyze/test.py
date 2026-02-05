"""Tests for analyze module."""

import numpy as np

from canvit_eval.ade20k.analyze import PolicyStats


def test_policy_stats_shapes() -> None:
    stats = PolicyStats(
        policy="c2f",
        n_runs=5,
        n_timesteps=10,
        mious=np.zeros(10),
        std=np.zeros(10),
        per_run=np.zeros((5, 10)),
    )
    assert stats.mious.shape == (10,)
    assert stats.std.shape == (10,)
    assert stats.per_run.shape == (5, 10)
