"""Tests for analyze module."""

import numpy as np

from canvit_eval.ade20k.analyze import bootstrap_ci


def test_bootstrap_ci_shape() -> None:
    per_image = np.random.rand(100, 10)
    ci_low, ci_high = bootstrap_ci(per_image, n_bootstrap=100)
    assert ci_low.shape == (10,)
    assert ci_high.shape == (10,)
    assert np.all(ci_low <= ci_high)


def test_bootstrap_ci_deterministic() -> None:
    per_image = np.random.rand(50, 5)
    ci1 = bootstrap_ci(per_image, seed=42)
    ci2 = bootstrap_ci(per_image, seed=42)
    np.testing.assert_array_equal(ci1[0], ci2[0])
    np.testing.assert_array_equal(ci1[1], ci2[1])
