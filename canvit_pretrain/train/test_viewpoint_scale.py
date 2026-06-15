"""Tests for foveated/square per-glimpse view-scale sampling."""

import torch

from .viewpoint import random_foveated_viewpoint, sample_view_scales

_CPU = torch.device("cpu")


def test_uniform_scales_range_includes_zoom_out():
    s = sample_view_scales(5000, _CPU, distribution="uniform", min_scale=0.5, max_scale=1.5)
    assert s.shape == (5000,)
    assert s.min() >= 0.5 - 1e-6 and s.max() <= 1.5 + 1e-6
    assert (s > 1.0).any()  # zoom-out is reachable via the uniform distribution


def test_safebox_scales_bounded_and_favor_zoom_in():
    s = sample_view_scales(50000, _CPU, distribution="safebox", min_scale=0.1, max_scale=1.0)
    # intrinsically (0, 1]; p(s) ∝ (1-s) so the mean sits well below the midpoint.
    assert s.min() >= 0.1 - 1e-6 and s.max() <= 1.0 + 1e-6
    assert float(s.mean()) < 0.5


def test_center_full_field_vs_safebox():
    scales = torch.full((2000,), 0.3)
    ff = random_foveated_viewpoint(2000, _CPU, scales=scales, center_mode="full_field")
    assert ff.centers.abs().max() <= 1.0
    assert ff.centers.abs().max() > 0.7  # genuinely uses the full field
    sb = random_foveated_viewpoint(2000, _CPU, scales=scales, center_mode="safebox")
    # safe box for s=0.3 is |center| <= 1 - s = 0.7 (crop fits, no overshoot).
    assert sb.centers.abs().max() <= 0.7 + 1e-6
    assert torch.equal(sb.scales, scales.float())


def test_safebox_center_scales_with_per_image_scale():
    # Larger scale -> tighter safe box (center range shrinks toward 0).
    scales = torch.cat([torch.full((1000,), 0.2), torch.full((1000,), 0.9)])
    vp = random_foveated_viewpoint(2000, _CPU, scales=scales, center_mode="safebox")
    big_box = vp.centers[:1000].abs().max()    # s=0.2 -> box 0.8
    small_box = vp.centers[1000:].abs().max()  # s=0.9 -> box 0.1
    assert big_box <= 0.8 + 1e-6 and small_box <= 0.1 + 1e-6
    assert big_box > small_box
