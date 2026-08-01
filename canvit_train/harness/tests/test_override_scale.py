"""`override_scale` pins eval glimpse SCALES while leaving CENTERS untouched.

The knob exists so a fixed-scale foveated backbone can be deployed under a policy whose
scales it never saw (C2F's quadtree is {1.0, 0.5, 0.25}; `fix_size = scale * H` makes
every such glimpse OOD). Two properties have to hold or the measurement is silently
wrong: centers must survive, and the default must be a bit-exact no-op — `in1k`'s
foveated `auto` deliberately routes to UNPINNED C2F for exp25/exp29 comparability.
"""

import torch

from canvit_train.harness.config import FoveatedScaleConfig
from canvit_train.harness.rollout.eval_viewpoints import OPEN_LOOP, open_loop_viewpoints

_B, _N = 3, 7


def _vps(policy, override, *, is_foveated=False, seed=0):
    torch.manual_seed(seed)  # `random` draws; identical seeds make the two runs comparable
    return open_loop_viewpoints(
        policy, batch_size=_B, device=torch.device("cpu"), n=_N, is_foveated=is_foveated,
        foveated_scale=FoveatedScaleConfig(mode="fixed", fixed_scale=2.0),
        foveated_eval_scale=2.0, override_scale=override,
    )


def test_default_is_a_bit_exact_no_op():
    """None must change nothing at all — this is what protects the historical defaults."""
    for policy in OPEN_LOOP:
        fov = policy == "fixation_grid"
        a, b = _vps(policy, None, is_foveated=fov), _vps(policy, None, is_foveated=fov)
        for x, y in zip(a, b, strict=True):
            assert torch.equal(x.centers, y.centers), policy
            assert torch.equal(x.scales, y.scales), policy


def test_override_pins_scales_and_keeps_centers():
    for policy in OPEN_LOOP:
        fov = policy == "fixation_grid"
        plain = _vps(policy, None, is_foveated=fov)
        pinned = _vps(policy, 2.0, is_foveated=fov)
        assert len(plain) == len(pinned) == _N, policy
        for p, q in zip(plain, pinned, strict=True):
            assert torch.equal(p.centers, q.centers), f"{policy}: centers must not move"
            assert torch.allclose(q.scales, torch.full_like(q.scales, 2.0)), policy
            assert q.scales.shape == p.scales.shape, policy


def test_c2f_is_multi_scale_until_pinned():
    """Guards the premise: unpinned C2F really does hand a foveated model several scales,
    so pinning is not a no-op dressed up as a fix."""
    plain = _vps("coarse_to_fine", None)
    distinct = {round(float(s), 4) for v in plain for s in v.scales.flatten()}
    assert len(distinct) > 1, f"expected a multi-scale quadtree, got {distinct}"
    pinned = {round(float(s), 4) for v in _vps("coarse_to_fine", 2.0) for s in v.scales.flatten()}
    assert pinned == {2.0}
