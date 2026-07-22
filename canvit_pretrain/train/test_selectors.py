"""P4a tests: PolicySelector / MixtureSelector through the P1 seam (fake net/encoder
— the real ones are covered in core's policy tests and ade20k/test_rl_train.py)."""

import torch

from .config import FoveatedScaleConfig
from .selector import MixtureSelector, PolicySelector, RandomSelector
from .viewpoint import ViewpointType

_B, _A = 4, 8
_CPU = torch.device("cpu")


class _FakeEncoder:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __call__(self, state):
        return torch.zeros(_B, 3, 32, 32)


def _make(mode: str, scores: torch.Tensor) -> PolicySelector:
    vp_flat = torch.stack(
        [torch.linspace(-0.5, 0.5, _A), torch.linspace(0.5, -0.5, _A), torch.full((_A,), 0.5)], dim=-1
    )
    rnd = RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.1)
    return PolicySelector(
        net=lambda f: scores, encoder=_FakeEncoder(), vp_flat=vp_flat, fallback=rnd, mode=mode
    )


def test_policy_selector_argmax_and_aux() -> None:
    scores = torch.zeros(_B, _A)
    scores[:, 3] = 5.0  # candidate 3 wins everywhere
    sel = _make("argmax", scores)
    ctx = sel.start_rollout(t0_type=ViewpointType.FULL, batch_size=_B, device=_CPU)
    assert sel.encoder.resets == 1  # type: ignore[attr-defined]
    vp = sel.select(vp_type=ViewpointType.RANDOM, ctx=ctx, t=1, batch_size=_B, device=_CPU, state=None)
    assert vp.name == "policy"
    assert torch.allclose(vp.centers, sel.vp_flat[3, :2].expand(_B, 2))
    assert sel.last_aux is not None and (sel.last_aux["flat_idx"] == 3).all()
    # FULL delegates to the random fallback (t0 anchor)
    full = sel.select(vp_type=ViewpointType.FULL, ctx=ctx, t=0, batch_size=_B, device=_CPU, state=None)
    assert full.name == "full" and (full.scales == 1.0).all()


def test_policy_selector_sample_mode() -> None:
    scores = torch.full((_B, _A), -100.0)
    scores[:, 5] = 100.0  # ~all probability mass on candidate 5
    sel = _make("sample", scores)
    ctx = sel.start_rollout(t0_type=ViewpointType.RANDOM, batch_size=_B, device=_CPU)
    sel.select(vp_type=ViewpointType.RANDOM, ctx=ctx, t=0, batch_size=_B, device=_CPU, state=None)
    assert sel.last_aux is not None and (sel.last_aux["flat_idx"] == 5).all()


def test_mixture_extremes_and_blend() -> None:
    scores = torch.zeros(_B, _A)
    scores[:, 0] = 5.0
    pol = _make("argmax", scores)
    rnd = pol.fallback
    mix = MixtureSelector(random_sel=rnd, policy_sel=pol, p_policy=0.0)
    ctx = mix.start_rollout(t0_type=ViewpointType.RANDOM, batch_size=_B, device=_CPU)

    torch.manual_seed(0)
    vp = mix.select(vp_type=ViewpointType.RANDOM, ctx=ctx, t=0, batch_size=_B, device=_CPU, state=None)
    assert vp.name == "random" and not mix.last_mask.any()  # p=0 -> today's behavior

    mix.p_policy = 1.0
    vp = mix.select(vp_type=ViewpointType.RANDOM, ctx=ctx, t=1, batch_size=_B, device=_CPU, state=None)
    assert vp.name == "policy" and mix.last_mask.all()

    mix.p_policy = 0.5
    torch.manual_seed(1)
    vp = mix.select(vp_type=ViewpointType.RANDOM, ctx=ctx, t=2, batch_size=_B, device=_CPU, state=None)
    assert vp.name == "mixture"
    # policy rows carry candidate 0's center; masked rows must match exactly
    assert torch.allclose(vp.centers[mix.last_mask], pol.vp_flat[0, :2].expand(int(mix.last_mask.sum()), 2))
