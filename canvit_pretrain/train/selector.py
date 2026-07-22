"""Viewpoint-selection seam for the unified harness (unification master plan §4.2).

A Selector decides where the next glimpse goes, consulted inside the rollout.
P1 ships RandomSelector only — a byte-for-byte extraction of training_step's
historical closures (``_foveated_random_vp`` / ``make_named_vp`` and the
per-rollout scale draw): same RNG calls in the same order, so the parity probe
digest is unchanged. PolicySelector / MixtureSelector (ε-curriculum) arrive in
P3/P4 — the ``state`` and ``t`` arguments exist for them and are unused here.
"""

from dataclasses import dataclass
from typing import Protocol

import torch
from canvit_pytorch import RecurrentState
from torch import Tensor

from .config import FoveatedScaleConfig
from .viewpoint import Viewpoint as NamedViewpoint
from .viewpoint import ViewpointType, random_foveated_viewpoint, sample_view_scales


@dataclass
class RolloutCtx:
    """Per-rollout selector state (one branch = one rollout)."""

    rollout_scales: Tensor | None  # frozen [B] view scale for foveated mode='per_rollout'


class Selector(Protocol):
    def start_rollout(
        self, *, t0_type: ViewpointType, batch_size: int, device: torch.device
    ) -> RolloutCtx: ...

    def select(
        self,
        *,
        vp_type: ViewpointType,
        ctx: RolloutCtx,
        t: int,
        batch_size: int,
        device: torch.device,
        state: RecurrentState,
    ) -> NamedViewpoint: ...


@dataclass
class RandomSelector:
    """The historical random viewing policy, patcher-aware and content-independent:
    uniform patcher -> safe-box-area sampler (p(s) ∝ (1-s), centers coupled to the
    safe box); foveated/square -> fixation-style viewpoints with the view scale
    drawn per FoveatedScaleConfig. This is the canonical random distribution the
    distribution tests pin (master plan §3)."""

    is_foveated: bool
    foveated_scale: FoveatedScaleConfig
    min_viewpoint_scale: float

    def start_rollout(
        self, *, t0_type: ViewpointType, batch_size: int, device: torch.device
    ) -> RolloutCtx:
        # Per-rollout scale: one scale per branch (per image), held across all of
        # this rollout's glimpses (per_rollout => constant scale within a rollout).
        # A FULL-start rollout is the scale-1 global anchor, so ALL its glimpses
        # (the full t0 AND its subsequent random glimpses) stay at scale=1 to keep
        # the rollout in-distribution; RANDOM-start rollouts use the sampled scale.
        rollout_scales: Tensor | None = None
        if self.is_foveated and self.foveated_scale.mode == "per_rollout":
            if t0_type == ViewpointType.FULL:
                rollout_scales = torch.ones(batch_size, device=device)
            else:
                rollout_scales = sample_view_scales(
                    batch_size, device,
                    distribution=self.foveated_scale.distribution,
                    min_scale=self.foveated_scale.min_scale,
                    max_scale=self.foveated_scale.max_scale,
                )
        return RolloutCtx(rollout_scales=rollout_scales)

    def _foveated_random_vp(
        self, rollout_scales: Tensor | None, batch_size: int, device: torch.device
    ) -> NamedViewpoint:
        """RANDOM viewpoint for the foveated/square path, with the view scale
        drawn per ``foveated_scale`` (see :class:`FoveatedScaleConfig`).
        ``rollout_scales`` is the frozen [B] scale for ``mode='per_rollout'``."""
        fs = self.foveated_scale
        if fs.mode == "fixed":
            scales = torch.full((batch_size,), float(fs.fixed_scale), device=device)
            center_mode = "full_field"
        else:
            if fs.mode == "per_rollout":
                assert rollout_scales is not None
                scales = rollout_scales
            else:  # per_glimpse
                scales = sample_view_scales(
                    batch_size, device, distribution=fs.distribution,
                    min_scale=fs.min_scale, max_scale=fs.max_scale,
                )
            center_mode = "safebox" if fs.distribution == "safebox" else "full_field"
        return random_foveated_viewpoint(batch_size, device, scales=scales, center_mode=center_mode)

    def select(
        self,
        *,
        vp_type: ViewpointType,
        ctx: RolloutCtx,
        t: int,
        batch_size: int,
        device: torch.device,
        state: RecurrentState,
    ) -> NamedViewpoint:
        """Create a NamedViewpoint (has .name for viz, convertible to canvit Viewpoint).

        Foveated/square path: RANDOM glimpses draw their view scale per
        ``foveated_scale`` (center per the chosen distribution). The FULL start
        glimpse is centered at fixation (center=0); its scale depends on mode:
        ``fixed`` -> the single training scale ``fixed_scale`` (so it matches every
        other glimpse; ``fixed_scale=1`` reproduces the original scale-1 full view),
        while ``per_rollout`` / ``per_glimpse`` keep it at scale=1 -- a full-image
        anchor that eases optimization (the RANDOM glimpses still zoom per the mode).
        Uniform path: existing safe-box-area sampler, FULL stays scale=1.
        """
        if vp_type == ViewpointType.RANDOM:
            if self.is_foveated:
                return self._foveated_random_vp(ctx.rollout_scales, batch_size, device)
            return NamedViewpoint.random(
                batch_size=batch_size, device=device, min_scale=self.min_viewpoint_scale
            )
        assert vp_type == ViewpointType.FULL
        if self.is_foveated and self.foveated_scale.mode == "fixed":
            return NamedViewpoint(
                name="full",
                centers=torch.zeros(batch_size, 2, device=device),
                scales=torch.full((batch_size,), float(self.foveated_scale.fixed_scale), device=device),
            )
        return NamedViewpoint.full_scene(batch_size=batch_size, device=device)
