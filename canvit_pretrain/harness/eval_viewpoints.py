"""One validation-viewpoint interface for all three tasks.

Before this module each task picked its validation trajectory a different way, not by
design but by inheritance — every task's ``evaluate()`` was lifted from a different
ancestor and kept that ancestor's habit:

  * distill  <- the old pretrain loop's ``validate()``  -> quadtree C2F (uniform) /
                the deterministic centre+3x3 fixation trajectory (foveated)
  * ade20k   <- the specialize probe, which TRAINED on random viewpoints, so it
                validated on random too -> IID random, with no knob at all
  * in1k     <- canvit_eval's deploy convention -> C2F, and it already had the knob

Nothing about the tasks requires this. The *metrics* genuinely differ (mIoU vs top-1/5
vs cosine-to-teacher) and stay task-owned; the viewpoint sequence is a rollout concern
and belongs here. This module is that seam: one option set, one place where the
patcher-awareness rule lives, and ``"policy"`` — deploy the learned scorer by argmax —
available to every task instead of only to the standalone RL trainer.

**Defaults are deliberately NOT unified.** Each task keeps exactly the trajectory it
used before (see ``HISTORICAL_DEFAULTS``): flipping ade20k to C2F would silently break
comparability with every specialize probe number and every exp24 run, and flipping
distill would break the exp22/23/26 val curves. The knob is shared; the default is
per-task and documented. ``"auto"`` means "whatever this task has always done", so
every existing config is a no-op through this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from canvit_pytorch.viewpoint import Viewpoint

if TYPE_CHECKING:  # the task configs import EvalPolicy from here, so stay dependency-light
    from canvit_pretrain.train.config import FoveatedScaleConfig

EvalPolicy = Literal["auto", "coarse_to_fine", "random", "full", "fixation_grid",
                     "entropy_coarse_to_fine", "policy"]

OPEN_LOOP: tuple[str, ...] = ("coarse_to_fine", "random", "full", "fixation_grid")
"""Policies whose whole trajectory is known before the rollout starts."""

CLOSED_LOOP: tuple[str, ...] = ("policy", "entropy_coarse_to_fine")
"""Policies that need the live canvas state to pick the next glimpse, so their
trajectory cannot be precomputed. Both are driven by a ``(t, state) -> Viewpoint``
chooser through the task's rollout."""

PAPER_TABLE4_C64: dict[str, list[float]] = {
    # arXiv:2603.22570 Table 4, canvas 64^2, ADE20K val, squish-512, t = 0..4.
    # Mirrored from CanViT-PyTorch-RL/docs/paper-tables.md, which records them as
    # "verbatim targets for harness verification". Use as the validation target for
    # any baseline implemented here, and as dashed reference lines on Figure-4B plots.
    "entropy_coarse_to_fine": [39.6, 42.2, 43.3, 44.1, 44.7],   # EG-C2F (deterministic)
    "coarse_to_fine":         [39.6, 41.3, 42.5, 43.6, 44.7],   # C2F (paper: mean of n=10)
}
"""Paper row -> our policy name, for the rows we can actually reproduce.

VERIFIED 2026-07-30 on full ADE20K val (squish-512, c64, eval batch 32):

    EG-C2F  ours 39.58 42.22 43.31 44.05 44.67   vs paper  max|delta| = 0.05
    C2F     ours 39.58 41.23 42.54 43.54 44.71   vs paper  max|delta| = 0.07

EG-C2F is DETERMINISTIC in the paper, so that agreement is a real validation of the port,
not a coincidence of averaging. C2F is stochastic there (mean of n=10), so 0.07 is within
its CI.

**``random`` is NOT the paper's F-IID — do not label it as such.** It measures
39.58/41.37/42.18/42.84/43.42, i.e. +0.17..+0.42 ABOVE the F-IID row, growing with t.
``open_loop_viewpoints`` does pass ``start_with_full_scene=True`` so t0 matches, but the
glimpses then follow the safe-box AREA law (p(s) ~ 1-s over [min_scale, max_scale]) rather
than F-IID's fixed fovea-sized scale — a different, measurably stronger random policy.
The paper's F-IID and R-IID rows are both unreachable from this module today."""

HISTORICAL_DEFAULTS: dict[str, tuple[str, str]] = {
    # task        (uniform patcher,   foveated/square patcher)
    "distill":    ("coarse_to_fine",  "fixation_grid"),
    "ade20k":     ("random",          "random"),
    "in1k":       ("coarse_to_fine",  "coarse_to_fine"),
}
"""What ``"auto"`` resolves to, per task and patcher — i.e. what each task did before
this module existed. Written down in ONE table so the divergence is visible instead of
scattered across three ``evaluate()`` methods. Notes on the non-obvious entries:

* **distill/foveated -> fixation_grid.** The quadtree's varying scales are out of
  distribution for a fixed-scale foveated model (``fix_size = scale * H``), so the old
  loop used a deterministic centre + shuffled-3x3 trajectory at the TRAINING scale.
* **ade20k -> random.** Inherited from the specialize probe, which trained on random
  viewpoints. Consistent for a probe; wrong the moment a policy is in the loop, which
  is why ``"policy"`` exists.
* **in1k/foveated -> coarse_to_fine.** This is the known OOD footgun that distill
  avoids, retained on purpose: exp25's foveated in1k arrays were measured under it and
  changing the default would make them non-comparable. Pass ``--cfg.eval-policy
  fixation_grid`` (or ``full``) for a scale-pinned foveated deploy.
"""


def resolve(policy: str, *, task: str, is_foveated: bool) -> str:
    """Map ``"auto"`` onto the task's historical trajectory; pass anything else through."""
    assert task in HISTORICAL_DEFAULTS, f"unknown task {task!r}"
    if policy == "auto":
        uniform_default, foveated_default = HISTORICAL_DEFAULTS[task]
        return foveated_default if is_foveated else uniform_default
    assert policy in OPEN_LOOP or policy in CLOSED_LOOP, f"unknown eval policy {policy!r}"
    return policy


def open_loop_viewpoints(
    policy: str,
    *,
    batch_size: int,
    device: torch.device,
    n: int,
    is_foveated: bool,
    foveated_scale: FoveatedScaleConfig,
    min_scale: float = 0.05,
    max_scale: float = 1.0,
    foveated_eval_scale: float = 1.0,
) -> list[Viewpoint]:
    """The precomputed trajectory for every policy except ``"policy"``.

    Delegates to the existing, tested generators rather than reimplementing them, so
    each option is bit-identical to the task that used to own it.
    """
    from canvit_pytorch.policies import repeated_full_scene

    from canvit_pretrain.ade20k.rollout import make_random_viewpoints
    from canvit_pretrain.train.viewpoint import make_eval_viewpoints, make_eval_viewpoints_foveated

    if policy == "coarse_to_fine":
        return make_eval_viewpoints(batch_size, device, n_viewpoints=n)
    if policy == "fixation_grid":
        return make_eval_viewpoints_foveated(batch_size, device, n_viewpoints=n,
                                             scale=foveated_eval_scale)
    if policy == "full":
        return repeated_full_scene(batch_size, device, n)
    if policy == "random":
        from canvit_pretrain.train.config import FoveatedScaleConfig
        # Only the foveated branch reads it, and there a WRONG scale is not a soft
        # mismatch — `fix_size = scale * H` puts every glimpse out of distribution and
        # mIoU falls as glimpses accumulate (job 15025338). So silently defaulting it for
        # a foveated model would hide the exact bug this codebase already paid for once.
        assert not (is_foveated and foveated_scale is None), (
            "eval_policy='random' on a foveated/square model needs the pretraining "
            "foveated_scale; defaulting it would put every glimpse out of distribution.")
        return make_random_viewpoints(
            batch_size, device, n, min_scale=min_scale, max_scale=max_scale,
            start_with_full_scene=True, is_foveated=is_foveated,
            foveated_scale=foveated_scale or FoveatedScaleConfig(),
        )
    if policy in CLOSED_LOOP:
        raise ValueError(
            f"eval_policy={policy!r} is closed-loop: the next viewpoint depends on the "
            "canvas state, so it cannot be precomputed. Drive it through the task's "
            "closed-loop rollout instead."
        )
    raise ValueError(f"unknown eval policy: {policy!r}")


def deploy_selector(joint: Any) -> Any:
    """The learned policy in DEPLOY configuration: pure argmax, no exploration.

    ``prime_on_policy=1.0`` is the deployed rule (and consumes no RNG, so a deploy eval
    cannot perturb the training stream). The scorer is switched to eval mode by
    :func:`deploy_rollout_viewpoints`, which owns the train/eval restore.
    """
    from dataclasses import replace

    assert joint is not None and getattr(joint, "scorer", None) is not None, (
        "eval_policy='policy' needs a trained scorer, but this run has no policy. Either "
        "train one (--preset policy_only / joint) or pick an open-loop eval policy."
    )
    return replace(joint.policy_selector, mode="argmax", prime_on_policy=1.0)


def deploy_rollout_viewpoints(
    *,
    joint: Any,
    advance: Any,
    t0_type: Any,
    batch_size: int,
    device: torch.device,
    n: int,
) -> list[Viewpoint]:
    """Run the closed-loop deploy rollout, returning the viewpoints it actually took.

    ``advance(viewpoint, state, t) -> RecurrentState`` is the task's own single-glimpse
    step (called with ``state=None`` at t0, so it owns its own state init); this function
    owns only the selection. The scorer runs in eval mode under ``no_grad`` — deploy
    semantics, and the reason a policy eval cannot leak gradient or BatchNorm statistics
    into training. Train mode is restored on the way out.
    """
    from canvit_pretrain.harness.rollout import _to_vp
    from canvit_pretrain.train.viewpoint import ViewpointType

    # t0 must be the FULL anchor: the scorer needs a canvas to read, and at t=0 there is
    # no state yet. FULL delegates to the RandomSelector fallback, which needs none. This
    # matches all three tasks' validation (C2F, random and fixation_grid all start full)
    # and the RL reference, whose episodes open on the full scene.
    assert t0_type == ViewpointType.FULL, (
        f"eval_policy='policy' requires a FULL t0 anchor, got {t0_type}")

    sel = deploy_selector(joint)
    scorer = joint.scorer
    was_training = scorer.training
    scorer.eval()
    try:
        with torch.no_grad():
            ctx = sel.start_rollout(t0_type=t0_type, batch_size=batch_size, device=device)
            state, taken = None, []
            for t in range(n):
                vp_type = t0_type if t == 0 else ViewpointType.RANDOM
                named = sel.select(vp_type=vp_type, ctx=ctx, t=t, batch_size=batch_size,
                                   device=device, state=state)
                vp = _to_vp(named)
                taken.append(vp)
                state = advance(vp, state, t)
    finally:
        if was_training:
            scorer.train()
    return taken


__all__ = [
    "HISTORICAL_DEFAULTS",
    "CLOSED_LOOP",
    "EntropyGuidedC2F",
    "OPEN_LOOP",
    "PAPER_TABLE4_C64",
    "closed_loop_rollout",
    "entropy_c2f_chooser",
    "EvalPolicy",
    "deploy_rollout_viewpoints",
    "deploy_selector",
    "open_loop_viewpoints",
    "resolve",
]


# --- EG-C2F: entropy-guided coarse-to-fine (the paper's strongest heuristic) ------


def _tile_masks(crops: list[tuple[float, float, float]], canvas_grid: int,
                device: Any) -> Any:
    """[n_crops, G, G] bool: which canvas cells fall inside each crop.

    Ported verbatim from ``canvit_eval/policies.py::_build_tile_masks`` — the
    implementation the published EG-C2F numbers were produced with.
    """
    import torch

    G = canvas_grid
    assert G > 0 and (G & (G - 1)) == 0, f"canvas_grid must be a power of 2, got {G}"
    coords = torch.linspace(-1 + 1 / G, 1 - 1 / G, G, device=device)
    crops_t = torch.tensor(crops, device=device)
    cy, cx, s = crops_t[:, 0], crops_t[:, 1], crops_t[:, 2]
    row_in = (coords.unsqueeze(0) - cy.unsqueeze(1)).abs() <= s.unsqueeze(1)
    col_in = (coords.unsqueeze(0) - cx.unsqueeze(1)).abs() <= s.unsqueeze(1)
    return row_in.unsqueeze(2) & col_in.unsqueeze(1)


class EntropyGuidedC2F:
    """Coarse-to-fine quadtree levels, visited in order of DECREASING per-tile probe entropy.

    A port of ``canvit_eval/policies.py::EntropyGuidedC2F``, which is what produced the
    published EG-C2F row (paper Table 4 / `PAPER_TABLE4_C64`). Closed-loop: the pick needs
    the live canvas, because the entropy map is read off the probe at each step.

    Schedule: 3 quadtree levels — 1 full scene, then 4 half-quadrants, then 16 quarter-tiles
    (21 timesteps total). At the harness default ``n_timesteps=5`` that is the full scene
    plus the 4 half-quadrants ranked by entropy, which is exactly the paper's t0..t4 row.
    """

    def __init__(self, *, seg: Any, batch_size: int, device: Any, canvas_grid: int,
                 n_levels: int = 3) -> None:
        from canvit_pytorch.policies import level_viewpoints

        self.seg, self.B, self.device, self.canvas_grid = seg, batch_size, device, canvas_grid
        self.levels = [level_viewpoints(lvl) for lvl in range(n_levels)]
        self.level_starts, t = [], 0
        for lvl in self.levels:
            self.level_starts.append(t)
            t += len(lvl)
        self.masks: list[Any] = [None] + [
            _tile_masks(self.levels[lvl], canvas_grid, device) for lvl in range(1, n_levels)]
        self.visited: list[Any] = [None] * n_levels

    def _entropy(self, state: Any) -> Any:
        """Per-cell probe entropy [B, G, G]. fp32 via ``head_logits`` (which disables
        autocast), so the ranking does not depend on the caller's amp context."""
        from canvit_pytorch.policy import entropy_from_logits, head_logits

        return entropy_from_logits(
            head_logits(self.seg, state.canvas, canvas_grid=self.canvas_grid)).float()

    def __call__(self, t: int, state: Any) -> Viewpoint:
        import torch

        level_idx = sum(1 for s in self.level_starts[1:] if t >= s)
        pos_in_level = t - self.level_starts[level_idx]
        crops = self.levels[level_idx]

        if level_idx == 0:                      # the full-scene anchor; no choice to make
            cy, cx, s = crops[0]
            return Viewpoint(
                centers=torch.tensor([[cy, cx]], device=self.device).expand(self.B, -1).contiguous(),
                scales=torch.full((self.B,), s, device=self.device))

        if pos_in_level == 0:                   # entering a level: nothing visited yet
            self.visited[level_idx] = torch.zeros(
                self.B, len(crops), dtype=torch.bool, device=self.device)

        ent, masks, visited = self._entropy(state), self.masks[level_idx], self.visited[level_idx]
        n_cells = masks.sum(dim=(1, 2)).clamp(min=1).float()
        # mean entropy per candidate tile, then take the highest UNVISITED one
        scores = (ent.unsqueeze(1) * masks.unsqueeze(0).float()).sum(dim=(2, 3)) / n_cells
        scores = scores.masked_fill(visited, float("-inf"))
        chosen = scores.argmax(dim=1)
        visited.scatter_(1, chosen.unsqueeze(1), True)

        selected = torch.tensor(crops, device=self.device)[chosen]
        return Viewpoint(centers=selected[:, :2].contiguous(), scales=selected[:, 2].contiguous())


def closed_loop_rollout(*, chooser: Any, advance: Any, n: int) -> list[Viewpoint]:
    """Drive a closed-loop eval policy: ``chooser(t, state) -> Viewpoint``, then
    ``advance(vp, state, t) -> state`` (called with ``state=None`` at t0, so the task owns
    its own state init). Shared by the learned policy and EG-C2F so both take exactly the
    same path through the task's rollout, and no metric code has to branch on which."""
    import torch

    with torch.no_grad():
        state, taken = None, []
        for t in range(n):
            vp = chooser(t, state)
            taken.append(vp)
            state = advance(vp, state, t)
    return taken


def entropy_c2f_chooser(*, seg: Any, batch_size: int, device: Any, canvas_grid: int) -> Any:
    """A fresh :class:`EntropyGuidedC2F` — fresh because it carries per-rollout `visited`
    state, so reusing one across batches would silently exclude already-picked tiles."""
    return EntropyGuidedC2F(seg=seg, batch_size=batch_size, device=device,
                            canvas_grid=canvas_grid)
