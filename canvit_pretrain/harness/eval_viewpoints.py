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

EvalPolicy = Literal["auto", "coarse_to_fine", "random", "full", "fixation_grid", "policy"]

OPEN_LOOP: tuple[str, ...] = ("coarse_to_fine", "random", "full", "fixation_grid")
"""Policies whose whole trajectory is known before the rollout starts. ``"policy"`` is
the only closed-loop one — it needs the live canvas state to pick the next glimpse."""

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
    assert policy in OPEN_LOOP or policy == "policy", f"unknown eval policy {policy!r}"
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
    if policy == "policy":
        raise ValueError(
            "eval_policy='policy' is closed-loop: the next viewpoint depends on the canvas "
            "state, so it cannot be precomputed. Drive it with deploy_selector() instead."
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
    "OPEN_LOOP",
    "EvalPolicy",
    "deploy_rollout_viewpoints",
    "deploy_selector",
    "open_loop_viewpoints",
    "resolve",
]
