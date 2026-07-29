"""Task-agnostic joint-policy builder for the unified harness (design §7 D-D crown jewel).

Generalizes ``train/joint.py::build_joint_policy`` (which hardwired the distill case:
the core model IS the canvit, INTRINSIC feature groups, no probe) to ANY task by
splitting two roles that distill happened to conflate:

  * ``canvit``       — the recurrent backbone; its ``.cfg`` gives canvas_dim + patcher_name.
  * ``encode_model`` — what the ``StateEncoder`` featurizes: an object exposing ``.canvit``
                       (and, for probe-entropy feature groups, ``.head``). For distill this
                       is ``SimpleNamespace(canvit=canvit)``; for ade20k/in1k it is the task
                       wrapper (``seg`` / ``clf``) so probe-aware features can reach the head.
  * ``feature_groups`` — the task's scorer features (``Task.policy_feature_groups()``):
                       INTRINSIC for distill/in1k, the full set (with ent/ent_delta) for the
                       spatial segmentation probe in ade20k.

Reuses the existing :class:`JointPolicy` container, objectives, selectors and scorer —
this only rewires the encoder so joint task+policy works for all three tasks (distill
already worked; this unlocks ade20k/in1k). The old ``build_joint_policy`` stays until
the big-bang cutover.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
from canvit_pytorch.policy import (
    StateEncoder,
    ViewpointScorer,
    candidate_viewpoints,
    fixation_candidates,
)

from canvit_pretrain.train.config import FoveatedScaleConfig, JointPolicyConfig
from canvit_pretrain.train.joint import JointPolicy
from canvit_pretrain.train.rl import PG, Objective, QReg
from canvit_pretrain.train.selector import PolicySelector, RandomSelector


def build_policy(
    *,
    canvit: Any,
    rl: JointPolicyConfig,
    feature_groups: tuple[str, ...],
    device: torch.device,
    canvas_grid: int,
    min_viewpoint_scale: float,
    foveated_scale: FoveatedScaleConfig,
    generator: torch.Generator,
    encode_model: Any | None = None,
) -> JointPolicy:
    """Assemble a :class:`JointPolicy` for any task. ``encode_model`` defaults to
    ``SimpleNamespace(canvit=canvit)`` (the distill case); pass the task wrapper
    (``seg``/``clf``) so probe-aware feature groups can read its head. ``feature_groups``
    comes from the task (INTRINSIC vs the probe-entropy set)."""
    if encode_model is None:
        encode_model = SimpleNamespace(canvit=canvit)

    is_foveated = getattr(canvit.cfg, "patcher_name", "uniform") in ("foveated", "square")
    if is_foveated:
        cand = fixation_candidates(rl.centers_per_axis)
        n_scale, scales, action_space = 1, (1.0,), "fixation"
    else:
        cand = candidate_viewpoints(rl.scales, rl.centers_per_axis)
        n_scale, scales, action_space = cand.shape[0], rl.scales, "safebox"
    vp_flat = cand.reshape(-1, 3).to(device)

    if rl.objective == "qreg":
        obj: Objective = QReg(prime_on_policy=rl.prime_on_policy, dueling=rl.dueling)
    else:
        obj = PG(entropy_bonus=rl.entropy_bonus, entropy_target=rl.entropy_target, alpha_lr=rl.alpha_lr)

    scorer = ViewpointScorer(
        canvas_dim=canvit.cfg.canvas_dim, width=rl.width, n_scale=n_scale, scales=scales,
        centers_per_axis=rl.centers_per_axis, block_layers=rl.block_layers, groups=feature_groups,
        dueling=isinstance(obj, QReg) and obj.dueling, action_space=action_space,
    ).to(device)
    scorer.train()

    encoder = StateEncoder(encode_model, canvas_grid=canvas_grid, feature_groups=feature_groups)

    random_sel = RandomSelector(
        is_foveated=is_foveated, foveated_scale=foveated_scale, min_viewpoint_scale=min_viewpoint_scale
    )
    policy_sel = PolicySelector(
        net=scorer, encoder=encoder, vp_flat=vp_flat, fallback=random_sel,
        mode="sample" if isinstance(obj, PG) else "argmax",
        prime_on_policy=rl.prime_on_policy if isinstance(obj, QReg) else 1.0,
        feats_detached=rl.feats_detached, select_bn_eval=rl.select_bn_eval, generator=generator,
    )

    jp = JointPolicy(
        policy_selector=policy_sel, random_selector=random_sel, scorer=scorer, objective=obj,
        rl_weight=rl.rl_weight, keep_random_branch=rl.keep_random_branch,
        target_momentum=rl.target_momentum, device=device,
        prime_target=rl.prime_on_policy, prime_warmup=rl.policy_warmup_steps,
    )
    if isinstance(obj, PG) and obj.entropy_target is not None:
        jp.log_alpha = torch.tensor(math.log(obj.entropy_bonus), device=device)
    return jp


__all__ = ["build_policy"]
