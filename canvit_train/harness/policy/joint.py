"""Joint task+policy training orchestrator (unification P4b).

Ties the P4a selector seam to the P3 RL objectives so a ViewpointScorer learns
*inside* the pretraining rollout: the distill task keeps training while the
policy drives the glimpses from its candidate grid (in-graph, master plan §4.3;
BatchNorm mode (a) — the train-mode scorer forward that picks the action is the
one the loss reads). OFF unless ``cfg.rl.use_rl`` — ``build_joint_policy`` is
only called then, so with it off ``training_step`` sees ``joint=None`` and runs
the byte-identical historical path (parity gate).

Reward (master plan §3): per glimpse, the fractional reduction in per-image
distill MSE ``r_t = (L_{t-1} - L_t) / L_{t-1}`` (measured detached by the caller),
standardized per depth by an online ``RunningNorm`` — the QReg regression target /
the PG advantage. The scorer + encoder are probe-free here (INTRINSIC feature
groups): distillation has no task probe, unlike the ADE20K policy trainer.
"""

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
import torch.distributed as dist
from canvit_pytorch.policy import (
    StateEncoder,
    ViewpointScorer,
    candidate_viewpoints,
    fixation_candidates,
)
from canvit_pytorch.policy.features import INTRINSIC_GROUPS
from torch import Tensor

from canvit_train.harness.config import FoveatedScaleConfig, JointPolicyConfig
from canvit_train.harness.policy.rl import (
    PG,
    VPG,
    Objective,
    QReg,
    RunningNorm,
    entropy_floor_step,
    pg_loss,
    qreg_loss,
    vpg_loss,
)
from canvit_train.harness.rollout.selector import PolicySelector, RandomSelector
from canvit_train.harness.rollout.viewpoint import ViewpointType


@dataclass
class JointPolicy:
    """Owns the policy machinery consulted by ``training_step`` when RL is on: the
    two selectors (policy / pure-random), the scorer module (its params go into the
    optimizer + checkpoint), the objective, the per-depth reward standardizers, and
    the PG entropy-floor dual variable."""

    policy_selector: PolicySelector
    random_selector: RandomSelector
    scorer: ViewpointScorer
    objective: Objective
    rl_weight: float
    keep_random_branch: bool
    target_momentum: float
    device: torch.device
    prime_target: float = 1.0  # ε-curriculum: prime_on_policy ramps 0 -> this...
    prime_warmup: int = 0  # ...over this many steps (0 = constant target)
    running: dict[int, RunningNorm] = field(default_factory=dict)
    log_alpha: Tensor | None = None
    last_step: dict = field(default_factory=dict)  # detached per-step policy metrics for logging

    def branch_selector(self, t0_type: ViewpointType) -> tuple[object, bool]:
        """(selector, is_policy_branch) for a branch with this t0 type. Policy
        branches are FULL-anchored (the caller forces t0=FULL and rewards t>=1);
        with ``keep_random_branch`` the RANDOM-start branches stay pure-random
        (distill-only, no policy loss). Off => every branch is a policy branch."""
        is_policy = (not self.keep_random_branch) or (t0_type == ViewpointType.FULL)
        return (self.policy_selector, True) if is_policy else (self.random_selector, False)

    def set_prime_for_step(self, step: int) -> None:
        """ε-curriculum: ramp prime_on_policy 0 -> prime_target over prime_warmup
        steps, then hold (QReg only; PG always samples on-policy, prime ignored)."""
        if not isinstance(self.objective, QReg):
            return
        frac = 1.0 if self.prime_warmup <= 0 else min(1.0, step / self.prime_warmup)
        self.policy_selector.prime_on_policy = self.prime_target * frac

    # --- DDP: the scorer is NOT DDP-wrapped (its forward is deep inside the rollout),
    # so sync it manually — broadcast once so all ranks start identical, AllReduce its
    # grads each step (else per-rank scorers drift + see √N grad noise, the head bug
    # task.py documents). RunningNorm stays per-rank (rl.py) — standardization is local.
    def broadcast(self, src: int = 0) -> None:
        for t in list(self.scorer.parameters()) + list(self.scorer.buffers()):
            dist.broadcast(t.data, src=src)

    def allreduce_grads(self) -> None:
        for p in self.scorer.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    def state_dict(self) -> dict:
        """Policy training state for the checkpoint sidecar (scorer + reward
        standardizers + PG dual variable). The optimizer/scheduler state — including
        the policy param group — rides in the main checkpoint."""
        return {
            "scorer": self.scorer.state_dict(),
            "running": {d: (rn.mean, rn.sq, rn.count) for d, rn in self.running.items()},
            "log_alpha": None if self.log_alpha is None else self.log_alpha.detach().cpu(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.scorer.load_state_dict(sd["scorer"])
        for d, (mean, sq, count) in sd.get("running", {}).items():
            rn = self._norm(int(d))
            rn.mean, rn.sq, rn.count = mean.to(self.device), sq.to(self.device), count
        if sd.get("log_alpha") is not None and self.log_alpha is not None:
            self.log_alpha = sd["log_alpha"].to(self.device)

    def _norm(self, depth: int) -> RunningNorm:
        rn = self.running.get(depth)
        if rn is None:  # created on first use — horizon varies per step (continue_prob)
            rn = RunningNorm(momentum=self.target_momentum, device=self.device)
            self.running[depth] = rn
        return rn

    @property
    def defers_credit(self) -> bool:
        """True => the reward is TERMINAL, so ``run_rollout`` must buffer the per-step
        policy aux and call :meth:`trajectory_loss` once at rollout end instead of
        :meth:`glimpse_loss` inside the loop (VPG). False => the historical inline path."""
        return isinstance(self.objective, VPG)

    def state_value(self, feats: Tensor) -> Tensor:
        """``V(s)`` [B] for the VPG baseline, read from the scorer's dueling value head.

        Deliberately a SECOND (very cheap: pool -> 64 -> 1) forward of ``vhead`` rather
        than recovering V from the score map's row mean: with ``dueling=True`` the map is
        ``V(s) + mean-zero A(s,a)`` so the row mean IS V analytically, but only to float
        error, and reading it that way would silently break if the dueling algebra ever
        changed. This keeps the baseline exact and the coupling explicit."""
        assert self.scorer.vhead is not None, (
            "VPG needs a learned V(s): build the ViewpointScorer with dueling=True"
        )
        return self.scorer.vhead(feats.float().mean(dim=(2, 3))).squeeze(-1)

    def trajectory_loss(self, *, scores: Tensor, flat_idx: Tensor, values: Tensor, reward: Tensor) -> Tensor:
        """The weighted policy loss for a WHOLE trajectory (VPG only). ``scores``
        [B, T, A] / ``values`` [B, T] carry grad; ``reward`` [B] is the detached terminal
        reward. Unlike :meth:`glimpse_loss` this is NOT per-depth, so there is no
        ``RunningNorm`` — VPG standardizes the ADVANTAGE per timestep over the batch
        instead of the reward stream over time (see :class:`VPG`)."""
        obj = self.objective
        assert isinstance(obj, VPG)
        loss, metrics = vpg_loss(
            scores, flat_idx, values, reward,
            entropy_bonus=obj.entropy_bonus, reinforce_weight=obj.reinforce_weight,
            baseline_weight=obj.baseline_weight, normalize_advantage=obj.normalize_advantage,
        )
        self.last_step = metrics
        return self.rl_weight * loss

    def glimpse_loss(self, *, depth: int, scores: Tensor, flat_idx: Tensor, reward: Tensor) -> Tensor:
        """The weighted policy loss for ONE glimpse. ``scores`` [B, A] carries grad
        (the train-mode scorer forward); ``flat_idx`` [B] is the taken candidate;
        ``reward`` [B] is the raw fractional distill-MSE reduction (detached)."""
        obj = self.objective
        if isinstance(obj, PG):
            target = self._norm(depth).normalize(reward, subtract_only=obj.z_subtract_only).detach()
            alpha = obj.entropy_bonus if self.log_alpha is None else self.log_alpha.exp()
            loss, entropy, _ = pg_loss(scores, flat_idx, target, alpha=alpha, crit_all=None)
            if self.log_alpha is not None:
                assert obj.entropy_target is not None
                entropy_floor_step(
                    log_alpha=self.log_alpha, entropy=entropy.detach(),
                    target=obj.entropy_target, alpha_lr=obj.alpha_lr, alpha_min=obj.entropy_bonus,
                )
        else:
            target = self._norm(depth).normalize(reward).detach()
            loss, _ = qreg_loss(scores, flat_idx, target)
        return self.rl_weight * loss


def build_joint_policy(
    *,
    core_model,
    rl: JointPolicyConfig,
    device: torch.device,
    canvas_grid: int,
    min_viewpoint_scale: float,
    foveated_scale: FoveatedScaleConfig,
    generator: torch.Generator,
) -> JointPolicy:
    """Assemble the scorer, encoder and selectors for the model's patcher family
    (fixation grid for foveated/square, safe-box grid for uniform), plus the
    objective and reward standardizers. PG here is score-function + entropy floor
    only — Q-Prop's control-variate critic (the standalone ADE trainer's `qprop`) is
    not wired into joint mode, so JointPolicyConfig deliberately exposes no knob."""
    if rl.objective == "vpg":
        # Deliberately NOT supported on this legacy path: VPG needs the rollout engine's
        # deferred-credit branch (terminal reward), which only harness/rollout.py has. The
        # old train/step.py rollout would silently drop the policy loss entirely.
        raise NotImplementedError(
            "objective='vpg' requires the unified harness rollout (terminal reward => "
            "deferred credit). Run it via `python -m canvit_train.harness.run <task>`, "
            "not the legacy train/loop.py path."
        )
    is_foveated = getattr(core_model.cfg, "patcher_name", "uniform") in ("foveated", "square")
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

    # `rl.feature_groups` is None-by-default now ("use the task's own set"); this legacy path
    # is distill-only, whose set is INTRINSIC_GROUPS — the value the field used to default to,
    # so this is a no-op for every existing distill run.
    groups = tuple(rl.feature_groups) if rl.feature_groups is not None else INTRINSIC_GROUPS

    scorer = ViewpointScorer(
        canvas_dim=core_model.cfg.canvas_dim, width=rl.width, n_scale=n_scale, scales=scales,
        centers_per_axis=rl.centers_per_axis, block_layers=rl.block_layers, groups=groups,
        dueling=isinstance(obj, QReg) and obj.dueling, action_space=action_space,
        readout=rl.policy_readout,
    ).to(device)
    scorer.train()

    # StateEncoder wants a segmentation model (seg.canvit.*); with INTRINSIC groups it
    # only touches seg.canvit.get_spatial / .init_state, so a shim onto the core model
    # is all it needs (no probe -> no seg.head).
    encoder = StateEncoder(
        SimpleNamespace(canvit=core_model), canvas_grid=canvas_grid, feature_groups=groups
    )

    random_sel = RandomSelector(
        is_foveated=is_foveated, foveated_scale=foveated_scale, min_viewpoint_scale=min_viewpoint_scale
    )
    policy_sel = PolicySelector(
        net=scorer, encoder=encoder, vp_flat=vp_flat, fallback=random_sel,
        mode="sample" if isinstance(obj, PG) else "argmax",
        prime_on_policy=rl.prime_on_policy if isinstance(obj, QReg) else 1.0,
        feats_detached=rl.feats_detached, generator=generator,
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
