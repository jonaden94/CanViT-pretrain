"""Generic RL training machinery for viewing policies (unification P3).

Ported from canvit_pytorch_rl.training.{config,stats,train}: the objective sum
type (QReg | PG), the online target standardizer (RunningNorm), the SAC-style
entropy-floor dual step, and the per-objective loss composition. Task-agnostic:
the reward is whatever per-glimpse fractional loss reduction the caller measured
(seg CE here, distill MSE later — master plan §3)."""

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

ALPHA_MAX = 10.0  # entropy-floor dual cap: a TIGHT cap binds and defeats the floor


@dataclass(frozen=True)
class QReg:
    """Value regression — the recipe: MSE at the taken cell, ε-greedy rollout, argmax deploy."""

    prime_on_policy: float = 0.5  # fraction of state-advancing glimpses taken by the net's argmax (DAgger)
    dueling: bool = True  # Q(s,a) = V(s) + mean-zero A(s,a); argmax (deploy) unchanged


@dataclass(frozen=True)
class PG:
    """Actor — the SAME net trained by on-policy score-function credit: softmax sampling
    over the candidate readout, loss -(z * log pi(a|s)) - alpha * H(pi) with alpha held
    by the entropy floor (dual ascent to entropy_target), argmax deploy. Dueling is
    structurally absent (softmax is shift-invariant: V(s) cancels)."""

    entropy_bonus: float = 0.01  # alpha init AND floor (alpha_min); fixed alpha when entropy_target=None
    entropy_target: float | None = 1.0  # nats; dual ascent holds mean policy entropy AT the target
    alpha_lr: float = 0.05  # dual step size on log(alpha)
    qprop: bool = False  # exact-expectation discrete Q-Prop: second net as control-variate critic
    z_subtract_only: bool = False  # advantage = reward - running mean, NO std division
    credit: Literal["immediate", "return"] = "immediate"


Objective = QReg | PG


@torch.no_grad()
def entropy_floor_step(*, log_alpha: Tensor, entropy: Tensor, target: float, alpha_lr: float, alpha_min: float) -> None:
    """One dual-ascent step on log(alpha), in place: alpha grows while mean policy
    entropy sits below the target, decays toward alpha_min above it. On-device, sync-free."""
    log_alpha += alpha_lr * (target - entropy)
    log_alpha.clamp_(min=math.log(alpha_min), max=math.log(ALPHA_MAX))


class RunningNorm:
    """Online global mean/std (EMA) of a scalar stream — standardizes the fractional
    reward across images/steps without per-scene statistics, so ONE sampled cell per
    scene is a valid SGD step. Adam-style bias correction -> unbiased from step 1.
    On-device, sync-free (under DDP each rank keeps its own EMA)."""

    def __init__(self, *, momentum: float, device: torch.device):
        self.m = momentum
        self.mean = torch.zeros((), device=device)
        self.sq = torch.zeros((), device=device)
        self.count = 0

    @torch.no_grad()
    def normalize(self, x: Tensor, *, subtract_only: bool = False) -> Tensor:
        self.count += 1
        self.mean = self.m * self.mean + (1 - self.m) * x.mean()
        self.sq = self.m * self.sq + (1 - self.m) * (x * x).mean()
        bc = 1 - self.m**self.count  # bias correction
        mean, sq = self.mean / bc, self.sq / bc
        if subtract_only:
            return x - mean
        return (x - mean) / (sq - mean**2).clamp_min(1e-4).sqrt()


def qreg_loss(pred_all: Tensor, flat_idx: Tensor, target: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """MSE between the predicted Q at the taken cell and the standardized reward.
    pred_all [N, A] (grad), flat_idx [N], target [N] (detached)."""
    pred_sel = pred_all.gather(1, flat_idx[:, None]).squeeze(1)
    loss = F.mse_loss(pred_sel, target)
    return loss, {"train_loss": loss, "q_sel": pred_sel.mean()}


def pg_loss(
    pred_all: Tensor,
    flat_idx: Tensor,
    target: Tensor,
    *,
    alpha: Tensor | float,
    crit_all: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Score-function credit on the on-policy taken cell (+ optional exact Q-Prop).
    Returns (loss, entropy, metrics). crit_all [N, A] (grad — its MSE is included)."""
    logp_all = F.log_softmax(pred_all, dim=1)
    pred_sel = logp_all.gather(1, flat_idx[:, None]).squeeze(1)
    entropy = -(logp_all.exp() * logp_all).sum(dim=1).mean()
    metrics: dict[str, Tensor] = {"policy_entropy": entropy, "taken_logp": pred_sel.mean(), "adv_std": target.std()}
    if crit_all is not None:  # Q-Prop: score-function on the residual + exact grad of E_pi[Q]
        crit_sel = crit_all.gather(1, flat_idx[:, None]).squeeze(1)
        critic_loss = F.mse_loss(crit_sel, target)
        analytic = (logp_all.exp() * crit_all.detach()).sum(dim=1).mean()
        loss = -((target - crit_sel.detach()) * pred_sel).mean() - analytic - alpha * entropy + critic_loss
        metrics |= {"critic_loss": critic_loss, "resid_std": (target - crit_sel.detach()).std()}
    else:
        loss = -(target * pred_sel).mean() - alpha * entropy
    return loss, entropy, metrics
