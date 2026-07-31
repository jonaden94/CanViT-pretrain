"""VPG — vanilla policy gradient with a learned baseline (the autoreg_tryout port).

Four things are worth pinning, none of which need a GPU or a real CanViT:

1. :func:`vpg_loss` is a faithful transcription of ``autoreg_tryout``'s
   ``_reinforce_loss`` (the ``rl_loss_stepwise=False`` branch) — checked against an
   independent reimplementation of that function, elementwise.
2. The two losses live on DISJOINT parameters: the score-function term cannot reach the
   value head (log_softmax is shift-invariant), and the baseline MSE cannot reach the
   score map. This is what makes reusing the dueling ``vhead`` as the baseline sound.
3. Credit is DEFERRED: the rollout buffers the trajectory and applies one policy loss at
   the end, using the terminal reward — so the gradient depends on the LAST glimpse's loss
   and every action step gets a gradient.
4. The historical inline objectives are untouched (``defers_credit`` is False for them,
   and the distill parity digest lives in ``test_rollout_parity.py``).

Run: ``.venv-cu126/bin/python -m pytest canvit_train/harness/tests/test_vpg.py``
"""

import torch
import torch.nn.functional as F

from canvit_train.train.rl import PG, VPG, QReg, vpg_loss

# --- 1. faithfulness to autoreg's _reinforce_loss ---------------------------------

def _autoreg_reinforce_loss(log_pis, baselines, entropies, R, *, reinforce_weight,
                            baseline_weight, entropy_weight, normalize_advantage):
    """Independent transcription of autoreg_tryout/utils/train.py::_reinforce_loss
    (baseline_clamp=None, i.e. rl_clamp_baseline_for_advantage=False as in exp12).
    Takes the same list-of-per-step-tensors shape the original does."""
    LP = torch.stack(log_pis, dim=1)
    B_t = torch.stack(baselines, dim=1)
    E_t = torch.stack(entropies, dim=1)
    R_exp = R.unsqueeze(1).expand_as(LP)
    B_t_det = B_t.detach()
    loss_baseline = F.mse_loss(B_t, R_exp) * baseline_weight
    advantage = R_exp - B_t_det
    if normalize_advantage:
        advantage = (advantage - advantage.mean(dim=0, keepdim=True)) / (
            advantage.std(dim=0, keepdim=True) + 1e-8
        )
    loss_reinforce = -(LP * advantage).sum(dim=1).mean() * reinforce_weight
    loss_entropy = -entropy_weight * E_t.mean()
    return loss_baseline + loss_reinforce + loss_entropy


def test_vpg_loss_matches_autoregs_reinforce_loss():
    torch.manual_seed(0)
    B, T, A = 8, 3, 5
    scores = torch.randn(B, T, A, requires_grad=True)
    values = torch.randn(B, T, requires_grad=True)
    idx = torch.randint(A, (B, T))
    reward = torch.randn(B)
    kw = dict(entropy_bonus=5e-3, reinforce_weight=1.0, baseline_weight=1.0,
              normalize_advantage=True)

    got, _ = vpg_loss(scores, idx, values, reward, **kw)

    # Rebuild the per-step lists the original signature expects, from the same tensors.
    logp_all = F.log_softmax(scores, dim=2)
    log_pis = [logp_all[:, t].gather(1, idx[:, t : t + 1]).squeeze(1) for t in range(T)]
    entropies = [-(logp_all[:, t].exp() * logp_all[:, t]).sum(dim=1) for t in range(T)]
    want = _autoreg_reinforce_loss(
        log_pis, [values[:, t] for t in range(T)], entropies, reward,
        reinforce_weight=1.0, baseline_weight=1.0, entropy_weight=5e-3,
        normalize_advantage=True,
    )
    assert torch.allclose(got, want, atol=1e-6), (got.item(), want.item())


def test_normalize_advantage_is_per_timestep_over_the_batch():
    """exp12 sets rl_normalize_advantage=True. The normalization must be over the BATCH at
    fixed t (each column standardized), not over the whole [B, T] block — with a terminal
    reward the columns differ only through V(s_t), so pooling them would erase the
    per-timestep signal the baseline provides."""
    torch.manual_seed(0)
    B, T, A = 64, 4, 3
    # Give each timestep a wildly different value scale; a per-column normalization must
    # flatten all of them to ~unit std, a global one would not.
    values = torch.arange(1.0, T + 1)[None, :].expand(B, T) * torch.randn(B, T)
    scores = torch.zeros(B, T, A, requires_grad=True)  # uniform => logp is a constant
    idx = torch.zeros(B, T, dtype=torch.long)
    reward = torch.randn(B)

    adv = reward[:, None].expand_as(values) - values
    norm = (adv - adv.mean(dim=0, keepdim=True)) / (adv.std(dim=0, keepdim=True) + 1e-8)
    assert torch.allclose(norm.mean(dim=0), torch.zeros(T), atol=1e-5)
    assert torch.allclose(norm.std(dim=0), torch.ones(T), atol=1e-2)

    _, m = vpg_loss(scores, idx, values, reward, entropy_bonus=0.0, reinforce_weight=1.0,
                    baseline_weight=1.0, normalize_advantage=True)
    assert abs(float(m["adv_std"]) - 1.0) < 0.05


def test_reward_is_broadcast_not_per_step():
    """A terminal reward means every action step sees the SAME return. Two runs whose
    rewards differ only in one image must produce identical loss for the other images'
    steps — i.e. the reward has no t dependence."""
    torch.manual_seed(0)
    B, T, A = 4, 3, 6
    scores = torch.randn(B, T, A)
    values = torch.zeros(B, T)
    idx = torch.randint(A, (B, T))
    r = torch.zeros(B)
    kw = dict(entropy_bonus=0.0, reinforce_weight=1.0, baseline_weight=0.0,
              normalize_advantage=False)
    # With V=0 and no normalization, loss = -(sum_t logp_t * R).mean(): linear in R, so a
    # unit reward on image 0 alone must equal (loss(R=e0) - loss(R=0)).
    l0, _ = vpg_loss(scores, idx, values, r, **kw)
    e0 = torch.zeros(B)
    e0[0] = 1.0
    l1, _ = vpg_loss(scores, idx, values, e0, **kw)
    logp_all = F.log_softmax(scores, dim=2)
    logp = logp_all.gather(2, idx[..., None]).squeeze(2)
    expected = -(logp[0].sum()) / B  # image 0's steps only, all sharing the same R
    assert torch.allclose(l1 - l0, expected, atol=1e-6)


# --- 2. the two losses sit on disjoint parameters ----------------------------------

class _ToyScorer(torch.nn.Module):
    """Mimics ViewpointScorer's dueling readout: a score map, made mean-zero, plus a
    scalar V(s) from its own head — the exact algebra VPG relies on."""

    def __init__(self, A: int):
        super().__init__()
        self.amap = torch.nn.Linear(4, A)
        self.vhead = torch.nn.Linear(4, 1)

    def forward(self, x):
        a = self.amap(x)
        return a - a.mean(dim=-1, keepdim=True) + self.vhead(x), self.vhead(x).squeeze(-1)


def test_score_function_term_cannot_reach_the_value_head():
    """log_softmax is shift-invariant, so the +V(s) shift drops out of the policy term.
    That is the whole justification for reading the baseline off the dueling head: the
    REINFORCE gradient provably leaves vhead alone."""
    torch.manual_seed(0)
    B, T, A = 6, 2, 5
    net = _ToyScorer(A)
    x = torch.randn(B, T, 4)
    scores, values = net(x)
    idx = torch.randint(A, (B, T))
    reward = torch.randn(B)

    # baseline_weight=0 => ONLY the score-function (+entropy) term is present.
    loss, _ = vpg_loss(scores, idx, values, reward, entropy_bonus=5e-3, reinforce_weight=1.0,
                       baseline_weight=0.0, normalize_advantage=True)
    loss.backward()
    assert net.amap.weight.grad.abs().max() > 0, "policy term must train the score map"
    assert net.vhead.weight.grad.abs().max() < 1e-6, net.vhead.weight.grad.abs().max()


def test_baseline_mse_trains_only_the_value_head():
    torch.manual_seed(0)
    B, T, A = 6, 2, 5
    net = _ToyScorer(A)
    x = torch.randn(B, T, 4)
    scores, values = net(x)
    idx = torch.randint(A, (B, T))
    reward = torch.randn(B)

    # reinforce_weight=0, entropy 0 => ONLY the MSE term.
    loss, _ = vpg_loss(scores, idx, values, reward, entropy_bonus=0.0, reinforce_weight=0.0,
                       baseline_weight=1.0, normalize_advantage=True)
    loss.backward()
    assert net.vhead.weight.grad.abs().max() > 0, "MSE must train the value head"
    assert net.amap.weight.grad is None or net.amap.weight.grad.abs().max() < 1e-6


def test_advantage_uses_a_detached_baseline():
    """autoreg: `advantage = R_exp - B_t.detach()`. If the baseline were live in the
    advantage, the policy term would push V to make the advantage large — a degenerate
    coupling. With reinforce-only weighting, vhead must get NO gradient at all."""
    torch.manual_seed(0)
    B, T, A = 6, 3, 4
    values = torch.randn(B, T, requires_grad=True)
    scores = torch.randn(B, T, A, requires_grad=True)
    loss, _ = vpg_loss(scores, torch.randint(A, (B, T)), values, torch.randn(B),
                       entropy_bonus=0.0, reinforce_weight=1.0, baseline_weight=0.0,
                       normalize_advantage=False)
    loss.backward()
    assert values.grad is None or values.grad.abs().max() < 1e-6


# --- 3./4. objective dispatch ------------------------------------------------------

def test_only_vpg_defers_credit():
    """`defers_credit` is the single switch the rollout branches on; if QReg/PG ever
    returned True their inline path (and the parity digest) would silently change."""
    from canvit_train.train.joint import JointPolicy

    def _jp(obj):
        return JointPolicy(
            policy_selector=None, random_selector=None, scorer=None, objective=obj,  # type: ignore[arg-type]
            rl_weight=1.0, keep_random_branch=False, target_momentum=0.997,
            device=torch.device("cpu"),
        )

    assert _jp(VPG()).defers_credit is True
    assert _jp(PG()).defers_credit is False
    assert _jp(QReg()).defers_credit is False


def test_vpg_defaults_are_autoreg_exp12s_values():
    """The recipe this port targets is
    jon_exp12_imagenet_rvm_fov_plus_rl_xareadout_nooverrides_bl: entropy 5e-3, reinforce
    and baseline weights 1.0, advantage normalization ON, reward NOT normalized (which is
    structural here — VPG has no reward normalizer at all)."""
    v = VPG()
    assert v.entropy_bonus == 5e-3
    assert v.reinforce_weight == 1.0
    assert v.baseline_weight == 1.0
    assert v.normalize_advantage is True
    assert v.value_bias_init is None
    assert not hasattr(v, "normalize_reward")


def test_config_exposes_vpg_and_keeps_the_old_default():
    from canvit_train.train.config import JointPolicyConfig

    c = JointPolicyConfig()
    assert c.objective == "qreg", "adding vpg must not change the default objective"
    assert c.vpg_entropy_bonus == 5e-3
    assert c.vpg_normalize_advantage is True


# --- the spec cell a terminal reward cannot honor ------------------------------------

def test_chunked_plus_coupled_policy_is_refused_for_vpg():
    """chunked task backward + policy_grad_to_backbone would free activations the deferred
    trajectory loss still needs. Caught at setup, not as an autograd error at step 0."""
    import pytest

    from canvit_train.harness.policy import check_credit_regime
    from canvit_train.harness.spec import BpttSpec, GroupOptim, TrainSpec
    from canvit_train.train.joint import JointPolicy

    def _jp(obj):
        return JointPolicy(
            policy_selector=None, random_selector=None, scorer=None, objective=obj,  # type: ignore[arg-type]
            rl_weight=1.0, keep_random_branch=False, target_momentum=0.997,
            device=torch.device("cpu"),
        )

    coupled_chunked = TrainSpec(
        train_backbone=True, train_policy=True, policy_weight=1.0,
        task_grad_to_backbone=True, policy_grad_to_backbone=True,
        bptt=BpttSpec(mode="chunked", chunk_size=2, horizon=4),
        optim={"backbone": GroupOptim(lr=1e-4), "policy": GroupOptim(lr=2e-4)},
    )
    with pytest.raises(ValueError, match="TERMINAL reward"):
        check_credit_regime(joint=_jp(VPG()), spec=coupled_chunked)

    # The same cell is fine for the inline objectives...
    check_credit_regime(joint=_jp(PG()), spec=coupled_chunked)
    check_credit_regime(joint=_jp(QReg()), spec=coupled_chunked)
    # ...and VPG is fine as soon as either half of the conflict is removed.
    from dataclasses import replace
    check_credit_regime(joint=_jp(VPG()),
                        spec=replace(coupled_chunked, bptt=BpttSpec(mode="full", horizon=4)))
    check_credit_regime(joint=_jp(VPG()),
                        spec=replace(coupled_chunked, policy_grad_to_backbone=False))
    check_credit_regime(joint=None, spec=coupled_chunked)


def test_legacy_distill_loop_refuses_vpg_rather_than_silently_building_pg():
    """train/joint.py::build_joint_policy dispatches on a string; before the guard,
    objective='vpg' fell through to `else: PG(...)` and the run would look healthy while
    training a different algorithm."""
    import pytest

    from canvit_train.train.config import FoveatedScaleConfig, JointPolicyConfig
    from canvit_train.train.joint import build_joint_policy

    with pytest.raises(NotImplementedError, match="harness"):
        build_joint_policy(
            core_model=None, rl=JointPolicyConfig(use_rl=True, objective="vpg"),
            device=torch.device("cpu"), canvas_grid=8, min_viewpoint_scale=0.05,
            foveated_scale=FoveatedScaleConfig(), generator=torch.Generator(),
        )
