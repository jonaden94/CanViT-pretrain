"""The shared validation-viewpoint knob.

The load-bearing claim of this module is NOT that the options work — it is that
unifying them changed nothing. Every task keeps the trajectory it had, so every
existing config, every specialize probe number and every exp22/23/24/26 val curve
stays comparable. That is what most of these tests pin, by running the OLD generator
and the new dispatch under the same seed and demanding identical tensors.

The rest cover the genuinely new capability: ``"policy"``, i.e. deploying a trained
scorer by argmax instead of replaying a fixed trajectory.
"""

from __future__ import annotations

import pytest
import torch

from canvit_train.harness.eval_viewpoints import (
    HISTORICAL_DEFAULTS,
    OPEN_LOOP,
    deploy_rollout_viewpoints,
    deploy_selector,
    open_loop_viewpoints,
    resolve,
)
from canvit_train.train.config import FoveatedScaleConfig

_B, _N, _DEV = 3, 5, torch.device("cpu")
_FS = FoveatedScaleConfig()


# --- the defaults did not move ----------------------------------------------
@pytest.mark.parametrize("task,is_fov,expected", [
    ("distill", False, "coarse_to_fine"),
    ("distill", True, "fixation_grid"),   # quadtree scales are OOD for fixed-scale foveated
    ("ade20k", False, "random"),          # inherited from the specialize probe
    ("ade20k", True, "random"),
    ("in1k", False, "coarse_to_fine"),    # canvit_eval's deploy convention
    ("in1k", True, "coarse_to_fine"),     # the known footgun, kept for exp25 comparability
])
def test_auto_resolves_to_each_tasks_historical_trajectory(task, is_fov, expected):
    assert resolve("auto", task=task, is_foveated=is_fov) == expected


def test_every_task_config_still_defaults_to_auto():
    """A task whose config drifted off "auto" would silently change its own history."""
    from canvit_train.ade20k.config import Ade20kConfig
    from canvit_train.in1k.config import In1kConfig
    from canvit_train.train.config import Config

    assert Config().eval_policy == "auto"
    assert Ade20kConfig().eval_policy == "auto"
    assert In1kConfig().eval_policy == "auto"


def test_distill_uniform_is_bit_identical_to_the_old_generator():
    from canvit_train.train.viewpoint import make_eval_viewpoints

    torch.manual_seed(0)
    old = make_eval_viewpoints(_B, _DEV, n_viewpoints=_N)
    torch.manual_seed(0)
    new = open_loop_viewpoints("coarse_to_fine", batch_size=_B, device=_DEV, n=_N,
                               is_foveated=False, foveated_scale=_FS)
    _assert_same_trajectory(old, new)


def test_distill_foveated_is_bit_identical_to_the_old_generator():
    from canvit_train.train.viewpoint import make_eval_viewpoints_foveated

    old = make_eval_viewpoints_foveated(_B, _DEV, n_viewpoints=_N, scale=2.0)
    new = open_loop_viewpoints("fixation_grid", batch_size=_B, device=_DEV, n=_N,
                               is_foveated=True, foveated_scale=_FS, foveated_eval_scale=2.0)
    _assert_same_trajectory(old, new)
    # the scale really is carried through — this is the whole point of fixation_grid
    assert torch.allclose(new[0].scales, torch.full((_B,), 2.0))


def test_ade20k_random_is_bit_identical_to_the_old_generator():
    from canvit_train.ade20k.rollout import make_random_viewpoints

    torch.manual_seed(0)
    old = make_random_viewpoints(_B, _DEV, _N, min_scale=0.05, max_scale=1.0,
                                 start_with_full_scene=True, is_foveated=False,
                                 foveated_scale=_FS)
    torch.manual_seed(0)
    new = open_loop_viewpoints("random", batch_size=_B, device=_DEV, n=_N,
                               is_foveated=False, foveated_scale=_FS)
    _assert_same_trajectory(old, new)


def _assert_same_trajectory(old, new):
    assert len(old) == len(new)
    for i, (a, b) in enumerate(zip(old, new)):
        torch.testing.assert_close(a.centers, b.centers, rtol=0, atol=0, msg=f"centers t{i}")
        torch.testing.assert_close(a.scales, b.scales, rtol=0, atol=0, msg=f"scales t{i}")


# --- the option set is the same for every task -------------------------------
@pytest.mark.parametrize("policy", OPEN_LOOP)
def test_every_open_loop_option_is_usable_by_every_task(policy):
    """The point of the unification: no option belongs to one task any more."""
    vps = open_loop_viewpoints(policy, batch_size=_B, device=_DEV, n=_N,
                               is_foveated=False, foveated_scale=_FS)
    assert len(vps) == _N
    for vp in vps:
        assert vp.centers.shape == (_B, 2) and vp.scales.shape == (_B,)


def test_policy_cannot_be_precomputed():
    with pytest.raises(ValueError, match="closed-loop"):
        open_loop_viewpoints("policy", batch_size=_B, device=_DEV, n=_N,
                             is_foveated=False, foveated_scale=_FS)


def test_unknown_policy_is_rejected_at_resolve():
    with pytest.raises(AssertionError, match="unknown eval policy"):
        resolve("greedy_oracle", task="ade20k", is_foveated=False)


def test_every_task_in_the_table_covers_both_patchers():
    for task, entry in HISTORICAL_DEFAULTS.items():
        assert len(entry) == 2, task
        for value in entry:
            assert value in OPEN_LOOP, f"{task}: {value} is not an open-loop default"


# --- deploying a trained scorer ----------------------------------------------
def test_deploy_without_a_policy_says_so():
    """The failure mode this replaces is silent: validating a policy run on random
    glimpses. An explicit error is the whole improvement."""
    with pytest.raises(AssertionError, match="needs a trained scorer"):
        deploy_selector(None)


def test_deploy_forces_pure_argmax():
    """Deployment must not explore, whatever the trainer's current epsilon is: a
    validation that took random glimpses would not measure the deployed policy, and
    drawing epsilon would consume RNG that training also uses."""
    joint = _tiny_joint(prime_on_policy=0.25)
    sel = deploy_selector(joint)
    assert sel.mode == "argmax" and sel.prime_on_policy == 1.0
    # ... and the trainer's own selector is untouched (deploy_selector copies).
    assert joint.policy_selector.prime_on_policy == 0.25


def test_deploy_requires_a_full_t0_anchor():
    from canvit_train.train.viewpoint import ViewpointType

    with pytest.raises(AssertionError, match="FULL t0 anchor"):
        deploy_rollout_viewpoints(joint=_tiny_joint(), advance=lambda vp, st, t: st,
                                  t0_type=ViewpointType.RANDOM, batch_size=_B,
                                  device=_DEV, n=_N)


def test_deploy_rollout_is_closed_loop_and_restores_train_mode():
    """Each glimpse must be chosen from the state the previous one produced (that is
    what 'closed loop' means), and a validation must not leave the scorer in eval mode
    — the next training step would then update no BatchNorm statistics."""
    from canvit_train.train.viewpoint import ViewpointType

    joint = _tiny_joint()
    joint.scorer.train()
    seen_states, advanced = [], []

    def advance(vp, state, t):
        seen_states.append(state)
        advanced.append(vp)
        return _FakeState(t)

    vps = deploy_rollout_viewpoints(joint=joint, advance=advance, t0_type=ViewpointType.FULL,
                                    batch_size=_B, device=_DEV, n=4)
    assert len(vps) == 4 and len(advanced) == 4
    assert seen_states[0] is None, "t0 must hand the task a None state to initialise"
    # every later selection saw the state the previous glimpse returned
    assert [s.t for s in seen_states[1:]] == [0, 1, 2]
    assert joint.scorer.training, "scorer left in eval mode would freeze its BatchNorm"


class _FakeState:
    def __init__(self, t):
        self.t = t


def _tiny_joint(prime_on_policy: float = 1.0):
    """A JointPolicy small enough to run on CPU: 2 candidates, a scorer that reads a
    stub encoder. Only the selection path is exercised here."""
    from dataclasses import dataclass

    import torch.nn as nn

    from canvit_train.train.config import FoveatedScaleConfig
    from canvit_train.train.selector import PolicySelector, RandomSelector

    class _Scorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([[1.0, -1.0]]))

        def forward(self, feats):
            return self.w.expand(feats.shape[0], -1)

    class _Encoder:
        def reset(self):
            pass

        def __call__(self, state):
            return torch.zeros(_B, 1)

    scorer = _Scorer()
    sel = PolicySelector(
        net=scorer, encoder=_Encoder(),
        vp_flat=torch.tensor([[0.0, 0.0, 0.5], [0.5, 0.5, 0.5]]),
        fallback=RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(),
                                min_viewpoint_scale=0.05),
        prime_on_policy=prime_on_policy,
    )

    @dataclass
    class _Joint:
        policy_selector: PolicySelector
        scorer: nn.Module

    return _Joint(policy_selector=sel, scorer=scorer)


# --- selection metric follows the eval policy --------------------------------
@pytest.mark.parametrize("eval_policy,expected", [
    ("auto", "miou_final"),
    ("random", "miou_final"),
    ("policy", "neg_ce_mean"),
])
def test_ade20k_best_metric_follows_the_eval_policy(eval_policy, expected):
    """Selecting a POLICY run on mIoU would put our checkpoints on a different axis
    from the qband reference band, which is defined on mean t1..tH CE."""
    from dataclasses import replace

    from canvit_train.ade20k.config import Ade20kConfig
    from canvit_train.tasks.ade20k.task import Ade20kRunTask

    task = Ade20kRunTask(replace(Ade20kConfig(), eval_policy=eval_policy))
    assert task.best_metric == expected
