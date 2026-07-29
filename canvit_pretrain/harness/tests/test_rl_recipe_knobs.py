"""The knobs that let the harness express the CanViT-PyTorch-RL recipe (doc 15 §A).

Each of these was a SILENT deviation: the harness ran, logged plausible numbers, and
simply trained under a different recipe than the gate-validated `ade20k/rl_train.py`.
Nothing surfaced them except reading both stacks side by side, which is why they are
pinned here against rl_train's own constants.

Two of the three fixes live in shared harness types (`GroupOptim.betas`, the policy
group's optimizer recipe in `resolve_spec`), so they apply to a policy on ANY task —
that is checked below rather than assumed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch import nn

from canvit_pretrain.harness.optim import build_optimizer_and_scheduler
from canvit_pretrain.harness.spec import GroupOptim, ScheduleSpec, TrainSpec
from canvit_pretrain.train.config import JointPolicyConfig

# The reference values, from canvit_pretrain/ade20k/rl_train.py's PolicyTrainConfig.
_RL_LR, _RL_WD = 2e-4, 1e-2
_RL_BETAS = (0.9, 0.95)
_RL_WARMUP_FRAC = 0.125


def test_policy_defaults_match_the_rl_recipe():
    pol = JointPolicyConfig()
    assert (pol.policy_lr, pol.policy_weight_decay) == (_RL_LR, _RL_WD)
    assert pol.policy_betas == _RL_BETAS, "beta2=0.95, NOT torch's 0.999"
    assert pol.policy_warmup_frac == _RL_WARMUP_FRAC


# --- GroupOptim.betas actually reaches the optimizer --------------------------
def test_betas_are_per_group_and_reach_adamw():
    """Before this field existed, every group silently got torch's (0.9, 0.999)."""
    spec = TrainSpec(
        train_backbone=True, train_head=True, train_policy=False,
        optim={"backbone": GroupOptim(lr=1e-3, betas=(0.9, 0.999)),
               "head": GroupOptim(lr=2e-4, betas=_RL_BETAS)},
    )
    opt, _ = build_optimizer_and_scheduler(
        spec, {"backbone": list(nn.Linear(2, 2).parameters()),
               "head": list(nn.Linear(2, 2).parameters())})
    by_lr = {g["lr"]: g["betas"] for g in opt.param_groups}
    assert by_lr[1e-3] == (0.9, 0.999)
    assert by_lr[2e-4] == _RL_BETAS, "per-group betas must not collapse to one value"


def test_default_betas_are_unchanged_for_existing_groups():
    """Task recipes keep torch's default, so no existing run moves."""
    assert GroupOptim(lr=1e-3).betas == (0.9, 0.999)


@pytest.mark.parametrize("bad", [(1.0, 0.95), (0.9, 1.0), (-0.1, 0.95)])
def test_invalid_betas_are_rejected(bad):
    assert GroupOptim(lr=1e-3, betas=bad).errors(group="policy")


# --- the policy group's schedule, on every task -------------------------------
@pytest.mark.parametrize("task_name", ["ade20k", "in1k"])
def test_policy_group_gets_the_rl_optimizer_recipe_on_every_task(task_name):
    """`resolve_spec` is shared, so this is one fix for all tasks — verified, not assumed.
    A bare GroupOptim would give the scorer torch's betas and warmup_steps=0 (no ramp)."""
    from dataclasses import replace

    from canvit_pretrain.harness.cli import resolve_spec

    if task_name == "ade20k":
        from canvit_pretrain.ade20k.config import Ade20kConfig
        from canvit_pretrain.tasks.ade20k.task import Ade20kRunTask
        task = Ade20kRunTask(replace(Ade20kConfig(), max_steps=8000))
    else:
        from canvit_pretrain.in1k.config import In1kConfig
        from canvit_pretrain.tasks.in1k.task import In1kRunTask
        task = In1kRunTask(replace(In1kConfig(), max_steps=8000), total_steps=8000)

    spec = resolve_spec(task, "policy_only", lr=1e-3, wd=1e-4)
    pol = spec.optim["policy"]
    assert pol.lr == _RL_LR and pol.weight_decay == _RL_WD
    assert pol.betas == _RL_BETAS
    assert pol.schedule.kind == "warmup_constant", "ramp then HOLD, as rl_train does"
    assert pol.schedule.warmup_steps == int(_RL_WARMUP_FRAC * 8000) == 1000


def test_distill_has_no_config_time_total_so_the_frac_warns_instead_of_vanishing(caplog):
    """Distill is SLURM-array-shaped (`steps_per_job`), so `policy_warmup_frac` has
    nothing to resolve against. The first version of this fix returned 0 there SILENTLY —
    i.e. reintroduced gap #2 on exactly the task the owner asked me to keep in mind."""
    import logging

    from canvit_pretrain.harness.cli import _policy_warmup_steps
    from canvit_pretrain.train.config import Config

    task = SimpleNamespace(cfg=Config(), name="distill")
    assert not hasattr(Config(), "max_steps"), "if distill gains max_steps, drop this test"
    with caplog.at_level(logging.WARNING):
        assert _policy_warmup_steps(task, JointPolicyConfig()) == 0
    assert "NO LR ramp" in caplog.text


def test_absolute_warmup_steps_win_and_need_no_total():
    from canvit_pretrain.harness.cli import _policy_warmup_steps
    from canvit_pretrain.train.config import Config

    task = SimpleNamespace(cfg=Config(), name="distill")
    pol = JointPolicyConfig(policy_warmup_steps=1000)
    assert _policy_warmup_steps(task, pol) == 1000  # the RL recipe's 0.125 * 8000


def test_policy_warmup_frac_zero_disables_the_ramp():
    from dataclasses import replace

    from canvit_pretrain.ade20k.config import Ade20kConfig
    from canvit_pretrain.harness.cli import resolve_spec
    from canvit_pretrain.tasks.ade20k.task import Ade20kRunTask

    task = Ade20kRunTask(replace(Ade20kConfig(), max_steps=8000),
                         rl=JointPolicyConfig(policy_warmup_frac=0.0))
    assert resolve_spec(task, "policy_only", lr=1e-3, wd=1e-4).optim["policy"].schedule.warmup_steps == 0


def test_the_ramp_actually_scales_the_lr():
    """warmup_constant must RAMP then HOLD — not decay, which is what a task schedule
    (onecycle/cosine) would have done to the scorer had it inherited one."""
    spec = TrainSpec(
        train_backbone=False, train_head=False, train_policy=True, task_weight=0.0,
        optim={"policy": GroupOptim(lr=_RL_LR, schedule=ScheduleSpec(
            kind="warmup_constant", warmup_steps=100))},
    )
    opt, sched = build_optimizer_and_scheduler(spec, {"policy": list(nn.Linear(2, 2).parameters())})
    lrs = []
    for _ in range(300):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] < lrs[50] < lrs[99] <= _RL_LR + 1e-12   # ramping
    assert lrs[150] == pytest.approx(_RL_LR)              # then held
    assert lrs[299] == pytest.approx(_RL_LR)


# --- augment: a shared INTERFACE, deliberately not shared code -----------------
def test_both_downstream_tasks_default_to_augmentation_on():
    """Every probe/finetune reference number was measured with it on."""
    from canvit_pretrain.ade20k.config import Ade20kConfig
    from canvit_pretrain.in1k.config import In1kConfig

    assert Ade20kConfig().augment is True
    assert In1kConfig().augment is True


def test_ade20k_augment_off_uses_the_val_transform_on_the_train_split():
    """The RL protocol. Asserted structurally because neutralising aug_scale_range /
    aug_flip_prob does NOT reproduce it — make_segmentation_train_transforms still
    applies RandomCropWithLabel and an unconditional PhotoMetricDistortion."""
    import inspect

    from canvit_pretrain.ade20k import data as ade_data

    src = inspect.getsource(ade_data.make_ade20k_loaders)
    assert "if cfg.augment:" in src
    # the off-branch must build the train dataset from the VAL transforms
    off_branch = src.split("else:", 1)[1]
    assert "img_transform=val_img_tf" in off_branch and "mask_transform=val_mask_tf" in off_branch
    assert 'split="training"' in off_branch


def test_in1k_augment_off_uses_the_val_preprocess():
    import inspect

    from canvit_pretrain.in1k import data as in1k_data

    src = inspect.getsource(in1k_data.make_train_loader)
    assert "cfg.augment" in src and "preprocess(cfg.scene_size)" in src


def test_the_five_recipe_constants_are_reachable_from_a_config():
    """Doc 15 §A's gap list, as a config-level check: everything the RL recipe needs can
    now be SET. (Whether a run then reproduces the band is a cluster question.)"""
    from dataclasses import replace

    from canvit_pretrain.ade20k.config import Ade20kConfig

    cfg = replace(Ade20kConfig(), augment=False, max_steps=8000, n_timesteps=5,
                  resize_mode="squish", batch_size=16)
    assert cfg.augment is False           # gap 3
    assert cfg.max_steps == 8000          # gap 4  (640k forwards / (16 * (1+4)))
    assert cfg.n_timesteps == 5           # rl_train train_horizon=4 -> t0 + 4
    pol = JointPolicyConfig()
    assert pol.policy_betas == _RL_BETAS  # gap 1
    assert pol.policy_warmup_frac > 0     # gap 2


# --- gap #6: the probe head (found 2026-07-29, blocking) ----------------------
def test_frozen_mode_honours_probe_repo():
    """`--preset policy_only` runs in FROZEN mode. Until 2026-07-29 `build_model` only
    loaded a published probe for `mode=="finetune"`, so a policy run silently trained its
    scorer against a RANDOM head — and the reward IS the probe's CE reduction, so that
    reward was noise. Asserted structurally: loading a real probe needs the network."""
    import inspect
    from dataclasses import replace

    from canvit_pretrain.ade20k.config import Ade20kConfig
    from canvit_pretrain.tasks.ade20k.task import Ade20kRunTask

    src = inspect.getsource(Ade20kRunTask.build_model)
    assert "if self.cfg.probe_repo:" in src, "must not be gated on mode any more"
    assert 'mode == "finetune" and self.cfg.probe_repo' not in src

    cfg = replace(Ade20kConfig(), mode="frozen", probe_repo="canvit/probe-x")
    assert cfg.probe_repo and cfg.mode == "frozen"  # the combination is now meaningful


def test_policy_without_a_probe_warns(caplog):
    """The failure is silent by nature, so it has to be announced."""
    import logging
    from dataclasses import replace

    import torch

    from canvit_pretrain.ade20k.config import Ade20kConfig
    from canvit_pretrain.tasks.ade20k.task import Ade20kRunTask

    task = Ade20kRunTask(replace(Ade20kConfig(), mode="frozen", probe_repo=None, canvas_grid=8))
    with caplog.at_level(logging.WARNING):
        try:
            task.build_policy(SimpleNamespace(cfg=SimpleNamespace(canvas_dim=8, patcher_name="uniform")),
                              device=torch.device("cpu"), canvas_grid=8,
                              generator=torch.Generator())
        except Exception:
            pass  # model stub is not buildable; the warning fires before that
    assert "reward is noise" in caplog.text
