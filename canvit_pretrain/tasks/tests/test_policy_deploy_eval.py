"""``eval_policy="policy"`` end to end, on real (tiny) models.

The bug this closes was silent, not loud: a ``--preset policy_only`` run trained a
scorer and then validated it on RANDOM glimpses, so every logged ``eval/miou_t*`` — and
the ``best.pt`` selection built on it — described a trajectory the policy never chose.
Nothing crashed; the numbers were simply about something else.

So the load-bearing assertion here is ``task.best_metric in task.evaluate(...)``: the
key the harness selects on must actually be produced by the eval that ran. Plus the
mechanical checks that the deploy path executes at all, and that switching to it really
changes the trajectory rather than quietly falling back.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from canvit_pytorch import CanViTForImageClassification, CanViTForSemanticSegmentation

from canvit_pretrain.ade20k.config import Ade20kConfig
from canvit_pretrain.ade20k.data import IGNORE_LABEL, NUM_CLASSES
from canvit_pretrain.harness.policy import build_policy
from canvit_pretrain.in1k.config import In1kConfig
from canvit_pretrain.tasks.ade20k.task import POLICY_FEATURE_GROUPS as ADE_GROUPS
from canvit_pretrain.tasks.ade20k.task import Ade20kRunTask
from canvit_pretrain.tasks.in1k.task import POLICY_FEATURE_GROUPS as IN1K_GROUPS
from canvit_pretrain.tasks.in1k.task import In1kRunTask
from canvit_pretrain.train.config import FoveatedScaleConfig, JointPolicyConfig

_B, _G, _IMG, _T = 2, 8, 224, 3
_DEV = torch.device("cpu")


def _joint_for(*, canvit, encode_model, groups):
    gen = torch.Generator(device=_DEV)
    gen.manual_seed(0)
    return build_policy(
        canvit=canvit, rl=JointPolicyConfig(use_rl=True, objective="qreg"), feature_groups=groups,
        device=_DEV, canvas_grid=_G, min_viewpoint_scale=0.05, foveated_scale=FoveatedScaleConfig(),
        generator=gen, encode_model=encode_model,
    )


def _ade_task(eval_policy: str) -> tuple[Ade20kRunTask, CanViTForSemanticSegmentation, list]:
    torch.manual_seed(0)
    seg = CanViTForSemanticSegmentation(
        backbone_name="vits16", model_config={}, num_classes=NUM_CLASSES).to(_DEV)
    seg.canvit.requires_grad_(False)
    cfg = replace(Ade20kConfig(), eval_policy=eval_policy, n_timesteps=_T, canvas_grid=_G,
                  viz_every=0, limit_val_batches=1)
    masks = torch.randint(0, NUM_CLASSES, (_B, _IMG, _IMG))
    masks[:, :8] = IGNORE_LABEL
    loader = [(torch.randn(_B, 3, _IMG, _IMG), masks)]
    return Ade20kRunTask(cfg), seg, loader


def test_ade20k_policy_deploy_eval_produces_its_own_selection_metric():
    task, seg, loader = _ade_task("policy")
    joint = _joint_for(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS)

    out = task.evaluate(model=seg, head=seg.head, val_loader=loader, device=_DEV, step=0,
                        joint=joint)

    assert task.best_metric == "neg_ce_mean"
    assert task.best_metric in out, (
        "the harness selects best.pt on this key; if evaluate() does not emit it, "
        "best.pt silently never updates")
    assert out["ce_mean"] > 0 and out["neg_ce_mean"] == -out["ce_mean"]
    # both axes are reported, so a policy run stays comparable to a probe run
    assert {f"ce_t{t}" for t in range(_T)} <= set(out)
    assert {f"miou_t{t}" for t in range(_T)} <= set(out)
    for v in out.values():
        assert torch.isfinite(torch.tensor(float(v)))


def test_ade20k_default_eval_is_untouched_and_selects_on_miou():
    task, seg, loader = _ade_task("auto")
    out = task.evaluate(model=seg, head=seg.head, val_loader=loader, device=_DEV, step=0)

    assert task.best_metric == "miou_final"
    assert task.best_metric in out
    # the probe path must not start paying for the policy path's full-res CE
    assert not any(k.startswith("ce_") for k in out), out


def test_ade20k_policy_deploy_actually_changes_the_trajectory():
    """Guards the failure this whole change is about: if 'policy' silently fell back to
    the random trajectory, every metric would still look plausible."""
    from canvit_pretrain.harness.eval_viewpoints import deploy_rollout_viewpoints, open_loop_viewpoints
    from canvit_pretrain.train.viewpoint import ViewpointType

    task, seg, _ = _ade_task("policy")
    joint = _joint_for(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS)
    images = torch.randn(_B, 3, _IMG, _IMG)

    def advance(vp, state, t):
        if state is None:
            state = seg.init_state(batch_size=_B, canvas_grid_size=_G)
        with torch.no_grad():
            return seg.canvit(image=images, state=state, viewpoint=vp).state

    torch.manual_seed(0)
    chosen = deploy_rollout_viewpoints(joint=joint, advance=advance, t0_type=ViewpointType.FULL,
                                       batch_size=_B, device=_DEV, n=_T)
    torch.manual_seed(0)
    random_traj = open_loop_viewpoints("random", batch_size=_B, device=_DEV, n=_T,
                                       is_foveated=False, foveated_scale=FoveatedScaleConfig())

    assert len(chosen) == _T
    # t0 is the shared FULL anchor; the glimpses AFTER it must come from the scorer.
    assert not torch.allclose(chosen[1].centers, random_traj[1].centers)
    # and the scorer picks from its discrete candidate table, not the continuous law
    table = joint.policy_selector.vp_flat
    for vp in chosen[1:]:
        for b in range(_B):
            assert (table[:, :2] == vp.centers[b]).all(dim=1).any(), "off-grid candidate"


def test_distill_deploy_viewpoints_runs_closed_loop():
    """Distill validates through core's ``forward_reduce``, which takes a precomputed
    list, so its policy path selects first and replays. This covers that selection."""
    from canvit_pytorch import create_backbone

    from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig
    from canvit_pretrain.tasks.distill.task import POLICY_FEATURE_GROUPS as DISTILL_GROUPS
    from canvit_pretrain.train.viz.validate import _deploy_viewpoints

    torch.manual_seed(0)
    model = CanViTForPretraining(
        backbone=create_backbone("vits16"), cfg=CanViTForPretrainingConfig(teacher_dim=32),
        glimpse_size_px=128, backbone_name="vits16", canvas_patch_grid_sizes=[_G],
    ).to(_DEV)
    model.eval()
    joint = _joint_for(canvit=model, encode_model=None, groups=DISTILL_GROUPS)

    with torch.no_grad():
        vps = _deploy_viewpoints(model=model, images=torch.randn(_B, 3, _IMG, _IMG),
                                 joint=joint, n=_T, canvas_grid_size=_G)

    assert len(vps) == _T
    table = joint.policy_selector.vp_flat
    for vp in vps[1:]:  # t0 is the FULL anchor; the rest come from the candidate table
        for b in range(_B):
            assert (table[:, :2] == vp.centers[b]).all(dim=1).any(), "off-grid candidate"


def test_in1k_policy_deploy_eval_runs():
    torch.manual_seed(0)
    clf = CanViTForImageClassification(
        backbone_name="vits16", model_config={}, n_classes=10, glimpse_grid_size=_G).to(_DEV)
    clf.canvit.requires_grad_(False)
    cfg = replace(In1kConfig(), eval_policy="policy", n_timesteps=_T, canvas_grid=_G,
                  limit_val_batches=1, eval_batch_size=_B)
    task = In1kRunTask(cfg, total_steps=1)
    joint = _joint_for(canvit=clf.canvit, encode_model=clf, groups=IN1K_GROUPS)
    loader = [(torch.randn(_B, 3, _IMG, _IMG), torch.randint(0, 10, (_B,)))]

    out = task.evaluate(model=clf, head=clf.head, val_loader=loader, device=_DEV, step=0,
                        joint=joint)

    assert task.best_metric in out
    assert 0.0 <= out["top1"] <= 1.0 and 0.0 <= out["top5"] <= 1.0
