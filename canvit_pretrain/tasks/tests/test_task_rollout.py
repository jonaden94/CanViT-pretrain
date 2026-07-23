"""Smoke tests: all three peer tasks drive the unified rollout engine, and the
TrainSpec grad regime routes gradients to exactly the right modules (design §5).

This is the concrete payoff of the harness — one engine, three tasks, the same
freeze/finetune knobs — validated on tiny CPU models. Joint task+policy per task
(the crown-jewel new capability) needs the per-task policy builder and is smoke-
tested once that lands (loop phase); here we cover the task-only cells.
"""

import torch
from canvit_pytorch import (
    CanViTForImageClassification,
    CanViTForSemanticSegmentation,
)

from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig
from canvit_pretrain.ade20k.data import IGNORE_LABEL, NUM_CLASSES
from canvit_pretrain.harness.policy import build_policy
from canvit_pretrain.harness.rollout import run_rollout
from canvit_pretrain.harness.spec import BpttSpec
from canvit_pretrain.tasks.ade20k.task import POLICY_FEATURE_GROUPS as ADE_GROUPS
from canvit_pretrain.tasks.ade20k.task import BoundAde20kTask
from canvit_pretrain.tasks.distill.task import BoundDistillTask
from canvit_pretrain.tasks.in1k.task import POLICY_FEATURE_GROUPS as IN1K_GROUPS
from canvit_pretrain.tasks.in1k.task import BoundIn1kTask
from canvit_pretrain.train.config import FoveatedScaleConfig, JointPolicyConfig
from canvit_pretrain.train.joint import build_joint_policy
from canvit_pretrain.train.selector import RandomSelector
from canvit_pretrain.train.task import DistillTask
from canvit_pretrain.train.viewpoint import ViewpointType

_B, _G, _IMG, _D, _C = 2, 8, 224, 384, 10
_DEV = torch.device("cpu")


def _selector() -> RandomSelector:
    return RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05)


def _zero_grads(*modules):
    for m in modules:
        for p in m.parameters():
            p.grad = None


def _has_grad(module) -> bool:
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def _no_grad(module) -> bool:
    return all(p.grad is None for p in module.parameters())


# --------------------------------------------------------------------------- #
# Distill (finetune only — its heads are inside the forward, no probe cell).
# --------------------------------------------------------------------------- #
def _distill_model() -> CanViTForPretraining:
    torch.manual_seed(0)
    from canvit_pytorch import create_backbone
    return CanViTForPretraining(
        backbone=create_backbone("vits16"), cfg=CanViTForPretrainingConfig(teacher_dim=_D),
        glimpse_size_px=128, backbone_name="vits16", canvas_patch_grid_sizes=[_G],
    ).to(_DEV)


def test_distill_finetune_trains_backbone():
    torch.manual_seed(1)
    model = _distill_model()
    task = BoundDistillTask(DistillTask(
        scene_target=torch.randn(_B, _G * _G, _D), cls_target=torch.randn(_B, _D),
        enable_scene_patches_loss=True, enable_scene_cls_loss=True,
    ))
    _zero_grads(model)
    r = run_rollout(
        model=model, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="full", horizon=2), branches=[ViewpointType.FULL, ViewpointType.RANDOM],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(),
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(model.backbone)


# --------------------------------------------------------------------------- #
# ADE20K (probe: frozen backbone / trains head ; finetune: trains backbone).
# --------------------------------------------------------------------------- #
def _seg() -> CanViTForSemanticSegmentation:
    torch.manual_seed(0)
    return CanViTForSemanticSegmentation(backbone_name="vits16", model_config={}, num_classes=NUM_CLASSES).to(_DEV)


def _seg_masks() -> torch.Tensor:
    m = torch.randint(0, NUM_CLASSES, (_B, _IMG, _IMG), device=_DEV)
    m[:, :8] = IGNORE_LABEL
    return m


def test_ade20k_probe_freezes_backbone_trains_head():
    torch.manual_seed(1)
    seg = _seg()
    seg.canvit.requires_grad_(False)
    seg.canvit.eval()
    task = BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G)
    _zero_grads(seg)
    r = run_rollout(
        model=seg, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="none", horizon=2), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(),
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(seg.head)
    assert _no_grad(seg.canvit)


def test_ade20k_finetune_trains_backbone():
    torch.manual_seed(1)
    seg = _seg()
    task = BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G)
    _zero_grads(seg)
    r = run_rollout(
        model=seg, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="full", horizon=2), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(),
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(seg.canvit)  # NEW capability: ADE20K full-model finetune


# --------------------------------------------------------------------------- #
# IN1k (frozen probe / finetune).
# --------------------------------------------------------------------------- #
def _clf() -> CanViTForImageClassification:
    torch.manual_seed(0)
    return CanViTForImageClassification(
        backbone_name="vits16", model_config={}, n_classes=_C, glimpse_grid_size=_G,
    ).to(_DEV)


def test_in1k_frozen_trains_head_only():
    torch.manual_seed(1)
    clf = _clf()
    clf.canvit.requires_grad_(False)
    clf.canvit.eval()
    task = BoundIn1kTask(clf=clf, targets=torch.randint(0, _C, (_B,)), canvas_grid=_G)
    _zero_grads(clf)
    r = run_rollout(
        model=clf, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="none", horizon=2), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(),
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(clf.head)
    assert _no_grad(clf.canvit)


def test_in1k_finetune_trains_backbone():
    torch.manual_seed(1)
    clf = _clf()
    task = BoundIn1kTask(clf=clf, targets=torch.randint(0, _C, (_B,)), canvas_grid=_G)
    _zero_grads(clf)
    r = run_rollout(
        model=clf, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="full", horizon=2), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(),
    )
    assert torch.isfinite(r.total_loss)
    assert _has_grad(clf.canvit)  # NEW capability: IN1k policy-free finetune via the unified engine


# --------------------------------------------------------------------------- #
# Joint task+policy through the unified engine (the P4b mechanism). Distill here
# (build_joint_policy fully supports it); per-task joint for ade20k/in1k needs the
# probe-aware policy builder (loop phase).
# --------------------------------------------------------------------------- #
def test_distill_joint_trains_task_and_scorer():
    torch.manual_seed(1)
    model = _distill_model()
    gen = torch.Generator(device=_DEV)
    gen.manual_seed(0)
    joint = build_joint_policy(
        core_model=model, rl=JointPolicyConfig(use_rl=True, objective="qreg"), device=_DEV,
        canvas_grid=_G, min_viewpoint_scale=0.05, foveated_scale=FoveatedScaleConfig(), generator=gen,
    )
    task = BoundDistillTask(DistillTask(
        scene_target=torch.randn(_B, _G * _G, _D), cls_target=torch.randn(_B, _D),
        enable_scene_patches_loss=True, enable_scene_cls_loss=True,
    ))
    _zero_grads(model, joint.scorer)
    r = run_rollout(
        model=model, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="chunked", chunk_size=2, horizon=4), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint,
    )
    assert torch.isfinite(r.total_loss)
    assert r.policy_metrics is not None and "reward_frac" in r.policy_metrics
    assert _has_grad(joint.scorer)   # policy loss trained the scorer
    assert _has_grad(model.backbone)  # task (distill) loss still trained the backbone


def _joint_for(*, canvit, encode_model, groups):
    gen = torch.Generator(device=_DEV)
    gen.manual_seed(0)
    return build_policy(
        canvit=canvit, rl=JointPolicyConfig(use_rl=True, objective="qreg"), feature_groups=groups,
        device=_DEV, canvas_grid=_G, min_viewpoint_scale=0.05, foveated_scale=FoveatedScaleConfig(),
        generator=gen, encode_model=encode_model,
    )


def test_ade20k_joint_trains_probe_and_scorer():
    # CROWN JEWEL: ADE20K joint task+policy (frozen backbone, train probe + policy) —
    # a capability the old ade20k trainer never had. Probe-aware scorer (ent features).
    torch.manual_seed(1)
    seg = _seg()
    seg.canvit.requires_grad_(False)
    seg.canvit.eval()
    joint = _joint_for(canvit=seg.canvit, encode_model=seg, groups=ADE_GROUPS)
    task = BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G)
    _zero_grads(seg, joint.scorer)
    r = run_rollout(
        model=seg, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="none", horizon=4), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint,
    )
    assert torch.isfinite(r.total_loss)
    assert r.policy_metrics is not None
    assert _has_grad(joint.scorer)   # policy loss trained the scorer
    assert _has_grad(seg.head)       # task CE trained the probe head
    assert _no_grad(seg.canvit)      # backbone stayed frozen


def test_in1k_joint_trains_head_and_scorer():
    # IN1k joint task+policy (frozen backbone, train head + policy); intrinsic scorer feats.
    torch.manual_seed(1)
    clf = _clf()
    clf.canvit.requires_grad_(False)
    clf.canvit.eval()
    joint = _joint_for(canvit=clf.canvit, encode_model=clf, groups=IN1K_GROUPS)
    task = BoundIn1kTask(clf=clf, targets=torch.randint(0, _C, (_B,)), canvas_grid=_G)
    _zero_grads(clf, joint.scorer)
    r = run_rollout(
        model=clf, images=torch.randn(_B, 3, _IMG, _IMG), task=task, selector=_selector(),
        bptt=BpttSpec(mode="none", horizon=4), branches=[ViewpointType.FULL],
        canvas_grid_size=_G, amp_ctx=torch.enable_grad(), joint=joint,
    )
    assert torch.isfinite(r.total_loss)
    assert r.policy_metrics is not None
    assert _has_grad(joint.scorer)
    assert _has_grad(clf.head)
    assert _no_grad(clf.canvit)
