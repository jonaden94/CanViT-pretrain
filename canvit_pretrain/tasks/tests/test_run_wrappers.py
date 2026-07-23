"""CPU unit tests for the run-level Task wrappers + the ``harness.run`` CLI glue.

Covers the pure-config surface (caps / default_spec / branches / feature groups /
RunTask protocol conformance), the trainable-param-group routing on tiny CPU models
(no HF download), the IN1k head=norm+head wrinkle, and the CLI ``_build_task`` /
``_resolve_spec`` preset matrix (head-aware). The model-loading + real-data training
path is covered by the GPU integration script (``unification_docs/harness_run_integration.py``).
"""

import torch
from canvit_pytorch import (
    CanViTForImageClassification,
    CanViTForSemanticSegmentation,
    create_backbone,
)

from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig
from canvit_pretrain.ade20k.config import Ade20kConfig
from canvit_pretrain.ade20k.data import NUM_CLASSES as ADE_CLASSES
from canvit_pretrain.harness.loop import apply_requires_grad
from canvit_pretrain.harness.run import RunTask, _build_task, _resolve_spec
from canvit_pretrain.harness.spec import TrainSpec
from canvit_pretrain.in1k.config import In1kConfig
from canvit_pretrain.tasks.ade20k.task import Ade20kRunTask
from canvit_pretrain.tasks.distill.task import DistillRunTask
from canvit_pretrain.tasks.in1k.task import In1kRunTask
from canvit_pretrain.train.config import Config

_G, _D, _C = 8, 384, 10


def _distill_cfg():
    return Config(webdataset_dir="/nonexistent", canvas_patch_grid_size=_G)


def _tiny_distill():
    return CanViTForPretraining(
        backbone=create_backbone("vits16"), cfg=CanViTForPretrainingConfig(teacher_dim=_D),
        glimpse_size_px=128, backbone_name="vits16", canvas_patch_grid_sizes=[_G],
    )


def _tiny_seg():
    return CanViTForSemanticSegmentation(backbone_name="vits16", model_config={}, num_classes=ADE_CLASSES)


def _tiny_clf():
    return CanViTForImageClassification(backbone_name="vits16", model_config={}, n_classes=_C, glimpse_grid_size=_G)


def _wrappers():
    return [
        Ade20kRunTask(Ade20kConfig(tracker="none")),
        In1kRunTask(In1kConfig(tracker="none")),
        DistillRunTask(_distill_cfg()),
    ]


# --- pure-config surface ---------------------------------------------------
def test_wrappers_satisfy_runtask_protocol():
    for t in _wrappers():
        assert isinstance(t, RunTask), t.name


def test_default_specs_validate():
    for t in _wrappers():
        spec = t.default_spec()
        spec.validate(t.caps())  # raises on incoherent spec
        # every trainable module has an optimizer group
        for m in spec.trainable_modules():
            assert m in spec.optim, (t.name, m)


def test_caps_and_feature_groups():
    a, i, d = _wrappers()
    assert a.caps().has_head and i.caps().has_head
    assert not d.caps().has_head  # distill heads live in the forward
    # ade20k is the only probe-aware (spatial-entropy) feature set
    assert "ent" in a.policy_feature_groups() or "ent_delta" in a.policy_feature_groups()
    assert "ent" not in i.policy_feature_groups()
    assert "ent" not in d.policy_feature_groups()


def test_branches_default():
    a, i, d = _wrappers()
    assert len(a.branches()) == 1 and len(i.branches()) == 1
    assert len(d.branches()) == 2  # 1 full + 1 random by default


# --- trainable param-group routing (tiny CPU models) ----------------------
def test_ade20k_param_groups_probe_vs_finetune():
    t = Ade20kRunTask(Ade20kConfig(tracker="none"))
    seg = _tiny_seg()
    probe = t.default_spec()  # frozen backbone, train head
    apply_requires_grad(model=seg, head=seg.head, joint=None, spec=probe)
    g = t.trainable_param_groups(model=seg, head=seg.head, joint=None, spec=probe)
    assert set(g) == {"head"}
    assert all(not p.requires_grad for p in seg.canvit.parameters())
    assert all(p.requires_grad for p in seg.head.parameters())

    ft = TrainSpec.finetune(optim=probe.optim | {"backbone": probe.optim["head"]})
    apply_requires_grad(model=seg, head=seg.head, joint=None, spec=ft)
    g = t.trainable_param_groups(model=seg, head=seg.head, joint=None, spec=ft)
    assert set(g) == {"backbone", "head"}
    assert all(p.requires_grad for p in seg.canvit.parameters())


def test_in1k_head_group_is_norm_plus_head():
    t = In1kRunTask(In1kConfig(tracker="none"))
    clf = _tiny_clf()
    spec = t.default_spec()
    g = t.trainable_param_groups(model=clf, head=clf.head, joint=None, spec=spec)
    head_ids = {id(p) for p in g["head"]}
    assert head_ids == {id(p) for p in list(clf.norm.parameters()) + list(clf.head.parameters())}
    assert head_ids  # non-empty


def test_distill_param_group_is_whole_model():
    t = DistillRunTask(_distill_cfg())
    model = _tiny_distill()
    spec = t.default_spec()
    g = t.trainable_param_groups(model=model, head=None, joint=None, spec=spec)
    assert set(g) == {"backbone"}
    assert len(g["backbone"]) == len(list(model.parameters()))


# --- CLI glue: _build_task + _resolve_spec preset matrix -------------------
def test_cli_preset_matrix_head_aware():
    ade = _build_task("ade20k", {})
    in1k = _build_task("in1k", {})
    distill = _build_task("distill", {"webdataset_dir": "/x"})
    # head-bearing tasks: all presets that make sense validate
    for t in (ade, in1k):
        for preset in ("default", "probe", "finetune", "policy_only", "joint"):
            _resolve_spec(t, preset, None, None).validate(t.caps())
    # headless distill: train_head is dropped, so finetune/joint still validate
    for preset in ("default", "finetune", "policy_only", "joint"):
        spec = _resolve_spec(distill, preset, None, None)
        spec.validate(distill.caps())
        assert not spec.train_head


def test_cli_overrides_apply():
    t = _build_task("ade20k", {"batch_size": 3, "n_timesteps": 7, "lr": 1e-2})
    assert t.cfg.batch_size == 3 and t.cfg.n_timesteps == 7 and t.cfg.peak_lr == 1e-2
