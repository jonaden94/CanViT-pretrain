"""CPU tests for the checkpoint → HF-format converter's pure logic."""

from pathlib import Path

import pytest

from .to_hf import build_config, extract_pretrain_view_scale, normalize_schema


def _raw(patcher: str, history: dict | None) -> dict:
    return {
        "backbone_name": "vitb16",
        "model_config": {"patcher_name": patcher, "teacher_dim": 768},
        "canvas_patch_grid_sizes": [32],
        "glimpse_grid_size": 8,
        "step": 200_000,
        "teacher_name": "dinov3_vitb16",
        "dataset": "in21k",
        "timestamp": "2026-02-02T00:00:00+00:00",
        "git_commit": "abc123",
        "state_dict": {},
        **({"training_config_history": history} if history is not None else {}),
    }


def _history(mode: str, fixed_scale: float) -> dict:
    return {
        "2026-02-01T00:00:00+00:00": {
            "foveated_scale.mode": mode,
            "foveated_scale.distribution": "loguniform",
            "foveated_scale.fixed_scale": fixed_scale,
            "foveated_scale.min_scale": 0.5,
            "foveated_scale.max_scale": 2.0,
        }
    }


def test_extract_fixed_foveated_scale():
    vs = extract_pretrain_view_scale(_raw("foveated", _history("fixed", 2.0)))
    assert vs == {
        "patcher_name": "foveated",
        "mode": "fixed",
        "distribution": "loguniform",
        "fixed_scale": 2.0,
        "min_scale": 0.5,
        "max_scale": 2.0,
    }


def test_extract_square_multiscale():
    vs = extract_pretrain_view_scale(_raw("square", _history("per_rollout", 1.0)))
    assert vs is not None and vs["patcher_name"] == "square" and vs["mode"] == "per_rollout"


def test_extract_uniform_is_none():
    # Uniform's OOD axis is glimpse crop pixels, not view-scale → not recorded.
    assert extract_pretrain_view_scale(_raw("uniform", _history("fixed", 2.0))) is None


def test_extract_no_history_is_none():
    # Older checkpoints predating training_config_history: unknown, not "1.0".
    assert extract_pretrain_view_scale(_raw("foveated", None)) is None


def test_extract_latest_history_entry_wins():
    hist = {
        "2026-01-01T00:00:00+00:00": {"foveated_scale.mode": "fixed", "foveated_scale.fixed_scale": 1.0},
        "2026-03-01T00:00:00+00:00": {"foveated_scale.mode": "fixed", "foveated_scale.fixed_scale": 2.0},
    }
    vs = extract_pretrain_view_scale(_raw("foveated", hist))
    assert vs is not None and vs["fixed_scale"] == 2.0


def test_build_config_embeds_view_scale():
    cfg = build_config(_raw("foveated", _history("fixed", 2.0)), Path("/x/step-200000.pt"))
    assert cfg["metadata"]["pretrain_view_scale"]["fixed_scale"] == 2.0
    assert cfg["backbone_name"] == "vitb16"
    assert cfg["metadata"]["teacher_name"] == "dinov3_vitb16"
    # Uniform → explicit None so eval treats it as "not applicable", not "unknown scare".
    cfg_u = build_config(_raw("uniform", None), Path("/x/step.pt"))
    assert cfg_u["metadata"]["pretrain_view_scale"] is None


# --- harness-schema checkpoints (nested metadata) --------------------------
def _harness_raw(patcher: str, *, patch_stride: int | None = None) -> dict:
    """A checkpoint shaped exactly as the harness writes it.

    Built by calling the REAL `DistillRunTask.model_config()` / `checkpoint_metadata()`
    rather than hand-writing the dicts — a hand-written fixture silently encoded the
    wrong `model_config` shape (flat instead of the harness's `{"canvit": {...}}`) and
    hid a live footgun that only showed up converting an actual checkpoint.
    """
    from types import SimpleNamespace

    from canvit_pretrain.tasks.distill.task import DistillRunTask
    from canvit_pretrain.train.config import Config, FoveatedScaleConfig

    cfg = Config(webdataset_dir=Path("/nonexistent"), patch_stride=patch_stride)
    cfg.model.patcher_name = patcher
    cfg.foveated_scale = FoveatedScaleConfig(mode="fixed", fixed_scale=2.0,
                                             distribution="uniform", min_scale=0.5, max_scale=1.0)
    task = DistillRunTask(cfg)
    stub = SimpleNamespace(cfg=cfg.model, canvas_patch_grid_sizes=[32])
    task_meta = task.checkpoint_metadata(stub)
    return {
        "step": 200_000,
        "model_state": {},
        "model_config": task.model_config(stub),
        "metadata": {**task_meta,
                     "training_config_history": {"2026-02-01T00:00:00+00:00": task_meta}},
    }


def test_normalize_leaves_legacy_untouched():
    legacy = _raw("foveated", _history("fixed", 2.0))
    assert normalize_schema(legacy) is legacy


def test_harness_checkpoint_converts_with_view_scale():
    """The footgun regression: before the shim this silently produced
    pretrain_view_scale=None because training_config_history was not top-level."""
    cfg = build_config(normalize_schema(_harness_raw("foveated")), Path("/x/step-200000.pt"))
    assert cfg["metadata"]["pretrain_view_scale"] == {
        "patcher_name": "foveated", "mode": "fixed", "distribution": "uniform",
        "fixed_scale": 2.0, "min_scale": 0.5, "max_scale": 1.0,
    }
    assert cfg["backbone_name"] == "vitb16"
    assert cfg["canvas_patch_grid_sizes"] == [32]
    assert cfg["glimpse_grid_size"] == 8


def test_harness_checkpoint_preserves_patch_stride():
    """Overlapping-patch models (exp21) are unrebuildable without patch_stride."""
    cfg = build_config(normalize_schema(_harness_raw("uniform", patch_stride=8)), Path("/x/s.pt"))
    assert cfg["patch_stride"] == 8
    # non-overlapping stays absent, so the config is byte-identical to before
    cfg_n = build_config(normalize_schema(_harness_raw("uniform")), Path("/x/s.pt"))
    assert "patch_stride" not in cfg_n


def test_harness_non_distill_checkpoint_rejected():
    """An ade20k/in1k checkpoint is not a pretraining model — fail loudly, not silently."""
    bad = {"model_state": {}, "model_config": {}, "metadata": {"task": "ade20k"}}
    with pytest.raises(KeyError, match="backbone_name"):
        normalize_schema(bad)
