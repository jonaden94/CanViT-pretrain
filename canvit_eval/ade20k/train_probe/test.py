"""Tests for ADE20K probe training."""

from pathlib import Path

from canvit_eval.ade20k.train_probe import Config, FeatureType


def test_config_required_fields() -> None:
    cfg = Config(model_repo="canvit/model", ade20k_root=Path("/data/ade20k"))
    assert cfg.n_timesteps == 10
    assert cfg.max_steps == 5000


def test_feature_types() -> None:
    features: list[FeatureType] = ["hidden", "predicted_norm", "teacher_glimpse"]
    assert len(features) == 3
