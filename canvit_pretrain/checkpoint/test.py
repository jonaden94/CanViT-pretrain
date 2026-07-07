"""Tests for checkpoint serialization."""

import tempfile
from pathlib import Path

import torch
from canvit_pytorch import create_backbone

from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig
from canvit_pretrain.checkpoint import load, save

_TEACHER_REPO = "facebook/dinov3-vits16-pretrain"
_TEACHER_NAME = "dinov3_vits16"
_DATASET = "in21k"


def _make_tiny_model(device: torch.device) -> CanViTForPretraining:
    """Create minimal CanViTForPretraining for testing (no pretrained weights needed)."""
    backbone = create_backbone("vits16").to(device)
    cfg = CanViTForPretrainingConfig(teacher_dim=384)
    return CanViTForPretraining(
        backbone=backbone,
        cfg=cfg,
        glimpse_size_px=8 * backbone.patch_size_px,
        backbone_name="vits16",
        canvas_patch_grid_sizes=[8, 16, 32],
    ).to(device)


def test_save_load_roundtrip() -> None:
    device = torch.device("cpu")
    model = _make_tiny_model(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pt"
        save(
            path, model, backbone_name="vits16",
            teacher_repo_id=_TEACHER_REPO, teacher_name=_TEACHER_NAME,
            dataset=_DATASET, glimpse_grid_size=8, scene_resolution=512,
            step=100, train_loss=0.5,
        )

        data = load(path, device)

        assert data["backbone_name"] == "vits16"
        assert data["canvas_patch_grid_sizes"] == [8, 16, 32]
        assert data["teacher_dim"] == 384
        assert data["teacher_repo_id"] == _TEACHER_REPO
        assert data["teacher_name"] == _TEACHER_NAME
        assert data["dataset"] == _DATASET
        assert data["glimpse_grid_size"] == 8
        assert data["scene_resolution"] == 512
        assert data["step"] == 100
        assert data["train_loss"] == 0.5

        # Verify state_dict loads into fresh model
        model2 = _make_tiny_model(device)
        model2.load_state_dict(data["state_dict"])
