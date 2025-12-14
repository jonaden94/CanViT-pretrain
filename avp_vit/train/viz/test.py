"""Tests for visualization utilities."""

import numpy as np
import torch
from matplotlib.figure import Figure

from avp_vit.glimpse import PixelBox
from avp_vit.train.viz import (
    fit_pca,
    imagenet_denormalize,
    pca_rgb,
    plot_pca_grid,
    plot_trajectory,
    timestep_colors,
)


class TestPCA:
    def test_fit_pca(self) -> None:
        features = np.random.randn(256, 384).astype(np.float32)
        pca = fit_pca(features)
        assert pca.n_components == 3

    def test_pca_rgb_shape(self) -> None:
        features = np.random.randn(256, 384).astype(np.float32)
        pca = fit_pca(features)
        rgb = pca_rgb(pca, features, 16, 16)
        assert rgb.shape == (16, 16, 3)

    def test_pca_rgb_bounds(self) -> None:
        features = np.random.randn(256, 384).astype(np.float32)
        pca = fit_pca(features)
        rgb = pca_rgb(pca, features, 16, 16)
        # Sigmoid output should be in (0, 1)
        assert (rgb >= 0).all()
        assert (rgb <= 1).all()


class TestImagenetDenormalize:
    def test_shape(self) -> None:
        img = torch.randn(3, 64, 64)
        result = imagenet_denormalize(img)
        assert result.shape == (64, 64, 3)

    def test_bounds(self) -> None:
        img = torch.randn(3, 64, 64)
        result = imagenet_denormalize(img)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_zero_input(self) -> None:
        img = torch.zeros(3, 64, 64)
        result = imagenet_denormalize(img)
        # Zero normalized -> ImageNet mean
        expected = torch.tensor([0.485, 0.456, 0.406])
        assert torch.allclose(result[0, 0], expected, atol=1e-5)

    def test_same_device(self) -> None:
        img = torch.randn(3, 32, 32)
        result = imagenet_denormalize(img)
        assert result.device == img.device


class TestTimestepColors:
    def test_returns_correct_count(self) -> None:
        colors = timestep_colors(5)
        assert len(colors) == 5

    def test_single_color(self) -> None:
        colors = timestep_colors(1)
        assert len(colors) == 1

    def test_rgba_tuples(self) -> None:
        colors = timestep_colors(3)
        for c in colors:
            assert len(c) == 4  # RGBA


class TestPlotTrajectory:
    def test_returns_figure(self) -> None:
        img = np.random.rand(64, 64, 3).astype(np.float32)
        boxes = [
            PixelBox(left=0, top=0, width=64, height=64, center_x=32, center_y=32),
            PixelBox(left=16, top=16, width=32, height=32, center_x=32, center_y=32),
        ]
        names = ["full", "center"]
        fig = plot_trajectory(img, boxes, names)
        assert isinstance(fig, Figure)

    def test_empty_boxes(self) -> None:
        img = np.random.rand(64, 64, 3).astype(np.float32)
        fig = plot_trajectory(img, [], [])
        assert isinstance(fig, Figure)


class TestPlotPcaGrid:
    def test_returns_figure(self) -> None:
        reference = np.random.randn(16, 64).astype(np.float32)
        samples = [np.random.randn(16, 64).astype(np.float32) for _ in range(3)]
        pca = fit_pca(reference)
        titles = ["t=0", "t=1", "t=2"]
        fig = plot_pca_grid(pca, reference, samples, grid_size=4, titles=titles)
        assert isinstance(fig, Figure)

    def test_single_sample(self) -> None:
        reference = np.random.randn(16, 64).astype(np.float32)
        samples = [np.random.randn(16, 64).astype(np.float32)]
        pca = fit_pca(reference)
        fig = plot_pca_grid(pca, reference, samples, grid_size=4, titles=["t=0"])
        assert isinstance(fig, Figure)
