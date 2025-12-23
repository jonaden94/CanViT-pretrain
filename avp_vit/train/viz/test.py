"""Tests for avp_vit.train.viz module."""

import numpy as np

from avp_vit.train.viz.pca import fit_pca, pca_rgb


class TestFitPca:
    def test_returns_pca(self) -> None:
        features = np.random.randn(100, 64).astype(np.float32)
        pca = fit_pca(features, n_components=12)
        assert pca is not None

    def test_returns_none_for_low_variance(self) -> None:
        features = np.ones((100, 64), dtype=np.float32)
        pca = fit_pca(features, n_components=12)
        assert pca is None

    def test_clamps_components(self) -> None:
        features = np.random.randn(5, 64).astype(np.float32)
        pca = fit_pca(features, n_components=12)
        assert pca is not None
        assert pca.n_components_ == 5


class TestPcaRgb:
    def test_output_shape(self) -> None:
        features = np.random.randn(64, 128).astype(np.float32)
        pca = fit_pca(features)
        assert pca is not None
        rgb = pca_rgb(pca, features, H=8, W=8)
        assert rgb.shape == (8, 8, 3)

    def test_none_pca_returns_gray(self) -> None:
        features = np.random.randn(64, 128).astype(np.float32)
        rgb = pca_rgb(None, features, H=8, W=8)
        assert rgb.shape == (8, 8, 3)
        assert np.allclose(rgb, 0.5)
