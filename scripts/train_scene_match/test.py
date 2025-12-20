"""Smoke tests for scene matching training module."""

from .config import Config


def test_config_defaults() -> None:
    """Config has sensible defaults."""
    cfg = Config()
    assert cfg.grid_size == 16
    assert cfg.batch_size == 128
    assert 0.0 <= cfg.p_reset <= 1.0
