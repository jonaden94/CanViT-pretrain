"""Tests for ADE20K dataset."""

from canvit_eval.ade20k.dataset import IGNORE_LABEL, NUM_CLASSES


def test_constants() -> None:
    assert NUM_CLASSES == 150
    assert IGNORE_LABEL == 255
