"""Tests for probe head."""

import torch

from canvit_eval.ade20k.dataset import NUM_CLASSES
from canvit_eval.ade20k.probe import ProbeHead


def test_output_shape() -> None:
    probe = ProbeHead(768)
    x = torch.randn(2, 32, 32, 768)
    out = probe(x)
    assert out.shape == (2, NUM_CLASSES, 32, 32)
