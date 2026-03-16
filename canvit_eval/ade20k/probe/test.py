"""Tests for probe (uses SegmentationProbe from canvit_utils)."""

import torch

from canvit_eval.ade20k.dataset import NUM_CLASSES
from canvit_probes import SegmentationProbe


def test_forward_shape() -> None:
    probe = SegmentationProbe(embed_dim=768, num_classes=NUM_CLASSES, dropout=0.1, use_ln=True)
    x = torch.randn(2, 32, 32, 768)
    out = probe(x)
    assert out.shape == (2, NUM_CLASSES, 32, 32)


def test_predict_rescales() -> None:
    probe = SegmentationProbe(embed_dim=768, num_classes=NUM_CLASSES, dropout=0.1, use_ln=True).eval()
    x = torch.randn(1, 32, 32, 768)
    out = probe.predict(x, rescale_to=(512, 512))
    assert out.shape == (1, NUM_CLASSES, 512, 512)


def test_no_dropout_in_eval() -> None:
    probe = SegmentationProbe(embed_dim=768, num_classes=NUM_CLASSES, dropout=0.5, use_ln=False).eval()
    x = torch.randn(1, 8, 8, 768)
    with torch.no_grad():
        out1 = probe.predict(x, rescale_to=(64, 64))
        out2 = probe.predict(x, rescale_to=(64, 64))
    assert torch.allclose(out1, out2)
