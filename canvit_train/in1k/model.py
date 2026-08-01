"""IN1k classifier construction — the mode-dependent head init.

Moved verbatim out of the (now deleted) standalone ``in1k/train.py`` during the
harness consolidation. It lived there because the harness task imported it from the
standalone so the two entry points could not diverge on head init; with one entry
point left, it belongs in a module of its own rather than inside a trainer.
"""

import logging

from canvit_pytorch import CanViTForImageClassification

from .config import NUM_CLASSES, In1kConfig

log = logging.getLogger(__name__)


def _resolve_probe_repo(cfg: In1kConfig) -> str:
    """The DINOv3 in1k linear probe fused into the finetune head. ``cfg.probe_repo``
    wins; otherwise derive from the pretrained checkpoint's ``backbone_name`` (metadata
    only — no model load). Works for a ``.pt`` as well as an HF dir / Hub id, or passing a
    checkpoint to ``--cfg.model-repo`` would fail here, before any weights are touched."""
    if cfg.probe_repo:
        return cfg.probe_repo
    from canvit_pytorch.model_source import read_backbone_name
    from dinov3_in1k_probes.repos import probe_repo

    backbone = read_backbone_name(cfg.model_repo)
    if backbone is None:
        raise KeyError(
            f"could not read backbone_name from {cfg.model_repo} to pick the DINOv3 probe; "
            f"pass --cfg.probe-repo explicitly")
    return probe_repo(backbone)


def build_classifier(cfg: In1kConfig, device) -> CanViTForImageClassification:
    """Construct the CanViT classifier for ``cfg.mode``.

    ``frozen``  -> fresh (random) LN+Linear head over the frozen backbone: the
        from-scratch linear-probe protocol (the canvit_eval baseline); the head trains
        from zero, so a random start is correct.
    ``finetune`` -> fuse the DINOv3 in1k linear probe into the head, reproducing the TPU
        flagship (``gcp_in1k_clf_ft/shared.py::load_classifier``). A random head would
        start at chance (loss = ln(1000)) and, at the tiny finetune LR, train far too
        slowly — so the fused probe is essential, not cosmetic.
    """
    if cfg.mode == "finetune":
        clf = CanViTForImageClassification.from_pretrained_with_probe(
            pretrained_repo=cfg.model_repo, probe_repo=_resolve_probe_repo(cfg),
        )
    else:
        clf = CanViTForImageClassification.from_pretrained_with_new_head(
            pretrained_repo=cfg.model_repo, n_classes=NUM_CLASSES,
        )
    return clf.to(device)
