"""Export the trained ADE20K probe out of an ``ade20k`` checkpoint into the HF layout.

Why this exists separately from ``to_hf``: ``to_hf`` publishes whole models (a distill
checkpoint becomes ``CanViTForPretrainingHFHub``, an in1k one the classifier layout) and
refuses an ``ade20k`` payload. But an ade20k probe checkpoint is not a model to publish --
its backbone is frozen and already published elsewhere, and the only new thing in it is the
150-class segmentation head.

That head is exactly what ``--cfg.probe-repo`` wants: policy training and probe finetuning
call ``CanViTForSemanticSegmentation.from_pretrained_with_probe(pretrained_repo=...,
probe_repo=...)``, and ``probe_repo`` is read with ``SegmentationProbe.from_pretrained``,
i.e. an HF directory, NOT a harness ``.pt``. Without this step a trained probe cannot be
used as the reward model for a policy run at all.

Usage:
    python -m canvit_train.checkpoint.probe_to_hf \
        --pt-path  <run>/checkpoints/best.pt \
        --out-dir  <run>/checkpoints/best-probe-hf

The pretrained BACKBONE is not written here and is not needed: it is whatever
``model_config["model_repo"]`` of the source run pointed at, and the caller passes it as
``--cfg.model-repo``. Both halves are required to rebuild the model.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from canvit_pytorch.probes import SegmentationProbe

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class Args:
    pt_path: Path
    """An `ade20k` harness checkpoint (best.pt / step-<N>.pt)."""
    out_dir: Path
    """Directory to write config.json + model.safetensors into."""
    dropout: float = 0.1
    """Probe dropout. Not recorded in the checkpoint, so it must match the source run's
    `--cfg.dropout` (0.1 is the Ade20kConfig default). It affects training only, but it is
    stored in the probe config and reused when the probe is loaded."""


def export_probe(pt_path: Path, out_dir: Path, dropout: float = 0.1) -> SegmentationProbe:
    raw = torch.load(pt_path, weights_only=False, map_location="cpu")
    task = (raw.get("metadata") or {}).get("task")
    if task != "ade20k":
        raise ValueError(f"{pt_path} is a {task!r} checkpoint; only ade20k carries a probe head")

    head = {k.removeprefix("head."): v for k, v in raw["model_state"].items()
            if k.startswith("head.")}
    if not head:
        raise ValueError(f"{pt_path} has no head.* weights")

    # Read the shape rather than trusting a flag: `conv.weight` is (num_classes, embed_dim,
    # 1, 1), and `ln.*` is present only when the probe was built with use_ln=True.
    num_classes, embed_dim = head["conv.weight"].shape[:2]
    use_ln = "ln.weight" in head

    probe = SegmentationProbe(embed_dim=embed_dim, num_classes=num_classes,
                              dropout=dropout, use_ln=use_ln)
    missing, unexpected = probe.load_state_dict(head, strict=True)
    assert not missing and not unexpected, f"missing={missing}, unexpected={unexpected}"

    out_dir.mkdir(parents=True, exist_ok=True)
    probe.save_pretrained(out_dir)
    log.info("step %s -> %s (embed_dim=%d, num_classes=%d, use_ln=%s, dropout=%s)",
             raw.get("step"), out_dir, embed_dim, num_classes, use_ln, dropout)
    log.info("pass it as --cfg.probe-repo %s", out_dir)
    log.info("the matching backbone is --cfg.model-repo %s",
             (raw.get("model_config") or {}).get("model_repo", "<unknown>"))
    return probe


def main() -> None:
    a = tyro.cli(Args)
    export_probe(a.pt_path, a.out_dir, a.dropout)


if __name__ == "__main__":
    main()
