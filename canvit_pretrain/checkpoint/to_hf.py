"""Convert a CanViT-pretrain ``.pt`` checkpoint into the local HF Hub layout that
``CanViTForPretrainingHFHub.from_pretrained(<dir>)`` reads.

Ported from the (now-archived) ``CanViT-specialize/scripts/pretrain_ckpt_to_hf_format.py``
into the unified repo. Adds one thing the old converter lacked: an explicit
``metadata.pretrain_view_scale`` recording the foveated/square view-scale the model
was pretrained with. That scale is a *training-time viewpoint* parameter
(``FoveatedScaleConfig``) — it is NOT part of ``model_config``, so without recording it
here a downstream eval has no way to know the in-distribution glimpse scale and silently
evaluates OOD (the footgun that broke run 15025338). CanViT-eval reads this field to
auto-set its view-scale. It is derived from the checkpoint's ``training_config_history``
(the authoritative record of the training config), and is ``None`` for uniform models
(view-scale is not their OOD axis — glimpse crop pixels are).

Output dir gets:
    config.json        — backbone_name, model_config, canvas_patch_grid_sizes, glimpse_grid_size, metadata
    model.safetensors  — model state_dict

Usage:
    python -m canvit_pretrain.checkpoint.to_hf --pt-path /path/to/step-NNNNNN.pt --out-dir /path/to/out
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Patchers whose in-distribution behavior depends on the pretraining view-scale.
_SCALE_SENSITIVE_PATCHERS = ("foveated", "square")


@dataclass
class Args:
    pt_path: Path
    out_dir: Path


def _migrate_standardizers_in_place(raw: dict) -> None:
    """Mirror canvit_pretrain's legacy → current standardizer migration."""
    if (scene_legacy := raw.get("scene_norm_state")) is None:
        return
    cls_legacy = raw["cls_norm_state"]
    grids = raw["canvas_patch_grid_sizes"]
    assert len(grids) == 1, f"Expected single grid size, got {grids}"
    G = str(grids[0])
    sd = raw["state_dict"]
    for prefix, legacy in [("scene_standardizers", scene_legacy), ("cls_standardizers", cls_legacy)]:
        for stat in ("mean", "var", "_initialized"):
            sd[f"{prefix}.{G}.{stat}"] = legacy[stat]
    del raw["scene_norm_state"], raw["cls_norm_state"]
    log.info("Migrated legacy standardizers (grid=%s)", G)


def extract_pretrain_view_scale(raw: dict) -> dict[str, Any] | None:
    """Recover the pretraining foveated/square view-scale from a checkpoint.

    The scale lives in ``training_config_history[ts]`` as flattened keys
    (``foveated_scale.mode``, ``foveated_scale.fixed_scale``, …; see
    ``train/loop.py:flatten_dict``). Returns ``None`` for uniform models or when
    no history is recorded (older checkpoints) — callers must treat ``None`` as
    "unknown", not "scale 1.0".
    """
    patcher = (raw.get("model_config") or {}).get("patcher_name")
    if patcher not in _SCALE_SENSITIVE_PATCHERS:
        return None
    history = raw.get("training_config_history") or {}
    if not history:
        return None
    # Entries are keyed by ISO-8601 timestamp; the most recent is the config the
    # run finished with. The view-scale is a model-defining choice and is
    # expected constant across a run, but taking the latest is the safe pick.
    latest = history[max(history)]
    mode = latest.get("foveated_scale.mode")
    if mode is None:
        return None
    return {
        "patcher_name": patcher,
        "mode": mode,
        "distribution": latest.get("foveated_scale.distribution"),
        "fixed_scale": latest.get("foveated_scale.fixed_scale"),
        "min_scale": latest.get("foveated_scale.min_scale"),
        "max_scale": latest.get("foveated_scale.max_scale"),
    }


def build_config(raw: dict, pt_path: Path) -> dict[str, Any]:
    """Assemble the config.json dict for the HF layout (pure; no I/O)."""
    config: dict[str, Any] = {
        "backbone_name": raw["backbone_name"],
        "model_config": raw["model_config"],
        "canvas_patch_grid_sizes": raw["canvas_patch_grid_sizes"],
        # Glimpse token-grid side (tokens per glimpse edge). Lets the HF-loaded
        # model / eval reconstruct the trained pixel glimpse size as
        # ``glimpse_grid_size * patch_size_px`` for any patch size. May be absent
        # in pre-this-field checkpoints (older runs) -> consumers fall back to 8.
        "glimpse_grid_size": raw.get("glimpse_grid_size"),
        "metadata": {
            "source_pt": str(pt_path),
            "step": raw.get("step"),
            "teacher_name": raw.get("teacher_name"),
            "dataset": raw.get("dataset"),
            "timestamp": raw.get("timestamp"),
            "git_commit": raw.get("git_commit"),
            # Explicit pretraining view-scale (None for uniform / unknown). See
            # extract_pretrain_view_scale + CanViT-eval config.resolve_view_scale.
            "pretrain_view_scale": extract_pretrain_view_scale(raw),
        },
    }

    # Overlapping-patch models (stride < patch_size) trained with a non-default
    # patch_stride, which lives OUTSIDE model_config (top-level training field)
    # and is needed to rebuild the patch-embed conv. Persist it ONLY when set,
    # so non-overlapping checkpoints produce a byte-for-byte identical config.
    patch_stride = raw.get("patch_stride")
    if patch_stride is not None:
        config["patch_stride"] = patch_stride
        log.info("Persisted patch_stride=%s (overlapping patches)", patch_stride)

    return config


def main(args: Args) -> None:
    log.info("Loading %s ...", args.pt_path)
    raw = torch.load(args.pt_path, map_location="cpu", weights_only=False)
    _migrate_standardizers_in_place(raw)

    config = build_config(raw, args.pt_path)
    vs = config["metadata"]["pretrain_view_scale"]
    log.info("pretrain_view_scale: %s", vs if vs is not None else "None (uniform / not recorded)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = args.out_dir / "config.json"
    sd_path = args.out_dir / "model.safetensors"
    cfg_path.write_text(json.dumps(config, indent=2, default=str))
    save_file(raw["state_dict"], sd_path)

    log.info("Wrote %s (%d params)", sd_path, len(raw["state_dict"]))
    log.info("Wrote %s", cfg_path)
    log.info("Load with: CanViTForPretrainingHFHub.from_pretrained(%r)", str(args.out_dir))


if __name__ == "__main__":
    main(tyro.cli(Args))
