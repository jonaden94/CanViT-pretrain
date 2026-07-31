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

IN1k CLASSIFIER checkpoints (``metadata.task == "in1k"``) are dispatched to a second
path that emits the ``CanViTForImageClassification.from_pretrained`` layout instead —
what ``CanViT-eval/tasks/in1k_clf.py`` loads. The standalone ``in1k/train.py`` used to
write that directory itself (``clf.save_pretrained(run_dir/"best-hf")``); when it was
deleted in the harness consolidation, the harness had no HF export at all, so an in1k
finetune could not be handed to canvit_eval. Same CLI, auto-detected from the payload.
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


def normalize_schema(raw: dict) -> dict:
    """Return the checkpoint in the FLAT schema this converter reads.

    Two writers produce checkpoints: the legacy ``train/loop.py`` trainer (flat —
    ``state_dict`` plus top-level ``backbone_name`` / ``canvas_patch_grid_sizes`` /
    ``training_config_history`` / …) and the unified harness
    (``harness/checkpoint.py``: ``model_state`` plus a nested ``metadata`` dict built
    from the task's ``checkpoint_metadata``). Without this, a harness checkpoint would
    KeyError on ``state_dict`` — or worse, silently yield ``pretrain_view_scale=None``
    because ``training_config_history`` is not at the top level, which is exactly the
    foveated OOD footgun this converter exists to prevent.

    Legacy payloads pass through untouched.
    """
    if "model_state" not in raw:
        return raw
    md = raw.get("metadata") or {}
    missing = [k for k in ("backbone_name", "canvas_patch_grid_sizes") if k not in md]
    if missing:
        raise KeyError(
            f"harness checkpoint metadata is missing {missing}; only a distill "
            f"checkpoint can be converted to the pretraining HF layout (this one is "
            f"task={md.get('task')!r})")
    # The harness nests the real architecture under model_config["canvit"] (the rest of
    # that dict is resume bookkeeping: task/teacher_dim/canvas_grid/backbone_name). The HF
    # layout — and `patcher_name`, which drives the whole view-scale footgun check — wants
    # the FLAT CanViTForPretrainingConfig, so unwrap it.
    mc = raw.get("model_config") or {}
    mc = mc.get("canvit", mc)
    return {
        **raw,
        "state_dict": raw["model_state"],
        "model_config": mc,
        "backbone_name": md["backbone_name"],
        "canvas_patch_grid_sizes": md["canvas_patch_grid_sizes"],
        "glimpse_grid_size": md.get("glimpse_grid_size"),
        "patch_stride": md.get("patch_stride"),
        "teacher_name": md.get("teacher_name"),
        "dataset": md.get("dataset"),
        "training_config_history": md.get("training_config_history"),
    }


def _scale_fields(entry: dict) -> dict[str, Any]:
    """The ``foveated_scale`` fields of one ``training_config_history`` entry.

    Legacy entries are FLAT (``foveated_scale.mode``, … — ``train/loop.py:flatten_dict``);
    harness entries carry the config as a NESTED ``foveated_scale`` dict.
    """
    if isinstance(nested := entry.get("foveated_scale"), dict):
        return nested
    return {k[len("foveated_scale."):]: v for k, v in entry.items()
            if k.startswith("foveated_scale.")}


def extract_pretrain_view_scale(raw: dict) -> dict[str, Any] | None:
    """Recover the pretraining foveated/square view-scale from a checkpoint.

    The scale lives in ``training_config_history[ts]``. Returns ``None`` for uniform
    models or when no history is recorded (older checkpoints) — callers must treat
    ``None`` as "unknown", not "scale 1.0".
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
    fields = _scale_fields(history[max(history)])
    mode = fields.get("mode")
    if mode is None:
        return None
    return {
        "patcher_name": patcher,
        "mode": mode,
        "distribution": fields.get("distribution"),
        "fixed_scale": fields.get("fixed_scale"),
        "min_scale": fields.get("min_scale"),
        "max_scale": fields.get("max_scale"),
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


def is_classifier_checkpoint(raw: dict) -> bool:
    """True for an IN1k classifier checkpoint, which needs the classification layout
    rather than the pretraining one. ``model_config`` is the harness in1k task's
    (``tasks/in1k/task.py::model_config``); a distill payload has no ``task`` key there."""
    return (raw.get("model_config") or {}).get("task") == "in1k"


def classifier_to_hf(raw: dict, out_dir: Path) -> None:
    """Write the ``CanViTForImageClassification.from_pretrained`` layout.

    Rebuilds the module and reuses the class's OWN ``save_pretrained`` (from
    ``PyTorchModelHubMixin``) rather than hand-assembling config.json + safetensors —
    the layout then cannot drift from what ``from_pretrained`` expects, which is the
    whole failure mode a second writer would introduce.

    The architecture is reconstructed with ``from_pretrained_with_new_head``, whose fresh
    random head is immediately overwritten by the checkpoint's own weights. That is
    correct for BOTH modes: ``finetune`` trained from a head built by
    ``from_pretrained_with_probe``, but the *architecture* either constructor produces is
    identical (LN(D) -> Linear(D, n_classes)) — only the init differs, and we load over it.
    """
    from canvit_pytorch import CanViTForImageClassification

    mc = raw["model_config"]
    repo, n_classes = mc["model_repo"], mc["n_classes"]
    log.info("in1k classifier (mode=%s, n_classes=%d) over %s", mc.get("mode"), n_classes, repo)
    clf = CanViTForImageClassification.from_pretrained_with_new_head(
        pretrained_repo=repo, n_classes=n_classes)
    # strict: a silently-partial load would publish a half-trained classifier.
    clf.load_state_dict(raw["model_state"], strict=True)
    assert clf.head.out_features == n_classes, (
        f"head has {clf.head.out_features} classes, checkpoint says {n_classes}")
    out_dir.mkdir(parents=True, exist_ok=True)
    clf.save_pretrained(out_dir)
    log.info("Wrote %s (step=%s)", out_dir, raw.get("step"))
    log.info("Load with: CanViTForImageClassification.from_pretrained(%r)", str(out_dir))


def main(args: Args) -> None:
    log.info("Loading %s ...", args.pt_path)
    raw = torch.load(args.pt_path, map_location="cpu", weights_only=False)
    if is_classifier_checkpoint(raw):
        classifier_to_hf(raw, args.out_dir)
        return
    raw = normalize_schema(raw)
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
