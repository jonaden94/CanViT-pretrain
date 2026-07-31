"""Unified LOCAL checkpoint I/O for the harness (design D-G).

Persists everything needed to resume + (optionally, later) publish: the full model
state (backbone + head — both are submodules of the task wrapper, so one
``state_dict()`` captures them), the optimizer/scheduler, the step, the resolved
``TrainSpec``, the model config, and free-form metadata (incl. ``pretrain_view_scale``
for the foveated footgun). The policy (scorer + reward standardizers + PG dual) rides
in a ``.policy.pt`` sidecar, schema-independent of the main file.

**Never touches the network.** Training writes local files only — exactly as
CanViT-train/specialize do today. HF publishing stays a separate, manual
``python -m canvit_train.checkpoint.to_hf`` step (owner decision D-G).
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from canvit_train.harness.spec import TrainSpec


def save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    step: int,
    task_name: str,
    spec: TrainSpec,
    model_config: dict | None = None,
    metadata: dict | None = None,
    joint: Any | None = None,
    extra: dict | None = None,
    update_latest: bool = True,
) -> Path:
    """Write a full local checkpoint (and a ``.policy.pt`` sidecar if ``joint`` given).
    Returns the path written. DDP-safe to call on rank 0 only (unwraps ``.module``)."""
    core = getattr(model, "module", model)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "task": task_name,
        "model_state": core.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "train_spec": asdict(spec),
        "model_config": model_config or {},
        "metadata": metadata or {},
        "extra": extra or {},
    }
    torch.save(payload, path)
    if joint is not None:
        torch.save(joint.state_dict(), path.with_suffix(".policy.pt"))
    if update_latest:
        update_latest_symlink(path.parent / "latest.pt", path)
    return path


def load_checkpoint(path: Path, device: torch.device | str) -> dict:
    """Load a checkpoint payload. Use ``restore_into`` to apply it to live modules."""
    return torch.load(path, map_location=device, weights_only=False)


def restore_into(
    payload: dict, *, model: Any, optimizer: Any = None, scheduler: Any = None,
    joint: Any | None = None, path: Path | None = None, device: torch.device | str = "cpu",
) -> int:
    """Apply a loaded payload to live modules; return the saved step. Optimizer/scheduler
    are restored only when provided (resume mode); a joint sidecar is restored when
    present next to ``path``."""
    core = getattr(model, "module", model)
    core.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if joint is not None and path is not None:
        sidecar = path.with_suffix(".policy.pt")
        if sidecar.exists():
            joint.load_state_dict(torch.load(sidecar, map_location=device, weights_only=False))
    return int(payload.get("step", 0))


def update_latest_symlink(link: Path, target: Path) -> None:
    """Point ``link`` at ``target`` (atomic replace). Best-effort: falls back to a
    tiny pointer file on filesystems without symlinks."""
    try:
        tmp = link.with_suffix(".tmp")
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        os.symlink(target.name, tmp)
        os.replace(tmp, link)
    except OSError:
        link.write_text(target.name)


def find_latest(ckpt_dir: Path) -> Path | None:
    """Resolve the newest checkpoint: the ``latest.pt`` symlink/pointer if present,
    else the highest ``step-<n>.pt``. None if the dir has no checkpoints."""
    if not ckpt_dir.is_dir():
        return None
    latest = ckpt_dir / "latest.pt"
    if latest.is_symlink() or latest.is_file():
        target = ckpt_dir / (os.readlink(latest) if latest.is_symlink() else latest.read_text().strip())
        if target.is_file():
            return target
    steps = sorted(ckpt_dir.glob("step-*.pt"), key=lambda p: int(p.stem.split("-")[1]))
    return steps[-1] if steps else None


__all__ = ["find_latest", "load_checkpoint", "restore_into", "save_checkpoint", "update_latest_symlink"]
