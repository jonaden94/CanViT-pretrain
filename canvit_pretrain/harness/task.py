"""The harness Task seam — the contract the three peer tasks implement (design §3.1).

A ``Task`` owns everything task-specific (data, targets, head, readout, loss,
reward signal, eval, checkpoint payload); the harness owns everything else. The
three tasks — ``tasks/distill``, ``tasks/ade20k``, ``tasks/in1k`` — are equal peers,
none privileged (design §1). The rollout engine only needs the per-batch
``RolloutTask`` view (``forward_glimpse`` / ``step_loss`` / ``per_image_loss``,
in ``rollout.py``); this module adds the run-level surface the neutral loop needs.

Per-task mapping (design §0):

    | task    | model (backbone owner)             | readout          | loss   | metric   |
    |---------|------------------------------------|------------------|--------|----------|
    | distill | CanViTForPretraining               | scene+cls preds  | MSE    | cos sim  |
    | ade20k  | CanViTForSemanticSegmentation      | canvas_hidden    | CE     | mIoU     |
    | in1k    | CanViTForImageClassification       | recurrent CLS    | CE     | top-1/5  |

NOTE: this is the interface only. Concrete task implementations live under
``canvit_pretrain/tasks/<name>/`` (unification task #14). The distill parity test
(``tests/test_rollout_parity.py``) already exercises a distill ``RolloutTask`` view.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from torch import Tensor

from canvit_pretrain.harness.rollout import RolloutTask
from canvit_pretrain.harness.spec import TaskCaps, TrainSpec


@runtime_checkable
class Task(Protocol):
    """Run-level task contract. ``bind`` produces the per-batch :class:`RolloutTask`
    the engine consumes; everything else is consulted once per run or per validation.
    """

    name: str

    # --- capabilities & defaults ------------------------------------------
    def caps(self) -> TaskCaps:
        """What this task supports (has_head / supports_policy) — the validator uses
        it to reject specs the task cannot honor (design §8)."""
        ...

    def default_spec(self) -> TrainSpec:
        """The task's canonical default config (e.g. distill = joint-capable chunked
        TBPTT; ade20k = frozen-backbone probe; in1k = frozen probe). Per-task defaults
        are data, not forked control flow (design §1)."""
        ...

    def extra_spec_warnings(self, spec: TrainSpec) -> list[str]:
        """Task-specific validation warnings beyond the generic ones in
        ``check_spec`` (e.g. distill warns on a frozen-backbone spec — distillation
        IS backbone training, so it is near-vacuous)."""
        ...

    # --- construction ------------------------------------------------------
    def build_model(self, cfg: Any, spec: TrainSpec) -> tuple[Any, Any]:
        """Return ``(model, head)``. ``model`` owns the CanViT backbone (a
        ``CanViTFor*`` wrapper); ``head`` is the task head module or ``None`` (distill's
        heads live inside its forward). The harness sets ``requires_grad`` on
        backbone/head/policy from the spec — the task must NOT freeze here."""
        ...

    def build_loaders(self, cfg: Any) -> tuple[Any, Any]:
        """``(train_loader, val_loader)``. Owns all task-specific data + DDP sharding
        (distill webdataset + normalizers; ade20k map dataset; in1k webdataset with
        ``with_epoch`` so the step-based loop derives ``steps_per_epoch``, design §Loop)."""
        ...

    def build_selector(self, cfg: Any, spec: TrainSpec, *, device: Any,
                       canvas_grid: int, is_foveated: bool) -> Any:
        """The random viewing policy for this run (patcher-aware). Joint/policy runs
        wrap it with the shared PolicySelector via the harness policy builder."""
        ...

    def policy_feature_groups(self) -> tuple[str, ...]:
        """StateEncoder feature groups for the scorer: INTRINSIC (distill, probe-free)
        vs probe-aware (ade20k/in1k) — design §3.1."""
        ...

    # --- per-batch (engine-facing) ----------------------------------------
    def batch_images(self, batch: Any, device: Any) -> Tensor:
        """Extract the [B,3,H,W] image tensor the engine threads through the rollout."""
        ...

    def bind(self, batch: Any, device: Any, *, model: Any, head: Any) -> RolloutTask:
        """Bind this batch's targets and return the per-glimpse :class:`RolloutTask`
        (``forward_glimpse`` / ``step_loss`` / ``per_image_loss``) the engine runs.
        Mirrors distill's fresh-per-batch ``DistillTask``."""
        ...

    # --- eval & checkpoint -------------------------------------------------
    def evaluate(self, *, model: Any, head: Any, val_loader: Any, cfg: Any,
                 device: Any, step: int) -> dict[str, float]:
        """Task metrics over the val set (distill cos/recon; ade20k mIoU-per-t; in1k
        top-1/5). The harness owns cadence; the task owns what is measured (design §10)."""
        ...

    def checkpoint_payload(self, *, model: Any, head: Any) -> dict:
        """Task-specific extras to persist LOCALLY beside the harness checkpoint
        (design D-G: persist all locally, never auto-push to HF)."""
        ...


__all__ = ["Task"]
