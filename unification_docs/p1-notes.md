# P1 — harness seams in training_step (DONE 2026-07-22)

## What changed

- **`train/selector.py`** (new): `Selector` protocol + `RolloutCtx` + `RandomSelector`.
  Byte-for-byte extraction of `training_step`'s historical closures
  (`_foveated_random_vp`, `make_named_vp`, the per-rollout foveated scale draw) —
  same RNG calls, same order. `select()` takes `state` and `t` (unused here) so the
  P3 `PolicySelector` can condition on the live canvas without an interface change.
- **`train/task.py`** (new): `Task` protocol + `LossOutput` + `DistillTask` — the
  historical `compute_loss` closure, including the DDP-Reducer comment (heads must
  produce preds inside the wrapped forward). `LossOutput` moved here; `step.py`
  re-exports it for compatibility.
- **`train/step.py`**: `training_step` gains optional `selector=None, task=None`
  kwargs; `None` builds the historical defaults from the existing kwargs, so every
  existing call site (loop.py, tests) is untouched and behavior is provably
  identical. The three closures are gone; call sites route through the seams.
- **`train/test_seams.py`** (new): spy-injection test — a custom Selector/Task is
  consulted with exactly the expected cardinality (per-branch `start_rollout`,
  per-glimpse `select`/`step_loss`).

## Gates

| gate | result |
|---|---|
| parity probe digest | `9a0100a1a3de3acd` — **byte-identical** to pre-refactor baseline |
| full pretrain suite | 61 passed (unchanged) |
| seam injection test | 1 passed (new) |

## Deliberate scope limits (for later phases)

- `loop.py` still calls `training_step` without seams (defaults built inside).
  Seam construction moves up to loop/config in **P2**, when the config grows a task
  switch and `DistillTask` takes over target-building from loop.py.
- `LossOutput` is still distill-shaped (`scene_*`/`cls_*` fields). Generalizing the
  metrics container is a P2 concern (ADE needs per-timestep CE, not scene/cls MSE).
- `t1_schedule` (all-RANDOM for t>=1) stays in `training_step`; it becomes
  selector-owned when the ε-curriculum lands (P4).
- The rollout/backward engine (`run_branch`, `ChunkState`, TBPTT chunking) is
  intentionally NOT extracted yet — P1 injected the two seams with minimal parity
  risk; the in-graph policy featurization (§4.3) reshapes `run_branch` in P3/P4.
