# harness_repro — production-scale fidelity A/B (the pre-cutover gate)

Ready-to-fire launchers that re-run **existing old-loop configs through the unified
harness** (`python -m canvit_train.harness.run`), so their curves can be overlaid on
the results you already have. This is the one remaining hole before the big-bang cutover:
everything so far is component-level (byte-exact distill rollout parity), short runs, or
probe-scale gates — this closes it at production scale on the *old* configs.

**Nothing here is submitted automatically.** Each script is a normal `bash …` launcher (it
calls `sbatch slurm_nhr/harness_train.sbatch`). Run when the queue is clear; they use a
separate `RUN_GROUP`/wandb project (`harness_repro`) so they never collide with the live
`jon_exp22_full_runs` arrays or logs.

## Scripts

| script | mirrors | proves |
|---|---|---|
| `distill-uniform16.sh` | `jon_exp22_full_runs/exp22-uniform16.sh` | uniform-patcher distill through the harness |
| `distill-fovi.sh` | `exp22-fovi.sh` | foveated patcher (random init) |
| `distill-fovi-teacherinit.sh` | `exp22-fovi-teacherinit.sh` | foveated + teacher-init (pairs with the above = teacher-init A/B) |
| `ade20k-finetune.sh` | a specialize ade20k finetune (TEMPLATE) | the `train_backbone` path (probe already gated) |

Each distill run is `~100k` steps (13 × 8192), not the full 2M — enough to see the curves
track. Config is identical to the mirrored exp22 script; only the training CODE differs
(harness vs `train/loop.py`). Flags gain the `--cfg.` prefix (the harness nests the task
config under `cfg`); `webdataset-dir` comes from `.envrc.grete` (already the
in21k-with-features set), so it isn't repeated.

## How to compare

Overlay `harness_repro/<run>` against the matching `jon_exp22_full_runs/<run>` in wandb:
- **distill:** train `loss` and the val reconstruction metrics should track closely
  (seeded → near-tight early, then follow). A persistent divergence = a real fidelity bug.
- **ade20k finetune:** mIoU within the ~0.007 run-to-run band measured for the probe.

## Commit pins (edit-safe A/B)

`PRETRAIN_COMMIT=bc63eee` (the unified harness) · `FOVI_COMMIT=c399d3b` (= exp22) ·
`PYTORCH_COMMIT=017ce9b` (current head — what `bc63eee` was tested against). **Caveat:**
exp22 used `PYTORCH_COMMIT=3277048`. For a strict *loop-only* A/B, either re-pin to
`3277048` (if `bc63eee` is API-compatible) or treat the `3277048→017ce9b` canvit_pytorch
drift as a second variable when reading curve differences.

## Scope

Representative, not exhaustive — the *old* combinations (the ones with existing baselines).
New combinations the unification enables (novel joint-policy modes, etc.) are out of scope
here; these establish that the ported functionality reproduces before the old loop is deleted.
