# 12 — old-loop → harness feature-parity audit (the pre-cutover code gate)

**Question (owner's claim 7):** would the big-bang cutover — deleting `train/__main__.py`
+ `train/loop.py` and repointing `base_train.sbatch` at the harness — remove any
functionality not reimplemented? This is the STATIC counterpart to the production A/B
(doc 11 §6): the runs prove what's reimplemented matches *numerically*; this proves
nothing is *silently dropped* — including features a given A/B config never exercises.

**Method:** read the old production path end to end — `slurm_nhr/base_train.sbatch`,
`canvit_train/train/__main__.py`, `canvit_train/train/loop.py` (1035 lines) — and
cross-check each operational feature against the harness (`harness/run.py` + `harness/loop.py`
+ `harness/cli.py` + `tasks/distill/task.py`). Not a grep-map; the loop was read (this
session's pass-4 read + this checklist pass). Verified runtime facts where a docstring was
ambiguous (EMA, backward_pass_autocast) rather than trusting comments.

## Parity table

| feature | old loop | harness | status |
|---|---|---|---|
| launcher: commit-pin / TMPDIR fallback / compile caches / DDP env / CFG_→flag / wandb | base_train.sbatch | harness_train.sbatch | ✅ mirrored |
| SIGUSR1 → checkpoint-after-step | loop.py `_handle_sigusr1` | loop.py `_install_sigusr1` | ✅ |
| FAILED marker + `scancel` crash-loop guard | loop.py `cancel_slurm_array` | loop.py `cancel_slurm_array` + `use_failed_marker` | ✅ |
| EMA-smoothed metric logging (total_loss, n_glimpses, branch series) | loop.py `ema.update`/`ema.items` | loop.py `smoothed` dict + `ema.update` (same set; adds `total_loss_raw`) | ✅ verified in code |
| grad clip (model + policy scorer, separately) | loop.py 145/149 | loop.py 283/287 | ✅ |
| amp / autocast (bf16) | loop.py 462 | run.py amp + amp_dtype | ✅ |
| wandb resume (run-id across array tasks) | loop.py 249-253 | run.py 295-296, 368 | ✅ |
| checkpoint payload (job_index, resume_state, normalizer, wandb_id) | loop.py save/load | run.py + checkpoint interop | ✅ |
| normalizer init from shard/tar (+ reset) | loop.py 546-581 | distill task build_loaders | ✅ |
| viz: pca_train + pca_val + graphs | loop.py `plot_multistep_pca`/`validate` | distill task `render_viz` + reused `validate()` | ✅ |
| validation phase (val_every, rank-0) | loop.py 689-734 | on_eval + distill evaluate | ✅ |
| seed (seed+rank, per-rank policy gen) | loop.py 323 | run.py 173 + policy | ✅ |
| SLURM-array shard-schedule resume | loop.py | distill (+now in1k) resume_state | ✅ |
| **matplotlib DDP-safety (MPLCONFIGDIR + Agg)** | `__main__.py` top | *was MISSING* | ✅ **FIXED** (run.py top) |
| **torch.compile gradient correctness (`backward_pass_autocast="off"`)** | loop.py:7 module top | *was MISSING* (default `same_as_forward`) | ✅ **FIXED** (run.py top) |
| optuna HPO (`n_trials`) | `__main__.py` study | *absent* | ⚠️ dropped capability |
| comet tracker | loop.py make_tracker | *rejected loudly* (`tracker=comet` errors) | ⚠️ intentional |
| `combo_kernels` inductor flag | loop.py (opt-in) | not threaded | ⚠️ minor (see below) |
| teacher `torch.compile` | loop.py `compile_teacher` | model-only | ⚠️ perf-only |

## The two real gaps found — both FIXED (this doc's commit)

1. **matplotlib DDP-safety.** The old `__main__.py` sets a per-rank/per-job `MPLCONFIGDIR`
   + `Agg` *before any matplotlib import*, because the default `~/.cache/matplotlib` on NFS
   races on the font cache across DDP ranks / concurrent jobs and **hangs**. The harness
   never did this, yet the distill viz path imports matplotlib — so a compiled production
   distill run *with viz under multi-rank DDP* could hang. Unexercised by the CPU tests and
   the short DDP smoke. Fixed at `harness/run.py` module top (mirrors `__main__.py`).
2. **`torch.compile` gradient correctness.** The old `train/loop.py` sets
   `torch._functorch.config.backward_pass_autocast = "off"` at import — its comment: since
   backward() runs outside autocast, compile's default `same_as_forward` *silently corrupts
   gradients*. `Config.compile=True` by default, so production compiles. The harness never
   set this (verified at runtime: `same_as_forward`), so **compiled distill through the
   harness would produce different gradients than the old loop** — invisible to the *eager*
   byte-parity digest and the CPU suite; only a compiled GPU run shows it. Fixed at
   `harness/run.py` module top. The production A/B (doc 11 §6, `compile=True`) now confirms
   this end-to-end.

## Flagged, NOT fixed (owner's call — none block the distill cutover)

- **optuna HPO (`n_trials>1`)** and **comet**: both **DEPRECATED** (owner-confirmed
  2026-07-25) — safe to drop, not a loss. `n_trials` defaults to 1 and no production run
  sets it (the old optuna wrapper is a single-trial no-op there); the harness doesn't import
  optuna at all. comet is not ported and `tracker=comet` raises loudly (doc 09); wandb + none
  are fully supported. Neither causes a problem in the harness today, so nothing to remove
  now — the cutover deletes them with the old loop. Only revisit if a stray import surfaces.
- **`combo_kernels`**: `Config.combo_kernels=False` default and exp22 does not set it, so
  production matches (harness leaves the inductor flag at its default). Only a gap if you
  set `combo_kernels=True` — the harness can't currently thread it. Trivial to add if wanted.
- **teacher `torch.compile`**: old compiles the frozen teacher too; harness compiles the
  model only → teacher forward is eager = **same numerics, slightly slower**. Perf, not
  fidelity.

## Verdict

The cutover would delete **nothing unimplemented** except optuna-HPO(`n_trials>1`) and
comet — both non-production, both flagged, neither silent. The two real hazards (mpl hang,
compiled-gradient divergence) are **fixed**. What remains before deleting the old loop is
purely the **numeric** confirmation: the production-scale A/B runs (doc 11 §6,
`harness_repro/`, run with `compile=True`), which now also exercise the compile fix.
