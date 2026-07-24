# 09 — Full CLI + checkpoint interop (2026-07-24)

Closes the two hard blockers the 7-claims audit found. Neither was architectural; both
were wiring on top of an already-validated core.

## 1. CLI: hand-rolled argparse → `tyro` over the task config

`harness/cli.py` (new) runs `tyro` over each task's OWN config dataclass — the idiom the
three standalone entry points already use (`train/__main__.py`, `ade20k/__main__.py`,
`in1k/__main__.py`). `harness/run.py::main` is now a 3-line delegator; `_build_task` and
`_resolve_spec` (the curated-subset argparse) are deleted.

**Invocation is now subcommand-based:**

```bash
# foveated pretraining — the config that was UNREACHABLE before (no --model.* flags)
python -m canvit_pretrain.harness.run distill \
    --cfg.model.patcher-name foveated --cfg.foveated-scale.mode per_rollout \
    --cfg.webdataset-dir "$WDS" --cfg.val-dir "$VAL" \
    --cfg.run-group fovi --cfg.init-backbone-from-teacher

python -m canvit_pretrain.harness.run ade20k --preset probe --cfg.model-repo "$REPO"
python -m canvit_pretrain.harness.run in1k  --preset joint --rl.use-rl True --opts.n-steps 50000
```

Now reachable (all were missing): the whole `--cfg.model.*` tree incl. **`patcher-name`
{uniform,foveated,square}** and the per-patcher geometry, `--cfg.foveated-scale.*`,
`--cfg.patch-stride`, `--cfg.canvas-patch-grid-size`, `--cfg.init-backbone-from-teacher`,
`--cfg.seed-ckpt` / `--cfg.hf-seed-ckpt`, `--cfg.run-group` / `--cfg.logs-dir`,
`--cfg.warmup-steps` / `--cfg.cosine-total-steps` / `--cfg.peak-lr`, `--cfg.compile`,
and `--rl.*` for ade20k/in1k.

### The task config is the single source of truth

`RunSettings` is **derived** from it, so a config that reproduced a run under the old
entry point reproduces it here and there is no second place to set the same thing:

| RunSettings | comes from |
|---|---|
| `n_steps` | distill `cfg.steps_per_job`, ade20k `cfg.max_steps` (in1k: `--opts.n-steps` required) |
| `eval_every` | `cfg.val_every` |
| `log_every` / `grad_clip` / `amp` / `seed` / `device` | same-named cfg fields |
| `compile` / `ema_alpha` / `seed_ckpt` | same-named cfg fields |
| `tracker` / `wandb_*` | same-named cfg fields |
| `run_dir` | `cfg.logs_dir / cfg.run_group / cfg.run_name` (train/loop.py 157-161) |

Only knobs with **no** config counterpart live in `--opts.*` (`HarnessOpts`):
`n_steps` override for smokes, `eval_every` override, `ckpt_every`, `viz_every`,
`start_step`, `ckpt_dir`, `resume`, `signal_checkpoint`, `use_failed_marker`,
`amp_dtype`, `log_grad_norms`, `log_timing`.

**This kills the `n_steps` footgun**: `n_steps` was defaulting to 100 independently of
`steps_per_job`, so a forgotten flag would train 100 steps and corrupt the WebDataset
shard schedule on resume. They are now coupled by construction.

### Deliberate hard failures (loud, not silent)

- **in1k without `--opts.n-steps`** → `ValueError`. `In1kConfig` schedules in *epochs*
  while the harness loop counts *steps*; there is no faithful automatic conversion, so
  it refuses to guess rather than silently run the wrong length.
- **`--cfg.tracker comet`** → `NotImplementedError`. Comet was never ported; previously
  it silently produced no tracker at all.

## 2. Checkpoint interop: harness ckpts are convertible again

Two writers exist: legacy `train/loop.py` (flat `state_dict` + top-level fields) and the
harness (`model_state` + nested `metadata`). `checkpoint/to_hf.py` grew
**`normalize_schema(raw)`**, which maps the harness shape onto the flat one and passes
legacy payloads through untouched (identity — asserted by test).

Fixed by this:
- Harness checkpoints previously **KeyError'd** on `raw["state_dict"]`.
- Worse, `training_config_history` was not top-level → `extract_pretrain_view_scale`
  returned `None` → **silent `pretrain_view_scale=None`**, exactly the foveated OOD
  footgun the field exists to prevent. Now recovered correctly.
- `extract_pretrain_view_scale` reads **both** history shapes: legacy FLAT
  (`foveated_scale.mode`, via `train/loop.py:flatten_dict`) and harness NESTED
  (`foveated_scale: {...}`).
- An ade20k/in1k checkpoint fed to the pretraining converter now raises a clear
  `KeyError` naming the task instead of producing a broken export.

`DistillRunTask.checkpoint_metadata` now also records **`canvas_patch_grid_sizes`**
(off the model), **`patch_stride`**, and the full **`foveated_scale`** dict. Without
`patch_stride` an overlapping-patch model (exp21) is unrebuildable — the patch-embed
conv comes back non-overlapping and the weights mismatch.

**Not done, deliberately:** loading OLD-format checkpoints into the harness. The owner
explicitly accepted dropping that ("we cannot expect this kind of backwards compatibility
after such a big refactor"). In-flight exp22 runs must finish under the old entry point.

## 3. wandb run continuity across a SLURM array

`run.py` now round-trips the wandb run id (`metadata["wandb_run_id"]` →
`prev_wandb_id`), matching `train/loop.py` 249-253/754. A 245-job array is ONE
experiment; without this each task opened its own run and the curves came out in 245
pieces.

## 4. Validation placement matches the old loop

`harness/loop.py` now validates **before** the update, at `step % eval_every == 0`
**including step 0** and excluding the job's last step (`train/loop.py:689`). Previously
it ran after the update and skipped step 0. Not cosmetic: it makes `step` the number of
updates the evaluated weights have had, so a resumed array task's val curve continues
where the previous one ended, and step 0 is the only record of the seeded model before
this job touched it.

Also added `torch.set_float32_matmul_precision("high")` in the CLI entry (TF32 — the old
`train/__main__.py` sets it before building anything; it changes speed and numerics).

## Verification

- Full `canvit_pretrain` suite green, **parity digest unchanged** (the loop edit is
  behaviour-preserving for the distill stream).
- New CPU tests: nested-config parsing (fovi flags land), config-derived RunSettings,
  in1k step-budget refusal, comet refusal, converter schema round-trip, patch_stride
  preservation, non-distill rejection, metadata completeness.
- GPU: 8/8 `harness_run_integration.py` configs re-confirmed after the changes
  (job 15037795), and a 2×A100 DDP distill run through the NEW CLI (job 15037796).

## Still open

- **Numeric quality gate through the harness** (harness ade20k probe mIoU vs the
  standalone's reference numbers). This was blocked by the CLI gap and is now runnable.
- Dedicated `slurm_nhr/` launchers pointing at `harness.run` — cutover work, still gated.
- ade20k `warmup_onecycle` (`NotImplementedError`) so harness ade20k can reproduce
  specialize's LR schedule exactly.
