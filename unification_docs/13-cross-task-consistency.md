# 13 — cross-task consistency audit (ade20k / in1k / distill under one harness)

**Question (owner, 2026-07-27):** doc 12 audited ONE axis — the old distill loop vs the
harness. It never asked whether the three tasks are consistent *with each other*, or
whether the ade20k/in1k **standalone** trainers lost anything on the way into the harness.
The ADE20K segmentation viz turned out to be missing on exactly that unchecked axis
(restored in a79e38b), and the trigger for this audit was the same class of bug: ade20k
had **no way to name its wandb run**, so exp24's three probes were all called `ade20k`.

Owner's rule for this audit: *"where stuff is not identical, this should be an extension,
not a regression … and check whether identicality to the earlier version is maybe still an
artifact that should be gotten rid of."*

**Method:** read end to end — `harness/cli.py`, `harness/run.py`, `harness/loop.py`,
`harness/rollout.py`, `harness/checkpoint.py`, all three `tasks/*/task.py`, all three
configs, both standalone trainers (`ade20k/train.py`, `in1k/train.py`), the old distill
loop's identity block, `slurm_nhr/harness_train.sbatch`, and the exp24/exp25 launchers —
then cross-tabulate every user-facing knob and every logged series per task. Runtime facts
verified where a docstring was ambiguous (tyro duplicate-flag precedence, strictness of
`restore_into`, whether `make_ade20k_loaders` shards by rank).

## Knob matrix (after the fixes below)

| knob | distill | ade20k | in1k | notes |
|---|---|---|---|---|
| `run_group` / `run_name` / `logs_dir` | ✅ cfg | ✅ cfg (**new**) | ✅ cfg (**new**) | one `_identity()` in cli.py resolves all three |
| wandb run name | cfg.run_name | cfg.run_name (**was hardcoded `"ade20k"`**) | cfg.run_name (**was the constant `"in1k-clf"`**) | both standalones honor it now too |
| run dir (`visualization/`) | derived | derived (**was `--opts.run-dir` only**) | derived (**same**) | `logs_dir/run_group/run_name` |
| ckpt dir | `run_dir/checkpoints` | ditto, else flat `probe_ckpt_dir` | ditto, else flat `clf_ckpt_dir` | **was: flat dir ALWAYS → cross-run clobber** |
| `seed` | ✅ cfg | ✅ cfg (**new**) | ✅ cfg | standalone ade20k had NO seed at all |
| `--opts.seed/-eval-every/-n-steps/-ckpt-every/-resume` | ✅ | ✅ | ✅ | already uniform |
| `--opts.ema-alpha` | ✅ (**new**) | ✅ (**new**) | ✅ (**new**) | was reachable only via distill's cfg |
| `limit_val_batches` | `val_samples` | ✅ (**new**) | ✅ | both ade20k entry points honor it |
| `viz_every` | ✅ pca_train/pca_val/graphs | ✅ seg_train/seg_val | ➖ no figures | see (f) |
| `steps_per_job` (array resume) | ✅ shard schedule | ➖ single job | ✅ shard schedule | see (g) |
| `compile` | ✅ cfg | ⛔ refused loudly | ⛔ refused loudly | see (d) |
| `seed_ckpt` (local weights) | ✅ cfg | ❌ | ❌ | see (e), deliberate |
| `tracker` / `wandb_project` / `_entity` / `_dir` | ✅ | ✅ | ✅ | `$WANDB_PROJECT` now defaults for distill too |
| DDP | ✅ shards by rank | ⛔ **refused** (3 guards) | ✅ shards by rank | see (c) |

## Fixed here

1. **Run identity, uniformly.** `run_group` / `run_name` / `logs_dir` added to
   `Ade20kConfig` + `In1kConfig`; `cli.py::_identity()` resolves name + run dir + ckpt dir
   + tracker identically for all three commands. `In1kConfig.run_name` no longer defaults
   to the constant `"in1k-clf"`, and unnamed runs get `{task}_{timestamp}` instead of a
   shared string. Both standalone trainers honor `cfg.run_name` (tracker name + checkpoint
   subdir) — in1k's field was previously read by the harness ONLY, so the standalone
   silently ignored it.
2. **Checkpoint clobber (real bug).** ade20k/in1k passed `ckpt_dir = opts.ckpt_dir or
   cfg.<flat>_ckpt_dir`, and those config defaults are never None — so `run_dir/checkpoints`
   could never win and two runs with default settings overwrote each other's `best.pt` /
   `step-N.pt`. Every exp24/exp25 launcher worked around it with a hand-written
   `OPT_CKPT_DIR=logs/$RUN_GROUP/$RUN_NAME/checkpoints`. Precedence is now
   `--opts.ckpt-dir` > `run_dir/checkpoints` > flat legacy dir (reached only without a
   `run_group`, i.e. the old behavior for anyone not using the convention).
3. **`Ade20kConfig.seed`.** The probe had no seed field anywhere: the harness hardcoded
   `seed=0` and the standalone never called `manual_seed` at all, which is why bit-parity
   gates against it were impossible (memory: `ade20k-probe-is-unseeded`). Both entry points
   honor `cfg.seed` now; 0 keeps what the harness was already doing.
4. **`timing/val_seconds`** — logged by `run()` for every task, under the key
   `ade20k/train.py` used. For a probe, validation (63 batches every 500 steps) is the
   dominant non-training cost; the series had been dropped entirely.
5. **`limit_val_batches` for ade20k**, mirroring in1k, honored in the standalone val loop
   and in `tasks/ade20k/task.py::evaluate`.
6. **`--opts.ema-alpha`** so the log-smoothing knob exists for all three (distill's
   `cfg.ema_alpha` remains its default; the probes' remains the harness default 0.1).
7. **Silent no-checkpoint run.** `train/loop.py:157` asserted `run_group is not None`. The
   harness instead derived `ckpt_dir=None` and trained happily, throwing every weight away
   at exit. Now a loud `WARNING` (not an error: the parity/smoke scripts deliberately run
   without a run dir).
8. **`$WANDB_PROJECT` / `$WANDB_ENTITY` defaults** on `Config`, which the two probe configs
   already had. Every launcher passes `CFG_WANDB_PROJECT` explicitly, so this changes only
   what a hand-run distill job does (land in the default project instead of asserting).
9. **`harness_train.sbatch`**: one identity block for all tasks (`--cfg.run-group /
   --cfg.run-name / --cfg.logs-dir`), `RUN_GROUP`/`RUN_NAME` required for all three, and
   the ade20k/in1k `--opts.run-dir` special case deleted. Existing launchers keep working
   unchanged: an explicit `CFG_*`/`OPT_*` lands in `_ARGS`, i.e. AFTER these flags, and
   tyro takes the last occurrence (verified).

10. **ade20k DDP is now REFUSED, not merely wrong** (owner's call, same day). New
    `TaskCaps.supports_ddp` (default True; ade20k False) → `check_spec` errors before the
    model is built; `Ade20kRunTask.build_loaders` raises independently, at the point where
    the reason lives; and `harness_train.sbatch` fails immediately on `TASK=ade20k` with
    `NGPU>1`, naming the variable that is wrong. See (c) for what a real multi-GPU ade20k
    would still need.
11. **Wrapper-level `compile` is refused for the probes.** New `TaskCaps.supports_compile`
    (default **False**; distill True, since it is the only task that calls the wrapper's
    forward). `run()` now raises instead of compiling nothing, so a future
    `--cfg.compile`-style flag on a probe config cannot silently pay the warmup and change
    zero. See (d).

## Flagged, NOT fixed (owner's call)

(a) **ade20k train mIoU.** `ade20k/train.py` logs `train_miou_mean` every `log_every`
(mean over per-timestep accumulators, reset each window); specialize additionally logged a
`train_miou_curve` figure. The harness logs **no** train-side mIoU. Restoring it faithfully
needs a 150-class confusion matrix per glimpse per step *plus* a window-reset signal the
neutral loop does not expose to tasks (`glimpse_metrics` receives only the loss object,
`final_metrics` only the last readout). Cheap approximations exist (EMA of a single-batch
mIoU via `final_metrics`, or pixel accuracy) but neither is the old series. Val mIoU per
timestep (`eval/miou_t{0..9}`) is logged every `val_every`.

(b) **`best_val_miou_t{t}` for every t.** The standalone logged a running best per
timestep; the harness mirrors only the single `best_metric` (`eval/best_miou_final`). Both
are a running max over series wandb already stores, so this is UI convenience.

(c) **Multi-GPU ade20k is unimplemented (now refused — see fix 10).** `make_ade20k_loaders`
builds a plain map-style `DataLoader(shuffle=True)` and takes no `world_size`/`rank`, and
`Ade20kRunTask.build_loaders` dropped both. Under DDP, ranks — seeded `seed + rank` — would
draw *different but overlapping* samples from the whole dataset rather than disjoint shards:
no error, just an effective batch that is not what the config says. Statistically benign for
the standard recipe (40k × 16 = 32 epochs over 20k images) and never exercised (every exp24
probe ran on 1 GPU), so the resolution is a refusal, not a port. Actually supporting it
needs a `DistributedSampler` **and** `set_epoch` plumbing through `run.py::_infinite`
(without `set_epoch`, `DistributedSampler` repeats the same order every epoch — a different
silent flaw), plus rank-aware validation. Nothing needs it today.

(d) **Compiling the probes for real** (the refusal in fix 11 only closes the trap).
`run()` compiles what the harness holds — the wrapper's `forward` — which only distill
calls; ade20k/in1k step `seg.canvit(...)` / `seg.head(...)`, so wrapper-level compile is a
no-op for them. Making it work means compiling `.canvit` explicitly. Plausibly worth it
(a probe is 8h and the frozen backbone forward, ×10 glimpses/step, is essentially the whole
cost), but it is a change plus a measurement, not a flag: the rollout carries recurrent
state and a fresh viewpoint per glimpse, so the risks are recompilation/graph breaks eating
the gain. Gate it as: 20 steps with and without, compare wall-clock AND step-0 mIoU
(unchanged), on the same GPU.

(e) **`seed_ckpt` for the probes: deliberately absent.** `restore_into` does a strict
`load_state_dict`, and a distill checkpoint's keys do not match a `CanViTForSemanticSegmentation`
/ `…ForImageClassification` wrapper, so exposing it would only produce key errors. Seeding
a probe from pretraining goes through `python -m canvit_pretrain.checkpoint.to_hf` →
`cfg.model_repo`, which is what exp24/exp25 do. (`--opts.resume` covers continuation from a
probe's own checkpoint.)

(f) **in1k renders no figures.** Neither the standalone nor specialize ever did, and there
is no obvious analogue of the seg overlay (a glimpse-trajectory + top-5 panel would be a
new feature, not a restoration). Not a regression.

(g) **ade20k has no `steps_per_job`.** Single-job by design: the map-style ADE20K loader has
no shard schedule to resume into, so cross-job continuation is plain checkpoint resume
(`--opts.resume True`). exp24 fits 40k steps in one 8h job.

(h) **distill's viz cadence still needs two knobs** (`val_every × viz_every_n_vals`) — an
old-loop artifact kept so an old config reproduces its cadence. `--opts.viz-every`
overrides it directly for anyone who wants one knob.

(i) **The restored ADE20K seg viz is harness-only.** `ade20k/train.py` renders no figures
(specialize's did). Wiring it into the standalone is throwaway work: the standalones are
deleted at cutover.

(j) **`resume` default differs by design** (distill True, probes False) — array-job
continuation vs a single-job probe where a re-run into a populated dir must start fresh.
Principled asymmetry, not an artifact.

Minor/cosmetic, left alone: `Config.device` is a `torch.device` while the probe configs use
`str`; distill's `val_samples` vs the probes' `limit_val_batches` naming.

## Verification

- `pytest canvit_pretrain/{harness,tasks,ade20k,in1k} train/test_seams.py` — 117 passed,
  including a new `test_run_identity_is_uniform_across_tasks` (name/run_dir/ckpt_dir for all
  three tasks, the no-`run_group` fallback, the `--opts` overrides) and two guard tests:
  `check_spec` errors under `is_dist` for a `supports_ddp=False` task, `build_loaders`
  raises at `world_size=2`, and the per-task `supports_compile` values.
- The sbatch guard was run against the real script's case block: `TASK=ade20k NGPU=2` exits
  1 with the message; `TASK=in1k NGPU=2` proceeds.
- CLI matrix by hand for all three tasks: named runs resolve to `LOGS_DIR/GROUP/NAME` +
  `ckpt_dir=None` (so `run()` derives `…/checkpoints`); unnamed runs get `{task}_{ts}` and
  fall back to the flat legacy dir; a duplicated `--cfg.run-name` takes the last value
  (the `CFG_RUN_NAME`-override path the sbatch relies on).
- **Real ADE20K harness run, A100-80GB**, passing only `--cfg.run-group _smoke_identity
  --cfg.run-name ade20k-derived-dirs --cfg.logs-dir …` (no `OPT_CKPT_DIR`, no
  `--opts.run-dir`) → `logs/_smoke_identity/ade20k-derived-dirs/{checkpoints/{best.pt,
  step-3.pt,latest.pt}, visualization/seg_train/step-{0,1,2}.png, visualization/seg_val/
  step-0.png}`, exit 0. `eval (4.6s)` / `eval (0.6s)` in the log = the new val-timing line,
  and 0.6s for a 2000-image val set = `limit_val_batches` honored.
- **Real standalone run** (`python -m canvit_pretrain.ade20k --run-name … --seed 7
  --limit-val-batches 2`): tracker name `smoke-standalone-named`, checkpoints under
  `probe_ckpt_dir/smoke-standalone-named/`, val capped — i.e. none of the three new
  standalone knobs is a dead flag.
- **Seed actually seeds** (3 real runs, `--max-steps 1`): same seed → the saved probe head
  is **bitwise identical** (max|Δ| = 0.0) and step-0 val mIoU matches to 6 decimals across
  all timesteps; seed 7 vs 8 → max|Δ| = 14.0 and different mIoU. Before this the standalone
  called `manual_seed` nowhere, so neither held.
