# RESUME STATE — unified-harness build (updated 2026-07-24)

## READ THIS FIRST AFTER A COMPACTION

**Commit stack on `main` (all additive; live trainers untouched; owner pushes):**
```
119059f  pass 3 — rest of single-GPU fidelity (seed-mode, compile, ckpt history, run_dir, distill PCA viz)
134bc51  pass 2 — core operational features (resume, SIGUSR1 ckpt, crash-guard, EMA/grad-norms)
88957e6  pass 1 — the unified harness itself (run() + three peer *RunTask wrappers)
480fd2a  <- pre-harness baseline (old distill loop intact here)
```
Everything is committed; the working tree's remaining changes are the OWNER's (exp22 `*.sh`)
plus generated `unification_docs/parity/record_*.json` — never stage those.

**Coding guidelines:** `~/.claude/CLAUDE.md` + this repo-session's `CLAUDE.md` now require invoking
the `andrej-karpathy-skills:karpathy-guidelines` skill at session start. Key corollary, learned here:
never claim "complete"/"drop-in" without stating the criterion AND checking it; when porting/replacing
code, read the target end to end — do NOT plan from a `grep` (that is exactly how pass 2 shipped incomplete).

**The WebDataset `job_index` multi-job resume is DONE** (pass 4, see its section below).

Pass 4's end-to-end re-read of `train/loop.py` then found FOUR more fidelity gaps (raw shards,
`reset_normalizer`, ckpt `model_config` on resume, scorer clipping) — **all four CLOSED in pass 5**,
see its section below.

Pass 6 then ported the full wandb metric richness + the distill validation phase — DONE, see below.

**IMMEDIATE NEXT TASK — DDP. This is the LAST item before the (owner-gated) cutover.**
BLOCKED: needs a multi-GPU node. Scope: manual grad-sync/broadcast for the in-rollout modules
(the loop already calls `joint.allreduce_grads()` when `is_dist`), the `ddp.all_reduce_mean` on every
logged scalar (train/loop.py 913 — the harness currently logs rank-0's local value), and the §9
support matrix (coupled policy-grad-into-backbone under DDP stays an assert).
SUCCESS CRITERIA: a 2-rank run where weights are identical across ranks after N steps and the loss
matches the 1-GPU run.
Cosmetic/observability, not correctness. Port from `train/loop.py`: per-branch distill EMA metrics,
flattened-config `log_parameters`, and the validation-phase PCA/curves + IN1k linear-probe readout
(the TRAINING-batch PCA viz is already ported). Owner constraint still in force: **distill viz only —
no ade20k/in1k viz.**
SUCCESS CRITERIA: a debug sbatch run logging to wandb project `canvit-pretrain` shows the same metric
set as an existing `jon_exp22` run for the same step range.

**Then, in order:** (a) **DDP — BLOCKED, needs a multi-GPU node** (verify: 2-rank, weights identical
across ranks, loss matches 1-GPU); (b) **big-bang cutover — needs DDP done + owner GREEN LIGHT**
(verify: REAL parity probe prints `9a0100a1a3de3acd`, full suite, and a GPU distill run matching an
existing one, BEFORE deleting anything).

---


Self-contained pickup point. Full design + running status: `07-unified-harness-design.md`
(read its "LOCKED DECISIONS" + "IMPLEMENTATION STATUS" first). Session memory (auto-loaded):
`unification-status.md`, `dataset-paths.md`.

## What this project is
Refactoring the three CanViT trainers (distill / ade20k / in1k) into ONE task-neutral harness
(`canvit_pretrain/harness/`) + three peer tasks (`canvit_pretrain/tasks/`), driven by an orthogonal
`TrainSpec`. All work so far is ADDITIVE — no existing trainer file has been modified, the live
distill/ade20k/in1k trainers are intact, nothing committed. All 7 sub-decisions locked (doc 07).

## Environment
- venv: `/user/henrich1/u25995/jonathan/repos/CanViT-pretrain/.venv-cu126/bin/python` (torch cu126).
- GPU: A100-80GB. Run from repo root `/user/henrich1/u25995/jonathan/repos/CanViT-pretrain`.
- Offline: prefix with `HF_HOME=/user/henrich1/u25995/.cache/huggingface HF_HUB_OFFLINE=1`.
  NO HF upload ever (D-G). Datasets: see `dataset-paths.md`.
- Big smoke checkpoints go to `/mnt/vast-nhr/projects/nib00021/jonathan/_harness_smoke_ckpts/`
  (disk-backed) — NOT the 4GB `/tmp` tmpfs (a full-finetune ckpt + AdamW state is >1GB and fills it).
- Distill parity digest = `9a0100a1a3de3acd` (CPU tripwire, `unification_docs/parity_probe.py`).

## DONE + VERIFIED (this session extends the earlier spine work)
### Harness spine (unchanged from before, all green)
`harness/{spec,rollout,optim,policy,checkpoint,loop,task}.py` + `tasks/{distill,ade20k,in1k}/task.py`
engine cores (Bound*Task). Distill parity byte-exact; config cross-product proven on CPU.

### NEW this session — run-level wrappers + single entry (unification task #15 → DONE)
- `harness/run.py` — **the single orchestration** `run(*, task, spec, settings)` (build_model →
  build_policy(joint) → apply_requires_grad → build_optimizer → build_loaders → build_selector →
  run_training_loop, with eval/log/ckpt hooks) + `RunSettings` (composed config, D-B) + a
  `RunTask` Protocol + an additive CLI. The CLI deliberately does NOT replace
  `python -m canvit_pretrain.train` (that repoint is the owner-gated big-bang cutover).
  **SUPERSEDED 2026-07-24 — see `09-cli-and-checkpoint.md`:** the curated `--task/--preset`
  argparse is gone, replaced by `harness/cli.py` (tyro over each task's own config, subcommand
  form `... harness.run distill --cfg.model.patcher-name foveated`). `RunSettings` is now
  DERIVED from the task config, and `to_hf` reads harness checkpoints.
- `tasks/{distill,ade20k,in1k}/task.py` — each gained a run-level `*RunTask` (DistillRunTask /
  Ade20kRunTask / In1kRunTask) implementing the full seam: caps/default_spec/build_model
  (from_pretrained_with_new_probe|head / create_model)/build_loaders (ade map / in1k wds+val /
  distill wds+normalizer-init)/build_selector/build_policy/trainable_param_groups (in1k head =
  norm+head)/bind/evaluate (ade mIoU-per-t / in1k top-1,5 / distill validate())/model_config.
- `harness/rollout.py` + `loop.py` — **threaded `spec.task_weight`** into `run_rollout` (design §4:
  `loss = task_weight*task_loss + policy_weight*policy_loss`; policy already scaled by
  `joint.rl_weight`). Guarded `== 1.0` so the parity path is byte-identical — **digest still
  `9a0100a1a3de3acd`** after the edit. (Closed task #18.)

### Validation (real cached pretrained CanViT, offline, A100)
- **51→53 harness/task CPU tests green** (`pytest canvit_pretrain/harness/tests canvit_pretrain/tasks/tests`):
  the earlier 42 + **9 new run-wrapper tests** (`tasks/tests/test_run_wrappers.py`) + **2 task_weight
  scaling tests** (`harness/tests/test_rollout_engine.py`). Parity test included → green.
- **Real-data GPU smokes (individual task cores)** — all PASS + checkpoint round-trip:
  `harness_realdata_train.py` (distill 150 steps, loss 2.00→1.97), `harness_realdata_ade20k.py`
  (probe 5.06→4.28, finetune 5.02→4.53), `harness_realdata_in1k.py` (frozen+finetune, finite).
- **Real-data GPU integration of the WHOLE `run()` path** — `unification_docs/harness_run_integration.py`:
  **8/8 configs PASS** across the full cross-product —
  ade20k {probe, finetune, **joint**}, in1k {frozen, finetune, **joint**}, distill {finetune ~1.96,
  **joint** ~1.96}. Every joint run trains head/probe + scorer together with `reward_frac` populated
  — the flagship NEW capability (joint on ade20k/in1k) the unification was for, now working end-to-end.
- **`Task.evaluate()` validated on real val data** — `unification_docs/harness_run_eval.py`: ALL 3 PASS.
  ade20k mIoU-per-t + in1k top-1/5 return valid-range metrics (~0 for the fresh heads, as expected);
  **distill `evaluate()` returned a real `val_metric=0.640`** (the `validate()` reuse ran end-to-end —
  cached teacher loaded offline, NOT the `{}` fallback). Eval mechanics for all three tasks confirmed.
- **Full `canvit_pretrain` suite green: 146 passed** (live trainers + harness + tasks) — additive edits
  broke nothing. Also `python -m canvit_pretrain.harness.run --task ade20k --preset probe` ran end-to-end.

### Single-GPU operational features (2026-07-24 pass 2) — CORE set DONE + validated
Ported the CORE task-neutral operational features (single-GPU resume / preemption / crash-safety).
**NOT yet a full drop-in** — the old `train/loop.py` CANNOT be deleted without regressing the "Still
missing" items below. All task-neutral, in the harness:
- **Resume**: `run()` does `find_latest` → `restore_into` (model/opt/sched/joint) BEFORE `build_loaders`
  (so distill's model-owned normalizer arrives initialized) → `start_step` via a `task.resume_start_step`
  hook (default `scheduler.last_epoch`; distill's SLURM-array `job_index` override landed in pass 4 below).
- **SIGUSR1 checkpoint-on-signal** (`install_sigusr1_handler` in run(); the loop polls a flag → saves at a
  safe boundary; `request_checkpoint()` is the non-signal test hook).
- **FAILED-marker + `cancel_slurm_array`** crash-loop guard (opt-in `use_failed_marker`).
- **Provenance** (`current_provenance()`) stamped into checkpoint metadata; `latest.pt` symlink.
- **EMA-smoothed loss + per-module grad-norms** in the log path (`ema_alpha`, `log_grad_norms`).
- **Dropped optuna** (owner: deprecated) — no sweep wrapper.
Validation: **run()-level resume PASSES on real data** (`harness_run_resume.py`: leg1→step5, resume→step9,
not restarted) + **6 new CPU ops tests** (`harness/tests/test_loop_ops.py`) + integration matrix re-run.
Built DDP-aware (rank-0-guarded); DDP itself still deferred (needs a multi-GPU node).

### Pass 3 (2026-07-24) — the rest of the single-GPU fidelity gap
Root cause of the pass-2 gap, recorded so it isn't repeated: pass 2 planned from a targeted `grep`
over `train/loop.py` instead of reading `training_loop` end to end, so features the grep didn't surface
were never in the plan (and I then overclaimed "feature-complete"). Pass 3 read it top-to-bottom.
Ported since:
- **seed-mode start** — `RunSettings.seed_ckpt` (weights-only, fresh opt/sched at step 0). Priority
  resume > seed > fresh, mirroring train/loop.py. Distill also honors `cfg.hf_seed_ckpt` in
  `build_model` (HF config wins over CLI defaults, else arch/weights mismatch).
- **`torch.compile`** (`RunSettings.compile`, uses the wrapper's own `.compile()` when present).
- **config + provenance HISTORY accumulated across resumes** (not a single snapshot) + distill stamps
  **`pretrain_view_scale`** / patcher / teacher info — what `to_hf` reads (the foveated footgun).
- **run_dir layout** (`RunSettings.run_dir` → `{run_dir}/checkpoints`, `{run_dir}/visualization`).
- **data% vs gpu% timing metrics** (`log_timing`).
- **distill PCA viz PORTED** (owner: distill only; no ade20k/in1k viz). New engine seam
  `run_rollout(collect_viz=, viz_task=)` + `task.viz_init`/`viz_frame`/`render_viz`, branch-0 only and
  **off by default so the parity path is untouched**. Reuses the existing `extract_sample0_viz` +
  `plot_multistep_pca` + `save_figure`, so content is identical and figures are saved **LOCALLY** to
  `{run_dir}/visualization/pca_train/step-N.png` (never uploaded — the old wandb path is not used).
  GPU-verified: real figures render correctly (init row + per-t rows, viewpoint boxes, hidden-PCA
  evolution, CosDis 0.9367→0.9613→0.9741).
  NOTE: a bug was caught here — the hooks live on the RUN-level task but the engine only sees the
  per-batch BOUND task, so they'd never fire; fixed via the explicit `viz_task` argument.

### Pass 4 (2026-07-24) — WebDataset `job_index` multi-job resume (the last CORRECTNESS gap) DONE
Production distill = a SLURM array where each task trains exactly `steps_per_job` steps over its own
shard slice and the next task must read the NEXT slice. The harness derived `start_step` from the
scheduler and passed a **hardcoded `job_index=0`** to `create_loaders`, so a real multi-job resume
would have silently re-read job 0's shards. Ported from `train/loop.py` (~354-417, ~499-512):
- distill `resume_start_step`: on the WebDataset path `start_step = (saved job_index + 1) *
  steps_per_job` (scheduler only on the sharded path); **raises** when the checkpoint has no
  `job_index`, and when `scheduler.last_epoch != start_step` (a mid-job/SIGUSR1 save, or a leg that
  ran a step count other than `steps_per_job`).
- `build_loaders`: passes the real `job_index`, then re-checks the schedule invariants
  (`world_size`/`batch_size_per_gpu`/`steps_per_job`/`samples_per_shard`) against the checkpoint and
  raises rather than silently shifting the slice offset.
- new task seam **`resume_state() -> dict`** (parallel to `resume_start_step`), written to checkpoint
  `metadata["resume_state"]` by `run()`; `{}` for ade20k/in1k.
Note: no separate `n_steps == steps_per_job` guard is needed — the mid-job assertion already catches a
short leg, and the loader's `total_samples == steps_per_job * batch` assert catches a long one.
`SLURM_ARRAY_TASK_ID` stays with the launcher rewrite: the index comes from the checkpoint, not the
environment, so array tasks chain without it.
Verified: **11 new CPU tests** (`tasks/tests/test_wds_resume.py`, incl. a parametrized case per schedule
input) + a **real 2-leg GPU run on the IN21k webdataset** (`unification_docs/harness_run_wds_resume.py`,
`ALL PASS`): leg 1 ran steps 0-127 on `shard-001751.tar` and saved `job_index=0`; leg 2 resumed at step
128, advanced to `job_index=1`, ended at 255, and read **`shard-002991.tar` — a disjoint slice**. All
four refusal cases raised (mid-job save; changed schedule inputs caught by both the job-boundary and the
invariant guard; missing `job_index`). Note a batch-size change ALONE isn't constructible on real data
(`steps_per_job * batch` must stay a multiple of `samples_per_shard`), so per-input isolation lives in the
CPU test. Parity digest still `9a0100a1a3de3acd`; 72 harness+tasks tests green; full suite 165.

### Pass 5 (2026-07-24) — the four gaps pass 4's end-to-end read surfaced. ALL FOUR CLOSED.
None of these were on any earlier list; they came out of reading `train/loop.py` top-to-bottom (the
discipline pass 2 skipped). Owner chose "close (a)-(d) first" and confirmed raw shards are still needed.
- **(a) Raw / no-feature WebDataset shards** (`train/loop.py` 566-575, 639-643) — the exp21
  on-the-fly-features path (memory `exp21-onthefly-stall-fix`). Those shards carry only jpg+json, so
  the frozen teacher makes the targets live. The harness called `init_normalizer_stats_from_tar`
  unconditionally and `bind()` did `raw_patches.to(...)` on a `None` → **it CRASHED**; every smoke
  passed only because exp22 uses with-features shards. FIXED: `build_loaders` dispatches on
  `train.has_features` (`init_normalizer_stats_from_tar_raw` for raw), and `bind()` computes targets
  via a new `_teacher_targets` helper when the batch carries none.
  Verified on the REAL no-feature IN1k webdataset (`unification_docs/harness_run_raw_shards.py`,
  `ALL PASS`): stats "computed live" from 256 samples, loss 2.033→2.014, checkpoint round-trips.
- **(b) `cfg.reset_normalizer` was ignored** (`train/loop.py` 546-548) → now
  `if cfg.reset_normalizer or not scene_norm.initialized`.
- **(c) On RESUME the checkpoint's `model_config` didn't override `cfg.model`** (`train/loop.py`
  254-261). FIXED: `run()` resolves + loads the resume checkpoint **before** `build_model` and passes
  `prior_model_config=`; distill rebuilds its arch from it (`dacite.from_dict`) and RESUME now
  correctly beats `hf_seed_ckpt`. Distill's `model_config` gained a `"canvit"` entry holding the full
  `asdict(cfg.model)` so the round-trip is lossless (nothing read `model_config` before, so this is
  additive). ade20k/in1k accept and ignore the kwarg — their arch comes from `cfg.model_repo`.
- **(d) Joint runs clipped the scorer TOGETHER with the model** (`train/loop.py` 874-878 clips
  `trainable` and `joint.scorer.parameters()` **separately**, each to `grad_clip`; one joint norm
  couples their magnitudes). FIXED in `harness/loop.py`: `trainable` now excludes the scorer's params
  by id, and the scorer gets its own clip **after** `allreduce_grads()` (the old order).
Verified: **8 new CPU tests** (`tasks/tests/test_distill_data.py` 7 + `harness/tests/test_loop_ops.py`
clip-split; full suite 165 → **173 passed**) + the real raw-shard GPU run + **8/8 integration configs
still PASS** (both joint configs with `reward_frac` populated) + parity digest `9a0100a1a3de3acd` + a
re-run of the 2-leg wds resume (`ALL PASS`) after the `run()` reordering.

### Pass 6 (2026-07-24) — full wandb metric richness + the distill validation phase. DONE.
The harness logged ~6 scalars; `train/loop.py` logs a whole EMA'd series set. Ported:
- **Per-branch series** `full/…` + `random/…` (loss, scene_patches_loss, scene_cls_loss, and the four
  cos-sims). The engine already returned per-branch `BranchResult`s, so this needed only two OPTIONAL
  task hooks — `glimpse_metrics(loss)` (per-glimpse scalars, averaged over the branch) and
  `final_metrics(readout)` (one-shot on the last readout) — plus a neutral
  `harness.loop.branch_metrics()` that groups by t0 type and averages, exactly like the historical
  `aggregate(list[BranchMetrics])`. Hookless tasks (ade20k/in1k) just get `{type}/loss`.
  The raw-space cosines compare against the TRUE raw teacher targets (passed through
  `BoundDistillTask(metric_refs=…)`), not `destandardize(standardize(x))`, matching train/loop.py.
- **EMA over every series, not just total_loss**, logged under the plain names (train/loop.py logs
  ONLY EMAs). The instantaneous total is kept as `total_loss_raw` — the one key we log that the old
  loop did not.
- **`train/` namespace** in `run()`'s tracker payload (`grad_norm/…` keeps its own), plus `train/lr`,
  `train/grad_norm` (the clip's total norm), `train/continue_prob`, `train/prime_on_policy`.
- **`log_parameters`**: the task's flattened config + `train_spec` + SLURM job id + trainable/total
  param counts (171 hyperparameters on a distill run).
- **The distill VALIDATION phase**, which was silently gutted: `evaluate()` built a throwaway
  `tracker="none"` and a `tempfile.mkdtemp()` run dir, so every `val/…` series and figure was
  DISCARDED. The `evaluate` seam now takes `tracker=`/`run_dir=` and distill passes the real ones,
  plus `probe` (IN1k linear probe), `log_curves`/`log_pca` on their historical val-count cadences,
  `foveated_eval_scale` (training scale for mode='fixed'), `log_spatial_stats` and `teacher_name`.
Verified on real data by `unification_docs/harness_metric_parity.py` (**ALL PASS**, GPU, and it
records instead of uploading so nothing is published): **92 metric keys** — every `train/loop.py`
series present, the `full/`+`random/` sets, 10 per-module grad-norm series, `val/scene_cos_*_t0..t9`,
`val/cls_cos_*_t0..t9`, **`val/in1k_tts_top1_t0..t9`** + `val/in1k_teacher_top1`, spatial stats — plus
171 hyperparameters and a PCA figure on disk. Parity digest `9a0100a1a3de3acd` held (rollout.py was
touched); full suite **174 passed**.
NOTE: the val-count cadence assumes the harness's `eval_every` == the config's `val_every`; the
launcher sets both at the cutover.

**STILL MISSING before the old loop can be deleted:**
- **DDP** (needs a multi-GPU node): manual grad-sync/broadcast, **all-reduce mean of each logged
  scalar** (train/loop.py 913 wraps every EMA in `ddp.all_reduce_mean`; the harness logs rank-0's
  local value), the §9 support matrix. This is now the ONLY remaining item before the cutover.

## NEXT STEPS (in order)
1. (optional) Port ADE20K's `WarmupOneCycleLR` into `harness/optim.py` (currently onecycle raises;
   the run wrappers default to warmup_constant/cosine). Not blocking.
2. (optional) Distill `evaluate()` reuses `validate()` (confirmed working on real data — val_metric
   returned; teacher loads offline). Viz/PCA/curves are off in the run-wrapper call; wiring those
   (+ probe/IN1k-acc during distill val) lands naturally at the cutover when loop.py is consolidated.
3. **DDP** (`harness/ddp.py`) — owner said SKIP for now. Single-GPU only. The loop already calls
   `joint.allreduce_grads()` when `is_dist`; the §9 support matrix + manual backbone AllReduce are TODO.
4. **(Needs owner GREEN LIGHT — destructive) big-bang cutover:** repoint `python -m canvit_pretrain.train`
   at `harness.run`, flip `train/step.py`→`run_rollout` (re-confirm the REAL parity probe
   `9a0100a1a3de3acd` + the 93-test pretrain suite), delete the old loops (`ade20k.train`,
   `ade20k.rl_train`, `in1k.train`, distill `train/loop.py`+`step.py`), rewrite `slurm_nhr/` launchers.
   Then the GPU acceptance gate. Everything above is additive scaffolding that makes this mechanical.

## GUARDRAILS (still in force)
- Do NOT commit/push unless the owner explicitly asks (everything currently uncommitted, for review).
- Do NOT do the destructive cutover (step 4) without an explicit green light.
- DDP skipped for now (owner). Single-GPU only.
- Keep the old distill loop until the cutover reproduces the digest, THEN delete (parity safeguard).

## Key files touched this session (all additive / parity-safe)
- NEW: `harness/run.py`, `tasks/tests/test_run_wrappers.py`, `unification_docs/harness_realdata_ade20k.py`,
  `harness_realdata_in1k.py`, `harness_run_integration.py`.
- APPENDED (run-level wrapper class only, cores unchanged): `tasks/{distill,ade20k,in1k}/task.py`
  (+ `ViewpointType` import each).
- EDITED (parity-safe, digest re-verified): `harness/rollout.py` + `harness/loop.py` (task_weight).
