# RESUME STATE — unified-harness build (updated 2026-07-24, overnight GPU session)

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
  `RunTask` Protocol + an **additive CLI** `python -m canvit_pretrain.harness.run --task {distill,
  ade20k,in1k} [--preset ...]`. The CLI deliberately does NOT replace `python -m canvit_pretrain.train`
  (that repoint is the owner-gated big-bang cutover).
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
