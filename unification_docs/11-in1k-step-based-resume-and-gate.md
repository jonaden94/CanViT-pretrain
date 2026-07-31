# 11 — in1k made step-based, per-task resume default, and the in1k top-1 gate

Follows docs 09 (CLI + checkpoint) and 10 (LR schedules + launchers + ade20k gate).
All work here is on `main`, committed together; the big-bang cutover is still gated.

## §1 Per-task `resume` default (owner decision, 2026-07-24)

The harness `RunSettings.resume` used to default `True` for every task. That is right for
**distill** (SLURM array jobs must continue across tasks) but a footgun for the **ade20k/
in1k probes**, whose launchers (`slurm/{ade20k,in1k}/train_*.sbatch`) are **single-job,
non-array**, and whose standalone counterparts have **no resume at all** — so resume-by-
default silently *continues* a one-shot probe re-run into a populated dir (this is what
contaminated the first ade20k gate run in doc 10).

**Decision:** `--opts.resume` now defaults to `None` → per-task default, mirroring the
existing `seed` pattern in `harness/cli.py`:

| task | default resume | why |
|---|---|---|
| distill | `True` | array jobs continue across tasks |
| ade20k | `False` | single-job probe; matches the no-resume standalone |
| in1k | `False` | single-job probe; matches the no-resume standalone |

Overridable with `--opts.resume True` (e.g. if in1k is ever arrayed — but see §3). Test:
`tasks/tests/test_run_wrappers.py::test_resume_default_is_per_task`. Verified live: the
in1k gate arms logged `FRESH mode: no checkpoint`.

## §2 in1k config made step-based (owner request, 2026-07-24)

`In1kConfig` was the only task config that spoke **epochs** (`epochs`, `warmup_epochs`,
`eval_every_epochs`, `limit_train_batches`). Since the train loader is an infinite
`resampled=True` WebDataset, "epoch" was only ever a *derived batch count* — so the
epoch vocabulary bought nothing and forced in1k to be the one task that required an
explicit `--opts.n-steps`. Now it matches `Ade20kConfig`:

| removed | added | old default → new default |
|---|---|---|
| `epochs=10` | `max_steps=200_000` | 10 epochs ≈ 200k steps @ batch 64 (1,281,167 imgs) |
| `warmup_epochs=0.5` | `warmup_steps=10_000` | 0.5 epoch ≈ 10k steps |
| `eval_every_epochs=1` | `val_every=20_000` | 1 epoch ≈ 20k steps |
| `limit_train_batches` | *(dropped)* | train length is now `max_steps` |

`in1k/train.py`'s nested `for epoch / for batch_i` loop became one `while step < max_steps`
loop over the infinite loader; the per-epoch train-mode reset (needed because `evaluate()`
flips the model to `eval()`) is now done after each validation. `In1kCmd.build()` derives
`n_steps`/`eval_every` from `cfg.max_steps`/`cfg.val_every` exactly like ade20k — the
`raise ValueError` for a missing `--opts.n-steps` is gone. `In1kRunTask.default_spec()`
uses `cfg.warmup_steps` directly. Full suite: **192 passed**.

The *training machinery* was already unified and step-based (all three tasks run the same
`run_training_loop`); this change only unified the config **vocabulary**. Other config
fields (`log_every`, `peak_lr`, `weight_decay`, `grad_clip`, `seed`, `amp`) were already
consistent across the three tasks — no sweeping rename was warranted.

## §3 in1k top-1 gate — PASS (job 15046042, 2026-07-24)

Same question as the ade20k gate: does the harness in1k task reproduce the standalone
probe? in1k is *seeded identically* on both sides (`torch.manual_seed(seed+rank)`) and the
harness task **shares** the standalone's loaders (`make_train_loader`/`make_val_loader`)
and eval fn (`in1k/train.evaluate`), so this is a strong reproduction test.

Full runs are ~60 GPU-h (200k steps; T=10/batch64 ≈ 1.1 s/step measured), so the gate is a
**capped code-path** run, not an accuracy run: frozen probe, `max_steps=1000`,
`warmup_steps=50`, `val_every=250`, T=10, batch=64, 25 val batches, over the uniform16
pretrained ckpt. Run-to-run noise is large at intermediate steps, so (like ade20k) compare
**bands**: n=3 each (`unification_docs/in1k_numeric_gate.sbatch`, arms 0–2 standalone,
3–5 harness, seeds 0/1/2). Harness evals at {0,250,500,750}, standalone at {250,500,750,
1000} — same gradient-step counts at the overlap {250,500,750}.

| step | standalone [min,max] mean | harness [min,max] mean | mean gap |
|---|---|---|---|
| 250 | [0.493, 0.598] 0.558 | [0.517, 0.589] 0.552 | −0.006 |
| 500 | [0.713, 0.724] 0.720 | [0.681, 0.799] 0.745 | +0.025 |
| 750 | [0.756, 0.789] 0.771 | [0.746, 0.838] 0.791 | +0.020 |

**PASS:** bands overlap at every comparable step; mean gaps ≤0.025, within the combined
seed spread; both plateau at ~0.77–0.79 top-1. Caveat: the harness shows a *wider* seed
spread than the standalone (esp. step 500) — not a failure (bands overlap), most likely
data-order nondeterminism (webdataset workers) across the two code paths; n=3 makes the
band estimate coarse.

## §4 Minor fidelity gap found (NOT fixed — flagged for the owner)

The first gate crashed the standalone arms with `AssertionError: total_steps (800) must
exceed warmup_steps (10000)` (my config error: capped `max_steps` but left the full-run
`warmup_steps`). The revealing part: the **standalone** `warmup_cosine_scheduler`
*asserts* `total_steps > warmup_steps` and fails loudly, while the **harness**
`warmup_cosine` (optim.py) *silently* runs the degenerate schedule (LR stuck in warmup →
top-1 ~0, no error). Same misconfig → crash vs silent-garbage. **FIXED (`dbf2eed`):**
`ScheduleSpec.errors()` now rejects `warmup_steps >= total_steps` for the decaying kinds,
so the harness fails loudly like the standalone. Test `test_schedule_warmup_must_be_below_total`.

## §5 in1k made resumable via the distill shard schedule (owner request, 2026-07-25)

Replaces the earlier `resampled=True` in1k train loader (infinite, no shard position → no
array resume) with the SAME resumable, shard-aligned schedule distill uses. Owner design
decisions: (1) **replace** (not coexist) — one loader path; (2) **keep a seeded
within-stream shuffle buffer** for in1k (distill has none).

- `in1k/data.py` reuses the pure `train.data.schedule` (`compute_shards_per_gpu` +
  `compute_schedule_slice`): a seeded global shard permutation, job `job_index` consuming a
  contiguous block, per-rank + per-worker (`split_by_worker`) split. A seeded
  `.shuffle(cfg.shuffle_buffer, seed=cfg.seed + job_index)` streaming buffer adds
  within-AND-across-shard mixing (bounded by the buffer window and the worker/job — it
  never crosses the job boundary, so resume stays shard-exact).
- `In1kConfig`: `steps_per_job` (shard window; None => single job of `max_steps`, enforced
  shard-aligned by `compute_shards_per_gpu`) + `shuffle_buffer`. `max_steps` is the LR-cosine
  horizon across ALL array jobs.
- `tasks/in1k/task.py` mirrors `DistillRunTask`'s resume wiring: `resume_start_step`
  (job_index → shard-aligned `start_step`), `_check_schedule_invariants` (refuse resume on
  changed world_size/batch/steps_per_job/samples_per_shard), `resume_state` (stores
  job_index). `In1kCmd`: per-job n_steps = `steps_per_job`, LR horizon = `max_steps`.
- Standalone `in1k/train.py` stays single-job (`job_index=0`, slice = whole run); the
  resumable array path is the harness.

Semantics (owner's mental model, confirmed): batch × steps_per_job is a whole multiple of
samples_per_shard, so a job ends shard-aligned; the next job auto-reads the next block via
the stored job_index; data order/coverage across resumed jobs = one monolithic job (distill
is order-identical; in1k is *sample*-identical per epoch but its per-job shuffle buffer
reorders finer, so not byte-identical). Last *partial* shard excluded. Per-job torch
re-seed means stochastic ops aren't a continuous stream (not bit-identical to monolithic).

**Validated on the cluster (2026-07-25):**
- **Resume (job 15048737):** job0 FRESH `job_index=0` steps 0–63 → job1 RESUME
  `start_step=64` `job_index=1` (next shard slice) steps 64–127. Auto-advance works.
- **Top-1 gate re-run (job 15048738), shard-aligned max_steps=1024:** harness/standalone
  bands overlap at every step, mean gap → 0.000 at the plateau (~0.80 top-1); same-seed
  pairs now coincide to ≤0.008 (tighter than the pre-schedule gate — both draw the identical
  seeded slice). Residual early wobble = cross-worker batch interleaving (`num_workers=8`),
  not a port bug. `unification_docs/in1k_resume_val.sbatch`, `in1k_numeric_gate.sbatch`.

## §6 Pre-cutover fidelity A/B — launchers ready (owner runs when queue is clear)

The one remaining hole before deleting `train/loop.py` is **production-scale** fidelity
(everything so far is component / short / probe scale). Ready-to-fire launchers now live in
`slurm/runs/harness_repro/` (see its README): they re-run existing old-loop configs
THROUGH THE HARNESS so the curves overlay on results you already have —
`distill-uniform16`, `distill-fovi`, `distill-fovi-teacherinit` (~100k steps each), and an
`ade20k-finetune` template. Pinned `PRETRAIN=bc63eee`. NOT submitted (owner's call;
sequence around the live exp22 arrays). Passing these clears claim 4 at scale and justifies
archiving the old repos.

## §7 Still open (owner-gated)

- **The production-scale A/B runs themselves** (launchers above are ready; not yet run).
- **Production in1k array launcher:** capability proven; a real array maps onto
  `harness_train.sbatch` (TASK=in1k, `--array`, `CFG_STEPS_PER_JOB`, `OPT_RESUME=True`) —
  not yet wired into a `runs/` script (single-job `train_in1k.sbatch` still covers probes).
- The big-bang cutover (deleting the old loop, repointing production launchers).
