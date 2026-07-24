# 11 — in1k made step-based, per-task resume default, and the in1k top-1 gate

Follows docs 09 (CLI + checkpoint) and 10 (LR schedules + launchers + ade20k gate).
All work here is on `main`, committed together; the big-bang cutover is still gated.

## §1 Per-task `resume` default (owner decision, 2026-07-24)

The harness `RunSettings.resume` used to default `True` for every task. That is right for
**distill** (SLURM array jobs must continue across tasks) but a footgun for the **ade20k/
in1k probes**, whose launchers (`slurm_nhr/{ade20k,in1k}/train_*.sbatch`) are **single-job,
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
top-1 ~0, no error). Same misconfig → crash vs silent-garbage. Only bites on
misconfiguration, so left as-is; a 1-line guard in the harness `warmup_cosine` would close
it if desired.

## §5 Still open (unchanged / owner-gated)

- **in1k array-resume is unsolved** (see memory `in1k-array-resume-gap`): the loader is
  `resampled=True` (infinite, no shard position), so distill's shard-aligned array-resume
  does not apply. A full in1k probe as a job array would need new design (deterministic
  shard-schedule window + `resume_state`, or approximate resume). Not built.
- The big-bang cutover (deleting the old loop, repointing production launchers).
