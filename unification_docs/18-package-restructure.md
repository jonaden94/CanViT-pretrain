# 18 — Package restructure: `train/` → `harness/` + `distill/`

**Date:** 2026-07-31 (same day as [17 — harness consolidation](17-harness-consolidation.md)
and the `CanViT-pretrain` → `CanViT-train` rename).
**Nature:** pure code motion. No logic edited, no behavior changed — see *Verification*.

## The problem

The tree did not match its own organizing principle. Expected: *`harness/` = shared,
`<task>/` = task-specific*. Actual: a folder called **`train/`** did double duty, holding

- the **shared substrate** (`config.FoveatedScaleConfig`/`JointPolicyConfig`, `viewpoint`,
  `selector`, `rl`, `joint`, `ema`, `dist`, `tracker`, `scheduler`, `utils`,
  `data/schedule`, the task-agnostic `viz` leaves), **and**
- **distill's own task code** (`config.Config`, `task.py`'s `DistillTask`, `model`,
  `probe`, the in21k `data/`, distill's validation `viz/`).

Plus `tasks/<t>/task.py` sat apart from `<t>/`, which reads as duplication but is not
(the adapter vs the task's library).

**Why it was that way, and it was not a design decision.** The package began as
`canvit_pretrain` — distill *was* the whole repo, so distill's files never needed a
task-name prefix. When specialize and RL were merged in, the newcomers reached *into*
`train/` for its primitives rather than the primitives moving out. Through that whole
period the gate was numeric parity against the old implementations, and the cheapest way
to keep that gate meaningful was for the harness to **import the old modules unchanged**:
every file move is a place a behavior change can hide inside a diff you can no longer
read. That caution paid for itself — the parity tests caught the `or 512` sentinel
clobber, the probe-BN pollution, and the 0.8× policy gradient. Tidying the tree mid-hunt
would have been reckless. Once the old implementations were deleted (doc 17), the reason
to hold still expired.

## The move

| from | to |
|---|---|
| `train/{viewpoint,selector,rl,joint,ema,dist,tracker,scheduler,utils}.py` | `harness/` |
| `train/config.py` — `FoveatedScaleConfig`, `JointPolicyConfig` | `harness/config.py` |
| `train/config.py` — `TEACHER_*`, `Config` | `distill/config.py` |
| `train/data/schedule.py` (shared: distill + in1k) | `harness/schedule.py` |
| `train/viz/{pca,disk,metrics}.py` (pure leaves, used by ade20k too) | `harness/viz/` |
| `train/task.py` (`DistillTask`, the per-step loss) | `distill/loss.py` |
| `train/{model,probe}.py`, rest of `train/data/`, rest of `train/viz/` | `distill/` |
| `tasks/{distill,ade20k,in1k}/task.py` | `{distill,ade20k,in1k}/task.py` |
| `tasks/tests/*`, `train/test*.py`, `train/viz/test.py` | `harness/tests/` |

`train/` and `tasks/` are gone. `config.py` was the only file split, at a contiguous
boundary (the two shared dataclasses precede `Config`); everything else moved whole.

**Renames avoided on purpose.** `Config` was *not* renamed to `DistillConfig` despite the
asymmetry with `Ade20kConfig` / `In1kConfig`. It buys symmetry only, and every rename is
risk without benefit here.

## Verification

The layout claim is cheap to check; the *no-behavior-change* claim is the one that needed
evidence. Five numeric gates, all recorded **before** the move and re-run after:

| gate | result |
|---|---|
| distill parity digest | `9a0100a1a3de3acd` — unchanged |
| `ade20k_probe` / `ade20k_finetune` pinning digests | `b9fd07bdac4f68bd` / `28fb8bff5010a010` — unchanged |
| `in1k_probe` / `in1k_finetune` pinning digests | `00ed4e8f2279b20f` / `6f4accd7c2ad3dba` — unchanged |
| full test suite | 307 passed, same count as baseline |
| `capability_matrix.md` regenerated | byte-identical |
| tyro CLI flag surface, all 3 tasks | 228 / 108 / 107 flags, byte-identical |
| `ruff check` | every non-import-order count identical to baseline |

The four ade20k/in1k digests were **created for this refactor** (commit `e07aa8a`) — those
paths previously had no end-to-end numeric guard, only shape/wiring tests, so the move
could have been argued safe but not measured safe. See `harness/tests/test_task_digests.py`.
They are *pinning* digests, not parity digests: they assert today's numbers equal the
recorded ones. A pre-existing bug is pinned in with everything else.

Two further hazards checked rather than assumed:

- **Checkpoints.** All 131 reachable `.pt` files were decoded with `pickletools`: the only
  unpickle targets are `OrderedDict`, torch's tensor rebuilders, storage classes, and
  `PosixPath` — **zero** canvit module references. The `canvit_pretrain` /
  `canvit_specialize` occurrences inside them are plain strings in config dicts. Module
  relocation therefore cannot break checkpoint loading.
- **Sibling repos.** None import this package (the single hit is a comment in the
  read-only `CanViT-specialize` fallback).

## What was deliberately left alone

- **Dated docs in this folder keep the old paths.** `07`, `08`, `12`, `ddp_validation.md`,
  `p0-notes.md` etc. are records of what was true when written; rewriting them would
  falsify the record. Read them as history — this doc is the current layout's authority.
- **`slurm/archive/**` keeps `canvit_pretrain.*` module paths.** Those launchers pin
  pre-rename commits whose `git archive` snapshot carries that package name — it is the
  only name under which those entry points exist. Do not modernize them.
- **One dead line removed:** the `pyproject.toml` per-file-ignore for
  `canvit_train/train/loop.py`, a file deleted back in doc 17.
