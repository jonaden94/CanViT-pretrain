# SLURM launchers

Everything here is submitted by hand — nothing in this tree runs automatically.

```
slurm/
├── harness_train.sbatch   THE launcher. Every new run goes through it.
├── env.sh                 Grete job setup (sources .envrc.grete, GWDG proxy)
├── submit.sh              convenience wrapper
├── runs/<group>/*.sh      per-experiment launchers — the copyable ones
└── archive/               reproduction-only; see below
```

> Naming note: this folder was `slurm_nhr/` until 2026-07-31. It took the name `slurm/`
> after the *original* `slurm/` — the Nibi / Alliance Canada tooling — was deleted as
> deprecated. Historical prose in `archive/` that says `slurm/…` may mean **that** deleted
> folder, not this one.

## Starting a new run

Copy the closest launcher from `runs/` and edit two things:

1. **The pins.** `PRETRAIN_COMMIT` / `PYTORCH_COMMIT` / `FOVI_COMMIT` snapshot the code for
   the whole run (offline `git archive` → `$TMPDIR` → `PYTHONPATH`). **Every launcher in
   `runs/` pins a pre-2026-07-31 commit**, so bumping them to current hashes is the normal
   first step — copying one unchanged reproduces old code, which is the point of pinning
   but rarely what you want for new work.
2. **The config.** `CFG_*` env vars map to `--cfg.foo-bar`; see any launcher for the
   pattern. Bool knobs have **no** `CFG_` form (tyro renders them as paired flags), so they
   go in `EXTRA_ARGS` — e.g. `EXTRA_ARGS="--preset policy_only --cfg.no-augment"`.

`runs/exp27/policy-lossfix-s0.sh` and `runs/exp25/in1k-uni16ti-803k.sh` are the most
heavily-commented examples.

## What's in `runs/` vs `archive/`

The split is **which interface a launcher drives**, not how old it is:

| | drives | usable as a template? |
|---|---|---|
| `runs/` | `canvit_train.harness.run` — the current single entry point | **yes**, bump the pins |
| `archive/` | `canvit_pretrain.{train,ade20k,in1k,ade20k.rl_train}` — entry points **deleted** in the 2026-07-31 consolidation | no — reproduction only |

`archive/` still *works*: each launcher pins a pre-consolidation commit, and `git archive`
restores that snapshot into the job's `$TMPDIR`, so the deleted entry points come back for
the duration of the job. That is how the exp22/exp23/exp27 comparisons stay reproducible.
**Do not "modernize" the `canvit_pretrain` module paths in there** — that is the only name
under which those entry points exist (the package was renamed the same day).

`archive/` holds: the old `base_train.sbatch` + the standalone `ade20k/` and `in1k/`
launchers, the pre-existing `_legacy/` `_scrap/` `_original/` `_workspace/` piles, and the
old-loop run groups (`jon_exp17_19_*`, `jon_exp20_*`, `jon_exp21_modulation`,
`jon_exp22_full_runs`, `test`, `perf`, `default`) — 211 files.

**Groups in `runs/` are kept intact even when they contain an old-loop arm.** `exp23` and
`exp26` are old-loop-vs-harness A/B *pairs*, and `exp27` includes `rl_train` reference arms;
splitting a pair across `runs/`/`archive/` would destroy the comparison's legibility. Those
arms reference `archive/base_train.sbatch` explicitly.

`harness_train.sbatch` is shared between new runs and the pre-rename pinned arms in `exp23`
/ `exp27`, so it **detects** whether the snapshot contains `canvit_train` or
`canvit_pretrain` and dispatches accordingly — see `_PKG` in that file.
