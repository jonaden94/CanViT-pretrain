# exp32 — exp28 restarted clean on the current code base

The same four pretrains as exp28, from scratch, each with a single ×0.1 LR drop followed by
204,800 more steps. **Config-identical to exp28**: only the run identity and the commit
pins differ, verified mechanically (rewriting `exp32`→`exp28` and stripping the two pin
lines makes every launcher byte-identical to its exp28 counterpart).

exp28 was cancelled part-way — 7 / 5 / 7 / 3 array tasks had completed on the
uniform16-teacherinit / fovi-teacherinit / uniform16 / fovi arms. exp32 does not resume it:
a new run group means an empty `logs/exp32_pretrain_lrdrop/<arm>/checkpoints/`, so distill's
default `--opts.resume` finds nothing and every arm starts at step 0.

## Pins

| repo | exp28 | exp32 |
|---|---|---|
| `canvit_train` | `455bdae` | **`009f262`** |
| `canvit_pytorch` | `1f5121b` | **`d616b7b`** |
| `fovi` | `c399d3b` | `c399d3b` (unchanged) |

All three verified to `git archive` cleanly, and the snapshots contain the current work
(`canvit_pytorch/model_source.py`, `canvit_train/checkpoint/test_self_describing.py`).

## What the new pins do and do not change

**Not the checkpoint format.** There is no new `.pt` format for pretraining. Distill
checkpoints always carried their full architecture under `model_config["canvit"]` — that is
why `to_hf` accepts them and refuses ade20k/in1k. The self-describing work applied to the
*downstream* tasks, whose checkpoints recorded only a `model_repo` path. An exp28 checkpoint
written by the old code loads under the new loader unchanged (verified on
`exp28-uniform16/checkpoints/step-8192.pt`).

**Not the training numerics.** The diff under `distill/` between the two pins is two config
defaults: a `val_dir` path spelling (the launcher passes `--cfg.val-dir` anyway) and
`wandb_dir` becoming `$WANDB_DIR`-driven (`.envrc.grete` sets it to the same value). The
parity digest and the full suite pass unchanged across the refactor.

**What it does give** is provenance: one clean run per arm under a single current pin, with
no mid-array pin change — which is what exp22's foveated arms suffered (a pin change jumped
their eval scale by +0.058 mid-run).

So exp32 should track exp28's curves where exp28 got to. The step-8192 reference table in
`../exp28_pretrain_lrdrop/README.md` applies here too, including the caveat that the
4-shard normalizer shifts the loss scale relative to exp22.

## Two phases, because there is no in-run LR-drop feature

| phase | launcher | learning rate | length |
|---|---|---|---|
| A | `exp32-<arm>.sh` | warmup 100k → constant 4e-4 | to the drop step |
| B | `exp32-<arm>-lrdrop.sh` | flat 4e-5, `warmup_steps=0` | 25 jobs = 204,800 steps |

The drop point is a FILENAME (`CFG_SEED_CKPT=.../step-<N>.pt`), so it cannot fire early,
late, or twice no matter how many array tasks fail. Phase B refuses to submit until that
file exists.

| arm | phase-A jobs | drop step | phase B |
|---|---|---|---|
| `exp32-uniform16-teacherinit` | 77 | 630,784 | yes |
| `exp32-fovi-teacherinit` | 138 | 1,130,496 | yes |
| `exp32-uniform16` | 176 | 1,441,792 | yes |
| `exp32-fovi` | 245 (full 2,007,040) | — | no |

The array is a **budget, not a schedule**: the job index comes from the checkpoint's resume
state, not `SLURM_ARRAY_TASK_ID`, so the step count advances only for jobs that succeed. If
tasks fail, phase A ends below target — resubmit the remainder until `step-<N>.pt` exists,
and only then launch phase B.

## Submit

```bash
bash slurm/runs/exp32_pretrain_lrdrop/exp32-uniform16.sh
bash slurm/runs/exp32_pretrain_lrdrop/exp32-uniform16-teacherinit.sh
bash slurm/runs/exp32_pretrain_lrdrop/exp32-fovi.sh
bash slurm/runs/exp32_pretrain_lrdrop/exp32-fovi-teacherinit.sh
```

~1,420 GPU-hours. At one array task at a time, `exp32-fovi` is ~20 days and the three drop
parents ~12 / 15 / 6 days. Phase B only once `bash slurm/status.sh` reports the arm READY.

## Not included

The four **in1k finetunes** (exp29) were cancelled alongside exp28 and are not part of this
group. Those are the runs that genuinely gain from the current code — an in1k checkpoint now
records its architecture instead of a `model_repo` pointer, so a finetuned model can be
loaded straight from its `.pt`. Worth re-running as its own group.

`slurm/status.sh` still refers to the exp28 group; point it at exp32 (or add a group) before
relying on it here.
