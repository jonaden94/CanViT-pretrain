# Verification campaign: exp32–exp35

Four groups of runs that verify, at full scale, that the three training objectives plus the
viewpoint policy behave as expected on the current code base. Each reproduces a campaign
that ran earlier and is judged against that earlier result, so a deviation is evidence of a
code change rather than of noise.

| group | what it trains | runs | reference | comparable on |
|---|---|---|---|---|
| `exp32_pretrain_lrdrop` | `distill` pretraining from scratch, one ×0.1 LR drop | 4 | exp22 / exp28 | train loss; the normalizer shifts the scale vs exp22 |
| `exp33_in1k_finetune` | `in1k` full finetunes from the exp22 models | 4 | exp25 | **yes**, top-1 |
| `exp34_ade20k_probe` | `ade20k` frozen probes from the exp22 models | 4 | exp30 | **yes**, CE and mIoU |
| `exp35_policy_qreg_10seed` | ADE20K viewpoint policy (Q-regression), 10 seeds | 10 | exp27 / exp31 | **yes**, CE and mIoU |

Each group's `slurm/runs/<group>/README.md` carries the detail — recipe provenance, source
checkpoints, reference tables, and the traps specific to that group. This document is the
overview and the shared procedure.

## Relationship to exp28–exp31

exp32–35 are **1:1 replications** of exp28–31 on the current code: rewriting the group
number back and removing the two pin lines makes every launcher byte-identical to its
predecessor. Only the run identity and the pinned commits differ.

Why they were re-run:

- **exp28 and exp29 were cancelled** part-way (22 and 1 array tasks completed respectively)
  to restart cleanly on current code.
- **exp30 and exp31 completed.** Their results stand and are now the *references* for exp34
  and exp35. Re-running them is a check that the loader/config refactor did not perturb
  training, not a recovery.

## Status at a glance

```bash
bash slurm/status.sh
```

One-shot, local job logs only, no network, changes nothing. It prints each run's current
step and best metric beside its reference, whether each `exp32` LR-drop phase is ready, and
an in-band verdict per policy seed. Note it still tracks the exp28–31 group names; point it
at exp32–35 before relying on it for this campaign.

Training logs contain the **full evaluation dictionaries inline** — every metric is in the
job log, not only in the tracker.

## Shared procedure

Every launcher is a small self-documenting script. Submit by running it:

```bash
bash slurm/runs/<group>/<run>.sh
```

All of them pin `TRAIN_COMMIT` / `PYTORCH_COMMIT` / `FOVI_COMMIT`, so the job runs a frozen
`git archive` snapshot and is immune to later edits of the clones. A new run group means
empty checkpoint directories, so every run starts at step 0 — there is nothing to clear.

## exp32 — pretraining with one ×0.1 LR drop

Four pretrains from scratch (uniform / foveated × with and without teacher init), each
followed by a single ×0.1 LR drop and 204,800 further steps.

**Two phases, because there is no in-run LR-drop feature.** Phase A is
`exp32-<arm>.sh` (warmup 100k → constant 4e-4, to the drop step); phase B is
`exp32-<arm>-lrdrop.sh` (flat 4e-5, 25 jobs). The drop point is a FILENAME
(`CFG_SEED_CKPT=.../step-<N>.pt`), so it cannot fire early, late, or twice however many
array tasks fail, and phase B refuses to submit until that file exists.

| arm | phase-A jobs | drop step | phase B |
|---|---|---|---|
| `exp32-uniform16-teacherinit` | 77 | 630,784 | yes |
| `exp32-fovi-teacherinit` | 138 | 1,130,496 | yes |
| `exp32-uniform16` | 176 | 1,441,792 | yes |
| `exp32-fovi` | 245 (full 2,007,040) | — | no |

The array is a **budget, not a schedule**: the job index comes from the checkpoint's resume
state rather than `SLURM_ARRAY_TASK_ID`, so the step count advances only for jobs that
succeed. If tasks fail, phase A ends below target — resubmit the remainder before phase B.

**Judge on** `val/scene_cos_norm_t9`, not `eval/val_metric`. The latter is the single scalar
the framework tracks for checkpoint selection (raw scene cosine at the last eval timestep);
the former is what selected exp22's and exp28's best checkpoints. First comparable point is
step 8192; exp22's values there were 1.8945 / 1.6106 / 1.8885 / 1.8377 (uniform16 /
uniform16-ti / fovi / fovi-ti), with a systematic offset expected from the 4-shard
normalizer.

**Evaluation policy** is `auto`, which resolves per patcher: uniform arms validate under
coarse-to-fine, foveated arms under a fixation grid.

## exp33 — ImageNet-1k full finetunes

Four finetunes, one per exp22 pretrained source, recipe from exp25 (the TPU recipe
batch-adapted for one A100). 49 array jobs each. `n_timesteps=4`, not the task default of
10. Foveated arms evaluate under `random` (coarse-to-fine is uniform-only and OOD for a
fixed-scale foveated model) with `--cfg.foveated-scale.fixed-scale 2.0`.

exp25's best `eval/top1` is the reference, and top-1 is untouched by the ADE20K mIoU
re-basing: **0.84954** (`uni16ti-803k`), **0.83692** (`fovi-ti-1196k`, an INCOMPLETE
reference — that run stopped at 320k of 401,408 steps), **0.83522** (`uni16-1516k`). The
fourth arm, `fovi-1901k`, has no earlier counterpart.

**Only one number is reported and it is measured at the final timestep** — the classifier
reads the CLS token of the last glimpse only, so there is no per-timestep top-1 series.

**Watch the first logged `train/full/loss`:** well below `ln(1000) ≈ 6.9`. Near 6.9 means
the pretrained probe was not fused and the finetune started from a random classifier — a
live bug once, which is why `CFG_PROBE_REPO` is load-bearing.

This is the group the current code most affects: an in1k checkpoint now records its
architecture instead of only a `model_repo` pointer, so a finetuned model — whose backbone
exists nowhere else — loads straight from its `.pt` via `load_classifier`.

## exp34 — ADE20K frozen probes

Four probe runs from the same four sources. Frozen backbone via the `probe` preset, 40k
steps, random-view training, `n_timesteps 10`, scene 512, `canvas_grid 32`. Single GPU —
the ADE20K task does not support DDP.

**`resize_mode=squish` for every arm including foveated.** That is the protocol every
earlier CanViT number was measured under. It distorts aspect ratio, so it is not the right
choice for a human-viewing comparison; whichever is used must be reported with the number.

exp30 is a valid reference on both CE and mIoU — it already included the mIoU
reduction-order fix. Best `miou_final`: 0.44479 (`uni16ti-803k`), 0.4434
(`fovi-ti-1196k`), 0.42321 (`uni16-1516k`); `fovi-1901k` has no earlier counterpart.
Ordering to expect: `uni16ti > fovi-ti > uni16`, all in the 0.41–0.45 band.

**The per-timestep `eval/miou_t*` are measured under *random* viewpoints, not
coarse-to-fine** — `eval_policy` is `auto` and ADE20K's default is IID random from a
full-scene anchor, matching how the probe trains. ADE20K train-mIoU is deliberately not
logged.

**Do not compare against exp24 or anything older**: those predate the mIoU reduction-order
fix and read ~0.2 pp lower by metric definition, not by quality.

## exp35 — ADE20K viewpoint policy, 10 seeds

A `ViewpointScorer` trained against a frozen backbone and probe so segmentation improves as
fast as possible per glimpse, deployed by argmax over its candidate grid. Recipe is exp27's
`lossfix` arm. `--preset policy_only`, 8000 steps, 5 timesteps, batch 16, canvas grid 64.

The frozen backbone needs no flag: `Ade20kConfig.model_repo`'s default already IS the
published model every policy checkpoint records. `CFG_MODEL_REPO` is deliberately unset,
which also makes this group independent of exp32–34.

| exp27 `lossfix`-s0 reference | value |
|---|---|
| best `eval/ce_mean` (mean t1–t4) | **0.68577** |
| `eval/miou_final` | 0.44848 |

Published band: **0.6853 ± 0.0007** → [0.6846, 0.6860]. Expect the ten seeds to cluster at
best `ce_mean` ≈ 0.685–0.686, `miou_final` ≈ 0.445–0.450.

**If a seed comes out materially BETTER than the band, distrust it.** That failure mode has
occurred: a resize default change put an arm at 0.6693 — 0.016 "better", ~20× the seed
spread — purely from the protocol change. `CFG_RESIZE_MODE=squish` is the measurement
contract.

**Free bit-identity check:** `ce_t0` / `miou_t0` are the full-image glimpse taken *before*
any policy action, so they depend only on the frozen backbone, probe, resize and eval path.
Matching exp31's to every printed digit isolates any difference to the policy itself —
which is exactly what this re-run tests.

## Using the results afterwards

Checkpoints from exp33 and exp34 are **self-describing**: they record the architecture, not
just a path to the model they started from, so they load without their source repo:

```python
from canvit_pytorch.model_source import load_classifier, load_segmentation
clf = load_classifier("logs/exp33_in1k_finetune/<run>/checkpoints/best.pt")
seg = load_segmentation("logs/exp34_ade20k_probe/<run>/checkpoints/best.pt")
```

An exp34 checkpoint also works directly as `--cfg.probe-repo` for a policy run, with no
conversion. Publishing still goes through `canvit_train.checkpoint.to_hf` (whole models) or
`canvit_train.checkpoint.probe_to_hf` (the segmentation head alone).

exp32's pretraining checkpoints are unchanged in format — distill checkpoints always carried
their full architecture, which is why `to_hf` accepts them and refuses the others.
