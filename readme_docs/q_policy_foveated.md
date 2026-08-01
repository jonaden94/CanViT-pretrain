# Training the Q viewpoint policy for a foveated model

How to train the ADE20K viewpoint policy — Q-regression (`objective=qreg`), the recipe the
published band is defined under — on a **foveated** CanViT backbone and a probe trained on
it.

> **This combination has never been run end to end.** Every Q-policy result so far is on a
> **uniform** backbone. The foveated path is implemented and unit-tested,
> and one bug specific to it has already been found and fixed, but no training run has
> completed on it. See [Status](#status) before trusting a number from it.

## What the policy actually chooses here

For a foveated model at a fixed view scale — the configuration below — **every glimpse is
the same foveation pattern and only the fixation centre changes**, t0 included. There is no
full-image glimpse anywhere in the rollout:

| | uniform model | foveated model, `mode=fixed` |
|---|---|---|
| t0 ("full") | whole image at scale 1 | **centred** foveation at `fixed_scale` — identical in kind to every other glimpse |
| later glimpses | a crop: position **and** scale | same foveation, different fixation centre |
| policy action space | safe-box grid, centres × scales | fixation heatmap, centres only — **no scale dimension** |

The window is `fix_size = scale × H`, so at `fixed_scale 2.0` it spans twice the image side.
The policy is therefore choosing where to point the fovea — where resolution is spent — not
which part of the scene is visible at all. That is a genuinely different decision problem
from the uniform case, which is the substantive reason this combination deserves its own
first run rather than being assumed equivalent.

`per_rollout` / `per_glimpse` behave differently: there t0 falls back to scale 1, a true
full-image anchor, and the random glimpses draw their own scale. Only `fixed` makes t0 match
the rest.

A policy run needs **two** things, and neither alone is enough:

| flag | what it is |
|---|---|
| `--cfg.model-repo` | the frozen pretrained backbone |
| `--cfg.probe-repo` | the trained segmentation head — **this is the reward model** |

The probe is load-bearing, not cosmetic: the reward is the fraction of the probe's
cross-entropy that a glimpse removes, so a random or mismatched head makes the reward pure
noise and the policy learns nothing.

Both flags accept a **training `.pt`**, a local HF directory, or a Hub id, so a probe you
just trained can be passed as the checkpoint the run wrote — no conversion step:

```bash
--cfg.model-repo logs/jon_exp22_full_runs/<pretrain>/checkpoints/step-N.pt \
--cfg.probe-repo logs/<probe-run>/checkpoints/best.pt
```

Only the head's `head.*` weights are read from that checkpoint, and its shape determines
`num_classes` / `embed_dim`, so any ADE20K run's checkpoint works as a probe source — old
ones included.

## Three settings that are not free choices

Each fails **silently** — a degraded metric, not an error.

| setting | rule | what goes wrong otherwise |
|---|---|---|
| `--cfg.canvas-grid` | match the grid the probe was **trained** at | the reward model is fed a canvas resolution it never saw |
| `--cfg.foveated-scale.fixed-scale` | match the backbone's pretraining scale (2.0 for the exp22 foveated models) | a foveated backbone derives `fix_size = scale × H`; another scale makes every glimpse out of distribution and mIoU decays as glimpses accumulate |
| `--cfg.resize-mode` | `squish` | the measurement contract every CanViT number is defined under; it has regressed once, putting an arm 0.016 "better" than the published band, ~20× the seed spread |

The grid is the one most easily got wrong by copying: the standard policy recipe uses
`canvas_grid 64` because the published probe was trained at 64. A probe trained at 32 needs
32.

## The ready-made launcher

`slurm/runs/policy_on_own_fovi_probe/policy-qreg-own-fovi-s0.sh` runs this on the exp22
foveated-teacherinit backbone with the ADE20K probe trained on it:

| flag | value |
|---|---|
| `--cfg.model-repo` | `logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt` |
| `--cfg.probe-repo` | `logs/exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best.pt` |

9000 steps (so the run is evaluated at 8000, matching the arm it is compared against), one
A100 per seed, ~1.5 h each.

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do
  SEED=$s bash slurm/runs/policy_on_own_fovi_probe/policy-qreg-own-fovi-s0.sh
done
```

It **refuses to submit until the probe run has finished**, testing for the final step
checkpoint rather than for `best.pt` — the latter appears at the first evaluation and is
rewritten whenever the metric improves, so it would otherwise hand the policy a probe a few
hundred steps old.

Both halves are the training checkpoints themselves — nothing is exported first. The HF
layout is for publishing, not for feeding one of your own runs into another.

If a checkpoint turns out to be unreadable by a colleague, check its mode rather than its
path: permissions live on the inode, so reaching a file through a different spelling of the
same directory changes nothing. Runs write `640` (group-readable) under the project's
current umask, but older trees may hold `600` files from a job that ran under a stricter
one; `chmod g+r` fixes those in place.

## Status

**Implemented and unit-tested; never trained end to end.** Worth being precise about which
parts are which, because "untested" would be too pessimistic and "it works" too optimistic.

What *is* covered:

- The action space branches on the patcher and has for a while: a foveated or square model
  gets `fixation_candidates(centers_per_axis)` — a pure *where to look* heatmap with **no
  scale dimension** — while a uniform model gets the safe-box grid with scales.
- A foveated-specific defect in exactly this path was found and fixed. The fixation
  candidate table hardcodes its scale column to `1.0`, which pinned every policy glimpse to
  full-field foveation *regardless of* `foveated_scale`. On a model pretrained at 2.0 — this
  one — the t0 anchor and the random glimpses used 2.0 while the policy's used 1.0, i.e. the
  policy looked out of distribution and only the policy did. The selector now asks the scale
  law instead, as the random path does.
- The suite exercises the foveated selector and the deployment path.

What is **not** covered: a completed training run. The Q objective, its ε-greedy curriculum
and its reward standardisation have only ever been driven by the safe-box action space. They
are action-space agnostic by construction, but that is an argument, not evidence.

Treat the first run as a smoke test. `eval/miou_t0` is taken *before* any policy action, so
it depends only on the frozen backbone, the probe, the resize and the eval path. It should
match a probe-only evaluation of the same pair; if it does not, something in the pairing is
wrong and the later timesteps mean nothing. (For a foveated model at a fixed scale, t0 is
the **centred** foveated glimpse — not a full-image view. See below.)

Expected scale for a working run, from the published band: best `eval/ce_mean` ≈ 0.685–0.686
and `miou_final` ≈ 0.445–0.450. **A result materially better than that is evidence of a
broken protocol, not of success** — that exact failure has happened here before.
