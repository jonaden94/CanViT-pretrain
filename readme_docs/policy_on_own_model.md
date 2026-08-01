# Training a viewpoint policy on your own model

How to train the ADE20K viewpoint policy on a backbone and probe you trained yourself,
rather than on the published pair.

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
--cfg.model-repo logs/jon_exp22_full_runs/<pretrain>/checkpoints/step-N-hf \
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

`slurm/runs/policy_on_own_fovi_probe/` runs this on the exp22 foveated-teacherinit backbone
with the ADE20K probe trained on it:

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do
  SEED=$s bash slurm/runs/policy_on_own_fovi_probe/policy-qreg-own-fovi-s0.sh
done
```

It **refuses to submit until the probe run has finished**, testing for the final step
checkpoint rather than for `best.pt` — the latter appears at the first evaluation and is
rewritten whenever the metric improves, so it would otherwise hand the policy a probe a few
hundred steps old.

The backbone is passed as its HF export rather than its `.pt` for a permissions reason, not
a technical one: the pretraining `.pt` files are mode 600 (owner only) while the exported
directories are group-readable, so the export is what a colleague can actually open.

## Status

**Config-validated, never trained.** The recipe is the standard Q-regression policy arm with
this backbone and probe swapped in and the grid matched to the probe. That arm runs against
the published *uniform* backbone; this one is foveated, so the policy's action space becomes
the fixation grid rather than the safe-box grid — a combination that has not been run.

Treat the first run as a smoke test. `eval/miou_t0` is the full-image glimpse taken *before*
any policy action, so it depends only on the frozen backbone, the probe, the resize and the
eval path. It should match a probe-only evaluation; if it does not, something in the pairing
is wrong and the later timesteps mean nothing.

Expected scale for a working run, from the published band: best `eval/ce_mean` ≈ 0.685–0.686
and `miou_final` ≈ 0.445–0.450. **A result materially better than that is evidence of a
broken protocol, not of success** — that exact failure has happened here before.
