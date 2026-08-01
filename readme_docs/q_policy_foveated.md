# Training the Q viewpoint policy for a foveated model

Trains the ADE20K viewpoint policy (Q-regression) on our foveated CanViT backbone and the
ADE20K probe trained on it. The policy learns **where to look next**; the backbone and probe
stay frozen.

> **Never run end to end.** Every Q-policy result so far is on a *uniform* backbone. The
> foveated path is implemented and covered by unit tests, and one bug specific to it has
> already been found and fixed — but no training run has completed on it. Treat the first
> run as a smoke test (see [Checking a run](#checking-a-run)).

## Run it

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do
  SEED=$s bash slurm/runs/policy_on_own_fovi_probe/policy-qreg-own-fovi-s0.sh
done
```

One A100 per seed, 9000 steps, ~1.5 h each. The launcher refuses to submit until the probe
run has finished, so it is safe to fire early.

## The two checkpoints it uses

| flag | value |
|---|---|
| `--cfg.model-repo` | `logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt` |
| `--cfg.probe-repo` | `logs/exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best.pt` |

Both are training checkpoints, passed directly — no conversion step.

The probe is **the reward model**, not a detail: the reward is the fraction of the probe's
cross-entropy that a glimpse removes, so a mismatched head makes the reward pure noise and
the policy learns nothing.

## What the policy chooses

Every glimpse is the **same foveation pattern**, and only the fixation centre changes — t0
included, which is simply the centred one. There is no full-image glimpse anywhere in the
rollout.

The foveation window is `fix_size = scale × H`, so at scale 2.0 it spans twice the image
side. The policy is choosing where to spend resolution, not what is visible. Its action
space is therefore a grid of **centres only, with no scale dimension**.

## Three settings that must match the checkpoints

Each fails **silently** — a worse number, not an error.

| setting | value here | why |
|---|---|---|
| `--cfg.canvas-grid` | `32` | must equal the grid the probe was trained at, or the reward model sees a canvas resolution it never saw. Easy to get wrong by copying: the uniform recipe uses 64 because *its* probe was trained at 64. |
| `--cfg.foveated-scale.fixed-scale` | `2.0` | must equal the backbone's pretraining scale, or every glimpse is out of distribution and mIoU decays as glimpses accumulate |
| `--cfg.resize-mode` | `squish` | the protocol every CanViT number is measured under |

Change the probe, and `canvas-grid` has to change with it.

## Checking a run

`eval/miou_t0` is measured *before* any policy action, so it depends only on the frozen
backbone, the probe, the resize and the eval path — never on the policy. It should match a
probe-only evaluation of the same pair. If it does not, the pairing is wrong and the later
timesteps mean nothing.

For a working run expect best `eval/ce_mean` ≈ 0.685–0.686 and `miou_final` ≈ 0.445–0.450.
**A result materially better than that is evidence of a broken protocol, not of success** —
that has happened here before, when a changed resize default put an arm 0.016 "better" than
the published band, about 20× the seed spread.
