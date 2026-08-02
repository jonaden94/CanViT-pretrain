# Training the Q viewpoint policy for a foveated model

Trains the ADE20K viewpoint policy (Q-regression) on our foveated CanViT backbone and the
ADE20K probe trained on it. The policy learns **where to look next**; the backbone and probe
stay frozen.

> **Never run end to end.** Every Q-policy result so far is on a *uniform* backbone. The
> foveated path is implemented, covered by unit tests, and the exact model this recipe
> builds has been constructed successfully from both checkpoints — but no training run has
> completed on it. Treat the first run as a smoke test (see [Checking a run](#checking-a-run)).

## Run it

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do
  SEED=$s bash slurm/runs/policy_on_own_fovi_probe/policy-qreg-own-fovi-s0.sh
done
```

One A100 per seed, 9000 steps. The equivalent uniform runs took 75 min each at canvas 64;
this one runs at canvas 32, so expect that or less.

The launcher refuses to submit unless the probe run has finished (it checks for the probe's
final `step-40000.pt`, not `best.pt`, which appears at the first evaluation and would hand
the policy a probe a few hundred steps old). That probe run **is** finished, so the guard is
already satisfied.

## The two checkpoints it uses

Absolute paths, readable by anyone in the `HPC_nib00021` project — the files are mode 640
and every parent directory is group-readable, so a collaborator can point at them directly
without copying:

| flag | value |
|---|---|
| `--cfg.model-repo` | `/mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt` |
| `--cfg.probe-repo` | `/mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train/logs/exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best.pt` |

Both are training checkpoints passed straight to the flags — no HF export, no conversion
step. Both flags accept a training `.pt`, a local HF directory, or a Hub id.

Verified by building the model this recipe specifies: the backbone comes out foveated at
scale 2.0 on a canvas grid of 32, and all 9 probe-head tensors are bit-identical to the ones
in the probe checkpoint, i.e. the reward model really is the trained probe and not a fresh
random head.

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
| `--cfg.resize-mode` | `squish` | the protocol every other CanViT number in this repo is measured under |

Change the probe, and `canvas-grid` has to change with it.

## Checking a run

**Step 1 — is the pairing right?** `eval/miou_t0` is measured *before* any policy action, so
it depends only on the frozen backbone, the probe, the resize and the eval path — never on
the policy. For this pair it must land at **0.377**. If it does not, backbone and probe are
mismatched and every later timestep is meaningless; fix that before reading anything else.

**Step 2 — does it beat random?** These two checkpoints under *random* viewpoints, which is
the bar a learned policy has to clear:

| | t0 | t1 | t2 | t3 | t4 |
|---|---|---|---|---|---|
| random viewpoints | 0.377 | 0.403 | 0.415 | 0.424 | **0.428** |

Both rows above were measured on 2026-08-02 with exactly this recipe's settings — full
ADE20K val, `n_timesteps 5`, `canvas_grid 32`, `squish`, `fixed_scale 2.0` — against
`best.pt`, so they are directly comparable to what the policy run reports:

```bash
python scripts/eval_ade20k_checkpoint.py \
  --ckpt   <probe .pt> --model-repo <backbone .pt> \
  --eval-policy random --n-timesteps 5 --canvas-grid 32 --resize-mode squish --fixed-scale 2.0
```

A trained policy should beat 0.428 at t4, and should beat random *earliest* — the claim is
that it reaches a given mIoU in fewer glimpses, so the gap at t1–t2 matters more than at t4.

Also check the **shape**: mIoU must rise monotonically across t1–t4. Falling mIoU as
glimpses accumulate is the signature of a scale mismatch between the rollout and the
backbone's pretraining scale.

**Do not borrow the uniform policy figures** (`ce_mean` ≈ 0.686, `miou_final` ≈ 0.448):
different backbone, different probe, canvas grid 64. They do not transfer, and a correct run
here would look broken against them.
