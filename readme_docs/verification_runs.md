# Verification campaign: exp28–exp31

Four groups of runs that re-verify, at full scale, that the three training
objectives behave exactly as they did before they were brought under one
framework. Each group reproduces a training campaign that was previously run
through a separate, now-removed code path, using the same recipe and the same
source checkpoints, and is judged against that earlier result.

| group | what it trains | runs | reference | comparable? |
|---|---|---|---|---|
| `exp28_pretrain_lrdrop` | `distill` pretraining from scratch, with one ×0.1 LR drop | 4 | exp22 | loss scale shifted — see below |
| `exp29_in1k_finetune` | `in1k` full finetunes from the exp22 models | 4 | exp25 | **yes**, on top-1 |
| `exp30_ade20k_probe` | `ade20k` frozen probes from the exp22 models | 4 | exp24 | **no**, on mIoU — see below |
| `exp31_policy_qreg_10seed` | ADE20K viewpoint policy (Q-regression), 10 seeds | 10 | exp27 | **yes**, on CE and mIoU |

Every run in the campaign pins `CanViT-train 455bdae` / `CanViT-PyTorch 1f5121b`
/ `fovi c399d3b`. Each group's `slurm/runs/<group>/README.md` carries the full
detail — recipe provenance, source-checkpoint identity, wandb run ids, submitted
job ids. This document is the procedure: how to launch, how to watch, how to
judge.

## Status at a glance

```bash
bash slurm/status.sh
```

One-shot, reads local job logs only, no network, changes nothing. For each group
it prints the current step, the best metric so far, the reference beside it,
whether each `exp28` second phase is ready to launch, and an in-band verdict per
`exp31` seed. It carries the interpretation rules inline, so start there rather
than scraping logs by hand.

The training logs contain the **full evaluation dictionaries inline** — every
metric is in the job log, not only in the tracker.

---

## exp28 — pretraining with one ×0.1 LR drop

Four `distill` pretrains from scratch — uniform and foveated patchers, each with
and without teacher-initialized weights — each followed by a single ×0.1 learning
rate drop and 204,800 further steps. This is the final empirical check that the
pretraining objective is unchanged.

### Two phases, because there is no in-run LR-drop feature

The available schedules are `warmup_constant`, `warmup_cosine`, and
`warmup_onecycle`; none has a "×0.1 at step N" milestone. The drop is therefore
achieved the way it always was: a **separate run**, seeded from a checkpoint
file, at a flat lower learning rate.

| phase | launcher | learning rate | length |
|---|---|---|---|
| A | `exp28-<arm>.sh` | warmup 100k → constant 4e-4 | to the drop step |
| B | `exp28-<arm>-lrdrop.sh` | flat 4e-5, `warmup_steps=0` | 25 jobs = 204,800 steps |

**The drop point is a filename, not a step comparison**
(`CFG_SEED_CKPT=.../step-<N>.pt`), so it cannot fire early, late, or twice no
matter how many array tasks fail. Phase B's launcher refuses to submit until that
file exists.

| arm | phase-A jobs | drop step | phase B |
|---|---|---|---|
| `exp28-uniform16-teacherinit` | 77 | 630,784 | yes |
| `exp28-fovi-teacherinit` | 138 | 1,130,496 | yes |
| `exp28-uniform16` | 176 | 1,441,792 | yes |
| `exp28-fovi` | 245 (full 2,007,040) | — | no |

### Procedure

```bash
# phase A — one launcher per arm
bash slurm/runs/exp28_pretrain_lrdrop/exp28-uniform16.sh
bash slurm/runs/exp28_pretrain_lrdrop/exp28-uniform16-teacherinit.sh
bash slurm/runs/exp28_pretrain_lrdrop/exp28-fovi.sh
bash slurm/runs/exp28_pretrain_lrdrop/exp28-fovi-teacherinit.sh
```

Then, **only once `bash slurm/status.sh` reports the arm READY** (i.e. its
`step-<N>.pt` exists):

```bash
bash slurm/runs/exp28_pretrain_lrdrop/exp28-uniform16-lrdrop.sh
bash slurm/runs/exp28_pretrain_lrdrop/exp28-uniform16-teacherinit-lrdrop.sh
bash slurm/runs/exp28_pretrain_lrdrop/exp28-fovi-teacherinit-lrdrop.sh
```

The array is a **budget, not a schedule**: the job index comes from the
checkpoint's resume state rather than from `SLURM_ARRAY_TASK_ID`, so the step
count advances only for jobs that *succeed*, and no shard is skipped or re-read
when a task dies. Consequence: if tasks fail, phase A finishes **below** its
target step — resubmit the remainder until `step-<N>.pt` exists, and only then
launch phase B. The resume path hard-errors rather than guessing if a
checkpoint's scheduler step disagrees with the schedule-derived step, so a
mid-job save cannot silently shift anything.

Cost: ~1,420 GPU-hours. At one array task at a time, `exp28-fovi` is ~20 days and
the three drop parents ~12 / 15 / 6 days. Widen the array concurrency if that is
too slow.

### How to judge

The single scalar the framework tracks for checkpoint selection is
`eval/val_metric` = raw scene cosine similarity to the teacher at the last
evaluation timestep. **For comparison against exp22, use
`val/scene_cos_norm_t9`** — the normalized per-timestep series, which is what
selected the earlier best checkpoints.

The first comparable point is step 8192, the end of each arm's first array job:

| arm | exp22 `train_loss` @ step 8192 |
|---|---|
| `uniform16` | 1.8945 |
| `uniform16-teacherinit` | 1.6106 |
| `fovi` | 1.8885 |
| `fovi-teacherinit` | 1.8377 |

exp28's step-8192 losses should land near these. A *systematic* offset is
expected and is explained by the normalizer change below; a large or
arm-dependent one is not.

### Accepted differences from exp22

1. **Target-normalizer statistics are pooled over 4 shards** (the current
   default) instead of exp22's effective single shard. The pooled estimate is more
   accurate but shifts the target normalization, so **the loss scale is not
   identical and the curves will not overlay exp22's exactly.** The shards also
   differ in identity, not only in count: exp22 drew one shard from the
   seed-dependent shuffled order, whereas the current code takes the sorted head
   for a seed-independent pick.
2. **The `CanViT-PyTorch` pin moved** — verified irrelevant to `distill`: the
   diff touches only ADE20K data, metrics, the classification and segmentation
   heads, and the policy. Nothing in the distillation training path, so
   pretraining numerics are unchanged.
3. exp22's two foveated arms suffered a **mid-array pin change** that jumped
   their evaluation scale by +0.058. exp28 has no such discontinuity, which is
   better but means the new foveated curve will not track the old one below its
   break step.

Everything else — peak LR 4e-4, 100k warmup, batch 64, 8192 steps per job,
validation and logging cadence, worker count, the full foveated patcher geometry,
the teacher-init flag — was copied value-for-value from the exp22 launchers.

### Evaluation policy

`exp28` leaves `eval_policy` at `auto`, which resolves per patcher: the **uniform
arms validate under coarse-to-fine**, the **foveated arms under a fixation grid**
(a deterministic centre plus a 3×3 grid at the training scale, the scale-safe
choice for a fixed-scale foveated model). This matches exp22.

---

## exp29 — ImageNet-1k full finetunes

Four full finetunes, one per exp22 pretrained source. The recipe is exp25's,
which is the original TPU ImageNet-1k finetune batch-adapted for a single A100 by
the recipe's own sanctioned rule (batch 256→64, peak LR 2.5e-5→6.25e-6, warmup
25k→100k, 100,080 steps at batch 256 → 401,408 at batch 64). Everything else is
byte-identical to the TPU recipe.

| run | source checkpoint | eval policy |
|---|---|---|
| `in1k-uni16ti-803k` | exp22-uniform16-teacherinit-lrdrop2-803k | coarse-to-fine |
| `in1k-uni16-1516k` | exp22-uniform16-lrdrop-1516k | coarse-to-fine |
| `in1k-fovi-ti-1196k` | exp22-fovi-teacherinit-lrdrop-1196k | random |
| `in1k-fovi-1901k` | exp22-fovi `step-1900544` — **new arm** | random |

The foveated arms use `random` because coarse-to-fine is uniform-only and
out-of-distribution for a fixed-scale foveated model; they also pass
`--cfg.foveated-scale.fixed-scale 2.0` so the rollout views at the pretraining
scale. Both choices follow exp25.

All four fuse the pretrained probe into the classifier head via `CFG_PROBE_REPO`.
This flag is load-bearing, not cosmetic — see the failure mode below.

### Procedure

```bash
bash slurm/runs/exp29_in1k_finetune/in1k-uni16ti-803k.sh
bash slurm/runs/exp29_in1k_finetune/in1k-uni16-1516k.sh
bash slurm/runs/exp29_in1k_finetune/in1k-fovi-ti-1196k.sh
bash slurm/runs/exp29_in1k_finetune/in1k-fovi-1901k.sh
```

49 array jobs each.

### How to judge

ImageNet-1k reports top-1, which the ADE20K mIoU re-basing does not touch, so
**exp25 is a valid reference**:

| run | exp25 best `eval/top1` | at step | exp25 run reached |
|---|---|---|---|
| `in1k-uni16ti-803k` | **0.84954** | 320,000 | 400,000 (finished) |
| `in1k-fovi-ti-1196k` | **0.83692** | 270,000 | 320,000 (**incomplete** — of 401,408) |
| `in1k-uni16-1516k` | **0.83522** | 360,000 | 400,000 (finished) |
| `in1k-fovi-1901k` | — | — | no reference (new arm) |

Same sources, same recipe, only the pins differ — so the first three should land
close to those numbers, and a large miss is a real signal rather than noise. Note
that `in1k-fovi-ti-1196k`'s reference is **incomplete**: exp25's run stopped at
320,000 of 401,408 steps, so exp29's counterpart trains longer and may
legitimately exceed it.

**Only one number is reported, and it is measured at the final timestep.** The
classifier reads the CLS token of the *last* glimpse only
(`in1k/eval.py`), so `eval/top1` is accuracy after all glimpses; there is no
per-timestep top-1 series. These runs use **4 timesteps** (`CFG_N_TIMESTEPS=4`,
following exp25 and the TPU recipe), not the task default of 10.

**The specific failure to watch for:** if the pretrained probe is not fused into
the head, the finetune starts from a random classifier and the loss opens near
`ln(1000) ≈ 6.9` at chance accuracy. That was a live bug once. Check that the
first logged `train/full/loss` is well below 6.9.

---

## exp30 — ADE20K frozen probes

Four frozen-probe runs, one per exp22 pretrained source — the same four sources
as exp29. The recipe is exp24's, reproduced by the `ade20k` task defaults: frozen
backbone via the `probe` preset, 40k steps, random-view training, 10 timesteps,
scene 512, canvas grid 32. The foveated arms pass
`--cfg.foveated-scale.fixed-scale 2.0`.

**`resize_mode=squish` for every arm, including foveated.** That is the protocol
every earlier CanViT number was measured under, so it is what makes these
comparable to exp24 and to the published values. It distorts aspect ratio, so it
is *not* the right choice for a human-viewing comparison — `center_crop`
preserves the geometry foveated sampling assumes, at the cost of cropping the
long side. Whichever is used must be reported with the number.

### Procedure

```bash
bash slurm/runs/exp30_ade20k_probe/ade20k-uni16ti-803k.sh
bash slurm/runs/exp30_ade20k_probe/ade20k-uni16-1516k.sh
bash slurm/runs/exp30_ade20k_probe/ade20k-fovi-ti-1196k.sh
bash slurm/runs/exp30_ade20k_probe/ade20k-fovi-1901k.sh
```

Single-GPU (the ADE20K task does not support DDP).

### How to judge — mIoU is NOT directly comparable to exp24

exp24's pin **predates the mIoU reduction-order fix**, and exp30 includes it. The
old path took the argmax at the 64×64 probe grid and then nearest-upsampled,
locking every predicted label into an 8×8 pixel block; the corrected path
upsamples the logits bilinearly and takes the argmax at full resolution. The old
order is strictly coarser — a bug, not an alternative convention — and the
corrected order is what `CanViT-eval` and every published ADE20K number use.

Measured on the full 2000-image ADE20K validation set, both reductions from the
same logits: **+0.15 / +0.17 / +0.18 / +0.19 pp at t1–t4**, growing with t as the
segmentation sharpens, reproducible to 0.01 pp across seeds. Cross-entropy is
unaffected (it has its own bilinear path).

So exp30's mIoU should come out **above** exp24's by roughly +0.2 pp at late
timesteps — a metric-definition change, **not** an improvement. Do not read it as
one. The gain beyond t4 was never measured, so treat +0.2 pp as a lower bound at
t9.

| exp24 reference (stale basis) | `miou_final` | `miou_mean` | exp30 expectation |
|---|---|---|---|
| `ade20k-uni16ti-803k` | 0.44443 | 0.43305 | ~0.446 |
| `ade20k-uni16-1516k` | 0.41953 | 0.40769 | ~0.421 |
| `ade20k-fovi-ti-1196k` | 0.43838 | 0.41877 | ~0.440 |
| `ade20k-fovi-1901k` | — | — | no reference (new arm) |

Two further reasons not to expect equality: exp24's probes were **unseeded**, and
exp24 logged **no cross-entropy at all**, so there is no re-basing-immune metric
to fall back on.

**Therefore the meaningful checks are relative, not absolute:** the ordering
`uni16ti > fovi-ti > uni16` should hold, all four should sit in the 0.41–0.45
band, and the four exp30 runs are mutually comparable because they share code and
a seed. exp30 is the first ADE20K probe set on the corrected mIoU basis — it
becomes the reference for what follows.

### Two things that surprise people

- **The per-timestep `eval/miou_t*` values are measured under *random*
  viewpoints, not coarse-to-fine.** `eval_policy` is left at `auto`, and the
  ADE20K default is IID random viewpoints from a full-scene anchor — inherited
  from the probe recipe, which *trains* on random viewpoints, so validating on
  random is the consistent choice. exp24 did the same, which is what preserves
  comparability.
- **ADE20K train-mIoU is deliberately not logged.** Training loss and
  per-timestep validation mIoU are unaffected. ADE20K runs from before this
  campaign have train-mIoU panels that these runs will not.

---

## exp31 — ADE20K viewpoint policy, 10 seeds

Ten seeds of the Q-regression viewpoint policy: a `ViewpointScorer` is trained
against a **frozen** backbone and probe so that segmentation improves as fast as
possible per glimpse, and is deployed by taking the argmax over its candidate
grid. The recipe is exp27's `lossfix` arm verbatim; only the pins and the run
group change.

`--preset policy_only`, 8000 steps, 5 timesteps, batch 16, canvas grid 64,
`eval_policy=policy`. `CFG_RESIZE_MODE=squish` is the measurement contract the
reference band is defined by — it has silently regressed once before, when a
default change to `center_crop` made an arm land 0.016 "better" than the band,
about 20× the seed spread. Do not drop it.

### The frozen backbone needs no extraction

The published ADE20K-policy checkpoints contain **no backbone** — `best.pt` holds
only the `ViewpointScorer` (452 tensors, 5.68M parameters), because policy
training freezes the backbone and never saves it. It is also unnecessary: all 21
published policy checkpoints record the same `model_repo`, and that string is
already `Ade20kConfig.model_repo`'s default. So the premise that one identical
backbone is shared by every published policy is correct — it just needs no flag,
and `CFG_MODEL_REPO` is deliberately unset.

That backbone is the published pretrain, **not** any exp22 or exp28 model, so
exp31 is independent of exp28/29/30 and directly comparable to exp27.

### Procedure

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do
  SEED=$s bash slurm/runs/exp31_policy_qreg_10seed/policy-qreg-s0.sh
done
```

One single-GPU job per seed; 8000 steps fits inside one walltime.

### How to judge

exp27's `lossfix` pin **does** include the mIoU re-basing, so unlike
exp30-vs-exp24, **both cross-entropy and mIoU are directly comparable**:

| exp27 `lossfix`-s0 reference | value |
|---|---|
| best `eval/ce_mean` (mean t1–t4) | **0.68577** |
| final `eval/ce_mean` | 0.68676 |
| `eval/miou_final` | 0.44848 |
| `eval/miou_mean` | 0.43101 |

Published band: **0.6853 ± 0.0007** mean t1–t4 validation CE → [0.6846, 0.6860].
Expect the ten seeds to cluster with best `ce_mean` ≈ 0.685–0.686 and
`miou_final` ≈ 0.445–0.450.

**If a seed comes out materially BETTER than the band, distrust it.** That exact
failure mode has occurred here: a resize-default change put an arm at 0.6693,
0.016 "better" than the band and ~20× the seed spread, purely from the protocol
change. A result that beats the reference is evidence of a broken protocol until
proven otherwise.

**Free bit-identity check:** t0 is the full-image glimpse taken *before* any
policy action, so `ce_t0` and `miou_t0` depend only on the frozen backbone, the
probe, the resize and the evaluation path. If they match the reference to every
printed digit, any remaining difference is policy-side only.
