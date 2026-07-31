# exp29 — ImageNet-1k full-model finetunes (harness, current code)

Four full finetunes, one per exp22 pretrained source. Recipe copied value-for-value from
**exp25**, which is the original `canvit_specialize` TPU in1k finetune batch-adapted for one
A100 by the recipe's own sanctioned rule (batch 256->64, `peak_lr` 2.5e-5->6.25e-6, warmup
25k->100k, 100,080@256 -> 401,408@64). Everything else byte-identical to the TPU recipe.

| run | source checkpoint | eval policy |
|---|---|---|
| `in1k-uni16ti-803k` | exp22-uniform16-teacherinit-lrdrop2-803k `step-16384-hf` | coarse_to_fine |
| `in1k-uni16-1516k` | exp22-uniform16-lrdrop-1516k `step-319488-hf` | coarse_to_fine |
| `in1k-fovi-ti-1196k` | exp22-fovi-teacherinit-lrdrop-1196k `step-155648-hf` | random |
| `in1k-fovi-1901k` | **exp22-fovi `step-1900544-hf` — NEW** | random |

The first three are exactly the sources exp24/exp25 used. **The fourth is new**: exp22-fovi
never had a downstream run. Its best `val/scene_cos_norm_t9` is 0.853339 at step 1,900,544
(max over all 238 val points, wandb run `r64ck13l`), converted with `to_hf` on 2026-07-31.
The pre-existing `step-516096-hf` export in that run is **not** the best checkpoint — do not
use it.

`EVAL_POLICY=random` for the foveated arms because coarse-to-fine is uniform-only and OOD for
a fixed-scale foveated model; those arms also pass `--cfg.foveated-scale.fixed-scale 2.0` so
the rollout views at the pretrain scale. Both choices follow exp25.

Uses the pretrained probe (`CFG_PROBE_REPO`) fused into the head — a random head here was a
real bug once (`8f780ba`), so this flag is load-bearing, not cosmetic.

## How to judge these results

in1k reports top-1, which the ADE20K mIoU re-basing does not touch, so **exp25 is a valid
reference**:

| exp25 reference | BEST `eval/top1` | at step | run reached |
|---|---|---|---|
| `in1k-uni16ti-803k` | **0.84954** | 320,000 | 400,000 (finished) |
| `in1k-fovi-ti-1196k` | **0.83692** | 270,000 | 320,000 (INCOMPLETE -- 320k of 401,408) |
| `in1k-uni16-1516k` | **0.83522** | 360,000 | 400,000 (finished) |
| `in1k-fovi-1901k` | — | — | no reference (new arm) |

Pulled from wandb (project `exp25`, max over history), 2026-07-31. An earlier version of this
table listed 0.84026 / 0.82786 / 0.82482 -- those were read from a MID-ARRAY job log: these are
49-job arrays and `ls *.log | tail -1` sorts alphabetically, so `job-X_9.log` outranks
`job-X_48.log`. All three were ~1 pp too low. Read final numbers from wandb, not from a log
picked by shell glob order.

Note `in1k-fovi-ti-1196k` stopped at 320,000 of 401,408 steps, so it is an INCOMPLETE
reference -- exp29's counterpart will train longer and may legitimately exceed it.

Same source checkpoints, same recipe, only the pins differ — so these three should land close
to those numbers. A large miss is a real signal, not noise.

**The specific failure to watch for:** if the pretrained probe is not fused into the head, the
finetune starts from a RANDOM classifier and the loss opens near `ln(1000) ~= 6.9` with
chance-level accuracy. That was a live bug once (fixed `8f780ba`), which is why
`CFG_PROBE_REPO` is load-bearing here rather than cosmetic. Check the first logged
`train/full/loss` is well below 6.9.

## Submitted job IDs (2026-07-31)

`in1k-uni16ti-803k` 15113362 · `in1k-uni16-1516k` 15113363 · `in1k-fovi-ti-1196k` 15113364 · `in1k-fovi-1901k` 15113365
