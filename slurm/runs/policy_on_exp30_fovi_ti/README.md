# Viewpoint policy on the foveated exp22 backbone + its exp30 probe

Starting point for training ADE20K viewpoint policies on top of the best foveated model.

## The model is two HF directories, not one `.pt`

| flag | path |
|---|---|
| `--cfg.model-repo` | `logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648-hf` |
| `--cfg.probe-repo` | `logs/exp30_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best-probe-hf` |

Both under `/mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train/`, group-readable.

Together they rebuild exp30's `best.pt` (step 30,500, `miou_final` 0.4434) **exactly** —
every head and backbone tensor bit-identical, no missing keys. The probe half was exported
from that checkpoint with:

```bash
python -m canvit_train.checkpoint.probe_to_hf \
    --pt-path logs/exp30_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best.pt \
    --out-dir logs/exp30_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best-probe-hf
```

The `.pt` cannot be passed directly: `probe_repo` is read with
`SegmentationProbe.from_pretrained`, and `to_hf` deliberately refuses `ade20k` payloads
(it publishes whole models, and an ade20k run's only new weights are the head).

The probe is **load-bearing, not cosmetic** — the reward is the fraction of the probe's CE
that a glimpse removes, so a random head makes the reward pure noise.

## Three settings that are not free choices

| setting | value | why |
|---|---|---|
| `--cfg.canvas-grid` | **32** | the exp30 probe was trained at 32. exp31 used 64 because *its* probe (`canvit/probe-ade20k-40k-s512-c64-in21k`) was trained at 64. Match the probe, not exp31. |
| `--cfg.foveated-scale.fixed-scale` | **2.0** | a foveated backbone derives `fix_size = scale * H`; any other scale makes every glimpse OOD. Does not crash — mIoU just falls as glimpses accumulate. |
| `--cfg.resize-mode` | **squish** | the protocol every earlier CanViT number was measured under. Has silently regressed once, putting an arm 0.016 "better" than the reference band. |

## Run

```bash
bash slurm/runs/policy_on_exp30_fovi_ti/policy-qreg-fovi-ti-s0.sh
# or 10 seeds:
for s in 0 1 2 3 4 5 6 7 8 9; do SEED=$s bash slurm/runs/policy_on_exp30_fovi_ti/policy-qreg-fovi-ti-s0.sh; done
```

Single A100, 8000 steps, ~8 h walltime. Set `PRETRAIN_COMMIT` / `PYTORCH_COMMIT` /
`FOVI_COMMIT` in the script to pin the code if the run matters.

## Status: config validated, never trained

The recipe is exp31's `lossfix` arm with this backbone+probe swapped in and the grid matched
to the probe. Verified by parsing it through the CLI: `preset=policy_only` →
`train_backbone/head/policy = False/False/True`, one `policy` optim group at lr 2e-4 betas
(0.9, 0.95), `best_metric=neg_ce_mean`, `augment=False`, `resize_mode=squish`,
`foveated_scale=fixed 2.0`, objective `qreg`, BN mode (b).

**No training step has been taken with it.** exp31 ran against the published *uniform*
backbone, so this is the first foveated policy arm — the action space becomes the fixation
grid rather than the safe-box grid. Treat the first run as a smoke test: `eval/miou_t0` is
the pre-policy full-image glimpse, so it should match a probe-only evaluation before any
later timestep is trusted.

exp31's numbers are **not** a reference here: different backbone, different probe,
different grid. See `../exp31_policy_qreg_10seed/README.md` for what that arm measured.
