# exp27 — does the UNIFIED HARNESS reproduce the CanViT-PyTorch-RL policy recipe?

wandb project `exp27`. Two arms, seed 0, same GPU class, same pinned commits.

| arm | script | entry point | status before this |
|---|---|---|---|
| A (control) | `policy-oldloop-s0.sh` | `python -m canvit_pretrain.ade20k.rl_train` | gate-validated 2026-07-23 |
| B | `policy-harness-s0.sh` | `harness.run ade20k --preset policy_only` | **never run at scale** |

## Why a control arm at all

The published reference is the RL repo's 8-seed qband band, **0.6853 ± 0.0007** mean
t1–t4 val CE (`CanViT-PyTorch-RL/docs/qband_results.md`, per-seed spread 0.6845…0.6865),
with EG-C2F-c64 at 0.6949. The P3 gate already reproduced it here (jobs 15025279 /
15025337 → 0.6855 / 0.6867).

Arm A is re-run anyway for two reasons:

1. The band was measured on the RL repo's machine. A local reference removes the
   hardware/stack question from the comparison entirely.
2. Those gate runs predate `845e401` (per-timestep mIoU in the deploy eval), so they
   logged **CE only**. Arm B reports CE *and* mIoU; without arm A, half the comparison
   has no counterpart.

**Rule (learned in exp23/exp26): never gate a production A/B on one baseline run.** If A
and B disagree, the next step is a second seed of A — not a verdict. See
[[../../unification_docs/14-parity-coverage.md]].

## Verdict rule

Judge on **mean(t1–t4) val CE**, the metric the qband band is defined by and the one
arm B selects `best.pt` on (`neg_ce_mean`).

- Arm B inside 0.6845…0.6865 → the harness reproduces the recipe. The policy path is
  production-gated and CanViT-RL work can proceed on it.
- Arm B outside, arm A inside → a harness-specific problem. First suspect: the one known
  remaining config difference, **reward `score_res` 128 (A) vs the probe's native 64 (B)**
  — doc 15 §A gap #5, owner-deferred. The reward *formula* is already identical.
- Both outside → not a harness question; look at the stack (pins, probe repo, data).

Also worth watching, independent of the verdict: **mIoU at t1–t4** (band: 42.65 → 44.97)
and `reward_frac` trending positive. A policy that is learning nothing shows a flat
`reward_frac` near 0 while CE sits at its t0 value.

## What had to be fixed before B was even meaningful

All 2026-07-28/29, committed in `cea4dee`, detailed in
[[../../unification_docs/15-rl-recipe-parity-and-open-items.md]] §A:

1. **Validation deployed no policy.** `--preset policy_only` validated on RANDOM
   glimpses and selected `best.pt` on that mIoU. Fixed by the shared `eval_policy` knob
   (`--cfg.eval-policy policy`) + `best_metric` following to `neg_ce_mean`.
2. **Adam betas** — every group silently got torch's `(0.9, 0.999)`; RL uses `(0.9, 0.95)`.
3. **No LR ramp** — the policy group fell through to `warmup_steps=0`.
4. **Augmentation** — rl_train trains on unaugmented images; the harness augmented
   unconditionally. Needed its own flag: neutralising the aug knobs does NOT disable
   augmentation (`RandomCropWithLabel` + `PhotoMetricDistortion` have no knob).
5. **The probe head (blocking).** `probe_repo` was gated on `mode=="finetune"`, but
   `policy_only` runs FROZEN — so it built a **fresh random head**. The reward *is* the
   probe's CE reduction, so the scorer would have trained on pure noise with nothing
   failing and every log looking healthy.

## The first attempt was voided by the resize protocol (2026-07-29)

Jobs 15093707 / 15093712 / 15093767 / 15093768 ran **center_crop on both arms** and were
cancelled. Arm A s0 completed at mean t1–t4 CE **0.6693** — 0.016 *better* than the band,
~20× its own 0.0007 seed spread. That was not a better policy, it was a different
measurement.

CanViT-PyTorch-RL squish-resizes image **and** mask to `scene_size` everywhere — its
`config.py` docstring calls this "the measurement contract every entry point builds on",
and its dataset class is literally `Ade20kSquish`. The qband band and the EG-C2F
baselines exist only under squish. At the P3 gate commit `7e5afac`, `rl_train.py:329`
hardcoded it. Commit **`1a0b452`** ("Lift val resize_mode into ADE/policy configs, default
center_crop") turned it into a knob and defaulted it to `center_crop`, silently
decoupling the reference from its band; neither launcher set it.

Fixed three ways: `PolicyTrainConfig.resize_mode` is pinned back to `squish` (it is the
frozen reference — `Ade20kConfig` deliberately keeps `center_crop` for new work), both
launchers now set it explicitly, and a policy run under any other mode logs a loud
not-band-comparable warning. Pinned by `test_rl_train_defaults_to_the_bands_squish_protocol`
and `test_harness_policy_run_warns_when_not_band_comparable`.

**Everything else audited clean** against the original repo (2026-07-29): same
`model_repo` and probe rule, same `make_val_transforms` function (equivalence-tested for
both modes in `unification_docs/specialize_equivalence.py`), full val split with no
limit/stride, eval CE at full 512², objective mean over t1–t4, and every recipe
hyperparameter matching `TrainConfig` (lr 2e-4, wd 1e-2, betas .9/.95, clip 1.0,
score_res 128, 640k forwards, batch 16, horizon 4, warmup 0.125, target_momentum 0.997,
scales (0.5,0.25), centers 16, width 128, blocks 3, prime_on_policy 0.5, dueling). The
only remaining diffs are immaterial to the metric: eval batch 32 vs 16, workers 4 vs 8.

## Gotcha for anyone editing arm B

`--cfg.no-augment` lives in `EXTRA_ARGS`, **not** as `CFG_AUGMENT=False`. tyro renders
bools as paired flags (`--cfg.augment` / `--cfg.no-augment`) while the launcher's
`CFG_FOO_BAR` mapping emits `--cfg.foo-bar VALUE`, which cannot express a flag. The same
applies to every other bool knob. Caught by a local smoke run before submitting; a
launcher-only test would have passed the wrong thing silently.

`ADE20K_ROOT` is **not** in `env.sh` — both scripts export it, as every exp24 script does.

## Submit

```bash
bash slurm_nhr/runs/exp27/policy-oldloop-s0.sh    # ~65 min on one A100
bash slurm_nhr/runs/exp27/policy-harness-s0.sh    # 8000 steps, same order
```

Pins: `PRETRAIN=cea4dee`, `PYTORCH=017ce9b`, `FOVI=c399d3b` on both arms. `rl_train.py`
is untouched by `cea4dee` (the session changed the harness path), so arm A pins "today's
code" without depending on any of this session's work.
