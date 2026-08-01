# exp35 — exp31 re-run clean on the current code base

Ten seeds of the Q-regression ADE20K viewpoint policy, under the current pins.
**Config-identical to exp31**: only the run identity and the two bumped commits differ,
verified mechanically.

Like exp30, exp31 **finished** — all ten seeds completed. exp35 is a deliberate re-run on
current code, not a recovery.

| repo | exp31 | exp35 |
|---|---|---|
| `canvit_train` | `455bdae` | **`009f262`** |
| `canvit_pytorch` | `1f5121b` | **`d616b7b`** |
| `fovi` | `c399d3b` | unchanged |

## Comparable to exp31, and that is the point

Same recipe (exp27's `lossfix` arm), same frozen backbone, same probe, same seeds. exp31's
pin already included the mIoU re-basing, so **both CE and mIoU are directly comparable**.
exp35 re-running clean on the current code is the check that the loader/config refactor did
not perturb policy training.

| exp27 `lossfix`-s0 reference | value |
|---|---|
| best `eval/ce_mean` (mean t1–t4) | **0.68577** |
| `eval/miou_final` | 0.44848 |

Published qband: **0.6853 ± 0.0007** → [0.6846, 0.6860]. Expect the ten seeds to cluster at
best `ce_mean` ≈ 0.685–0.686 and `miou_final` ≈ 0.445–0.450.

**If a seed comes out materially BETTER than the band, distrust it.** That failure mode has
occurred here: a resize-default change put an arm at 0.6693 — 0.016 "better", ~20× the seed
spread — purely from the protocol change. `CFG_RESIZE_MODE=squish` is the measurement
contract; do not drop it.

**Free bit-identity check:** `ce_t0` / `miou_t0` are the full-image glimpse taken *before*
any policy action, so they depend only on the frozen backbone, probe, resize and eval path.
If they match exp31's to every printed digit, any residual difference is policy-side only —
which is exactly what this re-run is testing.

## Backbone and probe (unchanged)

`CFG_MODEL_REPO` is deliberately unset: `Ade20kConfig.model_repo`'s default already IS the
published backbone every policy checkpoint records. `CFG_PROBE_REPO` is the published
c64 probe. Both stay Hub ids so this is config-identical to exp31.

Those could now be local `.pt` files instead — `--cfg.probe-repo` accepts a training
checkpoint directly — but changing them would make this something other than a replication.
See `../policy_on_exp30_fovi_ti/` for a policy arm that does use local `.pt` sources.

`--preset policy_only`, 8000 steps, 5 timesteps, batch 16, canvas grid 64,
`eval_policy=policy`, `--cfg.no-augment`. Single GPU per seed.

## Submit

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do
  SEED=$s bash slurm/runs/exp35_policy_qreg_10seed/policy-qreg-s0.sh
done
```
