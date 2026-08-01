# Viewpoint policy on our own foveated backbone + probe

The Q-regression ADE20K viewpoint policy trained on the exp22 foveated-teacherinit backbone
and the ADE20K probe trained on it, instead of on the published pair.

**The procedure, and the settings that must not be copied blindly, are documented in
[`readme_docs/q_policy_foveated.md`](../../../readme_docs/q_policy_foveated.md).**
Read that first — the three silent-failure knobs (canvas grid, foveated scale, resize mode)
are explained there.

## Sources

| flag | value |
|---|---|
| `--cfg.model-repo` | `logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt` |
| `--cfg.probe-repo` | `logs/exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints/best.pt` |

Both are the training checkpoints themselves — both flags read a `.pt` directly, so there
is no conversion step.

## Run

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do
  SEED=$s bash slurm/runs/policy_on_own_fovi_probe/policy-qreg-own-fovi-s0.sh
done
```

Single A100 per seed, 8000 steps, ~8 h walltime.

The launcher **refuses until the probe run has finished** (it tests for `step-40000.pt`, not
`best.pt`, which appears at the first evaluation). To use a different finished ADE20K run as
the probe, repoint `_PROBE_RUN` — and set `--cfg.canvas-grid` to the grid that probe was
trained at.

Set `TRAIN_COMMIT` / `PYTORCH_COMMIT` / `FOVI_COMMIT` before a run that matters; left empty,
the job uses the editable install and says so in its log.
