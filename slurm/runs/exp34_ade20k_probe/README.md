# exp34 — exp30 re-run clean on the current code base

The four ADE20K frozen-probe runs from exp30, under the current pins.
**Config-identical to exp30**: only the run identity and the two bumped commits differ,
verified mechanically.

Note exp30 **finished** — it was not cancelled (all four runs reached step 40,000). exp34
is a deliberate re-run on current code, not a recovery.

| repo | exp30 | exp34 |
|---|---|---|
| `canvit_train` | `455bdae` | **`009f262`** |
| `canvit_pytorch` | `1f5121b` | **`d616b7b`** |
| `fovi` | `c399d3b` | unchanged |

## What changes, and what does not

**The metric basis does NOT change.** exp30 already included the mIoU reduction-order fix
(`68b635f`), so exp34 is directly comparable to exp30 — unlike exp30-vs-exp24, which was
not. Same recipe, same sources, same seed: exp34 should reproduce exp30's numbers closely,
and a large deviation is a real signal.

| exp30 result | best `miou_final` |
|---|---|
| `ade20k-uni16ti-803k` | 0.44479 |
| `ade20k-fovi-ti-1196k` | **0.4434** (best.pt at step 30,500) |
| `ade20k-uni16-1516k` | 0.42321 |
| `ade20k-fovi-1901k` | no earlier counterpart |

Ordering to expect: `uni16ti > fovi-ti > uni16`, all in the 0.41–0.45 band.

**The checkpoints do change.** An ade20k checkpoint now records its architecture rather than
only a `model_repo` path, so an exp34 checkpoint can rebuild itself:

```python
from canvit_pytorch.model_source import load_segmentation
seg = load_segmentation("logs/exp34_ade20k_probe/<run>/checkpoints/best.pt")
```

For a frozen **probe** this matters less than for a finetune (the backbone is unchanged, so
`model_repo` + probe was always a valid reconstruction) — but it removes the dependency on
that path still resolving, which for exp30's checkpoints is a path only its owner can
traverse.

The probe half of an exp30/exp34 checkpoint is usable directly as `--cfg.probe-repo`
without conversion now; `canvit_train.checkpoint.probe_to_hf` remains for publishing.

## Recipe (unchanged)

Frozen backbone via the default `probe` preset, 40k steps, random-view training,
`n_timesteps 10`, scene 512, `canvas_grid 32`, `resize_mode=squish` for every arm including
foveated (the protocol every earlier CanViT number was measured under), foveated arms with
`--cfg.foveated-scale.fixed-scale 2.0`. Single GPU — the ADE20K task does not support DDP.

Sources are the same four exp22 HF exports as exp33; kept as HF dirs so this stays
config-identical to exp30.

ADE20K train-mIoU is deliberately not logged; train loss and per-timestep val mIoU are
unaffected.

## Submit

```bash
bash slurm/runs/exp34_ade20k_probe/ade20k-uni16ti-803k.sh
bash slurm/runs/exp34_ade20k_probe/ade20k-uni16-1516k.sh
bash slurm/runs/exp34_ade20k_probe/ade20k-fovi-ti-1196k.sh
bash slurm/runs/exp34_ade20k_probe/ade20k-fovi-1901k.sh
```
