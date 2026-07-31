# exp24 — ADE20K frozen-probe finetunes (through the harness)

Three ADE20K frozen-probe runs, one per pretrained source, from each source's
**best-`val/scene_cos_norm_t9`** checkpoint (converted to local HF format via `to_hf`).

| run | source | `model-repo` | resize |
|---|---|---|---|
| `ade20k-uni16ti-803k` | exp22-uniform16-teacherinit-lrdrop2-803k | step-16384-hf | squish |
| `ade20k-uni16-1516k` | exp22-uniform16-lrdrop-1516k | step-319488-hf | squish |
| `ade20k-fovi-ti-1196k` | exp22-fovi-teacherinit-lrdrop-1196k | step-155648-hf | squish + `foveated-scale.fixed-scale 2.0` |

**Recipe = the original specialize ade20k probe**, reproduced by the harness `ade20k` defaults
(mIoU gate passed vs the specialize-derived standalone): frozen backbone (default preset =
`TrainSpec.probe`), 40k steps, random-view training, `n_timesteps=10`, scene 512, `canvas_grid=32`,
val resize `squish` (the specialize reference protocol). The only non-essential difference from the
original is the entry point (harness vs `canvit_specialize`).

**Resize protocol:** all three (incl. foveated) eval under `squish`, for cross-run comparability
and because that is what the earlier CanViT / specialize numbers were measured under. It distorts
aspect ratio, so it is not the protocol to pick for a human-viewing comparison — there
`center_crop` preserves the geometry the foveated sampling assumes, at the cost of cropping the
long side. Both are supported for every patcher; the choice just has to be reported with the
number. `resize_mode` affects the val/eval resize only, not training.

Each run is a single 8h job (frozen probe → no cross-job resume). Submit:
```bash
cd /user/henrich1/u25995/jonathan/repos/CanViT-train
bash slurm/runs/exp24/ade20k-uni16ti-803k.sh
bash slurm/runs/exp24/ade20k-uni16-1516k.sh
bash slurm/runs/exp24/ade20k-fovi-ti-1196k.sh
```
in1k finetunes live in `slurm/runs/exp25/` (batch pending a memory smoke — see that README).
