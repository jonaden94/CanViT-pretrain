# exp33 — exp29 restarted clean on the current code base

The four ImageNet-1k full finetunes from exp29, re-run under the current pins.
**Config-identical to exp29**: only the run identity and the two bumped commits differ,
verified mechanically (rewriting `exp33`→`exp29` and stripping the pin lines makes every
launcher byte-identical to its exp29 counterpart).

exp29 was cancelled with 1 of 49 array tasks done on `in1k-uni16ti-803k` and 0 on the other
three, so almost nothing is lost. A new run group means empty checkpoint directories, so
each arm starts at step 0.

| repo | exp29 | exp33 |
|---|---|---|
| `canvit_train` | `455bdae` | **`009f262`** |
| `canvit_pytorch` | `1f5121b` | **`d616b7b`** |
| `fovi` | `c399d3b` | unchanged |

## This is the group the current code actually changes

Unlike the pretrains, an in1k checkpoint used to record only a `model_repo` **path** — a
pointer, not a description. That is a problem precisely for a finetune: finetuning updates
the backbone, so the checkpoint is the only place those weights exist, while the pointer
still names the model it *started* from. Such a checkpoint could not rebuild itself, which
is why `to_hf` was the only way to get a finetuned model out.

exp33's checkpoints record the architecture, so a finetuned model loads straight from its
`.pt`:

```python
from canvit_pytorch.model_source import load_classifier
clf = load_classifier("logs/exp33_in1k_finetune/<run>/checkpoints/best.pt")
```

`to_hf` still works and is still the right thing for publishing.

## Sources and recipe (unchanged from exp29)

| run | source checkpoint | eval policy |
|---|---|---|
| `in1k-uni16ti-803k` | exp22-uniform16-teacherinit-lrdrop2-803k `step-16384-hf` | coarse_to_fine |
| `in1k-uni16-1516k` | exp22-uniform16-lrdrop-1516k `step-319488-hf` | coarse_to_fine |
| `in1k-fovi-ti-1196k` | exp22-fovi-teacherinit-lrdrop-1196k `step-155648-hf` | random |
| `in1k-fovi-1901k` | exp22-fovi `step-1900544-hf` | random |

Sources deliberately stay the exp22 **HF exports**, not their `.pt` originals, so exp33 is
config-identical to exp29. Both now load identically, so switching to `.pt` would be a free
change — but it would be a change, and the point of this group is that nothing moved except
the pins.

`n_timesteps=4` (not the task default of 10), the DINOv3 probe fused into the head via
`CFG_PROBE_REPO`, foveated arms on `random` + `fixed-scale 2.0`. 49 array jobs each.

## How to judge

exp25's best `eval/top1` remains the reference, and top-1 is untouched by the ADE20K mIoU
re-basing:

| run | exp25 best `eval/top1` | note |
|---|---|---|
| `in1k-uni16ti-803k` | **0.84954** | |
| `in1k-fovi-ti-1196k` | **0.83692** | reference INCOMPLETE (320k of 401,408 steps) |
| `in1k-uni16-1516k` | **0.83522** | |
| `in1k-fovi-1901k` | — | no reference (new arm in exp29) |

**Watch the first logged `train/full/loss`:** well below `ln(1000) ≈ 6.9`. If it opens near
6.9 the pretrained probe was not fused and the finetune is starting from a random
classifier — a live bug once, which is why `CFG_PROBE_REPO` is load-bearing here.

## Submit

```bash
bash slurm/runs/exp33_in1k_finetune/in1k-uni16ti-803k.sh
bash slurm/runs/exp33_in1k_finetune/in1k-uni16-1516k.sh
bash slurm/runs/exp33_in1k_finetune/in1k-fovi-ti-1196k.sh
bash slurm/runs/exp33_in1k_finetune/in1k-fovi-1901k.sh
```
