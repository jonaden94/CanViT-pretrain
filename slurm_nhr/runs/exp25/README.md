# exp25 — ImageNet-1k full-model finetunes (through the harness)

Three IN1k **full-model** finetunes, one per pretrained source, from each source's
**best-`val/scene_cos_norm_t9`** checkpoint (converted to local HF format via `to_hf`) —
the in1k counterparts of the exp24 ade20k probes.

| run (= wandb name) | source | `model-repo` | eval policy |
|---|---|---|---|
| `in1k-uni16ti-803k` | exp22-uniform16-teacherinit-lrdrop2-803k | step-16384-hf | coarse_to_fine |
| `in1k-uni16-1516k` | exp22-uniform16-lrdrop-1516k | step-319488-hf | coarse_to_fine |
| `in1k-fovi-ti-1196k` | exp22-fovi-teacherinit-lrdrop-1196k | step-155648-hf | random (+ fixed-scale 2.0) |

The **wandb run name = the finetune base model**, so the three are distinguishable in the
`exp25` project. (The ade20k side in exp24 couldn't do this: the harness hardcodes ade20k's
wandb name to `"ade20k"`; in1k reads `cfg.run_name`, so `CFG_RUN_NAME` sets it.)

## Recipe = the original TPU in1k finetune (`canvit_specialize/.../gcp_in1k_clf_ft`), batch-adapted

Full-model AdamW finetune over a 4-glimpse rollout (full BPTT), t0 = full scene, linear
warmup → cosine-to-0. Adapted to one A100 by the recipe's OWN sanctioned rule
(yaml comment: *"Scale LR linearly with batch size if changed"*):

| HP | TPU (batch 256) | here (batch 64) |
|---|---|---|
| batch_size | 256 | **64** (÷4) |
| peak_lr | 2.5e-5 | **6.25e-6** (×64/256) |
| warmup_steps | 25 000 | **100 000** (×256/64, holds the ~5-epoch warmup) |
| total steps | 100 080 (20 ep) | **401 408** = 49×8192 (~20 ep) |
| weight_decay | 1e-4 | 1e-4 |
| grad_clip | 1.0 | 1.0 |
| label_smoothing | 0.1 | 0.1 |
| n_glimpses (= n_timesteps) | 4 | 4 |
| min_vp_scale | 0.05 | 0.05 |
| glimpse / canvas / scene | 128 / 32 / 512 | 128 / 32 / 512 (derived from the g128px-s512px model) |

Only batch, LR, and (warmup + total) steps move, all by the batch ratio → these runs are
comparable to the TPU checkpoints **up to the batch adaptation**. Linear LR scaling is a
heuristic (weaker for AdamW than SGD), so expect final top-1 *close but not identical* to a
true batch-256 run. Deliberate, owner-approved — NOT a bit-identical reproduction.

**Eval policy:** uniform → coarse-to-fine (owner's deliberate choice; the TPU *default* was
random — training-independent, only affects reported val top-1 / `best.pt` selection).
Foveated → random (c2f is uniform-only / OOD for a fixed-scale foveated backbone; the foveated
source also has no original TPU in1k counterpart — it's the uniform recipe on a foveated model).

## Array / resume

49-job array (`0-48%1`), 8192 steps/job = 128 shards/job — shard-aligned
(8192 × 64 = 524 288 = 128 × 4096-img shards). Cross-job resume uses the same shard schedule
the distill runs use (`--opts.resume True`; in1k's default is no-resume). 12 h/job: 8192
in1k-finetune steps ≈ distill's 8192@2h but heavier, so 12 h is a wide margin — a mid-job
timeout would land the checkpoint off a shard boundary and break the resume.

Pins: pretrain `8f780ba`, pytorch `017ce9b`, fovi `c399d3b` (same stack as exp23/exp24).

Submit:
```bash
cd /user/henrich1/u25995/jonathan/repos/CanViT-pretrain
bash slurm_nhr/runs/exp25/in1k-uni16ti-803k.sh
bash slurm_nhr/runs/exp25/in1k-uni16-1516k.sh
bash slurm_nhr/runs/exp25/in1k-fovi-ti-1196k.sh
```
