# CanViT-train

Pretraining and evaluation for CanViT.

## Structure

```
canvit_pretrain/    # Pretraining
canvit_eval/        # Evaluation (IN1k, ADE20k)
drac_imagenet/      # Fast indexed ImageFolder for IN21k
scripts/            # Utilities (benchmarks, feature export)
slurm/              # SLURM scripts: pretraining, IN1k, env setup
```

## Entry Points

```bash
# Pretraining
uv run -m canvit_pretrain.train --help

# IN1k probe evaluation
uv run -m canvit_eval.in1k --help

# ADE20k segmentation (4 subcommands)
uv run -m canvit_eval.ade20k --help
```

### ADE20K Subcommands

```bash
# CanViT canvas probe training
uv run -m canvit_eval.ade20k train --help

# CanViT canvas probe evaluation (with viewpoint policy)
uv run -m canvit_eval.ade20k evaluate --probe-ckpt best.pt --policy coarse_to_fine

# DINOv3 baseline probe training at a given resolution
uv run -m canvit_eval.ade20k train-dinov3-probe --resolution 128
uv run -m canvit_eval.ade20k train-dinov3-probe --resolution 512

# DINOv3 baseline probe evaluation (deterministic, single run)
uv run -m canvit_eval.ade20k eval-dinov3-probe --probe-ckpt dinov3_best.pt
```

Canvas probes and DINOv3 baselines are separate entry points — no flags, no polymorphism.
SLURM scripts for ADE20K live in `canvit_eval/ade20k/slurm/`.

## SLURM (Nibi cluster)

**Setup** (one-time):
```bash
cd ~/scratch
git clone git@github.com:m2b3/CanViT-train.git
uvx hf auth login
```

**Usage**:
```bash
cd ~/scratch/CanViT-train

# Pretraining & IN1k (scripts in slurm/)
sbatch slurm/train.sbatch
sbatch slurm/eval_in1k.sbatch
bash slurm/interactive.sh

# ADE20K (scripts in canvit_eval/ade20k/slurm/)
sbatch canvit_eval/ade20k/slurm/train_ade20k_probes.sbatch
sbatch canvit_eval/ade20k/slurm/eval_ade20k_policy.sbatch c2f /path/to/probe.pt
sbatch canvit_eval/ade20k/slurm/train_ade20k_dinov3_probe.sbatch 128
sbatch canvit_eval/ade20k/slurm/train_ade20k_dinov3_probe.sbatch 512
sbatch canvit_eval/ade20k/slurm/eval_ade20k_dinov3_probe.sbatch /path/to/dinov3_best.pt
```

**Key files**:
- `slurm/env.sh` - Environment setup (dataset paths, caches, Comet key)
- `outputs/` - Evaluation results (.pt files)
- `logs/` - Job stdout/stderr

## See Also

- `CLAUDE.md` - Development guidelines
- `canvit` package - Core model architecture (separate repo)
