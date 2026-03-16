# CanViT-pretrain

Self-supervised pretraining for CanViT: dense latent distillation from DINOv3.

## Structure

```
canvit_pretrain/    # Pretraining loop, data, checkpointing, viz
drac_imagenet/      # Fast indexed ImageFolder for IN21k
scripts/            # Feature export, checkpoint push, inspection
slurm/              # SLURM scripts for Nibi cluster
```

## Entry points

```bash
# Pretraining
uv run python -m canvit_pretrain.train --help

# Feature export (precompute DINOv3 teacher features)
uv run python scripts/export_in21k_features.py --help
```

## Related repos

| Repo | Role |
|------|------|
| [CanViT-PyTorch](https://github.com/m2b3/CanViT-PyTorch) | Core model + policies |
| [CanViT-probes](https://github.com/m2b3/CanViT-probes) | Probe definitions, datasets, metrics, training |
| [CanViT-eval](https://github.com/m2b3/CanViT-eval) | Evaluation (produces .pt result files) |
| [CanViT-Toward-AVFMs](https://github.com/m2b3/CanViT-Toward-AVFMs) | Paper (.pt → JSON → PDF) |
