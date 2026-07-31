"""CanViT training: all training modes behind one harness.

- ``distill``  — passive-to-active dense latent distillation from DINOv3 (pretraining)
- ``ade20k``   — ADE20K segmentation probe / finetune
- ``in1k``     — ImageNet-1k linear probe / full finetune
- plus the viewpoint-selection POLICY, trainable on any of the three (``--preset
  policy_only`` / ``joint``)

Single entry point: ``python -m canvit_train.harness.run <task> --preset <preset>``.
"""

from canvit_pytorch.model.pretraining import CanViTForPretraining, CanViTForPretrainingConfig

__all__ = [
    "CanViTForPretraining",
    "CanViTForPretrainingConfig",
]
