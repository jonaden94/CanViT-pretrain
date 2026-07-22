"""ADE20K segmentation-probe training on the stable CanViTForSemanticSegmentation
wrapper — ported from CanViT-specialize (unification master plan P2, decisions D3/D4).

Frozen backbone, per-timestep probe CE, patcher-aware glimpse routing (uniform
pre-crop vs foveated/square full-image). Entry: ``python -m canvit_pretrain.ade20k``.
"""
