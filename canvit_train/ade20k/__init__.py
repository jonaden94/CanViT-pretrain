"""ADE20K segmentation-probe training on the stable CanViTForSemanticSegmentation
wrapper — ported from CanViT-specialize (unification master plan P2, decisions D3/D4).

Frozen backbone, per-timestep probe CE, patcher-aware glimpse routing (uniform
pre-crop vs foveated/square full-image).

This package now holds only the SHARED ADE20K pieces — config, data/transforms, metrics
(incl. the policy reward `reward_ce`), rollout helpers, viz. The trainers that used to
live here (``train.py``, the standalone probe; ``rl_train.py``, the ported
CanViT-PyTorch-RL policy reference) were deleted in the 2026-07-31 consolidation; both
are reachable as ``python -m canvit_train.harness.run ade20k --preset {probe,policy_only}``.
"""
