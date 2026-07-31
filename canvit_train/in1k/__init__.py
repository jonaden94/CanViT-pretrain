"""ImageNet-1k classification task for the unified harness (unification P5, D2).

A FRESH CUDA implementation — the specialize repo's TPU/XLA classification path
is abandoned, not ported (master plan D2). Modeled on the ``ade20k`` task package:
a frozen-backbone linear probe (default) or full finetune of
``CanViTForImageClassification`` over a glimpse rollout, trained step-based (DDP,
``max_steps``) on the on-cluster ILSVRC data and validated against canvit_eval's
frozen-probe baseline.
"""
