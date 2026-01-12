"""Measure IN1k val accuracy at different CLS feature storage dtypes.

Single teacher inference pass, then compare probe accuracy when CLS
features are cast to different dtypes (simulating storage precision loss).

Dtypes tested: fp32, fp16 (our shard storage), bf16, fp8.

Usage (on cluster, after `source slurm/env.sh`):
    uv run -m scripts.dtype_accuracy_eval \
        --val-dir $IN1K_VAL_DIR \
        --teacher-ckpt $DINOV3_VITB16_CKPT

Results (2026-01-12, commit 747f2eb, H100, dinov3_vitb16, 512x512, IN1k val 50k):
    fp32 : 84.964%  (Δ = +0.000%)
    fp16 : 84.964%  (Δ = +0.000%)  <- our shard storage dtype
    bf16 : 84.968%  (Δ = +0.004%)
    fp8  : 84.962%  (Δ = -0.002%)

Conclusion: No measurable accuracy loss from fp16 storage. All dtypes equivalent.
"""
