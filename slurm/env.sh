# AVP-ViT environment setup for Nibi
# Source this: source slurm/env.sh

echo "[env] Setting up environment..."

export PATH=$HOME/.local/bin:$PATH
module load java/17.0.6 2>/dev/null && echo "[env] Loaded java/17.0.6" || true

# Fast local storage if in SLURM job
if [ -n "$SLURM_TMPDIR" ]; then
    export UV_CACHE_DIR="$SLURM_TMPDIR/.uv-cache"
    export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv"
    echo "[env] Using SLURM_TMPDIR for uv cache/venv"
else
    echo "[env] No SLURM_TMPDIR (interactive session)"
fi

# Caches on scratch (persistent)
export HF_HOME="$SCRATCH/.huggingface"
export TORCH_HOME="$SCRATCH/.torch"
export TORCH_COMPILE_CACHE_DIR="$SCRATCH/.torch_compile_cache"

# Nibi paths (single source of truth)
export AVP_TEACHER_CKPT=~/projects/def-skrishna/checkpoints/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
export AVP_TRAIN_DIR=/datashare/imagenet/winter21_whole
export AVP_VAL_DIR=/datashare/imagenet/ILSVRC2012/val
export AVP_INDEX_DIR="$SCRATCH/in21k_index"
export AVP_CKPT_DIR="$SCRATCH/avp_checkpoints"

# Comet
if [ -f ~/comet_api_key.txt ]; then
    export COMET_API_KEY=$(cat ~/comet_api_key.txt)
    echo "[env] Loaded COMET_API_KEY"
else
    echo "[env] No comet_api_key.txt (optional)"
fi

echo "[env] Done"

# Check critical paths (call with: check_critical || exit 1)
check_critical() {
    local fail=0
    [ ! -f "$AVP_TEACHER_CKPT" ] && echo "[check] FATAL: Teacher checkpoint not found: $AVP_TEACHER_CKPT" && fail=1
    [ ! -d "$AVP_TRAIN_DIR" ] && echo "[check] FATAL: Train dir not found: $AVP_TRAIN_DIR" && fail=1
    [ $fail -eq 1 ] && return 1
    echo "[check] Critical paths OK"
    return 0
}
