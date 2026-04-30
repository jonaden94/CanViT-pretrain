#!/bin/bash
# See slurm_nhr/tmux_resubmit/README.md for usage.

REPO="/user/henrich1/u25995/jonathan/repos/CanViT-pretrain"
STATE_FILE="$REPO/slurm_nhr/canvit_train_state"
# SBATCH_SCRIPT="$REPO/slurm_nhr/tmux_resubmit/train_v100_tmux_resubmit.sbatch"
SBATCH_SCRIPT="$REPO/slurm_nhr/tmux_resubmit/train_ddp_v100_tmux_resubmit.sbatch"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Start mode: create state file and exit ─────────────────────────────────
if [ "${1:-}" = "--start" ]; then
    shift
    JOBS=100
    RUN_NAME=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --jobs)      JOBS="$2";     shift 2 ;;
            --run-name)  RUN_NAME="$2"; shift 2 ;;
            *) echo "Unknown arg: $1"; exit 1 ;;
        esac
    done
    # Generate run name now if not provided — ensures consistency from the start
    if [ -z "$RUN_NAME" ]; then
        RUN_NAME="train-$(date '+%Y%m%d-%H%M%S')"
    fi
    LOG_DIR="$REPO/logs/$RUN_NAME"
    mkdir -p "$LOG_DIR"
    cat > "$STATE_FILE" <<EOF
REMAINING_JOBS=$JOBS
RUN_NAME=$RUN_NAME
LOG_DIR=$LOG_DIR
EOF
    log "State written: REMAINING_JOBS=$JOBS RUN_NAME=$RUN_NAME LOG_DIR=$LOG_DIR"
    log "Now start the watcher: bash slurm_nhr/tmux_resubmit/start_watcher.sh"
    exit 0
fi

# ── Resubmit mode ──────────────────────────────────────────────────────────

# Nothing to do if no state file
[ -f "$STATE_FILE" ] || exit 0

source "$STATE_FILE"

# Done
if [ "${REMAINING_JOBS:-0}" -le 0 ]; then
    log "All jobs completed. Removing state file."
    rm "$STATE_FILE"
    exit 0
fi

# Check for FAILED marker
CKPT_DIR="$REPO/checkpoints"
if [ -f "$CKPT_DIR/$RUN_NAME/FAILED" ]; then
    log "FAILED marker found at $CKPT_DIR/$RUN_NAME/FAILED — stopping. Delete marker to retry."
    rm "$STATE_FILE"
    exit 0
fi

# Job still alive (running or pending) — nothing to do
if [ -n "${LAST_JOB_ID:-}" ] && squeue -j "$LAST_JOB_ID" -h 2>/dev/null | grep -q .; then
    exit 0
fi

# Submit next job — always pass --run-name so the sbatch script never needs to infer it
cd "$REPO"
mkdir -p "$LOG_DIR"
SUBMIT_OUT=$(sbatch \
    --export=ALL,LOG_DIR="$LOG_DIR" \
    --output="$LOG_DIR/job-%j.log" \
    --error="$LOG_DIR/job-%j.log" \
    "$SBATCH_SCRIPT" --run-name "$RUN_NAME" 2>&1)
if echo "$SUBMIT_OUT" | grep -q "Submitted batch job"; then
    NEW_JOB_ID=$(echo "$SUBMIT_OUT" | grep -oP '\d+$')
    REMAINING_JOBS=$((REMAINING_JOBS - 1))
    cat > "$STATE_FILE" <<EOF
REMAINING_JOBS=$REMAINING_JOBS
RUN_NAME=$RUN_NAME
LOG_DIR=$LOG_DIR
LAST_JOB_ID=$NEW_JOB_ID
EOF
    log "Submitted job $NEW_JOB_ID (run: $RUN_NAME, remaining: $REMAINING_JOBS)"
else
    log "Submission failed: $SUBMIT_OUT"
fi
