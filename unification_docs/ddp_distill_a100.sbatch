#!/bin/bash
# Fuller DDP validation: the REAL distill task (--preset default = the "standard setting"
# that ran on 2x V100 / 4x A100 in the old repo) through the harness entry point, on 2x
# A100 over NCCL. Unlike ddp_smoke_a100.sbatch (isolated sync primitives on a tiny model),
# this exercises the whole path: real with-features shards, shard-split loaders, the
# manual broadcast of normalizer buffers, per-step allreduce_grads, wandb metrics.
#
# Short + no eval on purpose: this validates the TRAINING sync first (losses finite, ranks
# stay in step). A follow-up can add one eval to exercise the rank-0 eval barrier.
# Submit ONLY after the smoke (ddp_smoke_a100.sbatch) passes.
#SBATCH -p grete:shared
#SBATCH -A nib00021
#SBATCH --gpus-per-node=A100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 0-00:20:00
#SBATCH --qos=normal
#SBATCH -J canvit-ddp-distill
#SBATCH -o /user/henrich1/u25995/output/ddp-distill-%J.out

set -uo pipefail
cd /user/henrich1/u25995/jonathan/repos/CanViT-train

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((21000 + (SLURM_JOB_ID % 10000)))
export HF_HOME=/user/henrich1/u25995/.cache/huggingface
export HF_HUB_OFFLINE=1
export NCCL_DEBUG=WARN
WDS="/mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/webdataset-imagenet-21k-with-features"
# distill's create_loaders builds+asserts the val ImageFolder unconditionally, so the val
# paths are required even with eval off (--opts.eval-every 0).
VAL="/user/henrich1/u25995/jonathan/datasets/imagenet1k-val"
VAL_INDEX="/user/henrich1/u25995/jonathan/repos/_data_cache"
VENV=".venv-cu126"

echo "=== node $(hostname)  gpus: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
echo "=== WDS=$WDS"
ls -d "$WDS/train-shuffled" >/dev/null 2>&1 && echo "  train-shuffled/ present" || echo "  WARNING: train-shuffled/ missing"

srun --ntasks=2 \
    env WORLD_SIZE=2 MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
    "$VENV/bin/python" -m canvit_train.harness.run distill --preset default \
        --cfg.webdataset-dir "$WDS" --cfg.val-dir "$VAL" --cfg.val-index-dir "$VAL_INDEX" \
        --cfg.batch-size-per-gpu 8 --cfg.steps-per-job 4096 \
        `# steps_per_job(4096)*batch(8)=32768 = 8*samples_per_shard(4096): shard-aligned` \
        --cfg.log-every 5 --cfg.no-compile \
        --opts.n-steps 40 --opts.eval-every 0 --opts.ckpt-every 0 \
        --cfg.tracker wandb --cfg.wandb-project canvit-train \
        --cfg.run-group ddp-validation --cfg.run-name "ddp-distill-a100-$SLURM_JOB_ID"
echo "=== srun exit: $? ==="
