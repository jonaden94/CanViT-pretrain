#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-00:20:00
#SBATCH -A nib00021
#SBATCH --mem=64G
#SBATCH -o /user/henrich1/u25995/jonathan/repos/CanViT-pretrain/logs/in1k_gate/resume_val-%j.log
#SBATCH -e /user/henrich1/u25995/jonathan/repos/CanViT-pretrain/logs/in1k_gate/resume_val-%j.log
#SBATCH --job-name=in1k-resume

# Validates the NEW in1k resumable shard schedule end-to-end: two harness jobs sharing a
# checkpoint dir. Tiny shard-aligned config: batch=64, steps_per_job=64 (64*64 = 4096 =
# exactly 1 shard/job), max_steps=128 (2 jobs), warmup_steps=8, T=3, val skipped.
#   job 0: FRESH -> runs steps 0..63 on job_index=0's shard slice, writes latest.pt with
#          resume_state{job_index:0}, scheduler at step 64.
#   job 1: RESUME from latest.pt -> resume_start_step derives start_step=64 (=(0+1)*64) and
#          job_index=1, reads the NEXT shard slice, runs steps 64..127.
# PASS = job 0 "FRESH", job 1 "RESUME: start_step=64" + "job_index=1", both "run complete",
# and job 1's shard slice differs from job 0's (different job_index in the loader log).

cd /user/henrich1/u25995/jonathan/repos/CanViT-pretrain
source /user/henrich1/u25995/.bashrc
set -euo pipefail

source slurm_nhr/env.sh
mkdir -p logs/in1k_gate

VENV=.venv-cu126
CKPT=/user/henrich1/u25995/jonathan/repos/CanViT-pretrain/logs/jon_exp22_full_runs/exp22-uniform16/checkpoints/step-1515520-hf
DIR="$TMPDIR/in1k_resume_val"        # shared within the job; node-local is fine (one node)
rm -rf "$DIR"; mkdir -p "$DIR"

COMMON=(--cfg.model-repo "$CKPT" --cfg.mode frozen
        --cfg.max-steps 128 --cfg.steps-per-job 64 --cfg.batch-size 64
        --cfg.warmup-steps 8 --cfg.n-timesteps 3 --cfg.num-workers 1
        --cfg.val-dir /nonexistent --cfg.tracker none
        --opts.resume True --opts.ckpt-dir "$DIR")

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "==================== JOB 0 (fresh) ===================="
"$VENV/bin/python" -m canvit_pretrain.harness.run in1k "${COMMON[@]}"
echo "==================== JOB 1 (resume) ===================="
"$VENV/bin/python" -m canvit_pretrain.harness.run in1k "${COMMON[@]}"
echo "==================== RESUME VAL DONE ===================="
