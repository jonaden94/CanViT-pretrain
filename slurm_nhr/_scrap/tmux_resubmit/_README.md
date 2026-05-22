# Tmux-based job resubmitter for V100 training

Automates sequential SLURM job submission on the jupyter partition, which limits
users to 1 submitted job at a time. Instead of job arrays, jobs are submitted one
at a time by a persistent watcher running in a tmux session.

## Files

| File | Purpose |
|------|---------|
| `check_and_submit.sh` | Controller: manages state, checks queue, submits next job |
| `start_watcher.sh` | Watcher: calls `check_and_submit.sh` every 60s in a loop |
| `train_v100_tmux_resubmit.sbatch` | Single V100 job script |
| `train_ddp_v100_tmux_resubmit.sbatch` | 2× V100 DDP job script |

## Usage

**1. Initialize a new training run:**
```bash
bash slurm_nhr/tmux_resubmit/check_and_submit.sh --start --jobs 10
```
This creates a state file (`slurm_nhr/canvit_train_state`) with the run name
and job count. No jobs are submitted yet.

**2. Start the watcher in a tmux session:**
```bash
tmux new -s canvit
bash slurm_nhr/tmux_resubmit/start_watcher.sh
# Detach: Ctrl+B D
# Reattach: tmux attach -t canvit
```

The watcher submits the first job within 60 seconds, then resubmits after each job
finishes until the counter reaches 0.

**3. Resume an existing run** (only needed if the watcher died, you ran out of jobs, or the server rebooted):
```bash
bash slurm_nhr/tmux_resubmit/check_and_submit.sh --start --jobs 5 --run-name train-XXXXXXX
```
This only recreates the state file — you still need to start (or restart) the watcher in a tmux session afterwards (step 2).

## State file

`slurm_nhr/canvit_train_state` stores:
```
REMAINING_JOBS=10
RUN_NAME=train-20260429-XXXXXX
```

Delete this file to stop the watcher (it exits cleanly on the next tick).

## Switching between single and DDP

Edit `SBATCH_SCRIPT` in `check_and_submit.sh` to point at the desired script:
```bash
SBATCH_SCRIPT="$REPO/slurm_nhr/tmux_resubmit/train_v100_tmux_resubmit.sbatch"       # single GPU
SBATCH_SCRIPT="$REPO/slurm_nhr/tmux_resubmit/train_ddp_v100_tmux_resubmit.sbatch"   # 2× GPU DDP
```
