#!/bin/bash
# Definitive throughput experiment: harness-sync vs harness-async(fixed) vs old loop,
# ALL on ONE full A100, INTERLEAVED, repeated.
#
# Why a job rather than the interactive node: the interactive allocation is a `3g.40gb`
# MIG slice (42 of 108 SMs). The effect under test is a fixed CPU-side stall, which is
# hidden better when compute is slower, so MIG systematically UNDERSTATES it. Production
# ran on full cards; the number that matters must come from one.
#
# Why one job for all arms: every arm then runs on the SAME physical card in the same
# allocation. The retrospective log analysis had to correct for an A100 40GB/80GB mix
# after the fact; here that confound cannot arise.
#
# Why interleaved (sync, async, old, sync, async, old, ...): any drift over the job —
# thermal, a noisy neighbour, filesystem weather — is shared by all arms rather than
# landing on whichever ran last.
#
# Each arm is a separate PROCESS, so each gets a fresh CUDA context and no in-process
# state (compile cache, allocator, teacher) can leak between arms.
#
# Submit:  sbatch slurm_nhr/runs/perf/throughput_matrix.sh

#SBATCH --job-name=throughput_matrix
#SBATCH --partition=grete:shared
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/perf/throughput_matrix-%j.log
#SBATCH --error=logs/perf/throughput_matrix-%j.log

set -euo pipefail

REPO=/user/henrich1/u25995/jonathan/repos/CanViT-train
cd "$REPO"

# NB: source the preamble BEFORE `set -u` bites on unset vars in .bashrc
set +u
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
[ -f "$REPO/.envrc.grete" ] && source "$REPO/.envrc.grete"
set -u

export HF_HUB_OFFLINE=1

# --- COMMIT PINNING (same mechanism as base_train.sbatch) --------------------
# Without this the job reads the LIVE working tree, so an unrelated edit while it is
# queued or running silently changes what is measured. `git archive` reads the local
# object store only -- no network, no SSH, does not touch HEAD or the working tree.
# NOTE the scripts are also taken from the SNAPSHOT ($SRC), not from $REPO: this job
# invokes them by path, and pinning the importable package while running a live script
# would only half-pin the run.
_REPO_BASE=/user/henrich1/u25995/jonathan/repos
SRC="${TMPDIR:?TMPDIR must be set}/canvit_src"
: "${PRETRAIN_COMMIT:?set PRETRAIN_COMMIT}"
: "${PYTORCH_COMMIT:?set PYTORCH_COMMIT}"
: "${FOVI_COMMIT:?set FOVI_COMMIT}"
for pair in "CanViT-train:$PRETRAIN_COMMIT" "CanViT-PyTorch:$PYTORCH_COMMIT" "fovi:$FOVI_COMMIT"; do
    name="${pair%%:*}"; commit="${pair##*:}"
    mkdir -p "$SRC/$name"
    git -C "$_REPO_BASE/$name" archive "$commit" | tar -x -C "$SRC/$name"
    export PYTHONPATH="$SRC/$name${PYTHONPATH:+:$PYTHONPATH}"
    echo "Pinned $name -> $commit"
done
export PYTHONSAFEPATH=1   # the snapshot must win over cwd and the editable install
cd "$SRC/CanViT-train"

PY="$REPO/.venv-cu126/bin/python"
STEPS=${STEPS:-100}
WARMUP=${WARMUP:-30}
REPS=${REPS:-4}

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch;p=torch.cuda.get_device_properties(0);print(f'SMs={p.multi_processor_count} (full A100=108)')"
echo "steps=$STEPS warmup=$WARMUP reps=$REPS"
echo "code: $SRC/CanViT-train (pinned $PRETRAIN_COMMIT)"

declare -A RESULTS
for arm in sync async old; do RESULTS[$arm]=""; done

for rep in $(seq 1 "$REPS"); do
    echo ""
    echo "--- rep $rep/$REPS ---"
    for arm in sync async old; do
        case "$arm" in
            sync|async)
                out=$($PY $SRC/CanViT-train/unification_docs/throughput_ab.py --arm "$arm" \
                        --steps "$STEPS" --warmup "$WARMUP" 2>&1 | tail -40 || true) ;;
            old)
                out=$($PY $SRC/CanViT-train/unification_docs/throughput_oldloop.py --once \
                        --steps $((STEPS + WARMUP)) 2>&1 | tail -40 || true) ;;
        esac
        # `grep` exits 1 when the arm produced no MEDIAN_MS, and under `set -e` +
        # pipefail that killed the JOB before the diagnostic branch below could run —
        # which is exactly how rep 1 of job 15089880 was lost. Never let extraction fail.
        ms=$(echo "$out" | grep -oE "^MEDIAN_MS [0-9.]+" | awk '{print $2}' | tail -1 || true)
        if [ -z "$ms" ]; then
            echo "  [$arm] FAILED — no MEDIAN_MS. tail:"
            echo "$out" | tail -30 | sed 's/^/      /'
            continue
        fi
        printf "  %-6s %8s ms/step\n" "$arm" "$ms"
        RESULTS[$arm]="${RESULTS[$arm]} $ms"
    done
done

echo ""
echo "======================================================================"
echo "THROUGHPUT MATRIX (median ms/step per rep, full A100, interleaved)"
for arm in sync async old; do
    printf "  %-24s %s\n" "$arm" "${RESULTS[$arm]}"
done
export R_sync="${RESULTS[sync]}" R_async="${RESULTS[async]}" R_old="${RESULTS[old]}"
$PY - <<'EOF'
import os, statistics
res = {a: [float(x) for x in os.environ.get(f"R_{a}", "").split()] for a in ("sync","async","old")}
for a, v in res.items():
    if v:
        print(f"  {a:6s} mean {statistics.mean(v):8.1f} ms"
              + (f"  sd {statistics.stdev(v):5.1f}" if len(v) > 1 else ""))
if res["sync"] and res["async"]:
    # PAIRED per-rep ratios. A ratio of means is wrong here: grete:shared shares the
    # NODE, so a co-tenant inflates whichever reps overlap it (job 15091113 saw
    # sync 487/489/284/487 on one card). Within a rep the arms run back to back, so
    # the ratio survives contention that the absolute numbers do not.
    per = [(y/s-1)*100 for s, y in zip(res["sync"], res["async"])]
    print(f"\n  fix, paired per rep:        {' '.join(f'{d:+.1f}%' for d in per)}")
    print(f"  median of those:            {statistics.median(per):+.1f}%")
    print(f"  quietest rep (min abs ms):  "
          f"{(min(res['async'])/min(res['sync'])-1)*100:+.1f}%   <- least-contended estimate")
if res["old"] and res["async"]:
    o, y = statistics.mean(res["old"]), statistics.mean(res["async"])
    print(f"  fixed harness vs OLD LOOP:  {(y/o-1)*100:+.1f}%   <- 0% means fully explained")
if res["old"] and res["sync"]:
    o, s = statistics.mean(res["old"]), statistics.mean(res["sync"])
    print(f"  pre-fix harness vs OLD LOOP:{(s/o-1)*100:+.1f}%   <- production reported ~+10%")
EOF
