#!/bin/bash
#SBATCH -p grete:interactive
#SBATCH -G 2g.10gb # 1g.20gb, 2g.10gb
#SBATCH -c 4
#SBATCH -t 0-12:00:00
#SBATCH -A nib00021
#SBATCH -o slurm_nhr/_workspace/sync-%J.log
#SBATCH --mem=128G

# Copy the WebDataset (~2.1 TB, 326 shards) from project NFS to the
# Lustre RZG workspace. Workspace must already be allocated via
# allocate_workspace.sh — this script only resolves its path.
#
# Submit with:
#   sbatch slurm_nhr/_workspace/sync_to_workspace.sbatch

set -euo pipefail
log() { echo "[$(date '+%H:%M:%S')] $*"; }

WS_NAME=canvit-data
POOL=lustre-rzg
SRC=/mnt/vast-nhr/projects/nib00021/jonathan/datasets/webdataset-imagenet-1k
DEST_NAME=webdataset-imagenet-1k

log "=== Sync WebDataset to Lustre RZG workspace ==="
log "Host: $(hostname)"

WS_PATH=$(ws_list -F "$POOL" 2>/dev/null | awk -v name="$WS_NAME" '
    $1 == "id:" { current = $2 }
    /workspace directory/ && current == name { print $4; exit }
')

if [ -z "$WS_PATH" ]; then
    log "ERROR: workspace '$WS_NAME' not found on '$POOL'."
    log "       Run slurm_nhr/_workspace/allocate_workspace.sh first."
    exit 1
fi

DEST="$WS_PATH/$DEST_NAME"
log "Source:      $SRC"
log "Destination: $DEST"

mkdir -p "$DEST"

log "Starting rsync..."
rsync -a --info=progress2 --stats "$SRC/" "$DEST/"
log "rsync finished."

log "Verifying file counts..."
src_count=$(find "$SRC" -type f | wc -l)
dst_count=$(find "$DEST" -type f | wc -l)
log "  source: $src_count files"
log "  dest:   $dst_count files"
if [ "$src_count" -ne "$dst_count" ]; then
    log "ERROR: file count mismatch (src=$src_count, dst=$dst_count)"
    exit 1
fi

log "Verifying total sizes (bytes, exact)..."
src_size=$(du -sb "$SRC" | awk '{print $1}')
dst_size=$(du -sb "$DEST" | awk '{print $1}')
log "  source: $src_size"
log "  dest:   $dst_size"
if [ "$src_size" -ne "$dst_size" ]; then
    log "WARN: byte-size mismatch (src=$src_size, dst=$dst_size). Investigate."
fi

log "Done. Update WEBDATASET_DIR in .envrc.grete to: $DEST"
