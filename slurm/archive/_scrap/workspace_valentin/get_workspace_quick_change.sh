#!/bin/bash
set -euo pipefail

# 1. Environment Setup
set +u
source ~/.bashrc
set -u

export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'

# 2. Configuration
NEW_WS_NAME="gpupool_v6"           # Bumped to v6!
DURATION=30
EMAIL="valentin.hassler@uni-goettingen.de"
POOL="lustre-rzg"
DESTS=(
  "/mnt/lustre-grete/usr/u13879/datasets"
  "/mnt/lustre-grete/usr/u13879/curiosity/autoregressive_detection/data"
)

# 3. Allocate New Workspace
echo "Allocating new workspace ($NEW_WS_NAME)..."
export WS_DIR=$(ws_allocate -F "$POOL" \
                     --reminder 7 --mailaddress "$EMAIL" \
                     "$NEW_WS_NAME" "$DURATION")

echo "Successfully allocated new workspace at: $WS_DIR"

# 4. Re-populate from Master Source
echo "Populating workspace from master source. Your running training is entirely unaffected..."
echo "Starting data extraction at $(date)..."

# Explicitly tell Python where your custom library lives
export PYTHONPATH="/mnt/lustre-grete/usr/u13879/exploration/lib:${PYTHONPATH:-}"

# Use the BASE environment where boto3 and your custom libs are installed
conda run --no-capture-output -n curiosity python /mnt/lustre-grete/usr/u13879/exploration/lib/utilib/datasets/put_on_workspace.py

echo "Finished master source extraction at $(date)."

# 5. The Atomic Hot-Swap
echo "Data is ready! Swapping symlinks..."
for DEST_PARENT in "${DESTS[@]}"; do
  ln -sfn "$WS_DIR/ImageNet" "$DEST_PARENT/ImageNet"
  echo "Successfully swapped: $DEST_PARENT/ImageNet -> $WS_DIR/ImageNet"
done

echo "Workspace renewal fully complete! Your training is now reading from the fresh copy."