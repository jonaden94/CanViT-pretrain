#!/bin/bash
set -euo pipefail  # Moved to the top for maximum safety!

# --- Environment & Network Setup ---
export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'
export PYTHONPATH="/mnt/lustre-grete/usr/u13879/exploration/lib:${PYTHONPATH:-}"

# 1. Set some handy variables
WS_NAME=gpupool             # anything ASCII: letters, numbers, _ . -
DURATION=30                 # first chunk (max allowed on lustre‑rzg/ceph‑ssd)
EMAIL=valentin.hassler@uni-goettingen.de      # for reminder
POOL=lustre-rzg             # or ceph-ssd if you prefer

# 2. Allocate
WS_DIR=$(ws_allocate -F "$POOL" \
                     --reminder 7 --mailaddress "$EMAIL" \
                     "$WS_NAME" "$DURATION")

echo "Workspace lives at: $WS_DIR"

# 3. Data Extraction (Forced into 'curiosity' environment)
echo "Starting Python data extraction..."
conda run --no-capture-output -n curiosity python /mnt/lustre-grete/usr/u13879/exploration/lib/utilib/datasets/put_on_workspace.py
echo "Data extraction complete!"

# 4. Set up Symlinks
# Source directory
SRC="/mnt/lustre-rzg/workspaces/ws/nim00018/u13879-gpupool/ImageNet"

# Destination parent directories
DESTS=(
  "/mnt/lustre-grete/usr/u13879/datasets"
  "/mnt/lustre-grete/usr/u13879/curiosity/autoregressive_detection/data"
  "/mnt/vast-nhr/projects/nim00018/datasets"
)

for DEST_PARENT in "${DESTS[@]}"; do
  # Make sure the target parent folder exists
  mkdir -p "$DEST_PARENT"

  # Create or update the symlink
  ln -sfn "$SRC" "$DEST_PARENT/ImageNet"

  echo "Linked: $DEST_PARENT/ImageNet → $SRC"
done

# ======================================================================
# Extensions (Run manually when needed!)
# ======================================================================
# # Still on the Lustre workspace we created earlier
# POOL=lustre-rzg
# WS_NAME=gpupool

# # --- First extension: bump from 30 → 60 days total ---
# ws_extend -F "$POOL" "$WS_NAME" 30

# # # --- Second (and final) extension, later on: 60 → 90 days total ---
# # # Run this while at ~55‑59 days remaining
# # ws_extend -F "$POOL" "$WS_NAME" 90