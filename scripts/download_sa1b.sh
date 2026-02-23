#!/bin/bash
# Download SA-1B tarballs from Meta CDN to $SA1B_TAR_DIR.
#
# - Idempotent: skips tars that already exist at the expected size
# - Resumable: wget -c continues partial downloads (.tmp suffix until complete)
# - Safe to ctrl-C at any time: partial .tmp files will be resumed on next run
#
# Each tar is ~10.5 GB (gzipped, despite .tar extension), containing ~11k JPEGs + masks.
# At ~50 MB/s on Nibi login nodes, one tar takes ~3.5 min.
#
# Usage (from repo root, after `source slurm/env.sh`):
#   bash scripts/download_sa1b.sh              # all 1000 tars
#   bash scripts/download_sa1b.sh 3            # first 3 tars only

set -euo pipefail

TAR_DIR="${SA1B_TAR_DIR:?SA1B_TAR_DIR not set — source slurm/env.sh}"
LINKS="${SA1B_LINKS:?SA1B_LINKS not set — source slurm/env.sh}"
LIMIT="${1:-0}"

echo "=== SA-1B Download ==="
echo "TAR_DIR: $TAR_DIR"
echo "LINKS:   $LINKS"
echo "LIMIT:   ${LIMIT} (0 = all)"
echo "======================"

[ -f "$LINKS" ] || { echo "FATAL: links file not found: $LINKS" >&2; exit 1; }

mkdir -p "$TAR_DIR"

DOWNLOADED=0
SKIPPED=0

tail -n +2 "$LINKS" | while IFS=$'\t' read -r filename url; do
    if [ "$LIMIT" -gt 0 ] && [ "$((DOWNLOADED + SKIPPED))" -ge "$LIMIT" ]; then
        break
    fi

    dest="$TAR_DIR/$filename"
    tmp="$dest.tmp"

    if [ -f "$dest" ]; then
        echo "[skip] $filename"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "[download] $filename"
    wget -c -q --show-progress -O "$tmp" "$url"
    mv "$tmp" "$dest"
    DOWNLOADED=$((DOWNLOADED + 1))
    echo "[done] $filename ($(du -h "$dest" | cut -f1))"
done

echo "=== Complete: $DOWNLOADED downloaded, $SKIPPED skipped ==="
