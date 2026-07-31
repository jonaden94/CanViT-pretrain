#!/bin/bash
# Allocate (or look up, if it already exists) a Lustre RZG workspace for the
# CanViT WebDataset shards. Run on the login node — workspace tools are not
# available inside container/namespace contexts.
#
# `ws_allocate` is idempotent: if the workspace already exists, it just prints
# the path. No need for a separate ws_find lookup.
#
# Workspace lifetime: 30 days, max 2 extensions to 90 days total
# (see ws_extend.sh). Reminder email fires 7 days before expiry.

set -euo pipefail

WS_NAME=canvit-data
POOL=lustre-rzg
DURATION=30
EMAIL=jonathan.henrich@uni-goettingen.de
GROUPNAME=HPC_nib00021

WS_PATH=$(ws_allocate -F "$POOL" \
    --reminder 7 --mailaddress "$EMAIL" \
    --group --groupname "$GROUPNAME" \
    "$WS_NAME" "$DURATION")

echo
echo "Workspace path: $WS_PATH"
