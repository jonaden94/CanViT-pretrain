#!/bin/bash
# Extend the canvit-data Lustre RZG workspace to the next allowed duration.
#
# Lustre RZG allows 2 extensions, max lifetime 90 days.
# Extension targets are absolute totals: 30 (initial) → 60 → 90.
# This script reads the workspace's "available extensions" counter and picks
# the right target automatically. Run on the login node.

set -euo pipefail

WS_NAME=canvit-data
POOL=lustre-rzg

remaining=$(ws_list -F "$POOL" 2>/dev/null | awk -v name="$WS_NAME" '
    $1 == "id:" { current = $2 }
    /available extensions/ && current == name { print $4; exit }
')

case "$remaining" in
    "")
        echo "ERROR: workspace '$WS_NAME' not found on '$POOL'." >&2
        echo "       Run allocate_workspace.sh first." >&2
        exit 1
        ;;
    2) target=60 ;;
    1) target=90 ;;
    0)
        echo "ERROR: workspace '$WS_NAME' has no extensions remaining." >&2
        exit 1
        ;;
    *)
        echo "ERROR: unexpected 'available extensions' value: $remaining" >&2
        exit 1
        ;;
esac

echo "Extending '$WS_NAME' on '$POOL' to $target days total..."
ws_extend -F "$POOL" "$WS_NAME" "$target"
