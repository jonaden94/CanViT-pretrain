#!/bin/bash
# Extend the canvit-data Lustre RZG workspace by another full allocation period.
#
# IMPORTANT: the `days` argument to ws_extend is the NEW remaining time (counted
# from now), NOT a cumulative total. It is capped at the filesystem's per-
# allocation Time Limit, which is 30 days for lustre-rzg. So each extension just
# resets the remaining time back up to 30 days.
#
# Lustre RZG allows 2 extensions; max lifetime is therefore ~90 days
# (30 initial + 2 × 30). This script extends to the 30-day max, after first
# checking that an extension is still available. Run on the login node.

set -euo pipefail

WS_NAME=canvit-data
POOL=lustre-rzg
TIME_LIMIT=30

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
    0)
        echo "ERROR: workspace '$WS_NAME' has no extensions remaining." >&2
        exit 1
        ;;
esac

echo "Extending '$WS_NAME' on '$POOL' to $TIME_LIMIT days (extensions left: $remaining)..."
ws_extend -F "$POOL" "$WS_NAME" "$TIME_LIMIT"
