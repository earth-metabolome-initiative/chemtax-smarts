#!/bin/bash
# Show the queue state of chemtax-smarts shard jobs.
#
#   status.sh                 all chemtax jobs, once
#   status.sh classyfire      one dataset, once
#   status.sh classyfire 60   one dataset, refreshed every 60 seconds

set -euo pipefail

DATASET="${1:-}"
INTERVAL="${2:-0}"
NAME="chemtax${DATASET:+-$DATASET}"
FORMAT="%.18i %.14j %.9P %.8q %.8T %.11M %.6D %R"

show() {
    squeue -u "$USER" -n "$NAME" -o "$FORMAT"
}

if [ "$INTERVAL" -gt 0 ] 2>/dev/null; then
    watch -n "$INTERVAL" squeue -u "$USER" -n "$NAME" -o "$FORMAT"
else
    show
fi
