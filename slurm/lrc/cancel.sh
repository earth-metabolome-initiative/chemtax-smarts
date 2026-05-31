#!/bin/bash
# Cancel chemtax-smarts shard jobs.
#
#   cancel.sh              cancel every chemtax-* job
#   cancel.sh classyfire   cancel one dataset's jobs

set -euo pipefail

DATASET="${1:-}"
NAME="chemtax${DATASET:+-$DATASET}"

echo "Cancelling jobs named $NAME for $USER"
scancel -u "$USER" -n "$NAME"
