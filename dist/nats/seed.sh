#!/bin/bash
# Seed the NATS work queue with the remaining labels for one dataset, then exit.
# Run once before starting the workers. Pass extra binary args after the URL, for
# example --resume-from <merged-log> to skip labels already finished elsewhere.
#
#   seed.sh <npclassifier|classyfire> <nats-url> [extra binary args...]

set -euo pipefail

DATASET="${1:?Usage: seed.sh <npclassifier|classyfire> <nats-url> [extra args...]}"
NATS_URL="${2:?Usage: seed.sh <dataset> <nats-url> [extra args...]}"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${CHEMTAX_REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_DIR="${CHEMTAX_DATA_DIR:-$REPO_DIR/data/$DATASET}"
BINARY="$REPO_DIR/target/release/chemtax-smarts"

if [ ! -x "$BINARY" ]; then
    echo "ERROR: $BINARY missing. Build with: cargo build --release --features nats" >&2
    exit 1
fi

"$BINARY" \
    --dataset "$DATASET" \
    --data-dir "$DATA_DIR" \
    --nats-url "$NATS_URL" \
    --seed-queue \
    "$@"
