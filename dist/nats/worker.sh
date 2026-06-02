#!/bin/bash
# Run a chemtax-smarts worker: pull labels from the NATS queue, evolve each, and
# append to this machine's own results.jsonl. Run one per workstation. The first
# run downloads this machine's own dataset copy if it is not already present.
#
#   worker.sh <npclassifier|classyfire> <nats-url> [extra binary args...]

set -euo pipefail

DATASET="${1:?Usage: worker.sh <npclassifier|classyfire> <nats-url> [extra args...]}"
NATS_URL="${2:?Usage: worker.sh <dataset> <nats-url> [extra args...]}"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${CHEMTAX_REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_DIR="${CHEMTAX_DATA_DIR:-$REPO_DIR/data/$DATASET}"
OUTPUT_DIR="${CHEMTAX_OUTPUT_DIR:-$REPO_DIR/results/$DATASET}"
BINARY="$REPO_DIR/target/release/chemtax-smarts"

if [ ! -x "$BINARY" ]; then
    echo "ERROR: $BINARY missing. Build with: cargo build --release --features nats" >&2
    exit 1
fi

export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-$(nproc)}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"

echo "Host:    $(hostname)"
echo "Dataset: $DATASET"
echo "Queue:   $NATS_URL"
echo "Output:  $OUTPUT_DIR"
echo "Rayon:   $RAYON_NUM_THREADS threads"

"$BINARY" \
    --dataset "$DATASET" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --nats-url "$NATS_URL" \
    "$@"
