#!/bin/bash
# Download one dataset into the shared data directory before launching the shard
# array, so the workers all skip the download instead of racing Zenodo. Uses the
# binary's --download-only path: it fetches the files and exits without loading
# splits or evolving labels.

set -euo pipefail

DATASET="${1:?Usage: prefetch.sh <npclassifier|classyfire>}"

REPO_DIR="${CHEMTAX_REPO_DIR:-$HOME/chemtax-smarts}"
SCRATCH_ROOT="${CHEMTAX_SCRATCH_ROOT:-/global/scratch/users/$USER/chemtax-smarts}"
DATA_DIR="${CHEMTAX_DATA_DIR:-$SCRATCH_ROOT/data/$DATASET}"

load_user_cargo_environment() {
    if [ -f "$HOME/.cargo/env" ]; then
        # shellcheck source=/dev/null
        . "$HOME/.cargo/env"
    fi
}

load_user_cargo_environment
cd "$REPO_DIR"

BINARY="target/release/chemtax-smarts"
if [ ! -x "$BINARY" ]; then
    echo "ERROR: $BINARY is missing. Run bash slurm/lrc/setup_env.sh first."
    exit 1
fi

echo "Prefetching $DATASET into $DATA_DIR"
"$BINARY" \
    --dataset "$DATASET" \
    --data-dir "$DATA_DIR" \
    --output-dir "$SCRATCH_ROOT/results/$DATASET/_prefetch" \
    --download-only

echo "Done. Data dir: $DATA_DIR"
