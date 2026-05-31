#!/bin/bash
#SBATCH --job-name=chemtax-shard
#SBATCH --account=xxxxxxxxxxxxxxxx
#SBATCH --partition=lr6
#SBATCH --qos=lr_normal
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --requeue
#SBATCH --time=24:00:00
#SBATCH --output=/global/scratch/users/%u/chemtax-smarts/logs/worker_%A_%a.out
#SBATCH --error=/global/scratch/users/%u/chemtax-smarts/logs/worker_%A_%a.err

# One array task evolves one shard of the per-label task plan. The global shard
# index is the array task id plus the batch offset passed by submit.sh. Each
# shard writes to its own output directory and resumes from its own
# results.jsonl, so a requeued or restarted shard picks up where it stopped.

set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: sbatch slurm/lrc/array_job.sh <SHARD_OFFSET> <SHARD_COUNT> <OUTPUT_BASE> <RUN_ARGS...>"
    exit 1
fi

SHARD_OFFSET="$1"
SHARD_COUNT="$2"
OUTPUT_BASE="$3"
shift 3

SHARD_INDEX=$((SLURM_ARRAY_TASK_ID + SHARD_OFFSET))
SHARD_DIR="$OUTPUT_BASE/shards/$(printf 'shard-%05d' "$SHARD_INDEX")"
REPO_DIR="${CHEMTAX_REPO_DIR:-$HOME/chemtax-smarts}"
BINARY="target/release/chemtax-smarts"

clean_rust_compiler_environment() {
    unset CC CXX AR CFLAGS CXXFLAGS LDFLAGS
    unset CC_x86_64_unknown_linux_gnu CXX_x86_64_unknown_linux_gnu AR_x86_64_unknown_linux_gnu
    unset CFLAGS_x86_64_unknown_linux_gnu CXXFLAGS_x86_64_unknown_linux_gnu
}

load_user_cargo_environment() {
    if [ -f "$HOME/.cargo/env" ]; then
        # shellcheck source=/dev/null
        . "$HOME/.cargo/env"
    fi
}

clean_rust_compiler_environment
load_user_cargo_environment
cd "$REPO_DIR"

if [ ! -x "$BINARY" ]; then
    echo "ERROR: $BINARY is missing. Run bash slurm/lrc/setup_env.sh first."
    exit 1
fi

export RAYON_NUM_THREADS="${SLURM_CPUS_ON_NODE:-$(nproc)}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"

mkdir -p "$SHARD_DIR"

echo "Host:        $(hostname)"
echo "Start:       $(date)"
echo "Shard:       $SHARD_INDEX / $SHARD_COUNT"
echo "Output dir:  $SHARD_DIR"
echo "Rayon CPUs:  $RAYON_NUM_THREADS"
echo "Command:     $BINARY $* --shard-index $SHARD_INDEX --shard-count $SHARD_COUNT --output-dir $SHARD_DIR"

"$BINARY" "$@" \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$SHARD_COUNT" \
    --output-dir "$SHARD_DIR"

echo "Done:        $(date)"
