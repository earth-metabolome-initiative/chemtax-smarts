#!/bin/bash
# Submit chemtax-smarts shard arrays on Lawrencium.

set -euo pipefail

REPO_DIR="${CHEMTAX_REPO_DIR:-$HOME/chemtax-smarts}"
SCRATCH_ROOT="${CHEMTAX_SCRATCH_ROOT:-/global/scratch/users/$USER/chemtax-smarts}"
LOGS_DIR="$SCRATCH_ROOT/logs"
MAX_ARRAY_SIZE=1000

usage() {
    cat <<'USAGE'
Usage:
  bash slurm/lrc/submit.sh <npclassifier|classyfire> [OPTIONS]

Splits the dataset's per-label task plan into SHARDS shards and submits one
array task per shard. Each shard evolves a disjoint, size-balanced slice and
writes to its own directory under <output-dir>/shards/.

Options:
  --shards=N          Total shards the task plan is split into
                      (default: npclassifier 64, classyfire 512)
  --account=NAME      SLURM account / allocation (default: $CHEMTAX_LRC_ACCOUNT
                      or xxxxxxxxxxxxxxxx, set this to your allocation)
  --partition=PART    SLURM partition (default: lr6)
  --qos=QOS           SLURM QoS (default: lr_normal)
  --time=HH:MM:SS     Wall time per shard (default: 24:00:00)
  --concurrency=N     Max concurrent array tasks (default: 64)
  --offset=N          First zero-based shard index to submit (default: 0)
  --n-shards=N        How many shard indexes to submit (default: SHARDS - offset)
  --data-dir=PATH     Dataset cache directory
                      (default: $SCRATCH_ROOT/data/<dataset>)
  --output-dir=PATH   Output base directory
                      (default: $SCRATCH_ROOT/results/<dataset>)
  --resume-from=PATH  A prior results.jsonl whose labels every shard skips
                      without copying them (for example a local run's log).
                      Must use the same dataset vocabulary.
  --debug             One lr_debug shard for a quick end-to-end check
  --dry-run           Print sbatch commands without submitting
USAGE
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

DATASET="$1"
shift

ACCOUNT="${CHEMTAX_LRC_ACCOUNT:-xxxxxxxxxxxxxxxx}"
PARTITION="lr6"
QOS="lr_normal"
TIME="24:00:00"
CONCURRENCY=64
OFFSET=0
N_SHARDS=""
SHARDS=""
DATA_DIR=""
OUTPUT_DIR=""
RESUME_FROM=""
DEBUG=false
DRY_RUN=false

case "$DATASET" in
    npclassifier) DEFAULT_SHARDS=64 ;;
    classyfire)   DEFAULT_SHARDS=512 ;;
    -h|--help)    usage; exit 0 ;;
    *)            echo "Unknown dataset: $DATASET"; usage; exit 1 ;;
esac

for arg in "$@"; do
    case "$arg" in
        --shards=*)       SHARDS="${arg#*=}" ;;
        --account=*)      ACCOUNT="${arg#*=}" ;;
        --partition=*)    PARTITION="${arg#*=}" ;;
        --qos=*)          QOS="${arg#*=}" ;;
        --time=*)         TIME="${arg#*=}" ;;
        --concurrency=*)  CONCURRENCY="${arg#*=}" ;;
        --offset=*)       OFFSET="${arg#*=}" ;;
        --n-shards=*)     N_SHARDS="${arg#*=}" ;;
        --data-dir=*)     DATA_DIR="${arg#*=}" ;;
        --output-dir=*)   OUTPUT_DIR="${arg#*=}" ;;
        --resume-from=*)  RESUME_FROM="${arg#*=}" ;;
        --debug)          DEBUG=true ;;
        --dry-run)        DRY_RUN=true ;;
        -h|--help)        usage; exit 0 ;;
        *)                echo "Unknown option: $arg"; usage; exit 1 ;;
    esac
done

[ -n "$SHARDS" ] || SHARDS="$DEFAULT_SHARDS"
[ -n "$DATA_DIR" ] || DATA_DIR="$SCRATCH_ROOT/data/$DATASET"
[ -n "$OUTPUT_DIR" ] || OUTPUT_DIR="$SCRATCH_ROOT/results/$DATASET"

if [ "$DEBUG" = true ]; then
    PARTITION="lr4"
    QOS="lr_debug"
    TIME="00:30:00"
    CONCURRENCY=1
    N_SHARDS=1
fi

[ -n "$N_SHARDS" ] || N_SHARDS=$((SHARDS - OFFSET))
if [ "$N_SHARDS" -le 0 ]; then
    echo "No shards to submit."
    exit 0
fi
if [ $((OFFSET + N_SHARDS)) -gt "$SHARDS" ]; then
    echo "ERROR: offset + n-shards exceeds $SHARDS total shards."
    exit 1
fi

mkdir -p "$LOGS_DIR"
cd "$REPO_DIR"

RUN_ARGS=(
    --dataset "$DATASET"
    --data-dir "$DATA_DIR"
)
if [ -n "$RESUME_FROM" ]; then
    RUN_ARGS+=(--resume-from "$RESUME_FROM")
fi

JOB_NAME="chemtax-$DATASET"

echo "=== Lawrencium shard submission ==="
echo "Dataset:      $DATASET"
echo "Data dir:     $DATA_DIR"
echo "Output dir:   $OUTPUT_DIR"
[ -n "$RESUME_FROM" ] && echo "Resume from:  $RESUME_FROM"
echo "Shard range:  $OFFSET..$((OFFSET + N_SHARDS - 1)) / $SHARDS"
echo "Account:      $ACCOUNT"
echo "Partition:    $PARTITION"
echo "QoS:          $QOS"
echo "Time:         $TIME"
echo "Concurrency:  $CONCURRENCY"
echo "Logs:         $LOGS_DIR"
echo ""

SUBMITTED=0
while [ "$SUBMITTED" -lt "$N_SHARDS" ]; do
    BATCH_SIZE=$((N_SHARDS - SUBMITTED))
    if [ "$BATCH_SIZE" -gt "$MAX_ARRAY_SIZE" ]; then
        BATCH_SIZE="$MAX_ARRAY_SIZE"
    fi
    BATCH_MAX=$((BATCH_SIZE - 1))
    BATCH_OFFSET=$((OFFSET + SUBMITTED))

    SBATCH_CMD=(
        sbatch
        --account="$ACCOUNT"
        --partition="$PARTITION"
        --qos="$QOS"
        --time="$TIME"
        --job-name="$JOB_NAME"
        --array="0-${BATCH_MAX}%${CONCURRENCY}"
        --output="$LOGS_DIR/worker_${DATASET}_%A_%a.out"
        --error="$LOGS_DIR/worker_${DATASET}_%A_%a.err"
        slurm/lrc/array_job.sh
        "$BATCH_OFFSET" "$SHARDS" "$OUTPUT_DIR"
    )
    SBATCH_CMD+=("${RUN_ARGS[@]}")

    if [ "$DRY_RUN" = true ]; then
        printf '[DRY RUN] '
        printf '%q ' "${SBATCH_CMD[@]}"
        printf '\n'
    else
        echo "Submitting batch offset=$BATCH_OFFSET size=$BATCH_SIZE"
        "${SBATCH_CMD[@]}"
    fi

    SUBMITTED=$((SUBMITTED + BATCH_SIZE))
done
