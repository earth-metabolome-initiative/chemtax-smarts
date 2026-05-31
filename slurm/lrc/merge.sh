#!/bin/bash
# Concatenate the per-shard results.jsonl files into one results.jsonl once the
# array has finished. Shards evolve disjoint labels, so a plain concatenation is
# the merged result with no duplicates.

set -euo pipefail

DATASET="${1:?Usage: merge.sh <npclassifier|classyfire> [OUTPUT_DIR]}"
SCRATCH_ROOT="${CHEMTAX_SCRATCH_ROOT:-/global/scratch/users/$USER/chemtax-smarts}"
OUTPUT_DIR="${2:-$SCRATCH_ROOT/results/$DATASET}"
SHARDS_DIR="$OUTPUT_DIR/shards"
MERGED="$OUTPUT_DIR/results.jsonl"

if [ ! -d "$SHARDS_DIR" ]; then
    echo "No shard directory at $SHARDS_DIR"
    exit 1
fi

: > "$MERGED"
shard_count=0
for dir in "$SHARDS_DIR"/shard-*; do
    log="$dir/results.jsonl"
    [ -f "$log" ] || continue
    cat "$log" >> "$MERGED"
    shard_count=$((shard_count + 1))
done

labels=$(grep -c '' "$MERGED" || true)
echo "Merged $shard_count shard logs into $MERGED ($labels labels)."
