#!/bin/bash
# Gather each workstation's results.jsonl over Tailscale and merge them into one
# deduplicated log. Run on the host you want the final log on.
#
#   collect-merge.sh <npclassifier|classyfire> <host1> [host2 ...]
#
# Hosts are Tailscale names or IPs reachable over ssh. Each worker's output dir is
# assumed at $CHEMTAX_REMOTE_OUTPUT on the remote (default chemtax-smarts/results/<dataset>).

set -euo pipefail

DATASET="${1:?Usage: collect-merge.sh <dataset> <host> [host...]}"
shift
if [ "$#" -lt 1 ]; then
    echo "Give at least one worker host." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${CHEMTAX_REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BINARY="$REPO_DIR/target/release/chemtax-smarts"
COLLECT_DIR="${CHEMTAX_COLLECT_DIR:-$REPO_DIR/results/$DATASET/collected}"
REMOTE_OUTPUT="${CHEMTAX_REMOTE_OUTPUT:-chemtax-smarts/results/$DATASET}"
MERGED="${CHEMTAX_MERGED:-$REPO_DIR/results/$DATASET/results.jsonl}"

if [ ! -x "$BINARY" ]; then
    echo "ERROR: $BINARY missing. Build with: cargo build --release --features nats" >&2
    exit 1
fi

mkdir -p "$COLLECT_DIR"
for host in "$@"; do
    echo "Pulling results.jsonl from $host"
    rsync -az "$host:$REMOTE_OUTPUT/results.jsonl" "$COLLECT_DIR/$host.jsonl"
done

"$BINARY" --dataset "$DATASET" --merge-logs "$COLLECT_DIR" --merged-output "$MERGED"
echo "Merged log: $MERGED"
