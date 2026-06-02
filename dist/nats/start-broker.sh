#!/bin/bash
# Start a NATS server with JetStream for chemtax-smarts work distribution.
# Run this on one Tailscale node. Workers connect to nats://<this-tailscale-ip>:4222.

set -euo pipefail

STORE_DIR="${CHEMTAX_NATS_STORE:-$HOME/.chemtax-nats}"
ADDR="${CHEMTAX_NATS_ADDR:-0.0.0.0}"
PORT="${CHEMTAX_NATS_PORT:-4222}"

if ! command -v nats-server >/dev/null 2>&1; then
    echo "ERROR: nats-server not found. Install from https://github.com/nats-io/nats-server/releases" >&2
    exit 1
fi

mkdir -p "$STORE_DIR"

echo "Starting nats-server (JetStream) on $ADDR:$PORT"
echo "Store dir: $STORE_DIR"
echo "Workers connect with: nats://<this-tailscale-ip>:$PORT"
echo "(set CHEMTAX_NATS_ADDR to this node's Tailscale IP to bind only the tailnet)"

exec nats-server --jetstream --store_dir "$STORE_DIR" --addr "$ADDR" --port "$PORT"
