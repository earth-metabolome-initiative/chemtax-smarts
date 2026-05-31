#!/bin/bash
# Build the chemtax-smarts release binary the shard workers run.
#
# Run this on an lr6 compute node (for example inside `salloc --partition=lr6
# --qos=lr_normal --nodes=1 --time=1:00:00`) so `target-cpu=native` matches the
# CPUs the array jobs land on. Building on the login node can emit instructions
# the compute nodes do not have.

set -euo pipefail

REPO_DIR="${CHEMTAX_REPO_DIR:-$HOME/chemtax-smarts}"
SCRATCH_ROOT="${CHEMTAX_SCRATCH_ROOT:-/global/scratch/users/$USER/chemtax-smarts}"

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

mkdir -p "$SCRATCH_ROOT/logs"

export RUSTFLAGS="-C target-cpu=native"
cargo build --release --locked

echo "Built $REPO_DIR/target/release/chemtax-smarts"
echo "Scratch root: $SCRATCH_ROOT"
