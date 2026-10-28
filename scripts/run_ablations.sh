#!/usr/bin/env bash
# Each configuration has its own directory and can be resumed independently.
set -euo pipefail
CONFIG="${1:-configs/default.yaml}"
for mode in full full_scan fixed_sparse flow_full no_gate no_transport no_spectral no_phase no_cycle; do
  elst train --config "$CONFIG" --mode "$mode" --output "runs/$mode" --seed 42
done
