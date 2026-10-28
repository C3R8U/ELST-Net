#!/usr/bin/env bash
# Independent seeds make variation between training runs easier to inspect.
set -euo pipefail
CONFIG="${1:-configs/default.yaml}"
for seed in 41 42 43 44 45; do
  elst train --config "$CONFIG" --seed "$seed" --output "runs/seed_$seed"
done
