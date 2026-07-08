#!/usr/bin/env bash
# Step 2 — train the lightweight student (no LLM in the loop).
# Usage:  bash scripts/run_train.sh <dataset> [extra --set overrides...]
set -euo pipefail
DATASET="${1:-eedi}"; shift || true

python -m slimkt.train \
  --config configs/default.yaml \
  --dataset-config "configs/dataset/${DATASET}.yaml" \
  --set dataset.name="${DATASET}" "$@"
