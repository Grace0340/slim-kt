#!/usr/bin/env bash
# Step 3 — evaluate a trained checkpoint: AUC/ACC, cold-start AUC, inference
# latency + peak GPU memory, and interpretability artifacts.
# Usage:  bash scripts/run_eval.sh <dataset> <checkpoint.pt>
set -euo pipefail
DATASET="${1:-eedi}"
CKPT="${2:?path to checkpoint .pt required}"

python -m slimkt.evaluate \
  --config configs/default.yaml \
  --dataset-config "configs/dataset/${DATASET}.yaml" \
  --set dataset.name="${DATASET}" \
  --checkpoint "${CKPT}"
