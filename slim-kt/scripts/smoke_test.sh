#!/usr/bin/env bash
# P0 smoke test: end-to-end pipeline on SYNTHETIC data (no download, no LLM).
# Verifies the repo trains + evaluates on the AutoDL GPU before touching real data.
#
# Usage:  bash scripts/smoke_test.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/runs}"
NAME=synth

echo "[smoke] 1/4 generate synthetic data + teacher cache"
python -m slimkt.data.preprocess.make_synthetic --name "$NAME" \
  --num-users 500 --num-questions 300 --num-kcs 20 --embed-dim 384

echo "[smoke] 2/4 validate data contract"
python -m slimkt.data.preprocess.validate --dir "$DATA_ROOT/$NAME"

echo "[smoke] 3/4 train (5 epochs, small model)"
python -m slimkt.train --config configs/default.yaml \
  --set dataset.name="$NAME" dataset.has_options=true \
        model.d_model=64 model.n_blocks=2 train.epochs=5 train.early_stop_patience=5

echo "[smoke] 4/4 evaluate (cold-start + efficiency)"
CKPT="$OUTPUT_ROOT/${NAME}_sakt_fold0/best.pt"
python -m slimkt.evaluate --config configs/default.yaml \
  --set dataset.name="$NAME" dataset.has_options=true model.d_model=64 model.n_blocks=2 \
  --checkpoint "$CKPT"

echo "[smoke] DONE. See $OUTPUT_ROOT/${NAME}_sakt_fold0/{test_metrics.json,eval_metrics.json}"
