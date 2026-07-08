#!/usr/bin/env bash
# Step 1 — build the frozen-LLM teacher-signal cache (semantic embeddings,
# attributes, option weights). Runs the LLM ONCE, offline; the cache is reused
# by every training run.
#
# Usage:  bash scripts/run_teacher.sh <dataset>   e.g. bash scripts/run_teacher.sh eedi
set -euo pipefail
DATASET="${1:-eedi}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# If you serve the extraction LLM locally with vLLM on the same instance:
#   python -m vllm.entrypoints.openai.api_server --model Qwen2.5-7B-Instruct --port 8000 &
# then OPENAI_BASE_URL defaults to http://127.0.0.1:8000/v1
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

python -m slimkt.teacher.llm_teacher \
  --config configs/default.yaml \
  --dataset-config "configs/dataset/${DATASET}.yaml" \
  --set dataset.name="${DATASET}"
