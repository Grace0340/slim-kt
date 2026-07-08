#!/usr/bin/env bash
# XES3G5M on AutoDL: download from HF mirror -> preprocess -> build semantic teacher cache.
#
# Usage:
#   export DATA_ROOT=/root/autodl-tmp/slim-kt/data
#   bash scripts/preprocess_xes3g5m.sh
#   # optional raw text (unlocks LLM attribute/option teacher):
#   bash scripts/preprocess_xes3g5m.sh /root/autodl-tmp/raw/xes3g5m/metadata/questions.json
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
RAW="${RAW_DIR:-/root/autodl-tmp/raw/xes3g5m}"
OUT="${DATA_ROOT:-./slimkt_data}/xes3g5m"
QUESTIONS="${1:-}"   # optional path to questions.json

pip install -q -U huggingface_hub pyarrow >/dev/null 2>&1 || true

# 1) download interactions + precomputed embeddings from HF mirror
python scripts/download_xes3g5m.py "$RAW"

# 2) preprocess to SLIM-KT CSVs (+ precomputed_qid_emb.npy); merge text if provided
if [ -n "$QUESTIONS" ]; then
  python -m slimkt.data.preprocess.xes3g5m --raw "$RAW" --questions "$QUESTIONS" --out "$OUT"
else
  python -m slimkt.data.preprocess.xes3g5m --raw "$RAW" --out "$OUT"
fi

# 3) validate the produced folder
python -m slimkt.data.preprocess.validate --dir "$OUT"

# 4) build the semantic teacher cache from the authors' precomputed embeddings
python -m slimkt.teacher.llm_teacher \
  --config configs/default.yaml --dataset-config configs/dataset/xes3g5m.yaml \
  --set dataset.name=xes3g5m --no-llm --precomputed-emb auto

echo "[xes3g5m] ready. Semantic cold-start pipeline can now train on xes3g5m."
