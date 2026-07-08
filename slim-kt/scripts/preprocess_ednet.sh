#!/usr/bin/env bash
# Convert a raw EdNet-KT1 dump into SLIM-KT CSVs, then validate.
# Usage:  bash scripts/preprocess_ednet.sh /path/to/EdNet-KT1 /path/to/questions.csv [max_users]
set -euo pipefail
RAW="${1:?path to EdNet-KT1 folder (u*.csv) required}"
QUESTIONS="${2:?path to contents/questions.csv required}"
MAX_USERS="${3:-20000}"
OUT="${DATA_ROOT:-./slimkt_data}/ednet"

python -m slimkt.data.preprocess.ednet \
  --raw "$RAW" --questions "$QUESTIONS" --out "$OUT" \
  --max-users "$MAX_USERS" --min-interactions 5
python -m slimkt.data.preprocess.validate --dir "$OUT"
