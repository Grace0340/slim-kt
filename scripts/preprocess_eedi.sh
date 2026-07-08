#!/usr/bin/env bash
# Convert a raw Eedi dump into SLIM-KT CSVs, then validate.
# Usage:  bash scripts/preprocess_eedi.sh /path/to/eedi_raw
set -euo pipefail
RAW="${1:?path to extracted Eedi dataset required}"
OUT="${DATA_ROOT:-./slimkt_data}/eedi"

python -m slimkt.data.preprocess.eedi --raw "$RAW" --out "$OUT"
python -m slimkt.data.preprocess.validate --dir "$OUT"
