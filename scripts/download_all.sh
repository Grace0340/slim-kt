#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-click download of all public datasets used by SLIM-KT.
#
#   XES3G5M   (HuggingFace mirror)  -> interactions + precomputed RoBERTa emb.
#   DBE-KT22  (HuggingFace mirror)  -> text-rich DB-course KT
#   Eedi      (NeurIPS'20, Azure)   -> image-stem KT (coarse subject proxy)
#   EdNet     (Riiid, OPTIONAL)     -> only if you pass --with-ednet
#
# Raw dumps land under $RAW_ROOT; run the preprocessors afterwards to build the
# model-ready CSVs under $DATA_ROOT (see DATA.md, step 2).
#
# Usage:
#   bash scripts/download_all.sh                 # XES3G5M + DBE-KT22 + Eedi
#   RAW_ROOT=/data/raw bash scripts/download_all.sh
#   bash scripts/download_all.sh --with-ednet    # also fetch EdNet-KT1
# ---------------------------------------------------------------------------
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/root/autodl-tmp/raw}"
# in-China HuggingFace mirror; unset/override if you have direct HF access
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WITH_EDNET=0
for a in "$@"; do [ "$a" = "--with-ednet" ] && WITH_EDNET=1; done

echo "[download_all] RAW_ROOT=$RAW_ROOT  HF_ENDPOINT=$HF_ENDPOINT"
mkdir -p "$RAW_ROOT"

echo "== [1/3] XES3G5M =="
python "$HERE/download_xes3g5m.py" "$RAW_ROOT/xes3g5m"

echo "== [2/3] DBE-KT22 =="
python "$HERE/download_dbe_kt22.py" --out "$RAW_ROOT/dbe_kt22"

echo "== [3/3] Eedi (NeurIPS 2020) =="
python "$HERE/download_eedi.py" "$RAW_ROOT/eedi"

if [ "$WITH_EDNET" = "1" ]; then
  echo "== [+] EdNet-KT1 (optional) =="
  python "$HERE/download_ednet.py" "$RAW_ROOT/ednet"
fi

cat <<EOF

[download_all] raw data ready under: $RAW_ROOT
Next (step 2 in DATA.md): build model-ready CSVs, e.g.

  export DATA_ROOT=\${DATA_ROOT:-/root/autodl-tmp/slimkt_data}
  python -m slimkt.data.preprocess.eedi     --raw $RAW_ROOT/eedi     --out \$DATA_ROOT/eedi
  python -m slimkt.data.preprocess.dbe_kt22 --raw $RAW_ROOT/dbe_kt22 --out \$DATA_ROOT/dbe_kt22
  python -m slimkt.data.preprocess.xes3g5m  --raw $RAW_ROOT/xes3g5m  --out \$DATA_ROOT/xes3g5m
EOF
