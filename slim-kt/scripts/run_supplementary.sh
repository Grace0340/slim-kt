#!/usr/bin/env bash
# Supplementary experiments for Table S1: bootstrap, option recovery, sem_akt.
# Run after run_core.sh (or standalone on synth for smoke validation).
#
#   DATASET=synth DATA_ROOT=./data OUTPUT_ROOT=./runs bash scripts/run_supplementary.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/runs}"
cd "$REPO_ROOT"

DATASET="${DATASET:?set DATASET}"
DS_PREFIX="${DS_PREFIX:-${DATASET}_sakt_fold0}"
SEEDS="${SEEDS:-42 1 7}"
EPOCHS="${EPOCHS:-30}"

CB="--config configs/default.yaml"
if [[ -f "configs/dataset/${DATASET}.yaml" ]]; then
  CB="$CB --dataset-config configs/dataset/${DATASET}.yaml"
fi
SB="dataset.name=${DATASET} train.epochs=${EPOCHS} train.early_stop_patience=5"

run_variant () {
  local tag="$1"; shift
  local seed="$1"; shift
  local d="$OUTPUT_ROOT/${DS_PREFIX}_${tag}_s${seed}"
  echo "==== [$DATASET/$tag seed=$seed] train ===="
  python -m slimkt.train $CB --set $SB "$@" seed="$seed" train.run_name="${tag}_s${seed}"
  echo "==== [$DATASET/$tag seed=$seed] eval ===="
  python -m slimkt.evaluate $CB --set $SB "$@" seed="$seed" train.run_name="${tag}_s${seed}" \
    --checkpoint "$d/best.pt"
}

# --- optional: AKT backbone + semantic encoder (Table S1 row) ---
if [[ "${RUN_SEM_AKT:-1}" == "1" ]]; then
  for s in $SEEDS; do
    if [[ ! -f "$OUTPUT_ROOT/${DS_PREFIX}_sem_akt_s${s}/eval_metrics.json" ]]; then
      run_variant sem_akt "$s" \
        model.arch=slimkt model.backbone=akt \
        model.use_semantic=true model.use_id_embedding=false \
        model.lambda_attr=0 model.lambda_opt=0
    fi
  done
fi

# --- paired bootstrap (sem_cs vs id_cs) ---
if [[ "${RUN_BOOTSTRAP:-1}" == "1" ]]; then
  DS="$DS_PREFIX" SEEDS="$SEEDS" OUTPUT_ROOT="$OUTPUT_ROOT" \
    python scripts/bootstrap_significance.py --dataset "$DATASET" --metric cold
  DS="$DS_PREFIX" SEEDS="$SEEDS" OUTPUT_ROOT="$OUTPUT_ROOT" \
    python scripts/bootstrap_significance.py --dataset "$DATASET" --metric overall
fi

# --- option recovery (DBE raw dir or synth items) ---
if [[ "${RUN_OPTION_RECOVERY:-1}" == "1" ]]; then
  CACHE="$DATA_ROOT/teacher_cache/$DATASET"
  if [[ -d "$CACHE" && -f "$CACHE/teacher_options.npy" ]]; then
    if [[ -n "${DBE_RAW:-}" && -d "$DBE_RAW" ]]; then
      python scripts/option_recovery.py --cache "$CACHE" --raw "$DBE_RAW"
    elif [[ -f "$DATA_ROOT/$DATASET/items.csv" ]]; then
      python scripts/option_recovery.py --cache "$CACHE" \
        --items "$DATA_ROOT/$DATASET/items.csv" || true
    fi
  fi
fi

# --- export Table S1 rows ---
python scripts/export_table_s1.py --output-root "$OUTPUT_ROOT" \
  --csv "$REPO_ROOT/results/table_s1.csv" \
  --include-template

echo "${DATASET}_SUPPLEMENTARY_DONE"
