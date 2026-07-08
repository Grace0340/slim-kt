#!/usr/bin/env bash
# Generic CORE comparison (DKT / DKVMN / AKT / SAKT-ID / SLIM-KT-sem) x seeds for
# any preprocessed dataset with a teacher cache. Usage:
#   DATASET=dbe_kt22 bash scripts/run_core.sh
#   DATASET=eedi EPOCHS=30 SEEDS="42 1 7" bash scripts/run_core.sh
set -euo pipefail
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate slimkt
  export PATH=/root/miniconda3/envs/slimkt/bin:$PATH
fi
export DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/slim-kt/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/slim-kt/runs}"
cd /root/autodl-tmp/slim-kt

DATASET="${DATASET:?set DATASET, e.g. dbe_kt22}"
EPOCHS="${EPOCHS:-30}"
SEEDS="${SEEDS:-42 1 7}"
VARIANTS="${VARIANTS:-dkt dkvmn akt id_cs sem_cs sem_akt}"
CB="--config configs/default.yaml --dataset-config configs/dataset/${DATASET}.yaml"
SB="dataset.name=${DATASET} dataset.batch_size=512 dataset.num_workers=8 train.epochs=${EPOCHS} train.early_stop_patience=5"

run () {
  local tag="$1"; shift
  local d="$OUTPUT_ROOT/${DATASET}_sakt_fold0_${tag}"
  echo "==================== [$DATASET/$tag] train ===================="
  python -m slimkt.train $CB --set $SB "$@" train.run_name="$tag"
  echo "==================== [$DATASET/$tag] eval  ===================="
  python -m slimkt.evaluate $CB --set $SB "$@" train.run_name="$tag" --checkpoint "$d/best.pt"
}

args_for () {
  case "$1" in
    dkt)    echo "model.arch=dkt   model.lambda_sem=0 model.lambda_attr=0 model.lambda_opt=0" ;;
    dkvmn)  echo "model.arch=dkvmn model.lambda_sem=0 model.lambda_attr=0 model.lambda_opt=0" ;;
    akt)    echo "model.arch=akt   model.lambda_sem=0 model.lambda_attr=0 model.lambda_opt=0" ;;
    id_cs)  echo "model.use_semantic=false model.use_id_embedding=true  model.lambda_attr=0 model.lambda_opt=0" ;;
    sem_cs) echo "model.use_semantic=true  model.use_id_embedding=false model.lambda_attr=0 model.lambda_opt=0" ;;
    sem_akt) echo "model.arch=slimkt model.backbone=akt model.use_semantic=true model.use_id_embedding=false model.lambda_attr=0 model.lambda_opt=0" ;;
  esac
}

for s in $SEEDS; do
  for v in $VARIANTS; do
    run "${v}_s${s}" seed=$s $(args_for "$v")
  done
done

echo "${DATASET}_CORE_DONE"
