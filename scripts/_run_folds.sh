#!/usr/bin/env bash
# 5-fold CV robustness for the headline comparison + strongest baseline.
# Fixed seed=42, learner partition varies by dataset.fold (folds 1-4; fold0 exists).
# Variants: id_cs (SAKT-ID), sem_cs (SLIM-KT sem), akt (ID baseline).
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate slimkt
export DATA_ROOT=/root/autodl-tmp/slim-kt/data
export OUTPUT_ROOT=/root/autodl-tmp/slim-kt/runs
export PYTHONUNBUFFERED=1
cd /root/autodl-tmp/slim-kt

SEED=42
FOLDS="${FOLDS:-1 2 3 4}"
VARIANTS="${VARIANTS:-id_cs sem_cs akt}"
DATASETS="${DATASETS:-dbe_kt22 xes3g5m eedi}"

args_for () {
  case "$1" in
    akt)    echo "model.arch=akt   model.lambda_sem=0 model.lambda_attr=0 model.lambda_opt=0" ;;
    id_cs)  echo "model.use_semantic=false model.use_id_embedding=true  model.lambda_attr=0 model.lambda_opt=0" ;;
    sem_cs) echo "model.use_semantic=true  model.use_id_embedding=false model.lambda_attr=0 model.lambda_opt=0" ;;
  esac
}

for DS in $DATASETS; do
  CB="--config configs/default.yaml --dataset-config configs/dataset/${DS}.yaml"
  SB="dataset.name=${DS} dataset.batch_size=512 dataset.num_workers=8 train.epochs=30 train.early_stop_patience=5"
  for f in $FOLDS; do
    for v in $VARIANTS; do
      tag="${v}_s${SEED}"
      # baselines keep backbone=sakt; dir uses cfg.model.backbone -> always 'sakt'
      d="$OUTPUT_ROOT/${DS}_sakt_fold${f}_${tag}"
      if [ -f "$d/eval_metrics.json" ]; then echo "[skip] $d"; continue; fi
      echo "==================== [$DS fold$f/$tag] train ===================="
      python -m slimkt.train  $CB --set $SB dataset.fold=$f seed=$SEED $(args_for "$v") train.run_name="$tag"
      echo "==================== [$DS fold$f/$tag] eval  ===================="
      python -m slimkt.evaluate $CB --set $SB dataset.fold=$f seed=$SEED $(args_for "$v") train.run_name="$tag" --checkpoint "$d/best.pt"
    done
  done
done
echo "FOLDS_DONE"
