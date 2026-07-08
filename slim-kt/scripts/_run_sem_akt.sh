#!/usr/bin/env bash
# Train + eval the AKT+semantic student (Table S1 backbone-generalization row).
# 3 datasets x 3 seeds. Skips runs whose eval_metrics.json already exists.
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate slimkt
export DATA_ROOT=/root/autodl-tmp/slim-kt/data
export OUTPUT_ROOT=/root/autodl-tmp/slim-kt/runs
export PYTHONUNBUFFERED=1
cd /root/autodl-tmp/slim-kt

EPOCHS=30
SEEDS="42 1 7"
DATASETS="${DATASETS:-dbe_kt22 xes3g5m eedi}"

for DS in $DATASETS; do
  CB="--config configs/default.yaml"
  [ -f "configs/dataset/${DS}.yaml" ] && CB="$CB --dataset-config configs/dataset/${DS}.yaml"
  SB="dataset.name=${DS} dataset.batch_size=512 dataset.num_workers=8 train.epochs=${EPOCHS} train.early_stop_patience=5"
  MB="model.arch=slimkt model.backbone=akt model.use_semantic=true model.use_id_embedding=false model.lambda_attr=0 model.lambda_opt=0"
  for s in $SEEDS; do
    tag="sem_akt_s${s}"
    # train.py names the dir {dataset}_{backbone}_fold0_{run_name}; backbone=akt here
    d="$OUTPUT_ROOT/${DS}_akt_fold0_${tag}"
    if [ -f "$d/eval_metrics.json" ]; then
      echo "[skip] $d already done"; continue
    fi
    echo "==================== [$DS/$tag] train ===================="
    python -m slimkt.train $CB --set $SB $MB seed=$s train.run_name="$tag"
    echo "==================== [$DS/$tag] eval  ===================="
    python -m slimkt.evaluate $CB --set $SB $MB seed=$s train.run_name="$tag" --checkpoint "$d/best.pt"
  done
done
echo "SEM_AKT_DONE"
