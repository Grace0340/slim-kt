#!/usr/bin/env bash
# 1-epoch smoke test: verify the full SLIM-KT train->eval path on XES3G5M using
# the freshly built teacher cache (sem + difficulty + options). Fast sanity check
# before launching the full experiment matrix.
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate slimkt
export PATH=/root/miniconda3/envs/slimkt/bin:$PATH
export DATA_ROOT=/root/autodl-tmp/slim-kt/data
export OUTPUT_ROOT=/root/autodl-tmp/slim-kt/runs
cd /root/autodl-tmp/slim-kt

COMMON="--config configs/default.yaml --dataset-config configs/dataset/xes3g5m.yaml \
--set dataset.name=xes3g5m dataset.batch_size=512 dataset.num_workers=8 \
train.epochs=1 train.early_stop_patience=5 \
model.use_semantic=true model.use_id_embedding=false \
model.lambda_attr=0.5 model.lambda_opt=0.5"

echo "==================== [smoke] train ===================="
python -m slimkt.train $COMMON train.run_name=smoke
echo "==================== [smoke] eval ====================="
python -m slimkt.evaluate $COMMON train.run_name=smoke \
  --checkpoint "$OUTPUT_ROOT/xes3g5m_sakt_fold0_smoke/best.pt"
echo "SMOKE_DONE"
