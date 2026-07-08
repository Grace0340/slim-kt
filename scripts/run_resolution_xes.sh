#!/usr/bin/env bash
# Mechanism study: cold-AUC vs semantic resolution on XES3G5M. Trains sem_cs with
# K-means-quantized embeddings (K distinct prototypes) for several K, 3 seeds each.
# The full-resolution point is the existing sem_cs_s* runs. Build the degraded
# tables first with make_degraded_sem.py.
set -euo pipefail
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate slimkt
  export PATH=/root/miniconda3/envs/slimkt/bin:$PATH
fi
export DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/slim-kt/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/slim-kt/runs}"
cd /root/autodl-tmp/slim-kt

KS="${KS:-50 200 800 3200}"
SEEDS="${SEEDS:-42 1 7}"
CACHE="$DATA_ROOT/teacher_cache/xes3g5m"
CB="--config configs/default.yaml --dataset-config configs/dataset/xes3g5m.yaml"
SB="dataset.name=xes3g5m dataset.batch_size=512 dataset.num_workers=8 train.epochs=30 train.early_stop_patience=5 \
model.use_semantic=true model.use_id_embedding=false model.lambda_attr=0 model.lambda_opt=0"

python scripts/make_degraded_sem.py --cache "$CACHE" --ks $KS

run () {
  local tag="$1"; shift
  local d="$OUTPUT_ROOT/xes3g5m_sakt_fold0_${tag}"
  echo "==================== [$tag] train ===================="
  python -m slimkt.train $CB --set $SB "$@" train.run_name="$tag"
  echo "==================== [$tag] eval  ===================="
  python -m slimkt.evaluate $CB --set $SB "$@" train.run_name="$tag" --checkpoint "$d/best.pt"
}

for K in $KS; do
  for s in $SEEDS; do
    run "semK${K}_s${s}" seed=$s "model.sem_override_path=$CACHE/sem_k${K}.npy"
  done
done

echo "RESOLUTION_DONE"
