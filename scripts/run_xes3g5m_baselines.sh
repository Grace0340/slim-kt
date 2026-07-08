#!/usr/bin/env bash
# Classic ID-based KT baselines on XES3G5M (DKT / DKVMN / AKT), 3 seeds each,
# same fold0 / dataloader / cold-start protocol as the SLIM-KT variants. Pure KT
# objective (all distillation lambdas = 0). Set EPOCHS=1 for a smoke test.
set -euo pipefail
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate slimkt
  export PATH=/root/miniconda3/envs/slimkt/bin:$PATH
fi
export DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/slim-kt/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/slim-kt/runs}"
cd /root/autodl-tmp/slim-kt

EPOCHS="${EPOCHS:-30}"
SEEDS="${SEEDS:-42 1 7}"
ARCHS="${ARCHS:-dkt dkvmn akt}"
CB="--config configs/default.yaml --dataset-config configs/dataset/xes3g5m.yaml"
SB="dataset.name=xes3g5m dataset.batch_size=512 dataset.num_workers=8 train.epochs=${EPOCHS} train.early_stop_patience=5 \
model.lambda_sem=0 model.lambda_attr=0 model.lambda_opt=0"

run () {
  local tag="$1"; shift
  local d="$OUTPUT_ROOT/xes3g5m_sakt_fold0_${tag}"
  echo "==================== [$tag] train ===================="
  python -m slimkt.train $CB --set $SB "$@" train.run_name="$tag"
  echo "==================== [$tag] eval  ===================="
  python -m slimkt.evaluate $CB --set $SB "$@" train.run_name="$tag" --checkpoint "$d/best.pt"
}

for a in $ARCHS; do
  for s in $SEEDS; do
    run "${a}_s${s}" seed=$s model.arch=$a
  done
done

echo "BASELINES_DONE"
