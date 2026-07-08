#!/usr/bin/env bash
# Multi-seed robustness + lambda sweep for the XES3G5M ablations.
#   * 5 variants x 3 seeds -> mean/std of AUC & cold-AUC (are the gaps real?)
#   * slim_full lambda sweep {0.05,0.1,0.2} @ seed42 (+ existing 0.5) -> can small
#     distillation weights turn the aux losses neutral/positive?
# Within a seed all variants share the same cold-start split (cfg.seed), so
# per-seed comparisons are matched. Runs train+eval into per-run dirs.
set -euo pipefail
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate slimkt
  export PATH=/root/miniconda3/envs/slimkt/bin:$PATH
fi
export DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/slim-kt/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/slim-kt/runs}"
cd /root/autodl-tmp/slim-kt

SEEDS="42 1 7"
CB="--config configs/default.yaml --dataset-config configs/dataset/xes3g5m.yaml"
SB="dataset.name=xes3g5m dataset.batch_size=512 dataset.num_workers=8 train.epochs=30 train.early_stop_patience=5"

run () {
  local tag="$1"; shift
  local d="$OUTPUT_ROOT/xes3g5m_sakt_fold0_${tag}"
  echo "==================== [$tag] train ===================="
  python -m slimkt.train $CB --set $SB "$@" train.run_name="$tag"
  echo "==================== [$tag] eval  ===================="
  python -m slimkt.evaluate $CB --set $SB "$@" train.run_name="$tag" --checkpoint "$d/best.pt"
}

for s in $SEEDS; do
  run "id_cs_s${s}"     seed=$s model.use_semantic=false model.use_id_embedding=true  model.lambda_attr=0   model.lambda_opt=0
  run "sem_cs_s${s}"    seed=$s model.use_semantic=true  model.use_id_embedding=false model.lambda_attr=0   model.lambda_opt=0
  run "slim_attr_s${s}" seed=$s model.use_semantic=true  model.use_id_embedding=false model.lambda_attr=0.5 model.lambda_opt=0
  run "slim_opt_s${s}"  seed=$s model.use_semantic=true  model.use_id_embedding=false model.lambda_attr=0   model.lambda_opt=0.5
  run "slim_full_s${s}" seed=$s model.use_semantic=true  model.use_id_embedding=false model.lambda_attr=0.5 model.lambda_opt=0.5
done

for l in 0.05 0.1 0.2; do
  run "slim_full_l${l/./}_s42" seed=42 model.use_semantic=true model.use_id_embedding=false model.lambda_attr=$l model.lambda_opt=$l
done

echo "MULTISEED_DONE"
