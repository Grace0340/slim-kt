#!/usr/bin/env bash
# Re-run ONLY the attribute-distillation variants after populating teacher_kc.npy
# with ground-truth concept grounding. id_cs / sem_cs / slim_opt use lambda_attr=0
# and are unaffected, so we skip them. Backs up the previous (KC-less) metrics.
set -euo pipefail
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate slimkt
  export PATH=/root/miniconda3/envs/slimkt/bin:$PATH
fi
export DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/slim-kt/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/slim-kt/runs}"
cd /root/autodl-tmp/slim-kt

COMMON="--config configs/default.yaml --dataset-config configs/dataset/xes3g5m.yaml \
--set dataset.name=xes3g5m dataset.batch_size=512 dataset.num_workers=8 \
train.epochs=30 train.early_stop_patience=5"

run () {
  local name="$1"; shift
  local d="$OUTPUT_ROOT/xes3g5m_sakt_fold0_${name}"
  # preserve the previous (no-KC) metrics for comparison
  [ -f "$d/eval_metrics.json" ] && cp "$d/eval_metrics.json" "$d/eval_metrics_noKC.json"
  [ -f "$d/test_metrics.json" ] && cp "$d/test_metrics.json" "$d/test_metrics_noKC.json"
  echo "==================== [$name] train (KC-grounded) ===================="
  python -m slimkt.train $COMMON "$@" train.run_name="$name"
  echo "==================== [$name] eval ====================="
  python -m slimkt.evaluate $COMMON "$@" train.run_name="$name" \
    --checkpoint "$d/best.pt"
}

run slim_attr    model.use_semantic=true  model.use_id_embedding=false model.lambda_attr=0.5 model.lambda_opt=0
run slim_full    model.use_semantic=true  model.use_id_embedding=false model.lambda_attr=0.5 model.lambda_opt=0.5

echo "KC_RERUN_DONE"
