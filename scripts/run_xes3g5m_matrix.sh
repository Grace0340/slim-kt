#!/usr/bin/env bash
# SLIM-KT experiment matrix on XES3G5M (cold-start + ablations).
# Prereqs:
#   - preprocessed data at $DATA_ROOT/xes3g5m (interactions/items/splits/precomputed_qid_emb.npy)
#   - teacher cache at $DATA_ROOT/teacher_cache/xes3g5m (teacher_sem + difficulty + options)
#
# Usage:
#   export DATA_ROOT=/root/autodl-tmp/slim-kt/data
#   export OUTPUT_ROOT=/root/autodl-tmp/slim-kt/runs
#   bash scripts/run_xes3g5m_matrix.sh
set -euo pipefail

# Activate the project env so `python` resolves under non-interactive SSH.
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate slimkt
  export PATH=/root/miniconda3/envs/slimkt/bin:$PATH
fi

export DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/slim-kt/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/slim-kt/runs}"

COMMON="--config configs/default.yaml --dataset-config configs/dataset/xes3g5m.yaml \
--set dataset.name=xes3g5m dataset.batch_size=512 dataset.num_workers=8 \
train.epochs=30 train.early_stop_patience=5"

run () {
  local name="$1"; shift
  echo "==================== [$name] train ===================="
  python -m slimkt.train $COMMON "$@" train.run_name="$name"
  echo "==================== [$name] eval ====================="
  python -m slimkt.evaluate $COMMON "$@" train.run_name="$name" \
    --checkpoint "$OUTPUT_ROOT/xes3g5m_sakt_fold0_${name}/best.pt"
}

# 1) ID-only baseline (should collapse on cold-start new items)
run id_cs        model.use_semantic=false model.use_id_embedding=true  model.lambda_attr=0   model.lambda_opt=0

# 2) Semantic-only (ID-free): the cold-start core of SLIM-KT
run sem_cs       model.use_semantic=true  model.use_id_embedding=false model.lambda_attr=0   model.lambda_opt=0

# 3) + attribute (difficulty) distillation only
run slim_attr    model.use_semantic=true  model.use_id_embedding=false model.lambda_attr=0.5 model.lambda_opt=0

# 4) + option-weight distillation only
run slim_opt     model.use_semantic=true  model.use_id_embedding=false model.lambda_attr=0   model.lambda_opt=0.5

# 5) full SLIM-KT (semantic + attribute + option)
run slim_full    model.use_semantic=true  model.use_id_embedding=false model.lambda_attr=0.5 model.lambda_opt=0.5

echo "ALL DONE. Metrics: $OUTPUT_ROOT/xes3g5m_sakt_fold0_*/{test_metrics.json,eval_metrics.json}"
