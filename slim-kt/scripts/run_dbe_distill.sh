#!/usr/bin/env bash
# LLM attribute-distillation ablation on DBE-KT22 (needs the LLM teacher cache
# built by run_teacher_dbe.sh: real difficulty + option labels). sem_cs (no
# distillation) already exists from the core run and serves as the matched anchor.
#   slim_attr : + difficulty/KC distillation   (lambda_attr=0.5)
#   slim_opt  : + option-weight distillation   (lambda_opt=0.5)
#   slim_full : both                            (0.5/0.5) and a neutral 0.05 check
set -euo pipefail
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate slimkt
  export PATH=/root/miniconda3/envs/slimkt/bin:$PATH
fi
export DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/slim-kt/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/slim-kt/runs}"
cd /root/autodl-tmp/slim-kt

SEEDS="${SEEDS:-42 1 7}"
CB="--config configs/default.yaml --dataset-config configs/dataset/dbe_kt22.yaml"
SB="dataset.name=dbe_kt22 dataset.batch_size=512 dataset.num_workers=8 train.epochs=30 train.early_stop_patience=5 \
model.use_semantic=true model.use_id_embedding=false"

run () {
  local tag="$1"; shift
  local d="$OUTPUT_ROOT/dbe_kt22_sakt_fold0_${tag}"
  echo "==================== [$tag] train ===================="
  python -m slimkt.train $CB --set $SB "$@" train.run_name="$tag"
  echo "==================== [$tag] eval  ===================="
  python -m slimkt.evaluate $CB --set $SB "$@" train.run_name="$tag" --checkpoint "$d/best.pt"
}

for s in $SEEDS; do
  run "slim_attr_s${s}" seed=$s model.lambda_attr=0.5 model.lambda_opt=0
  run "slim_opt_s${s}"  seed=$s model.lambda_attr=0   model.lambda_opt=0.5
  run "slim_full_s${s}" seed=$s model.lambda_attr=0.5 model.lambda_opt=0.5
  run "slim_full_l005_s${s}" seed=$s model.lambda_attr=0.05 model.lambda_opt=0.05
done

echo "DBE_DISTILL_DONE"
