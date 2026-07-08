#!/usr/bin/env bash
# Generality check on Eedi (NeurIPS 2020): does the "ID-free semantic -> cold-start"
# win reproduce on a second dataset? Eedi's semantic embeddings come from the
# subject-hierarchy proxy text; difficulty/options are absent (image stems), so we
# run the CORE comparison only: DKT / DKVMN / AKT / SAKT-ID (id_cs) / SLIM-KT (sem_cs),
# 3 seeds each, fold0, same cold-start protocol. Set EPOCHS=1 for a smoke test.
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
VARIANTS="${VARIANTS:-dkt dkvmn akt id_cs sem_cs}"
CB="--config configs/default.yaml --dataset-config configs/dataset/eedi.yaml"
SB="dataset.name=eedi dataset.batch_size=512 dataset.num_workers=8 train.epochs=${EPOCHS} train.early_stop_patience=5"

run () {
  local tag="$1"; shift
  local d="$OUTPUT_ROOT/eedi_sakt_fold0_${tag}"
  echo "==================== [$tag] train ===================="
  python -m slimkt.train $CB --set $SB "$@" train.run_name="$tag"
  echo "==================== [$tag] eval  ===================="
  python -m slimkt.evaluate $CB --set $SB "$@" train.run_name="$tag" --checkpoint "$d/best.pt"
}

args_for () {
  case "$1" in
    dkt)    echo "model.arch=dkt   model.lambda_sem=0 model.lambda_attr=0 model.lambda_opt=0" ;;
    dkvmn)  echo "model.arch=dkvmn model.lambda_sem=0 model.lambda_attr=0 model.lambda_opt=0" ;;
    akt)    echo "model.arch=akt   model.lambda_sem=0 model.lambda_attr=0 model.lambda_opt=0" ;;
    id_cs)  echo "model.use_semantic=false model.use_id_embedding=true  model.lambda_attr=0 model.lambda_opt=0" ;;
    sem_cs) echo "model.use_semantic=true  model.use_id_embedding=false model.lambda_attr=0 model.lambda_opt=0" ;;
  esac
}

for s in $SEEDS; do
  for v in $VARIANTS; do
    run "${v}_s${s}" seed=$s $(args_for "$v")
  done
done

echo "EEDI_DONE"
