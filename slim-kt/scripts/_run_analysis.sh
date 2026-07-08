#!/usr/bin/env bash
# No-training supplementary analyses: bootstrap significance + option recovery.
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate slimkt
export DATA_ROOT=/root/autodl-tmp/slim-kt/data
export OUTPUT_ROOT=/root/autodl-tmp/slim-kt/runs
export PYTHONUNBUFFERED=1
cd /root/autodl-tmp/slim-kt

echo "########## bootstrap significance (sem_cs vs id_cs) ##########"
for D in xes3g5m dbe_kt22 eedi; do
  echo "----- $D cold -----"
  DS=${D}_sakt_fold0 SEEDS="42 1 7" OUTPUT_ROOT=$OUTPUT_ROOT \
    python scripts/bootstrap_significance.py --dataset "$D" --metric cold --n-boot 5000
  echo "----- $D overall -----"
  DS=${D}_sakt_fold0 SEEDS="42 1 7" OUTPUT_ROOT=$OUTPUT_ROOT \
    python scripts/bootstrap_significance.py --dataset "$D" --metric overall --n-boot 5000
done

echo "########## option recovery (DBE-KT22 answer key) ##########"
python scripts/option_recovery.py --cache data/teacher_cache/dbe_kt22 \
  --items data/dbe_kt22/items.csv --raw /root/autodl-tmp/raw/dbe_kt22 || true

echo "ANALYSIS_DONE"
