#!/usr/bin/env bash
# Overall-AUC paired bootstrap (SLIM-KT sem vs SAKT-ID) for all three datasets.
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate slimkt
export DATA_ROOT=/root/autodl-tmp/slim-kt/data
export OUTPUT_ROOT=/root/autodl-tmp/slim-kt/runs
export PYTHONUNBUFFERED=1
cd /root/autodl-tmp/slim-kt

for D in dbe_kt22 xes3g5m eedi; do
  echo "----- $D overall -----"
  DS=${D}_sakt_fold0 SEEDS="42 1 7" \
    python scripts/bootstrap_significance.py --dataset "$D" --metric overall --n-boot 2000
done
echo "OVERALL_BOOT_DONE"
