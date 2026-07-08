#!/usr/bin/env bash
# Cold-start-only paired bootstrap (the paper's headline metric) for dbe + eedi.
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate slimkt
export DATA_ROOT=/root/autodl-tmp/slim-kt/data
export OUTPUT_ROOT=/root/autodl-tmp/slim-kt/runs
export PYTHONUNBUFFERED=1
cd /root/autodl-tmp/slim-kt

for D in dbe_kt22 eedi; do
  echo "----- $D cold -----"
  DS=${D}_sakt_fold0 SEEDS="42 1 7" \
    python scripts/bootstrap_significance.py --dataset "$D" --metric cold --n-boot 2000
done
echo "COLD_BOOT_DONE"
