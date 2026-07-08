"""Aggregate multi-seed + lambda-sweep XES3G5M results into mean/std tables."""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

root = os.environ.get("OUTPUT_ROOT", "/root/autodl-tmp/slim-kt/runs")
# DS controls which run dirs to aggregate, e.g. xes3g5m_sakt_fold0 or eedi_sakt_fold0
DS = os.environ.get("DS", "xes3g5m_sakt_fold0")

core_re = re.compile(
    rf"{DS}_(dkt|dkvmn|akt|id_cs|sem_cs|sem_akt|slim_attr|slim_opt|slim_full)_s(\d+)$")
sweep_re = re.compile(rf"{DS}_(slim_full_l\w+)_s(\d+)$")


def load(d):
    ev = os.path.join(d, "eval_metrics.json")
    return json.load(open(ev)) if os.path.exists(ev) else None


core = defaultdict(lambda: defaultdict(list))   # variant -> metric -> [vals]
sweep = {}                                      # tag -> metrics
for d in sorted(glob.glob(os.path.join(root, f"{DS}_*"))):
    base = os.path.basename(d)
    m = core_re.match(base)
    if m:
        met = load(d)
        if met:
            for k in ("auc", "acc", "rmse", "cold_auc",
                      "latency_ms_per_interaction", "peak_gpu_mem_mb"):
                if met.get(k) is not None:
                    core[m.group(1)][k].append(met[k])
        continue
    s = sweep_re.match(base)
    if s:
        met = load(d)
        if met:
            sweep[s.group(1)] = met


order = ["dkt", "dkvmn", "akt", "id_cs", "sem_cs", "sem_akt", "slim_attr", "slim_opt", "slim_full"]
print("=== baselines + SLIM-KT variants x seeds (mean +/- std) ===")
hdr = (f"{'variant':<11} {'n':>2} {'AUC':>16} {'coldAUC':>16} {'ACC':>16} "
       f"{'RMSE':>16} {'lat_ms/it':>10} {'peakMB':>9}")
print(hdr); print("-" * len(hdr))
def ms(v):
    a = np.array(v, dtype=float)
    return f"{a.mean():.4f}+/-{a.std():.4f}" if a.size else "-"
def mean1(v, p=3):
    a = np.array(v, dtype=float)
    return f"{a.mean():.{p}f}" if a.size else "-"
for name in order:
    if name in core:
        c = core[name]
        n = len(c.get("auc", []))
        print(f"{name:<11} {n:>2} {ms(c.get('auc',[])):>16} {ms(c.get('cold_auc',[])):>16} "
              f"{ms(c.get('acc',[])):>16} {ms(c.get('rmse',[])):>16} "
              f"{mean1(c.get('latency_ms_per_interaction',[])):>10} "
              f"{mean1(c.get('peak_gpu_mem_mb',[]),1):>9}")

if sweep:
    print("\n=== slim_full lambda sweep (seed42) ===")
    print(f"{'tag':<22} {'AUC':>8} {'coldAUC':>8} {'ACC':>8} {'RMSE':>8}")
    for tag in sorted(sweep):
        m = sweep[tag]
        print(f"{tag:<22} {m.get('auc',0):>8.4f} {m.get('cold_auc',0):>8.4f} "
              f"{m.get('acc',0):>8.4f} {m.get('rmse',0):>8.4f}")
