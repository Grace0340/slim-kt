"""Aggregate the XES3G5M resolution sweep: mean +/- std of AUC / cold-AUC per K,
plus the full-resolution sem_cs anchor. Prints a table sorted by K so the
'resolution -> cold-start gain' curve is directly readable."""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.environ.get("OUTPUT_ROOT", "/root/autodl-tmp/slim-kt/runs")
PREFIX = "xes3g5m_sakt_fold0"
FULL_K = int(os.environ.get("FULL_K", "7230"))  # distinct embeddings at full res

by_k = defaultdict(list)
for d in glob.glob(os.path.join(ROOT, f"{PREFIX}_*")):
    name = os.path.basename(d)
    m = re.search(rf"^{PREFIX}_semK(\d+)_s(\d+)$", name)
    if m:
        k = int(m.group(1))
    elif re.search(rf"^{PREFIX}_sem_cs_s(\d+)$", name):
        k = FULL_K
    else:
        continue
    mp = os.path.join(d, "eval_metrics.json")
    if os.path.exists(mp):
        with open(mp) as f:
            by_k[k].append(json.load(f))


def agg(rows, key):
    vals = [r[key] for r in rows if key in r]
    return (np.mean(vals), np.std(vals), len(vals)) if vals else (float("nan"), float("nan"), 0)


print("=== XES3G5M: semantic resolution (K distinct embeddings) vs performance ===")
print(f"{'K':>7} {'uniq%':>7} {'n':>3}   {'AUC':>16}   {'coldAUC':>16}")
print("-" * 60)
for k in sorted(by_k):
    rows = by_k[k]
    am, asd, n = agg(rows, "auc")
    cm, csd, _ = agg(rows, "cold_auc")
    uniq = 100.0 * min(k, FULL_K) / 7652.0
    print(f"{k:>7} {uniq:>6.1f}% {n:>3}   {am:.4f}+/-{asd:.4f}   {cm:.4f}+/-{csd:.4f}")
