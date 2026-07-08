"""Aggregate 5-fold CV (seed 42) for the headline variants into mean+/-std."""
import glob
import json
import os

import numpy as np

ROOT = os.environ.get("OUTPUT_ROOT", "runs")
DATASETS = ["xes3g5m", "dbe_kt22", "eedi"]
VARIANTS = {"sem_cs": "SLIM-KT (sem)", "id_cs": "SAKT-ID", "akt": "AKT"}

rows = []
for ds in DATASETS:
    perfold = {}
    for v in VARIANTS:
        aucs, colds = [], []
        for f in range(5):
            fn = os.path.join(ROOT, f"{ds}_sakt_fold{f}_{v}_s42", "eval_metrics.json")
            if not os.path.exists(fn):
                continue
            m = json.load(open(fn))
            aucs.append(m["auc"]); colds.append(m["cold_auc"])
        if aucs:
            perfold[v] = (np.mean(aucs), np.std(aucs), np.mean(colds), np.std(colds), len(aucs))
    for v, name in VARIANTS.items():
        if v in perfold:
            a, sa, c, sc, n = perfold[v]
            rows.append((ds, name, n, a, sa, c, sc))
            print(f"{ds:9s} {name:14s} folds={n} AUC={a:.4f}+/-{sa:.4f} cold={c:.4f}+/-{sc:.4f}")
    if "sem_cs" in perfold and "id_cs" in perfold:
        d = perfold["sem_cs"][2] - perfold["id_cs"][2]
        print(f"  -> {ds} cold-start gain (sem-id), 5-fold mean = {d:+.4f}")

with open("results/folds_summary.json", "w") as f:
    json.dump([{"dataset": r[0], "variant": r[1], "folds": r[2], "auc": r[3],
                "auc_std": r[4], "cold_auc": r[5], "cold_auc_std": r[6]} for r in rows],
              f, indent=2)
print("[agg_folds] wrote results/folds_summary.json")
