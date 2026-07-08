"""Plot the XES3G5M semantic-resolution mechanism curve (AUC & cold-AUC vs K)."""
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.environ.get("OUTPUT_ROOT", "/root/autodl-tmp/slim-kt/runs")
PREFIX = "xes3g5m_sakt_fold0"
FULL_K = 7230
NQ = 7652

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
        by_k[k].append(json.load(open(mp)))

ks = sorted(by_k)
uniq = [100.0 * min(k, FULL_K) / NQ for k in ks]


def ms(k, key):
    v = [r[key] for r in by_k[k] if key in r]
    return np.mean(v), np.std(v)


auc = np.array([ms(k, "auc") for k in ks])
cold = np.array([ms(k, "cold_auc") for k in ks])

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.errorbar(uniq, auc[:, 0], yerr=auc[:, 1], marker="o", capsize=3, lw=2,
            color="#1f77b4", label="Overall AUC")
ax.errorbar(uniq, cold[:, 0], yerr=cold[:, 1], marker="s", capsize=3, lw=2,
            color="#d62728", label="Cold-start AUC")
ax.axhline(0.699, ls="--", lw=1.2, color="#7f7f7f", label="SAKT-ID cold-start (0.699)")
ax.axvspan(0, 5, color="#ffd9d9", alpha=0.5, zorder=0)
ax.text(1.0, ax.get_ylim()[0] + 0.004, "Eedi-like\ncoarse regime", fontsize=8, color="#a33")

ax.set_xscale("log")
ax.set_xlabel("Semantic resolution: distinct item embeddings (% of 7652 questions)")
ax.set_ylabel("AUC")
ax.set_title("Item-semantic resolution drives KT accuracy (XES3G5M)")
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(loc="lower right", fontsize=9)
for x, y, k in zip(uniq, auc[:, 0], ks):
    ax.annotate(f"K={k}", (x, y), textcoords="offset points", xytext=(0, 7),
                fontsize=7, ha="center", color="#1f77b4")
fig.tight_layout()
out = os.path.join(ROOT, "resolution_curve.png")
fig.savefig(out, dpi=180)
print(f"saved {out}")
