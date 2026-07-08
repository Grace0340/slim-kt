"""Aggregate XES3G5M matrix results into a comparison table."""
import glob
import json
import os

root = os.environ.get("OUTPUT_ROOT", "/root/autodl-tmp/slim-kt/runs")
order = ["id_cs", "sem_cs", "slim_attr", "slim_opt", "slim_full"]

rows = []
for name in order:
    d = os.path.join(root, f"xes3g5m_sakt_fold0_{name}")
    ev = os.path.join(d, "eval_metrics.json")
    if not os.path.exists(ev):
        continue
    m = json.load(open(ev))
    eff = m.get("efficiency", {}) or {}
    rows.append((
        name,
        m.get("auc"), m.get("acc"), m.get("rmse"),
        m.get("cold_auc"),
        eff.get("latency_ms_per_interaction"),
        eff.get("peak_gpu_mem_mb"),
    ))

hdr = f"{'variant':<11} {'AUC':>7} {'ACC':>7} {'RMSE':>7} {'coldAUC':>8} {'lat_ms/it':>10} {'peakMB':>8}"
print(hdr)
print("-" * len(hdr))
def f(x, p=4):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else "-"
for r in rows:
    name, auc, acc, rmse, cauc, lat, mem = r
    print(f"{name:<11} {f(auc):>7} {f(acc):>7} {f(rmse):>7} {f(cauc):>8} {f(lat,3):>10} {f(mem,1):>8}")
