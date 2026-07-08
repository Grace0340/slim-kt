"""Aggregate run outputs into Table~S1 rows (CSV + optional LaTeX snippet).

Reads eval_metrics.json under OUTPUT_ROOT, bootstrap JSON, option_recovery JSON.

Usage:
  OUTPUT_ROOT=./runs python scripts/export_table_s1.py --out ../overleaf/tables/table_s1_rows.tex
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# experiment registry: tag pattern -> display name
CORE_VARIANTS = {
    "id_cs": "SAKT-ID",
    "sem_cs": "Text-Embed-SAKT (SLIM-KT sem)",
    "sem_akt": "AKT + semantic encoder",
    "slim_attr": "SLIM-KT + attr",
    "slim_opt": "SLIM-KT + opt",
    "slim_full": "SLIM-KT + attr + opt",
    "dkt": "DKT",
    "dkvmn": "DKVMN",
    "akt": "AKT",
}

DATASETS = ["xes3g5m", "dbe_kt22", "eedi", "synth"]
core_re_tpl = r"{ds}_sakt_fold0_{variant}_s(\d+)$"


def _load_json(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def _mean_std(vals):
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return "—", "—", 0
    if a.size == 1:
        return f"{a.mean():.4f}", "—", 1
    return f"{a.mean():.4f}$\\pm${a.std():.4f}", f"{a.std():.4f}", a.size


def collect_core(root: str) -> list[dict]:
    rows = []
    for ds in DATASETS:
        for variant, name in CORE_VARIANTS.items():
            # sem_akt lives under {ds}_akt_fold0_ (backbone=akt); others under _sakt_fold0_
            bb = "akt" if variant == "sem_akt" else "sakt"
            prefix = f"{ds}_{bb}_fold0"
            pat = re.compile(rf"{ds}_{bb}_fold0_{variant}_s(\d+)$")
            aucs, colds = [], []
            for d in glob.glob(os.path.join(root, f"{prefix}_{variant}_s*")):
                base = os.path.basename(d)
                if not pat.match(base):
                    continue
                met = _load_json(os.path.join(d, "eval_metrics.json"))
                if met:
                    aucs.append(met.get("auc"))
                    if met.get("cold_auc") is not None:
                        colds.append(met["cold_auc"])
            if not aucs:
                continue
            auc_s, _, n = _mean_std(aucs)
            cold_s, _, _ = _mean_std(colds)
            rows.append({
                "experiment": name,
                "dataset": ds,
                "seeds": n,
                "auc": auc_s.replace("$\\pm$", "±"),
                "cold_auc": cold_s.replace("$\\pm$", "±"),
                "status": "Done" if n >= 3 else ("Partial" if n else "TBD"),
            })
    return rows


def collect_bootstrap(root: str) -> list[dict]:
    rows = []
    for path in glob.glob(os.path.join(root, "*_bootstrap_*.json")) + glob.glob(os.path.join(root, "*_bootstrap.json")):
        data = _load_json(path)
        if not data:
            continue
        ds = data.get("dataset", "?")
        metric = data.get("metric", "?")
        # aggregate across seeds: mean delta, max p-value (most conservative)
        deltas = [s.get("delta") for s in data.get("seeds", {}).values() if s.get("delta") is not None]
        pvals = [s.get("p_value") for s in data.get("seeds", {}).values() if s.get("p_value") is not None]
        if not deltas:
            continue
        mean_delta = float(np.mean(deltas))
        max_p = float(np.max(pvals)) if pvals else float("nan")
        rows.append({
            "experiment": f"Bootstrap sem vs ID ({metric}, {len(deltas)} seeds)",
            "dataset": ds,
            "seeds": len(deltas),
            "auc": f"$\\Delta={mean_delta:.4f}$",
            "cold_auc": f"$p\\le{max_p:.3f}$",
            "status": "Done",
        })
    return rows


def collect_option_recovery(root: str) -> list[dict]:
    rows = []
    for path in glob.glob(os.path.join(root, "**/option_recovery.json"), recursive=True):
        data = _load_json(path)
        if not data:
            continue
        ds = os.path.basename(os.path.dirname(path))
        rate = data.get("recovery_rate", float("nan"))
        n = data.get("evaluated_items", 0)
        rows.append({
            "experiment": "Option adequacy = answer key",
            "dataset": ds,
            "seeds": "—",
            "auc": f"{100*rate:.1f}\\%",
            "cold_auc": f"$n={n}$",
            "status": "Done",
        })
    return rows


def template_rows() -> list[dict]:
    """Placeholder rows for experiments not yet run locally."""
    pending = [
        ("LOKT (LLM-in-the-loop)", "xes3g5m", "Same cold-start split"),
        ("Attribute-aware LLM-KT (external baseline)", "xes3g5m", "Same cold-start split"),
        ("Fold~1 replication", "xes3g5m", "5-fold CV extension"),
        ("Extended seeds ($n=5$)", "dbe_kt22", "Variance reduction"),
    ]
    return [{
        "experiment": exp,
        "dataset": ds,
        "seeds": "TBD",
        "auc": "—",
        "cold_auc": "—",
        "status": f"TBD ({note})",
    } for exp, ds, note in pending]


def to_latex(rows: list[dict]) -> str:
    lines = [
        "% Auto-generated by scripts/export_table_s1.py — do not hand-edit rows marked AUTO",
        "\\begin{tabular}{llcccl}",
        "\\toprule",
        "Experiment & Dataset & Seeds & AUC & Cold-start AUC & Status \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['experiment']} & {r['dataset']} & {r['seeds']} & {r['auc']} & "
            f"{r['cold_auc']} & {r['status']} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", os.path.join(_REPO, "runs")))
    ap.add_argument("--csv", default=os.path.join(_REPO, "results", "table_s1.csv"))
    ap.add_argument("--out", default=None, help="LaTeX fragment path")
    ap.add_argument("--include-template", action="store_true",
                    help="append TBD placeholder rows for LLM baselines etc.")
    args = ap.parse_args()

    rows = collect_core(args.output_root)
    rows += collect_bootstrap(args.output_root)
    rows += collect_option_recovery(args.output_root)
    if args.include_template:
        done_keys = {(r["experiment"], r["dataset"]) for r in rows}
        for t in template_rows():
            if (t["experiment"], t["dataset"]) not in done_keys:
                rows.append(t)

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    import csv

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "dataset", "seeds", "auc", "cold_auc", "status"])
        w.writeheader()
        w.writerows(rows)
    print(f"[export] CSV -> {args.csv} ({len(rows)} rows)")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(to_latex(rows))
        print(f"[export] LaTeX -> {args.out}")


if __name__ == "__main__":
    main()
