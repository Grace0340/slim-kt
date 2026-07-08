"""Paired bootstrap significance test: SLIM-KT (sem_cs) vs SAKT-ID (id_cs).

Compares cold-start (or overall) AUC on the *same* test interactions by
resampling interaction indices. Requires per-run checkpoints for matched seeds.

Usage:
  OUTPUT_ROOT=./runs DS=eedi_sakt_fold0 SEEDS="42" \\
    python scripts/bootstrap_significance.py

Writes:
  <OUTPUT_ROOT>/<DS>_bootstrap.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

# allow `python scripts/bootstrap_significance.py` from repo root
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Fast rank-based AUC (Mann-Whitney U), handles ties via average ranks."""
    n = y.shape[0]
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    sp = p[order]
    # average ranks for ties
    ranks_sorted = np.arange(1, n + 1, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sp[j] == sp[i]:
            j += 1
        if j - i > 1:
            ranks_sorted[i:j] = (i + 1 + j) / 2.0
        i = j
    ranks[order] = ranks_sorted
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# per-variant model overrides so the rebuilt architecture matches the checkpoint
_TAG_OVERRIDES = {
    "sem_cs": ["model.use_semantic=true", "model.use_id_embedding=false",
               "model.lambda_attr=0", "model.lambda_opt=0"],
    "id_cs": ["model.use_semantic=false", "model.use_id_embedding=true",
              "model.lambda_attr=0", "model.lambda_opt=0"],
    "sem_akt": ["model.arch=slimkt", "model.backbone=akt", "model.use_semantic=true",
                "model.use_id_embedding=false", "model.lambda_attr=0", "model.lambda_opt=0"],
}


def _collect_probs(cfg_path, ds_cfg, dataset, seed, tag, ckpt_path, device):
    import torch

    from slimkt.config import load_config
    from slimkt.data.datasets import build_dataloaders
    from slimkt.models.factory import build_model
    from slimkt.teacher.llm_teacher import TeacherCache
    from slimkt.utils import set_seed

    overrides = [f"dataset.name={dataset}", f"seed={seed}"] + _TAG_OVERRIDES.get(tag, [])
    cfg = load_config(cfg_path, ds_cfg, overrides)
    set_seed(seed)
    _, _, test_dl, stats = build_dataloaders(cfg)
    cfg.model["num_kcs"] = stats.num_kcs
    cache = TeacherCache.load(os.path.join(cfg.paths.teacher_cache, cfg.dataset.name))
    teacher_sem = torch.as_tensor(cache.sem, dtype=torch.float32)
    model = build_model(cfg, stats, teacher_sem).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()

    ys, ps, colds = [], [], []
    with torch.no_grad():
        for batch in test_dl:
            b = {k: v.to(device) for k, v in batch.items()}
            prob = torch.sigmoid(model(b)["logit"]).cpu().numpy()
            mask = batch["mask"].numpy().astype(bool)
            ys.append(batch["r"].numpy()[mask])
            ps.append(prob[mask])
            colds.append(batch["cold"].numpy().astype(bool)[mask])
    y = np.concatenate(ys)
    pr = np.concatenate(ps)
    cold = np.concatenate(colds)
    return y, pr, cold


def paired_bootstrap(y, p_a, p_b, cold, n_boot=2000, seed=0, cold_only=True, max_n=200000):
    """Return observed delta (a-b), 95% CI, one-sided p-value (H0: delta<=0).

    Observed AUCs use the full array; the bootstrap CI/p resamples (subsampled to
    ``max_n`` for tractability on large datasets)."""
    rng = np.random.default_rng(seed)
    idx = np.where(cold)[0] if cold_only else np.arange(len(y))
    if idx.size == 0 or int(y[idx].sum()) in (0, idx.size):
        return {"n": int(idx.size), "delta": float("nan"),
                "ci95": [float("nan"), float("nan")], "p_value": float("nan")}

    ya_full, pa_full, pb_full = y[idx], p_a[idx], p_b[idx]
    obs = _auc(ya_full, pa_full) - _auc(ya_full, pb_full)

    # subsample for the resampling loop if the array is very large
    if idx.size > max_n:
        sub = rng.choice(idx.size, size=max_n, replace=False)
        ya, pa, pb = ya_full[sub], pa_full[sub], pb_full[sub]
    else:
        ya, pa, pb = ya_full, pa_full, pb_full

    n = ya.shape[0]
    deltas = np.empty(n_boot, dtype=np.float64)
    k = 0
    for _ in range(n_boot):
        samp = rng.integers(0, n, size=n)
        ys = ya[samp]
        if int(ys.sum()) in (0, n):
            continue
        deltas[k] = _auc(ys, pa[samp]) - _auc(ys, pb[samp])
        k += 1
    deltas = deltas[:k]
    if deltas.size == 0:
        return {"n": int(idx.size), "delta": float(obs),
                "ci95": [float("nan"), float("nan")], "p_value": float("nan")}
    p_val = float((deltas <= 0).mean())  # one-sided: sem better if delta>0
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"n": int(idx.size), "boot_n": int(n), "delta": float(obs),
            "ci95": [float(lo), float(hi)], "p_value": p_val}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_REPO, "configs/default.yaml"))
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", os.path.join(_REPO, "runs")))
    ap.add_argument("--ds-prefix", default=os.environ.get("DS", "eedi_sakt_fold0"))
    ap.add_argument("--seeds", default=os.environ.get("SEEDS", "42 1 7"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--metric", choices=("cold", "overall"), default="cold")
    args = ap.parse_args()

    import torch

    from slimkt.utils import get_device

    device = get_device()
    seeds = [int(s) for s in args.seeds.split()]
    ds_cfg = args.dataset_config
    dataset = args.dataset
    if dataset is None:
        m = re.match(r"^(\w+)_sakt", args.ds_prefix)
        dataset = m.group(1) if m else "eedi"
    if ds_cfg is None:
        cand = os.path.join(_REPO, f"configs/dataset/{dataset}.yaml")
        ds_cfg = cand if os.path.exists(cand) else None

    per_seed = {}
    for seed in seeds:
        sem_dir = os.path.join(args.output_root, f"{args.ds_prefix}_sem_cs_s{seed}")
        id_dir = os.path.join(args.output_root, f"{args.ds_prefix}_id_cs_s{seed}")
        sem_ckpt = os.path.join(sem_dir, "best.pt")
        id_ckpt = os.path.join(id_dir, "best.pt")
        if not (os.path.exists(sem_ckpt) and os.path.exists(id_ckpt)):
            print(f"[skip] seed={seed}: missing checkpoint(s)")
            continue
        y, p_sem, cold = _collect_probs(args.config, ds_cfg, dataset, seed, "sem_cs", sem_ckpt, device)
        _, p_id, _ = _collect_probs(args.config, ds_cfg, dataset, seed, "id_cs", id_ckpt, device)
        cold_only = args.metric == "cold"
        stat = paired_bootstrap(y, p_sem, p_id, cold, n_boot=args.n_boot, seed=seed, cold_only=cold_only)
        stat["sem_auc"] = _auc(y[cold], p_sem[cold]) if cold_only else _auc(y, p_sem)
        stat["id_auc"] = _auc(y[cold], p_id[cold]) if cold_only else _auc(y, p_id)
        per_seed[str(seed)] = stat
        print(f"seed={seed} delta={stat['delta']:.4f} p={stat['p_value']:.4f} n={stat['n']}")

    out = {
        "dataset": dataset,
        "ds_prefix": args.ds_prefix,
        "metric": args.metric,
        "comparison": "sem_cs vs id_cs",
        "seeds": per_seed,
    }
    out_path = os.path.join(args.output_root, f"{args.ds_prefix}_bootstrap_{args.metric}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[bootstrap] wrote {out_path}")


if __name__ == "__main__":
    main()
