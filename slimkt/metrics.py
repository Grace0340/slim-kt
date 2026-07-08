"""Predictive metrics + efficiency (latency / GPU memory) benchmarking."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error


def kt_metrics(y_true: np.ndarray, y_prob: np.ndarray,
               cold_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """AUC / ACC / RMSE overall and (optionally) on cold-start positions only."""
    out: Dict[str, float] = {}
    out.update(_binary_scores(y_true, y_prob, prefix=""))
    if cold_mask is not None and cold_mask.any():
        out.update(_binary_scores(y_true[cold_mask], y_prob[cold_mask], prefix="cold_"))
    return out


def _binary_scores(y_true, y_prob, prefix: str) -> Dict[str, float]:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {f"{prefix}auc": float("nan"),
                f"{prefix}acc": float("nan"),
                f"{prefix}rmse": float("nan")}
    return {
        f"{prefix}auc": float(roc_auc_score(y_true, y_prob)),
        f"{prefix}acc": float(accuracy_score(y_true, (y_prob >= 0.5).astype(int))),
        f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_prob))),
    }


def benchmark_efficiency(model, sample_batch, device: str,
                         warmup: int = 20, iters: int = 200) -> Dict[str, float]:
    """Measure per-batch inference latency and peak GPU memory (H1 evidence)."""
    import torch

    model.eval()
    batch = {k: v.to(device) for k, v in sample_batch.items()}
    with torch.no_grad():
        for _ in range(warmup):
            model(batch)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters

    bs = sample_batch["q"].size(0)
    res = {
        "latency_ms_per_batch": dt * 1e3,
        "latency_ms_per_interaction": dt * 1e3 / max(bs, 1),
        "throughput_batches_per_s": 1.0 / dt if dt > 0 else float("inf"),
    }
    if device == "cuda":
        res["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
    return res
