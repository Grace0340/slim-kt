"""Shared helpers for the preprocessing converters.

Output contract (consumed by slimkt.data.datasets):
  <out>/interactions.csv   uid, order, question_id, kc_id, correct[, option_id]
  <out>/items.csv          question_id, text[, option_0..option_k][, kc_id]
  <out>/splits/fold{k}.json {"train":[uid...], "valid":[...], "test":[...]}
  <out>/preprocess_report.json   provenance + basic statistics
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def resolve_out(name: str, out_arg: Optional[str]) -> str:
    if out_arg:
        out = out_arg
    else:
        root = os.environ.get("DATA_ROOT", os.path.join(os.getcwd(), "slimkt_data"))
        out = os.path.join(root, name)
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "splits"), exist_ok=True)
    return out


def write_interactions(df: pd.DataFrame, out: str) -> None:
    cols = ["uid", "order", "question_id", "kc_id", "correct"]
    if "option_id" in df.columns:
        cols.append("option_id")
    df = df[cols].copy()
    df["uid"] = df["uid"].astype(np.int64)
    df["question_id"] = df["question_id"].astype(np.int64)
    df["kc_id"] = df["kc_id"].fillna(-1).astype(np.int64)
    df["correct"] = df["correct"].astype(np.int64).clip(0, 1)
    if "option_id" in df.columns:
        df["option_id"] = df["option_id"].fillna(-1).astype(np.int64)
    df.to_csv(os.path.join(out, "interactions.csv"), index=False)


def write_items(items: pd.DataFrame, out: str) -> None:
    if "question_id" not in items.columns:
        raise ValueError("items must contain question_id")
    items.to_csv(os.path.join(out, "items.csv"), index=False)


def write_folds(uids: List[int], out: str, num_folds: int = 5,
                valid_frac: float = 0.1, seed: int = 42) -> None:
    """Learner-level K-fold CV. fold k: test = group k, valid = valid_frac of the
    rest, train = remainder. Reproducible under `seed`."""
    rng = np.random.default_rng(seed)
    uids = np.array(sorted(set(int(u) for u in uids)))
    rng.shuffle(uids)
    groups = np.array_split(uids, num_folds)
    for k in range(num_folds):
        test = groups[k]
        rest = np.concatenate([groups[j] for j in range(num_folds) if j != k])
        rng.shuffle(rest)
        n_valid = int(round(valid_frac * len(rest)))
        valid, train = rest[:n_valid], rest[n_valid:]
        with open(os.path.join(out, "splits", f"fold{k}.json"), "w", encoding="utf-8") as f:
            json.dump({"train": train.tolist(), "valid": valid.tolist(),
                       "test": test.tolist()}, f)


def write_report(out: str, report: Dict) -> None:
    with open(os.path.join(out, "preprocess_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def basic_stats(df: pd.DataFrame) -> Dict:
    return {
        "interactions": int(len(df)),
        "learners": int(df["uid"].nunique()),
        "questions": int(df["question_id"].nunique()),
        "kcs": int(df["kc_id"].nunique()),
        "mean_correct": float(df["correct"].mean()),
        "has_options": bool("option_id" in df.columns),
    }
