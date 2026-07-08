"""Validate a preprocessed dataset folder against the SLIM-KT data contract.

Run after preprocessing to catch problems before training / teacher extraction:
  python -m slimkt.data.preprocess.validate --dir ./slimkt_data/eedi
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="preprocessed dataset folder")
    args = ap.parse_args()
    d = args.dir

    problems, warnings = [], []

    inter_p = os.path.join(d, "interactions.csv")
    items_p = os.path.join(d, "items.csv")
    if not os.path.exists(inter_p):
        problems.append("interactions.csv missing")
    if not os.path.exists(items_p):
        problems.append("items.csv missing")
    if problems:
        _report(problems, warnings, {})
        return

    inter = pd.read_csv(inter_p)
    items = pd.read_csv(items_p)

    for c in ["uid", "order", "question_id", "kc_id", "correct"]:
        if c not in inter.columns:
            problems.append(f"interactions.csv missing column {c!r}")
    if "question_id" not in items.columns:
        problems.append("items.csv missing question_id")

    if not problems:
        if not set(inter["correct"].unique()).issubset({0, 1}):
            problems.append("correct must be 0/1")
        # item-text coverage (needed by the semantic teacher)
        q_seen = set(inter["question_id"].unique())
        q_items = set(items["question_id"].unique())
        missing_items = len(q_seen - q_items)
        if missing_items:
            warnings.append(f"{missing_items} questions in interactions have no items.csv row")
        if "text" in items.columns:
            empty_txt = int((items["text"].fillna("").str.len() == 0).sum())
            if empty_txt:
                warnings.append(f"{empty_txt} items have empty text (teacher embedding will be zero)")
        else:
            warnings.append("items.csv has no 'text' column -> L_sem/cold-start unavailable")
        if "option_id" in inter.columns:
            bad = int(((inter["option_id"] < -1)).sum())
            if bad:
                warnings.append(f"{bad} option_id < -1")
        # fold files
        folds = glob.glob(os.path.join(d, "splits", "fold*.json"))
        if not folds:
            warnings.append("no fold split files under splits/ (random split will be used)")

    stats = {
        "interactions": int(len(inter)),
        "learners": int(inter["uid"].nunique()),
        "questions_in_logs": int(inter["question_id"].nunique()),
        "questions_in_items": int(items["question_id"].nunique()),
        "kcs": int(inter["kc_id"].nunique()),
        "mean_correct": round(float(inter["correct"].mean()), 4),
        "has_options": bool("option_id" in inter.columns),
    }
    _report(problems, warnings, stats)


def _report(problems, warnings, stats):
    print("=== SLIM-KT dataset validation ===")
    print("stats:", json.dumps(stats, indent=2))
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print("  -", w)
    if problems:
        print("\nPROBLEMS (must fix):")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("\nOK: dataset conforms to the SLIM-KT contract.")


if __name__ == "__main__":
    main()
