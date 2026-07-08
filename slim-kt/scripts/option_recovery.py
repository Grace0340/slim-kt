"""Option-label recovery: fraction of items where the teacher's ``adequate``
label lands on the answer-key option.

Alignment is text-based to avoid question_id vs.\ contiguous-index confusion:
teacher_options.npy is indexed by the *contiguous question index* (row r of
items.csv -> q_idx = r+1, PAD = 0), and each option slot j corresponds to
column ``option_j`` of items.csv (the same order the LLM was prompted with). The
answer key comes from the raw Question_Choices.csv ``is_correct`` flag, matched
to items.csv option columns by normalized text.

Usage (DBE-KT22):
  python scripts/option_recovery.py --cache data/teacher_cache/dbe_kt22 \\
      --items data/dbe_kt22/items.csv --raw /root/autodl-tmp/raw/dbe_kt22
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def _norm(s) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.replace("\r", " ").replace("\n", " ").split()).strip().lower()


def _correct_text_by_qid(raw_dir: str) -> dict[int, str]:
    ch = pd.read_csv(os.path.join(raw_dir, "Question_Choices.csv"))
    out: dict[int, str] = {}
    for _, row in ch.iterrows():
        if bool(row.get("is_correct", False)):
            out[int(row["question_id"])] = _norm(row["choice_text"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--items", required=True, help="items.csv (row order == q_idx)")
    ap.add_argument("--raw", required=True, help="dir with Question_Choices.csv")
    ap.add_argument("--adequate-label", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    opt = np.load(os.path.join(args.cache, "teacher_options.npy"))  # [num_q, num_opt], -1 missing
    top_label = args.adequate_label
    if top_label is None:
        valid = opt[opt >= 0]
        top_label = int(valid.max()) if valid.size else 3

    items = pd.read_csv(args.items)
    opt_cols = [c for c in items.columns if c.startswith("option_")]
    correct_text = _correct_text_by_qid(args.raw)

    adequate_hits, argmax_hits, total, skipped = 0, 0, 0, 0
    rows = []
    for r, item in items.iterrows():
        q_idx = r + 1  # PAD = 0
        qid = int(item["question_id"])
        if qid not in correct_text or q_idx >= opt.shape[0]:
            skipped += 1
            continue
        texts = [_norm(item[c]) for c in opt_cols]
        # locate the answer-key slot by text match
        corr_slot = next((j for j, t in enumerate(texts) if t and t == correct_text[qid]), None)
        if corr_slot is None:
            skipped += 1
            continue
        row = opt[q_idx]
        if (row < 0).all():
            skipped += 1
            continue
        # restrict argmax to slots that actually have an option / a label
        valid_slots = [j for j in range(len(texts)) if texts[j] and row[j] >= 0]
        if not valid_slots:
            skipped += 1
            continue
        pred_slot = max(valid_slots, key=lambda j: row[j])
        adequate = int(row[corr_slot]) == top_label
        argmax_ok = pred_slot == corr_slot
        adequate_hits += int(adequate)
        argmax_hits += int(argmax_ok)
        total += 1
        rows.append({"q_idx": int(q_idx), "question_id": qid, "correct_slot": corr_slot,
                     "pred_slot": pred_slot, "adequate_on_correct": adequate})

    result = {
        "cache": args.cache,
        "adequate_label": top_label,
        "evaluated_items": total,
        "skipped_items": skipped,
        "adequate_on_correct_rate": adequate_hits / total if total else float("nan"),
        "argmax_match_rate": argmax_hits / total if total else float("nan"),
        "adequate_on_correct_n": adequate_hits,
        "argmax_match_n": argmax_hits,
    }
    out = args.out or os.path.join(args.cache, "option_recovery.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[option_recovery] wrote {out}")


if __name__ == "__main__":
    main()
