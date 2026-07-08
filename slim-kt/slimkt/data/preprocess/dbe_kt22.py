"""DBE-KT22 (ANU database-course exercises) -> SLIM-KT CSVs.

A small but genuinely TEXT-RICH knowledge-tracing dataset: every question has a
distinct natural-language stem (``question_text``), real answer choices, and
multi-KC tags. This is the regime where distilled item semantics should help most
(near 1:1 question<->embedding), complementing XES3G5M and contrasting with Eedi
(whose "semantics" are only coarse subject labels).

Raw files (HuggingFace ``Unggi/dbe-kt22_raw_data``):
  Transaction.csv               student_id, question_id, answer_state (bool=correct),
                                answer_choice_id, start_time, ...
  Questions.csv                 id, question_text (clean), question_title, difficulty, ...
  Question_Choices.csv          id, choice_text, is_correct, question_id
  KCs.csv                       id, name, description
  Question_KC_Relationships.csv question_id, knowledgecomponent_id  (multi-KC)

Usage:
  python -m slimkt.data.preprocess.dbe_kt22 --raw /root/autodl-tmp/raw/dbe_kt22 \
    --out $DATA_ROOT/dbe_kt22
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import pandas as pd

from . import common


def _clean(s) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.replace("\r", " ").replace("\n", " ").split()).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess DBE-KT22 into SLIM-KT CSVs.")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--num-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = common.resolve_out("dbe_kt22", args.out)
    tx = pd.read_csv(os.path.join(args.raw, "Transaction.csv"))
    q = pd.read_csv(os.path.join(args.raw, "Questions.csv"))
    ch = pd.read_csv(os.path.join(args.raw, "Question_Choices.csv"))
    qkc = pd.read_csv(os.path.join(args.raw, "Question_KC_Relationships.csv"))
    print(f"[dbe_kt22] tx={tx.shape} questions={q.shape} choices={ch.shape} out={out}")

    # ---- ordering: real timestamp, fall back to transaction id ----
    dt = pd.to_datetime(tx["start_time"], errors="coerce", utc=True)
    order = dt.astype("int64").where(dt.notna(), tx["id"])

    # ---- representative KC per question (first tag) + within-question choice ordinal ----
    kc_map: Dict[int, int] = (
        qkc.sort_values("id").groupby("question_id")["knowledgecomponent_id"].first().to_dict())
    choice_ord: Dict[int, int] = {}
    opt_map: Dict[int, List[str]] = {}
    max_opt = 0
    for qid, g in ch.sort_values("id").groupby("question_id"):
        texts = [_clean(t) for t in g["choice_text"].tolist()]
        opt_map[int(qid)] = texts
        max_opt = max(max_opt, len(texts))
        for j, cid in enumerate(g["id"].tolist()):
            choice_ord[int(cid)] = j

    inter = pd.DataFrame({
        "uid": tx["student_id"].astype("int64"),
        "order": order.astype("int64"),
        "question_id": tx["question_id"].astype("int64"),
        "correct": tx["answer_state"].astype(bool).astype("int64"),
        "option_id": tx["answer_choice_id"].map(choice_ord).astype("Int64"),
    })
    inter["kc_id"] = inter["question_id"].map(kc_map).fillna(-1).astype("int64")

    # ---- items.csv: distinct question text + choices + representative KC ----
    text_map: Dict[int, str] = {}
    for _, row in q.iterrows():
        t = _clean(row.get("question_text"))
        if not t:
            t = _clean(row.get("question_title"))
        text_map[int(row["id"])] = t

    qids = sorted(set(inter["question_id"].unique()) | set(text_map.keys()))
    items = pd.DataFrame({"question_id": qids})
    items["text"] = items["question_id"].map(lambda x: text_map.get(int(x), ""))
    items["kc_id"] = items["question_id"].map(kc_map).fillna(-1).astype("int64")
    for j in range(max_opt):
        items[f"option_{j}"] = items["question_id"].map(
            lambda x: opt_map.get(int(x), [None] * max_opt)[j]
            if j < len(opt_map.get(int(x), [])) else None)

    common.write_interactions(inter, out)
    common.write_items(items, out)
    common.write_folds(inter["uid"].tolist(), out, args.num_folds, seed=args.seed)

    stats = common.basic_stats(inter)
    n_text = int(sum(1 for t in items["text"] if isinstance(t, str) and t))
    common.write_report(out, {
        "dataset": "dbe_kt22",
        "source": "HuggingFace: Unggi/dbe-kt22_raw_data (ADA Dataverse mirror)",
        "item_text": "real natural-language question stems",
        "questions_with_text": n_text,
        "num_options": max_opt,
        **stats,
    })
    print(f"[dbe_kt22] done: {stats} | questions_with_text={n_text} num_options={max_opt}")


if __name__ == "__main__":
    main()
