"""Eedi (NeurIPS 2020 Education Challenge / Diagnostic Questions) -> SLIM-KT CSVs.

Expected raw files (Tasks 1 & 2 release). Point --raw at the extracted folder;
the script searches recursively for the standard filenames:

  train_task_1_2.csv              QuestionId, UserId, AnswerId, IsCorrect, CorrectAnswer, AnswerValue
  question_metadata_task_1_2.csv  QuestionId, SubjectId          (SubjectId = "[3, 71, 197]")
  subject_metadata.csv            SubjectId, Name, ParentId, Level
  answer_metadata_task_1_2.csv    AnswerId, DateAnswered, ...    (optional, gives real timestamps)

Item "text" note: Eedi questions are IMAGES, so there is no stem text in the
release. We build a semantic proxy from the subject/construct-name hierarchy
(general -> specific), which is what the LLM teacher embeds. If you later supply
real stems (e.g. via OCR) in a CSV `QuestionId,text`, pass --question-text to use it.

Usage:
  python -m slimkt.data.preprocess.eedi --raw /path/to/eedi --out ./slimkt_data/eedi
"""
from __future__ import annotations

import argparse
import ast
import glob
import os
from typing import Dict, List, Optional

import pandas as pd

from . import common


def _find(raw: str, filename: str) -> Optional[str]:
    hits = glob.glob(os.path.join(raw, "**", filename), recursive=True)
    return hits[0] if hits else None


def _parse_subject_ids(cell) -> List[int]:
    if isinstance(cell, list):
        return [int(x) for x in cell]
    try:
        return [int(x) for x in ast.literal_eval(str(cell))]
    except (ValueError, SyntaxError):
        return []


def build_subject_text(subject_ids: List[int], subj: pd.DataFrame) -> tuple[str, int]:
    """Return (hierarchy text, representative kc id = deepest subject)."""
    rows = subj[subj["SubjectId"].isin(subject_ids)]
    if rows.empty:
        return "", -1
    rows = rows.sort_values("Level")
    names = [str(n) for n in rows["Name"].tolist() if isinstance(n, str)]
    text = "; ".join(names)
    deepest = rows.iloc[rows["Level"].argmax()]
    return text, int(deepest["SubjectId"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess Eedi into SLIM-KT CSVs.")
    ap.add_argument("--raw", required=True, help="root folder of the extracted Eedi dataset")
    ap.add_argument("--out", default=None)
    ap.add_argument("--question-text", default=None,
                    help="optional CSV QuestionId,text with real stems (overrides proxy)")
    ap.add_argument("--num-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-users", type=int, default=None, help="cap learners (debug)")
    args = ap.parse_args()

    ans_path = _find(args.raw, "train_task_1_2.csv")
    qmeta_path = _find(args.raw, "question_metadata_task_1_2.csv")
    subj_path = _find(args.raw, "subject_metadata.csv")
    amet_path = _find(args.raw, "answer_metadata_task_1_2.csv")
    if not (ans_path and qmeta_path and subj_path):
        raise FileNotFoundError(
            "Could not locate train_task_1_2.csv / question_metadata_task_1_2.csv / "
            "subject_metadata.csv under --raw. Check the path.")

    out = common.resolve_out("eedi", args.out)
    print(f"[eedi] answers={ans_path}\n[eedi] out={out}")

    ans = pd.read_csv(ans_path)
    if args.max_users:
        keep = ans["UserId"].drop_duplicates().head(args.max_users)
        ans = ans[ans["UserId"].isin(keep)]

    # ordering: prefer real DateAnswered from answer metadata, else AnswerId
    if amet_path:
        amet = pd.read_csv(amet_path, usecols=["AnswerId", "DateAnswered"])
        ans = ans.merge(amet, on="AnswerId", how="left")
        dt = pd.to_datetime(ans["DateAnswered"], errors="coerce")
        # real timestamp where available, else fall back to AnswerId (monotonic)
        ans["order"] = dt.astype("int64").where(dt.notna(), ans["AnswerId"])
    else:
        ans["order"] = ans["AnswerId"]

    inter = pd.DataFrame({
        "uid": ans["UserId"],
        "order": ans["order"],
        "question_id": ans["QuestionId"],
        "correct": ans["IsCorrect"],
        "option_id": ans["AnswerValue"].astype("Int64") - 1,  # 1..4 -> 0..3
    })

    # ---- items + KC from subject hierarchy ----
    qmeta = pd.read_csv(qmeta_path)
    subj = pd.read_csv(subj_path)
    subj["Level"] = pd.to_numeric(subj["Level"], errors="coerce").fillna(0).astype(int)

    text_map: Dict[int, str] = {}
    kc_map: Dict[int, int] = {}
    for _, row in qmeta.iterrows():
        qid = int(row["QuestionId"])
        sids = _parse_subject_ids(row["SubjectId"])
        txt, kc = build_subject_text(sids, subj)
        text_map[qid] = txt
        kc_map[qid] = kc

    if args.question_text and os.path.exists(args.question_text):
        real = pd.read_csv(args.question_text)
        text_map.update({int(r.QuestionId): str(r.text) for r in real.itertuples()})

    inter["kc_id"] = inter["question_id"].map(kc_map).fillna(-1).astype("int64")

    items = pd.DataFrame({"question_id": sorted(text_map.keys())})
    items["text"] = items["question_id"].map(text_map)
    items["kc_id"] = items["question_id"].map(kc_map).fillna(-1).astype("int64")
    # Eedi is 4-option MCQ but option strings are in images -> placeholders.
    for j in range(4):
        items[f"option_{j}"] = f"Option {chr(ord('A') + j)}"

    common.write_interactions(inter, out)
    common.write_items(items, out)
    common.write_folds(inter["uid"].tolist(), out, args.num_folds, seed=args.seed)
    stats = common.basic_stats(inter)
    common.write_report(out, {
        "dataset": "eedi", "source": {"answers": ans_path, "question_meta": qmeta_path,
                                       "subject_meta": subj_path, "answer_meta": amet_path},
        "item_text": "subject-hierarchy proxy (stems are images)"
                     if not args.question_text else "real stems provided",
        "num_options": 4, **stats,
    })
    print(f"[eedi] done: {stats}")


if __name__ == "__main__":
    main()
