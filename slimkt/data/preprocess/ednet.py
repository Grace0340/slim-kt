"""EdNet-KT1 -> SLIM-KT CSVs.

Expected raw layout (EdNet-KT1 release):
  <raw>/u*.csv                    per-learner logs: timestamp, solving_id, question_id, user_answer, elapsed_time
  <questions>/questions.csv       question_id, bundle_id, explanation_id, correct_answer, part, tags

Correctness is not stored per interaction in KT1; it is computed by joining the
learner's `user_answer` (a-d) with the question's `correct_answer`.

Item "text" note: EdNet does not release question stems (copyright). We build a
semantic proxy from `part` (TOEIC section) + skill `tags`. Provide --tag-names
(CSV: tag_id,name) to make the proxy human-readable; otherwise tag ids are used.
If you have real stems, pass --question-text (CSV: question_id,text).

Usage:
  python -m slimkt.data.preprocess.ednet \
      --raw /path/to/EdNet-KT1 --questions /path/to/contents/questions.csv \
      --out ./slimkt_data/ednet --max-users 20000 --min-interactions 5
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional

import pandas as pd

from . import common

_ANS2IDX = {"a": 0, "b": 1, "c": 2, "d": 3}


def _qid_to_int(q) -> int:
    """Parse ids like 'q7900', 'u12345', 'b55' or plain numbers -> int."""
    s = str(q).strip()
    i = 0
    while i < len(s) and not s[i].isdigit():
        i += 1
    return int(s[i:]) if i < len(s) else int(float(s))


def _parse_tags(cell) -> List[int]:
    s = str(cell).strip()
    if not s or s in ("-1", "nan"):
        return []
    sep = ";" if ";" in s else (" " if " " in s else ",")
    out = []
    for tok in s.split(sep):
        tok = tok.strip()
        if tok and tok not in ("-1",):
            try:
                out.append(int(float(tok)))
            except ValueError:
                pass
    return out


def _iter_user_files(raw: str, max_users: Optional[int]):
    files = sorted(glob.glob(os.path.join(raw, "u*.csv")))
    if not files:  # maybe nested
        files = sorted(glob.glob(os.path.join(raw, "**", "u*.csv"), recursive=True))
    if max_users:
        files = files[:max_users]
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess EdNet-KT1 into SLIM-KT CSVs.")
    ap.add_argument("--raw", required=True, help="folder containing per-user u*.csv")
    ap.add_argument("--questions", required=True, help="contents/questions.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag-names", default=None, help="optional CSV tag_id,name")
    ap.add_argument("--question-text", default=None, help="optional CSV question_id,text")
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--min-interactions", type=int, default=5)
    ap.add_argument("--num-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = common.resolve_out("ednet", args.out)
    q = pd.read_csv(args.questions)
    q["qid"] = q["question_id"].map(_qid_to_int)
    q["correct_answer"] = q["correct_answer"].astype(str).str.lower().str[0]
    q["tag_list"] = q["tags"].map(_parse_tags)
    q["kc_id"] = q["tag_list"].map(lambda t: t[0] if t else -1)
    correct_by_q: Dict[int, str] = dict(zip(q["qid"], q["correct_answer"]))
    kc_by_q: Dict[int, int] = dict(zip(q["qid"], q["kc_id"]))

    files = _iter_user_files(args.raw, args.max_users)
    if not files:
        raise FileNotFoundError(f"No u*.csv found under {args.raw}")
    print(f"[ednet] {len(files)} user files; questions={args.questions}\n[ednet] out={out}")

    frames = []
    for i, fp in enumerate(files):
        uid = _qid_to_int(os.path.splitext(os.path.basename(fp))[0])  # u12345 -> 12345
        try:
            d = pd.read_csv(fp, usecols=["timestamp", "question_id", "user_answer"])
        except Exception:  # noqa: BLE001
            continue
        if len(d) < args.min_interactions:
            continue
        d["uid"] = uid
        d["qid"] = d["question_id"].map(_qid_to_int)
        ans = d["user_answer"].astype(str).str.lower().str[0]
        d["correct"] = [int(a == correct_by_q.get(qq, "")) for a, qq in zip(ans, d["qid"])]
        d["option_id"] = ans.map(_ANS2IDX).fillna(-1).astype("Int64")
        d["kc_id"] = d["qid"].map(kc_by_q).fillna(-1).astype("int64")
        frames.append(d[["uid", "timestamp", "qid", "kc_id", "correct", "option_id"]])
        if (i + 1) % 5000 == 0:
            print(f"[ednet] processed {i + 1}/{len(files)} users")

    inter = pd.concat(frames, ignore_index=True)
    inter = inter.rename(columns={"timestamp": "order", "qid": "question_id"})

    # ---- items text proxy ----
    tag_name: Dict[int, str] = {}
    if args.tag_names and os.path.exists(args.tag_names):
        tn = pd.read_csv(args.tag_names)
        tag_name = {int(r[0]): str(r[1]) for r in tn.itertuples(index=False)}

    def item_text(row) -> str:
        tags = row["tag_list"]
        tag_str = ", ".join(tag_name.get(t, f"tag{t}") for t in tags) if tags else "unspecified"
        return f"TOEIC Part {row['part']}. Skills: {tag_str}"

    q["text"] = q.apply(item_text, axis=1)
    if args.question_text and os.path.exists(args.question_text):
        real = pd.read_csv(args.question_text)
        rmap = {_qid_to_int(r.question_id): str(r.text) for r in real.itertuples()}
        q["text"] = q.apply(lambda r: rmap.get(r["qid"], r["text"]), axis=1)

    items = q[["qid", "text", "kc_id"]].rename(columns={"qid": "question_id"})
    for j in range(4):
        items[f"option_{j}"] = f"Option {chr(ord('A') + j)}"

    common.write_interactions(inter, out)
    common.write_items(items, out)
    common.write_folds(inter["uid"].tolist(), out, args.num_folds, seed=args.seed)
    stats = common.basic_stats(inter)
    common.write_report(out, {
        "dataset": "ednet-kt1",
        "source": {"raw": args.raw, "questions": args.questions},
        "users_processed": int(inter["uid"].nunique()),
        "item_text": "part+tags proxy (stems not released)"
                     if not args.question_text else "real stems provided",
        "num_options": 4, **stats,
    })
    print(f"[ednet] done: {stats}")


if __name__ == "__main__":
    main()
