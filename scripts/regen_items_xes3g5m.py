"""Regenerate ONLY items.csv for XES3G5M (text + option_* columns) from questions.json.

Rebuilding the full dataset re-expands 5.5M interactions and needs lots of RAM
(OOM in AutoDL's 2GB no-GPU mode). But items.csv depends only on questions.json
and the existing question-id list, so this lightweight script refreshes it in
place without touching interactions.csv / splits / precomputed_qid_emb.npy.

Usage:
  python scripts/regen_items_xes3g5m.py \
    --items /root/autodl-tmp/slim-kt/data/xes3g5m/items.csv \
    --questions /root/autodl-tmp/raw/xes3g5m/metadata/questions.json
"""
from __future__ import annotations

import argparse

import pandas as pd

from slimkt.data.preprocess.xes3g5m import _item_text, _load_questions, _options_to_list


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="existing items.csv to overwrite")
    ap.add_argument("--questions", required=True, help="metadata/questions.json")
    args = ap.parse_args()

    qids = pd.read_csv(args.items, usecols=["question_id"])["question_id"].tolist()
    questions = _load_questions(args.questions)

    items = pd.DataFrame({"question_id": qids})
    items["text"] = items["question_id"].map(lambda q: _item_text(questions.get(int(q), {})))

    max_opt = 0
    opt_map = {}
    for q in qids:
        opts = _options_to_list(questions.get(int(q), {}).get("options"))
        if opts:
            opt_map[int(q)] = opts
            max_opt = max(max_opt, len(opts))
    for j in range(max_opt):
        items[f"option_{j}"] = items["question_id"].map(
            lambda q: opt_map.get(int(q), [None] * max_opt)[j] if j < len(opt_map.get(int(q), [])) else None)

    items.to_csv(args.items, index=False)
    n_mc = sum(1 for v in opt_map.values() if v)
    print(f"[regen] wrote {args.items}: {len(items)} items, {max_opt} option slots, "
          f"{n_mc} multiple-choice questions with options")


if __name__ == "__main__":
    main()
