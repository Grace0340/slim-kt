"""XES3G5M (HuggingFace parquet) -> SLIM-KT CSVs + precomputed semantic embeddings.

Source layout produced by scripts/download_xes3g5m.py:
  <raw>/interaction_sequences/**/train-*.parquet   pyKT sequences, columns:
        fold, uid, questions, concepts, responses, timestamps, selectmasks, is_repeat
        (each of questions..is_repeat is a length-200 list; selectmasks == -1 marks
         padding / history-only positions)
  <raw>/interaction_sequences/**/test-*.parquet    official held-out test sequences
  <raw>/content_metadata/**/question-*.parquet     column `embeddings` (7652 x 768,
        row order == internal question id) -> authors' RoBERTa question semantics

The HF mirror has NO raw question text. Pass --questions /path/to/questions.json
(from the Drive package) to additionally fill items.csv text/options and thereby
enable the LLM attribute/option teacher. Without it, items.csv text is empty and
we rely on the precomputed embeddings (exported to precomputed_qid_emb.npy) for
the semantic / cold-start signal.

Splits follow pyKT: the official test set is shared across folds; the train/valid
CV fold rotates using the per-row `fold` id.

Usage:
  python -m slimkt.data.preprocess.xes3g5m --raw /root/autodl-tmp/raw/xes3g5m --out $DATA_ROOT/xes3g5m
  python -m slimkt.data.preprocess.xes3g5m --raw ... --questions .../questions.json --out ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import common

TEST_UID_OFFSET = 10_000_000  # keep official-test uids from colliding with train uids


def _find_parquets(raw: str, subdir: str, name_contains: str) -> List[str]:
    hits = sorted(glob.glob(os.path.join(raw, subdir, "**", "*.parquet"), recursive=True))
    return [h for h in hits if name_contains in os.path.basename(h).lower()]


def _to_list(cell) -> List:
    if isinstance(cell, (list, tuple, np.ndarray)):
        return list(cell)
    return str(cell).split(",")


def _expand(df: pd.DataFrame, uid_offset: int) -> List[dict]:
    """Expand pyKT sequence rows (list-valued columns) into per-interaction records.

    The HF `interaction_sequences` is KC-level: a multi-KC question is split into
    several consecutive positions, and `is_repeat==1` marks the repeated ones.
    We collapse back to QUESTION level by keeping only `is_repeat==0` positions
    (one row per question, tagged with its first leaf KC). This recovers the
    official ~5.55M question-level interaction count.
    """
    rows: List[dict] = []
    has_ts = "timestamps" in df.columns
    has_sm = "selectmasks" in df.columns
    has_ir = "is_repeat" in df.columns
    for i, row in df.iterrows():
        qs = _to_list(row["questions"])
        cs = _to_list(row["concepts"])
        rs = _to_list(row["responses"])
        ts = _to_list(row["timestamps"]) if has_ts else None
        sm = _to_list(row["selectmasks"]) if has_sm else None
        ir = _to_list(row["is_repeat"]) if has_ir else None
        uid = uid_offset + int(row["uid"]) if "uid" in df.columns else uid_offset + int(i)
        order = 0
        for j in range(len(qs)):
            try:
                q = int(qs[j]); resp = int(rs[j])
            except (ValueError, TypeError, IndexError):
                continue
            if q < 0 or resp < 0:
                continue
            if sm is not None and j < len(sm) and int(sm[j]) == -1:
                continue
            if ir is not None and j < len(ir) and int(ir[j]) == 1:
                continue  # repeated KC of a multi-KC question -> collapse to question level
            kc = int(cs[j]) if (j < len(cs) and int(cs[j]) >= 0) else -1
            t = int(ts[j]) if (ts is not None and j < len(ts) and int(ts[j]) >= 0) else order
            rows.append({"uid": uid, "order": t, "question_id": q, "kc_id": kc, "correct": resp})
            order += 1
    return rows


def _uid_fold_map(train_df: pd.DataFrame) -> Dict[int, int]:
    """Map uid -> its CV fold id (pyKT assigns one fold per learner)."""
    fold_map: Dict[int, int] = {}
    if "fold" not in train_df.columns:
        return fold_map
    for _, row in train_df[["uid", "fold"]].iterrows():
        fold_map.setdefault(int(row["uid"]), int(row["fold"]))
    return fold_map


def _load_embeddings(raw: str) -> Optional[np.ndarray]:
    files = _find_parquets(raw, "content_metadata", "question")
    if not files:
        print("[xes3g5m] WARNING: no question embedding parquet found; semantic cache will be empty.")
        return None
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    if "embeddings" not in df.columns:
        print(f"[xes3g5m] WARNING: 'embeddings' column missing (cols={list(df.columns)}).")
        return None
    emb = np.asarray([np.asarray(v, dtype=np.float32) for v in df["embeddings"].tolist()],
                     dtype=np.float32)
    print(f"[xes3g5m] loaded question embeddings: {emb.shape} (row idx == internal qid)")
    return emb


def _load_questions(path: Optional[str]) -> Dict[int, dict]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[int, dict] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = v
        except (ValueError, TypeError):
            continue
    print(f"[xes3g5m] loaded raw question text for {len(out)} questions")
    return out


def _options_to_list(opts) -> List[str]:
    """XES3G5M stores options as a dict {'A':..,'B':..} for multi-choice and an
    empty dict for fill-in-the-blank. Return an ordered (A,B,C,D,...) list."""
    if isinstance(opts, dict) and opts:
        return [str(opts[k]) for k in sorted(opts.keys())]
    if isinstance(opts, list) and opts:
        return [str(o) for o in opts]
    return []


def _item_text(q: dict) -> str:
    parts = [str(q.get("content", "") or "")]
    kcr = q.get("kc_routes")
    if kcr:
        parts.append("知识点: " + ("; ".join(map(str, kcr)) if isinstance(kcr, list) else str(kcr)))
    ana = q.get("analysis")
    if ana:
        parts.append("解析: " + str(ana))
    return " ".join(p for p in parts if p and p not in ("None", "nan")).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess XES3G5M (HF parquet) into SLIM-KT CSVs.")
    ap.add_argument("--raw", required=True, help="folder from download_xes3g5m.py")
    ap.add_argument("--questions", default=None, help="optional metadata/questions.json (raw text)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-users", type=int, default=None, help="cap learners (debug)")
    args = ap.parse_args()

    train_files = _find_parquets(args.raw, "interaction_sequences", "train")
    test_files = _find_parquets(args.raw, "interaction_sequences", "test")
    if not train_files:
        raise FileNotFoundError(
            f"No train parquet under {args.raw}/interaction_sequences. "
            "Run scripts/download_xes3g5m.py first (HF_ENDPOINT=https://hf-mirror.com).")

    out = common.resolve_out("xes3g5m", args.out)
    print(f"[xes3g5m] train={train_files}\n[xes3g5m] test={test_files}\n[xes3g5m] out={out}")

    train_df = pd.concat([pd.read_parquet(f) for f in train_files], ignore_index=True)
    recs = _expand(train_df, uid_offset=0)
    fold_map = _uid_fold_map(train_df)

    test_uids: set = set()
    if test_files:
        test_df = pd.concat([pd.read_parquet(f) for f in test_files], ignore_index=True)
        test_recs = _expand(test_df, uid_offset=TEST_UID_OFFSET)
        test_uids = {r["uid"] for r in test_recs}
        recs.extend(test_recs)

    inter = pd.DataFrame(recs)
    if args.max_users:
        keep = set(pd.Index(inter["uid"].unique())[: args.max_users])
        inter = inter[inter["uid"].isin(keep)]
        test_uids &= keep
        fold_map = {u: f for u, f in fold_map.items() if u in keep}

    # ---- precomputed semantic embeddings (row idx == internal qid) ----
    emb = _load_embeddings(args.raw)
    if emb is not None:
        max_q = int(inter["question_id"].max())
        if max_q >= emb.shape[0]:
            print(f"[xes3g5m] WARNING: max qid {max_q} >= #embeddings {emb.shape[0]}; "
                  "row-order alignment may be off.")
        np.save(os.path.join(out, "precomputed_qid_emb.npy"), emb)
        print(f"[xes3g5m] saved precomputed_qid_emb.npy {emb.shape} -> {out}")

    # ---- items.csv (real text only if questions.json supplied) ----
    questions = _load_questions(args.questions)
    n_emb = emb.shape[0] if emb is not None else 0
    qids = sorted(set(inter["question_id"].unique()) | set(questions.keys()) | set(range(n_emb)))
    items = pd.DataFrame({"question_id": qids})
    items["text"] = items["question_id"].map(lambda q: _item_text(questions.get(int(q), {})))

    max_opt = 0
    opt_map: Dict[int, List[str]] = {}
    for q in qids:
        opt_list = _options_to_list(questions.get(int(q), {}).get("options"))
        if opt_list:
            opt_map[q] = opt_list
            max_opt = max(max_opt, len(opt_list))
    for j in range(max_opt):
        items[f"option_{j}"] = items["question_id"].map(
            lambda q: opt_map.get(int(q), [None] * max_opt)[j] if j < len(opt_map.get(int(q), [])) else None)

    common.write_interactions(inter, out)
    common.write_items(items, out)
    _write_folds(inter, fold_map, test_uids, out)

    stats = common.basic_stats(inter)
    common.write_report(out, {
        "dataset": "xes3g5m",
        "source": "HuggingFace: Atomi/XES3G5M_{interaction_sequences,content_metadata}",
        "raw": args.raw,
        "has_raw_text": bool(questions),
        "num_options": max_opt,
        "precomputed_embedding_dim": int(emb.shape[1]) if emb is not None else 0,
        "official_test_users": len(test_uids),
        "note": ("real question text merged from questions.json" if questions else
                 "NO raw text; semantic signal = precomputed_qid_emb.npy (Pillar 1 only)"),
        **stats,
    })
    print(f"[xes3g5m] done: {stats}")
    if not questions:
        print("[xes3g5m] reminder: pass --questions questions.json to enable LLM attribute/option distillation.")


def _write_folds(inter: pd.DataFrame, fold_map: Dict[int, int], test_uids: set, out: str,
                 num_folds: int = 5, seed: int = 42) -> None:
    """pyKT protocol: shared official test across folds; valid = uids whose CV fold==k."""
    os.makedirs(os.path.join(out, "splits"), exist_ok=True)
    test = sorted(int(u) for u in test_uids)
    train_uids = [int(u) for u in inter["uid"].unique() if int(u) not in test_uids]

    if fold_map:
        folds_present = sorted(set(int(v) for v in fold_map.values()))
        for k in folds_present:
            valid = sorted(u for u in train_uids if fold_map.get(u) == k)
            train = sorted(u for u in train_uids if fold_map.get(u) != k)
            with open(os.path.join(out, "splits", f"fold{k}.json"), "w", encoding="utf-8") as f:
                json.dump({"train": train, "valid": valid, "test": test}, f)
    else:
        # no fold column: random learner K-fold, official test still shared
        rng = np.random.default_rng(seed)
        arr = np.array(sorted(train_uids)); rng.shuffle(arr)
        groups = np.array_split(arr, num_folds)
        for k in range(num_folds):
            valid = sorted(int(u) for u in groups[k].tolist())
            train = sorted(int(u) for u in np.concatenate(
                [groups[j] for j in range(num_folds) if j != k]).tolist())
            with open(os.path.join(out, "splits", f"fold{k}.json"), "w", encoding="utf-8") as f:
                json.dump({"train": train, "valid": valid, "test": test}, f)


if __name__ == "__main__":
    main()
