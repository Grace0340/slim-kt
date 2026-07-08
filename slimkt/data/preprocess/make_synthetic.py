"""Generate a small synthetic KT dataset AND a matching teacher cache.

Purpose: a zero-dependency end-to-end smoke test of the whole SLIM-KT pipeline
(train -> eval) on an AutoDL GPU, with NO dataset download and NO LLM. The
synthetic task is deliberately made learnable from item semantics (item
embeddings cluster by knowledge concept), so cold-start prediction is non-trivial
and H1/H2 plumbing can be exercised.

Outputs (aligned to the contiguous question index used by DatasetStats):
  <data_root>/<name>/interactions.csv , items.csv , splits/fold*.json
  <teacher_cache>/<name>/teacher_{sem,difficulty,kc,options}.npy

Usage:
  python -m slimkt.data.preprocess.make_synthetic --name synth \
      --num-users 500 --num-questions 300 --num-kcs 20 --embed-dim 384
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import common
from ...teacher.llm_teacher import TeacherCache


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic KT data + teacher cache (smoke test).")
    ap.add_argument("--name", default="synth")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "./slimkt_data"))
    ap.add_argument("--teacher-cache", default=None,
                    help="default: <data-root>/teacher_cache")
    ap.add_argument("--num-users", type=int, default=500)
    ap.add_argument("--num-questions", type=int, default=300)
    ap.add_argument("--num-kcs", type=int, default=20)
    ap.add_argument("--num-options", type=int, default=4)
    ap.add_argument("--num-option-labels", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=100)
    ap.add_argument("--embed-dim", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    Q, K, O = args.num_questions, args.num_kcs, args.num_options

    # ids are 1-based and contiguous so that question idx == question_id and
    # kc idx == kc_id (matches DatasetStats remapping; index 0 = PAD).
    q_ids = np.arange(1, Q + 1)
    q_kc = rng.integers(1, K + 1, size=Q)                    # kc per question
    q_diff_raw = rng.normal(0.0, 1.0, size=Q)                # latent difficulty
    q_correct_opt = rng.integers(0, O, size=Q)               # correct option idx

    # per-KC semantic center -> item embeddings cluster by concept (cold start signal)
    kc_center = rng.normal(0, 1, size=(K + 1, args.embed_dim))
    sem = np.zeros((Q + 1, args.embed_dim), dtype=np.float32)
    for j, qid in enumerate(q_ids):
        v = kc_center[q_kc[j]] + 0.3 * rng.normal(0, 1, size=args.embed_dim)
        sem[qid] = v / (np.linalg.norm(v) + 1e-8)

    # per-user ability, per-(user,kc) offset
    ability = rng.normal(0, 1, size=args.num_users)
    kc_off = rng.normal(0, 0.5, size=(args.num_users, K + 1))

    rows = []
    for u in range(args.num_users):
        L = int(np.clip(rng.normal(args.seq_len, args.seq_len * 0.2), 10, args.seq_len * 2))
        visited = rng.integers(0, Q, size=L)                 # positional into q_ids
        for t, qi in enumerate(visited):
            qid = int(q_ids[qi])
            kc = int(q_kc[qi])
            logit = ability[u] + kc_off[u, kc] - q_diff_raw[qi]
            p = 1.0 / (1.0 + np.exp(-logit))
            correct = int(rng.random() < p)
            if correct:
                opt = int(q_correct_opt[qi])
            else:
                choices = [o for o in range(O) if o != q_correct_opt[qi]]
                opt = int(rng.choice(choices)) if choices else 0
            rows.append((u, t, qid, kc, correct, opt))

    inter = pd.DataFrame(rows, columns=["uid", "order", "question_id", "kc_id", "correct", "option_id"])

    # items.csv with pseudo-text per KC (embedded by the (fake) teacher)
    items = pd.DataFrame({"question_id": q_ids})
    items["text"] = [f"synthetic item on concept {int(k)}" for k in q_kc]
    items["kc_id"] = q_kc
    for j in range(O):
        items[f"option_{j}"] = f"Option {chr(ord('A') + j)}"

    out = common.resolve_out(args.name, os.path.join(args.data_root, args.name))
    common.write_interactions(inter, out)
    common.write_items(items, out)
    common.write_folds(inter["uid"].tolist(), out, num_folds=5, seed=args.seed)

    # ---- synthetic teacher cache (aligned to question idx) ----
    difficulty = np.full(Q + 1, np.nan, dtype=np.float32)
    difficulty[q_ids] = (1.0 / (1.0 + np.exp(-q_diff_raw))).astype(np.float32)  # -> [0,1]
    kc_multi = np.zeros((Q + 1, K + 1), dtype=np.float32)
    kc_multi[q_ids, q_kc] = 1.0
    options = np.full((Q + 1, O), -1, dtype=np.int64)
    top = args.num_option_labels - 1
    for j, qid in enumerate(q_ids):
        options[qid] = rng.integers(0, max(top, 1), size=O)   # distractors low
        options[qid, q_correct_opt[j]] = top                  # correct = most adequate

    cache = TeacherCache(sem=sem, difficulty=difficulty, kc_multi=kc_multi, options=options)
    tc_root = args.teacher_cache or os.path.join(args.data_root, "teacher_cache")
    tc_dir = os.path.join(tc_root, args.name)
    cache.save(tc_dir)

    correct_by_qid = {int(qid): int(q_correct_opt[j]) for j, qid in enumerate(q_ids)}
    common.write_report(out, {
        "dataset": f"{args.name} (SYNTHETIC smoke test)",
        "num_options": O, "embed_dim": args.embed_dim,
        "teacher_cache": tc_dir,
        "correct_option_by_qid": correct_by_qid,
        **common.basic_stats(inter),
    })
    print(f"[synthetic] data -> {out}")
    print(f"[synthetic] teacher cache -> {tc_dir}")
    print(f"[synthetic] stats: {common.basic_stats(inter)}")


if __name__ == "__main__":
    main()
