"""Offline teacher preprocessing cost accounting (Limitation vi).

Reports, per dataset:
  * number of items with text (LLM calls = attributes + options)
  * estimated LLM input+output tokens (Qwen2.5 tokenizer if available, else
    a word-count approximation)
  * semantic-embedding wall-clock: for encoder-based datasets (MiniLM) we
    re-encode the item texts and time it; for datasets shipping precomputed
    vectors (XES3G5M RoBERTa) we report that no encoding is needed.

Usage:
  python scripts/teacher_cost.py --data-root data --out results/teacher_cost.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np


def _load_tokenizer():
    # local path first to avoid a slow/blocked HF network call
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer
        for name in ("/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
                     "Qwen/Qwen2.5-7B-Instruct"):
            try:
                return AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            except Exception:
                continue
    except Exception:
        pass
    return None


def _count_tokens(tok, text: str) -> int:
    if not text:
        return 0
    if tok is not None:
        return len(tok(text, add_special_tokens=False)["input_ids"])
    return int(len(text.split()) * 1.3)  # rough fallback


def cost_for_dataset(data_root: str, name: str, tok) -> dict:
    cache = os.path.join(data_root, "teacher_cache", name)
    raw = os.path.join(cache, "teacher_raw.jsonl")
    sem_path = os.path.join(cache, "teacher_sem.npy")
    out: dict = {"dataset": name}

    # --- LLM extraction accounting from the cached raw jsonl ---
    n_items = 0
    out_tokens = 0
    n_attr_calls = 0
    n_opt_calls = 0
    if os.path.exists(raw):
        with open(raw, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n_items += 1
                if "attributes" in rec:
                    n_attr_calls += 1
                    out_tokens += _count_tokens(tok, json.dumps(rec["attributes"], ensure_ascii=False))
                if "option_weights" in rec:
                    n_opt_calls += 1
                    out_tokens += _count_tokens(tok, json.dumps(rec["option_weights"], ensure_ascii=False))
    out["n_items_with_text"] = n_items
    out["llm_calls"] = n_attr_calls + n_opt_calls
    out["llm_output_tokens_est"] = int(out_tokens)
    out["tokenizer"] = "Qwen2.5" if tok is not None else "wordcount_x1.3"

    # --- semantic embedding cost ---
    if os.path.exists(sem_path):
        sem = np.load(sem_path, mmap_mode="r")
        out["sem_shape"] = list(sem.shape)
    # XES3G5M ships precomputed RoBERTa vectors; DBE/Eedi use MiniLM encoding.
    precomputed = glob.glob(os.path.join(data_root, name, "precomputed_qid_emb.npy"))
    out["semantics_source"] = "precomputed_roberta" if precomputed else "encoded_minilm"
    return out


def time_minilm_encode(data_root: str, name: str, model_name: str) -> dict:
    """Re-encode item texts once and time it (encoder datasets only)."""
    import pandas as pd
    items_csv = os.path.join(data_root, name, "items.csv")
    if not os.path.exists(items_csv):
        return {"encode_error": "no items.csv"}
    df = pd.read_csv(items_csv)
    texts = [str(t) for t in df.get("text", []) if isinstance(t, str) and t.strip()]
    if not texts:
        return {"encode_error": "no texts"}
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(model_name, device=dev)
    # warmup
    m.encode(texts[: min(16, len(texts))], batch_size=64, show_progress_bar=False)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    m.encode(texts, batch_size=64, show_progress_bar=False)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return {"encode_items": len(texts), "encode_seconds": round(dt, 3),
            "encode_items_per_s": round(len(texts) / dt, 1), "encode_device": dev,
            "encoder": model_name}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--datasets", nargs="+", default=["xes3g5m", "dbe_kt22", "eedi"])
    ap.add_argument("--minilm", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--time-encode", action="store_true",
                    help="also re-encode MiniLM datasets to measure wall-clock")
    ap.add_argument("--out", default="results/teacher_cost.json")
    args = ap.parse_args()

    tok = _load_tokenizer()
    report = []
    for ds in args.datasets:
        row = cost_for_dataset(args.data_root, ds, tok)
        if args.time_encode and row.get("semantics_source") == "encoded_minilm":
            row.update(time_minilm_encode(args.data_root, ds, args.minilm))
        report.append(row)
        print(json.dumps(row, ensure_ascii=False))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[teacher_cost] wrote {args.out}")


if __name__ == "__main__":
    main()
