"""Regenerate ONLY teacher_kc.npy (ground-truth concept grounding) in place.

Loads the preprocessed dataset, builds the [num_q, num_kc] KC indicator from the
real (question -> concept) labels, and overwrites teacher_kc.npy without touching
the LLM-derived tensors (sem / difficulty / options). Use after preprocessing so
the L_attr KC term and the mastery head get real supervision without re-running
the (slow, GPU-bound) LLM extraction.

  python -m scripts.regen_teacher_kc --config configs/default.yaml \
    --dataset-config configs/dataset/xes3g5m.yaml --set dataset.name=xes3g5m
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from slimkt.config import add_config_args, load_config
from slimkt.data.datasets import load_dataset
from slimkt.teacher.llm_teacher import _build_kc_multi


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild teacher_kc.npy from dataset KC labels.")
    add_config_args(p)
    args = p.parse_args()
    cfg = load_config(args.config, args.dataset_config, args.set)

    sequences, stats = load_dataset(cfg.paths.data_root, cfg.dataset.name, cfg)
    kc_multi = _build_kc_multi(sequences, stats.num_questions, stats.num_kcs)

    cache_dir = os.path.join(cfg.paths.teacher_cache, cfg.dataset.name)
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, "teacher_kc.npy")
    np.save(out, kc_multi)
    print(f"[regen_teacher_kc] wrote {out} shape={kc_multi.shape} "
          f"nonzero_rows={int((kc_multi.sum(1) > 0).sum())} total_ones={int(kc_multi.sum())}")


if __name__ == "__main__":
    main()
