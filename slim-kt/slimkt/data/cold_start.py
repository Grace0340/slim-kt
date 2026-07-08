"""Cold-start / extreme few-shot (rho) splitting.

Definition used in the paper:
  * a fraction ``new_item_frac`` of questions are designated *new* (cold) items;
  * for each new item, only a fraction ``rho`` of its interactions may appear in
    the training set (rho = 0.001 is the extreme few-shot regime; rho = 0.0 means
    the item is entirely unseen at training). All remaining interactions of new
    items are moved to the cold-start test pool.

This module returns index masks over interactions so that the same underlying
data can be evaluated under warm and cold-start conditions without re-loading.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

import numpy as np


def select_new_items(num_questions: int, new_item_frac: float, seed: int = 42) -> np.ndarray:
    """Deterministically choose the held-out *new* question idxs (1..num_questions-1).

    Used identically by training (to exclude them from the loss) and evaluation
    (to compute cold-start AUC), so both share the exact same item hold-out.
    Returns a sorted int array.
    """
    rng = np.random.default_rng(seed)
    items = np.arange(1, num_questions)  # 0 is PAD
    k = int(round(new_item_frac * len(items)))
    if k <= 0:
        return np.array([], dtype=np.int64)
    chosen = rng.choice(items, size=k, replace=False)
    return np.sort(chosen).astype(np.int64)


@dataclass
class ColdStartSplit:
    new_items: Set[int]                       # question idxs held out as "new"
    train_visible_mask: Dict[int, np.ndarray] = field(default_factory=dict)
    # per-sequence boolean mask marking cold-start test positions is built on the fly


def make_cold_start_split(
    stats,
    interactions_per_item: Dict[int, int],
    new_item_frac: float,
    rho: float,
    seed: int = 42,
) -> ColdStartSplit:
    """Choose new items and the small visible fraction of their responses.

    Args:
        stats: DatasetStats (uses num_questions).
        interactions_per_item: question idx -> count of interactions.
        new_item_frac: fraction of items to treat as new.
        rho: fraction of each new item's interactions visible during training.
    """
    rng = np.random.default_rng(seed)
    all_items = np.array(sorted(interactions_per_item.keys()))
    n_new = int(round(new_item_frac * len(all_items)))
    new_items = set(rng.choice(all_items, size=n_new, replace=False).tolist())

    visible: Dict[int, np.ndarray] = {}
    for qid in new_items:
        c = interactions_per_item.get(qid, 0)
        k = int(np.floor(rho * c))
        # positional indices (within the item's interaction stream) kept for training
        visible[qid] = np.sort(rng.choice(c, size=k, replace=False)) if k > 0 and c > 0 \
            else np.array([], dtype=np.int64)
    return ColdStartSplit(new_items=new_items, train_visible_mask=visible)


def is_cold_position(qid: int, occurrence_index: int, split: ColdStartSplit) -> bool:
    """True if this (question, its k-th occurrence) is a held-out cold-start target."""
    if qid not in split.new_items:
        return False
    return occurrence_index not in split.train_visible_mask.get(qid, np.array([]))


def count_interactions_per_item(sequences: List[Dict[str, np.ndarray]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for s in sequences:
        for q in s["q"]:
            counts[int(q)] = counts.get(int(q), 0) + 1
    return counts
