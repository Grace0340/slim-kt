"""pyKT-style knowledge-tracing sequence dataset.

Expected preprocessed inputs under ``<data_root>/<dataset>/``:

  interactions.csv   columns: uid, order (or timestamp), question_id, kc_id, correct[, option_id]
  items.csv          columns: question_id, text[, option_0..option_k][, kc_id]   (item text for the teacher)
  splits/fold{k}.json {"train": [uid,...], "valid": [uid,...], "test": [uid,...]}  (optional; else random)

Each learner history is chunked into windows of ``max_seq_len`` and left-padded
with ``PAD=0``. Question / KC ids are remapped to a contiguous range at load time
so that id 0 is reserved for padding.

TODO(slim-kt): provide dataset-specific preprocessing scripts that convert the
raw ASSISTments / EdNet / Junyi / Eedi dumps into the two CSVs above. Reusing
``pykt-toolkit`` preprocessing is recommended for comparability.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except Exception:  # allow import without torch for config inspection
    Dataset = object  # type: ignore
    DataLoader = None  # type: ignore
    torch = None  # type: ignore

PAD = 0


@dataclass
class DatasetStats:
    num_questions: int          # includes PAD at index 0
    num_kcs: int                # includes PAD at index 0
    num_options: int
    qid2idx: Dict[int, int]
    kc2idx: Dict[int, int]
    item_text: Dict[int, str]   # question idx -> concatenated stem+options text
    item_options: Dict[int, List[str]] = field(default_factory=dict)  # question idx -> option strings


class KTSequenceDataset(Dataset):
    """Windows of (question, kc, response) with next-step prediction targets.

    Cold-start support:
      * ``new_items`` — question idxs designated as *new* (held out).
      * training set (``is_train=True``): interactions on new items are removed
        from the *loss* (``loss_mask=0``), except a fraction ``rho`` kept as
        few-shot; they still remain in the sequence as context.
      * evaluation set (``is_train=False``): ``loss_mask == mask`` (evaluate all)
        and a per-position ``cold`` flag marks new-item positions for cold-start AUC.
    """

    def __init__(self, sequences: List[Dict[str, np.ndarray]], max_seq_len: int,
                 new_items: Optional[np.ndarray] = None, is_train: bool = False,
                 rho: float = 0.0, seed: int = 42):
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.new_items = new_items  # sorted np.ndarray or None
        self.is_train = is_train
        self.rho = rho
        self.seed = seed

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, i: int):
        s = self.sequences[i]
        L = self.max_seq_len
        q = np.full(L, PAD, dtype=np.int64)
        kc = np.full(L, PAD, dtype=np.int64)
        r = np.zeros(L, dtype=np.float32)
        opt = np.full(L, PAD, dtype=np.int64)
        mask = np.zeros(L, dtype=np.float32)
        cold = np.zeros(L, dtype=np.float32)
        n = len(s["q"])
        q[:n], kc[:n], r[:n], mask[:n] = s["q"], s["kc"], s["r"], 1.0
        if "opt" in s:
            opt[:n] = s["opt"]
        loss_mask = mask.copy()

        if self.new_items is not None and n > 0:
            is_new = np.isin(q[:n], self.new_items)
            cold[:n] = is_new.astype(np.float32)
            if self.is_train and is_new.any():
                # exclude new items from the training loss, keep a rho fraction (few-shot)
                excl = is_new.copy()
                if self.rho > 0:
                    uid = int(s.get("uid", i))
                    thr = int(self.rho * 100000)
                    for t in np.nonzero(is_new)[0]:
                        h = (uid * 1000003 + int(q[t]) * 97 + int(t)) % 100000
                        if h < thr:
                            excl[t] = False  # kept as few-shot
                loss_mask[:n][excl] = 0.0

        return {
            "q": torch.from_numpy(q),
            "kc": torch.from_numpy(kc),
            "r": torch.from_numpy(r),
            "opt": torch.from_numpy(opt),
            "mask": torch.from_numpy(mask),
            "loss_mask": torch.from_numpy(loss_mask),
            "cold": torch.from_numpy(cold),
        }


def _remap(values: pd.Series) -> Tuple[Dict[int, int], int]:
    uniq = sorted(pd.unique(values.dropna()).tolist())
    mapping = {int(v): i + 1 for i, v in enumerate(uniq)}  # +1: reserve 0 for PAD
    return mapping, len(uniq) + 1


def _read_item_text(items_path: str, qid2idx: Dict[int, int]) -> Dict[int, str]:
    text: Dict[int, str] = {}
    if not os.path.exists(items_path):
        return text
    df = pd.read_csv(items_path)
    opt_cols = [c for c in df.columns if c.startswith("option_")]
    for _, row in df.iterrows():
        qid = int(row["question_id"])
        if qid not in qid2idx:
            continue
        parts = [str(row.get("text", ""))]
        for c in opt_cols:
            v = row.get(c)
            if isinstance(v, str) and v.strip():
                parts.append(f"[{c}] {v}")
        text[qid2idx[qid]] = " ".join(p for p in parts if p and p != "nan").strip()
    return text


def _read_item_options(items_path: str, qid2idx: Dict[int, int]) -> Tuple[Dict[int, List[str]], int]:
    """Read option_0..option_k columns from items.csv into {question idx -> [opt strings]}.

    Returns the per-item option lists (only non-empty options kept) and the max
    number of option slots seen (used as num_options when interactions.csv has no
    explicit option_id column, e.g. XES3G5M)."""
    opts: Dict[int, List[str]] = {}
    max_opts = 0
    if not os.path.exists(items_path):
        return opts, max_opts
    df = pd.read_csv(items_path)
    opt_cols = sorted([c for c in df.columns if c.startswith("option_")],
                      key=lambda c: int(c.split("_")[1]) if c.split("_")[1].isdigit() else 0)
    if not opt_cols:
        return opts, max_opts
    for _, row in df.iterrows():
        qid = int(row["question_id"])
        if qid not in qid2idx:
            continue
        vals = [str(row.get(c, "")).strip() for c in opt_cols]
        vals = [v for v in vals if v and v.lower() != "nan"]
        if vals:
            opts[qid2idx[qid]] = vals
            max_opts = max(max_opts, len(vals))
    return opts, max_opts


def load_dataset(data_root: str, name: str, cfg) -> Tuple[List[Dict[str, np.ndarray]], DatasetStats]:
    ddir = os.path.join(data_root, name)
    inter_csv = os.path.join(ddir, "interactions.csv")

    # Building windowed sequences from a large interactions.csv (a Python groupby
    # over all learners) is expensive and identical across seeds/variants, so cache
    # the (sequences, stats) tuple keyed by the windowing params. Rebuild only when
    # the source CSVs are newer than the cache.
    import pickle
    cache_path = os.path.join(
        ddir, f"_seqcache_L{cfg.dataset.max_seq_len}_m{cfg.dataset.min_seq_len}.pkl")
    items_csv = os.path.join(ddir, "items.csv")
    srcs = [p for p in (inter_csv, items_csv) if os.path.exists(p)]
    if os.path.exists(cache_path) and srcs and \
            os.path.getmtime(cache_path) >= max(os.path.getmtime(p) for p in srcs):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    df = pd.read_csv(inter_csv)
    order_col = "order" if "order" in df.columns else "timestamp"
    df = df.sort_values(["uid", order_col])

    qid2idx, num_q = _remap(df["question_id"])
    kc2idx, num_kc = _remap(df["kc_id"])
    items_path = os.path.join(ddir, "items.csv")
    item_text = _read_item_text(items_path, qid2idx)
    item_options, max_item_opts = _read_item_options(items_path, qid2idx)
    # option_id in interactions gives the chosen-option space; otherwise fall back
    # to the number of option slots declared in items.csv (XES3G5M-style).
    num_opt = int(df["option_id"].max()) + 1 if "option_id" in df.columns else max_item_opts

    max_len, min_len = cfg.dataset.max_seq_len, cfg.dataset.min_seq_len
    # Vectorized windowing: map the id columns once, then slice per learner using
    # uid boundaries on the sorted frame (avoids a slow per-group pandas loop over
    # hundreds of thousands of learners).
    uids = df["uid"].to_numpy()
    q_all = df["question_id"].map(qid2idx).to_numpy(np.int64)
    kc_all = df["kc_id"].map(kc2idx).fillna(PAD).to_numpy(np.int64)
    r_all = df["correct"].to_numpy(np.float32)
    has_opt = "option_id" in df.columns
    opt_all = df["option_id"].to_numpy(np.int64) if has_opt else None

    sequences: List[Dict[str, np.ndarray]] = []
    n = len(uids)
    if n:
        bnd = np.flatnonzero(np.diff(uids)) + 1
        starts = np.concatenate(([0], bnd)).tolist()
        ends = np.concatenate((bnd, [n])).tolist()
        for si, ei in zip(starts, ends):
            uid = int(uids[si])
            for start in range(si, ei, max_len):
                stop = min(start + max_len, ei)
                if stop - start < min_len:
                    continue
                seq = {"uid": uid, "q": q_all[start:stop], "kc": kc_all[start:stop],
                       "r": r_all[start:stop]}
                if has_opt:
                    seq["opt"] = opt_all[start:stop]
                sequences.append(seq)

    stats = DatasetStats(num_q, num_kc, num_opt, qid2idx, kc2idx, item_text, item_options)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump((sequences, stats), f, protocol=4)
    except OSError:
        pass  # caching is best-effort; fall back to rebuilding next time
    return sequences, stats


def _fold_split(sequences, data_root, name, fold) -> Tuple[list, list, list]:
    split_path = os.path.join(data_root, name, "splits", f"fold{fold}.json")
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            sp = json.load(f)
        by = {k: set(v) for k, v in sp.items()}
        tr = [s for s in sequences if s["uid"] in by["train"]]
        va = [s for s in sequences if s["uid"] in by["valid"]]
        te = [s for s in sequences if s["uid"] in by["test"]]
        return tr, va, te
    # fallback: random 70/10/20 by sequence
    rng = np.random.default_rng(1234 + fold)
    idx = rng.permutation(len(sequences))
    n = len(sequences)
    tr_end, va_end = int(0.7 * n), int(0.8 * n)
    pick = lambda ids: [sequences[i] for i in ids]
    return pick(idx[:tr_end]), pick(idx[tr_end:va_end]), pick(idx[va_end:])


def build_dataloaders(cfg) -> Tuple["DataLoader", "DataLoader", "DataLoader", DatasetStats]:
    from .cold_start import select_new_items

    sequences, stats = load_dataset(cfg.paths.data_root, cfg.dataset.name, cfg)
    tr, va, te = _fold_split(sequences, cfg.paths.data_root, cfg.dataset.name, cfg.dataset.fold)

    new_items = None
    if bool(cfg.get_dotted("cold_start.enabled", False)):
        new_items = select_new_items(stats.num_questions,
                                     cfg.cold_start.new_item_frac, cfg.seed)

    def dl(seqs, shuffle, is_train):
        ds = KTSequenceDataset(seqs, cfg.dataset.max_seq_len, new_items=new_items,
                               is_train=is_train, rho=cfg.get_dotted("cold_start.rho", 0.0),
                               seed=cfg.seed)
        return DataLoader(ds, batch_size=cfg.dataset.batch_size, shuffle=shuffle,
                          num_workers=cfg.dataset.num_workers, pin_memory=True, drop_last=False)

    return dl(tr, True, True), dl(va, False, False), dl(te, False, False), stats
