"""Frozen LLM teacher: builds and caches per-item signals.

Signals cached per question (indexed by the contiguous question idx from
DatasetStats):
  * semantic embedding      -> teacher_sem.npy         [num_q, embed_dim]
  * difficulty (scalar)     -> teacher_difficulty.npy  [num_q]
  * required-KC multi-hot   -> teacher_kc.npy          [num_q, num_kc]
  * misconception multi-hot -> teacher_misc.npy        [num_q, num_misc]
  * option ordinal labels   -> teacher_options.npy     [num_q, num_options]  (-1 = missing)
  * raw JSON per item       -> teacher_raw.jsonl

Run once, offline. The training loop only reads the .npy tensors; the LLM is
never called during training or inference.

CLI:
  python -m slimkt.teacher.llm_teacher --config configs/default.yaml \
    --dataset-config configs/dataset/eedi.yaml --set dataset.name=eedi
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..config import add_config_args, load_config
from ..utils import get_logger
from . import prompts

log = get_logger("slimkt.teacher")


@dataclass
class TeacherCache:
    sem: np.ndarray                 # [num_q, embed_dim]
    difficulty: np.ndarray          # [num_q]
    kc_multi: np.ndarray            # [num_q, num_kc]
    options: Optional[np.ndarray]   # [num_q, num_options] ordinal labels, or None

    @classmethod
    def load(cls, cache_dir: str) -> "TeacherCache":
        opt_path = os.path.join(cache_dir, "teacher_options.npy")
        return cls(
            sem=np.load(os.path.join(cache_dir, "teacher_sem.npy")),
            difficulty=np.load(os.path.join(cache_dir, "teacher_difficulty.npy")),
            kc_multi=np.load(os.path.join(cache_dir, "teacher_kc.npy")),
            options=np.load(opt_path) if os.path.exists(opt_path) else None,
        )

    def save(self, cache_dir: str) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(os.path.join(cache_dir, "teacher_sem.npy"), self.sem)
        np.save(os.path.join(cache_dir, "teacher_difficulty.npy"), self.difficulty)
        np.save(os.path.join(cache_dir, "teacher_kc.npy"), self.kc_multi)
        if self.options is not None:
            np.save(os.path.join(cache_dir, "teacher_options.npy"), self.options)


class LLMTeacher:
    """Wraps a frozen sentence encoder + an OpenAI-compatible chat LLM."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._embedder = None
        self._client = None

    # ---- lazy heavy deps ----
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            log.info("Loading sentence embedder: %s", self.cfg.teacher.embedder)
            self._embedder = SentenceTransformer(self.cfg.teacher.embedder)
        return self._embedder

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.cfg.teacher.llm_base_url,
                api_key=self.cfg.teacher.llm_api_key,
            )
        return self._client

    # ---- semantic embeddings ----
    def embed_items(self, texts: List[str]) -> np.ndarray:
        emb = self.embedder.encode(texts, batch_size=64, show_progress_bar=True,
                                   normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

    # ---- LLM JSON call with one retry ----
    def _chat_json(self, messages) -> dict:
        for attempt in range(2):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.teacher.llm_model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=self.cfg.teacher.max_new_tokens,
                )
                content = resp.choices[0].message.content
                return _parse_json(content)
            except Exception as e:  # noqa: BLE001
                log.warning("LLM call failed (attempt %d): %s", attempt + 1, e)
        return {}

    def extract_attributes(self, text: str) -> dict:
        return self._chat_json(prompts.build_attribute_prompt(text))

    def extract_option_weights(self, text: str, options: List[str]) -> dict:
        n = self.cfg.teacher.num_option_labels
        return self._chat_json(prompts.build_option_prompt(text, options, n))


def _parse_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content[content.find("{"):]
    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def _load_precomputed_sem(cfg, stats, num_q: int, path: str) -> np.ndarray:
    """Map an [n_qid, dim] embedding matrix (row == original qid) onto the
    contiguous question idx space (PAD row 0 stays zero)."""
    if path == "auto":
        path = os.path.join(cfg.paths.data_root, cfg.dataset.name, "precomputed_qid_emb.npy")
    emb = np.load(path)
    dim = emb.shape[1]
    if dim != cfg.teacher.embed_dim:
        log.warning("precomputed embedding dim %d != cfg.teacher.embed_dim %d; using %d. "
                    "Set teacher.embed_dim=%d in the dataset config.", dim, cfg.teacher.embed_dim, dim, dim)
    sem = np.zeros((num_q, dim), dtype=np.float32)
    hit = 0
    for qid, idx in stats.qid2idx.items():
        if 0 <= int(qid) < emb.shape[0]:
            sem[idx] = emb[int(qid)]
            hit += 1
    log.info("precomputed embeddings mapped for %d/%d questions from %s", hit, num_q - 1, path)
    return sem


def _build_kc_multi(sequences, num_q: int, num_kc: int) -> np.ndarray:
    """Concept grounding: build the [num_q, num_kc] ground-truth KC indicator from
    the dataset's own (question -> concept) labels observed in the interactions
    (already remapped to contiguous idx space). This gives the student's per-KC
    mastery head real supervision (interpretability + the L_attr KC term), and is
    independent of the LLM. PAD (idx 0) rows/cols stay zero."""
    kc_multi = np.zeros((num_q, num_kc), dtype=np.float32)
    for s in sequences:
        qa = np.asarray(s["q"]); ka = np.asarray(s["kc"])
        nz = (qa > 0) & (ka > 0)
        if nz.any():
            kc_multi[qa[nz], ka[nz]] = 1.0
    log.info("ground-truth KC multi-hot: %d/%d questions carry >=1 concept",
             int((kc_multi.sum(1) > 0).sum()), num_q - 1)
    return kc_multi


def build_cache(cfg, no_llm: bool = False, precomputed_emb: Optional[str] = None) -> TeacherCache:
    """Materialize teacher signals for every item and cache them to disk.

    If ``no_llm`` is True, only semantic embeddings are computed (attribute /
    option-weight losses then get masked out during training). Use this to get a
    quick real-data run going before serving an extraction LLM.

    If ``precomputed_emb`` is given, the semantic embeddings are loaded from that
    ``.npy`` (row index == original question_id, e.g. XES3G5M's authors' RoBERTa
    vectors) instead of running the sentence encoder. Rows are placed at the
    contiguous question idx via ``stats.qid2idx``. Pass ``"auto"`` to read
    ``<data_root>/<name>/precomputed_qid_emb.npy``.
    """
    from ..data.datasets import load_dataset

    sequences, stats = load_dataset(cfg.paths.data_root, cfg.dataset.name, cfg)
    num_q, num_kc = stats.num_questions, stats.num_kcs
    teacher = LLMTeacher(cfg)

    # 1) semantic embeddings for every question idx (PAD row 0 stays zero)
    texts = [stats.item_text.get(i, "") for i in range(num_q)]
    # Items with real text drive both the encoder path and the LLM loop; compute
    # once so it is defined regardless of the embedding source.
    nonempty = [i for i in range(1, num_q) if texts[i]]
    if precomputed_emb:
        sem = _load_precomputed_sem(cfg, stats, num_q, precomputed_emb)
    else:
        sem = np.zeros((num_q, cfg.teacher.embed_dim), dtype=np.float32)
        if nonempty:
            sem[nonempty] = teacher.embed_items([texts[i] for i in nonempty])
        else:
            log.warning("No item text found; semantic embeddings are all zero. "
                        "Populate items.csv or pass --precomputed-emb for cold-start to work.")

    # 2) attributes + 3) option weights via the LLM (JSON), cached per item
    difficulty = np.full(num_q, np.nan, dtype=np.float32)
    kc_multi = _build_kc_multi(sequences, num_q, num_kc)
    num_opt = max(stats.num_options, 0)
    options = np.full((num_q, num_opt), -1, dtype=np.int64) if (cfg.dataset.has_options and num_opt) else None

    cache_dir = os.path.join(cfg.paths.teacher_cache, cfg.dataset.name)
    os.makedirs(cache_dir, exist_ok=True)
    raw_path = os.path.join(cache_dir, "teacher_raw.jsonl")

    def _extract_one(i: int) -> dict:
        """Run the LLM attribute (+ option-weight) extraction for question idx i.

        Writes into the per-index rows of the shared difficulty/options arrays
        (each thread owns a distinct row i, so no lock is needed there) and
        returns the raw record for ordered JSONL logging."""
        attrs = teacher.extract_attributes(texts[i])
        rec = {"q_idx": i, "attributes": attrs}
        if "difficulty" in attrs:
            try:
                difficulty[i] = float(attrs["difficulty"])
            except (TypeError, ValueError):
                pass
        # required_kcs: map any that match known kc names is dataset-specific;
        # here we only record raw names. TODO(slim-kt): map names -> kc idx when
        # a KC vocabulary is available, then set kc_multi[i, idx] = 1.
        item_opts = stats.item_options.get(i, [])
        if options is not None and item_opts:
            opt_json = teacher.extract_option_weights(texts[i], options=item_opts)
            rec["option_weights"] = opt_json
            for k, lbl in (opt_json.get("weights", {}) or {}).items():
                try:
                    j = int(k)
                    if 0 <= j < num_opt:
                        options[i, j] = prompts.label_to_ordinal(str(lbl), cfg.teacher.num_option_labels)
                except ValueError:
                    pass
        return rec

    todo = [] if no_llm else nonempty
    conc = max(1, int(getattr(cfg.teacher, "request_concurrency", 8)))
    log.info("Extracting LLM attributes/options for %d items with concurrency=%d", len(todo), conc)
    with open(raw_path, "w", encoding="utf-8") as raw_f:
        with ThreadPoolExecutor(max_workers=conc) as ex:
            # ex.map preserves input order and streams results, giving ordered
            # JSONL output plus periodic progress without a manual lock.
            for done, rec in enumerate(ex.map(_extract_one, todo), 1):
                raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if done % 200 == 0 or done == len(todo):
                    log.info("attributes/options: %d/%d", done, len(todo))

    cache = TeacherCache(sem=sem, difficulty=difficulty, kc_multi=kc_multi, options=options)
    cache.save(cache_dir)
    log.info("Teacher cache written to %s", cache_dir)
    return cache


def main() -> None:
    p = argparse.ArgumentParser(description="Build the frozen-LLM teacher cache.")
    add_config_args(p)
    p.add_argument("--no-llm", action="store_true",
                   help="only build semantic embeddings (skip LLM attribute/option extraction)")
    p.add_argument("--precomputed-emb", default=None,
                   help="load semantic embeddings from a .npy (row==qid) instead of the "
                        "sentence encoder; use 'auto' for <data_root>/<name>/precomputed_qid_emb.npy")
    args = p.parse_args()
    cfg = load_config(args.config, args.dataset_config, args.set)
    build_cache(cfg, no_llm=args.no_llm, precomputed_emb=args.precomputed_emb)


if __name__ == "__main__":
    main()
