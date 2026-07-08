"""The lightweight SLIM-KT student.

Item representation is produced by a *semantic item encoder* that projects the
frozen-LLM embedding into the model space, optionally added to a learnable ID
embedding. Setting ``use_id_embedding=False`` makes the model ID-free, so unseen
items are represented purely by their distilled semantics — this is what enables
cold start without question identifiers.

Heads:
  * response head   -> P(correct) on the queried question (main KT objective)
  * mastery head    -> per-KC mastery in [0,1] (interpretability output)
  * difficulty head -> scalar, distilled from the teacher (attribute distillation)
  * option head     -> per-option ordinal logits, distilled from teacher weights

The teacher tensors are injected as a frozen buffer indexed by question idx.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .backbones import build_backbone


class SemanticItemEncoder(nn.Module):
    """Item representation from distilled semantics and/or a learnable ID.

    * ``use_semantic=True, use_id_embedding=False`` -> SLIM-KT (ID-free, cold-start ready)
    * ``use_semantic=False, use_id_embedding=True`` -> ID-only baseline (collapses on new items)
    * both True -> hybrid
    At least one must be enabled; if neither is set, ID is used as a fallback.
    """

    def __init__(self, embed_dim: int, d_model: int, num_questions: int,
                 use_id_embedding: bool, use_semantic: bool, dropout: float):
        super().__init__()
        self.use_semantic = use_semantic
        self.use_id = use_id_embedding or (not use_semantic)
        if self.use_semantic:
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, d_model), nn.LayerNorm(d_model), nn.ReLU(), nn.Dropout(dropout)
            )
        if self.use_id:
            self.id_emb = nn.Embedding(num_questions, d_model, padding_idx=0)

    def forward(self, q_idx: torch.Tensor, teacher_sem: torch.Tensor) -> torch.Tensor:
        # teacher_sem: [B, L, embed_dim] gathered for each q_idx
        e = None
        if self.use_semantic:
            e = self.proj(teacher_sem)
        if self.use_id:
            e = self.id_emb(q_idx) if e is None else e + self.id_emb(q_idx)
        return e


class SlimKTStudent(nn.Module):
    def __init__(self, cfg, stats, teacher_sem: torch.Tensor):
        super().__init__()
        m = cfg.model
        d = m.d_model
        self.num_kcs = stats.num_kcs
        self.num_options = stats.num_options
        self.embed_dim = teacher_sem.size(1)

        # frozen teacher semantic table [num_q, embed_dim]
        self.register_buffer("teacher_sem", teacher_sem)

        use_semantic = bool(cfg.get_dotted("model.use_semantic", True))
        self.item_encoder = SemanticItemEncoder(
            self.embed_dim, d, stats.num_questions, m.use_id_embedding, use_semantic, m.dropout
        )
        # interaction embedding = item embedding + response embedding
        self.resp_emb = nn.Embedding(3, d, padding_idx=0)  # 0 pad, 1 wrong, 2 correct
        self.backbone = build_backbone(m.backbone, d, m.n_heads, m.n_blocks, m.dropout)

        self.response_head = nn.Linear(d, 1)
        self.mastery_head = nn.Linear(d, self.num_kcs)
        self.difficulty_head = nn.Linear(d, 1)
        if self.num_options:
            # ordinal logits over labels for each option slot
            self.option_head = nn.Linear(d, self.num_options)

    def _gather_sem(self, q: torch.Tensor) -> torch.Tensor:
        # q: [B, L] -> [B, L, embed_dim]
        return self.teacher_sem[q]

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q, r, mask = batch["q"], batch["r"], batch["mask"]
        pad = mask == 0

        item_emb = self.item_encoder(q, self._gather_sem(q))           # [B,L,d]
        resp_tok = torch.where(mask > 0, r.long() + 1, torch.zeros_like(q))  # 1/2
        inter_emb = item_emb + self.resp_emb(resp_tok)                 # past interaction

        # shift: predict q_t from interactions strictly before t
        query = item_emb
        inter_shifted = torch.zeros_like(inter_emb)
        inter_shifted[:, 1:] = inter_emb[:, :-1]

        h = self.backbone(query, inter_shifted, key_padding_mask=pad)  # [B,L,d]

        out = {
            "logit": self.response_head(h).squeeze(-1),   # [B,L]
            "mastery": torch.sigmoid(self.mastery_head(h)),
            "difficulty": self.difficulty_head(item_emb).squeeze(-1),  # per-item
            "hidden": h,
        }
        if self.num_options:
            out["option_logit"] = self.option_head(item_emb)  # [B,L,num_options]
        out["attn"] = getattr(self.backbone, "last_attn", None)
        return out
