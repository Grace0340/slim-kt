"""Classic ID-based knowledge-tracing baselines for the SLIM-KT comparison table.

All baselines are *question-level* and *ID-based* (they embed the question id and,
where applicable, its concept), so they share SLIM-KT's dataloader, splits,
cold-start protocol, loss (masked BCE on ``logit``) and evaluation. Being
ID-based, they have no representation for unseen questions and therefore collapse
on cold-start new items — which is exactly the contrast SLIM-KT is designed to
overcome.

Implemented:
  * DKT   — Deep Knowledge Tracing (Piech et al., 2015): LSTM over interaction
            embeddings, per-question output, gathered at the queried question.
  * DKVMN — Dynamic Key-Value Memory Networks (Zhang et al., 2017): static key
            memory + dynamic value memory with erase/add writes.
  * AKT   — Context-aware Attentive KT (Ghosh et al., 2020): Rasch question
            embeddings + monotonic (distance-decayed) causal attention.

Each ``forward(batch)`` returns at least ``{"logit": [B, L]}`` (pre-sigmoid, next
-step correctness for the queried question). ``mastery`` / ``attn`` are provided
where meaningful for the interpretability dump.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _shift(x: torch.Tensor) -> torch.Tensor:
    """Right-shift along time so position t only sees interactions strictly < t."""
    out = torch.zeros_like(x)
    out[:, 1:] = x[:, :-1]
    return out


def _inter_index(q: torch.Tensor, r: torch.Tensor, mask: torch.Tensor, num_q: int) -> torch.Tensor:
    """Encode (question, correct) as a single id in [0, 2*num_q); PAD -> 0."""
    idx = q + num_q * (r > 0.5).long()
    return idx * (mask > 0).long()


class DKT(nn.Module):
    def __init__(self, cfg, stats):
        super().__init__()
        d = cfg.model.d_model
        self.num_q = stats.num_questions
        self.inter_emb = nn.Embedding(2 * self.num_q, d, padding_idx=0)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.dropout = nn.Dropout(cfg.model.dropout)
        self.out = nn.Linear(d, self.num_q)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q, r, mask = batch["q"], batch["r"], batch["mask"]
        e = self.inter_emb(_inter_index(q, r, mask, self.num_q))
        h, _ = self.lstm(_shift(e))
        y = self.out(self.dropout(h))                       # [B, L, num_q]
        logit = y.gather(-1, q.unsqueeze(-1)).squeeze(-1)   # [B, L]
        return {"logit": logit}


class DKVMN(nn.Module):
    def __init__(self, cfg, stats, mem_size: int = 20):
        super().__init__()
        d = cfg.model.d_model
        self.num_q = stats.num_questions
        self.N = mem_size
        self.q_emb = nn.Embedding(self.num_q, d, padding_idx=0)        # key (correlation) embedding
        self.qa_emb = nn.Embedding(2 * self.num_q, d, padding_idx=0)   # value write embedding
        self.Mk = nn.Parameter(torch.randn(self.N, d) * 0.1)          # static key memory
        self.Mv0 = nn.Parameter(torch.randn(self.N, d) * 0.1)         # initial value memory
        self.erase = nn.Linear(d, d)
        self.add = nn.Linear(d, d)
        self.summary = nn.Linear(2 * d, d)
        self.pred = nn.Linear(d, 1)
        self.dropout = nn.Dropout(cfg.model.dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q, r, mask = batch["q"], batch["r"], batch["mask"]
        B, L = q.shape
        k = self.q_emb(q)                                     # [B, L, d]
        w = torch.softmax(k @ self.Mk.t(), dim=-1)            # [B, L, N] correlation weights
        va = self.qa_emb(_inter_index(q, r, mask, self.num_q))  # [B, L, d]
        Mv = self.Mv0.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, N, d]

        logits = []
        for t in range(L):
            wt = w[:, t]                                      # [B, N]
            rt = torch.bmm(wt.unsqueeze(1), Mv).squeeze(1)    # read from memory BEFORE write -> [B, d]
            ft = torch.tanh(self.summary(torch.cat([rt, k[:, t]], dim=-1)))
            logits.append(self.pred(self.dropout(ft)).squeeze(-1))
            e = torch.sigmoid(self.erase(va[:, t]))           # [B, d]
            a = torch.tanh(self.add(va[:, t]))                # [B, d]
            wt_ = wt.unsqueeze(-1)                             # [B, N, 1]
            Mv = Mv * (1 - wt_ * e.unsqueeze(1)) + wt_ * a.unsqueeze(1)
        return {"logit": torch.stack(logits, dim=1)}          # [B, L]


class _MonotonicAttention(nn.Module):
    """Causal multi-head attention with AKT-style per-head distance decay."""

    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.h = n_heads
        self.dk = d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.zeros(n_heads))  # softplus -> nonneg decay
        self.last_attn = None

    def forward(self, query, key, value, pad_mask):
        B, L, _ = query.shape
        Q = self.q(query).view(B, L, self.h, self.dk).transpose(1, 2)  # [B,h,L,dk]
        K = self.k(key).view(B, L, self.h, self.dk).transpose(1, 2)
        V = self.v(value).view(B, L, self.h, self.dk).transpose(1, 2)
        scores = (Q @ K.transpose(-1, -2)) / math.sqrt(self.dk)        # [B,h,L,L]

        idx = torch.arange(L, device=query.device)
        future = idx[None, :] > idx[:, None]                          # True = disallow
        dist = (idx[:, None] - idx[None, :]).clamp(min=0).float()     # (t - tau) for the past
        gamma = F.softplus(self.gamma).view(1, self.h, 1, 1)
        scores = scores - gamma * dist[None, None]                    # distance-decayed logits
        scores = scores.masked_fill(future[None, None], float("-inf"))
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)                                 # rows that are all -inf -> 0
        self.last_attn = attn.mean(1).detach()
        out = (self.drop(attn) @ V).transpose(1, 2).reshape(B, L, -1)
        return self.o(out)


class AKT(nn.Module):
    """Compact AKT: Rasch question embeddings + monotonic attention retriever."""

    def __init__(self, cfg, stats):
        super().__init__()
        d = cfg.model.d_model
        h, nb, dr = cfg.model.n_heads, cfg.model.n_blocks, cfg.model.dropout
        self.num_q, self.num_kc = stats.num_questions, stats.num_kcs
        self.concept = nn.Embedding(self.num_kc, d, padding_idx=0)      # c_{c_t}
        self.concept_var = nn.Embedding(self.num_kc, d, padding_idx=0)  # d_{c_t}
        self.q_diff = nn.Embedding(self.num_q, 1, padding_idx=0)        # scalar mu_{q_t}
        self.resp = nn.Embedding(3, d, padding_idx=0)                   # response embedding
        self.resp_var = nn.Embedding(3, d, padding_idx=0)
        self.enc = nn.ModuleList([_MonotonicAttention(d, h, dr) for _ in range(nb)])
        self.enc_ffn = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dr),
                                                    nn.Linear(d, d)) for _ in range(nb)])
        self.enc_ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(nb)])
        self.enc_ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(nb)])
        self.retriever = _MonotonicAttention(d, h, dr)
        self.dropout = nn.Dropout(dr)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(dr), nn.Linear(d, 1))
        self.last_attn = None

    def _rasch_q(self, q, kc):
        mu = self.q_diff(q)                                   # [B,L,1]
        return self.concept(kc) + mu * self.concept_var(kc)   # [B,L,d]

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q, kc, r, mask = batch["q"], batch["kc"], batch["r"], batch["mask"]
        pad = mask == 0
        resp_tok = torch.where(mask > 0, r.long() + 1, torch.zeros_like(q))  # 1 wrong / 2 correct
        qemb = self._rasch_q(q, kc)                           # question representation
        mu = self.q_diff(q)
        xemb = qemb + self.resp(resp_tok) + mu * self.resp_var(resp_tok)     # interaction

        # knowledge encoder over PAST interactions (shifted)
        h = _shift(xemb)
        for blk, ffn, ln1, ln2 in zip(self.enc, self.enc_ffn, self.enc_ln1, self.enc_ln2):
            h = ln1(h + self.dropout(blk(h, h, h, pad)))
            h = ln2(h + self.dropout(ffn(h)))
        # retrieve knowledge relevant to the current question
        retrieved = self.retriever(qemb, h, h, pad)           # [B,L,d]
        self.last_attn = self.retriever.last_attn
        logit = self.out(torch.cat([retrieved, qemb], dim=-1)).squeeze(-1)
        return {"logit": logit, "attn": self.last_attn}


def build_baseline(arch: str, cfg, stats) -> nn.Module:
    arch = arch.lower()
    if arch == "dkt":
        return DKT(cfg, stats)
    if arch == "dkvmn":
        return DKVMN(cfg, stats)
    if arch == "akt":
        return AKT(cfg, stats)
    raise ValueError(f"unknown baseline arch {arch!r}")
