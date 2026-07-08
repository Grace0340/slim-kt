"""Sequence backbones for the KT student.

SAKTBackbone is implemented in full (self-attentive KT, Pandey & Karypis 2019).
AKTBackbone is a stub with the interface fixed and the monotonic-attention
mechanism (Ghosh et al. 2020) left as TODO(slim-kt).

Both take a sequence of *interaction* embeddings (previous question + response)
as keys/values and *query* embeddings (current question) and return a
per-position hidden state used by the mastery / response heads.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


def _causal_mask(L: int, device) -> torch.Tensor:
    # True = disallow attention (upper triangle, strictly future)
    return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)


class SAKTBackbone(nn.Module):
    """Self-attentive KT backbone.

    Query   : current-question embedding q_t.
    Key/Val : past interaction embedding x_{<t} (question + response).
    Output  : hidden state h_t predicting performance on q_t.
    """

    def __init__(self, d_model: int, n_heads: int, n_blocks: int, dropout: float,
                 max_len: int = 1024):
        super().__init__()
        self.pos = _PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
             for _ in range(n_blocks)]
        )
        self.ffns = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
                           nn.Linear(d_model, d_model)) for _ in range(n_blocks)]
        )
        self.ln1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.last_attn = None  # cached for interpretability

    def forward(self, query_emb: torch.Tensor, inter_emb: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # query_emb, inter_emb: [B, L, d]; key_padding_mask: [B, L] True=pad
        L = query_emb.size(1)
        x = self.pos(inter_emb)
        q = self.pos(query_emb)
        attn_mask = _causal_mask(L, query_emb.device)
        h = q
        for blk, ffn, ln1, ln2 in zip(self.blocks, self.ffns, self.ln1, self.ln2):
            attn_out, attn_w = blk(h, x, x, attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask, need_weights=True,
                                   average_attn_weights=True)
            self.last_attn = attn_w.detach()
            h = ln1(h + self.dropout(attn_out))
            h = ln2(h + self.dropout(ffn(h)))
        return h


class _MonotonicMHA(nn.Module):
    """Causal multi-head attention with AKT-style per-head distance decay
    (Ghosh et al. 2020). Query attends to keys with a learned exponential
    decay in temporal distance, on top of a strict causal mask.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.zeros(n_heads))  # softplus -> nonneg decay
        self.last_attn = None

    def forward(self, query, key, value, key_padding_mask=None):
        B, L, _ = query.shape
        Q = self.q(query).view(B, L, self.h, self.dk).transpose(1, 2)   # [B,h,L,dk]
        K = self.k(key).view(B, L, self.h, self.dk).transpose(1, 2)
        V = self.v(value).view(B, L, self.h, self.dk).transpose(1, 2)
        scores = (Q @ K.transpose(-1, -2)) / math.sqrt(self.dk)         # [B,h,L,L]

        idx = torch.arange(L, device=query.device)
        future = idx[None, :] > idx[:, None]                           # True = disallow (strict causal)
        dist = (idx[:, None] - idx[None, :]).clamp(min=0).float()      # (t - tau) over the past
        gamma = F.softplus(self.gamma).view(1, self.h, 1, 1)
        scores = scores - gamma * dist[None, None]                     # distance-decayed logits
        scores = scores.masked_fill(future[None, None], float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)                                  # all -inf rows -> 0
        self.last_attn = attn.mean(1).detach()
        out = (self.drop(attn) @ V).transpose(1, 2).reshape(B, L, -1)
        return self.o(out)


class AKTBackbone(nn.Module):
    """Context-aware attentive KT backbone (Ghosh et al. 2020).

    Uses monotonic (distance-decayed) multi-head attention instead of SAKT's
    position-encoded vanilla attention. Interface matches SAKTBackbone: the
    current-question query attends to strictly-preceding interaction embeddings.
    """

    def __init__(self, d_model: int, n_heads: int, n_blocks: int, dropout: float,
                 max_len: int = 1024):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_MonotonicMHA(d_model, n_heads, dropout) for _ in range(n_blocks)]
        )
        self.ffns = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
                           nn.Linear(d_model, d_model)) for _ in range(n_blocks)]
        )
        self.ln1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.last_attn = None

    def forward(self, query_emb: torch.Tensor, inter_emb: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = query_emb
        x = inter_emb
        for blk, ffn, ln1, ln2 in zip(self.blocks, self.ffns, self.ln1, self.ln2):
            attn_out = blk(h, x, x, key_padding_mask=key_padding_mask)
            self.last_attn = blk.last_attn
            h = ln1(h + self.dropout(attn_out))
            h = ln2(h + self.dropout(ffn(h)))
        return h


def build_backbone(name: str, d_model: int, n_heads: int, n_blocks: int, dropout: float):
    name = name.lower()
    if name == "sakt":
        return SAKTBackbone(d_model, n_heads, n_blocks, dropout)
    if name == "akt":
        return AKTBackbone(d_model, n_heads, n_blocks, dropout)
    raise ValueError(f"unknown backbone {name!r}")
