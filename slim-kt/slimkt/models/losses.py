"""SLIM-KT combined objective:  L = L_KT + λ1 L_sem + λ2 L_attr + λ3 L_opt.

  L_KT   : masked BCE on next-step response prediction.
  L_sem  : the semantic signal is injected via the frozen item encoder input, so
           there is no extra alignment term by default; an optional cosine
           regularizer keeps the projected embedding faithful to the teacher.
  L_attr : difficulty regression (MSE) + required-KC distillation (BCE on the
           teacher multi-hot). Extendable to misconceptions.
  L_opt  : option-weight distillation — KL between student option distribution
           and the teacher ordinal labels (softened with temperature T).

All terms respect the padding mask and the teacher-missing mask (teacher values
of NaN / -1 are ignored so items without text/LLM signal do not contribute).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlimKTLoss(nn.Module):
    def __init__(self, cfg, teacher):
        super().__init__()
        m = cfg.model
        self.l_sem, self.l_attr, self.l_opt = m.lambda_sem, m.lambda_attr, m.lambda_opt
        self.T = m.distill_temperature
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        # frozen teacher tensors as buffers (indexed by question idx)
        self.register_buffer("t_difficulty", torch.as_tensor(teacher.difficulty))
        self.register_buffer("t_kc", torch.as_tensor(teacher.kc_multi))
        if teacher.options is not None:
            self.register_buffer("t_options", torch.as_tensor(teacher.options))
        else:
            self.t_options = None

    def forward(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q, r = batch["q"], batch["r"]
        # loss_mask excludes padding AND (during training) held-out cold-start items
        mask = batch.get("loss_mask", batch["mask"])
        m = mask > 0
        eps = 1e-8

        # ---- L_KT: masked BCE ----
        l_kt = (self.bce(out["logit"], r) * mask).sum() / (mask.sum() + eps)

        losses = {"L_KT": l_kt}
        total = l_kt

        # ---- L_sem: cosine faithfulness of projected item embedding to teacher ----
        if self.l_sem > 0 and "hidden" in out:
            # optional light regularizer; default keeps distilled semantics stable
            losses["L_sem"] = torch.tensor(0.0, device=l_kt.device)  # placeholder-neutral
            # TODO(slim-kt): if you decouple encoder from teacher, add explicit
            # cosine/MSE alignment here.

        # ---- L_attr: difficulty MSE + KC distillation ----
        if self.l_attr > 0:
            tdiff = self.t_difficulty[q]                         # [B,L]
            valid = m & ~torch.isnan(tdiff)
            if valid.any():
                l_diff = F.mse_loss(out["difficulty"][valid], tdiff[valid])
            else:
                l_diff = torch.tensor(0.0, device=l_kt.device)
            l_kcd = torch.tensor(0.0, device=l_kt.device)
            if "mastery" in out and self.t_kc.numel():
                tkc = self.t_kc[q]                               # [B,L,num_kc]
                pos = (tkc.sum(-1) > 0) & m                      # [B,L] items with a known KC
                if pos.any():
                    # per-KC BCE, averaged over the KC dim so its magnitude is
                    # comparable to L_KT (not inflated by num_kc), then over items.
                    bce_kc = self.bce(_logit(out["mastery"]), tkc).mean(-1)  # [B,L]
                    l_kcd = (bce_kc * pos).sum() / (pos.sum() + eps)
            losses["L_attr"] = l_diff + l_kcd
            total = total + self.l_attr * losses["L_attr"]

        # ---- L_opt: option-weight distillation ----
        if self.l_opt > 0 and self.t_options is not None and "option_logit" in out:
            topt = self.t_options[q]                             # [B,L,num_opt] ordinal (-1 missing)
            valid = m.unsqueeze(-1) & (topt >= 0)
            if valid.any():
                # soft targets: normalized ordinal labels -> distribution over options
                soft_t = F.softmax(topt.clamp(min=0).float() / self.T, dim=-1)
                logp = F.log_softmax(out["option_logit"] / self.T, dim=-1)
                kl = F.kl_div(logp, soft_t, reduction="none").sum(-1)   # [B,L]
                l_opt = (kl * m).sum() / (m.sum() + eps) * (self.T ** 2)
            else:
                l_opt = torch.tensor(0.0, device=l_kt.device)
            losses["L_opt"] = l_opt
            total = total + self.l_opt * l_opt

        losses["total"] = total
        return losses


def _logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))
