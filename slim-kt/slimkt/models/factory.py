"""Model factory: pick the SLIM-KT student or a classic ID-based baseline.

Selected via ``model.arch`` (default ``slimkt``). Baselines ignore the teacher
semantics and are trained with the pure KT (BCE) objective (set the distillation
lambdas to 0 for them).
"""
from __future__ import annotations


def build_model(cfg, stats, teacher_sem):
    arch = str(cfg.get_dotted("model.arch", "slimkt")).lower()
    if arch == "slimkt":
        from .student import SlimKTStudent
        return SlimKTStudent(cfg, stats, teacher_sem)
    from .baselines import build_baseline
    return build_baseline(arch, cfg, stats)
