"""Prompt templates for the frozen LLM teacher.

Two extractions per question:
  1. ATTRIBUTES  — difficulty (scalar 0-1), required knowledge concepts, likely
     misconceptions. Returned as strict JSON so it can be parsed and distilled.
  2. OPTION WEIGHTS — for multiple-choice items, an ordinal "adequacy" label per
     option (TCOW-style, cf. Kim et al., EMNLP Findings 2025). Higher = more
     indicative of mastery; the correct option is typically highest.

Keep prompts deterministic (temperature 0) and force JSON to make parsing robust.
"""
from __future__ import annotations

import json
from typing import Dict, List

ATTRIBUTE_SYSTEM = (
    "You are an expert assessment designer. Given a question, estimate its "
    "pedagogical attributes. Be concise and calibrated. Respond with STRICT JSON "
    "only, no prose."
)

ATTRIBUTE_SCHEMA = {
    "difficulty": "float in [0,1], 0=very easy, 1=very hard",
    "required_kcs": "list of short knowledge-concept names needed to solve it",
    "misconceptions": "list of short descriptions of likely wrong ideas",
}


def build_attribute_prompt(question_text: str) -> List[Dict[str, str]]:
    user = (
        "Question:\n"
        f"{question_text}\n\n"
        "Return JSON with exactly these keys:\n"
        f"{json.dumps(ATTRIBUTE_SCHEMA, ensure_ascii=False, indent=2)}\n"
        'Example: {"difficulty": 0.6, "required_kcs": ["fractions","division"], '
        '"misconceptions": ["treats numerator and denominator independently"]}'
    )
    return [
        {"role": "system", "content": ATTRIBUTE_SYSTEM},
        {"role": "user", "content": user},
    ]


OPTION_SYSTEM = (
    "You are an expert tutor scoring how much choosing each answer option reveals "
    "about a learner's mastery. Respond with STRICT JSON only."
)


def build_option_prompt(question_text: str, options: List[str], num_labels: int) -> List[Dict[str, str]]:
    labels = _ordinal_labels(num_labels)
    opts = "\n".join(f"({i}) {o}" for i, o in enumerate(options))
    user = (
        "Question and options:\n"
        f"{question_text}\n{opts}\n\n"
        f"For EACH option index, assign one adequacy label from {labels} "
        "(higher = more indicative of mastery; the fully correct option is "
        "'adequate'). Return JSON: {\"weights\": {\"0\": \"inadequate\", ...}}."
    )
    return [
        {"role": "system", "content": OPTION_SYSTEM},
        {"role": "user", "content": user},
    ]


def _ordinal_labels(k: int) -> List[str]:
    base = ["inadequate", "somewhat inadequate", "somewhat adequate", "adequate"]
    if k <= len(base):
        # evenly sample k labels spanning the scale
        idx = [round(i * (len(base) - 1) / (k - 1)) for i in range(k)] if k > 1 else [len(base) - 1]
        return [base[i] for i in idx]
    return base + [f"level_{i}" for i in range(len(base), k)]


def label_to_ordinal(label: str, num_labels: int) -> int:
    labels = _ordinal_labels(num_labels)
    label = label.strip().lower()
    return labels.index(label) if label in labels else num_labels - 1
