"""
Confidence scoring (без ML).

Идея:
- основа: качество лучшего кандидата (0..1)
- штраф: неоднозначность (маленький gap между 1 и 2)
- штраф: если поле было задано пользователем, но не сматчилось
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ConfidenceInputs:
    best_score: float           # 0..1, score of best candidate
    gap: float                  # 0..1, best_score - second_best_score (0 if no second)
    region_given: bool
    district_given: bool
    town_given: bool
    region_matched: bool
    district_matched: bool
    town_matched: bool
    status: str                 # ok / weak / ambiguous / ignored


def confidence(ci: ConfidenceInputs) -> float:
    if ci.status == "ignored":
        return 0.0

    # base from best score
    s = ci.best_score

    # ambiguity penalty
    if ci.status == "ambiguous":
        s *= 0.55
    else:
        # if gap is small, reduce
        if ci.gap < 0.05:
            s *= 0.80
        elif ci.gap < 0.10:
            s *= 0.90

    # penalties for "given but not matched"
    if ci.town_given and not ci.town_matched:
        s -= 0.35
    if ci.district_given and not ci.district_matched:
        s -= 0.15
    if ci.region_given and not ci.region_matched:
        s -= 0.20

    # small bonus for having strong hierarchy matches
    if ci.region_matched:
        s += 0.05
    if ci.region_matched and ci.district_matched:
        s += 0.03
    if ci.region_matched and ci.district_matched and ci.town_matched:
        s += 0.05

    s = max(0.0, min(1.0, s))
    return round(s, 3)