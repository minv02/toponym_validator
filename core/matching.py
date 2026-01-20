"""
Нечёткое сопоставление топонимов.

Используется rapidfuzz:
- token_sort_ratio

Важно:
- функции возвращают не только match, но и score (0..1)
- пороги score_cutoff задаются снаружи (для региона/района/НП могут отличаться)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from rapidfuzz import fuzz, process


@dataclass(frozen=True)
class MatchResult:
    """Result of fuzzy matching against a choice list."""
    match: Optional[str]
    score: float  # 0..1

    @property
    def is_match(self) -> bool:
        return self.match is not None and self.score > 0.0


def best_match(
    query: Optional[str],
    choices: List[str],
    score_cutoff: int = 80,
) -> Tuple[Optional[str], float]:
    """
    Backward-compatible helper.

    Returns:
        (best_choice or None, score in 0..1)
    """
    if query is None or not choices:
        return None, 0.0
    m = process.extractOne(
        query,
        choices,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=score_cutoff,
    )
    if m is None:
        return None, 0.0
    return m[0], m[1] / 100.0


def best_match_r(
    query: Optional[str],
    choices: List[str],
    score_cutoff: int = 80,
) -> MatchResult:
    """Typed wrapper returning MatchResult."""
    m, s = best_match(query=query, choices=choices, score_cutoff=score_cutoff)
    return MatchResult(match=m, score=s)