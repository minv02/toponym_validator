"""
Инструменты для fuzzy matching.

Важное правило проекта:
matching НЕ принимает решений "что исправлять".
Он только считает похожесть и выбирает кандидатов.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from rapidfuzz import fuzz, process


@dataclass(frozen=True)
class MatchResult:
    match: Optional[str]         # canonical value
    score: float                 # 0..1
    best_norm_key: Optional[str] # key in dict if applicable


def similarity(a: Optional[str], b: Optional[str]) -> float:
    """Similarity 0..1 using WRatio; None -> 0."""
    if not a or not b:
        return 0.0
    return fuzz.WRatio(a, b) / 100.0


def best_from_norm_map(
    query_norm: Optional[str],
    norm_to_canon: Dict[str, str],
    cutoff: float,
) -> MatchResult:
    """
    Match query (normalized) against keys of norm_to_canon and return canonical.
    cutoff: 0..1
    """
    if not query_norm or not norm_to_canon:
        return MatchResult(None, 0.0, None)

    keys = list(norm_to_canon.keys())
    m = process.extractOne(query_norm, keys, scorer=fuzz.WRatio, score_cutoff=int(cutoff * 100))
    if not m:
        return MatchResult(None, 0.0, None)

    norm_key = m[0]
    score = m[1] / 100.0
    return MatchResult(norm_to_canon[norm_key], score, norm_key)


def top_n_from_norm_map(
    query_norm: Optional[str],
    norm_to_canon: Dict[str, str],
    n: int = 5,
) -> List[Tuple[str, float]]:
    """Return top-N canonical candidates with scores (0..1)."""
    if not query_norm or not norm_to_canon:
        return []
    keys = list(norm_to_canon.keys())
    matches = process.extract(query_norm, keys, scorer=fuzz.WRatio, limit=n)
    out: List[Tuple[str, float]] = []
    for norm_key, sc, _ in matches:
        out.append((norm_to_canon[norm_key], sc / 100.0))
    return out