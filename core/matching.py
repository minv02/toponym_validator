"""
Нечёткое сопоставление топонимов.

Используется rapidfuzz:
- token_sort_ratio
- partial_ratio

Регион всегда матчитcя первым, затем район, затем НП.
"""

from rapidfuzz import fuzz, process
from typing import List, Tuple, Optional

def best_match(
    query: Optional[str],
    choices: List[str],
    score_cutoff: int = 80
) -> Tuple[Optional[str], float]:
    if query is None or not choices:
        return None, 0.0
    match = process.extractOne(
        query,
        choices,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=score_cutoff
    )
    if match is None:
        return None, 0.0
    return match[0], match[1] / 100.0