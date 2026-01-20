"""
Детерминированное обогащение по эталону.

Здесь НЕТ fuzzy.
Только строгие правила: если (region, town) однозначно определяет district -> восстановим.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


@dataclass(frozen=True)
class RefRow:
    region: str
    district: str
    town: str
    town_type: Optional[str] = None
    population: Optional[float] = None


class RefLookup:
    """Fast deterministic lookups over reference rows (canonical values)."""

    def __init__(self, ref_df: pd.DataFrame):
        self.rdt_exists: Set[Tuple[str, str, str]] = set()
        self.rt_to_districts: Dict[Tuple[str, str], Set[str]] = {}

        for _, rr in ref_df.iterrows():
            r = str(rr["region"])
            d = str(rr["district"])
            t = str(rr["town"])
            self.rdt_exists.add((r, d, t))
            self.rt_to_districts.setdefault((r, t), set()).add(d)

    def district_by_region_town(self, region: str, town: str) -> Optional[str]:
        ds = self.rt_to_districts.get((region, town), set())
        if len(ds) == 1:
            return next(iter(ds))
        return None

    def is_consistent(self, region: str, district: str, town: str) -> bool:
        return (region, district, town) in self.rdt_exists