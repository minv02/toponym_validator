"""
Обогащение данных.

Логика:
- если регион и город валидны и однозначны → восстановить район
- если город омонимичен → использовать регион как дизамбигуатор
"""

from typing import Dict, Optional
import pandas as pd

def enrich_row(row: Dict, ref: pd.DataFrame, log: list) -> Dict:
    result = row.copy()

    if result.get("country") is None and result.get("region") is not None:
        result["country"] = "Россия"
        log.append("Страна восстановлена как Россия по региону")

    if result.get("district") is None and result.get("town") and result.get("region"):
        subset = ref[
            (ref["region"] == result["region"]) &
            (ref["town"] == result["town"])
        ]
        if len(subset) == 1:
            result["district"] = subset.iloc[0]["district"]
            log.append("Район восстановлен по региону+городу")

    return result