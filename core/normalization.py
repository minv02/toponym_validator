"""
Нормализация входных строк.

Концепция:
- привести все 'пустоты' к None
- унифицировать регистр
- удалить служебные слова (г., с., дер. и т.п.)
"""

import re
from typing import Optional

EMPTY_VALUES = {"", " ", "na", "nan", "null", "none", "-"}

def normalize_empty(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float):
        return None
    v = str(value).strip().lower()
    if v in EMPTY_VALUES:
        return None
    return v

SERVICE_WORDS = [
    "город", "г.", "село", "деревня", "поселок", "посёлок", "пгт"
]

def normalize_location_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.lower()
    for w in SERVICE_WORDS:
        v = re.sub(rf"\b{w}\b", "", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v or None