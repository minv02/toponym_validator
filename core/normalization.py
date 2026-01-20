"""
Нормализация входных строк и эталона.

Принципы:
- стандартизируем пустоты в None
- минимальная, но стабильная нормализация строк (lower, ё->е, пробелы)
- отдельные нормализаторы для region / district / town (не смешиваем правила!)
"""

from __future__ import annotations

import re
from typing import Optional, Any

EMPTY_VALUES = {"", " ", "na", "nan", "null", "none", "-", "—", "n/a", "нет", "не знаю"}


def normalize_empty(value: Any) -> Optional[str]:
    """Convert various empty-like inputs to None; otherwise return stripped string."""
    if value is None:
        return None
    # pandas NaN
    try:
        # float('nan') != float('nan')
        if isinstance(value, float) and value != value:
            return None
    except Exception:
        pass

    v = str(value).strip()
    if v == "":
        return None
    if v.strip().lower() in EMPTY_VALUES:
        return None
    return v


def _base_text(s: str) -> str:
    """Base normalization shared by all location fields."""
    # важное: Excel часто содержит NBSP (\xa0) и табы
    s = s.replace("\xa0", " ").replace("\t", " ")
    s = s.strip().lower()
    s = s.replace("ё", "е")
    s = s.replace("—", "-").replace("–", "-")
    # схлопываем любые whitespace (в т.ч. двойные пробелы)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Town type tokens often appear in survey input; remove for town matching only.
TOWN_TYPE_TOKENS = [
    # базовые (как в примере)
    "город", "г.",
    "село", "с.",
    "деревня",
    "поселок", "посёлок",
    "пгт",

    # реально часто встречается в твоём файле
    "хутор",
    "станица",
    "аул",
    "улус",
    "починок",
    "слобода",
    "местечко",
    "выселок",
    "заимка",
    "кордон",
    "участок",

    # жд-шные и станционные форматы
    "станция",
    "разъезд",
    "ж.д. станция",
    "ж.д. разъезд",
    "ж.д. казарма",

    # составные типы (из town_type)
    "рабочий посёлок",
    "городской посёлок",
    "пгт (рабочий посёлок)",
    "посёлок станции",
    "посёлок при станции",
    "посёлок ж.д. станции",
    "посёлок ж.д. разъезда",

    # обобщённое
    "населённый пункт",
]


DISTRICT_TOKENS = [
    # базовое
    "район", "р-н", "рн",

    # округа
    "округ",
    "муниципальный округ",
    "городской округ",

    # муниципальные сущности
    "муниципальный район",
    "муниципальное образование",
    "муниципальное",
    "образование",

    # часто встречающиеся шаблоны
    "городской",
    "муниципальный",
    "город",

    # сокращения (встречаются в данных)
    "го",  # "Городской округ" -> иногда как "ГО"
]


def normalize_region(value: Any) -> Optional[str]:
    """Normalize region name (keep meaningful words like 'область', 'край')."""
    v = normalize_empty(value)
    if v is None:
        return None
    return _base_text(v)


def normalize_district(value: Any) -> Optional[str]:
    """
    Normalize district name:
    - base normalize
    - remove tokens like 'район', 'муниципальный район', 'р-н'
    """
    v = normalize_empty(value)
    if v is None:
        return None
    s = _base_text(v)
    # remove punctuation to stabilize boundaries
    s = re.sub(r"[.,;:()\"']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    for tok in sorted(DISTRICT_TOKENS, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(tok)}\b", " ", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def normalize_town(value: Any) -> Optional[str]:
    v = normalize_empty(value)
    if v is None:
        return None
    s = _base_text(v)

    # ✅ НОВОЕ: вырезаем уточнения в скобках
    s = re.sub(r"\([^)]*\)", " ", s)

    s = re.sub(r"[.,;:()\"']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    for tok in sorted(TOWN_TYPE_TOKENS, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(tok)}\b", " ", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def normalize_country(value: Any) -> Optional[str]:
    """Normalize country; returns canonical 'Россия' for known variants."""
    v = normalize_empty(value)
    if v is None:
        return None
    s = _base_text(v)
    if s in {"россия", "russia", "rf", "russian federation", "россйия"}:
        return "Россия"
    return v.strip()