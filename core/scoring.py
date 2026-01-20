"""
Эвристический скор уверенности.

Принципы:
- +0.4 за точное совпадение региона
- +0.3 за город
- +0.2 за район
- +0.1 за восстановленное поле
Максимум = 1.0
"""

def confidence_score(row: dict, restored_fields: int) -> float:
    score = 0.0
    if row.get("region"):
        score += 0.4
    if row.get("town"):
        score += 0.3
    if row.get("district"):
        score += 0.2
    score += min(restored_fields * 0.1, 0.1)
    return round(min(score, 1.0), 3)