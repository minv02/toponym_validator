import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import gradio as gr
import pandas as pd

from core.normalization import normalize_empty, normalize_location_name
from core.enrichment import enrich_row
from core.matching import best_match_r
from core.scoring import confidence_score

EXPECTED_COLS = ["country", "region", "district", "town"]

# Пороги можно тонко настроить (регион проще, район/НП — сложнее)
REGION_CUTOFF = 86
DISTRICT_CUTOFF = 84
TOWN_CUTOFF = 84


def _to_input_path(file_obj) -> str:
    return file_obj.name if hasattr(file_obj, "name") else str(file_obj)


def _standardize_row(raw_row: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for k, v in raw_row.items():
        nv = normalize_empty(v)
        if k in EXPECTED_COLS:
            nv = normalize_location_name(nv)
        clean[k] = nv
    return clean


def _compute_stats(out_df: pd.DataFrame) -> Dict[str, Any]:
    valid_pct = round((out_df["confidence"] >= 0.6).mean() * 100, 1)
    mean_conf = round(float(out_df["confidence"].mean()), 3)
    return {
        "% валидных записей (confidence>=0.6)": valid_pct,
        "средняя уверенность": mean_conf,
    }


def _mk_field_log(
    input_norm: Any,
    out_value: Any,
    is_valid: bool,
) -> Dict[str, bool]:
    """
    Формирует лог по одному полю в формате:
    { "is_valid": bool, "is_corrected": bool }
    """
    # corrected = алгоритм изменил значение относительно нормализованного ввода
    # Если ввода не было (None), но out появилось — считаем corrected=True (восстановление)
    if input_norm is None:
        is_corrected = out_value is not None
    else:
        is_corrected = (out_value != input_norm)
    return {"is_valid": bool(is_valid), "is_corrected": bool(is_corrected)}


def process(file_obj, progress=gr.Progress(track_tqdm=False)) -> Tuple[str, Dict[str, Any]]:
    input_path = _to_input_path(file_obj)

    base_dir = Path(__file__).resolve().parent
    ref_path = base_dir / "regions.csv"

    ref = pd.read_csv(ref_path)
    df_in = pd.read_excel(input_path)

    for c in EXPECTED_COLS:
        if c not in df_in.columns:
            df_in[c] = None

    # Подготовим быстрые справочники
    ref["region"] = ref["region"].astype(str)
    ref["district"] = ref["district"].astype(str)
    ref["town"] = ref["town"].astype(str)

    regions = sorted(ref["region"].unique().tolist())

    # Для дизамбигуации
    districts_by_region = {
        r: sorted(ref.loc[ref["region"] == r, "district"].unique().tolist())
        for r in regions
    }
    towns_by_region = {
        r: sorted(ref.loc[ref["region"] == r, "town"].unique().tolist())
        for r in regions
    }
    # город по регион+район (если район известен — сузим)
    towns_by_region_district = {}
    for r in regions:
        sub = ref[ref["region"] == r]
        for d in sub["district"].unique().tolist():
            towns_by_region_district[(r, d)] = sorted(
                sub.loc[sub["district"] == d, "town"].unique().tolist()
            )

    total = len(df_in)
    results = []

    progress(0, desc="Подготовка…")

    for i, (_, row) in enumerate(df_in.iterrows(), start=1):
        raw_row = row.to_dict()
        in_payload = {f"in_{k}": raw_row.get(k) for k in raw_row.keys()}
        clean = _standardize_row(raw_row)

        # --- 1) Fuzzy-валидация/коррекция региона ---
        r_in = clean.get("region")
        r_m = best_match_r(r_in, regions, score_cutoff=REGION_CUTOFF)
        out_region = r_m.match if r_m.match is not None else r_in
        region_valid = r_m.match is not None

        # --- 2) Район: используем регион как дизамбигуатор ---
        d_in = clean.get("district")
        district_choices = districts_by_region.get(out_region, []) if out_region else []
        # если регион не определили — fallback на все районы РФ
        if not district_choices:
            district_choices = sorted(ref["district"].unique().tolist())
        d_m = best_match_r(d_in, district_choices, score_cutoff=DISTRICT_CUTOFF)
        out_district = d_m.match if d_m.match is not None else d_in
        district_valid = d_m.match is not None

        # --- 3) НП: если район валиден, сужаем выбор до (регион, район) ---
        t_in = clean.get("town")
        town_choices = []
        if out_region and out_district and district_valid:
            town_choices = towns_by_region_district.get((out_region, out_district), [])
        if not town_choices and out_region:
            town_choices = towns_by_region.get(out_region, [])
        if not town_choices:
            town_choices = sorted(ref["town"].unique().tolist())

        t_m = best_match_r(t_in, town_choices, score_cutoff=TOWN_CUTOFF)
        out_town = t_m.match if t_m.match is not None else t_in
        town_valid = t_m.match is not None

        # --- 4) Страна: правило "если пропущена, но РФ-локация валидна → Россия" ---
        # (полностью "игнорировать не-Россию" можно усилить позже — сейчас фиксируем out_country)
        country_in = clean.get("country")
        out_country = country_in
        row_notes = []
        if country_in is None:
            # сигнал, что РФ: валидный регион/город/район
            if region_valid or town_valid or district_valid:
                out_country = "Россия"
                row_notes.append("country_restored_to_russia")
        else:
            # нормализатор приводит к lower, но тут хотим каноническое
            if country_in in ("россия", "russia", "rf", "russian federation"):
                out_country = "Россия"

        # --- 5) Обогащение административных уровней по справочнику ---
        # enrich_row ожидает dict с ключами country/region/district/town
        algo_log_text = []
        enriched = enrich_row(
            {
                "country": out_country,
                "region": out_region,
                "district": out_district,
                "town": out_town,
            },
            ref,
            algo_log_text,
        )

        out_country = enriched.get("country")
        out_region = enriched.get("region")
        out_district = enriched.get("district")
        out_town = enriched.get("town")

        # --- 6) Структурированный JSON-лог ---
        log_obj = {
            "region": _mk_field_log(r_in, out_region, region_valid),
            "district": _mk_field_log(d_in, out_district, district_valid),
            "town": _mk_field_log(t_in, out_town, town_valid),
        }
        # если хочешь — можно добавить служебный блок, не мешая основной структуре:
        if row_notes or algo_log_text:
            log_obj["_meta"] = {
                "notes": row_notes,
                "actions": algo_log_text,
                "scores": {
                    "region": round(r_m.score, 3),
                    "district": round(d_m.score, 3),
                    "town": round(t_m.score, 3),
                },
            }

        out_payload = {
            "out_country": out_country,
            "out_region": out_region,
            "out_district": out_district,
            "out_town": out_town,
        }

        restored_fields = int(out_country != country_in and country_in is None) + len(algo_log_text)
        meta = {
            "log": json.dumps(log_obj, ensure_ascii=False),
            "confidence": confidence_score(
                {"region": out_region, "district": out_district, "town": out_town},
                restored_fields=restored_fields,
            ),
        }

        results.append({**in_payload, **out_payload, **meta})

        progress(i / max(total, 1), desc=f"Обработка: {i}/{total}")

    out_df = pd.DataFrame(results)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        out_path = tmp.name

    out_df.to_excel(out_path, index=False)

    stats = _compute_stats(out_df)
    return out_path, stats


with gr.Blocks() as demo:
    gr.Markdown(
        "### Валидация и обогащение мест рождения (РФ)\n"
        "1) Загрузите `.xlsx` с колонками `country, region, district, town` (часть колонок может отсутствовать)\n"
        "2) Нажмите **Обработать** (прогресс обновляется по строкам)\n"
        "3) Скачайте результат: **in_*** и **out_*** поля + `log` (JSON) + `confidence`"
    )

    file_in = gr.File(label="Входной файл (.xlsx)", file_types=[".xlsx"])
    btn = gr.Button("Обработать", variant="primary")
    file_out = gr.File(label="Выходной файл (.xlsx)")
    stats = gr.JSON(label="Статистика выполнения")

    btn.click(process, inputs=[file_in], outputs=[file_out, stats])

demo.queue()
demo.launch()