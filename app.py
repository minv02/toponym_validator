import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import gradio as gr
import pandas as pd

from core.normalization import normalize_empty, normalize_location_name
from core.enrichment import enrich_row
from core.scoring import confidence_score

EXPECTED_COLS = ["country", "region", "district", "town"]


def _to_input_path(file_obj) -> str:
    """Extract a filesystem path from a Gradio File object."""
    return file_obj.name if hasattr(file_obj, "name") else str(file_obj)


def _standardize_row(raw_row: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize raw row values: empty-like -> None, location strings normalized."""
    clean: Dict[str, Any] = {}
    for k, v in raw_row.items():
        nv = normalize_empty(v)
        if k in EXPECTED_COLS:
            nv = normalize_location_name(nv)
        clean[k] = nv
    return clean


def _compute_stats(out_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute UI stats from output dataframe."""
    # валидной считаем запись с уверенностью >= 0.6 (можно потом вынести в настройку)
    valid_pct = round((out_df["confidence"] >= 0.6).mean() * 100, 1)

    # % восстановлений по полям — считаем по логам (простая эвристика)
    log_series = out_df["log"].fillna("")
    restored_country = log_series.str.contains("Страна восстановлена", regex=False).mean() * 100
    restored_district = log_series.str.contains("Район восстановлен", regex=False).mean() * 100

    # общий скор уверенности
    mean_conf = round(float(out_df["confidence"].mean()), 3)

    return {
        "% валидных записей (confidence>=0.6)": valid_pct,
        "% восстановлена country": round(restored_country, 1),
        "% восстановлен district": round(restored_district, 1),
        "средняя уверенность": mean_conf,
    }


def process(file_obj, progress=gr.Progress(track_tqdm=False)) -> Tuple[str, Dict[str, Any]]:
    """
    Validate + enrich survey locations against regions.csv and return:
    - path to output xlsx
    - run statistics
    """
    input_path = _to_input_path(file_obj)

    base_dir = Path(__file__).resolve().parent
    # ref_path = base_dir / "regions.csv"
    ref_path = "/Users/nikita/Desktop/PythonProjects/migunov_hse_practice/src/data/regions.csv"

    ref = pd.read_csv(ref_path)
    df_in = pd.read_excel(input_path)

    # гарантируем наличие ожидаемых колонок (если каких-то нет — создадим пустые)
    for c in EXPECTED_COLS:
        if c not in df_in.columns:
            df_in[c] = None

    total = len(df_in)
    results = []

    progress(0, desc="Подготовка…")

    for i, (_, row) in enumerate(df_in.iterrows(), start=1):
        log = []

        raw_row = row.to_dict()

        # входные значения сохраняем "как есть"
        in_payload = {f"in_{k}": raw_row.get(k) for k in raw_row.keys()}

        # стандартизируем / нормализуем то, что пойдёт в алгоритм
        clean = _standardize_row(raw_row)

        # обогащение (внутри также восстанавливается Россия по региону)
        enriched = enrich_row(clean, ref, log)

        # выходные значения складываем в out_
        out_payload = {f"out_{k}": enriched.get(k) for k in EXPECTED_COLS}

        # метаданные
        restored_fields = len(log)
        meta = {
            "log": "; ".join(log) if log else "",
            "confidence": confidence_score(enriched, restored_fields=restored_fields),
        }

        results.append({**in_payload, **out_payload, **meta})

        progress(i / max(total, 1), desc=f"Обработка: {i}/{total}")

    out_df = pd.DataFrame(results)

    # кроссплатформенно: пишем во временный xlsx
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        out_path = tmp.name

    out_df.to_excel(out_path, index=False)

    stats = _compute_stats(out_df)
    return out_path, stats


with gr.Blocks() as demo:
    gr.Markdown(
        "### Валидация и обогащение мест рождения (РФ)\n"
        "1) Загрузите `.xlsx` с колонками `country, region, district, town` (часть колонок может отсутствовать)\n"
        "2) Нажмите **Обработать**\n"
        "3) Скачайте результат: в нём будут **и входные (in_*)**, и **выходные (out_*)** поля + `log`, `confidence`"
    )

    file_in = gr.File(label="Входной файл (.xlsx)", file_types=[".xlsx"])
    btn = gr.Button("Обработать", variant="primary")
    file_out = gr.File(label="Выходной файл (.xlsx)")
    stats = gr.JSON(label="Статистика выполнения")

    btn.click(process, inputs=[file_in], outputs=[file_out, stats])

demo.queue()  # важно для прогресса/очереди
demo.launch()
