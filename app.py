import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

from core.enrichment import RefLookup
from core.matching import best_from_norm_map, similarity, top_n_from_norm_map
from core.normalization import (
    normalize_country,
    normalize_district,
    normalize_region,
    normalize_town,
)
from core.scoring import ConfidenceInputs, confidence


# --- Cutoffs ---
REGION_CUTOFF = 0.86
DISTRICT_CUTOFF = 0.84
TOWN_CUTOFF = 0.84

# Candidate acceptance thresholds
CANDIDATE_MIN_SCORE = 0.78
AMBIGUITY_GAP_MIN = 0.06  # if gap < this -> ambiguous

# Special handling for "only town" rows (no region, no district)
ONLY_TOWN_MIN_TOWN_SCORE = 0.84  # accept global town match if town similarity >= this
ONLY_TOWN_NEAR_EPS = 0.03        # population tie-break window for near-best candidates


@dataclass
class Candidate:
    region: str
    district: str
    town: str
    score_region: float
    score_district: float
    score_town: float
    @property
    def total(self) -> float:
        # Weighted: region strongest, then town, then district
        return 0.45 * self.score_region + 0.35 * self.score_town + 0.20 * self.score_district


def _to_input_path(file_obj) -> str:
    return file_obj.name if hasattr(file_obj, "name") else str(file_obj)


def _clean_colname(name: Any) -> str:
    if name is None:
        return ""
    return str(name).strip()


def process(
    file_obj,
    col_country: str,
    col_region: str,
    col_district: str,
    col_town: str,
    REGION_CUTOFF: float,
    DISTRICT_CUTOFF: float,
    TOWN_CUTOFF: float,
    CANDIDATE_MIN_SCORE: float,
    AMBIGUITY_GAP_MIN: float,
    progress=gr.Progress(track_tqdm=False),
) -> Tuple[str, Dict[str, Any]]:
    input_path = _to_input_path(file_obj)
    base_dir = Path(__file__).resolve().parent
    ref_path = base_dir / "regions.csv"

    ref = pd.read_csv(ref_path)
    df_in = pd.read_excel(input_path)

    # ensure columns exist in ref
    for c in ["region", "district", "town"]:
        if c not in ref.columns:
            raise ValueError(f"regions.csv must contain column '{c}'")

    ref["region"] = ref["region"].astype(str)
    ref["district"] = ref["district"].astype(str)
    ref["town"] = ref["town"].astype(str)

    # population lookup (если колонки нет или NaN -> 0)
    pop_by_rdt: Dict[Tuple[str, str, str], float] = {}
    if "population" in ref.columns:
        for _, rr in ref.iterrows():
            r = str(rr["region"])
            d = str(rr["district"])
            t = str(rr["town"])
            try:
                p = float(rr["population"]) if rr["population"] == rr["population"] else 0.0
            except Exception:
                p = 0.0
            pop_by_rdt[(r, d, t)] = p

    # --- Build normalized indices from reference ---
    region_norm_to_canon: Dict[str, str] = {}
    district_norm_to_canon_by_region: Dict[str, Dict[str, str]] = {}

    town_rows_by_region: Dict[str, List[Tuple[str, str]]] = {}  # region -> list of (district, town)

    # Global index: normalized town -> all canonical rows
    town_norm_to_rows_global: Dict[str, List[Tuple[str, str, str]]] = {}  # norm_town -> [(r,d,t)]

    for _, rr in ref.iterrows():
        r = rr["region"]
        d = rr["district"]
        t = rr["town"]

        rn = normalize_region(r)
        dn = normalize_district(d)
        tn = normalize_town(t)

        if rn:
            region_norm_to_canon[rn] = r

        if rn and dn:
            district_norm_to_canon_by_region.setdefault(r, {})
            district_norm_to_canon_by_region[r][dn] = d

        town_rows_by_region.setdefault(r, []).append((d, t))

        if tn:
            town_norm_to_rows_global.setdefault(tn, []).append((r, d, t))

    ref_lookup = RefLookup(ref)

    # --- UI column mapping ---
    col_country = _clean_colname(col_country)
    col_region = _clean_colname(col_region)
    col_district = _clean_colname(col_district)
    col_town = _clean_colname(col_town)

    def get_val(row_dict: Dict[str, Any], col_name: str) -> Any:
        if not col_name:
            return None
        return row_dict.get(col_name, None)

    missing_cols = []
    for label, cname in [
        ("Страна", col_country),
        ("Регион", col_region),
        ("Район", col_district),
        ("Город", col_town),
    ]:
        if cname and cname not in df_in.columns:
            missing_cols.append(f"{label}: '{cname}'")

    # --- run ---
    total = len(df_in)
    results: List[Dict[str, Any]] = []
    progress(0, desc="Подготовка…")

    for i, (_, row) in enumerate(df_in.iterrows(), start=1):
        raw_row = row.to_dict()
        in_payload = {f"in_{k}": raw_row.get(k) for k in raw_row.keys()}

        country_raw = get_val(raw_row, col_country)
        region_raw = get_val(raw_row, col_region)
        district_raw = get_val(raw_row, col_district)
        town_raw = get_val(raw_row, col_town)

        # normalize inputs
        country_in = normalize_country(country_raw)
        region_in = normalize_region(region_raw)
        district_in = normalize_district(district_raw)
        town_in = normalize_town(town_raw)

        region_given = region_in is not None
        district_given = district_in is not None
        town_given = town_in is not None
        only_town_mode = bool(town_in) and (region_in is None) and (district_in is None)

        meta_notes: List[str] = []
        meta_actions: List[str] = []
        status = "ok"

        # --- country rule: ignore non-Russia ---
        if country_in is not None and country_in != "Россия":
            out = {
                "out_country": country_in,
                "out_region": None,
                "out_district": None,
                "out_town": None,
            }
            log_obj = {
                "region": {"is_valid": False, "is_corrected": False},
                "district": {"is_valid": False, "is_corrected": False},
                "town": {"is_valid": False, "is_corrected": False},
                "_meta": {
                    "status": "ignored",
                    "notes": ["ignored_non_russia"],
                    "actions": [],
                    "scores": {"best": 0.0, "second": 0.0, "gap": 0.0},
                    "top_candidates": [],
                    "mapping": {
                        "country_col": col_country,
                        "region_col": col_region,
                        "district_col": col_district,
                        "town_col": col_town,
                    },
                    "tiebreak_by_population": False,
                    **({"missing_input_columns": missing_cols} if missing_cols else {}),
                },
            }
            results.append(
                {**in_payload, **out, "log": json.dumps(log_obj, ensure_ascii=False), "confidence": 0.0}
            )
            progress(i / max(total, 1), desc=f"Обработка: {i}/{total}")
            continue

        # --- Step 1: match region (if provided) ---
        region_match = best_from_norm_map(region_in, region_norm_to_canon, cutoff=REGION_CUTOFF)
        region_canon = region_match.match if region_match.match else None
        region_matched = region_canon is not None

        # --- Candidate generation ---
        candidates: List[Candidate] = []

        if town_in:
            if region_canon:
                # candidates within matched region
                for (d, t) in town_rows_by_region.get(region_canon, []):
                    score_r = 1.0
                    score_t = similarity(town_in, normalize_town(t))
                    score_d = similarity(district_in, normalize_district(d)) if district_in else 0.0
                    candidates.append(Candidate(region_canon, d, t, score_r, score_d, score_t))
            else:
                # global search by town
                rows = town_norm_to_rows_global.get(town_in, [])

                if rows:
                    # exact normalized key hit
                    for (r, d, t) in rows:
                        score_r = similarity(region_in, normalize_region(r)) if region_in else 0.0
                        score_t = 1.0
                        score_d = similarity(district_in, normalize_district(d)) if district_in else 0.0
                        candidates.append(Candidate(r, d, t, score_r, score_d, score_t))
                else:
                    # fuzzy fallback: find closest town keys globally
                    meta_notes.append("global_town_fuzzy_fallback")
                    norm_keys = list(town_norm_to_rows_global.keys())
                    tmp_map = {k: k for k in norm_keys}
                    top_keys = top_n_from_norm_map(town_in, tmp_map, n=7)  # [(norm_key, score)]
                    for norm_key, sc_key in top_keys:
                        for (r, d, t) in town_norm_to_rows_global.get(norm_key, []):
                            score_r = similarity(region_in, normalize_region(r)) if region_in else 0.0
                            score_t = sc_key
                            score_d = similarity(district_in, normalize_district(d)) if district_in else 0.0
                            candidates.append(Candidate(r, d, t, score_r, score_d, score_t))
        else:
            meta_notes.append("no_town_provided")
            status = "weak"

        # If we have district given and region matched, we can reduce candidates to that district group
        if candidates and district_in and region_canon:
            dist_index = district_norm_to_canon_by_region.get(region_canon, {})
            dist_match = best_from_norm_map(district_in, dist_index, cutoff=DISTRICT_CUTOFF)

            # Restrict only if confident
            if dist_match.match and dist_match.score >= DISTRICT_CUTOFF:
                dcanon = dist_match.match
                before = len(candidates)
                candidates = [c for c in candidates if c.district == dcanon]
                after = len(candidates)
                meta_actions.append(f"candidates_restricted_by_district:{before}->{after}")
            else:
                meta_notes.append("district_not_restricting_candidates_due_to_low_confidence")

        # --- Choose best candidate ---
        best: Optional[Candidate] = None
        second: Optional[Candidate] = None
        if candidates:
            candidates.sort(key=lambda c: c.total, reverse=True)
            best = candidates[0]
            second = candidates[1] if len(candidates) > 1 else None

        best_score = best.total if best else 0.0
        second_score = second.total if second else 0.0
        gap = max(0.0, best_score - second_score)

        accepted = False
        ambiguous = False
        tiebreak_by_population = False

        # homonymy hint for town-only global cases
        if town_in and (not region_canon) and len({(c.region, c.district) for c in candidates}) > 1:
            meta_notes.append("possible_homonymy_town_multiple_regions_or_districts")

        # --- Acceptance logic (main) ---
        if best and best_score >= CANDIDATE_MIN_SCORE:
            if second is not None and gap < AMBIGUITY_GAP_MIN:
                ambiguous = True
                status = "ambiguous"
                meta_notes.append("ambiguous_top_candidates_close")

                # population tie-break among near-best candidates
                near = [c for c in candidates if c.total >= (best_score - ONLY_TOWN_NEAR_EPS)]
                if "population" in ref.columns and len(near) > 1:
                    near.sort(
                        key=lambda c: pop_by_rdt.get((c.region, c.district, c.town), 0.0),
                        reverse=True,
                    )
                    chosen = near[0]
                    if chosen != best:
                        best = chosen
                        best_score = best.total
                        tiebreak_by_population = True
                        meta_notes.append("population_tiebreak_applied")

                if only_town_mode:
                    meta_notes.append("only_town_global_population_choice")

                accepted = True
            else:
                accepted = True
                status = "ok"
        else:
            status = "weak"

        # --- Acceptance override for ONLY_TOWN_MODE ---
        # If user provided ONLY town (no region, no district) and we have a best candidate,
        # accept it when town similarity is sufficiently high.
        if (not accepted) and only_town_mode and best is not None:
            if best.score_town >= ONLY_TOWN_MIN_TOWN_SCORE:
                accepted = True
                meta_notes.append("only_town_override_accept")
                meta_notes.append("only_town_global_population_choice")
                # keep status weak to be honest; we still output the chosen best row
            else:
                meta_notes.append("only_town_not_accepted_low_town_score")

        # --- Build outputs ---
        out_country = country_in or ("Россия" if accepted or region_matched else None)

        out_region = None
        out_district = None
        out_town = None

        if accepted and best:
            out_region = best.region
            out_district = best.district
            out_town = best.town
            meta_actions.append("accepted_best_reference_row")
        else:
            if region_matched:
                out_region = region_canon
                meta_actions.append("region_validated_only")

        # --- Field validity / corrected flags ---
        region_is_valid = bool(out_region) and (region_matched or (accepted and best and best.score_region >= REGION_CUTOFF))
        district_is_valid = bool(out_district) and accepted
        town_is_valid = bool(out_town) and accepted

        region_is_corrected = (
            (region_in is None and out_region is not None)
            or (region_in is not None and out_region is not None and normalize_region(out_region) != region_in)
        )
        district_is_corrected = (
            (district_in is None and out_district is not None)
            or (district_in is not None and out_district is not None and normalize_district(out_district) != district_in)
        )
        town_is_corrected = (
            (town_in is None and out_town is not None)
            or (town_in is not None and out_town is not None and normalize_town(out_town) != town_in)
        )

        # --- Deterministic consistency guard ---
        if accepted and best and not ref_lookup.is_consistent(best.region, best.district, best.town):
            status = "weak"
            meta_notes.append("reference_inconsistency_guard_triggered")
            out_region = out_district = out_town = None
            region_is_valid = district_is_valid = town_is_valid = False
            accepted = False  # also prevent false validity

        # --- Log candidates (top-3) for debugging if not ok ---
        top_candidates = []
        if candidates:
            for c in candidates[:3]:
                top_candidates.append(
                    {
                        "region": c.region,
                        "district": c.district,
                        "town": c.town,
                        "total": round(c.total, 3),
                        "town_score": round(c.score_town, 3),
                        "population": pop_by_rdt.get((c.region, c.district, c.town), 0.0),
                    }
                )

        log_obj = {
            "region": {"is_valid": region_is_valid, "is_corrected": region_is_corrected},
            "district": {"is_valid": district_is_valid, "is_corrected": district_is_corrected},
            "town": {"is_valid": town_is_valid, "is_corrected": town_is_corrected},
            "_meta": {
                "status": ("ignored" if status == "ignored" else status),
                "notes": meta_notes + (["ambiguous_top_candidates_close"] if ambiguous else []),
                "actions": meta_actions,
                "scores": {"best": round(best_score, 3), "second": round(second_score, 3), "gap": round(gap, 3)},
                "top_candidates": top_candidates if status in {"weak", "ambiguous"} else [],
                "mapping": {
                    "country_col": col_country,
                    "region_col": col_region,
                    "district_col": col_district,
                    "town_col": col_town,
                },
                "tiebreak_by_population": tiebreak_by_population,
                **({"missing_input_columns": missing_cols} if missing_cols else {}),
            },
        }

        out_payload = {
            "out_country": out_country,
            "out_region": out_region,
            "out_district": out_district,
            "out_town": out_town,
        }

        ci = ConfidenceInputs(
            best_score=best_score,
            gap=gap,
            region_given=region_given,
            district_given=district_given,
            town_given=town_given,
            region_matched=region_is_valid,
            district_matched=district_is_valid,
            town_matched=town_is_valid,
            status=("ignored" if status == "ignored" else status),
        )
        conf = confidence(ci)

        results.append(
            {**in_payload, **out_payload, "log": json.dumps(log_obj, ensure_ascii=False), "confidence": conf}
        )

        progress(i / max(total, 1), desc=f"Обработка: {i}/{total}")

    out_df = pd.DataFrame(results)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        out_path = tmp.name
    out_df.to_excel(out_path, index=False)

    stats = {
        "% валидных записей (confidence>=0.6)": round((out_df["confidence"] >= 0.6).mean() * 100, 1),
        "средняя уверенность": round(float(out_df["confidence"].mean()), 3),
    }
    return out_path, stats


# ---------------- UI (вид и порядок сохраняем) ----------------
with gr.Blocks() as demo:
    gr.Markdown(
        "### Валидация и обогащение мест рождения (РФ)\n"
        "1) Загрузите `.xlsx`\n"
        "2) Укажите **точные имена колонок** во входном файле (как в Excel)\n"
        "3) Нажмите **Обработать**\n"
        "4) Скачайте результат: все входные колонки (`in_*`) + выходные (`out_*`) + `log` (JSON) + `confidence`"
    )

    file_in = gr.File(label="Входной файл (.xlsx)", file_types=[".xlsx"])

    gr.Markdown("#### Сопоставление колонок входного файла")
    col_country = gr.Textbox(label="Наименование поля: Страна", placeholder="например: country или Страна", value="country")
    col_region = gr.Textbox(label="Наименование поля: Регион", placeholder="например: region или Регион", value="region")
    col_district = gr.Textbox(label="Наименование поля: Район", placeholder="например: district или Район", value="district")
    col_town = gr.Textbox(label="Наименование поля: Город", placeholder="например: town или Город", value="town")

    with gr.Accordion("Расширенные настройки сопоставления (для опытных пользователей)", open=False):
        gr.Markdown(
            """
            ⚠️ **Внимание:** изменения этих параметров влияют на строгость алгоритма сопоставления  
            и могут существенно изменить результаты.

            **Общий принцип:**
            - большее значение → строже (меньше исправлений, больше пропусков)
            - меньшее значение → мягче (больше исправлений, выше риск ложных совпадений)

            **Рекомендации:**
            - если данных много и качество среднее → оставьте значения по умолчанию
            - если данных мало и важна полнота → можно аккуратно снизить пороги
            - меняйте параметры по одному и анализируйте логи (`log`, `top_candidates`)
            """
        )

        ui_region_cutoff = gr.Slider(
            0.5, 1.0, value=REGION_CUTOFF, step=0.01,
            label="Порог совпадения региона (REGION_CUTOFF)"
        )
        gr.Markdown(
            "Определяет, насколько строго сопоставляется субъект РФ. "
            "Рекомендуется менять **крайне редко**."
        )

        ui_district_cutoff = gr.Slider(
            0.5, 1.0, value=DISTRICT_CUTOFF, step=0.01,
            label="Порог совпадения района (DISTRICT_CUTOFF)"
        )
        gr.Markdown(
            "Влияет на распознавание районов и муниципальных округов. "
            "Снижение может помочь при орфографических ошибках, "
            "но повышает риск неверной привязки."
        )

        ui_town_cutoff = gr.Slider(
            0.5, 1.0, value=TOWN_CUTOFF, step=0.01,
            label="Порог совпадения населённого пункта (TOWN_CUTOFF)"
        )
        gr.Markdown(
            "Основной параметр для городов и сёл. "
            "Чаще всего именно его имеет смысл корректировать."
        )

        ui_candidate_min = gr.Slider(
            0.5, 1.0, value=CANDIDATE_MIN_SCORE, step=0.01,
            label="Минимальный итоговый скор кандидата (CANDIDATE_MIN_SCORE)"
        )
        gr.Markdown(
            "Определяет, считается ли найденный вариант приемлемым в целом. "
            "Снижение увеличивает количество автоматически принятых совпадений."
        )

        ui_gap_min = gr.Slider(
            0.0, 0.3, value=AMBIGUITY_GAP_MIN, step=0.01,
            label="Порог неоднозначности между лучшими вариантами (AMBIGUITY_GAP_MIN)"
        )
        gr.Markdown(
            "Если разница между двумя лучшими вариантами меньше этого значения, "
            "результат считается неоднозначным и помечается в логах."
        )

    btn = gr.Button("Обработать", variant="primary")
    file_out = gr.File(label="Выходной файл (.xlsx)")
    stats = gr.JSON(label="Статистика выполнения")

    btn.click(
        process,
        inputs=[file_in, col_country, col_region, col_district, col_town,
                ui_region_cutoff, ui_district_cutoff, ui_town_cutoff, ui_candidate_min, ui_gap_min],
        outputs=[file_out, stats],
    )

demo.queue()
demo.launch()