import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

from core.enrichment import RefLookup
from core.matching import best_from_norm_map, similarity, top_n_from_norm_map
from core.normalization import normalize_country, normalize_district, normalize_region, normalize_town
from core.scoring import ConfidenceInputs, confidence


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


def _is_missing(v: Any) -> bool:
    """Robust missing detection for raw values (None, NaN/NA, '', whitespace)."""
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def _apply_missing(v: Any, fill_missing: bool, missing_value: str) -> Any:
    """
    If fill_missing enabled, replace missing-like values with missing_value.
    If disabled, keep missing as empty (None), so Excel cell remains empty.
    """
    if not fill_missing:
        return None if _is_missing(v) else v
    return missing_value if _is_missing(v) else v


def _max_tie_candidates(
    cands: List[Candidate],
    pop_by_rdt: Dict[Tuple[str, str, str], float],
    town_type_by_rdt: Dict[Tuple[str, str, str], Optional[str]],
) -> List[Dict[str, Any]]:
    """
    If several candidates have exactly the same max score (after rounding),
    return them as a list of dicts. Otherwise return [].
    """
    if not cands:
        return []

    scores = [round(c.total, 4) for c in cands]
    max_s = max(scores)
    tied = [c for c in cands if round(c.total, 4) == max_s]
    if len(tied) <= 1:
        return []

    out: List[Dict[str, Any]] = []
    for c in tied:
        key = (c.region, c.district, c.town)
        out.append(
            {
                "region": c.region,
                "district": c.district,
                "town": c.town,
                "total": round(c.total, 4),
                "population": pop_by_rdt.get(key, 0.0),
                "town_type": town_type_by_rdt.get(key, None),
            }
        )
    return out


def process(
    file_obj,
    col_country: str,
    col_region: str,
    col_district: str,
    col_town: str,
    # --- advanced: missing policy
    fill_missing: bool,
    missing_value: str,
    # --- advanced: cutoffs & selection policy
    region_cutoff: float,
    district_cutoff: float,
    town_cutoff: float,
    candidate_min_score: float,
    ambiguity_gap_min: float,
    progress=gr.Progress(track_tqdm=False),
) -> Tuple[str, Dict[str, Any]]:
    # normalize missing_value from UI
    missing_value = str(missing_value or "").strip()
    if missing_value == "":
        missing_value = "99999999"

    # normalize thresholds (safety)
    region_cutoff = float(region_cutoff)
    district_cutoff = float(district_cutoff)
    town_cutoff = float(town_cutoff)
    candidate_min_score = float(candidate_min_score)
    ambiguity_gap_min = float(ambiguity_gap_min)

    # constants used in behavior
    ONLY_TOWN_MIN_TOWN_SCORE = max(town_cutoff, 0.84)  # keep conservative
    ONLY_TOWN_NEAR_EPS = 0.03

    input_path = _to_input_path(file_obj)
    base_dir = Path(__file__).resolve().parent
    ref_path = base_dir / "regions.csv"

    ref = pd.read_csv(ref_path)
    df_in = pd.read_excel(input_path)

    for c in ["region", "district", "town"]:
        if c not in ref.columns:
            raise ValueError(f"regions.csv must contain column '{c}'")

    ref["region"] = ref["region"].astype(str)
    ref["district"] = ref["district"].astype(str)
    ref["town"] = ref["town"].astype(str)

    # Lookups for enrichment outputs
    pop_by_rdt: Dict[Tuple[str, str, str], float] = {}
    town_type_by_rdt: Dict[Tuple[str, str, str], Optional[str]] = {}

    has_population = "population" in ref.columns
    has_town_type = "town_type" in ref.columns

    if has_population:
        for _, rr in ref.iterrows():
            key = (str(rr["region"]), str(rr["district"]), str(rr["town"]))
            try:
                p = float(rr["population"]) if rr["population"] == rr["population"] else 0.0
            except Exception:
                p = 0.0
            pop_by_rdt[key] = p

    if has_town_type:
        for _, rr in ref.iterrows():
            key = (str(rr["region"]), str(rr["district"]), str(rr["town"]))
            v = rr["town_type"]
            town_type_by_rdt[key] = None if _is_missing(v) else str(v)

    # --- indices ---
    region_norm_to_canon: Dict[str, str] = {}
    district_norm_to_canon_by_region: Dict[str, Dict[str, str]] = {}
    town_rows_by_region: Dict[str, List[Tuple[str, str]]] = {}
    town_norm_to_rows_global: Dict[str, List[Tuple[str, str, str]]] = {}

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

    # --- UI mapping ---
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

        # === RULE: if whole row empty -> do not fill with missing_value ===
        whole_row_empty = (
            _is_missing(country_raw)
            and _is_missing(region_raw)
            and _is_missing(district_raw)
            and _is_missing(town_raw)
        )
        if whole_row_empty:
            log_obj = {
                "region": {"is_valid": False, "is_corrected": False},
                "district": {"is_valid": False, "is_corrected": False},
                "town": {"is_valid": False, "is_corrected": False},
                "_meta": {
                    "status": "empty_row_passthrough",
                    "notes": ["row_is_empty_no_filling_applied"],
                    "actions": [],
                    "scores": {"best": 0.0, "second": 0.0, "gap": 0.0},
                    "top_candidates": [],
                    "mapping": {
                        "country_col": col_country,
                        "region_col": col_region,
                        "district_col": col_district,
                        "town_col": col_town,
                    },
                    "fill_missing": fill_missing,
                    "missing_value": missing_value,
                    "params": {
                        "region_cutoff": region_cutoff,
                        "district_cutoff": district_cutoff,
                        "town_cutoff": town_cutoff,
                        "candidate_min_score": candidate_min_score,
                        "ambiguity_gap_min": ambiguity_gap_min,
                    },
                    **({"missing_input_columns": missing_cols} if missing_cols else {}),
                },
            }
            results.append(
                {
                    **in_payload,
                    "out_country": None,
                    "out_region": None,
                    "out_district": None,
                    "out_town": None,
                    "out_town_type": None,
                    "out_population": None,
                    "candidates": "[]",
                    "log": json.dumps(log_obj, ensure_ascii=False),
                    "confidence": 0.0,
                }
            )
            progress(i / max(total, 1), desc=f"Обработка: {i}/{total}")
            continue

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
        tiebreak_by_population = False

        # fill_missing can be disabled per-row by rules:
        # RULE: if country explicitly not Russia -> do not fill
        fill_missing_effective = fill_missing
        if country_in is not None and country_in != "Россия":
            fill_missing_effective = False

        # ---- non-Russia passthrough ----
        if country_in is not None and country_in != "Россия":
            out_country = _apply_missing(country_raw, fill_missing_effective, missing_value)
            out_region = _apply_missing(region_raw, fill_missing_effective, missing_value)
            out_district = _apply_missing(district_raw, fill_missing_effective, missing_value)
            out_town = _apply_missing(town_raw, fill_missing_effective, missing_value)

            log_obj = {
                "region": {"is_valid": False, "is_corrected": False},
                "district": {"is_valid": False, "is_corrected": False},
                "town": {"is_valid": False, "is_corrected": False},
                "_meta": {
                    "status": "non_russia_passthrough",
                    "notes": [
                        "non_russia_passthrough_return_original_fields",
                        "missing_fill_disabled_for_non_russia",
                    ],
                    "actions": [],
                    "scores": {"best": 0.0, "second": 0.0, "gap": 0.0},
                    "top_candidates": [],
                    "mapping": {
                        "country_col": col_country,
                        "region_col": col_region,
                        "district_col": col_district,
                        "town_col": col_town,
                    },
                    "fill_missing": fill_missing,
                    "fill_missing_effective": fill_missing_effective,
                    "missing_value": missing_value,
                    "params": {
                        "region_cutoff": region_cutoff,
                        "district_cutoff": district_cutoff,
                        "town_cutoff": town_cutoff,
                        "candidate_min_score": candidate_min_score,
                        "ambiguity_gap_min": ambiguity_gap_min,
                    },
                    **({"missing_input_columns": missing_cols} if missing_cols else {}),
                },
            }

            results.append(
                {
                    **in_payload,
                    "out_country": out_country,
                    "out_region": out_region,
                    "out_district": out_district,
                    "out_town": out_town,
                    "out_town_type": None,
                    "out_population": None,
                    "candidates": "[]",
                    "log": json.dumps(log_obj, ensure_ascii=False),
                    "confidence": 0.0,
                }
            )
            progress(i / max(total, 1), desc=f"Обработка: {i}/{total}")
            continue

        # ---- region match & Russia inference ----
        region_match = best_from_norm_map(region_in, region_norm_to_canon, cutoff=region_cutoff)
        region_canon = region_match.match if region_match.match else None
        region_matched = region_canon is not None

        inferred_russia = (country_in is None and region_matched)
        is_russia_context = (country_in == "Россия") or inferred_russia
        if inferred_russia:
            meta_notes.append("country_inferred_russia_by_region")

        # ---- Russia-context & no town mode ----
        if is_russia_context and not town_in:
            out_country = "Россия"
            out_region = region_canon if region_canon else None
            out_district = None
            out_town = None

            if district_in and out_region:
                dist_index = district_norm_to_canon_by_region.get(out_region, {})
                dist_match = best_from_norm_map(district_in, dist_index, cutoff=district_cutoff)
                if dist_match.match:
                    out_district = dist_match.match
                    meta_actions.append("district_corrected_or_validated_by_region")

            meta_actions.append("region_corrected_or_validated" if out_region else "region_missing_or_not_matched")

            out_country = _apply_missing(out_country, fill_missing_effective, missing_value)
            out_region = _apply_missing(out_region, fill_missing_effective, missing_value)
            out_district = _apply_missing(out_district, fill_missing_effective, missing_value)
            out_town = _apply_missing(out_town, fill_missing_effective, missing_value)

            log_obj = {
                "region": {
                    "is_valid": bool(region_matched),
                    "is_corrected": bool(region_matched and region_in and normalize_region(region_canon) != region_in),
                },
                "district": {
                    "is_valid": bool(out_district and (not fill_missing_effective or out_district != missing_value)),
                    "is_corrected": bool(out_district),
                },
                "town": {"is_valid": False, "is_corrected": False},
                "_meta": {
                    "status": "russia_no_town_mode",
                    "notes": meta_notes,
                    "actions": meta_actions,
                    "scores": {"region": round(region_match.score, 3), "district": 0.0, "town": 0.0},
                    "top_candidates": [],
                    "mapping": {
                        "country_col": col_country,
                        "region_col": col_region,
                        "district_col": col_district,
                        "town_col": col_town,
                    },
                    "fill_missing": fill_missing,
                    "fill_missing_effective": fill_missing_effective,
                    "missing_value": missing_value,
                    "params": {
                        "region_cutoff": region_cutoff,
                        "district_cutoff": district_cutoff,
                        "town_cutoff": town_cutoff,
                        "candidate_min_score": candidate_min_score,
                        "ambiguity_gap_min": ambiguity_gap_min,
                    },
                    **({"missing_input_columns": missing_cols} if missing_cols else {}),
                },
            }

            conf = confidence(
                ConfidenceInputs(
                    best_score=min(1.0, 0.6 if region_matched else 0.0),
                    gap=0.0,
                    region_given=region_given,
                    district_given=district_given,
                    town_given=town_given,
                    region_matched=bool(region_matched),
                    district_matched=bool(out_district and (not fill_missing_effective or out_district != missing_value)),
                    town_matched=False,
                    status="weak",
                )
            )

            results.append(
                {
                    **in_payload,
                    "out_country": out_country,
                    "out_region": out_region,
                    "out_district": out_district,
                    "out_town": out_town,
                    "out_town_type": None,
                    "out_population": None,
                    "candidates": "[]",
                    "log": json.dumps(log_obj, ensure_ascii=False),
                    "confidence": conf,
                }
            )
            progress(i / max(total, 1), desc=f"Обработка: {i}/{total}")
            continue

        # ---- candidate generation (town present) ----
        candidates: List[Candidate] = []

        if town_in:
            if region_canon:
                for (d, t) in town_rows_by_region.get(region_canon, []):
                    score_r = 1.0
                    score_t = similarity(town_in, normalize_town(t))
                    score_d = similarity(district_in, normalize_district(d)) if district_in else 0.0
                    candidates.append(Candidate(region_canon, d, t, score_r, score_d, score_t))
            else:
                rows = town_norm_to_rows_global.get(town_in, [])
                if rows:
                    for (r, d, t) in rows:
                        score_r = similarity(region_in, normalize_region(r)) if region_in else 0.0
                        score_t = 1.0
                        score_d = similarity(district_in, normalize_district(d)) if district_in else 0.0
                        candidates.append(Candidate(r, d, t, score_r, score_d, score_t))
                else:
                    meta_notes.append("global_town_fuzzy_fallback")
                    norm_keys = list(town_norm_to_rows_global.keys())
                    tmp_map = {k: k for k in norm_keys}
                    top_keys = top_n_from_norm_map(town_in, tmp_map, n=7)
                    for norm_key, sc_key in top_keys:
                        for (r, d, t) in town_norm_to_rows_global.get(norm_key, []):
                            score_r = similarity(region_in, normalize_region(r)) if region_in else 0.0
                            score_t = sc_key
                            score_d = similarity(district_in, normalize_district(d)) if district_in else 0.0
                            candidates.append(Candidate(r, d, t, score_r, score_d, score_t))
        else:
            status = "weak"
            meta_notes.append("no_town_provided_unexpected")

        # Restrict by district if confident and region matched
        if candidates and district_in and region_canon:
            dist_index = district_norm_to_canon_by_region.get(region_canon, {})
            dist_match = best_from_norm_map(district_in, dist_index, cutoff=district_cutoff)
            if dist_match.match and dist_match.score >= district_cutoff:
                dcanon = dist_match.match
                before = len(candidates)
                candidates = [c for c in candidates if c.district == dcanon]
                after = len(candidates)
                meta_actions.append(f"candidates_restricted_by_district:{before}->{after}")
            else:
                meta_notes.append("district_not_restricting_candidates_due_to_low_confidence")

        if candidates:
            candidates.sort(key=lambda c: c.total, reverse=True)

        # ---- max-score tie => outputs empty (None), regardless of fill_missing_effective ----
        tie_list = _max_tie_candidates(candidates, pop_by_rdt, town_type_by_rdt)
        if tie_list:
            log_obj = {
                "region": {"is_valid": False, "is_corrected": False},
                "district": {"is_valid": False, "is_corrected": False},
                "town": {"is_valid": False, "is_corrected": False},
                "_meta": {
                    "status": "max_score_tie_multiple_candidates",
                    "notes": meta_notes + ["max_score_tie_multiple_candidates"],
                    "actions": meta_actions,
                    "scores": {
                        "best": round(candidates[0].total, 4) if candidates else 0.0,
                        "second": round(candidates[1].total, 4) if len(candidates) > 1 else 0.0,
                        "gap": 0.0,
                    },
                    "top_candidates": tie_list,
                    "mapping": {
                        "country_col": col_country,
                        "region_col": col_region,
                        "district_col": col_district,
                        "town_col": col_town,
                    },
                    "fill_missing": fill_missing,
                    "fill_missing_effective": fill_missing_effective,
                    "missing_value": missing_value,
                    "params": {
                        "region_cutoff": region_cutoff,
                        "district_cutoff": district_cutoff,
                        "town_cutoff": town_cutoff,
                        "candidate_min_score": candidate_min_score,
                        "ambiguity_gap_min": ambiguity_gap_min,
                    },
                    "note": "tie_case_outputs_are_intentionally_empty",
                    **({"missing_input_columns": missing_cols} if missing_cols else {}),
                },
            }

            results.append(
                {
                    **in_payload,
                    "out_country": None,
                    "out_region": None,
                    "out_district": None,
                    "out_town": None,
                    "out_town_type": None,
                    "out_population": None,
                    "candidates": json.dumps(tie_list, ensure_ascii=False),
                    "log": json.dumps(log_obj, ensure_ascii=False),
                    "confidence": 0.0,
                }
            )
            progress(i / max(total, 1), desc=f"Обработка: {i}/{total}")
            continue

        best: Optional[Candidate] = candidates[0] if candidates else None
        second: Optional[Candidate] = candidates[1] if candidates and len(candidates) > 1 else None

        best_score = best.total if best else 0.0
        second_score = second.total if second else 0.0
        gap = max(0.0, best_score - second_score)

        accepted = False
        ambiguous = False

        if best and best_score >= candidate_min_score:
            if second is not None and gap < ambiguity_gap_min:
                ambiguous = True
                status = "ambiguous"
                meta_notes.append("ambiguous_top_candidates_close")

                near = [c for c in candidates if c.total >= (best_score - ONLY_TOWN_NEAR_EPS)]
                if has_population and len(near) > 1:
                    near.sort(key=lambda c: pop_by_rdt.get((c.region, c.district, c.town), 0.0), reverse=True)
                    chosen = near[0]
                    if chosen != best:
                        best = chosen
                        best_score = best.total
                        tiebreak_by_population = True
                        meta_notes.append("population_tiebreak_applied")
                accepted = True
            else:
                accepted = True
                status = "ok"
        else:
            status = "weak"

        if (not accepted) and only_town_mode and best is not None and best.score_town >= ONLY_TOWN_MIN_TOWN_SCORE:
            accepted = True
            meta_notes.append("only_town_override_accept")

        # ---- outputs: do NOT erase raw town/district if not accepted ----
        if accepted and best:
            # NEW RULE: if country was missing but we accepted a reference candidate => restore to "Россия"
            if country_in is None:
                out_country = "Россия"
                meta_actions.append("country_restored_to_russia_by_accepted_candidate")
            else:
                out_country = country_in

            out_region = best.region
            out_district = best.district
            out_town = best.town
            meta_actions.append("accepted_best_reference_row")

            key = (best.region, best.district, best.town)
            out_town_type = town_type_by_rdt.get(key, None)
            out_population = pop_by_rdt.get(key, None)
        else:
            out_country = country_in or ("Россия" if (region_matched or is_russia_context) else None)
            out_region = region_canon if region_matched else region_raw
            out_district = district_raw
            out_town = town_raw
            out_town_type = None
            out_population = None
            if region_matched:
                meta_actions.append("region_validated_only")

        if accepted and best and not ref_lookup.is_consistent(best.region, best.district, best.town):
            status = "weak"
            meta_notes.append("reference_inconsistency_guard_triggered")
            accepted = False
            out_country = country_in or ("Россия" if (region_matched or is_russia_context) else None)
            out_region = region_canon if region_matched else region_raw
            out_district = district_raw
            out_town = town_raw
            out_town_type = None
            out_population = None

        # missing policy applied to normal branches only (effective)
        out_country = _apply_missing(out_country, fill_missing_effective, missing_value)
        out_region = _apply_missing(out_region, fill_missing_effective, missing_value)
        out_district = _apply_missing(out_district, fill_missing_effective, missing_value)
        out_town = _apply_missing(out_town, fill_missing_effective, missing_value)

        region_is_valid = bool(out_region) and (not fill_missing_effective or out_region != missing_value)
        district_is_valid = accepted and bool(out_district) and (not fill_missing_effective or out_district != missing_value)
        town_is_valid = accepted and bool(out_town) and (not fill_missing_effective or out_town != missing_value)

        region_is_corrected = bool(region_in and out_region and normalize_region(out_region) != region_in)
        district_is_corrected = bool(district_in and out_district and normalize_district(out_district) != district_in)
        town_is_corrected = bool(town_in and out_town and normalize_town(out_town) != town_in)

        top_candidates = []
        for c in candidates[:3]:
            key = (c.region, c.district, c.town)
            top_candidates.append(
                {
                    "region": c.region,
                    "district": c.district,
                    "town": c.town,
                    "total": round(c.total, 3),
                    "town_score": round(c.score_town, 3),
                    "population": pop_by_rdt.get(key, 0.0),
                    "town_type": town_type_by_rdt.get(key, None),
                }
            )

        log_obj = {
            "region": {"is_valid": region_is_valid, "is_corrected": region_is_corrected},
            "district": {"is_valid": district_is_valid, "is_corrected": district_is_corrected},
            "town": {"is_valid": town_is_valid, "is_corrected": town_is_corrected},
            "_meta": {
                "status": status,
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
                "fill_missing": fill_missing,
                "fill_missing_effective": fill_missing_effective,
                "missing_value": missing_value,
                "params": {
                    "region_cutoff": region_cutoff,
                    "district_cutoff": district_cutoff,
                    "town_cutoff": town_cutoff,
                    "candidate_min_score": candidate_min_score,
                    "ambiguity_gap_min": ambiguity_gap_min,
                },
                **({"missing_input_columns": missing_cols} if missing_cols else {}),
            },
        }

        conf = confidence(
            ConfidenceInputs(
                best_score=best_score,
                gap=gap,
                region_given=region_given,
                district_given=district_given,
                town_given=town_given,
                region_matched=region_is_valid,
                district_matched=district_is_valid,
                town_matched=town_is_valid,
                status=status,
            )
        )

        results.append(
            {
                **in_payload,
                "out_country": out_country,
                "out_region": out_region,
                "out_district": out_district,
                "out_town": out_town,
                "out_town_type": out_town_type,
                "out_population": out_population,
                "candidates": "[]",
                "log": json.dumps(log_obj, ensure_ascii=False),
                "confidence": conf,
            }
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
        "4) Скачайте результат: все входные колонки (`in_*`) + выходные (`out_*`) + "
        "`out_town_type`, `out_population` + `candidates` + `log` (JSON) + `confidence`"
    )

    file_in = gr.File(label="Входной файл (.xlsx)", file_types=[".xlsx"])

    gr.Markdown("#### Сопоставление колонок входного файла")
    col_country = gr.Textbox(label="Наименование поля: Страна", placeholder="например: country или Страна", value="country")
    col_region = gr.Textbox(label="Наименование поля: Регион", placeholder="например: region или Регион", value="region")
    col_district = gr.Textbox(label="Наименование поля: Район", placeholder="например: district или Район", value="district")
    col_town = gr.Textbox(label="Наименование поля: Город", placeholder="например: town или Город", value="town")

    # --- Advanced: cutoffs ---
    with gr.Accordion("Расширенные настройки сопоставления (cutoffs)", open=False):
        gr.Markdown(
            """
            ⚠️ Эти параметры управляют строгостью сопоставления.
            - больше → строже (меньше автокоррекций, больше пропусков)
            - меньше → мягче (больше автокоррекций, выше риск ложных совпадений)
            """
        )
        ui_region_cutoff = gr.Slider(0.5, 1.0, value=0.86, step=0.01, label="Порог совпадения региона (REGION_CUTOFF)")
        ui_district_cutoff = gr.Slider(0.5, 1.0, value=0.84, step=0.01, label="Порог совпадения района (DISTRICT_CUTOFF)")
        ui_town_cutoff = gr.Slider(0.5, 1.0, value=0.84, step=0.01, label="Порог совпадения города (TOWN_CUTOFF)")
        ui_candidate_min = gr.Slider(0.5, 1.0, value=0.78, step=0.01, label="Минимальный итоговый скор кандидата (CANDIDATE_MIN_SCORE)")
        ui_gap_min = gr.Slider(0.0, 0.3, value=0.06, step=0.01, label="Порог неоднозначности (AMBIGUITY_GAP_MIN)")

    # --- Advanced: missing policy ---
    with gr.Accordion("Расширенные настройки (пропуски)", open=False):
        gr.Markdown(
            """
            Если включено — пустые значения в `out_*` будут заменены на указанную строку.

            **Исключения:**
            1) Если `country` явно не Россия → строка passthrough, пропуски НЕ заполняются.
            2) Если все поля (country/region/district/town) пустые → пропуски НЕ заполняются.
            3) Если `candidates` не пустой (tie max-score) → `out_*` остаются пустыми.
            4) Если `country` был пустой, но выбран лучший кандидат из справочника → `out_country` восстанавливается как "Россия".
            """
        )
        fill_missing = gr.Checkbox(label="Заполнить пропуски", value=True)
        missing_value = gr.Textbox(label="Вместо пропуска заполнять", value="99999999")

    btn = gr.Button("Обработать", variant="primary")
    file_out = gr.File(label="Выходной файл (.xlsx)")
    stats = gr.JSON(label="Статистика выполнения")

    btn.click(
        process,
        inputs=[
            file_in,
            col_country,
            col_region,
            col_district,
            col_town,
            fill_missing,
            missing_value,
            ui_region_cutoff,
            ui_district_cutoff,
            ui_town_cutoff,
            ui_candidate_min,
            ui_gap_min,
        ],
        outputs=[file_out, stats],
    )

demo.queue()
demo.launch()