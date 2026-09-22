"""Dynamic CSV schema detection with role profiles and user overrides."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import AppConfig, get_config
from models.domain_types import (
    ColumnProfile,
    DetectedSchema,
    SchemaMappingOverride,
    VolumeAggregationMode,
)

_NORMALIZE = re.compile(r"[^a-z0-9]+")

ROLE_AMOUNT = "amount"
ROLE_COUNT = "count"
ROLE_DATE = "date"
ROLE_BUSINESS_UNIT = "business_unit"
ROLE_PRODUCT = "product"
ROLE_STATUS = "status"
ROLE_CASE_ID = "case_id"
ROLE_REASON_CODE = "reason_code"
ROLE_CATEGORY = "category"
ROLE_CATEGORY_CANDIDATE = "category_candidate"


def _normalize_name(name: str) -> str:
    return _NORMALIZE.sub("_", str(name).strip().lower()).strip("_")


def _score_name_match(column: str, candidates: Sequence[str]) -> float:
    norm = _normalize_name(column)
    if not norm:
        return 0.0
    best = 0.0
    for cand in candidates:
        c = _normalize_name(cand)
        if norm == c:
            return 1.0
        if c in norm or norm in c:
            best = max(best, 0.75)
        if norm.endswith(c) or norm.startswith(c):
            best = max(best, 0.6)
    return best


def _date_parse_ratio(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    sample = series.head(min(len(series), 5000))
    parsed = pd.to_datetime(sample, errors="coerce", utc=False)
    return float(parsed.notna().mean())


def _numeric_ratio(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    sample = series.head(min(len(series), 5000))
    converted = pd.to_numeric(sample, errors="coerce")
    return float(converted.notna().mean())


def _sample_values(series: pd.Series, limit: int = 5) -> List[str]:
    vals = series.dropna().astype(str).unique().tolist()[:limit]
    return vals


def _rank_by_role(
    df: pd.DataFrame,
    candidates: Sequence[str],
    *,
    require_numeric: bool = False,
    cfg: AppConfig,
    exclude: Sequence[str],
) -> List[Tuple[str, float, ColumnProfile]]:
    exclude_set = set(exclude)
    ranked: List[Tuple[str, float, ColumnProfile]] = []
    for col in df.columns:
        col_s = str(col)
        if col_s in exclude_set:
            continue
        name_score = _score_name_match(col_s, candidates)
        parse_ratio = _date_parse_ratio(df[col]) if not require_numeric else None
        num_ratio = _numeric_ratio(df[col]) if require_numeric else None
        if require_numeric:
            if (num_ratio or 0.0) < cfg.csv.min_numeric_ratio and name_score < 0.55:
                continue
            combined = 0.5 * (num_ratio or 0.0) + 0.5 * name_score
            role = candidates[0] if candidates else "numeric"
        else:
            if (parse_ratio or 0.0) < cfg.csv.min_date_parse_ratio and name_score < 0.55:
                continue
            combined = 0.55 * (parse_ratio or 0.0) + 0.45 * name_score
            role = ROLE_DATE
        profile = ColumnProfile(
            column_name=col_s,
            inferred_role=role,
            confidence=combined,
            parse_ratio=parse_ratio,
            numeric_ratio=num_ratio,
            nunique=int(df[col].nunique(dropna=True)),
            sample_values=_sample_values(df[col]),
        )
        ranked.append((col_s, combined, profile))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _rank_category_candidates(
    df: pd.DataFrame,
    cfg: AppConfig,
    reserved: Sequence[str],
) -> List[Tuple[str, float, ColumnProfile]]:
    reserved_set = set(reserved)
    ranked: List[Tuple[str, float, ColumnProfile]] = []
    for col in df.columns:
        col_s = str(col)
        if col_s in reserved_set:
            continue
        series = df[col]
        nunique = int(series.nunique(dropna=True))
        if nunique <= 1 or nunique > max(500, len(series) * 0.5):
            continue
        name_score = _score_name_match(col_s, cfg.csv.category_column_candidates)
        name_score = max(name_score, _score_name_match(col_s, cfg.csv.segment_column_candidates))
        cardinality_score = 1.0 - min(1.0, abs(np.log1p(nunique) - np.log1p(20)) / 5.0)
        combined = 0.6 * name_score + 0.4 * cardinality_score
        if combined < 0.3 and name_score < 0.5:
            continue
        profile = ColumnProfile(
            column_name=col_s,
            inferred_role=ROLE_CATEGORY_CANDIDATE,
            confidence=combined,
            nunique=nunique,
            sample_values=_sample_values(series),
        )
        ranked.append((col_s, combined, profile))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _is_cleared_mapping(value: Optional[str]) -> bool:
    return value is not None and str(value).strip() == ""


def _rank_identifier_columns(
    df: pd.DataFrame,
    candidates: Sequence[str],
    *,
    role: str,
    exclude: Sequence[str],
    min_score: float = 0.35,
    cfg: Optional[AppConfig] = None,
) -> List[Tuple[str, float, ColumnProfile]]:
    """Rank identifier-like columns by name; do not use date-parse heuristics."""
    exclude_set = set(exclude)
    ranked: List[Tuple[str, float, ColumnProfile]] = []
    n_rows = max(1, len(df))
    for col in df.columns:
        col_s = str(col)
        if col_s in exclude_set:
            continue
        name_score = _score_name_match(col_s, candidates)
        if name_score < min_score:
            continue
        if role == ROLE_CASE_ID and cfg is not None:
            count_name_score = _score_name_match(col_s, cfg.csv.count_column_candidates)
            norm = _normalize_name(col_s)
            if count_name_score >= name_score and (
                "count" in norm.split("_") or norm.endswith("_count") or norm.startswith("count_")
            ):
                continue
        nunique = int(df[col].nunique(dropna=True))
        cardinality = min(1.0, nunique / n_rows)
        combined = 0.85 * name_score + 0.15 * cardinality
        profile = ColumnProfile(
            column_name=col_s,
            inferred_role=role,
            confidence=combined,
            nunique=nunique,
            sample_values=_sample_values(df[col]),
        )
        ranked.append((col_s, combined, profile))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _rank_count_columns(
    df: pd.DataFrame,
    cfg: AppConfig,
    exclude: Sequence[str],
) -> List[Tuple[str, float, ColumnProfile]]:
    """Numeric count columns; never pick reason/case identifier columns."""
    exclude_set = set(exclude)
    ranked: List[Tuple[str, float, ColumnProfile]] = []
    for col in df.columns:
        col_s = str(col)
        if col_s in exclude_set:
            continue
        if _score_name_match(col_s, cfg.csv.reason_code_candidates) >= 0.55:
            continue
        case_name_score = _score_name_match(col_s, cfg.csv.case_id_candidates)
        name_score = _score_name_match(col_s, cfg.csv.count_column_candidates)
        norm = _normalize_name(col_s)
        is_count_like = (
            name_score >= 0.55
            and ("count" in norm.split("_") or norm.endswith("_count") or norm.startswith("count_"))
        )
        if case_name_score >= 0.55 and not is_count_like:
            continue
        num_ratio = _numeric_ratio(df[col])
        if num_ratio < cfg.csv.min_numeric_ratio and name_score < 0.55:
            continue
        combined = 0.5 * num_ratio + 0.5 * name_score
        profile = ColumnProfile(
            column_name=col_s,
            inferred_role=ROLE_COUNT,
            confidence=combined,
            numeric_ratio=num_ratio,
            nunique=int(df[col].nunique(dropna=True)),
            sample_values=_sample_values(df[col]),
        )
        ranked.append((col_s, combined, profile))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _pick_best(
    ranked: List[Tuple[str, float, ColumnProfile]],
    min_score: float = 0.35,
) -> Optional[Tuple[str, ColumnProfile]]:
    if not ranked:
        return None
    col, score, profile = ranked[0]
    if score < min_score:
        return None
    return col, profile


def _infer_volume_mode(
    case_id: Optional[str],
    count_col: Optional[str],
    count_profile: Optional[ColumnProfile],
    override: Optional[str],
) -> VolumeAggregationMode:
    if override:
        try:
            return VolumeAggregationMode(override)
        except ValueError:
            pass
    if case_id:
        return VolumeAggregationMode.UNIQUE_CASE
    if count_col and count_profile and (count_profile.numeric_ratio or 0) >= 0.7:
        return VolumeAggregationMode.SUM_COUNT
    return VolumeAggregationMode.ROW_COUNT


def _resolve_mapping(
    override_val: Optional[str],
    current: Optional[str],
    df_cols: Optional[set[str]],
    *,
    required: bool = False,
) -> Optional[str]:
    if override_val is None:
        return current
    if _is_cleared_mapping(override_val):
        return None
    col = str(override_val).strip()
    if df_cols is not None and col not in df_cols:
        raise ValueError(f"Override column not found: {col}")
    return col


def apply_schema_overrides(
    schema: DetectedSchema,
    overrides: SchemaMappingOverride,
    df: Optional[pd.DataFrame] = None,
) -> DetectedSchema:
    """Apply user mapping overrides while preserving profiles where possible."""
    df_cols = {str(c) for c in df.columns} if df is not None else None

    date_col = _resolve_mapping(overrides.date_column, schema.date_column, df_cols, required=True)
    if not date_col:
        raise ValueError("date_column is required")
    amount_col = _resolve_mapping(overrides.amount_column, schema.amount_column, df_cols)
    count_col = _resolve_mapping(overrides.count_column, schema.count_column, df_cols)
    bu_col = _resolve_mapping(overrides.business_unit_column, schema.business_unit_column, df_cols)
    product_col = _resolve_mapping(overrides.product_column, schema.product_column, df_cols)
    status_col = _resolve_mapping(overrides.status_column, schema.status_column, df_cols)
    case_col = _resolve_mapping(overrides.case_id_column, schema.case_id_column, df_cols)
    reason_col = _resolve_mapping(overrides.reason_code_column, schema.reason_code_column, df_cols)
    category_col = _resolve_mapping(overrides.category_column, schema.category_column, df_cols)

    volume_mode = _infer_volume_mode(
        case_col,
        count_col,
        schema.column_profiles.get(count_col) if count_col else None,
        overrides.volume_aggregation_mode,
    )

    segment_cols = list(
        dict.fromkeys(
            [c for c in [category_col, bu_col, product_col, reason_col] if c]
            + schema.category_candidate_columns
        )
    )

    notes = list(schema.detection_notes) + ["user_overrides_applied"]
    profiles = dict(schema.column_profiles)
    for col, role in [
        (date_col, ROLE_DATE),
        (amount_col, ROLE_AMOUNT),
        (count_col, ROLE_COUNT),
        (bu_col, ROLE_BUSINESS_UNIT),
        (product_col, ROLE_PRODUCT),
        (status_col, ROLE_STATUS),
        (case_col, ROLE_CASE_ID),
        (reason_col, ROLE_REASON_CODE),
        (category_col, ROLE_CATEGORY),
    ]:
        if col and col not in profiles:
            profiles[col] = ColumnProfile(column_name=col, inferred_role=role, confidence=1.0)

    return DetectedSchema(
        date_column=date_col,
        amount_column=amount_col,
        count_column=count_col,
        business_unit_column=bu_col,
        product_column=product_col,
        status_column=status_col,
        case_id_column=case_col,
        reason_code_column=reason_col,
        category_column=category_col,
        category_candidate_columns=schema.category_candidate_columns,
        segment_columns=segment_cols,
        column_profiles=profiles,
        volume_aggregation_mode=volume_mode,
        all_columns=schema.all_columns,
        row_count=schema.row_count,
        detection_notes=notes,
        user_overrides_applied=True,
    )


def detect_csv_schema(
    df: pd.DataFrame,
    config: Optional[AppConfig] = None,
    overrides: Optional[SchemaMappingOverride] = None,
) -> DetectedSchema:
    """
    Infer semantic CSV roles with per-column confidence profiles.

    Amount is optional; volume-only datasets are supported via row/case/count modes.
    """
    cfg = config or get_config()
    notes: List[str] = []
    if df is None or df.empty:
        raise ValueError("Cannot detect schema on empty DataFrame")

    work = df.copy()
    work.columns = [str(c) for c in work.columns]
    profiles: Dict[str, ColumnProfile] = {}

    date_rank = _rank_by_role(
        work,
        cfg.csv.date_column_candidates,
        require_numeric=False,
        cfg=cfg,
        exclude=[],
    )
    if not date_rank:
        raise ValueError("No date-like column detected")
    date_pick = _pick_best(date_rank, min_score=0.4)
    assert date_pick is not None
    date_col, date_profile = date_pick
    date_profile = ColumnProfile(
        column_name=date_col,
        inferred_role=ROLE_DATE,
        confidence=date_profile.confidence,
        parse_ratio=date_profile.parse_ratio,
        nunique=date_profile.nunique,
        sample_values=date_profile.sample_values,
    )
    profiles[date_col] = date_profile
    notes.append(f"date_column={date_col};confidence={date_profile.confidence:.3f}")

    reserved = [date_col]
    amount_rank = _rank_by_role(
        work,
        cfg.csv.amount_column_candidates,
        require_numeric=True,
        cfg=cfg,
        exclude=reserved,
    )
    amount_col: Optional[str] = None
    amount_pick = _pick_best(amount_rank, min_score=0.35)
    if amount_pick:
        amount_col, amount_profile = amount_pick
        amount_profile = ColumnProfile(
            column_name=amount_col,
            inferred_role=ROLE_AMOUNT,
            confidence=amount_profile.confidence,
            numeric_ratio=amount_profile.numeric_ratio,
            nunique=amount_profile.nunique,
            sample_values=amount_profile.sample_values,
        )
        profiles[amount_col] = amount_profile
        reserved.append(amount_col)
        notes.append(f"amount_column={amount_col};confidence={amount_profile.confidence:.3f}")
    else:
        notes.append("amount_column_not_detected_volume_only_ok")

    case_col: Optional[str] = None
    case_rank = _rank_identifier_columns(
        work,
        cfg.csv.case_id_candidates,
        role=ROLE_CASE_ID,
        exclude=reserved,
        min_score=0.45,
        cfg=cfg,
    )
    case_pick = _pick_best(case_rank, min_score=0.45)
    if case_pick:
        case_col, case_profile = case_pick
        profiles[case_col] = case_profile
        reserved.append(case_col)
        notes.append(f"case_id_column={case_col};confidence={case_profile.confidence:.3f}")

    reason_col: Optional[str] = None
    reason_rank = _rank_identifier_columns(
        work,
        cfg.csv.reason_code_candidates,
        role=ROLE_REASON_CODE,
        exclude=reserved,
        min_score=0.4,
    )
    reason_pick = _pick_best(reason_rank, min_score=0.4)
    if reason_pick:
        reason_col, reason_profile = reason_pick
        profiles[reason_col] = reason_profile
        reserved.append(reason_col)

    count_col: Optional[str] = None
    count_rank = _rank_count_columns(work, cfg, exclude=reserved)
    count_pick = _pick_best(count_rank, min_score=0.35)
    if count_pick:
        count_col, count_profile = count_pick
        profiles[count_col] = count_profile
        reserved.append(count_col)

    role_specs = [
        (cfg.csv.business_unit_candidates, ROLE_BUSINESS_UNIT),
        (cfg.csv.product_column_candidates, ROLE_PRODUCT),
        (cfg.csv.status_column_candidates, ROLE_STATUS),
        (cfg.csv.category_column_candidates, ROLE_CATEGORY),
    ]
    assigned: Dict[str, Optional[str]] = {
        ROLE_BUSINESS_UNIT: None,
        ROLE_PRODUCT: None,
        ROLE_STATUS: None,
        ROLE_CATEGORY: None,
    }

    for candidates, role in role_specs:
        ranked = _rank_identifier_columns(
            work,
            candidates,
            role=role,
            exclude=reserved,
            min_score=0.4,
        )
        pick = _pick_best(ranked, min_score=0.4)
        if pick:
            col, prof = pick
            profiles[col] = prof
            assigned[role] = col
            reserved.append(col)

    cat_candidates_rank = _rank_category_candidates(work, cfg, reserved)
    category_candidate_columns = [
        c for c, _, _ in cat_candidates_rank[:8] if c not in reserved
    ]
    for col, score, prof in cat_candidates_rank[:8]:
        if col not in reserved:
            profiles[col] = prof

    volume_mode = _infer_volume_mode(
        case_col,
        count_col,
        profiles.get(count_col) if count_col else None,
        None,
    )

    segment_columns = list(
        dict.fromkeys(
                [
                    c
                    for c in [
                        assigned[ROLE_CATEGORY],
                        assigned[ROLE_BUSINESS_UNIT],
                        assigned[ROLE_PRODUCT],
                        reason_col,
                    ]
                    if c
                ]
            + category_candidate_columns
        )
    )

    schema = DetectedSchema(
        date_column=date_col,
        amount_column=amount_col,
        count_column=count_col,
        business_unit_column=assigned[ROLE_BUSINESS_UNIT],
        product_column=assigned[ROLE_PRODUCT],
        status_column=assigned[ROLE_STATUS],
        case_id_column=case_col,
        reason_code_column=reason_col,
        category_column=assigned[ROLE_CATEGORY],
        category_candidate_columns=category_candidate_columns,
        segment_columns=segment_columns,
        column_profiles=profiles,
        volume_aggregation_mode=volume_mode,
        all_columns=list(work.columns),
        row_count=int(len(work)),
        detection_notes=notes,
    )

    if overrides is not None:
        schema = apply_schema_overrides(schema, overrides, df=work)
    return schema
