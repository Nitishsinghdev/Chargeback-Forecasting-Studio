"""Validate ingested CSV rows against a detected schema."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from config import AppConfig, get_config
from models.domain_types import DetectedSchema, ValidationIssue, ValidationReport, VolumeAggregationMode


def _month_range_stats(dates: pd.Series) -> tuple[int, List[str]]:
    if dates.empty:
        return 0, []
    months = dates.dt.to_period("M")
    if months.empty:
        return 0, []
    start, end = months.min(), months.max()
    full = pd.period_range(start, end, freq="M")
    present = set(months.dropna())
    missing = [str(p) for p in full if p not in present]
    return int(len(full)), missing


def _compute_totals(cleaned: pd.DataFrame, schema: DetectedSchema) -> tuple[float, float]:
    total_liability = 0.0
    if schema.amount_column and schema.amount_column in cleaned.columns:
        total_liability = float(pd.to_numeric(cleaned[schema.amount_column], errors="coerce").fillna(0).sum())

    if schema.volume_aggregation_mode == VolumeAggregationMode.UNIQUE_CASE and schema.case_id_column:
        total_volume = float(cleaned[schema.case_id_column].nunique(dropna=True))
    elif schema.volume_aggregation_mode == VolumeAggregationMode.SUM_COUNT and schema.count_column:
        total_volume = float(pd.to_numeric(cleaned[schema.count_column], errors="coerce").fillna(0).sum())
    else:
        total_volume = float(len(cleaned))
    return total_volume, total_liability


def _readiness(is_valid: bool, warnings: List[str], valid_rows: int, min_train: int) -> str:
    if not is_valid or valid_rows == 0:
        return "not_ready"
    if valid_rows < min_train:
        return "needs_review"
    if warnings:
        return "needs_review"
    return "ready"


def validate_dataframe(
    df: pd.DataFrame,
    schema: DetectedSchema,
    config: Optional[AppConfig] = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Clean and validate rows; returns cleaned frame and structured report.

    Invalid rows are dropped; errors vs warnings are recorded explicitly.
    """
    cfg = config or get_config()
    issues: list[ValidationIssue] = []
    empty_report_kwargs = dict(
        uploaded_row_count=0,
        valid_row_count=0,
        rejected_row_count=0,
        duplicate_row_count=0,
        earliest_date=None,
        latest_date=None,
        month_count=0,
        missing_months=[],
        total_volume=0.0,
        total_liability=0.0,
        warnings=[],
        readiness="not_ready",
    )

    if df is None or df.empty:
        report = ValidationReport(
            is_valid=False,
            issues=[
                ValidationIssue(
                    severity="error",
                    code="empty_input",
                    message="Input DataFrame is empty",
                )
            ],
            **empty_report_kwargs,
        )
        return pd.DataFrame(), report

    uploaded = int(len(df))
    work = df.copy()
    work.columns = [str(c) for c in work.columns]

    if schema.date_column not in work.columns:
        issues.append(
            ValidationIssue(
                severity="error",
                code="missing_column",
                message=f"Required date column missing: {schema.date_column}",
                column=schema.date_column,
            )
        )
        report = ValidationReport(is_valid=False, issues=issues, uploaded_row_count=uploaded, **{
            **empty_report_kwargs,
            "rejected_row_count": uploaded,
        })
        return pd.DataFrame(), report

    if schema.amount_column and schema.amount_column not in work.columns:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="amount_column_missing",
                message=f"Configured amount column missing: {schema.amount_column}",
                column=schema.amount_column,
            )
        )

    work["_parsed_date"] = pd.to_datetime(work[schema.date_column], errors="coerce")
    bad_dates = work["_parsed_date"].isna()
    rejected_indexes: List[int] = []
    for idx in work.index[bad_dates]:
        rejected_indexes.append(int(idx))
        if len([i for i in issues if i.code == "invalid_date"]) < 20:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid_date",
                    message="Unparseable date value",
                    column=schema.date_column,
                    row_index=int(idx) if isinstance(idx, int) else None,
                )
            )
    if bad_dates.sum() > 20:
        issues.append(
            ValidationIssue(
                severity="error",
                code="invalid_date_truncated",
                message=f"Additional invalid dates suppressed: {int(bad_dates.sum()) - 20}",
                column=schema.date_column,
            )
        )

    bad_amounts = pd.Series(False, index=work.index)
    if schema.amount_column and schema.amount_column in work.columns:
        work["_parsed_amount"] = pd.to_numeric(work[schema.amount_column], errors="coerce")
        bad_amounts = work["_parsed_amount"].isna()
        for idx in work.index[bad_amounts]:
            if int(idx) not in rejected_indexes:
                rejected_indexes.append(int(idx))
            if len([i for i in issues if i.code == "invalid_amount"]) < 20:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="invalid_amount",
                        message="Unparseable numeric amount",
                        column=schema.amount_column,
                        row_index=int(idx) if isinstance(idx, int) else None,
                    )
                )

    valid_mask = (~bad_dates) & (~bad_amounts)
    cleaned = work.loc[valid_mask].copy()
    cleaned[schema.date_column] = cleaned["_parsed_date"]
    drop_cols = ["_parsed_date"]
    if "_parsed_amount" in cleaned.columns:
        cleaned[schema.amount_column] = cleaned["_parsed_amount"].astype(float)
        drop_cols.append("_parsed_amount")
    cleaned = cleaned.drop(columns=[c for c in drop_cols if c in cleaned.columns])

    if cleaned.empty:
        issues.append(
            ValidationIssue(
                severity="error",
                code="no_valid_rows",
                message="All rows failed validation",
            )
        )

    dup_cols = [schema.date_column]
    if schema.case_id_column and schema.case_id_column in cleaned.columns:
        dup_cols.append(schema.case_id_column)
    elif schema.amount_column and schema.amount_column in cleaned.columns:
        dup_cols.append(schema.amount_column)
    dup_mask = cleaned.duplicated(subset=dup_cols, keep=False) if not cleaned.empty else pd.Series(dtype=bool)
    duplicate_count = int(dup_mask.sum()) if not cleaned.empty else 0
    if duplicate_count:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="duplicate_keys",
                message=f"Found {duplicate_count} duplicate key rows; monthly aggregation will combine",
            )
        )

    earliest = latest = None
    month_count = 0
    missing_months: List[str] = []
    if not cleaned.empty:
        dseries = pd.to_datetime(cleaned[schema.date_column])
        earliest = datetime(dseries.min().year, dseries.min().month, dseries.min().day)
        latest = datetime(dseries.max().year, dseries.max().month, dseries.max().day)
        month_count, missing_months = _month_range_stats(dseries)

    total_volume, total_liability = _compute_totals(cleaned, schema) if not cleaned.empty else (0.0, 0.0)

    warnings = [i.message for i in issues if i.severity == "warning"]
    has_errors = any(i.severity == "error" for i in issues) or cleaned.empty
    readiness = _readiness(not has_errors, warnings, int(len(cleaned)), cfg.validation.min_train_months)

    sample_limit = cfg.csv.rejected_sample_limit
    rejected_sample: List[Dict[str, Any]] = []
    if rejected_indexes:
        sample_idx = rejected_indexes[:sample_limit]
        rejected_sample = work.loc[sample_idx].astype(str).to_dict(orient="records")

    report = ValidationReport(
        is_valid=not has_errors,
        issues=issues,
        uploaded_row_count=uploaded,
        valid_row_count=int(len(cleaned)),
        rejected_row_count=int(uploaded - len(cleaned)),
        duplicate_row_count=duplicate_count,
        earliest_date=earliest,
        latest_date=latest,
        month_count=month_count,
        missing_months=missing_months,
        total_volume=total_volume,
        total_liability=total_liability,
        warnings=warnings,
        readiness=readiness,
        rejected_row_indexes=rejected_indexes[:sample_limit],
        rejected_records_sample=rejected_sample,
    )
    return cleaned, report
