"""Monthly aggregation and series preparation."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import pandas as pd

from config import AppConfig, get_config
from models.domain_types import DetectedSchema, ForecastMetric, MonthlySeries, VolumeAggregationMode
from services.numeric_parse import to_datetime_flexible, to_numeric_currency


def _to_month_start(ts: pd.Timestamp) -> datetime:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return datetime(ts.year, ts.month, 1)


def _apply_filters(
    df: pd.DataFrame,
    schema: DetectedSchema,
    segment_column: Optional[str],
    segment_value: Optional[str],
    business_unit: Optional[str],
    product: Optional[str],
) -> pd.DataFrame:
    work = df.copy()
    if segment_column and segment_value is not None:
        if segment_column not in work.columns:
            raise ValueError(f"Segment column not found: {segment_column}")
        work = work[work[segment_column].astype(str) == str(segment_value)]
    if business_unit and schema.business_unit_column:
        work = work[work[schema.business_unit_column].astype(str) == str(business_unit)]
    if product and schema.product_column:
        work = work[work[schema.product_column].astype(str) == str(product)]
    return work


def aggregate_to_monthly(
    df: pd.DataFrame,
    schema: DetectedSchema,
    metric: ForecastMetric = ForecastMetric.VOLUME,
    segment_value: Optional[str] = None,
    segment_column: Optional[str] = None,
    business_unit: Optional[str] = None,
    product: Optional[str] = None,
    history_months: Optional[int] = None,
    config: Optional[AppConfig] = None,
) -> MonthlySeries:
    """
    Aggregate to calendar months for volume or liability.

    Volume supports unique case count, raw row count, or summed pre-aggregated count.
    History is trimmed to the most recent ``history_months`` (default from config).
    """
    cfg = config or get_config()
    window = history_months if history_months is not None else cfg.forecast.history_months

    if df.empty:
        return MonthlySeries([], [], segment_key=segment_value, metric=metric.value, metadata={"empty": True})

    work = _apply_filters(df, schema, segment_column, segment_value, business_unit, product)
    if work.empty:
        return MonthlySeries([], [], segment_key=segment_value, metric=metric.value, metadata={"empty_after_filter": True})

    work = work.copy()
    work["_month"] = to_datetime_flexible(work[schema.date_column]).dt.to_period("M").dt.to_timestamp()

    if metric == ForecastMetric.LIABILITY:
        if not schema.amount_column or schema.amount_column not in work.columns:
            raise ValueError("Liability metric requested but no amount column is mapped")
        work[schema.amount_column] = to_numeric_currency(work[schema.amount_column]).fillna(0.0)
        grouped = work.groupby("_month", as_index=False)[schema.amount_column].sum()
        grouped = grouped.sort_values("_month")
        values = [float(v) for v in grouped[schema.amount_column]]
    else:
        rows: List[dict] = []
        for month, grp in work.groupby("_month"):
            if schema.volume_aggregation_mode == VolumeAggregationMode.UNIQUE_CASE and schema.case_id_column:
                val = float(grp[schema.case_id_column].nunique(dropna=True))
            elif schema.volume_aggregation_mode == VolumeAggregationMode.SUM_COUNT and schema.count_column:
                val = float(to_numeric_currency(grp[schema.count_column]).fillna(0).sum())
            else:
                val = float(len(grp))
            rows.append({"_month": month, "value": val})
        grouped = pd.DataFrame(rows).sort_values("_month")
        values = [float(v) for v in grouped["value"]]

    periods = [_to_month_start(v) for v in grouped["_month"]]
    if window and len(periods) > window:
        periods = periods[-window:]
        values = values[-window:]

    meta = {
        "n_source_rows": int(len(work)),
        "segment_column": segment_column,
        "segment_value": segment_value,
        "business_unit": business_unit,
        "product": product,
        "volume_mode": schema.volume_aggregation_mode.value,
        "history_months_applied": window,
        "metric": metric.value,
    }
    return MonthlySeries(
        period_index=periods,
        values=values,
        segment_key=segment_value,
        metric=metric.value,
        metadata=meta,
    )


def monthly_series_to_frame(series: MonthlySeries) -> pd.DataFrame:
    """Convert MonthlySeries to DataFrame (period, value, optional metric)."""
    return pd.DataFrame(
        {
            "period": series.period_index,
            "value": series.values,
            "metric": series.metric,
        }
    )


def list_segment_values(df: pd.DataFrame, segment_column: str) -> List[str]:
    """Distinct segment labels as strings, sorted."""
    if segment_column not in df.columns:
        return []
    vals = df[segment_column].dropna().astype(str).unique().tolist()
    return sorted(vals)


def list_column_values(df: pd.DataFrame, column: Optional[str]) -> List[str]:
    if not column or column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())
