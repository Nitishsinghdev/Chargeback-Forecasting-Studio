"""Orchestrates ingestion, modeling, events, comparisons, and KPIs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from config import AppConfig, get_config
from models.analysis_metadata import AnalysisMetadataStore, persist_forecast_run
from models.domain_types import (
    ForecastExplanation,
    ForecastKpis,
    ForecastMetric,
    ForecastPoint,
    ForecastRunResult,
    MetricForecastBundle,
    ModelKind,
    MonthlySeries,
    ParsedEvent,
    SameMonthHistoricalStats,
    SchemaMappingOverride,
    SegmentComparison,
    event_has_adjustable_magnitude,
)
from services.csv_ingest import read_csv_limited
from services.csv_schema import detect_csv_schema
from services.csv_validation import validate_dataframe
from services.event_features import apply_event_adjustments, exogenous_for_sarimax
from services.event_parser import parse_event_batch
from services.monthly_processor import aggregate_to_monthly, list_column_values, list_segment_values, monthly_series_to_frame
from services.sarimax_search import forecast_with_selection, select_sarimax_model

FileLike = Union[str, Any]


def _next_months(last_period: datetime, count: int) -> List[datetime]:
    start = pd.Timestamp(last_period).to_period("M") + 1
    return [datetime(p.year, p.month, 1) for p in pd.period_range(start, periods=count, freq="M")]


def _pct_change(base: float, new: float) -> Optional[float]:
    if not np.isfinite(base) or abs(base) < 1e-12:
        return None
    return float((new - base) / base * 100.0)


def _same_month_history(
    history: MonthlySeries,
    forecast_period: datetime,
    max_years: int = 3,
) -> SameMonthHistoricalStats:
    target_month = forecast_period.month
    pairs = [
        (p, v)
        for p, v in zip(history.period_index, history.values)
        if p.month == target_month and p < forecast_period
    ]
    pairs.sort(key=lambda x: x[0], reverse=True)
    pairs = pairs[:max_years]
    pairs.reverse()
    values = [float(v) for _, v in pairs]
    periods = [p for p, _ in pairs]
    avg = float(np.mean(values)) if values else None
    mn = float(np.min(values)) if values else None
    mx = float(np.max(values)) if values else None
    trend = None
    if len(values) >= 2 and abs(values[0]) > 1e-12:
        trend = float((values[-1] - values[0]) / abs(values[0]) * 100.0)
    return SameMonthHistoricalStats(
        calendar_month=target_month,
        historical_values=values,
        historical_periods=periods,
        avg=avg,
        min=mn,
        max=mx,
        three_year_trend_pct=trend,
        years_available=len(values),
    )


def _build_segment_comparisons(
    history: MonthlySeries,
    forecasts: List[ForecastPoint],
    segment_label: str,
    metric: ForecastMetric,
) -> List[SegmentComparison]:
    out: List[SegmentComparison] = []
    for fp in forecasts:
        stats = _same_month_history(history, fp.period)
        ref_val = stats.historical_values[-1] if stats.historical_values else None
        ref_period = stats.historical_periods[-1] if stats.historical_periods else fp.period
        if ref_val is None:
            continue
        out.append(
            SegmentComparison(
                segment=segment_label,
                metric=metric.value,
                reference_period=ref_period,
                reference_value=float(ref_val),
                forecast_period=fp.period,
                forecast_baseline=float(fp.baseline),
                forecast_adjusted=float(fp.adjusted),
                delta_baseline=float(fp.baseline - ref_val),
                delta_adjusted=float(fp.adjusted - ref_val),
                pct_change_baseline=_pct_change(ref_val, fp.baseline),
                pct_change_adjusted=_pct_change(ref_val, fp.adjusted),
                same_month_stats=stats,
            )
        )
    return out


def _filter_events(
    events: Sequence[ParsedEvent],
    *,
    require_approved: bool,
) -> List[ParsedEvent]:
    out: List[ParsedEvent] = []
    for e in events:
        if require_approved and not e.approved:
            continue
        out.append(e)
    return out


def _build_explanations(
    selection,
    events: Sequence[ParsedEvent],
    segment_label: str,
    metric: ForecastMetric,
) -> ForecastExplanation:
    model_rationale = (
        f"Selected {selection.model_kind.value} for {metric.value} "
        f"with validation strategy {selection.validation_strategy.value}."
    )
    if selection.order:
        model_rationale += f" SARIMAX order={selection.order}, seasonal={selection.seasonal_order}."
    if selection.fallback_reason:
        model_rationale += f" Fallback reason: {selection.fallback_reason}."

    applied = [e for e in events if event_has_adjustable_magnitude(e)]
    if applied:
        event_rationale = (
            f"Applied {len(applied)} magnitude-bearing event(s) to {metric.value} baseline for '{segment_label}'."
        )
    else:
        event_rationale = "No magnitude-bearing events; adjusted forecast equals baseline."

    val = selection.metrics
    val_parts = []
    if val.mape is not None:
        val_parts.append(f"MAPE={val.mape:.2f}%")
    if val.rmse is not None:
        val_parts.append(f"RMSE={val.rmse:.2f}")
    validation_rationale = "Validation metrics: " + (", ".join(val_parts) if val_parts else "insufficient holdout data")

    caveats: List[str] = list(val.notes)
    if selection.fallback_reason:
        caveats.append(selection.fallback_reason)

    summary = (
        f"{len(applied)} event adjustment(s); metric={metric.value}; "
        f"model={selection.model_kind.value}; segment={segment_label}."
    )
    return ForecastExplanation(
        summary=summary,
        model_rationale=model_rationale,
        event_rationale=event_rationale,
        validation_rationale=validation_rationale,
        caveats=caveats,
    )


def _build_kpis(
    forecasts: List[ForecastPoint],
    selection,
    metric: ForecastMetric,
) -> ForecastKpis:
    baseline_total = float(sum(f.baseline for f in forecasts))
    adjusted_total = float(sum(f.adjusted for f in forecasts))
    impact = float(sum(f.event_adjustment for f in forecasts))
    n = max(1, len(forecasts))
    return ForecastKpis(
        total_baseline_horizon=baseline_total,
        total_adjusted_horizon=adjusted_total,
        total_event_impact=impact,
        avg_monthly_baseline=baseline_total / n,
        avg_monthly_adjusted=adjusted_total / n,
        model_kind=selection.model_kind,
        fallback_used=selection.model_kind != ModelKind.SARIMAX or selection.fallback_reason is not None,
        fallback_reason=selection.fallback_reason,
        validation_mape=selection.metrics.mape,
        metric=metric.value,
    )


class ForecastOrchestrator:
    """
    High-level API for Flask routes: load CSV, forecast, optional persistence.

    Supports separate volume and liability paths plus per-segment forecast methods.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or get_config()
        self.metadata_store = AnalysisMetadataStore(self.config)

    def prepare_dataframe(
        self,
        source: FileLike,
        schema_overrides: Optional[SchemaMappingOverride] = None,
        **read_csv_kwargs,
    ) -> tuple[pd.DataFrame, Any, Any]:
        """Load, detect schema (with optional overrides), validate."""
        raw = read_csv_limited(source, config=self.config, **read_csv_kwargs)
        schema = detect_csv_schema(raw, config=self.config, overrides=schema_overrides)
        cleaned, report = validate_dataframe(raw, schema, config=self.config)
        return cleaned, schema, report

    def forecast_metric_path(
        self,
        cleaned: pd.DataFrame,
        schema,
        metric: ForecastMetric,
        events: Sequence[ParsedEvent],
        *,
        segment_column: Optional[str] = None,
        segment_value: Optional[str] = None,
        business_unit: Optional[str] = None,
        product: Optional[str] = None,
        segment_label: Optional[str] = None,
    ) -> MetricForecastBundle:
        """Run modeling pipeline for a single metric and segment context."""
        if metric == ForecastMetric.LIABILITY and not schema.supports_liability:
            raise ValueError("Liability forecast requested but schema has no amount column")

        history = aggregate_to_monthly(
            cleaned,
            schema,
            metric=metric,
            segment_value=segment_value,
            segment_column=segment_column,
            business_unit=business_unit,
            product=product,
            config=self.config,
        )
        if len(history.values) < 2:
            raise ValueError(f"Insufficient monthly history for metric={metric.value}")

        horizon = self.config.forecast.horizon_months
        forecast_periods = _next_months(history.period_index[-1], horizon)
        label = segment_label or (str(segment_value) if segment_value is not None else "all")
        if business_unit:
            label = f"{label}|bu={business_unit}"
        if product:
            label = f"{label}|product={product}"

        exog_full = exogenous_for_sarimax(
            history,
            forecast_periods,
            events,
            segment_filter=str(segment_value) if segment_value is not None else None,
            metric=metric.value,
            business_unit=business_unit,
            product=product,
        )
        exog_hist = exog_full[: len(history.values)] if exog_full is not None else None

        selection = select_sarimax_model(history.values, exog=exog_hist, config=self.config)
        exog_future = exog_full[-horizon:] if exog_full is not None and len(exog_full) >= horizon else None
        baseline, intervals = forecast_with_selection(
            history.values,
            horizon,
            selection,
            exog_history=exog_hist,
            exog_future=exog_future,
            config=self.config,
        )

        mults, adds = apply_event_adjustments(
            baseline,
            forecast_periods,
            events,
            history,
            segment_filter=str(segment_value) if segment_value is not None else None,
            metric=metric.value,
            business_unit=business_unit,
            product=product,
        )

        lower_band: Optional[List[float]] = None
        upper_band: Optional[List[float]] = None
        if intervals is not None:
            lower_band, upper_band = intervals

        forecasts: List[ForecastPoint] = []
        for i, period in enumerate(forecast_periods):
            b = float(baseline[i])
            adj = float(b * mults[i] + adds[i])
            lo = up = None
            if lower_band is not None and upper_band is not None:
                lo = float(lower_band[i] * mults[i] + adds[i])
                up = float(upper_band[i] * mults[i] + adds[i])
            forecasts.append(
                ForecastPoint(
                    period=period,
                    baseline=b,
                    adjusted=adj,
                    lower=lo,
                    upper=up,
                    event_adjustment=adj - b,
                    metric=metric.value,
                )
            )

        comparisons = _build_segment_comparisons(history, forecasts, label, metric)
        explanations = _build_explanations(selection, events, label, metric)
        kpis = _build_kpis(forecasts, selection, metric)
        return MetricForecastBundle(
            metric=metric,
            forecasts=forecasts,
            model_selection=selection,
            monthly_history=history,
            segment_comparisons=comparisons,
            explanations=explanations,
            kpis=kpis,
        )

    def run_from_dataframe(
        self,
        df: pd.DataFrame,
        schema=None,
        event_texts: Optional[Sequence[str]] = None,
        events: Optional[Sequence[ParsedEvent]] = None,
        segment_column: Optional[str] = None,
        segment_value: Optional[str] = None,
        business_unit: Optional[str] = None,
        product: Optional[str] = None,
        metrics: Optional[Sequence[ForecastMetric]] = None,
        require_approved_events: bool = False,
        schema_overrides: Optional[SchemaMappingOverride] = None,
    ) -> ForecastRunResult:
        """Run full pipeline; produces volume and liability bundles when available."""
        if schema is None:
            schema = detect_csv_schema(df, config=self.config, overrides=schema_overrides)
        elif schema_overrides is not None:
            from services.csv_schema import apply_schema_overrides

            schema = apply_schema_overrides(schema, schema_overrides, df=df)

        cleaned, report = validate_dataframe(df, schema, config=self.config)
        if not report.is_valid:
            raise ValueError(f"Validation failed with {len(report.issues)} issue(s)")

        seg_col = segment_column
        seg_val = segment_value
        if seg_col is None and schema.segment_columns:
            seg_col = schema.segment_columns[0]
        if seg_col == schema.case_id_column:
            seg_col = next(
                (c for c in schema.segment_columns if c != schema.case_id_column),
                None,
            )
        if seg_col and seg_val is None:
            values = list_segment_values(cleaned, seg_col)
            seg_val = values[0] if values else None

        if events is None:
            events = parse_event_batch(
                event_texts or [],
                segment_candidates=list_segment_values(cleaned, seg_col) if seg_col else [],
                business_units=list_column_values(cleaned, schema.business_unit_column),
                products=list_column_values(cleaned, schema.product_column),
            )
        events = _filter_events(events, require_approved=require_approved_events)

        if metrics is None:
            requested = [ForecastMetric.VOLUME]
            if schema.supports_liability:
                requested.append(ForecastMetric.LIABILITY)
        else:
            requested = list(metrics)

        bundles: Dict[ForecastMetric, MetricForecastBundle] = {}
        for metric in requested:
            if metric == ForecastMetric.LIABILITY and not schema.supports_liability:
                continue
            try:
                bundles[metric] = self.forecast_metric_path(
                    cleaned,
                    schema,
                    metric,
                    events,
                    segment_column=seg_col,
                    segment_value=seg_val,
                    business_unit=business_unit,
                    product=product,
                )
            except ValueError:
                if metric == ForecastMetric.VOLUME:
                    raise

        primary = ForecastMetric.VOLUME if ForecastMetric.VOLUME in bundles else next(iter(bundles))
        primary_bundle = bundles[primary]

        return ForecastRunResult(
            forecasts=primary_bundle.forecasts,
            model_selection=primary_bundle.model_selection,
            segment_comparisons=primary_bundle.segment_comparisons,
            explanations=primary_bundle.explanations,
            kpis=primary_bundle.kpis,
            monthly_history=primary_bundle.monthly_history,
            events_applied=events,
            primary_metric=primary,
            volume=bundles.get(ForecastMetric.VOLUME),
            liability=bundles.get(ForecastMetric.LIABILITY),
        )

    def run_segment_forecasts(
        self,
        df: pd.DataFrame,
        segment_column: str,
        schema=None,
        metric: ForecastMetric = ForecastMetric.VOLUME,
        event_texts: Optional[Sequence[str]] = None,
        events: Optional[Sequence[ParsedEvent]] = None,
        require_approved_events: bool = False,
        max_segments: Optional[int] = None,
    ) -> Dict[str, MetricForecastBundle]:
        """Forecast independently for each segment value (public Flask helper)."""
        if schema is None:
            schema = detect_csv_schema(df, config=self.config)
        cleaned, report = validate_dataframe(df, schema, config=self.config)
        if not report.is_valid:
            raise ValueError("Validation failed")

        segments = list_segment_values(cleaned, segment_column)
        if max_segments:
            segments = segments[:max_segments]

        if events is None:
            events = parse_event_batch(
                event_texts or [],
                segment_candidates=segments,
                business_units=list_column_values(cleaned, schema.business_unit_column),
                products=list_column_values(cleaned, schema.product_column),
            )
        else:
            events = list(events)
        events = _filter_events(events, require_approved=require_approved_events)

        out: Dict[str, MetricForecastBundle] = {}
        for seg in segments:
            try:
                out[seg] = self.forecast_metric_path(
                    cleaned,
                    schema,
                    metric,
                    events,
                    segment_column=segment_column,
                    segment_value=seg,
                    segment_label=seg,
                )
            except ValueError:
                continue
        return out

    def run_from_csv(
        self,
        source: FileLike,
        event_texts: Optional[Sequence[str]] = None,
        segment_column: Optional[str] = None,
        segment_value: Optional[str] = None,
        schema_overrides: Optional[SchemaMappingOverride] = None,
        **read_csv_kwargs,
    ) -> ForecastRunResult:
        """End-to-end run from CSV path or file-like object."""
        cleaned, schema, report = self.prepare_dataframe(
            source, schema_overrides=schema_overrides, **read_csv_kwargs
        )
        if not report.is_valid:
            raise ValueError("CSV validation failed")
        return self.run_from_dataframe(
            cleaned,
            schema=schema,
            event_texts=event_texts,
            segment_column=segment_column,
            segment_value=segment_value,
            schema_overrides=None,
        )

    def run_and_persist(
        self,
        source: FileLike,
        source_filename: Optional[str] = None,
        event_texts: Optional[Sequence[str]] = None,
        schema_overrides: Optional[SchemaMappingOverride] = None,
        **read_csv_kwargs,
    ) -> tuple[ForecastRunResult, int]:
        """Run forecast and store metadata; returns result and analysis run id."""
        run = self.metadata_store.create_run(source_filename=source_filename, status="running")
        try:
            cleaned, schema, report = self.prepare_dataframe(
                source, schema_overrides=schema_overrides, **read_csv_kwargs
            )
            if not report.is_valid:
                raise ValueError("CSV validation failed")
            result = self.run_from_dataframe(cleaned, schema=schema, event_texts=event_texts)
            forecast_payload = {
                "primary_metric": result.primary_metric.value,
                "volume": _bundle_to_dict(result.volume),
                "liability": _bundle_to_dict(result.liability),
                "forecasts": [
                    {
                        "period": f.period.isoformat(),
                        "baseline": f.baseline,
                        "adjusted": f.adjusted,
                        "lower": f.lower,
                        "upper": f.upper,
                        "event_adjustment": f.event_adjustment,
                        "metric": f.metric,
                    }
                    for f in result.forecasts
                ],
            }
            persist_forecast_run(
                self.metadata_store,
                run.id,  # type: ignore[arg-type]
                schema=_schema_to_dict(schema),
                validation=validation_report_to_dict(report),
                model_kind=result.model_selection.model_kind.value,
                fallback_reason=result.model_selection.fallback_reason,
                metrics=result.model_selection.metrics.__dict__,
                forecast=forecast_payload,
                events=[_event_to_dict(e) for e in result.events_applied],
            )
            return result, int(run.id)
        except Exception as exc:
            if run.id is not None:
                run.status = "failed"
                run.error_message = str(exc)
                self.metadata_store.update_run(run)
            raise

    def export_forecast_dataframe(self, result: ForecastRunResult) -> pd.DataFrame:
        """Build export-ready forecast DataFrame (apply export_safety before HTTP response)."""
        rows = []
        for f in result.forecasts:
            rows.append(
                {
                    "period": f.period.strftime("%Y-%m"),
                    "metric": f.metric,
                    "baseline": f.baseline,
                    "adjusted": f.adjusted,
                    "event_adjustment": f.event_adjustment,
                    "lower": f.lower,
                    "upper": f.upper,
                }
            )
        return pd.DataFrame(rows)

    def export_dual_forecast_dataframe(self, result: ForecastRunResult) -> pd.DataFrame:
        """Combined volume + liability forecast rows when both paths exist."""
        frames = []
        if result.volume:
            frames.append(self.export_bundle_dataframe(result.volume))
        if result.liability:
            frames.append(self.export_bundle_dataframe(result.liability))
        if not frames:
            return self.export_forecast_dataframe(result)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def export_bundle_dataframe(bundle: MetricForecastBundle) -> pd.DataFrame:
        rows = []
        for f in bundle.forecasts:
            rows.append(
                {
                    "period": f.period.strftime("%Y-%m"),
                    "metric": bundle.metric.value,
                    "baseline": f.baseline,
                    "adjusted": f.adjusted,
                    "event_adjustment": f.event_adjustment,
                    "lower": f.lower,
                    "upper": f.upper,
                }
            )
        return pd.DataFrame(rows)

    def export_history_dataframe(self, result: ForecastRunResult) -> pd.DataFrame:
        return monthly_series_to_frame(result.monthly_history)


def _schema_to_dict(schema) -> Dict[str, Any]:
    return {
        "date_column": schema.date_column,
        "amount_column": schema.amount_column,
        "count_column": schema.count_column,
        "business_unit_column": schema.business_unit_column,
        "product_column": schema.product_column,
        "status_column": schema.status_column,
        "case_id_column": schema.case_id_column,
        "reason_code_column": schema.reason_code_column,
        "category_column": schema.category_column,
        "category_candidate_columns": schema.category_candidate_columns,
        "segment_columns": schema.segment_columns,
        "volume_aggregation_mode": schema.volume_aggregation_mode.value,
        "column_profiles": {k: v.__dict__ for k, v in schema.column_profiles.items()},
        "user_overrides_applied": schema.user_overrides_applied,
    }


def validation_report_to_dict(report) -> Dict[str, Any]:
    return {
        "is_valid": report.is_valid,
        "uploaded_row_count": report.uploaded_row_count,
        "valid_row_count": report.valid_row_count,
        "rejected_row_count": report.rejected_row_count,
        "invalid_row_count": report.invalid_row_count,
        "duplicate_row_count": report.duplicate_row_count,
        "earliest_date": report.earliest_date.isoformat() if report.earliest_date else None,
        "latest_date": report.latest_date.isoformat() if report.latest_date else None,
        "month_count": report.month_count,
        "missing_months": report.missing_months,
        "total_volume": report.total_volume,
        "total_liability": report.total_liability,
        "warnings": report.warnings,
        "readiness": report.readiness,
        "rejected_row_indexes": report.rejected_row_indexes,
        "rejected_records_sample": report.rejected_records_sample,
        "issues": [i.__dict__ for i in report.issues],
    }


def _event_to_dict(event: ParsedEvent) -> Dict[str, Any]:
    return {
        "raw_text": event.raw_text,
        "event_name": event.event_name,
        "effect_type": event.effect_type,
        "effect_value": event.effect_value,
        "direction": event.direction,
        "metric": event.metric,
        "business_unit": event.business_unit,
        "product": event.product,
        "start_period": event.start_period.isoformat() if event.start_period else None,
        "end_period": event.end_period.isoformat() if event.end_period else None,
        "segment_hint": event.segment_hint,
        "confidence": event.confidence,
        "parse_notes": event.parse_notes,
        "notes": event.notes,
        "approved": event.approved,
        "annotation": event.annotation,
    }


def _bundle_to_dict(bundle: Optional[MetricForecastBundle]) -> Optional[Dict[str, Any]]:
    if bundle is None:
        return None
    return {
        "metric": bundle.metric.value,
        "forecasts": [
            {
                "period": f.period.isoformat(),
                "baseline": f.baseline,
                "adjusted": f.adjusted,
                "lower": f.lower,
                "upper": f.upper,
                "event_adjustment": f.event_adjustment,
            }
            for f in bundle.forecasts
        ],
        "kpis": {
            **{k: v for k, v in bundle.kpis.__dict__.items() if k != "model_kind"},
            "model_kind": bundle.kpis.model_kind.value,
        },
    }


def forecast_result_to_dict(result: ForecastRunResult) -> Dict[str, Any]:
    """JSON-serializable summary for Flask jsonify."""
    return {
        "primary_metric": result.primary_metric.value,
        "forecasts": [
            {
                "period": f.period.isoformat(),
                "baseline": f.baseline,
                "adjusted": f.adjusted,
                "lower": f.lower,
                "upper": f.upper,
                "event_adjustment": f.event_adjustment,
                "metric": f.metric,
            }
            for f in result.forecasts
        ],
        "volume": _bundle_to_dict(result.volume),
        "liability": _bundle_to_dict(result.liability),
        "kpis": {
            **{k: v for k, v in result.kpis.__dict__.items() if k != "model_kind"},
            "model_kind": result.kpis.model_kind.value,
        },
        "model_selection": {
            "model_kind": result.model_selection.model_kind.value,
            "order": result.model_selection.order,
            "seasonal_order": result.model_selection.seasonal_order,
            "fallback_reason": result.model_selection.fallback_reason,
            "metrics": result.model_selection.metrics.__dict__,
        },
        "segment_comparisons": [
            {
                **c.__dict__,
                "same_month_stats": c.same_month_stats.__dict__ if c.same_month_stats else None,
            }
            for c in result.segment_comparisons
        ],
        "explanations": result.explanations.__dict__,
        "events": [_event_to_dict(e) for e in result.events_applied],
    }
