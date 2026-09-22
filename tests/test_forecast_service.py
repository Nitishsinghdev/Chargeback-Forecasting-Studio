"""Orchestrator: dual metrics, events, exports."""

from __future__ import annotations

from datetime import datetime

from models.domain_types import ForecastMetric, ParsedEvent
from services.csv_schema import detect_csv_schema
from services.csv_validation import validate_dataframe
from services.forecast_service import ForecastOrchestrator


def test_end_to_end_sample_forecast(sample_df, fast_config):
    schema = detect_csv_schema(sample_df, config=fast_config)
    cleaned, report = validate_dataframe(sample_df, schema, config=fast_config)
    assert report.is_valid
    orch = ForecastOrchestrator(fast_config)
    result = orch.run_from_dataframe(
        cleaned,
        schema=schema,
        metrics=[ForecastMetric.VOLUME, ForecastMetric.LIABILITY],
        require_approved_events=False,
    )
    assert len(result.forecasts) == fast_config.forecast.horizon_months
    assert result.volume is not None
    assert result.liability is not None
    assert result.kpis.total_baseline_horizon > 0
    dual = orch.export_dual_forecast_dataframe(result)
    assert set(dual["metric"].unique()) >= {"volume", "liability"}


def test_approved_event_adjusts_forecast(sample_df, fast_config):
    schema = detect_csv_schema(sample_df, config=fast_config)
    cleaned, _ = validate_dataframe(sample_df, schema, config=fast_config)
    orch = ForecastOrchestrator(fast_config)
    baseline = orch.run_from_dataframe(cleaned, schema=schema, events=[], require_approved_events=True)
    last = baseline.monthly_history.period_index[-1]
    start = datetime(last.year + (1 if last.month == 12 else 0), (last.month % 12) + 1, 1)
    ev = ParsedEvent(
        raw_text="test bump",
        event_name="bump",
        effect_type="percent",
        effect_value=20.0,
        direction="increase",
        metric="volume",
        business_unit=None,
        product=None,
        start_period=start,
        end_period=start,
        segment_hint=None,
        confidence=1.0,
        approved=True,
    )
    adjusted = orch.run_from_dataframe(
        cleaned, schema=schema, events=[ev], require_approved_events=True
    )
    assert adjusted.kpis.total_adjusted_horizon >= adjusted.kpis.total_baseline_horizon


def test_run_from_csv_path(sample_csv_path, fast_config):
    orch = ForecastOrchestrator(fast_config)
    result = orch.run_from_csv(sample_csv_path)
    assert len(result.forecasts) == 3
