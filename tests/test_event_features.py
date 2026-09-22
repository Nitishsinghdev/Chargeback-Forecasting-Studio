"""Event adjustments and exogenous features."""

from __future__ import annotations

from datetime import datetime

from models.domain_types import MonthlySeries, ParsedEvent, event_has_adjustable_magnitude
from services.event_features import apply_event_adjustments, build_event_regressors, exogenous_for_sarimax


def _history(n: int = 24) -> MonthlySeries:
    periods = [datetime(2023, m, 1) for m in range(1, 13)] + [datetime(2024, m, 1) for m in range(1, 13)]
    periods = periods[:n]
    return MonthlySeries(period_index=periods, values=[100.0 + i for i in range(n)])


def test_no_number_event_does_not_adjust():
    ev = ParsedEvent(
        raw_text="Policy change",
        event_name="Policy",
        effect_type="unknown",
        effect_value=None,
        direction="unknown",
        metric="volume",
        business_unit=None,
        product=None,
        start_period=datetime(2024, 6, 1),
        end_period=datetime(2024, 6, 1),
        segment_hint=None,
        confidence=0.2,
    )
    assert not event_has_adjustable_magnitude(ev)
    baseline = [100.0, 100.0, 100.0]
    forecast_periods = [datetime(2024, 7, 1), datetime(2024, 8, 1), datetime(2024, 9, 1)]
    mults, adds = apply_event_adjustments(baseline, forecast_periods, [ev], _history())
    assert mults == [1.0, 1.0, 1.0]
    assert adds == [0.0, 0.0, 0.0]


def test_percent_event_applies_to_matching_month():
    ev = ParsedEvent(
        raw_text="Spike",
        event_name="Spike",
        effect_type="percent",
        effect_value=10.0,
        direction="increase",
        metric="volume",
        business_unit=None,
        product=None,
        start_period=datetime(2024, 7, 1),
        end_period=datetime(2024, 7, 1),
        segment_hint=None,
        confidence=0.9,
        approved=True,
    )
    forecast_periods = [datetime(2024, 7, 1), datetime(2024, 8, 1)]
    mults, adds = apply_event_adjustments([50.0, 50.0], forecast_periods, [ev], _history())
    assert mults[0] == 1.1
    assert mults[1] == 1.0
    assert adds == [0.0, 0.0]


def test_exogenous_none_when_no_adjustments():
    hist = _history()
    future = [datetime(2024, 7, 1)]
    x = exogenous_for_sarimax(hist, future, [], metric="volume")
    assert x is None


def test_build_regressors_covers_history_and_future():
    ev = ParsedEvent(
        raw_text="x",
        event_name="x",
        effect_type="absolute",
        effect_value=500.0,
        direction="increase",
        metric="liability",
        business_unit=None,
        product=None,
        start_period=datetime(2024, 1, 1),
        end_period=datetime(2024, 1, 1),
        segment_hint=None,
        confidence=0.8,
    )
    hist = _history(24)
    future = [datetime(2024, 7, 1)]
    reg = build_event_regressors(hist, future, [ev], metric="liability")
    assert len(reg) >= 13
    jan = reg[reg["period"] == datetime(2024, 1, 1)]
    assert not jan.empty
    assert float(jan["event_abs_adjustment"].iloc[0]) == 500.0
