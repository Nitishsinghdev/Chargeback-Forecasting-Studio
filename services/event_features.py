"""Build exogenous event features aligned to monthly forecast index."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from models.domain_types import MonthlySeries, ParsedEvent, event_has_adjustable_magnitude


def _month_start(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, 1)


def _periods_between(start: datetime, end: datetime) -> List[datetime]:
    start_p = pd.Timestamp(start).to_period("M")
    end_p = pd.Timestamp(end).to_period("M")
    if end_p < start_p:
        start_p, end_p = end_p, start_p
    return [datetime(p.year, p.month, 1) for p in pd.period_range(start_p, end_p, freq="M")]


def _event_applies_to_context(
    event: ParsedEvent,
    *,
    metric: str,
    segment_filter: Optional[str],
    business_unit: Optional[str],
    product: Optional[str],
) -> bool:
    if not event_has_adjustable_magnitude(event):
        return False

    em = (event.metric or "unknown").lower()
    if em not in {"unknown", "both", metric.lower()}:
        return False

    if segment_filter and event.segment_hint:
        if str(event.segment_hint) != str(segment_filter):
            return False
    if business_unit and event.business_unit:
        if str(event.business_unit) != str(business_unit):
            return False
    if product and event.product:
        if str(event.product) != str(product):
            return False
    return True


def build_event_regressors(
    history: MonthlySeries,
    forecast_periods: Sequence[datetime],
    events: Sequence[ParsedEvent],
    segment_filter: Optional[str] = None,
    metric: str = "volume",
    business_unit: Optional[str] = None,
    product: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create per-period event multiplier and additive adjustment columns.

    Events without numeric magnitude never adjust the baseline.
    """
    all_periods = list(history.period_index) + list(forecast_periods)
    if not all_periods:
        return pd.DataFrame(
            columns=["period", "event_pct_multiplier", "event_abs_adjustment"]
        )

    idx = sorted({_month_start(p) for p in all_periods})
    pct = np.ones(len(idx), dtype=float)
    add = np.zeros(len(idx), dtype=float)
    period_to_i = {p: i for i, p in enumerate(idx)}

    for event in events:
        if not _event_applies_to_context(
            event,
            metric=metric,
            segment_filter=segment_filter,
            business_unit=business_unit,
            product=product,
        ):
            continue
        if event.start_period is None:
            continue
        end = event.end_period or event.start_period
        for p in _periods_between(event.start_period, end):
            key = _month_start(p)
            if key not in period_to_i:
                continue
            i = period_to_i[key]
            if event.effect_type == "percent" and event.effect_value is not None:
                pct[i] *= 1.0 + (event.effect_value / 100.0)
            elif event.effect_type == "absolute" and event.effect_value is not None:
                add[i] += event.effect_value

    return pd.DataFrame(
        {
            "period": idx,
            "event_pct_multiplier": pct,
            "event_abs_adjustment": add,
        }
    )


def apply_event_adjustments(
    baseline_values: Sequence[float],
    forecast_periods: Sequence[datetime],
    events: Sequence[ParsedEvent],
    history: MonthlySeries,
    segment_filter: Optional[str] = None,
    metric: str = "volume",
    business_unit: Optional[str] = None,
    product: Optional[str] = None,
) -> Tuple[List[float], List[float]]:
    """Return (multipliers, additive) aligned to forecast_periods only."""
    reg = build_event_regressors(
        history=history,
        forecast_periods=forecast_periods,
        events=events,
        segment_filter=segment_filter,
        metric=metric,
        business_unit=business_unit,
        product=product,
    )
    reg = reg.set_index("period")
    mults: List[float] = []
    adds: List[float] = []
    for p in forecast_periods:
        key = _month_start(p)
        if key in reg.index:
            mults.append(float(reg.loc[key, "event_pct_multiplier"]))
            adds.append(float(reg.loc[key, "event_abs_adjustment"]))
        else:
            mults.append(1.0)
            adds.append(0.0)
    return mults, adds


def exogenous_for_sarimax(
    history: MonthlySeries,
    forecast_periods: Sequence[datetime],
    events: Sequence[ParsedEvent],
    segment_filter: Optional[str] = None,
    metric: str = "volume",
    business_unit: Optional[str] = None,
    product: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Exogenous matrix for in-sample + forecast horizon (history + future)."""
    reg = build_event_regressors(
        history,
        forecast_periods,
        events,
        segment_filter,
        metric=metric,
        business_unit=business_unit,
        product=product,
    )
    if reg.empty:
        return None
    hist_set = {_month_start(p) for p in history.period_index}
    fc_set = [_month_start(p) for p in forecast_periods]
    ordered = sorted(hist_set.union(fc_set))
    reg = reg.set_index("period").reindex(ordered).fillna(
        {"event_pct_multiplier": 1.0, "event_abs_adjustment": 0.0}
    )
    x = reg[["event_pct_multiplier", "event_abs_adjustment"]].to_numpy(dtype=float)
    if np.allclose(x[:, 0], 1.0) and np.allclose(x[:, 1], 0.0):
        return None
    return x
