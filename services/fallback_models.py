"""Fallback forecasters when SARIMAX search fails or is insufficient."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

from config import AppConfig, get_config
from models.domain_types import ModelKind

_EPS = 1e-12


def _as_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def forecast_naive(
    history: Sequence[float],
    horizon: int,
) -> Tuple[List[float], ModelKind, Optional[str]]:
    if len(history) == 0:
        return [0.0] * horizon, ModelKind.NAIVE, "empty_history"
    last = float(history[-1])
    return [last] * horizon, ModelKind.NAIVE, None


def forecast_seasonal_naive(
    history: Sequence[float],
    horizon: int,
    seasonal_period: int,
) -> Tuple[List[float], ModelKind, Optional[str]]:
    y = _as_array(history)
    if len(y) == 0:
        return [0.0] * horizon, ModelKind.SEASONAL_NAIVE, "empty_history"
    if len(y) < seasonal_period:
        vals, _, _ = forecast_naive(history, horizon)
        return (
            vals,
            ModelKind.SEASONAL_NAIVE,
            f"insufficient_history_for_seasonal_period_{seasonal_period}",
        )
    out: List[float] = []
    for h in range(1, horizon + 1):
        idx = len(y) - seasonal_period + ((h - 1) % seasonal_period)
        idx = max(0, min(len(y) - 1, idx))
        out.append(float(y[idx]))
    return out, ModelKind.SEASONAL_NAIVE, None


def forecast_moving_average(
    history: Sequence[float],
    horizon: int,
    window: int,
) -> Tuple[List[float], ModelKind, Optional[str]]:
    y = _as_array(history)
    if len(y) == 0:
        return [0.0] * horizon, ModelKind.MOVING_AVERAGE, "empty_history"
    w = max(1, min(window, len(y)))
    level = float(np.mean(y[-w:]))
    return [level] * horizon, ModelKind.MOVING_AVERAGE, None


def forecast_mean(
    history: Sequence[float],
    horizon: int,
) -> Tuple[List[float], ModelKind, Optional[str]]:
    y = _as_array(history)
    if len(y) == 0:
        return [0.0] * horizon, ModelKind.MEAN, "empty_history"
    mu = float(np.mean(y))
    return [mu] * horizon, ModelKind.MEAN, None


def forecast_linear_trend(
    history: Sequence[float],
    horizon: int,
) -> Tuple[List[float], ModelKind, Optional[str]]:
    y = _as_array(history)
    if len(y) < 2:
        vals, _, reason = forecast_naive(history, horizon)
        return vals, ModelKind.LINEAR_TREND, reason or "insufficient_history_for_trend"
    x = np.arange(len(y), dtype=float).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    future_x = np.arange(len(y), len(y) + horizon, dtype=float).reshape(-1, 1)
    preds = model.predict(future_x)
    return [float(v) for v in preds], ModelKind.LINEAR_TREND, None


def forecast_holt_winters_simple(
    history: Sequence[float],
    horizon: int,
    seasonal_period: int,
) -> Tuple[List[float], ModelKind, Optional[str]]:
    y = _as_array(history)
    if len(y) < max(3, seasonal_period + 1):
        return forecast_seasonal_naive(history, horizon, seasonal_period)
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        model = ExponentialSmoothing(
            y,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_period,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True, use_brute=False)
        preds = fit.forecast(horizon)
        return [float(v) for v in preds], ModelKind.HOLT_WINTERS, None
    except Exception as exc:  # noqa: BLE001 - explicit fallback path
        sn, _, reason = forecast_seasonal_naive(history, horizon, seasonal_period)
        return sn, ModelKind.HOLT_WINTERS, f"holt_winters_failed:{type(exc).__name__}:{reason or ''}"


def run_fallback_chain(
    history: Sequence[float],
    horizon: int,
    seasonal_period: int,
    config: Optional[AppConfig] = None,
) -> Tuple[List[float], ModelKind, str]:
    """
    Required order: seasonal-naive, then recent moving average; optional fallbacks after.
    """
    cfg = config or get_config()
    window = cfg.forecast.moving_average_window
    chain = [
        lambda: forecast_seasonal_naive(history, horizon, seasonal_period),
        lambda: forecast_moving_average(history, horizon, window),
        lambda: forecast_holt_winters_simple(history, horizon, seasonal_period),
        lambda: forecast_linear_trend(history, horizon),
        lambda: forecast_mean(history, horizon),
        lambda: forecast_naive(history, horizon),
    ]
    reasons: List[str] = []
    for fn in chain:
        vals, kind, reason = fn()
        if reason:
            reasons.append(f"{kind.value}:{reason}")
        else:
            return vals, kind, "sarimax_unavailable_or_failed"
    vals, kind, reason = forecast_naive(history, horizon)
    reasons.append(reason or "unknown")
    return vals, kind, ";".join(reasons)
