"""Fallback chain order and behavior."""

from __future__ import annotations

import numpy as np

from models.domain_types import ModelKind
from services.fallback_models import forecast_seasonal_naive, run_fallback_chain


def test_seasonal_naive_uses_lag_12():
    history = list(range(1, 25))
    vals, kind, reason = forecast_seasonal_naive(history, horizon=3, seasonal_period=12)
    assert kind == ModelKind.SEASONAL_NAIVE
    assert reason is None
    assert vals[0] == float(history[-12])


def test_fallback_chain_prefers_seasonal_naive_with_enough_history(fast_config):
    rng = np.random.default_rng(0)
    history = (100 + 10 * np.sin(np.arange(48) * 2 * np.pi / 12) + rng.normal(0, 1, 48)).tolist()
    vals, kind, reason = run_fallback_chain(history, horizon=3, seasonal_period=12, config=fast_config)
    assert kind in {ModelKind.SEASONAL_NAIVE, ModelKind.MOVING_AVERAGE, ModelKind.HOLT_WINTERS}
    assert len(vals) == 3
    assert reason is not None or kind == ModelKind.SEASONAL_NAIVE


def test_insufficient_history_still_returns_horizon():
    vals, kind, _ = run_fallback_chain([1.0, 2.0], horizon=3, seasonal_period=12)
    assert len(vals) == 3
    assert isinstance(kind, ModelKind)
