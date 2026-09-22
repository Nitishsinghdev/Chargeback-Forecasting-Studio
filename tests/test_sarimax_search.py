"""SARIMAX selection, 3-month horizon, leakage-safe training slices."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from config import ForecastConfig
from dataclasses import replace
from models.domain_types import ModelKind, ValidationStrategy
from services.sarimax_search import _rolling_validation, forecast_with_selection, select_sarimax_model


def _seasonal_history(n: int = 36) -> list[float]:
    t = np.arange(n, dtype=float)
    return (120 + 8 * np.sin(2 * np.pi * t / 12) + 0.3 * t).tolist()


def test_select_sarimax_or_fallback_fast(fast_config):
    y = _seasonal_history(36)
    result = select_sarimax_model(y, config=fast_config)
    assert result.model_kind in set(ModelKind)
    assert result.metrics.n_obs >= 0
    if result.model_kind == ModelKind.SARIMAX:
        assert result.order is not None
        assert result.validation_strategy == ValidationStrategy.HOLDOUT_AND_ROLLING


def test_forecast_horizon_three_months(fast_config):
    y = _seasonal_history(36)
    selection = select_sarimax_model(y, config=fast_config)
    cfg = replace(fast_config, forecast=replace(fast_config.forecast, horizon_months=3))
    baseline, bands = forecast_with_selection(y, 3, selection, config=cfg)
    assert len(baseline) == 3
    if bands is not None:
        lower, upper = bands
        assert len(lower) == 3 and len(upper) == 3


def test_insufficient_history_activates_fallback(fast_config):
    y = [1.0, 2.0, 3.0, 4.0]
    result = select_sarimax_model(y, config=fast_config)
    assert result.model_kind != ModelKind.SARIMAX or result.fallback_reason
    assert result.fallback_reason is not None
    assert "insufficient_history" in (result.fallback_reason or "")


def test_rolling_validation_train_end_before_test(fast_config):
    """Training arrays passed to fit must not include holdout months (no leakage)."""
    y = np.array(_seasonal_history(30), dtype=float)
    holdout = fast_config.validation.holdout_months
    train_lengths: list[int] = []

    def capture_train(y_train, steps, order, seasonal_order, exog_train=None, exog_test=None, cfg=None):
        train_lengths.append(len(y_train))
        return np.full(steps, float(np.mean(y_train)))

    order = (0, 0, 0)
    seasonal = (0, 0, 0, 12)
    with patch("services.sarimax_search._fit_predict", side_effect=capture_train):
        _rolling_validation(y, order, seasonal, holdout, folds=2, cfg=fast_config)

    assert train_lengths, "expected at least one rolling fold"
    for length in train_lengths:
        assert length <= len(y) - holdout
        assert length >= fast_config.validation.min_train_months or length < len(y) - holdout


def test_holdout_validation_train_excludes_holdout_slice(fast_config):
    y = np.array(_seasonal_history(24), dtype=float)
    holdout = fast_config.validation.holdout_months
    seen_train_len: list[int] = []

    def capture(y_train, steps, order, seasonal_order, exog_train=None, exog_test=None, cfg=None):
        seen_train_len.append(len(y_train))
        return np.full(steps, float(np.mean(y_train)))

    order = (0, 0, 0)
    seasonal = (0, 0, 0, 12)
    with patch("services.sarimax_search._fit_predict", side_effect=capture):
        from services.sarimax_search import _holdout_validation

        preds = _holdout_validation(y, order, seasonal, holdout, None, fast_config)

    assert preds is not None
    assert seen_train_len == [len(y) - holdout]
    assert seen_train_len[0] + holdout == len(y)
