"""Controlled SARIMAX search with holdout and rolling validation."""

from __future__ import annotations

import itertools
import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import AppConfig, get_config
from models.domain_types import ModelKind, ModelSelectionResult, ValidationStrategy
from services.fallback_models import run_fallback_chain
from services.metrics import combine_metric_sets, compute_safe_metrics

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


def _candidate_orders(cfg: AppConfig) -> List[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    s = cfg.validation.seasonal_period
    combos = list(
        itertools.product(
            cfg.sarimax.p_range,
            cfg.sarimax.d_range,
            cfg.sarimax.q_range,
            cfg.sarimax.P_range,
            cfg.sarimax.D_range,
            cfg.sarimax.Q_range,
        )
    )
    orders = [((p, d, q), (P, D, Q, s)) for p, d, q, P, D, Q in combos]
    # Prefer simpler models first (sum of parameters)
    orders.sort(key=lambda x: sum(x[0]) + sum(x[1][:3]))
    return orders[: cfg.sarimax.max_candidates]


def _fit_predict(
    y_train: np.ndarray,
    steps: int,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    exog_train: Optional[np.ndarray] = None,
    exog_test: Optional[np.ndarray] = None,
    cfg: Optional[AppConfig] = None,
) -> Optional[np.ndarray]:
    cfg = cfg or get_config()
    try:
        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=cfg.sarimax.enforce_stationarity,
            enforce_invertibility=cfg.sarimax.enforce_invertibility,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(disp=False, low_memory=True)
        pred = res.forecast(steps=steps, exog=exog_test)
        return np.asarray(pred, dtype=float)
    except Exception:
        return None


def _holdout_validation(
    y: np.ndarray,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    holdout: int,
    exog: Optional[np.ndarray] = None,
    cfg: Optional[AppConfig] = None,
) -> Optional[np.ndarray]:
    if len(y) <= holdout + 1:
        return None
    train, test = y[:-holdout], y[-holdout:]
    ex_train = exog[:-holdout] if exog is not None else None
    ex_test = exog[-holdout:] if exog is not None else None
    preds = _fit_predict(train, holdout, order, seasonal_order, ex_train, ex_test, cfg)
    return preds


def _rolling_validation(
    y: np.ndarray,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    holdout: int,
    folds: int,
    exog: Optional[np.ndarray] = None,
    cfg: Optional[AppConfig] = None,
) -> List[np.ndarray]:
    preds_folds: List[np.ndarray] = []
    min_train = (cfg or get_config()).validation.min_train_months
    for f in range(folds):
        test_end = len(y) - f * holdout
        test_start = test_end - holdout
        if test_start < min_train:
            break
        train = y[:test_start]
        test = y[test_start:test_end]
        if len(test) == 0:
            continue
        ex_train = exog[:test_start] if exog is not None else None
        ex_test = exog[test_start:test_end] if exog is not None else None
        pred = _fit_predict(train, len(test), order, seasonal_order, ex_train, ex_test, cfg)
        if pred is None or len(pred) != len(test):
            continue
        preds_folds.append(pred)
    return preds_folds


def select_sarimax_model(
    history_values: Sequence[float],
    exog: Optional[np.ndarray] = None,
    config: Optional[AppConfig] = None,
) -> ModelSelectionResult:
    """
    Search SARIMAX orders using holdout + rolling validation without leakage.

    Training slices always end before the evaluated horizon.
    """
    cfg = config or get_config()
    y = np.asarray(history_values, dtype=float)
    y = y[np.isfinite(y)]
    notes: List[str] = []
    holdout = cfg.validation.holdout_months
    folds = cfg.validation.rolling_folds
    min_train = cfg.validation.min_train_months

    if len(y) < min_train:
        fb, kind, reason = run_fallback_chain(
            y.tolist(), cfg.forecast.horizon_months, cfg.validation.seasonal_period, config=cfg
        )
        metrics = compute_safe_metrics(y[-min(holdout, len(y)) :] if len(y) else [], fb[: min(holdout, len(fb))])
        return ModelSelectionResult(
            model_kind=kind,
            order=None,
            seasonal_order=None,
            metrics=metrics,
            validation_strategy=ValidationStrategy.HOLDOUT,
            fallback_reason=f"insufficient_history:{len(y)}<{min_train};{reason}",
            search_notes=notes,
            extra={"fallback_forecast": fb},
        )

    if exog is not None and len(exog) != len(y):
        notes.append("exog_length_mismatch_ignored")
        exog = None

    best: Optional[ModelSelectionResult] = None
    best_score = float("inf")

    for order, seasonal_order in _candidate_orders(cfg):
        holdout_pred = _holdout_validation(y, order, seasonal_order, holdout, exog, cfg)
        if holdout_pred is None:
            continue
        actual = y[-holdout:]
        holdout_metrics = compute_safe_metrics(actual, holdout_pred)

        rolling_preds = _rolling_validation(y, order, seasonal_order, holdout, folds, exog, cfg)
        rolling_metrics_list = []
        for i, pred in enumerate(rolling_preds):
            test_end = len(y) - i * holdout
            test_start = test_end - holdout
            actual_fold = y[test_start:test_end]
            rolling_metrics_list.append(compute_safe_metrics(actual_fold, pred))
        combined = combine_metric_sets([holdout_metrics, *rolling_metrics_list])

        score = combined.rmse if combined.rmse is not None else combined.mae
        if score is None:
            score = float("inf")
        if score < best_score:
            best_score = score
            best = ModelSelectionResult(
                model_kind=ModelKind.SARIMAX,
                order=order,
                seasonal_order=seasonal_order,
                metrics=combined,
                validation_strategy=ValidationStrategy.HOLDOUT_AND_ROLLING,
                fallback_reason=None,
                search_notes=[f"selected_score={score:.6f}"],
            )

    if best is not None:
        return best

    fb, kind, reason = run_fallback_chain(
        y.tolist(), cfg.forecast.horizon_months, cfg.validation.seasonal_period, config=cfg
    )
    metrics = compute_safe_metrics(y[-holdout:], fb[:holdout])
    return ModelSelectionResult(
        model_kind=kind,
        order=None,
        seasonal_order=None,
        metrics=metrics,
        validation_strategy=ValidationStrategy.HOLDOUT,
        fallback_reason=f"sarimax_search_exhausted;{reason}",
        search_notes=notes,
        extra={"fallback_forecast": fb},
    )


def forecast_with_selection(
    history_values: Sequence[float],
    horizon: int,
    selection: ModelSelectionResult,
    exog_history: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
    config: Optional[AppConfig] = None,
) -> Tuple[List[float], Optional[Tuple[List[float], List[float]]]]:
    """
    Produce baseline forecast path for selected model.

    Returns (point_forecast, optional_confidence_intervals_as_(lower, upper)).
    """
    cfg = config or get_config()
    y = np.asarray(history_values, dtype=float)

    if selection.model_kind != ModelKind.SARIMAX or selection.order is None:
        fb = selection.extra.get("fallback_forecast")
        if fb is None:
            fb, _, _ = run_fallback_chain(
                y.tolist(), horizon, cfg.validation.seasonal_period, config=cfg
            )
        return list(fb[:horizon]), None

    order = selection.order
    seasonal_order = selection.seasonal_order or (0, 0, 0, cfg.validation.seasonal_period)
    try:
        model = SARIMAX(
            y,
            exog=exog_history,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=cfg.sarimax.enforce_stationarity,
            enforce_invertibility=cfg.sarimax.enforce_invertibility,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(disp=False, low_memory=True)
        pred = res.get_forecast(steps=horizon, exog=exog_future)
        mean = [float(v) for v in pred.predicted_mean]
        conf = pred.conf_int(alpha=1.0 - cfg.forecast.confidence_level)
        lower = [float(v) for v in conf.iloc[:, 0]]
        upper = [float(v) for v in conf.iloc[:, 1]]
        return mean, (lower, upper)
    except Exception:
        fb, _, _ = run_fallback_chain(
            y.tolist(), horizon, cfg.validation.seasonal_period, config=cfg
        )
        return list(fb[:horizon]), None
