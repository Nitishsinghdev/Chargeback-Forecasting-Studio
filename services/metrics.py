"""Safe forecast error metrics without divide-by-zero explosions."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np

from models.domain_types import SafeMetricSet

_EPS = 1e-12


def _as_float_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected 1-D numeric sequence")
    return arr


def safe_mape(actual: Sequence[float], predicted: Sequence[float]) -> tuple[Optional[float], List[str]]:
    """Mean absolute percentage error; skips zero actuals."""
    notes: List[str] = []
    y = _as_float_array(actual)
    yhat = _as_float_array(predicted)
    if y.shape != yhat.shape:
        raise ValueError("actual and predicted must have the same length")
    mask = np.isfinite(y) & np.isfinite(yhat) & (np.abs(y) > _EPS)
    if mask.sum() == 0:
        notes.append("mape_undefined_all_zero_or_nonfinite_actuals")
        return None, notes
    pct = np.abs((y[mask] - yhat[mask]) / y[mask])
    return float(np.mean(pct) * 100.0), notes


def safe_smape(actual: Sequence[float], predicted: Sequence[float]) -> tuple[Optional[float], List[str]]:
    notes: List[str] = []
    y = _as_float_array(actual)
    yhat = _as_float_array(predicted)
    denom = np.abs(y) + np.abs(yhat)
    mask = np.isfinite(y) & np.isfinite(yhat) & (denom > _EPS)
    if mask.sum() == 0:
        notes.append("smape_undefined_zero_denominator")
        return None, notes
    smape = 200.0 * np.abs(y[mask] - yhat[mask]) / denom[mask]
    return float(np.mean(smape)), notes


def safe_rmse(actual: Sequence[float], predicted: Sequence[float]) -> Optional[float]:
    y = _as_float_array(actual)
    yhat = _as_float_array(predicted)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() == 0:
        return None
    err = y[mask] - yhat[mask]
    return float(np.sqrt(np.mean(err ** 2)))


def safe_mae(actual: Sequence[float], predicted: Sequence[float]) -> Optional[float]:
    y = _as_float_array(actual)
    yhat = _as_float_array(predicted)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs(y[mask] - yhat[mask])))


def safe_bias(actual: Sequence[float], predicted: Sequence[float]) -> Optional[float]:
    y = _as_float_array(actual)
    yhat = _as_float_array(predicted)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() == 0:
        return None
    return float(np.mean(yhat[mask] - y[mask]))


def compute_safe_metrics(actual: Sequence[float], predicted: Sequence[float]) -> SafeMetricSet:
    """Aggregate safe metrics for holdout or rolling validation."""
    y = _as_float_array(actual)
    yhat = _as_float_array(predicted)
    mask = np.isfinite(y) & np.isfinite(yhat)
    notes: List[str] = []
    mape, mape_notes = safe_mape(actual, predicted)
    notes.extend(mape_notes)
    smape, smape_notes = safe_smape(actual, predicted)
    notes.extend(smape_notes)
    return SafeMetricSet(
        mape=mape,
        smape=smape,
        rmse=safe_rmse(actual, predicted),
        mae=safe_mae(actual, predicted),
        bias=safe_bias(actual, predicted),
        n_obs=int(mask.sum()),
        notes=notes,
    )


def combine_metric_sets(metric_sets: Iterable[SafeMetricSet]) -> SafeMetricSet:
    """Average available scalar metrics across folds (equal weight)."""
    sets = list(metric_sets)
    if not sets:
        return SafeMetricSet(None, None, None, None, None, 0, ["no_metric_sets"])

    def _avg(getter) -> Optional[float]:
        vals = [getter(m) for m in sets if getter(m) is not None]
        if not vals:
            return None
        return float(np.mean(vals))

    notes: List[str] = []
    for m in sets:
        notes.extend(m.notes)
    return SafeMetricSet(
        mape=_avg(lambda m: m.mape),
        smape=_avg(lambda m: m.smape),
        rmse=_avg(lambda m: m.rmse),
        mae=_avg(lambda m: m.mae),
        bias=_avg(lambda m: m.bias),
        n_obs=int(sum(m.n_obs for m in sets)),
        notes=list(dict.fromkeys(notes)),
    )
