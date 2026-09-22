"""Safe metrics with zero actuals."""

from __future__ import annotations

import numpy as np

from services.metrics import combine_metric_sets, compute_safe_metrics, safe_mape


def test_mape_skips_zero_actuals():
    actual = [0.0, 0.0, 100.0]
    predicted = [10.0, 20.0, 90.0]
    mape, notes = safe_mape(actual, predicted)
    assert mape is not None
    assert abs(mape - 10.0) < 1e-6
    assert not notes or "mape_undefined" not in notes[0]


def test_mape_all_zeros_undefined():
    mape, notes = safe_mape([0.0, 0.0], [1.0, 2.0])
    assert mape is None
    assert any("mape_undefined" in n for n in notes)


def test_compute_safe_metrics_finite():
    metrics = compute_safe_metrics([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    assert metrics.rmse is not None
    assert metrics.mae is not None
    assert metrics.n_obs == 3


def test_combine_metric_sets_averages():
    a = compute_safe_metrics([10.0, 20.0], [11.0, 18.0])
    b = compute_safe_metrics([10.0, 20.0], [9.0, 22.0])
    combined = combine_metric_sets([a, b])
    assert combined.rmse is not None
    assert combined.n_obs == a.n_obs + b.n_obs
