"""CSV export formula-injection mitigation."""

from __future__ import annotations

import pandas as pd

from services.export_safety import (
    dataframe_to_safe_csv,
    is_potentially_dangerous_csv_value,
    sanitize_dataframe_for_export,
)


def test_formula_prefix_applied(fast_config):
    df = pd.DataFrame({"note": ["=SUM(A1:A2)", "+cmd", "normal"]})
    safe = sanitize_dataframe_for_export(df, config=fast_config)
    assert str(safe["note"].iloc[0]).startswith("'")
    assert str(safe["note"].iloc[2]) == "normal"


def test_dataframe_to_safe_csv_string(fast_config):
    df = pd.DataFrame({"x": ["@evil"]})
    csv_text = dataframe_to_safe_csv(df, config=fast_config)
    assert csv_text.startswith("x")
    assert "'@evil" in csv_text or "@evil" in csv_text


def test_numeric_not_flagged():
    assert not is_potentially_dangerous_csv_value(42)
    assert is_potentially_dangerous_csv_value("-1+2")
