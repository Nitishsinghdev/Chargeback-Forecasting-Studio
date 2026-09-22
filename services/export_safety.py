"""CSV export helpers that mitigate formula injection."""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence, Union

import pandas as pd

from config import AppConfig, get_config

_FORMULA_TRIGGER = re.compile(r"^[\s]*[=+\-@|\t\r]")


def _sanitize_cell(value: Any, prefix: str) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    text = str(value)
    if _FORMULA_TRIGGER.match(text):
        return prefix + text
    return value


def sanitize_dataframe_for_export(
    df: pd.DataFrame,
    config: AppConfig | None = None,
) -> pd.DataFrame:
    """Return a copy with dangerous leading characters neutralized."""
    cfg = config or get_config()
    prefix = cfg.export_formula_prefix
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(lambda v: _sanitize_cell(v, prefix))
    return out


def sanitize_rows_for_export(
    rows: Sequence[Sequence[Any]],
    config: AppConfig | None = None,
) -> List[List[Any]]:
    cfg = config or get_config()
    prefix = cfg.export_formula_prefix
    return [[_sanitize_cell(cell, prefix) for cell in row] for row in rows]


def dataframe_to_safe_csv(
    df: pd.DataFrame,
    config: AppConfig | None = None,
    **to_csv_kwargs: Any,
) -> str:
    """Serialize DataFrame to CSV string with formula injection mitigation."""
    safe_df = sanitize_dataframe_for_export(df, config=config)
    return safe_df.to_csv(index=False, **to_csv_kwargs)


def is_potentially_dangerous_csv_value(value: Union[str, Any]) -> bool:
    """Heuristic check for spreadsheet formula injection."""
    if value is None:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return False
    return bool(_FORMULA_TRIGGER.match(str(value)))
