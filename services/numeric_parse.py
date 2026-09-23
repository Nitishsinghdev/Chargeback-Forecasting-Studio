"""Currency-aware numeric parsing shared by detection, validation, and aggregation."""

from __future__ import annotations

import re
import warnings

import pandas as pd

# Symbols commonly prefixed/suffixed to monetary values in exported reports.
CURRENCY_SYMBOLS = "$€£¥₹₽₩₪₫₦₱₨฿"

_CURRENCY_CHARS = re.compile(rf"[{re.escape(CURRENCY_SYMBOLS)}]")
_CURRENCY_CODE = re.compile(r"\b(?:USD|EUR|GBP|INR|JPY|CNY|AUD|CAD|CHF|SGD|AED|ZAR)\b", re.IGNORECASE)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TRAILING_MINUS = re.compile(r"^(\d[\d.]*)-$")


def _clean_scalar(value: str) -> str:
    """Strip currency decoration so a numeric string remains."""
    text = value.strip()
    if not text:
        return ""
    negative = False
    # Accounting notation: (1,234.56) means -1234.56
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()
    text = _CURRENCY_CHARS.sub("", text)
    text = _CURRENCY_CODE.sub("", text)
    text = text.replace("\u00a0", " ").strip()
    text = _THOUSANDS.sub("", text)
    trailing = _TRAILING_MINUS.match(text)
    if trailing:
        negative = True
        text = trailing.group(1)
    text = text.replace(" ", "")
    if text.endswith("%"):
        text = text[:-1]
    if negative and text and not text.startswith("-"):
        text = f"-{text}"
    return text


def to_numeric_currency(series: pd.Series) -> pd.Series:
    """
    Convert a series to floats, tolerating currency symbols and thousands separators.

    Handles values such as ``$1,234.56``, ``USD 500``, ``(250.00)``, and ``1 234,00``
    is intentionally not treated as European decimal comma, to avoid ambiguity.
    """
    direct = pd.to_numeric(series, errors="coerce")
    if direct.notna().all():
        return direct.astype(float)

    text = series.astype(str).map(_clean_scalar)
    cleaned = pd.to_numeric(text, errors="coerce")
    return direct.fillna(cleaned).astype(float)


def numeric_ratio_currency(series: pd.Series, sample_size: int = 5000) -> float:
    """Share of values parseable as numbers once currency decoration is removed."""
    if series.empty:
        return 0.0
    sample = series.head(min(len(series), sample_size))
    return float(to_numeric_currency(sample).notna().mean())


def to_datetime_flexible(series: pd.Series, sample_size: int = 5000) -> pd.Series:
    """
    Parse dates without discarding rows that use a different format.

    A single ``pd.to_datetime`` call locks onto one inferred format and coerces every
    other layout to NaT, which silently rejects valid rows in files that mix
    ``2025-01-31``, ``31/01/2025`` and ``31-Jan-2025``. Values that fail the fast
    vectorised pass are retried per element, day-first as a last attempt.
    """
    if series.empty:
        return pd.to_datetime(series, errors="coerce")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(series, errors="coerce")

    text = series.astype(str).str.strip()
    pending = parsed.isna() & series.notna() & text.ne("") & text.str.lower().ne("nan")
    if not pending.any():
        return parsed

    subset = series[pending]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        retry = pd.to_datetime(subset, errors="coerce", format="mixed")
        remaining = retry.isna()
        if remaining.any():
            retry.loc[remaining] = pd.to_datetime(
                subset[remaining], errors="coerce", format="mixed", dayfirst=True
            )
    parsed.loc[pending] = retry
    return parsed


def date_parse_ratio_flexible(series: pd.Series, sample_size: int = 5000) -> float:
    """Share of values parseable as dates, tolerating mixed formats."""
    if series.empty:
        return 0.0
    sample = series.head(min(len(series), sample_size))
    return float(to_datetime_flexible(sample).notna().mean())


def currency_marker_ratio(series: pd.Series, sample_size: int = 5000) -> float:
    """Share of values carrying a currency symbol or ISO currency code."""
    if series.empty:
        return 0.0
    sample = series.head(min(len(series), sample_size)).dropna()
    if sample.empty:
        return 0.0
    text = sample.astype(str)
    marked = text.str.contains(_CURRENCY_CHARS, regex=True) | text.str.contains(_CURRENCY_CODE, regex=True)
    return float(marked.mean())
