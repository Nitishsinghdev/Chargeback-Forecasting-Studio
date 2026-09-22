"""Synthetic sample CSV shape, determinism, and realistic patterns."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from services.csv_schema import detect_csv_schema
from services.csv_validation import validate_dataframe

EXPECTED_COLUMNS = [
    "Chargeback Date",
    "Chargeback ID",
    "Amount",
    "Business Unit",
    "Product",
    "Reason Code",
    "Status",
]


def test_sample_csv_columns_and_row_count(sample_csv_path: Path):
    df = pd.read_csv(sample_csv_path)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) >= 5000
    assert df["Chargeback ID"].is_unique


def test_sample_csv_covers_at_least_36_months(sample_df: pd.DataFrame, fast_config):
    schema = detect_csv_schema(sample_df, config=fast_config)
    _, report = validate_dataframe(sample_df, schema, config=fast_config)
    assert report.month_count >= 36
    assert report.earliest_date is not None
    assert report.latest_date is not None
    span_months = (
        (report.latest_date.year - report.earliest_date.year) * 12
        + report.latest_date.month
        - report.earliest_date.month
        + 1
    )
    assert span_months >= 36


def test_sample_december_seasonality(sample_df: pd.DataFrame):
    df = sample_df.copy()
    df["_month"] = pd.to_datetime(df["Chargeback Date"]).dt.month
    monthly_counts = df.groupby("_month").size()
    assert monthly_counts[12] > monthly_counts[2]


def test_sample_business_unit_segment_mix(sample_df: pd.DataFrame):
    bus = set(sample_df["Business Unit"].unique())
    assert bus == {"Enterprise", "Consumer", "Wholesale"}
    assert sample_df.groupby("Business Unit").size().nunique() > 1


def test_regeneration_is_deterministic(project_root: Path, sample_csv_path: Path):
    script = project_root / "sample_data" / "generate_sample_chargebacks.py"
    before = sample_csv_path.read_bytes()
    subprocess.run(
        [sys.executable, str(script)],
        cwd=str(project_root),
        check=True,
        capture_output=True,
    )
    after = sample_csv_path.read_bytes()
    assert before == after
