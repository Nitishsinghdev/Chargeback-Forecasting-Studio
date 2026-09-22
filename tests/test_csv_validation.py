"""Validation: invalid rows, duplicates, missing months."""

from __future__ import annotations

import pandas as pd

from services.csv_schema import detect_csv_schema
from services.csv_validation import validate_dataframe


def test_sample_csv_validates_ready(sample_df, fast_config):
    schema = detect_csv_schema(sample_df, config=fast_config)
    cleaned, report = validate_dataframe(sample_df, schema, config=fast_config)
    assert report.is_valid
    assert report.valid_row_count == len(sample_df)
    assert report.month_count >= 36
    assert report.readiness in {"ready", "needs_review"}
    assert report.total_liability > 0
    assert report.total_volume > 0
    assert len(cleaned) == report.valid_row_count


def test_invalid_date_and_amount_rejected(fast_config):
    df = pd.DataFrame(
        {
            "Chargeback Date": ["2024-01-15", "not-a-date", "2024-02-01"],
            "Chargeback ID": ["CB-1", "CB-2", "CB-3"],
            "Amount": [100.0, 200.0, "bad"],
            "Business Unit": ["Enterprise"] * 3,
            "Product": ["Cloud Services"] * 3,
            "Reason Code": ["4837"] * 3,
            "Status": ["Open"] * 3,
        }
    )
    schema = detect_csv_schema(df, config=fast_config)
    cleaned, report = validate_dataframe(df, schema, config=fast_config)
    assert not report.is_valid or report.rejected_row_count >= 2
    assert report.valid_row_count == 1
    assert len(cleaned) == 1
    codes = {i.code for i in report.issues}
    assert "invalid_date" in codes or "invalid_amount" in codes


def test_duplicate_keys_warn(fast_config):
    from models.domain_types import SchemaMappingOverride
    from services.csv_schema import apply_schema_overrides

    df = pd.DataFrame(
        {
            "Chargeback Date": ["2024-01-15", "2024-01-15"],
            "Chargeback ID": ["CB-DUP", "CB-DUP"],
            "Amount": [50.0, 60.0],
            "Business Unit": ["Consumer", "Consumer"],
            "Product": ["Mobile Plans", "Mobile Plans"],
            "Reason Code": ["4837", "4837"],
            "Status": ["Open", "Open"],
        }
    )
    schema = apply_schema_overrides(
        detect_csv_schema(df, config=fast_config),
        SchemaMappingOverride(
            case_id_column="Chargeback ID",
            reason_code_column="Reason Code",
            count_column="",
        ),
        df=df,
    )
    assert schema.case_id_column == "Chargeback ID"
    assert schema.count_column is None
    _, report = validate_dataframe(df, schema, config=fast_config)
    assert report.duplicate_row_count == 2
    assert any(i.code == "duplicate_keys" for i in report.issues)


def test_missing_months_detected(fast_config):
    rows = []
    for month in ["2024-01", "2024-02", "2024-04"]:
        rows.append(
            {
                "Chargeback Date": f"{month}-10",
                "Chargeback ID": f"CB-{month}",
                "Amount": 100.0,
                "Business Unit": "Enterprise",
                "Product": "Cloud Services",
                "Reason Code": "4837",
                "Status": "Open",
            }
        )
    df = pd.DataFrame(rows)
    schema = detect_csv_schema(df, config=fast_config)
    _, report = validate_dataframe(df, schema, config=fast_config)
    assert "2024-03" in report.missing_months
