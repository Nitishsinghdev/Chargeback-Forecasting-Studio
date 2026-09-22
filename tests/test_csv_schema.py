"""Dynamic schema detection and override mapping."""

from __future__ import annotations

import pandas as pd
import pytest

from models.domain_types import SchemaMappingOverride, VolumeAggregationMode
from services.csv_schema import apply_schema_overrides, detect_csv_schema


def test_detects_core_chargeback_columns(sample_df):
    schema = detect_csv_schema(sample_df)
    assert schema.date_column == "Chargeback Date"
    assert schema.amount_column == "Amount"
    assert schema.business_unit_column == "Business Unit"
    assert schema.product_column == "Product"
    assert schema.status_column == "Status"
    assert schema.supports_liability is True
    assert "Business Unit" in schema.segment_columns


def test_chargeback_id_should_be_case_id_not_count(sample_df):
    """Regression: Chargeback ID must win over Reason Code for case_id / volume mode."""
    schema = detect_csv_schema(sample_df)
    assert schema.case_id_column == "Chargeback ID"
    assert schema.reason_code_column == "Reason Code"
    assert schema.count_column is None
    assert schema.volume_aggregation_mode == VolumeAggregationMode.UNIQUE_CASE


def test_override_restores_unique_case_volume(sample_df):
    base = detect_csv_schema(sample_df)
    overrides = SchemaMappingOverride(
        case_id_column="Chargeback ID",
        count_column="",
        reason_code_column="Reason Code",
    )
    schema = apply_schema_overrides(base, overrides, df=sample_df)
    assert schema.case_id_column == "Chargeback ID"
    assert schema.count_column is None
    assert schema.volume_aggregation_mode == VolumeAggregationMode.UNIQUE_CASE


def test_empty_string_clears_inferred_optional_role():
    df = pd.DataFrame(
        {
            "month": ["2024-01-01", "2024-02-01"],
            "case_count": [120, 130],
            "total_amount": [50000.0, 52000.0],
        }
    )
    base = detect_csv_schema(df)
    assert base.count_column is not None
    cleared = apply_schema_overrides(
        base,
        SchemaMappingOverride(count_column=""),
        df=df,
    )
    assert cleared.count_column is None
    assert cleared.date_column == base.date_column


def test_column_profiles_have_confidence(sample_df):
    schema = detect_csv_schema(sample_df)
    date_prof = schema.column_profiles["Chargeback Date"]
    assert date_prof.inferred_role == "date"
    assert date_prof.confidence >= 0.4
    assert date_prof.parse_ratio is not None and date_prof.parse_ratio >= 0.85


def test_apply_overrides_remaps_roles(sample_df):
    base = detect_csv_schema(sample_df)
    overrides = SchemaMappingOverride(
        date_column="Chargeback Date",
        amount_column="Amount",
        case_id_column="Chargeback ID",
        business_unit_column="Business Unit",
        product_column="Product",
        volume_aggregation_mode=VolumeAggregationMode.ROW_COUNT.value,
    )
    updated = apply_schema_overrides(base, overrides, df=sample_df)
    assert updated.volume_aggregation_mode == VolumeAggregationMode.ROW_COUNT
    assert updated.user_overrides_applied is True
    assert "user_overrides_applied" in updated.detection_notes


def test_override_unknown_column_raises(sample_df):
    base = detect_csv_schema(sample_df)
    overrides = SchemaMappingOverride(date_column="Not A Column")
    with pytest.raises(ValueError, match="not found"):
        apply_schema_overrides(base, overrides, df=sample_df)


def test_preaggregated_monthly_schema():
    df = pd.DataFrame(
        {
            "month": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "case_count": [120, 130, 125],
            "total_amount": [50000.0, 52000.0, 51000.0],
        }
    )
    schema = detect_csv_schema(df)
    assert schema.date_column == "month"
    assert schema.count_column is not None
    assert schema.amount_column is not None


def test_empty_frame_raises():
    with pytest.raises(ValueError, match="empty"):
        detect_csv_schema(pd.DataFrame())
