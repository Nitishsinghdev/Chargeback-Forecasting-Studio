"""Monthly aggregation: charge-level vs pre-aggregated counts."""

from __future__ import annotations

import pandas as pd

from models.domain_types import ForecastMetric, SchemaMappingOverride, VolumeAggregationMode
from services.csv_schema import apply_schema_overrides, detect_csv_schema
from services.csv_validation import validate_dataframe
from services.monthly_processor import aggregate_to_monthly


def test_unique_case_volume_from_sample(sample_df, fast_config):
    schema = detect_csv_schema(sample_df, config=fast_config)
    schema = apply_schema_overrides(
        schema,
        SchemaMappingOverride(case_id_column="Chargeback ID", count_column=None),
        df=sample_df,
    )
    cleaned, _ = validate_dataframe(sample_df, schema, config=fast_config)
    series = aggregate_to_monthly(cleaned, schema, metric=ForecastMetric.VOLUME, config=fast_config)
    assert len(series.period_index) >= 36
    assert len(series.values) == len(series.period_index)
    assert all(v > 0 for v in series.values)
    assert series.metadata["volume_mode"] == VolumeAggregationMode.UNIQUE_CASE.value


def test_row_count_override_changes_volume(sample_df, fast_config):
    schema = detect_csv_schema(sample_df, config=fast_config)
    schema = apply_schema_overrides(
        schema,
        SchemaMappingOverride(volume_aggregation_mode=VolumeAggregationMode.ROW_COUNT.value),
        df=sample_df,
    )
    cleaned, _ = validate_dataframe(sample_df, schema, config=fast_config)
    unique = aggregate_to_monthly(cleaned, schema, metric=ForecastMetric.VOLUME, config=fast_config)
    row_schema = schema
    assert row_schema.volume_aggregation_mode == VolumeAggregationMode.ROW_COUNT
    row_series = aggregate_to_monthly(cleaned, row_schema, metric=ForecastMetric.VOLUME, config=fast_config)
    assert sum(row_series.values) >= sum(unique.values)


def test_liability_sums_amounts(sample_df, fast_config):
    schema = detect_csv_schema(sample_df, config=fast_config)
    cleaned, _ = validate_dataframe(sample_df, schema, config=fast_config)
    series = aggregate_to_monthly(cleaned, schema, metric=ForecastMetric.LIABILITY, config=fast_config)
    assert len(series.values) >= 36
    assert sum(series.values) > 0


def test_preaggregated_sum_count():
    df = pd.DataFrame(
        {
            "billing_date": ["2024-01-01", "2024-01-01", "2024-02-01"],
            "volume": [10, 5, 20],
            "net_amount": [1000.0, 500.0, 2000.0],
        }
    )
    schema = detect_csv_schema(df)
    schema = apply_schema_overrides(
        schema,
        SchemaMappingOverride(
            date_column="billing_date",
            count_column="volume",
            amount_column="net_amount",
            volume_aggregation_mode=VolumeAggregationMode.SUM_COUNT.value,
        ),
        df=df,
    )
    cleaned, report = validate_dataframe(df, schema)
    assert report.is_valid
    vol = aggregate_to_monthly(cleaned, schema, metric=ForecastMetric.VOLUME)
    assert vol.values[0] == 15.0
    liab = aggregate_to_monthly(cleaned, schema, metric=ForecastMetric.LIABILITY)
    assert liab.values[0] == 1500.0


def test_business_unit_filter(sample_df, fast_config):
    schema = detect_csv_schema(sample_df, config=fast_config)
    cleaned, _ = validate_dataframe(sample_df, schema, config=fast_config)
    all_series = aggregate_to_monthly(cleaned, schema, config=fast_config)
    ent = aggregate_to_monthly(
        cleaned, schema, business_unit="Enterprise", config=fast_config
    )
    assert sum(ent.values) < sum(all_series.values)
