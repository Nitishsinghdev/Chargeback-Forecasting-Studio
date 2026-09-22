"""Shared domain datatypes for services and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class ModelKind(str, Enum):
    """Forecast model identifier."""

    SARIMAX = "sarimax"
    SEASONAL_NAIVE = "seasonal_naive"
    MOVING_AVERAGE = "moving_average"
    NAIVE = "naive"
    HOLT_WINTERS = "holt_winters"
    LINEAR_TREND = "linear_trend"
    MEAN = "mean"


class ValidationStrategy(str, Enum):
    HOLDOUT = "holdout"
    ROLLING = "rolling"
    HOLDOUT_AND_ROLLING = "holdout_and_rolling"


class VolumeAggregationMode(str, Enum):
    """How monthly volume is computed from row-level data."""

    UNIQUE_CASE = "unique_case"
    ROW_COUNT = "row_count"
    SUM_COUNT = "sum_count"


class ForecastMetric(str, Enum):
    VOLUME = "volume"
    LIABILITY = "liability"


@dataclass(frozen=True)
class ColumnProfile:
    """Detection profile for a single CSV column."""

    column_name: str
    inferred_role: str
    confidence: float
    parse_ratio: Optional[float] = None
    numeric_ratio: Optional[float] = None
    nunique: Optional[int] = None
    sample_values: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SchemaMappingOverride:
    """
    User-supplied column mapping overrides (Flask form / API).

    Optional role fields: ``None`` keeps auto-detected value; ``""`` clears mapping.
    """

    date_column: Optional[str] = None
    amount_column: Optional[str] = None
    count_column: Optional[str] = None
    business_unit_column: Optional[str] = None
    product_column: Optional[str] = None
    status_column: Optional[str] = None
    case_id_column: Optional[str] = None
    reason_code_column: Optional[str] = None
    category_column: Optional[str] = None
    volume_aggregation_mode: Optional[str] = None


@dataclass(frozen=True)
class DetectedSchema:
    """Result of dynamic CSV schema detection with role assignments."""

    date_column: str
    amount_column: Optional[str]
    count_column: Optional[str]
    business_unit_column: Optional[str]
    product_column: Optional[str]
    status_column: Optional[str]
    case_id_column: Optional[str]
    reason_code_column: Optional[str]
    category_column: Optional[str]
    category_candidate_columns: List[str]
    segment_columns: List[str]
    column_profiles: Dict[str, ColumnProfile]
    volume_aggregation_mode: VolumeAggregationMode
    all_columns: List[str]
    row_count: int
    detection_notes: List[str] = field(default_factory=list)
    user_overrides_applied: bool = False

    @property
    def supports_liability(self) -> bool:
        return self.amount_column is not None

    @property
    def supports_volume(self) -> bool:
        return True


@dataclass(frozen=True)
class ValidationIssue:
    """Single validation finding."""

    severity: str  # "error" | "warning"
    code: str
    message: str
    column: Optional[str] = None
    row_index: Optional[int] = None


@dataclass(frozen=True)
class ValidationReport:
    """Aggregated CSV validation outcome with readiness summary."""

    is_valid: bool
    issues: List[ValidationIssue]
    uploaded_row_count: int
    valid_row_count: int
    rejected_row_count: int
    duplicate_row_count: int
    earliest_date: Optional[datetime]
    latest_date: Optional[datetime]
    month_count: int
    missing_months: List[str]
    total_volume: float
    total_liability: float
    warnings: List[str]
    readiness: str
    rejected_row_indexes: List[int] = field(default_factory=list)
    rejected_records_sample: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def invalid_row_count(self) -> int:
        """Backward-compatible alias for rejected rows."""
        return self.rejected_row_count


@dataclass(frozen=True)
class MonthlySeries:
    """Monthly aggregated target series."""

    period_index: List[datetime]
    values: List[float]
    segment_key: Optional[str] = None
    metric: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedEvent:
    """Structured event extracted from natural language."""

    raw_text: str
    event_name: Optional[str]
    effect_type: str  # "percent" | "absolute" | "unknown"
    effect_value: Optional[float]
    direction: Optional[str]  # increase | decrease | neutral | unknown
    metric: Optional[str]  # volume | liability | both | unknown
    business_unit: Optional[str]
    product: Optional[str]
    start_period: Optional[datetime]
    end_period: Optional[datetime]
    segment_hint: Optional[str]
    confidence: float
    parse_notes: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    approved: bool = False
    annotation: Optional[str] = None


def event_has_adjustable_magnitude(event: ParsedEvent) -> bool:
    """True only when a numeric magnitude is present and parseable."""
    if event.effect_value is None:
        return False
    if event.effect_type not in {"percent", "absolute"}:
        return False
    return True


@dataclass(frozen=True)
class SafeMetricSet:
    """Validation metrics with safe handling of zeros and small denominators."""

    mape: Optional[float]
    smape: Optional[float]
    rmse: Optional[float]
    mae: Optional[float]
    bias: Optional[float]
    n_obs: int
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelSelectionResult:
    """Outcome of model search including fallback metadata."""

    model_kind: ModelKind
    order: Optional[tuple[int, ...]]
    seasonal_order: Optional[tuple[int, ...]]
    metrics: SafeMetricSet
    validation_strategy: ValidationStrategy
    fallback_reason: Optional[str]
    search_notes: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ForecastPoint:
    """Single forecast period."""

    period: datetime
    baseline: float
    adjusted: float
    lower: Optional[float]
    upper: Optional[float]
    event_adjustment: float
    metric: Optional[str] = None


@dataclass(frozen=True)
class SameMonthHistoricalStats:
    """Same calendar month statistics across prior years."""

    calendar_month: int
    historical_values: List[float]
    historical_periods: List[datetime]
    avg: Optional[float]
    min: Optional[float]
    max: Optional[float]
    three_year_trend_pct: Optional[float]
    years_available: int


@dataclass(frozen=True)
class SegmentComparison:
    """Same-month comparison for a segment and metric."""

    segment: str
    metric: str
    reference_period: datetime
    reference_value: float
    forecast_period: datetime
    forecast_baseline: float
    forecast_adjusted: float
    delta_baseline: float
    delta_adjusted: float
    pct_change_baseline: Optional[float]
    pct_change_adjusted: Optional[float]
    same_month_stats: Optional[SameMonthHistoricalStats] = None


@dataclass(frozen=True)
class ForecastExplanation:
    """Human-readable explanation fragments."""

    summary: str
    model_rationale: str
    event_rationale: str
    validation_rationale: str
    caveats: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ForecastKpis:
    """High-level KPIs for a forecast run."""

    total_baseline_horizon: float
    total_adjusted_horizon: float
    total_event_impact: float
    avg_monthly_baseline: float
    avg_monthly_adjusted: float
    model_kind: ModelKind
    fallback_used: bool
    fallback_reason: Optional[str]
    validation_mape: Optional[float]
    metric: Optional[str] = None


@dataclass(frozen=True)
class MetricForecastBundle:
    """Forecast output for a single metric path."""

    metric: ForecastMetric
    forecasts: List[ForecastPoint]
    model_selection: ModelSelectionResult
    monthly_history: MonthlySeries
    segment_comparisons: List[SegmentComparison]
    explanations: ForecastExplanation
    kpis: ForecastKpis


@dataclass
class ForecastRunResult:
    """Complete orchestrated forecast output (dual-metric capable)."""

    forecasts: List[ForecastPoint]
    model_selection: ModelSelectionResult
    segment_comparisons: List[SegmentComparison]
    explanations: ForecastExplanation
    kpis: ForecastKpis
    monthly_history: MonthlySeries
    events_applied: Sequence[ParsedEvent]
    primary_metric: ForecastMetric = ForecastMetric.VOLUME
    volume: Optional[MetricForecastBundle] = None
    liability: Optional[MetricForecastBundle] = None
    segmented_results: Dict[str, MetricForecastBundle] = field(default_factory=dict)
