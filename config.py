"""Application configuration loaded from environment with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class SarimaxSearchConfig:
    """Bounds for controlled SARIMAX hyperparameter search."""

    p_range: Tuple[int, ...] = (0, 1, 2)
    d_range: Tuple[int, ...] = (0, 1)
    q_range: Tuple[int, ...] = (0, 1, 2)
    P_range: Tuple[int, ...] = (0, 1)
    D_range: Tuple[int, ...] = (0, 1)
    Q_range: Tuple[int, ...] = (0, 1)
    max_candidates: int = 48
    fit_timeout_seconds: float = 120.0
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True


@dataclass(frozen=True)
class ValidationConfig:
    """Time-series validation settings (no test leakage into training)."""

    holdout_months: int = 3
    rolling_folds: int = 3
    min_train_months: int = 12
    seasonal_period: int = 12


@dataclass(frozen=True)
class ForecastConfig:
    """Forecast horizon and baseline behavior."""

    horizon_months: int = 3
    confidence_level: float = 0.95
    history_months: int = 36
    moving_average_window: int = 3


@dataclass(frozen=True)
class CsvConfig:
    """CSV ingestion limits and detection heuristics."""

    max_rows_sample: int = 10_000_000
    max_file_bytes: int = 250 * 1024 * 1024
    min_date_parse_ratio: float = 0.85
    min_numeric_ratio: float = 0.85
    date_column_candidates: Tuple[str, ...] = field(
        default_factory=lambda: (
            "date",
            "period",
            "month",
            "timestamp",
            "billing_date",
            "charge_date",
            "posting_date",
        )
    )
    amount_column_candidates: Tuple[str, ...] = field(
        default_factory=lambda: (
            "amount",
            "charge",
            "chargeback",
            "value",
            "cost",
            "total",
            "net_amount",
        )
    )
    segment_column_candidates: Tuple[str, ...] = field(
        default_factory=lambda: (
            "segment",
            "category",
            "product",
            "service",
            "region",
            "department",
            "cost_center",
            "gl_code",
        )
    )
    count_column_candidates: Tuple[str, ...] = field(
        default_factory=lambda: (
            "count",
            "volume",
            "qty",
            "quantity",
            "case_count",
            "transactions",
        )
    )
    business_unit_candidates: Tuple[str, ...] = field(
        default_factory=lambda: (
            "business_unit",
            "bu",
            "unit",
            "division",
            "org",
        )
    )
    product_column_candidates: Tuple[str, ...] = field(
        default_factory=lambda: ("product", "product_name", "sku", "offering")
    )
    status_column_candidates: Tuple[str, ...] = field(
        default_factory=lambda: ("status", "state", "dispute_status", "case_status")
    )
    case_id_candidates: Tuple[str, ...] = field(
        default_factory=lambda: (
            "chargeback_id",
            "chargeback id",
            "case_id",
            "case",
            "dispute_id",
            "dispute",
            "reference_id",
            "reference",
            "ticket_id",
            "claim_id",
            "id",
        )
    )
    reason_code_candidates: Tuple[str, ...] = field(
        default_factory=lambda: ("reason_code", "reason", "chargeback_reason", "code")
    )
    category_column_candidates: Tuple[str, ...] = field(
        default_factory=lambda: ("category", "type", "chargeback_type", "reason_category")
    )
    rejected_sample_limit: int = 50


@dataclass(frozen=True)
class AppConfig:
    """Root configuration object for domain services."""

    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    database_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "data" / "analysis.db"
    )
    csv: CsvConfig = field(default_factory=CsvConfig)
    sarimax: SarimaxSearchConfig = field(default_factory=SarimaxSearchConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    export_formula_prefix: str = "'"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build configuration from environment variables."""
        base = Path(os.getenv("CHARGEBACK_APP_BASE_DIR", str(Path(__file__).resolve().parent)))
        db_path = Path(
            os.getenv(
                "CHARGEBACK_DATABASE_PATH",
                str(base / "data" / "analysis.db"),
            )
        )
        val_cfg = ValidationConfig(
            holdout_months=_env_int("CHARGEBACK_HOLDOUT_MONTHS", 3),
            rolling_folds=_env_int("CHARGEBACK_ROLLING_FOLDS", 3),
            min_train_months=_env_int("CHARGEBACK_MIN_TRAIN_MONTHS", 12),
            seasonal_period=_env_int("CHARGEBACK_SEASONAL_PERIOD", 12),
        )
        fc_cfg = ForecastConfig(
            horizon_months=_env_int("CHARGEBACK_FORECAST_HORIZON", 3),
            confidence_level=_env_float("CHARGEBACK_CONFIDENCE_LEVEL", 0.95),
            history_months=_env_int("CHARGEBACK_HISTORY_MONTHS", 36),
            moving_average_window=_env_int("CHARGEBACK_MOVING_AVERAGE_WINDOW", 3),
        )
        csv_cfg = CsvConfig(
            max_rows_sample=_env_int("CHARGEBACK_CSV_MAX_ROWS_SAMPLE", 10_000_000),
            max_file_bytes=_env_int("CHARGEBACK_CSV_MAX_FILE_BYTES", 250 * 1024 * 1024),
            min_date_parse_ratio=_env_float("CHARGEBACK_CSV_MIN_DATE_PARSE_RATIO", 0.85),
            min_numeric_ratio=_env_float("CHARGEBACK_CSV_MIN_NUMERIC_RATIO", 0.85),
            rejected_sample_limit=_env_int("CHARGEBACK_REJECTED_SAMPLE_LIMIT", 50),
        )
        sarimax_cfg = SarimaxSearchConfig(
            max_candidates=_env_int("CHARGEBACK_SARIMAX_MAX_CANDIDATES", 48),
            fit_timeout_seconds=_env_float("CHARGEBACK_SARIMAX_FIT_TIMEOUT", 120.0),
            enforce_stationarity=_env_bool("CHARGEBACK_SARIMAX_ENFORCE_STATIONARITY", True),
            enforce_invertibility=_env_bool("CHARGEBACK_SARIMAX_ENFORCE_INVERTIBILITY", True),
        )
        return cls(
            base_dir=base,
            database_path=db_path,
            csv=csv_cfg,
            sarimax=sarimax_cfg,
            validation=val_cfg,
            forecast=fc_cfg,
            export_formula_prefix=os.getenv("CHARGEBACK_EXPORT_FORMULA_PREFIX", "'"),
            log_level=os.getenv("CHARGEBACK_LOG_LEVEL", "INFO"),
        )


def get_config() -> AppConfig:
    """Return cached-style config singleton for the process."""
    if not hasattr(get_config, "_instance"):
        get_config._instance = AppConfig.from_env()  # type: ignore[attr-defined]
    return get_config._instance  # type: ignore[attr-defined]
