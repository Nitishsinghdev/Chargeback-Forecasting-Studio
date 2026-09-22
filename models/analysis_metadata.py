"""SQLite persistence for analysis runs, uploads, and forecast metadata."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from config import AppConfig, get_config


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class AnalysisRunRecord:
    """Metadata for a single end-to-end analysis."""

    id: Optional[int] = None
    created_at: str = field(default_factory=_utc_now_iso)
    source_filename: Optional[str] = None
    detected_schema_json: Optional[str] = None
    validation_summary_json: Optional[str] = None
    model_kind: Optional[str] = None
    fallback_reason: Optional[str] = None
    metrics_json: Optional[str] = None
    forecast_json: Optional[str] = None
    events_json: Optional[str] = None
    status: str = "pending"
    error_message: Optional[str] = None


class AnalysisMetadataStore:
    """Lightweight SQLite store for analysis artifacts."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self._config = config or get_config()
        self._db_path = Path(self._config.database_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def database_path(self) -> Path:
        return self._db_path

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source_filename TEXT,
                    detected_schema_json TEXT,
                    validation_summary_json TEXT,
                    model_kind TEXT,
                    fallback_reason TEXT,
                    metrics_json TEXT,
                    forecast_json TEXT,
                    events_json TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_analysis_runs_created
                    ON analysis_runs(created_at DESC);
                """
            )

    def create_run(
        self,
        source_filename: Optional[str] = None,
        status: str = "pending",
    ) -> AnalysisRunRecord:
        record = AnalysisRunRecord(source_filename=source_filename, status=status)
        with self._connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO analysis_runs (created_at, source_filename, status)
                VALUES (?, ?, ?)
                """,
                (record.created_at, record.source_filename, record.status),
            )
            record.id = int(cur.lastrowid)
        return record

    def update_run(self, record: AnalysisRunRecord) -> AnalysisRunRecord:
        if record.id is None:
            raise ValueError("Cannot update run without id")
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE analysis_runs SET
                    detected_schema_json = ?,
                    validation_summary_json = ?,
                    model_kind = ?,
                    fallback_reason = ?,
                    metrics_json = ?,
                    forecast_json = ?,
                    events_json = ?,
                    status = ?,
                    error_message = ?
                WHERE id = ?
                """,
                (
                    record.detected_schema_json,
                    record.validation_summary_json,
                    record.model_kind,
                    record.fallback_reason,
                    record.metrics_json,
                    record.forecast_json,
                    record.events_json,
                    record.status,
                    record.error_message,
                    record.id,
                ),
            )
        return record

    def get_run(self, run_id: int) -> Optional[AnalysisRunRecord]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_runs(self, limit: int = 50) -> List[AnalysisRunRecord]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM analysis_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> AnalysisRunRecord:
        return AnalysisRunRecord(
            id=int(row["id"]),
            created_at=str(row["created_at"]),
            source_filename=row["source_filename"],
            detected_schema_json=row["detected_schema_json"],
            validation_summary_json=row["validation_summary_json"],
            model_kind=row["model_kind"],
            fallback_reason=row["fallback_reason"],
            metrics_json=row["metrics_json"],
            forecast_json=row["forecast_json"],
            events_json=row["events_json"],
            status=str(row["status"]),
            error_message=row["error_message"],
        )

    @staticmethod
    def dump_json(payload: Any) -> str:
        return json.dumps(payload, default=str)

    @staticmethod
    def load_json(payload: Optional[str]) -> Any:
        if not payload:
            return None
        return json.loads(payload)


def persist_forecast_run(
    store: AnalysisMetadataStore,
    run_id: int,
    *,
    schema: Dict[str, Any],
    validation: Dict[str, Any],
    model_kind: str,
    fallback_reason: Optional[str],
    metrics: Dict[str, Any],
    forecast: Dict[str, Any],
    events: List[Dict[str, Any]],
    status: str = "completed",
    error_message: Optional[str] = None,
) -> AnalysisRunRecord:
    """Convenience helper to update a run with serialized payloads."""
    record = store.get_run(run_id)
    if record is None:
        raise ValueError(f"Analysis run {run_id} not found")
    record.detected_schema_json = store.dump_json(schema)
    record.validation_summary_json = store.dump_json(validation)
    record.model_kind = model_kind
    record.fallback_reason = fallback_reason
    record.metrics_json = store.dump_json(metrics)
    record.forecast_json = store.dump_json(forecast)
    record.events_json = store.dump_json(events)
    record.status = status
    record.error_message = error_message
    return store.update_run(record)
