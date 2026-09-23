"""Flask executive UI for chargeback forecasting (owns routes/templates only)."""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from flask import (
    Flask,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session as flask_session,
    url_for,
)
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import AppConfig, CsvConfig, ForecastConfig, ValidationConfig, get_config
from models.domain_types import (
    ForecastMetric,
    ForecastRunResult,
    MetricForecastBundle,
    ParsedEvent,
    SchemaMappingOverride,
    VolumeAggregationMode,
)
from services.csv_schema import apply_schema_overrides, detect_csv_schema
from services.csv_validation import validate_dataframe
from services.event_parser import parse_event_batch, parse_event_text
from services.export_safety import dataframe_to_safe_csv, sanitize_dataframe_for_export
from models.analysis_metadata import persist_forecast_run
from services.forecast_service import (
    ForecastOrchestrator,
    forecast_result_to_dict,
    validation_report_to_dict,
    _schema_to_dict,
    _event_to_dict,
    _bundle_to_dict,
)
from services.monthly_processor import list_column_values, list_segment_values
from services.csv_ingest import read_csv_limited

APP_ROOT = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {".csv"}
SESSION_TTL_SECONDS = 3600 * 4
MAX_SEGMENTS_BREAKDOWN = 12

WIZARD_STEPS = ["upload", "schema", "validation", "configure", "events", "dashboard"]

ALERT_LEVEL_CHOICES = ("low", "medium", "high", "critical")
ALERT_LEVEL_TO_BOOTSTRAP = {
    "low": "info",
    "medium": "warning",
    "high": "danger",
    "critical": "danger",
}

_log = logging.getLogger("chargeback_app")

_sessions_lock = threading.RLock()
_sessions: Dict[str, "AnalysisSession"] = {}


@dataclass
class SessionSettings:
    history_months: int = 36
    horizon_months: int = 3
    confidence_level: float = 0.95
    holdout_months: int = 3
    min_train_months: int = 12
    rolling_folds: int = 3
    currency_code: str = "USD"
    mape_alert_threshold: float = 25.0
    mape_strong_threshold: float = 15.0
    mape_acceptable_threshold: float = 25.0
    global_alert_level: str = "medium"
    mape_alert_level: str = ""
    monthly_volume_alert_threshold: Optional[float] = None
    monthly_volume_alert_level: str = ""
    monthly_liability_alert_threshold: Optional[float] = None
    monthly_liability_alert_level: str = ""
    percent_growth_alert_threshold: Optional[float] = 20.0
    percent_growth_alert_level: str = ""
    ci_width_alert_threshold_pct: Optional[float] = 40.0
    ci_width_alert_level: str = ""
    segment_column: str = ""
    segment_value: str = ""
    business_unit: str = ""
    product: str = ""
    include_liability: bool = True
    run_segment_breakdown: bool = True


@dataclass
class AnalysisSession:
    session_id: str
    csrf_token: str
    created_at: float
    original_filename: str = ""
    raw_df: Optional[pd.DataFrame] = None
    cleaned_df: Optional[pd.DataFrame] = None
    schema: Any = None
    validation_report: Any = None
    settings: SessionSettings = field(default_factory=SessionSettings)
    events: List[ParsedEvent] = field(default_factory=list)
    event_text_batch: str = ""
    result: Optional[ForecastRunResult] = None
    segment_bundles: Dict[str, MetricForecastBundle] = field(default_factory=dict)
    analysis_run_id: Optional[int] = None
    temp_paths: List[Path] = field(default_factory=list)
    step: str = "upload"

    def touch(self) -> None:
        self.created_at = time.time()


def _is_production() -> bool:
    env = os.getenv("FLASK_ENV", os.getenv("ENV", "")).lower()
    return env in {"production", "prod"}


def _secret_key() -> str:
    key = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY")
    if not key:
        if _is_production():
            raise RuntimeError("FLASK_SECRET_KEY must be set in production")
        key = secrets.token_hex(32)
        _log.warning("Using ephemeral secret key; set FLASK_SECRET_KEY for production.")
    return key


def _format_number(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(num) >= 1_000_000:
        return f"{num:,.0f}"
    if abs(num) >= 1000:
        return f"{num:,.2f}"
    return f"{num:,.4g}"


def _format_optional(value: Any, suffix: str = "", default: str = "—") -> str:
    if value is None:
        return default
    return f"{value:.2f}{suffix}" if isinstance(value, (int, float)) else str(value)


def _log_file_path() -> Path:
    """Location of the rotating application log."""
    return APP_ROOT / "logs" / "app.log"


def _configure_logging(level: int) -> None:
    """Log to console and to a rotating file so failures can be diagnosed later."""
    logging.basicConfig(level=level)
    path = _log_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    already = any(
        isinstance(h, RotatingFileHandler) and Path(getattr(h, "baseFilename", "")) == path
        for h in root.handlers
    )
    if already:
        return
    handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler.setLevel(level)
    root.addHandler(handler)


def _wants_json_response() -> bool:
    """True only for API-style callers, so browsers still get the HTML error page."""
    if request.path.endswith("/api/parse-events") or "/api/" in request.path:
        return True
    if request.is_json:
        return True
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return True
    accept = request.accept_mimetypes
    return bool(accept["application/json"] > accept["text/html"])


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = _secret_key()
    app.config["MAX_CONTENT_LENGTH"] = get_config().csv.max_file_bytes

    _configure_logging(getattr(logging, get_config().log_level.upper(), logging.INFO))

    app.jinja_env.filters["format_number"] = _format_number
    app.jinja_env.filters["format_optional"] = _format_optional

    @app.before_request
    def _purge_stale_sessions() -> None:
        _purge_expired_sessions()

    @app.context_processor
    def _inject_globals() -> Dict[str, Any]:
        def step_index(name: str) -> int:
            try:
                return WIZARD_STEPS.index(name)
            except ValueError:
                return -1

        sid = getattr(g, "session_id", None)
        csrf = _session_csrf(sid) if sid else getattr(g, "upload_csrf", "") or flask_session.get("upload_csrf", "")
        return {
            "step_index": step_index,
            "format_number": _format_number,
            "format_optional": _format_optional,
            "session_id": sid,
            "csrf_token": csrf,
            "current_step": getattr(g, "wizard_step", "upload"),
        }

    @app.errorhandler(HTTPException)
    def _http_error(exc: HTTPException):
        if _wants_json_response():
            return jsonify({"error": exc.description or exc.name}), exc.code
        return (
            render_template(
                "error.html",
                title=exc.name,
                message=exc.description or "Request could not be completed.",
            ),
            exc.code,
        )

    @app.errorhandler(Exception)
    def _unhandled_error(exc: Exception):
        reference = uuid.uuid4().hex[:8]
        _log.exception("Unhandled error [%s] on %s %s: %s", reference, request.method, request.path, exc)
        msg = (
            "An unexpected error occurred. Please try again, or share reference "
            f"{reference} with support."
        )
        if _wants_json_response():
            return jsonify({"error": msg, "reference": reference}), 500
        return render_template(
            "error.html",
            title="Error",
            message=msg,
            reference=reference,
            log_path=str(_log_file_path()),
        ), 500

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(_exc: RequestEntityTooLarge):
        flash("Uploaded file exceeds the maximum allowed size.", "danger")
        return redirect(url_for("upload"))

    register_routes(app)
    return app


def register_routes(app: Flask) -> None:
    @app.route("/")
    def index():
        g.wizard_step = "upload"
        return render_template("index.html")

    @app.route("/upload", methods=["GET", "POST"])
    def upload():
        g.wizard_step = "upload"
        if request.method == "GET":
            if not flask_session.get("upload_csrf"):
                flask_session["upload_csrf"] = secrets.token_urlsafe(32)
            g.upload_csrf = flask_session["upload_csrf"]
            return render_template("upload.html")

        _validate_csrf_form(None)
        file = request.files.get("csv_file")
        if not file or not file.filename:
            flash("Please choose a CSV file.", "warning")
            return redirect(url_for("upload"))

        filename = secure_filename(file.filename)
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            flash("Only .csv files are allowed.", "danger")
            return redirect(url_for("upload"))

        cfg = get_config()
        data = file.read()
        if len(data) > cfg.csv.max_file_bytes:
            flash("File exceeds maximum size limit.", "danger")
            return redirect(url_for("upload"))

        try:
            raw = read_csv_limited(BytesIO(data), config=cfg)
        except Exception as exc:
            _log.warning("CSV read failed: %s", exc)
            flash(f"Could not read CSV: {exc}", "danger")
            return redirect(url_for("upload"))

        if raw.empty:
            flash("CSV appears empty.", "danger")
            return redirect(url_for("upload"))

        sid = str(uuid.uuid4())
        csrf = secrets.token_urlsafe(32)
        sess = AnalysisSession(session_id=sid, csrf_token=csrf, created_at=time.time())
        sess.original_filename = filename
        sess.raw_df = raw
        try:
            sess.schema = detect_csv_schema(raw, config=cfg)
        except ValueError as exc:
            flash(str(exc), "danger")
            return redirect(url_for("upload"))

        _store_session(sess)
        flask_session["cb_sid"] = sid
        return redirect(url_for("session_schema", session_id=sid))

    @app.route("/s/<session_id>/schema", methods=["GET", "POST"])
    def session_schema(session_id: str):
        sess = _load_session(session_id)
        g.session_id = session_id
        g.wizard_step = "schema"
        if sess.schema is None or sess.raw_df is None:
            abort(404)

        if request.method == "GET":
            return render_template(
                "schema.html",
                schema=sess.schema,
                columns=sess.schema.all_columns,
                volume_modes=[m.value for m in VolumeAggregationMode],
            )

        _validate_csrf_form(sess)
        try:
            overrides = _overrides_from_form(request.form, sess.schema.all_columns)
            schema = apply_schema_overrides(sess.schema, overrides, df=sess.raw_df)
            cleaned, report = validate_dataframe(sess.raw_df, schema, config=_session_config(sess))
        except ValueError as exc:
            flash(str(exc), "danger")
            return redirect(url_for("session_schema", session_id=session_id))

        sess.schema = schema
        sess.cleaned_df = cleaned
        sess.validation_report = report
        sess.step = "validation"
        sess.touch()
        return redirect(url_for("session_validation", session_id=session_id))

    @app.route("/s/<session_id>/validation")
    def session_validation(session_id: str):
        sess = _load_session(session_id)
        g.session_id = session_id
        g.wizard_step = "validation"
        if sess.validation_report is None:
            abort(404)
        return render_template("validation.html", report=sess.validation_report)

    @app.route("/s/<session_id>/configure", methods=["GET", "POST"])
    def session_configure(session_id: str):
        sess = _load_session(session_id)
        g.session_id = session_id
        g.wizard_step = "configure"
        if sess.cleaned_df is None or sess.schema is None:
            abort(404)

        if request.method == "GET":
            return render_template(
                "configure.html",
                settings=sess.settings,
                segment_columns=sess.schema.segment_columns,
                segment_values=_segment_values(sess),
                business_units=list_column_values(sess.cleaned_df, sess.schema.business_unit_column),
                products=list_column_values(sess.cleaned_df, sess.schema.product_column),
                alert_levels=ALERT_LEVEL_CHOICES,
            )

        _validate_csrf_form(sess)
        sess.settings = _settings_from_form(request.form)
        sess.step = "events"
        sess.touch()
        return redirect(url_for("session_events", session_id=session_id))

    @app.route("/s/<session_id>/events", methods=["GET", "POST"])
    def session_events(session_id: str):
        sess = _load_session(session_id)
        g.session_id = session_id
        g.wizard_step = "events"
        if sess.cleaned_df is None or sess.schema is None:
            abort(404)

        if request.method == "GET":
            return render_template(
                "events.html",
                events=[_event_to_dict(e) for e in sess.events],
                event_text_batch=sess.event_text_batch,
            )

        _validate_csrf_form(sess)
        batch = (request.form.get("event_text_batch") or "").strip()
        sess.event_text_batch = batch
        action = request.form.get("action", "parse")

        seg_col = sess.settings.segment_column or (
            sess.schema.segment_columns[0] if sess.schema.segment_columns else None
        )
        segments = list_segment_values(sess.cleaned_df, seg_col) if seg_col else []
        bus = list_column_values(sess.cleaned_df, sess.schema.business_unit_column)
        prods = list_column_values(sess.cleaned_df, sess.schema.product_column)

        lines = [ln.strip() for ln in batch.splitlines() if ln.strip()]
        sess.events = parse_event_batch(lines, segment_candidates=segments, business_units=bus, products=prods)

        if action == "forecast":
            try:
                _run_forecast(sess)
            except ValueError as exc:
                flash(str(exc), "danger")
                return redirect(url_for("session_events", session_id=session_id))
            except Exception as exc:
                _log.exception("Forecast failed")
                flash("Forecast could not be completed. Check configuration and data.", "danger")
                return redirect(url_for("session_events", session_id=session_id))
            return redirect(url_for("session_dashboard", session_id=session_id))

        sess.step = "events"
        sess.touch()
        flash("Events parsed. Review approvals below.", "info")
        return redirect(url_for("session_events", session_id=session_id))

    @app.route("/s/<session_id>/events/update", methods=["POST"])
    def session_events_update(session_id: str):
        sess = _load_session(session_id)
        _validate_csrf_form(sess)
        count = int(request.form.get("event_count", 0) or 0)
        updated: List[ParsedEvent] = []
        for i in range(count):
            raw = request.form.get(f"raw_{i}", "")
            approved = request.form.get(f"approved_{i}") == "on"
            annotation = request.form.get(f"annotation_{i}") or None
            base = next((e for e in sess.events if e.raw_text == raw), None)
            if base is None:
                base = parse_event_text(raw)
            base.approved = approved
            base.annotation = annotation
            updated.append(base)
        sess.events = updated
        sess.touch()
        flash("Event approvals saved.", "success")
        return redirect(url_for("session_events", session_id=session_id))

    @app.route("/s/<session_id>/api/parse-events", methods=["POST"])
    def api_parse_events(session_id: str):
        sess = _load_session(session_id)
        _validate_csrf_header(sess)
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        seg_col = sess.settings.segment_column or (
            sess.schema.segment_columns[0] if sess.schema and sess.schema.segment_columns else None
        )
        segments = list_segment_values(sess.cleaned_df, seg_col) if sess.cleaned_df is not None and seg_col else []
        bus = list_column_values(sess.cleaned_df, sess.schema.business_unit_column) if sess.cleaned_df is not None else []
        prods = list_column_values(sess.cleaned_df, sess.schema.product_column) if sess.cleaned_df is not None else []
        events = parse_event_batch(
            [ln.strip() for ln in text.splitlines() if ln.strip()],
            segment_candidates=segments,
            business_units=bus,
            products=prods,
        )
        rows = "".join(
            f"<li><strong>{e.event_name or 'Event'}</strong>: {e.effect_type} {e.effect_value or ''} "
            f"({int(e.confidence * 100)}% conf.)</li>"
            for e in events
        )
        html = f"<ul class='mb-0'>{rows}</ul>" if rows else "<p class='text-muted mb-0'>No events parsed.</p>"
        return jsonify({"events": [_event_to_dict(e) for e in events], "html": html})

    @app.route("/s/<session_id>/dashboard")
    def session_dashboard(session_id: str):
        sess = _load_session(session_id)
        g.session_id = session_id
        g.wizard_step = "dashboard"
        if sess.result is None:
            flash("Generate a forecast first.", "warning")
            return redirect(url_for("session_events", session_id=session_id))

        ctx = _dashboard_context(sess)
        return render_template("dashboard.html", **ctx)

    @app.route("/s/<session_id>/export/rejected.csv")
    def export_rejected(session_id: str):
        sess = _load_session(session_id)
        if sess.raw_df is None or sess.validation_report is None:
            abort(404)
        idx = sess.validation_report.rejected_row_indexes
        if not idx:
            flash("No rejected rows to export.", "info")
            return redirect(url_for("session_validation", session_id=session_id))
        subset = sess.raw_df.loc[idx].copy()
        csv_str = dataframe_to_safe_csv(subset, config=_session_config(sess))
        bio = BytesIO(csv_str.encode("utf-8"))
        return send_file(bio, mimetype="text/csv", as_attachment=True, download_name="rejected_rows.csv")

    @app.route("/s/<session_id>/export/forecast.<fmt>")
    def export_forecast(session_id: str, fmt: str):
        sess = _load_session(session_id)
        if sess.result is None:
            abort(404)
        orch = ForecastOrchestrator(_session_config(sess))
        df = orch.export_dual_forecast_dataframe(sess.result)
        df = sanitize_dataframe_for_export(df, config=_session_config(sess))
        fmt = fmt.lower()
        if fmt == "csv":
            bio = BytesIO(dataframe_to_safe_csv(df, config=_session_config(sess)).encode("utf-8"))
            return send_file(bio, mimetype="text/csv", as_attachment=True, download_name="forecast.csv")
        if fmt == "json":
            payload = forecast_result_to_dict(sess.result)
            bio = BytesIO(json.dumps(payload, indent=2, default=str).encode("utf-8"))
            return send_file(bio, mimetype="application/json", as_attachment=True, download_name="forecast.json")
        if fmt == "xlsx":
            bio = BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="forecast")
                hist = orch.export_history_dataframe(sess.result)
                sanitize_dataframe_for_export(hist, _session_config(sess)).to_excel(
                    writer, index=False, sheet_name="history"
                )
            bio.seek(0)
            return send_file(
                bio,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                as_attachment=True,
                download_name="forecast.xlsx",
            )
        abort(404)

    @app.route("/s/<session_id>/export/audit.<fmt>")
    def export_audit(session_id: str, fmt: str):
        sess = _load_session(session_id)
        audit = _build_audit_payload(sess)
        fmt = fmt.lower()
        if fmt == "json":
            bio = BytesIO(json.dumps(audit, indent=2, default=str).encode("utf-8"))
            return send_file(bio, mimetype="application/json", as_attachment=True, download_name="audit.json")
        if fmt == "csv":
            flat = pd.DataFrame(
                [
                    {
                        "session_id": audit["session_id"],
                        "generated_at": audit["generated_at"],
                        "filename": audit.get("source_filename"),
                        "model": audit.get("model_kind"),
                        "validation_readiness": audit.get("validation_readiness"),
                        "events_approved": audit.get("events_approved_count"),
                    }
                ]
            )
            bio = BytesIO(dataframe_to_safe_csv(flat, config=_session_config(sess)).encode("utf-8"))
            return send_file(bio, mimetype="text/csv", as_attachment=True, download_name="audit_summary.csv")
        abort(404)


def _run_forecast(sess: AnalysisSession) -> None:
    cfg = _session_config(sess)
    orch = ForecastOrchestrator(cfg)
    assert sess.cleaned_df is not None and sess.schema is not None

    metrics: List[ForecastMetric] = [ForecastMetric.VOLUME]
    if sess.settings.include_liability and sess.schema.supports_liability:
        metrics.append(ForecastMetric.LIABILITY)

    seg_col = sess.settings.segment_column or None
    seg_val = sess.settings.segment_value or None
    bu = sess.settings.business_unit or None
    product = sess.settings.product or None

    result = orch.run_from_dataframe(
        sess.cleaned_df,
        schema=sess.schema,
        events=sess.events,
        segment_column=seg_col,
        segment_value=seg_val if seg_val else None,
        business_unit=bu,
        product=product,
        metrics=metrics,
        require_approved_events=True,
    )
    sess.result = result
    sess.segment_bundles = {}

    if sess.settings.run_segment_breakdown and seg_col:
        try:
            sess.segment_bundles = orch.run_segment_forecasts(
                sess.cleaned_df,
                segment_column=seg_col,
                schema=sess.schema,
                metric=ForecastMetric.VOLUME,
                events=sess.events,
                require_approved_events=True,
                max_segments=MAX_SEGMENTS_BREAKDOWN,
            )
        except ValueError:
            sess.segment_bundles = {}

    try:
        run = orch.metadata_store.create_run(source_filename=sess.original_filename, status="running")
        assert run.id is not None
        forecast_payload = {
            "primary_metric": result.primary_metric.value,
            "volume": _bundle_to_dict(result.volume),
            "liability": _bundle_to_dict(result.liability),
            "forecasts": [
                {
                    "period": f.period.isoformat(),
                    "baseline": f.baseline,
                    "adjusted": f.adjusted,
                    "lower": f.lower,
                    "upper": f.upper,
                    "event_adjustment": f.event_adjustment,
                    "metric": f.metric,
                }
                for f in result.forecasts
            ],
        }
        persist_forecast_run(
            orch.metadata_store,
            int(run.id),
            schema=_schema_to_dict(sess.schema),
            validation=validation_report_to_dict(sess.validation_report),
            model_kind=result.model_selection.model_kind.value,
            fallback_reason=result.model_selection.fallback_reason,
            metrics=result.model_selection.metrics.__dict__,
            forecast=forecast_payload,
            events=[_event_to_dict(e) for e in sess.events],
        )
        sess.analysis_run_id = int(run.id)
    except Exception as exc:
        _log.info("Persistence skipped or failed (non-fatal): %s", exc)

    sess.step = "dashboard"
    sess.touch()


def _dashboard_context(sess: AnalysisSession) -> Dict[str, Any]:
    result = sess.result
    assert result is not None
    settings = sess.settings
    currency = settings.currency_code or "USD"

    alerts = _build_alerts(sess, result)
    kpi_cards = _build_kpi_cards(sess, result, currency, alerts=alerts)

    charts = {
        "main": _chart_main(result),
        "baseline_adjusted": _chart_baseline_adjusted(result),
        "seasonality": _chart_seasonality(result),
        "liability": _chart_liability(result) if result.liability else None,
        "segments": _chart_segments(sess.segment_bundles) if sess.segment_bundles else None,
        "heatmap": _chart_heatmap(sess.segment_bundles) if sess.segment_bundles else None,
        "diagnostics": _chart_diagnostics(result),
    }

    rows: List[Dict[str, Any]] = []
    for f in result.forecasts:
        rows.append(
            {
                "period": f.period.strftime("%Y-%m"),
                "metric": f.metric,
                "baseline": f.baseline,
                "adjusted": f.adjusted,
                "event_adjustment": f.event_adjustment,
                "lower": f.lower,
                "upper": f.upper,
            }
        )
    if result.liability:
        for f in result.liability.forecasts:
            rows.append(
                {
                    "period": f.period.strftime("%Y-%m"),
                    "metric": "liability",
                    "baseline": f.baseline,
                    "adjusted": f.adjusted,
                    "event_adjustment": f.event_adjustment,
                    "lower": f.lower,
                    "upper": f.upper,
                }
            )

    ms = result.model_selection
    model_selection = {
        "model_kind": ms.model_kind.value,
        "validation_strategy": ms.validation_strategy.value,
        "fallback_reason": ms.fallback_reason,
        "metrics": ms.metrics.__dict__,
        "search_notes": ms.search_notes,
    }

    return {
        "kpi_cards": kpi_cards,
        "charts": charts,
        "forecast_rows": rows,
        "explanations": result.explanations.__dict__,
        "model_selection": model_selection,
        "alerts": alerts,
    }


def _fmt_kpi(value: float, metric: Optional[str]) -> str:
    if metric == "liability":
        return f"{value:,.0f}"
    return f"{value:,.0f}"


def _parse_optional_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return float(text)


def _normalize_alert_level(raw: Optional[str]) -> str:
    level = (raw or "medium").strip().lower()
    return level if level in ALERT_LEVEL_CHOICES else "medium"


def _alert_bootstrap_level(settings: SessionSettings, level_field: str) -> str:
    specific = (getattr(settings, level_field, "") or "").strip()
    level = _normalize_alert_level(specific or settings.global_alert_level)
    return ALERT_LEVEL_TO_BOOTSTRAP[level]


def _model_quality_label(mape: Optional[float], settings: SessionSettings) -> tuple[str, str, str]:
    """Returns (label, hint, tone css suffix)."""
    if mape is None:
        return (
            "Insufficient data",
            "Validation MAPE not available from holdout",
            "insufficient",
        )
    if mape <= settings.mape_strong_threshold:
        return (
            "Strong",
            f"MAPE {mape:.1f}% ≤ strong bound {settings.mape_strong_threshold}%",
            "strong",
        )
    if mape <= settings.mape_acceptable_threshold:
        return (
            "Acceptable",
            f"MAPE {mape:.1f}% ≤ acceptable bound {settings.mape_acceptable_threshold}%",
            "acceptable",
        )
    return (
        "Weak",
        f"MAPE {mape:.1f}% > acceptable bound {settings.mape_acceptable_threshold}%",
        "weak",
    )


def _volume_forecast_points(result: ForecastRunResult) -> List[Any]:
    points = [f for f in result.forecasts if (f.metric or ForecastMetric.VOLUME.value) == ForecastMetric.VOLUME.value]
    return points if points else list(result.forecasts)


def _forecast_risk_label(threshold_alerts: List[Dict[str, str]]) -> tuple[str, str]:
    if not threshold_alerts:
        return "Minimal", "No configured threshold crossings"
    danger = sum(1 for a in threshold_alerts if a.get("level") == "danger")
    warning = sum(1 for a in threshold_alerts if a.get("level") == "warning")
    if danger >= 2:
        return "Critical", f"{danger} high-severity threshold alert(s)"
    if danger == 1:
        return "Elevated", "One high-severity threshold alert"
    if warning:
        return "Moderate", f"{warning} medium-severity threshold alert(s)"
    return "Low", f"{len(threshold_alerts)} informational threshold alert(s)"


def _highest_segment(sess: AnalysisSession) -> Optional[tuple[str, float]]:
    if not sess.segment_bundles:
        return None
    best_name = ""
    best_total = float("-inf")
    for name, bundle in sess.segment_bundles.items():
        total = sum(f.adjusted for f in bundle.forecasts)
        if total > best_total:
            best_total = total
            best_name = name
    if best_name and best_total > float("-inf"):
        return best_name, best_total
    return None


def _build_kpi_cards(
    sess: AnalysisSession,
    result: ForecastRunResult,
    currency: str,
    alerts: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    settings = sess.settings
    cards: List[Dict[str, Any]] = []
    vr = sess.validation_report

    hist = result.monthly_history
    if hist and hist.values:
        cards.append(
            {
                "label": "Latest month volume",
                "value": _fmt_kpi(float(hist.values[-1]), "volume"),
                "hint": hist.period_index[-1].strftime("%Y-%m"),
            }
        )
        cards.append(
            {
                "label": "Historical volume total",
                "value": _fmt_kpi(float(sum(hist.values)), "volume"),
                "hint": f"{len(hist.values)} months in model history",
            }
        )
    elif vr is not None and vr.total_volume is not None:
        cards.append(
            {
                "label": "Historical volume total",
                "value": _fmt_kpi(float(vr.total_volume), "volume"),
                "hint": "From validation summary",
            }
        )

    vol_fc = _volume_forecast_points(result)
    if vol_fc:
        n = min(3, len(vol_fc))
        cards.append(
            {
                "label": f"Expected volume ({n}-mo, {currency})",
                "value": _fmt_kpi(float(sum(f.adjusted for f in vol_fc[:n])), "volume"),
                "hint": "Sum of adjusted forecast",
            }
        )

    if result.liability:
        lhist = result.liability.monthly_history
        if lhist and lhist.values:
            cards.append(
                {
                    "label": "Latest month liability",
                    "value": _fmt_kpi(float(lhist.values[-1]), "liability"),
                    "hint": lhist.period_index[-1].strftime("%Y-%m"),
                }
            )
            cards.append(
                {
                    "label": "Historical liability total",
                    "value": _fmt_kpi(float(sum(lhist.values)), "liability"),
                    "hint": f"{len(lhist.values)} months in model history",
                }
            )
        elif vr is not None and vr.total_liability is not None:
            cards.append(
                {
                    "label": "Historical liability total",
                    "value": _fmt_kpi(float(vr.total_liability), "liability"),
                    "hint": "From validation summary",
                }
            )
        lfc = result.liability.forecasts
        if lfc:
            n = min(3, len(lfc))
            cards.append(
                {
                    "label": f"Expected liability ({n}-mo, {currency})",
                    "value": _fmt_kpi(float(sum(f.adjusted for f in lfc[:n])), "liability"),
                    "hint": "Sum of adjusted forecast",
                }
            )

    quality, qhint, qtone = _model_quality_label(result.kpis.validation_mape, settings)
    cards.append({"label": "Model quality", "value": quality, "hint": qhint, "tone": qtone})

    approved = sum(1 for e in sess.events if e.approved)
    if sess.events or approved:
        cards.append(
            {
                "label": "Approved events",
                "value": str(approved),
                "hint": f"{len(sess.events)} parsed" if sess.events else "None parsed",
            }
        )

    threshold_alerts, _info = _split_alerts(alerts if alerts is not None else _build_alerts(sess, result))
    risk, rhint = _forecast_risk_label(threshold_alerts)
    cards.append(
        {
            "label": "Forecast risk",
            "value": risk,
            "hint": rhint,
            "tone": "weak" if risk in {"Critical", "Elevated"} else ("acceptable" if risk == "Moderate" else "strong"),
        }
    )

    top_seg = _highest_segment(sess)
    if top_seg:
        name, total = top_seg
        cards.append(
            {
                "label": "Highest segment (horizon adj.)",
                "value": name,
                "hint": _fmt_kpi(total, "volume"),
            }
        )

    if result.kpis.validation_mape is not None:
        cards.append(
            {
                "label": "Validation MAPE",
                "value": f"{result.kpis.validation_mape:.1f}%",
                "hint": f"Alert if > {settings.mape_alert_threshold}%",
            }
        )

    return cards


def _split_alerts(alerts: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    threshold: List[Dict[str, str]] = []
    info: List[Dict[str, str]] = []
    for a in alerts:
        if a.get("kind") == "threshold":
            threshold.append(a)
        else:
            info.append(a)
    return threshold, info


def _append_alert(
    alerts: List[Dict[str, str]],
    *,
    kind: str,
    level: str,
    message: str,
) -> None:
    alerts.append({"kind": kind, "level": level, "message": message})


def _chart_main(result: ForecastRunResult) -> Dict[str, Any]:
    hist = result.monthly_history
    hx = [p.strftime("%Y-%m") for p in hist.period_index]
    hy = list(hist.values)
    fx = [f.period.strftime("%Y-%m") for f in result.forecasts]
    fb = [f.baseline for f in result.forecasts]
    fa = [f.adjusted for f in result.forecasts]
    lo = [f.lower for f in result.forecasts]
    up = [f.upper for f in result.forecasts]

    traces = [
        {"type": "scatter", "mode": "lines+markers", "name": "History", "x": hx, "y": hy, "line": {"color": "#0a2540"}},
        {"type": "scatter", "mode": "lines+markers", "name": "Baseline FC", "x": fx, "y": fb, "line": {"dash": "dot", "color": "#1e5a8a"}},
        {"type": "scatter", "mode": "lines+markers", "name": "Adjusted FC", "x": fx, "y": fa, "line": {"color": "#4da3d9"}},
    ]
    if any(v is not None for v in lo) and any(v is not None for v in up):
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Upper band",
                "x": fx,
                "y": up,
                "line": {"width": 0},
                "showlegend": False,
            }
        )
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Confidence band",
                "x": fx,
                "y": lo,
                "fill": "tonexty",
                "fillcolor": "rgba(77,163,217,0.2)",
                "line": {"width": 0},
            }
        )
    return {"data": traces, "layout": {"title": "Historical actuals & forecast", "xaxis": {"title": "Month"}, "yaxis": {"title": "Value"}}}


def _chart_baseline_adjusted(result: ForecastRunResult) -> Dict[str, Any]:
    fx = [f.period.strftime("%Y-%m") for f in result.forecasts]
    return {
        "data": [
            {"type": "bar", "name": "Baseline", "x": fx, "y": [f.baseline for f in result.forecasts], "marker": {"color": "#1e5a8a"}},
            {"type": "bar", "name": "Adjusted", "x": fx, "y": [f.adjusted for f in result.forecasts], "marker": {"color": "#4da3d9"}},
        ],
        "layout": {"barmode": "group", "title": "Baseline vs adjusted"},
    }


def _chart_seasonality(result: ForecastRunResult) -> Optional[Dict[str, Any]]:
    if not result.segment_comparisons:
        return None
    comp = result.segment_comparisons[0]
    stats = comp.same_month_stats
    if not stats or not stats.historical_values:
        return None
    labels = [p.strftime("%Y-%m") for p in stats.historical_periods]
    return {
        "data": [
            {
                "type": "bar",
                "name": f"Month {stats.calendar_month}",
                "x": labels,
                "y": stats.historical_values,
                "marker": {"color": "#0a2540"},
            }
        ],
        "layout": {"title": "Same calendar month — prior years"},
    }


def _chart_liability(result: ForecastRunResult) -> Optional[Dict[str, Any]]:
    bundle = result.liability
    if not bundle:
        return None
    fake = ForecastRunResult(
        forecasts=bundle.forecasts,
        model_selection=bundle.model_selection,
        segment_comparisons=bundle.segment_comparisons,
        explanations=bundle.explanations,
        kpis=bundle.kpis,
        monthly_history=bundle.monthly_history,
        events_applied=result.events_applied,
        primary_metric=ForecastMetric.LIABILITY,
    )
    return _chart_main(fake)


def _chart_segments(bundles: Dict[str, MetricForecastBundle]) -> Optional[Dict[str, Any]]:
    if not bundles:
        return None
    labels = list(bundles.keys())
    totals = [sum(f.adjusted for f in b.forecasts) for b in bundles.values()]
    return {
        "data": [{"type": "bar", "x": labels, "y": totals, "marker": {"color": "#1e5a8a"}}],
        "layout": {"title": "Adjusted horizon total by segment", "xaxis": {"tickangle": -35}},
    }


def _chart_heatmap(bundles: Dict[str, MetricForecastBundle]) -> Optional[Dict[str, Any]]:
    if not bundles:
        return None
    segments = list(bundles.keys())
    periods = [f.period.strftime("%Y-%m") for f in next(iter(bundles.values())).forecasts]
    z = []
    for seg in segments:
        z.append([f.adjusted for f in bundles[seg].forecasts])
    return {
        "data": [
            {
                "type": "heatmap",
                "x": periods,
                "y": segments,
                "z": z,
                "colorscale": "Blues",
            }
        ],
        "layout": {"title": "Adjusted forecast by segment & month"},
    }


def _chart_diagnostics(result: ForecastRunResult) -> Optional[Dict[str, Any]]:
    ms = result.model_selection
    metrics = ms.metrics
    names = []
    values = []
    for label, val in [("MAPE", metrics.mape), ("sMAPE", metrics.smape), ("RMSE", metrics.rmse), ("MAE", metrics.mae)]:
        if val is not None:
            names.append(label)
            values.append(val)
    if not names:
        return None
    return {
        "data": [{"type": "bar", "x": names, "y": values, "marker": {"color": "#4da3d9"}}],
        "layout": {"title": f"Validation metrics ({ms.validation_strategy.value})"},
    }


def _build_alerts(sess: AnalysisSession, result: ForecastRunResult) -> List[Dict[str, str]]:
    alerts: List[Dict[str, str]] = []
    settings = sess.settings
    vol_fc = _volume_forecast_points(result)

    if settings.monthly_volume_alert_threshold is not None and vol_fc:
        peak = max(f.adjusted for f in vol_fc)
        if peak > settings.monthly_volume_alert_threshold:
            worst = max(vol_fc, key=lambda f: f.adjusted)
            _append_alert(
                alerts,
                kind="threshold",
                level=_alert_bootstrap_level(settings, "monthly_volume_alert_level"),
                message=(
                    f"Monthly volume forecast {worst.adjusted:,.0f} ({worst.period.strftime('%Y-%m')}) "
                    f"exceeds alert threshold {settings.monthly_volume_alert_threshold:,.0f}."
                ),
            )

    if (
        settings.monthly_liability_alert_threshold is not None
        and result.liability
        and result.liability.forecasts
    ):
        lfc = result.liability.forecasts
        peak = max(f.adjusted for f in lfc)
        if peak > settings.monthly_liability_alert_threshold:
            worst = max(lfc, key=lambda f: f.adjusted)
            _append_alert(
                alerts,
                kind="threshold",
                level=_alert_bootstrap_level(settings, "monthly_liability_alert_level"),
                message=(
                    f"Monthly liability forecast {worst.adjusted:,.0f} ({worst.period.strftime('%Y-%m')}) "
                    f"exceeds alert threshold {settings.monthly_liability_alert_threshold:,.0f}."
                ),
            )

    if settings.percent_growth_alert_threshold is not None and vol_fc:
        hist = result.monthly_history
        prev_val = float(hist.values[-1]) if hist and hist.values else None
        for fp in vol_fc:
            if prev_val is None or abs(prev_val) < 1e-12:
                prev_val = fp.baseline
                continue
            growth = (fp.adjusted - prev_val) / abs(prev_val) * 100.0
            if abs(growth) > settings.percent_growth_alert_threshold:
                _append_alert(
                    alerts,
                    kind="threshold",
                    level=_alert_bootstrap_level(settings, "percent_growth_alert_level"),
                    message=(
                        f"Volume change {growth:+.1f}% into {fp.period.strftime('%Y-%m')} "
                        f"exceeds ±{settings.percent_growth_alert_threshold}% alert bound."
                    ),
                )
                break
            prev_val = fp.adjusted

    if settings.ci_width_alert_threshold_pct is not None:
        for fp in vol_fc:
            if fp.lower is None or fp.upper is None:
                continue
            base = abs(fp.adjusted) if abs(fp.adjusted) > 1e-12 else abs(fp.baseline)
            if base < 1e-12:
                continue
            width_pct = (fp.upper - fp.lower) / base * 100.0
            if width_pct > settings.ci_width_alert_threshold_pct:
                _append_alert(
                    alerts,
                    kind="threshold",
                    level=_alert_bootstrap_level(settings, "ci_width_alert_level"),
                    message=(
                        f"Confidence interval width {width_pct:.1f}% for {fp.period.strftime('%Y-%m')} "
                        f"exceeds {settings.ci_width_alert_threshold_pct}% alert threshold."
                    ),
                )
                break

    if result.kpis.validation_mape is not None and result.kpis.validation_mape > settings.mape_alert_threshold:
        _append_alert(
            alerts,
            kind="threshold",
            level=_alert_bootstrap_level(settings, "mape_alert_level"),
            message=(
                f"Validation MAPE {result.kpis.validation_mape:.1f}% exceeds threshold "
                f"{settings.mape_alert_threshold}%."
            ),
        )

    if sess.validation_report and sess.validation_report.warnings:
        _append_alert(
            alerts,
            kind="info",
            level="warning",
            message=f"{len(sess.validation_report.warnings)} validation warning(s) — review data quality.",
        )
    if result.kpis.fallback_used:
        _append_alert(
            alerts,
            kind="info",
            level="warning",
            message=f"Fallback model in use ({result.kpis.model_kind.value}). {result.kpis.fallback_reason or ''}".strip(),
        )
    unapproved = [e for e in sess.events if not e.approved and e.raw_text]
    if unapproved:
        _append_alert(
            alerts,
            kind="info",
            level="info",
            message=f"{len(unapproved)} event(s) not approved — excluded from adjustments.",
        )
    if sess.validation_report and sess.validation_report.missing_months:
        _append_alert(
            alerts,
            kind="info",
            level="info",
            message=(
                f"Missing months in history: {', '.join(sess.validation_report.missing_months[:6])}"
                + ("…" if len(sess.validation_report.missing_months) > 6 else "")
            ),
        )
    return alerts


def _build_audit_payload(sess: AnalysisSession) -> Dict[str, Any]:
    result = sess.result
    payload: Dict[str, Any] = {
        "session_id": sess.session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_filename": sess.original_filename,
        "analysis_run_id": sess.analysis_run_id,
        "settings": sess.settings.__dict__,
        "schema": _schema_to_dict(sess.schema) if sess.schema else None,
        "validation": validation_report_to_dict(sess.validation_report) if sess.validation_report else None,
        "events": [_event_to_dict(e) for e in sess.events],
        "events_approved_count": sum(1 for e in sess.events if e.approved),
    }
    if result:
        payload["forecast"] = forecast_result_to_dict(result)
        payload["model_kind"] = result.model_selection.model_kind.value
    if sess.validation_report:
        payload["validation_readiness"] = sess.validation_report.readiness
    return payload


def _session_config(sess: AnalysisSession) -> AppConfig:
    base = AppConfig.from_env()
    s = sess.settings
    fc = replace(
        base.forecast,
        horizon_months=s.horizon_months,
        history_months=s.history_months,
        confidence_level=s.confidence_level,
    )
    val = replace(
        base.validation,
        holdout_months=s.holdout_months,
        min_train_months=s.min_train_months,
        rolling_folds=s.rolling_folds,
    )
    return replace(base, forecast=fc, validation=val)


def _settings_from_form(form) -> SessionSettings:
    return SessionSettings(
        history_months=int(form.get("history_months", 36)),
        horizon_months=int(form.get("horizon_months", 3)),
        confidence_level=float(form.get("confidence_level", 0.95)),
        holdout_months=int(form.get("holdout_months", 3)),
        min_train_months=int(form.get("min_train_months", 12)),
        currency_code=(form.get("currency_code") or "USD").strip()[:8],
        mape_alert_threshold=float(form.get("mape_alert_threshold", 25)),
        mape_strong_threshold=float(form.get("mape_strong_threshold", 15)),
        mape_acceptable_threshold=float(form.get("mape_acceptable_threshold", 25)),
        global_alert_level=_normalize_alert_level(form.get("global_alert_level")),
        mape_alert_level=(form.get("mape_alert_level") or "").strip().lower(),
        monthly_volume_alert_threshold=_parse_optional_float(form.get("monthly_volume_alert_threshold")),
        monthly_volume_alert_level=(form.get("monthly_volume_alert_level") or "").strip().lower(),
        monthly_liability_alert_threshold=_parse_optional_float(form.get("monthly_liability_alert_threshold")),
        monthly_liability_alert_level=(form.get("monthly_liability_alert_level") or "").strip().lower(),
        percent_growth_alert_threshold=_parse_optional_float(form.get("percent_growth_alert_threshold")),
        percent_growth_alert_level=(form.get("percent_growth_alert_level") or "").strip().lower(),
        ci_width_alert_threshold_pct=_parse_optional_float(form.get("ci_width_alert_threshold_pct")),
        ci_width_alert_level=(form.get("ci_width_alert_level") or "").strip().lower(),
        rolling_folds=int(form.get("rolling_folds", 3)),
        segment_column=(form.get("segment_column") or "").strip(),
        segment_value=(form.get("segment_value") or "").strip(),
        business_unit=(form.get("business_unit") or "").strip(),
        product=(form.get("product") or "").strip(),
        include_liability=form.get("include_liability") == "on",
        run_segment_breakdown=form.get("run_segment_breakdown") == "on",
    )


FIELD_LABELS: Dict[str, str] = {
    "date_column": "Date column",
    "amount_column": "Amount / liability column",
    "count_column": "Count column",
    "case_id_column": "Case ID column",
    "business_unit_column": "Business unit column",
    "product_column": "Product column",
    "category_column": "Category column",
    "status_column": "Status column",
    "reason_code_column": "Reason code column",
}


def _overrides_from_form(form, columns: List[str]) -> SchemaMappingOverride:
    cols = set(columns)
    labels = {v.casefold() for v in FIELD_LABELS.values()}

    def pick(name: str) -> Optional[str]:
        if name not in form:
            # Field absent from the submission: leave the detected mapping untouched.
            return None
        val = (form.get(name) or "").strip()
        if not val or val.casefold() in labels:
            # Explicit "none" (or a label submitted by autofill) clears the mapping,
            # so a wrong auto-detected column can actually be removed by the user.
            return ""
        if val not in cols:
            field = FIELD_LABELS.get(name, name)
            raise ValueError(
                f"'{val}' is not a column in the uploaded file. "
                f"Please choose a valid option for {field}."
            )
        return val

    date_choice = pick("date_column")
    if date_choice == "":
        raise ValueError("A date column is required. Please choose one.")

    return SchemaMappingOverride(
        date_column=date_choice,
        amount_column=pick("amount_column"),
        count_column=pick("count_column"),
        case_id_column=pick("case_id_column"),
        business_unit_column=pick("business_unit_column"),
        product_column=pick("product_column"),
        category_column=pick("category_column"),
        status_column=pick("status_column"),
        reason_code_column=pick("reason_code_column"),
        volume_aggregation_mode=(form.get("volume_aggregation_mode") or None),
    )


def _segment_values(sess: AnalysisSession) -> List[str]:
    if sess.cleaned_df is None:
        return []
    col = sess.settings.segment_column or (
        sess.schema.segment_columns[0] if sess.schema and sess.schema.segment_columns else None
    )
    if not col:
        return []
    return list_segment_values(sess.cleaned_df, col)


def _validate_csrf_form(sess: Optional[AnalysisSession]) -> None:
    token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
    if sess:
        expected = sess.csrf_token
    else:
        expected = flask_session.get("upload_csrf") or _session_csrf_from_cookie()
    if not token or not expected or not secrets.compare_digest(token, expected):
        abort(403)


def _validate_csrf_header(sess: AnalysisSession) -> None:
    token = request.headers.get("X-CSRF-Token")
    if not token or not secrets.compare_digest(token, sess.csrf_token):
        abort(403)


def _session_csrf(session_id: Optional[str]) -> str:
    if not session_id:
        return ""
    sess = _sessions.get(session_id)
    return sess.csrf_token if sess else ""


def _session_csrf_from_cookie() -> str:
    sid = flask_session.get("cb_sid")
    if not sid:
        return ""
    return _session_csrf(sid)


def _store_session(sess: AnalysisSession) -> None:
    with _sessions_lock:
        _sessions[sess.session_id] = sess


def _load_session(session_id: str) -> AnalysisSession:
    with _sessions_lock:
        sess = _sessions.get(session_id)
    if sess is None:
        abort(404)
    sess.touch()
    return sess


def _purge_expired_sessions() -> None:
    now = time.time()
    with _sessions_lock:
        expired = [sid for sid, s in _sessions.items() if now - s.created_at > SESSION_TTL_SECONDS]
        for sid in expired:
            sess = _sessions.pop(sid, None)
            if sess:
                for p in sess.temp_paths:
                    try:
                        p.unlink(missing_ok=True)
                    except OSError:
                        pass


def list_routes(app: Flask) -> List[str]:
    rules = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint != "static":
            rules.append(f"{','.join(sorted(rule.methods - {'HEAD', 'OPTIONS'}))} {rule.rule}")
    return sorted(rules)


app = create_app()

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host=os.getenv("FLASK_HOST", "127.0.0.1"), port=int(os.getenv("FLASK_PORT", "5000")), debug=debug)
