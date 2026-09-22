"""Shared fixtures: sample CSV, fast SARIMAX config, Flask test client."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AppConfig, SarimaxSearchConfig, get_config  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def sample_csv_path(project_root: Path) -> Path:
    path = project_root / "sample_data" / "sample_chargebacks.csv"
    assert path.is_file(), f"Missing sample CSV at {path}"
    return path


@pytest.fixture(scope="session")
def sample_df(sample_csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(sample_csv_path)


@pytest.fixture
def reset_config_cache():
    if hasattr(get_config, "_instance"):
        delattr(get_config, "_instance")
    yield
    if hasattr(get_config, "_instance"):
        delattr(get_config, "_instance")


@pytest.fixture
def fast_config(reset_config_cache, tmp_path, monkeypatch) -> AppConfig:
    """Small SARIMAX grid and isolated SQLite path for speed."""
    monkeypatch.setenv("FLASK_SECRET_KEY", "test-secret-key-for-pytest-only")
    monkeypatch.setenv("CHARGEBACK_DATABASE_PATH", str(tmp_path / "test_analysis.db"))
    base = AppConfig.from_env()
    sarimax = SarimaxSearchConfig(
        p_range=(0, 1),
        d_range=(0,),
        q_range=(0, 1),
        P_range=(0, 1),
        D_range=(0,),
        Q_range=(0,),
        max_candidates=6,
        fit_timeout_seconds=30.0,
    )
    cfg = replace(base, sarimax=sarimax, database_path=tmp_path / "test_analysis.db")
    get_config._instance = cfg  # type: ignore[attr-defined]
    return cfg


@pytest.fixture
def flask_app(fast_config, monkeypatch):
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setenv("FLASK_SECRET_KEY", "test-secret-key-for-pytest-only")
    import app as app_module

    application = app_module.create_app()
    application.config["TESTING"] = True
    application.config["WTF_CSRF_ENABLED"] = False
    return application


@pytest.fixture
def client(flask_app):
    return flask_app.test_client()


def upload_sample_csv(client, sample_csv_path: Path):
    """POST upload wizard step; returns (session_id, csrf_token)."""
    with sample_csv_path.open("rb") as fh:
        data = fh.read()
    with client.session_transaction() as sess:
        sess["upload_csrf"] = "upload-csrf-token"
    resp = client.post(
        "/upload",
        data={
            "csrf_token": "upload-csrf-token",
            "csv_file": (BytesIO(data), "sample_chargebacks.csv"),
        },
        content_type="multipart/form-data",
        follow_redirects=False,
    )
    assert resp.status_code == 302
    location = resp.headers["Location"]
    parts = location.rstrip("/").split("/")
    session_id = parts[2] if len(parts) >= 3 and parts[1] == "s" else parts[-2]
    import app as app_module

    with app_module._sessions_lock:
        analysis = app_module._sessions[session_id]
    return session_id, analysis.csrf_token


@pytest.fixture
def uploaded_session(client, sample_csv_path):
    return upload_sample_csv(client, sample_csv_path)

