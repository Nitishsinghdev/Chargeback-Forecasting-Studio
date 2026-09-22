"""Flask wizard workflow, CSRF, and export routes."""

from __future__ import annotations

import json

from conftest import upload_sample_csv


def test_index_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_upload_rejects_non_csv(client):
    with client.session_transaction() as sess:
        sess["upload_csrf"] = "tok"
    resp = client.post(
        "/upload",
        data={"csrf_token": "tok", "csv_file": (b"not csv", "notes.txt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 302


def test_upload_bad_csrf(client, sample_csv_path):
    with sample_csv_path.open("rb") as fh:
        data = fh.read()
    with client.session_transaction() as sess:
        sess["upload_csrf"] = "expected"
    resp = client.post(
        "/upload",
        data={
            "csrf_token": "wrong",
            "csv_file": (data, "sample_chargebacks.csv"),
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 403


def test_wizard_schema_to_validation(client, uploaded_session):
    session_id, csrf = uploaded_session
    resp = client.get(f"/s/{session_id}/schema")
    assert resp.status_code == 200
    resp = client.post(
        f"/s/{session_id}/schema",
        data={
            "csrf_token": csrf,
            "date_column": "Chargeback Date",
            "amount_column": "Amount",
            "case_id_column": "Chargeback ID",
            "business_unit_column": "Business Unit",
            "product_column": "Product",
            "reason_code_column": "Reason Code",
            "status_column": "Status",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"validation" in resp.data.lower() or b"Validation" in resp.data


def _confirm_schema(client, session_id: str, csrf: str) -> None:
    client.post(
        f"/s/{session_id}/schema",
        data={
            "csrf_token": csrf,
            "date_column": "Chargeback Date",
            "amount_column": "Amount",
            "case_id_column": "Chargeback ID",
            "business_unit_column": "Business Unit",
            "product_column": "Product",
        },
        follow_redirects=True,
    )


def test_api_parse_events_requires_csrf(client, uploaded_session):
    session_id, csrf = uploaded_session
    _confirm_schema(client, session_id, csrf)
    resp = client.post(
        f"/s/{session_id}/api/parse-events",
        json={"text": "Promo: +5% volume in 2025-07"},
        headers={"X-CSRF-Token": "bad"},
    )
    assert resp.status_code == 403
    resp = client.post(
        f"/s/{session_id}/api/parse-events",
        json={"text": "Promo: +5% volume in 2025-07"},
        headers={"X-CSRF-Token": csrf},
    )
    assert resp.status_code == 200
    payload = json.loads(resp.data)
    assert "events" in payload


def test_export_forecast_csv_after_run(client, uploaded_session, sample_csv_path, fast_config):
    session_id, csrf = uploaded_session
    client.post(
        f"/s/{session_id}/schema",
        data={
            "csrf_token": csrf,
            "date_column": "Chargeback Date",
            "amount_column": "Amount",
            "case_id_column": "Chargeback ID",
        },
        follow_redirects=True,
    )
    client.post(
        f"/s/{session_id}/configure",
        data={
            "csrf_token": csrf,
            "history_months": "36",
            "horizon_months": "3",
            "include_liability": "on",
        },
        follow_redirects=True,
    )
    client.post(
        f"/s/{session_id}/events",
        data={
            "csrf_token": csrf,
            "event_text_batch": "",
            "action": "forecast",
        },
        follow_redirects=True,
    )
    resp = client.get(f"/s/{session_id}/export/forecast.csv")
    assert resp.status_code == 200
    assert b"period" in resp.data


def test_unknown_session_404(client):
    assert client.get("/s/00000000-0000-0000-0000-000000000000/schema").status_code == 404


def test_configure_requires_csrf(client, uploaded_session):
    session_id, csrf = uploaded_session
    _confirm_schema(client, session_id, csrf)
    resp = client.post(
        f"/s/{session_id}/configure",
        data={"csrf_token": "invalid", "history_months": "24"},
    )
    assert resp.status_code == 403
