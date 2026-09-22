# Chargeback Forecasting Studio

Executive-facing Flask application for ingesting chargeback CSVs, validating data quality, configuring forecast scenarios, parsing business events, and producing **volume** and **liability** forecasts with SARIMAX (and controlled fallbacks).

**Start here:** [GETTING_STARTED.md](GETTING_STARTED.md) — Windows install, start/stop/restart, first analysis, and troubleshooting.

> **Sample data:** `sample_data/sample_chargebacks.csv` is **fully synthetic**—generated for demos and automated tests. It does not contain real customer or financial records. Regenerate with `python sample_data/generate_sample_chargebacks.py`.

---

## Objective

Help finance and operations teams:

1. Upload row-level or pre-aggregated chargeback history.
2. Confirm automatic column mapping (with overrides).
3. Review validation readiness (gaps, duplicates, rejected rows).
4. Configure horizon, segments, and optional liability path.
5. Enter natural-language **events** (promotions, policy changes) with explicit approval.
6. View dashboard KPIs, same-month history comparisons, and export audit-safe artifacts.

---

## Stack

| Layer | Technology |
|--------|------------|
| UI | Flask 3, Jinja2 templates, Plotly (client-side) |
| Data | pandas, numpy |
| Models | statsmodels SARIMAX, scikit-learn fallbacks, Holt-Winters |
| Persistence | SQLite (`data/analysis.db`) |
| Tests | pytest |

---

## Installation and running

Step-by-step Windows instructions (full path, virtual environment, browser URL, restart, Oct–Dec settings) are in **[GETTING_STARTED.md](GETTING_STARTED.md)**.

Short version:

```powershell
cd "C:\Users\nitising\OneDrive - AMDOCS\Desktop\Cursor Projects\Forecasting Studio\chargeback_forecasting_tool"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000**. Leave the terminal open. Press Ctrl+C to stop.

Override host/port with `FLASK_HOST`, `FLASK_PORT`, `FLASK_DEBUG`.

Wizard flow: **Upload → Schema → Validation → Configure → Events → Dashboard**.

---

## Project layout

```
chargeback_forecasting_tool/
├── app.py                 # Flask routes, sessions, CSRF, exports
├── config.py              # Environment-driven AppConfig
├── models/                # Domain types, SQLite metadata
├── services/              # CSV, validation, monthly agg, SARIMAX, events
├── templates/             # Wizard + dashboard HTML
├── static/                # CSS/JS
├── sample_data/
│   ├── sample_chargebacks.csv      # Synthetic 48-month chargeback-level data
│   └── generate_sample_chargebacks.py
├── tests/                 # pytest suite
├── .env.example
├── GETTING_STARTED.md     # How to start, stop, and run the first analysis
└── README.md
```

---

## CSV mapping (dynamic schema)

The detector scores columns by **name similarity** and **parse/numeric ratios**:

| Role | Typical headers |
|------|-----------------|
| Date | `Chargeback Date`, `period`, `billing_date`, … |
| Amount (liability) | `Amount`, `chargeback`, `net_amount`, … |
| Case ID (unique volume) | `Chargeback ID`, `dispute_id`, … |
| Count (pre-aggregated volume) | `volume`, `case_count`, … |
| Business unit | `Business Unit`, `bu`, `division`, … |
| Product | `Product`, `sku`, … |
| Reason / category | `Reason Code`, `category`, … |
| Status | `Status`, `dispute_status`, … |

**Volume aggregation modes**

- `unique_case` — distinct case IDs per month (default when ID column exists).
- `row_count` — one row = one chargeback event.
- `sum_count` — sum a numeric count column (monthly extracts).

Users can override mappings on the **Schema** step; invalid column names raise a clear error.

**Clearing an inferred optional role:** In `SchemaMappingOverride`, an empty string (`""`) for an optional field (amount, count, case ID, business unit, product, status, reason code, category) clears that role and sets the column mapping to `None`. Omitted fields (`None`) leave the detected mapping unchanged. The date column is required and cannot be cleared this way.

---

## Validation & readiness

`validate_dataframe` produces a structured report:

- Rejects unparseable **dates** and **amounts** (when amount is mapped).
- Flags **duplicate keys** (date + case ID or date + amount) as warnings.
- Computes **missing months** in the observed range.
- Summarizes **total volume** (per aggregation mode) and **total liability**.
- **Readiness** labels: `not_ready`, `needs_review`, `ready` (considers errors, warnings, and `min_train_months`).

Rejected rows can be exported from the UI as `rejected_rows.csv`.

---

## SARIMAX & rolling validation

- Controlled grid over `(p,d,q)(P,D,Q,s)` with `s = CHARGEBACK_SEASONAL_PERIOD` (default 12).
- **Holdout** (`CHARGEBACK_HOLDOUT_MONTHS`, default 3): train on all but last *h* months, score on holdout.
- **Rolling folds** (`CHARGEBACK_ROLLING_FOLDS`): expanding window; training always ends **before** each test slice (no leakage).
- Best candidate by combined RMSE/MAE across holdout + folds.
- Default **forecast horizon**: 3 months (`CHARGEBACK_FORECAST_HORIZON`).
- **History window**: last 36 months by default (`CHARGEBACK_HISTORY_MONTHS`).

If history is shorter than `CHARGEBACK_MIN_TRAIN_MONTHS` or no SARIMAX candidate fits, a **fallback chain** runs (see below).

---

## Same-month comparisons

For each forecast month, the engine collects prior-year values for the **same calendar month** (up to three years) and builds segment comparisons (baseline vs adjusted vs last same-month actual).

---

## Events & approvals

Natural-language lines are parsed for:

- **Percent** or **absolute** effects (no number → no forecast adjustment).
- **Periods** (`Jan 2026`, `2025-07`, ranges `from … to …`).
- **Business unit** / **product** / segment hints matched against dataset vocabularies.

Only **approved** events with numeric magnitudes adjust the baseline (`require_approved_events=True` on forecast). Unapproved events appear in dashboard alerts but do not change numbers.

---

## Scenarios (Configure step)

| Setting | Purpose |
|---------|---------|
| `history_months` | Trim training window |
| `horizon_months` | Months to forecast (default 3) |
| `holdout_months` / `rolling_folds` | Validation depth |
| `segment_column` / `segment_value` | Filter series |
| `business_unit` / `product` | Additional filters |
| `include_liability` | Second metric path when amount exists |
| `run_segment_breakdown` | Per-segment volume mini-forecasts (cap 12) |
| `mape_alert_threshold` | Dashboard alert if validation MAPE exceeds threshold |

---

## Liability vs volume

- **Volume** — case count, row count, or summed counts (see aggregation mode).
- **Liability** — monthly sum of amount column; skipped when no amount column is mapped.

Exports include both metrics when available (`export_dual_forecast_dataframe`).

---

## Metrics & quality labels

Validation metrics (holdout / rolling): **MAPE**, **sMAPE**, **RMSE**, **MAE**, **bias**.

- MAPE skips zero actuals to avoid divide-by-zero.
- Combined metrics average across folds with equal weight.
- Default quality labels are **Strong** (MAPE ≤ 10%), **Acceptable** (≤ 25%),
  **Weak** (> 25%), and **Insufficient data** when validation cannot be scored.
- The Strong and Acceptable boundaries can be changed in the Configure screen's
  advanced panel for each analysis.

Dashboard shows model kind, fallback reason, validation MAPE, and the plain-language
quality label when holdout data exists. The same advanced panel configures monthly
volume, monthly liability, growth, and confidence-interval-width thresholds plus
their Low/Medium/High/Critical alert levels.

---

## Fallback models (order)

When SARIMAX is unavailable or search fails:

1. Seasonal naive (lag 12)
2. Moving average (`CHARGEBACK_MOVING_AVERAGE_WINDOW`)
3. Holt-Winters (additive)
4. Linear trend
5. Mean
6. Naive (last value)

---

## Security

- **CSRF** tokens on form posts and `X-CSRF-Token` for JSON event parse API.
- **Session TTL** (4 hours) with in-memory session store (not suitable for multi-worker production without shared store).
- **Upload limits**: extension `.csv`, default maximum **250 MB** (`CHARGEBACK_CSV_MAX_FILE_BYTES`). Large files stay in process memory, so plan for several times the file size in RAM.
- **`secure_filename`** on uploads.
- **Formula injection**: CSV exports prefix cells starting with `=`, `+`, `-`, `@`, `|`, tab, CR.
- Set **`FLASK_SECRET_KEY`** in production (`FLASK_ENV=production` requires it).

---

## Limitations

- In-memory sessions are lost on process restart and are not shared across workers.
- SARIMAX search is CPU-bound; large candidate grids can be slow (tune `CHARGEBACK_SARIMAX_MAX_CANDIDATES`).
- Event parsing is heuristic—not a substitute for structured scenario files.
- SQLite persistence is best-effort from the dashboard (failures are logged, non-fatal).

---

## Testing

```bash
pytest tests/ -q
```

The suite uses a **reduced SARIMAX grid** via fixtures (`tests/conftest.py`) to keep runtime reasonable (~1–2 minutes on a laptop). Coverage includes:

| Area | Test modules |
|------|----------------|
| Synthetic sample CSV | `test_sample_data.py` |
| Dynamic schema & overrides | `test_csv_schema.py` |
| Validation (invalid rows, duplicates, gaps) | `test_csv_validation.py` |
| Charge-level & pre-aggregated monthly agg | `test_monthly_processor.py` |
| CSV row/byte limits | `test_csv_ingest.py` |
| Event parsing & adjustments | `test_event_parser.py`, `test_event_features.py` |
| SARIMAX + 3-month horizon, holdout/rolling leakage guards | `test_sarimax_search.py` |
| Fallback chain order | `test_fallback_models.py` |
| Safe metrics (zero actuals) | `test_metrics.py` |
| Export formula injection | `test_export_safety.py` |
| Orchestrator dual metrics & exports | `test_forecast_service.py` |
| Flask wizard, CSRF, exports | `test_flask_app.py` |

Regression tests cover schema role ordering (case ID vs numeric reason codes), event percent/date/BU parsing, and duplicate-key warnings via override mapping (including clearing a mistaken count column with `count_column=""`).

---


## Environment variables

See [`.env.example`](.env.example) for the full list. Common entries:

| Variable | Default | Meaning |
|----------|---------|---------|
| `FLASK_SECRET_KEY` | (ephemeral in dev) | Session signing |
| `CHARGEBACK_FORECAST_HORIZON` | 3 | Months ahead |
| `CHARGEBACK_HISTORY_MONTHS` | 36 | Training window cap |
| `CHARGEBACK_HOLDOUT_MONTHS` | 3 | Holdout size |
| `CHARGEBACK_ROLLING_FOLDS` | 3 | Rolling validation folds |
| `CHARGEBACK_MIN_TRAIN_MONTHS` | 12 | Minimum history for “ready” / SARIMAX |
| `CHARGEBACK_SARIMAX_MAX_CANDIDATES` | 48 | Grid cap |
| `CHARGEBACK_EXPORT_FORMULA_PREFIX` | `'` | CSV injection prefix |
| `CHARGEBACK_CSV_MAX_FILE_BYTES` | `262144000` (250 MB) | Upload size limit |
| `CHARGEBACK_CSV_MAX_ROWS_SAMPLE` | `10000000` | Maximum rows loaded from a CSV |

---

## Sample CSV (synthetic)

- **Path:** `sample_data/sample_chargebacks.csv`
- **Span:** 48 months (Jan 2022 – Dec 2025), **8,000+** chargeback rows
- **Columns:** Chargeback Date, Chargeback ID, Amount, Business Unit, Product, Reason Code, Status
- **Patterns:** Holiday seasonality (Dec peak), gentle BU trends, product mix differences

Regenerate deterministically:

```bash
python sample_data/generate_sample_chargebacks.py
```
