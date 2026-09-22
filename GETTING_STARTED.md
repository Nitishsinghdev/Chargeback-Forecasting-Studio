# Getting Started — Chargeback Seasonal Trend & Forecasting Studio

This guide shows how to install, start, stop, restart, and run the first analysis on **Windows**.

The application runs **only on your computer**. Chargeback files are not sent to an external business portal or AI API. The browser may still download Bootstrap, Font Awesome, and Plotly from public CDNs for the user interface.

---

## 1. Prerequisites

- Windows 10 or 11
- **Python 3.11 or 3.12** installed  
  Check in PowerShell:

  ```powershell
  python --version
  ```

- A web browser (Edge or Chrome)
- Optional: at least **4 GB free RAM** for large CSVs (100 MB+ files need more)

If `python` is not recognized, install Python from [python.org](https://www.python.org/downloads/) and tick **Add python.exe to PATH**.

---

## 2. Open the project folder

Use the **full path**. Do not run these commands from `C:\WINDOWS\system32`.

```powershell
cd "C:\Users\nitising\OneDrive - AMDOCS\Desktop\Cursor Projects\Forecasting Studio\chargeback_forecasting_tool"
```

Confirm you are in the right place:

```powershell
dir app.py, requirements.txt
```

You should see both files.

---

## 3. One-time setup (first run only)

Create a virtual environment and install libraries:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activate script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Optional: copy environment defaults:

```powershell
copy .env.example .env
```

The app reads many settings from the process environment. A `.env` file is a convenient place to keep `FLASK_SECRET_KEY`; this codebase does not auto-load `.env` unless you set those variables in the shell. For local use, defaults work without a `.env` file.

Upload limit defaults:

- Maximum file size: **250 MB**
- Maximum rows loaded: **10 million**

---

## 4. Start the tool

Every time you use the app:

```powershell
cd "C:\Users\nitising\OneDrive - AMDOCS\Desktop\Cursor Projects\Forecasting Studio\chargeback_forecasting_tool"
.\.venv\Scripts\Activate.ps1
python app.py
```

Wait until you see:

```text
 * Running on http://127.0.0.1:5000
```

**Leave this PowerShell window open.** Closing it or pressing Ctrl+C stops the server.

Open a browser to:

**http://127.0.0.1:5000**

Use `http://`, not `https://`.

---

## 5. Stop the tool

In the PowerShell window that is running `python app.py`, press **Ctrl+C**.

---

## 6. Restart the tool

1. Press **Ctrl+C** in the running window (or close it).
2. If port 5000 is still in use (old process), free it:

   ```powershell
   Get-NetTCPConnection -LocalPort 5000 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
   ```

3. Start again with `python app.py` from the project folder (step 4).

If you change `config.py` or upload limits, you **must restart**. An old process on port 5000 will keep the previous settings (for example the old 100 MB limit).

---

## 7. First analysis (recommended path)

1. Click **Upload CSV**.
2. Choose a file:
   - Sample (synthetic): `sample_data\sample_chargebacks.csv`
   - Your chargeback extract: `.csv` only, up to 250 MB
3. **Schema:** confirm Date, Amount, Case ID, Business Unit, Product. Correct any wrong mapping. Empty optional fields clear a guessed column.
4. **Validation:** review valid vs rejected rows, date range, missing months. Download rejected rows if needed. Continue only if the report is ready (or you accept warnings).
5. **Configure** — suggested values for an Oct–Dec forecast:

   | Setting | Recommended value |
   |---------|-------------------|
   | History months | 36 |
   | Forecast horizon | 3 |
   | Confidence level | 0.95 |
   | Holdout months | 3 |
   | Min training months | 12 |
   | Rolling validation folds | 3 |
   | Primary segment column | Leave blank for total portfolio |
   | Include liability | On, if an amount column exists |

   For **October–December**, the last month in the CSV must be **September** of that year. If Oct–Dec actuals are already in the file, the tool will forecast the *next* three months after the last date (often Jan–Mar).

6. **Events:** optional. Type a business factor in plain language, review the parsed fields, and **approve** before it can change the forecast. No number means annotation only.
7. Generate forecast and open the **Dashboard**.

Large files (100 MB+) can take several minutes on upload, validation, and SARIMAX training. Do not close the browser tab.

---

## 8. What “good” model quality looks like

On the dashboard **Model selection** card:

- **Model:** `sarimax` and **Fallback:** `None` means the primary model ran.
- **MAPE** around **10% or lower** is typically labeled **Strong** (default threshold).
- **RMSE** is in the same units as the series (counts for volume, currency for liability).

---

## 9. Troubleshooting

| Problem | What to do |
|---------|------------|
| `Cannot find path ...\chargeback_forecasting_tool` | You started PowerShell in the wrong folder. Use the full `cd` path in step 2. |
| Browser: connection failed | The server is not running. Start `python app.py` and wait for the “Running on” line. |
| `Address already in use` / old size limit still applies | Another `python app.py` is still on port 5000. Stop it (step 6) then start again. |
| File too large (105 MB) after raising the limit | Restart so Flask picks up 250 MB. Confirm `Running on http://127.0.0.1:5000` in a **new** process. |
| HTTPS / “secure connection failed” | Use **http://127.0.0.1:5000** |
| Upload hangs | Normal for large CSVs. Wait; check RAM. |
| Charts missing / unstyled page | CDNs blocked (offline or firewall). Analysis still runs locally; UI libraries may not load. |
| `python` not found | Install Python 3.12 and reopen PowerShell. |
| Port 5000 blocked | `$env:FLASK_PORT="5050"; python app.py` then open http://127.0.0.1:5050 |

---

## 10. Privacy

- Uploaded rows stay in **server memory** for the session (about 4 hours, or until restart).
- SQLite `data\analysis.db` stores **run metadata** (filename, mappings, metrics, approved events), not a full copy of the CSV unless you export.
- Browser cookies store only a session id, not the dataset.

---

## 11. More detail

See [README.md](README.md) for schema mapping, SARIMAX, fallbacks, security, tests, and environment variables.
