"""
Deterministic synthetic chargeback-level CSV for demos and tests.

Run from project root:
    python sample_data/generate_sample_chargebacks.py
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "sample_chargebacks.csv"

BUSINESS_UNITS = ("Enterprise", "Consumer", "Wholesale")
PRODUCTS = {
    "Enterprise": ("Cloud Services", "Managed Support", "Platform API"),
    "Consumer": ("Mobile Plans", "Streaming Add-on", "Device Protection"),
    "Wholesale": ("Bulk Voice", "Partner Resale", "IoT SIM"),
}
REASON_CODES = ("4837", "4855", "4863", "4870", "4871", "7030")
STATUSES = ("Open", "Pending", "Won", "Lost")

# 48 months: Jan 2022 – Dec 2025
START = datetime(2022, 1, 1)
N_MONTHS = 48


def _month_start(offset: int) -> datetime:
    year = START.year + (START.month - 1 + offset) // 12
    month = (START.month - 1 + offset) % 12 + 1
    return datetime(year, month, 1)


def _seasonality(month: int) -> float:
    """Calendar-month multipliers (retail / holiday chargeback pattern)."""
    table = {
        1: 0.88,
        2: 0.90,
        3: 0.95,
        4: 0.98,
        5: 1.00,
        6: 1.02,
        7: 1.05,
        8: 1.03,
        9: 1.00,
        10: 1.08,
        11: 1.15,
        12: 1.28,
    }
    return table[month]


def _bu_trend(bu: str, month_index: int) -> float:
    drift = {
        "Enterprise": 1.0 + month_index * 0.004,
        "Consumer": 1.0 + month_index * 0.002,
        "Wholesale": 1.0 - month_index * 0.001,
    }
    return drift[bu]


def _product_weight(product: str) -> float:
    weights = {
        "Cloud Services": 1.35,
        "Managed Support": 1.05,
        "Platform API": 0.85,
        "Mobile Plans": 1.20,
        "Streaming Add-on": 0.75,
        "Device Protection": 0.95,
        "Bulk Voice": 1.10,
        "Partner Resale": 0.90,
        "IoT SIM": 0.70,
    }
    return weights.get(product, 1.0)


def _daily_count(month_index: int, bu: str, product: str, rng: random.Random) -> int:
    mstart = _month_start(month_index)
    base = 18 * _seasonality(mstart.month) * _bu_trend(bu, month_index) * _product_weight(product)
    # Mild pseudo-random jitter, deterministic via rng
    jitter = rng.uniform(0.82, 1.18)
    return max(3, int(round(base * jitter)))


def _amount(bu: str, reason: str, rng: random.Random) -> float:
    bu_base = {"Enterprise": 420.0, "Consumer": 185.0, "Wholesale": 310.0}[bu]
    reason_mult = {"4837": 1.15, "4855": 0.95, "4863": 1.05, "4870": 1.25, "4871": 0.88, "7030": 1.40}.get(
        reason, 1.0
    )
    raw = bu_base * reason_mult * rng.uniform(0.55, 1.65)
    return round(raw, 2)


def generate_rows() -> list[dict[str, str]]:
    rng = random.Random(20260321)
    rows: list[dict[str, str]] = []
    cb_seq = 1_000_001

    for month_index in range(N_MONTHS):
        mstart = _month_start(month_index)
        if month_index < N_MONTHS - 1:
            days_in_month = (_month_start(month_index + 1) - timedelta(days=1)).day
        else:
            days_in_month = 31

        for bu in BUSINESS_UNITS:
            for product in PRODUCTS[bu]:
                n = _daily_count(month_index, bu, product, rng)
                for _ in range(n):
                    day = rng.randint(1, min(days_in_month, 28 if mstart.month == 2 else days_in_month))
                    charge_date = datetime(mstart.year, mstart.month, day)
                    reason = rng.choice(REASON_CODES)
                    status = rng.choices(STATUSES, weights=[12, 18, 35, 35], k=1)[0]
                    rows.append(
                        {
                            "Chargeback Date": charge_date.strftime("%Y-%m-%d"),
                            "Chargeback ID": f"CB-{cb_seq}",
                            "Amount": f"{_amount(bu, reason, rng):.2f}",
                            "Business Unit": bu,
                            "Product": product,
                            "Reason Code": reason,
                            "Status": status,
                        }
                    )
                    cb_seq += 1

    rows.sort(key=lambda r: (r["Chargeback Date"], r["Chargeback ID"]))
    return rows


def main() -> None:
    rows = generate_rows()
    fieldnames = [
        "Chargeback Date",
        "Chargeback ID",
        "Amount",
        "Business Unit",
        "Product",
        "Reason Code",
        "Status",
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
