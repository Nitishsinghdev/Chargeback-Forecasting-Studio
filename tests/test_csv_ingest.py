"""CSV ingestion limits."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import pytest

from dataclasses import replace

from config import CsvConfig
from services.csv_ingest import read_csv_limited


def test_default_upload_limit_is_250_mb():
    cfg = CsvConfig()
    assert cfg.max_file_bytes == 250 * 1024 * 1024
    assert cfg.max_rows_sample >= 10_000_000


def test_read_csv_respects_max_rows(fast_config):
    rows = "date,amount\n" + "\n".join(f"2024-01-{d:02d},1" for d in range(1, 29))
    cfg = replace(fast_config, csv=replace(fast_config.csv, max_rows_sample=10))
    df = read_csv_limited(StringIO(rows), config=cfg)
    assert len(df) == 10


def test_read_csv_file_too_large(tmp_path, fast_config):
    path = tmp_path / "big.csv"
    path.write_text("date,amount\n2024-01-01,1\n")
    cfg = replace(fast_config, csv=replace(fast_config.csv, max_file_bytes=5))
    with pytest.raises(ValueError, match="max_file_bytes"):
        read_csv_limited(path, config=cfg)
