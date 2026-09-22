"""Safe CSV loading with size and row limits."""

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import BinaryIO, TextIO, Union

import pandas as pd

from config import AppConfig, get_config

FileLike = Union[str, Path, BytesIO, StringIO, BinaryIO, TextIO]


def read_csv_limited(
    source: FileLike,
    config: AppConfig | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Load CSV from path or file-like object with configured limits.

    Extra kwargs are forwarded to pandas.read_csv (except nrows, which is capped).
    """
    cfg = config or get_config()
    kwargs = dict(read_csv_kwargs)
    nrows = kwargs.pop("nrows", cfg.csv.max_rows_sample)
    kwargs["nrows"] = min(int(nrows), cfg.csv.max_rows_sample)

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.stat().st_size > cfg.csv.max_file_bytes:
            raise ValueError(
                f"CSV exceeds max_file_bytes ({cfg.csv.max_file_bytes})"
            )
        return pd.read_csv(path, **kwargs)

    size = _file_like_size(source)
    if size is not None and size > cfg.csv.max_file_bytes:
        raise ValueError(f"CSV exceeds max_file_bytes ({cfg.csv.max_file_bytes})")

    return pd.read_csv(source, **kwargs)


def _file_like_size(source: FileLike) -> int | None:
    """Best-effort size for in-memory uploads without consuming the stream."""
    getbuffer = getattr(source, "getbuffer", None)
    if callable(getbuffer):
        try:
            return int(len(getbuffer()))
        except (TypeError, ValueError):
            return None
    getvalue = getattr(source, "getvalue", None)
    if callable(getvalue):
        try:
            value = getvalue()
            return len(value) if isinstance(value, (bytes, str)) else None
        except (TypeError, ValueError):
            return None
    return None
