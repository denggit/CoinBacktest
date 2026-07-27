#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared models and timestamp helpers for Binance futures metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from config.loader import TIMEZONE
except ImportError:  # pragma: no cover
    TIMEZONE = "+8"

BINANCE_VISION_BASE_URL = "https://data.binance.vision"
METRICS_PERIOD = "5m"
EXPECTED_ROWS_PER_DAY = 288
METRIC_COLUMNS: tuple[str, ...] = (
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)
DEFAULT_RELATIVE_WINDOWS: tuple[str, ...] = ("5m", "15m", "30m", "1h", "4h", "1d")


def timezone_offset() -> pd.Timedelta:
    value = str(TIMEZONE).strip()
    if value.startswith("+"):
        return pd.Timedelta(hours=float(value[1:] or 0))
    if value.startswith("-"):
        return -pd.Timedelta(hours=float(value[1:] or 0))
    return pd.Timedelta(0)


def normalize_symbol(symbol: str) -> str:
    text = str(symbol).strip().upper().replace("_", "-")
    if text.endswith("-SWAP"):
        parts = text.split("-")
        if len(parts) >= 2:
            return f"{parts[0]}{parts[1]}"
    return text.replace("-", "")


def parse_archive_day(value: Any) -> date:
    return pd.Timestamp(value).date()


def iter_days(start: Any, end: Any) -> list[date]:
    start_day = parse_archive_day(start)
    end_day = parse_archive_day(end)
    if end_day < start_day:
        raise ValueError("end_date must be greater than or equal to start_date")
    count = (end_day - start_day).days + 1
    return [start_day + timedelta(days=i) for i in range(count)]


def parse_timedelta(value: str | pd.Timedelta) -> pd.Timedelta:
    delta = pd.Timedelta(value)
    if delta < pd.Timedelta(0):
        raise ValueError("timedelta must be non-negative")
    return delta


def window_column_tag(value: str | pd.Timedelta) -> str:
    delta = pd.Timedelta(value)
    seconds = int(delta.total_seconds())
    if seconds % 86_400 == 0:
        return f"{seconds // 86_400}d"
    if seconds % 3_600 == 0:
        return f"{seconds // 3_600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


@dataclass(frozen=True)
class BinanceMetricsCoverage:
    symbol: str
    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    complete_days: int
    partial_days: int
    missing_days: int
    error_days: int


@dataclass(frozen=True)
class BinanceMetricsDayResult:
    day_utc: date
    status: str
    rows: int
    frame: pd.DataFrame | None
    source_url: str
    checksum_sha256: str = ""
    checksum_verified: bool = False
    error: str = ""
    used_local_archive: bool = False


@dataclass(frozen=True)
class BinanceMetricsDownloadSummary:
    symbol: str
    requested_days: int
    downloaded_days: int
    skipped_days: int
    partial_days: int
    missing_days: int
    error_days: int
    rows_written: int
    elapsed_seconds: float
    db_path: Path


class BinanceMetricsDownloadError(RuntimeError):
    """Raised when a Binance archive day cannot be fetched or validated."""
