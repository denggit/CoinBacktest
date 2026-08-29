#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local Binance perpetual funding archive loader.

This module is intentionally local-only.  It normalizes archived Binance
funding CSVs into the same naive UTC+offset clock used by the CoinBacktest
trade-bar loaders.  Strategy/backtest code should consume this interface rather
than parsing research CSVs directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class BinanceFundingArchiveLoader:
    """Read a local Binance funding-rate CSV without network access.

    Expected columns are compatible with Binance ``/fapi/v1/fundingRate``
    archives, including ``fundingTime`` and ``fundingRate``.  ``markPrice`` is
    optional.  Returned index is local-naive time using ``timezone_offset_hours``.
    """

    def __init__(self, csv_path: str | Path, *, timezone_offset_hours: float = 8.0) -> None:
        self.csv_path = Path(csv_path)
        self.timezone_offset_hours = float(timezone_offset_hours)

    def load(self, start: Any, end: Any) -> pd.DataFrame:
        if not self.csv_path.exists():
            return pd.DataFrame(columns=["funding_rate", "mark_price", "source"])

        frame = pd.read_csv(self.csv_path)
        if frame.empty:
            return pd.DataFrame(columns=["funding_rate", "mark_price", "source"])
        required = {"fundingTime", "fundingRate"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"funding archive missing required columns: {sorted(missing)}")

        # Binance archive timestamps are UTC.  utc=True also handles strings
        # that already carry +00:00 and epoch-like strings consistently.
        ts = pd.to_datetime(frame["fundingTime"], utc=True, errors="coerce", format="mixed")
        local = (ts.dt.tz_convert(None) + pd.Timedelta(hours=self.timezone_offset_hours)).dt.round("s")
        out = pd.DataFrame(index=pd.DatetimeIndex(local, name="timestamp"))
        out["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="coerce").to_numpy()
        if "markPrice" in frame.columns:
            out["mark_price"] = pd.to_numeric(frame["markPrice"], errors="coerce").to_numpy()
        else:
            out["mark_price"] = float("nan")
        out["source"] = "BINANCE_ETHUSDT_PROXY"
        out = out[~out.index.isna()].sort_index()
        out = out[~out.index.duplicated(keep="last")]
        out = out.dropna(subset=["funding_rate"])

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tzinfo is not None:
            start_ts = start_ts.tz_convert(None)
        if end_ts.tzinfo is not None:
            end_ts = end_ts.tz_convert(None)
        return out.loc[(out.index >= start_ts) & (out.index <= end_ts)].copy()

    def coverage(self) -> dict[str, object]:
        frame = self.load(pd.Timestamp("1970-01-01"), pd.Timestamp("2100-01-01"))
        return {
            "rows": int(len(frame)),
            "start": frame.index.min() if not frame.empty else None,
            "end": frame.index.max() if not frame.empty else None,
            "source": "BINANCE_ETHUSDT_PROXY",
            "path": str(self.csv_path),
        }
