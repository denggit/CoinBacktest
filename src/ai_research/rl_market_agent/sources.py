#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local-only source adapter for the clean-sheet RL market-agent dataset.

This layer deliberately delegates all market-data access to ``src.data_feed``.
It never downloads or rebuilds missing history; R00 is an audit/dataset job, not
an ingestion job.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_feed.okx_loader import OKXDataLoader
from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader


class SourceRepository:
    def __init__(self, *, symbol: str, data_dir: str | Path | None = None) -> None:
        self.symbol = symbol
        self.data_dir = None if data_dir is None else Path(data_dir)
        self._default_data_dir = Path(__file__).resolve().parents[3] / "data"
        self._kline_cache: dict[str, pd.DataFrame] = {}

    @property
    def data_root(self) -> Path:
        return self._default_data_dir if self.data_dir is None else self.data_dir

    def _require_db(self, name: str) -> None:
        path = self.data_root / name
        if not path.exists():
            raise FileNotFoundError(f"required local data cache does not exist: {path}")

    @staticmethod
    def _slice(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=getattr(frame, "columns", None))
        out = frame.copy()
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce"))
        out = out.loc[~out.index.isna()]
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out.loc[(out.index >= start) & (out.index <= end)]

    def load_kline(self, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # OKXDataLoader currently exposes an all-local-table read. HTF tables are
        # small enough to cache once in-process and slice per shard.
        if timeframe not in self._kline_cache:
            self._require_db("crypto_history.db")
            loader = OKXDataLoader(
                symbol=self.symbol,
                timeframe=timeframe,
                db_dir=None if self.data_dir is None else str(self.data_dir),
            )
            self._kline_cache[timeframe] = loader.load_local_data()
        return self._slice(self._kline_cache[timeframe], start, end)

    def load_trade_bars(self, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        self._require_db("okx_trade_bars.db")
        loader = OKXTradeBarLoader(
            symbol=self.symbol,
            timeframe=timeframe,
            data_dir=None if self.data_dir is None else self.data_dir,
        )
        return loader.fetch_data_by_date_range(start, end, build_missing=False, cvd_mode="range")

    def load_range_bars(
        self,
        range_pct: float,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        self._require_db("okx_range_bars.db")
        loader = OKXRangeBarLoader(
            symbol=self.symbol,
            range_pct=range_pct,
            data_dir=None if self.data_dir is None else self.data_dir,
            initialize_db=False,
        )
        return loader.load_local_data(start_date=start, end_date=end, columns=columns)

    def load_footprint(
        self,
        range_pct: float,
        price_step: float,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        self._require_db("okx_range_footprints.db")
        loader = OKXRangeFootprintLoader(
            symbol=self.symbol,
            range_pct=range_pct,
            price_step=price_step,
            data_dir=None if self.data_dir is None else self.data_dir,
        )
        return loader.load_local_data(start_date=start, end_date=end, columns=columns)
