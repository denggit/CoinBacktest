#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified causal OHLCV loading for R03.3.3.

2020-2021 use the public ordinary OKX 1m K-line loader. 2022 onward use the
existing public 1m Trade Bar loader, restricted to the common OHLCV contract.
No missing Trade fields are fabricated for the early period.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_feed.okx_loader import OKXDataLoader
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader

from .config import MarketStateContinuityConfig


COMMON_COLUMNS = ("open", "high", "low", "close", "volume")


def _normalise_ohlcv(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*COMMON_COLUMNS, "source"])
    out = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce"))
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    out.index = index
    out = out.loc[~out.index.isna()]
    out = out.loc[~out.index.duplicated(keep="last")].sort_index(kind="stable")
    missing = sorted(set(COMMON_COLUMNS) - set(out.columns))
    if missing:
        raise RuntimeError(f"{source} loader missing common OHLCV columns: {missing}")
    out = out.loc[:, list(COMMON_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])
    out["volume"] = out["volume"].fillna(0.0)
    out["source"] = source
    out.index.name = "timestamp"
    return out


@dataclass
class UnifiedOHLCVLoader:
    config: MarketStateContinuityConfig
    data_dir: str | Path | None = None

    def __post_init__(self) -> None:
        self.kline_loader = OKXDataLoader(
            symbol=self.config.symbol,
            timeframe=self.config.source_timeframe,
            db_dir=self.data_dir,
        )
        self.trade_loader = OKXTradeBarLoader(
            symbol=self.config.symbol,
            timeframe=self.config.source_timeframe,
            data_dir=self.data_dir,
            align_with_okx_loader_timezone=True,
        )

    def _load_ordinary(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if start > end:
            return pd.DataFrame(columns=[*COMMON_COLUMNS, "source"])
        frame = self.kline_loader.fetch_data_by_date_range(start, end)
        return _normalise_ohlcv(frame, "ordinary_kline")

    def _load_trade_bar(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if start > end:
            return pd.DataFrame(columns=[*COMMON_COLUMNS, "source"])
        frame = self.trade_loader.fetch_data_by_date_range(
            start,
            end,
            build_missing=False,
            force_rebuild=False,
            cvd_mode="range",
        )
        return _normalise_ohlcv(frame, "trade_bar")

    def fetch_data_by_date_range(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
        begin = pd.Timestamp(start)
        finish = pd.Timestamp(end)
        if begin > finish:
            return pd.DataFrame(columns=[*COMMON_COLUMNS, "source"])
        ordinary_end = pd.Timestamp(self.config.ordinary_kline_end)
        trade_start = pd.Timestamp(self.config.trade_bar_start)
        parts: list[pd.DataFrame] = []
        if begin <= ordinary_end:
            parts.append(self._load_ordinary(begin, min(finish, ordinary_end)))
        if finish >= trade_start:
            parts.append(self._load_trade_bar(max(begin, trade_start), finish))
        nonempty = [part for part in parts if not part.empty]
        if not nonempty:
            return pd.DataFrame(columns=[*COMMON_COLUMNS, "source"])
        out = pd.concat(nonempty, axis=0)
        out = out.loc[~out.index.duplicated(keep="last")].sort_index(kind="stable")
        return out


@dataclass(frozen=True)
class StateDataPreflight:
    status: str
    samples: tuple[dict[str, object], ...]
    boundary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {"status": self.status, "samples": list(self.samples), "boundary": self.boundary}


def run_state_data_preflight(
    loader: UnifiedOHLCVLoader,
    config: MarketStateContinuityConfig,
) -> StateDataPreflight:
    sample_dates = ("2020-06-15", "2021-06-15", "2023-06-15", "2025-06-15")
    rows: list[dict[str, object]] = []
    status = "PASS"
    for raw in sample_dates:
        start = pd.Timestamp(raw)
        end = start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        bars = loader.fetch_data_by_date_range(start, end)
        expected_source = "ordinary_kline" if start.year <= 2021 else "trade_bar"
        sources = sorted(str(item) for item in bars.get("source", pd.Series(dtype=str)).dropna().unique())
        numeric = bars[list(COMMON_COLUMNS)].to_numpy(dtype=float, copy=False) if not bars.empty else np.empty((0, 5))
        valid = (
            not bars.empty
            and len(bars) >= 1_350
            and bars.index.is_monotonic_increasing
            and bars.index.is_unique
            and np.isfinite(numeric).all()
            and sources == [expected_source]
        )
        if not valid:
            status = "BLOCKED"
        rows.append(
            {
                "date": str(start.date()),
                "rows": int(len(bars)),
                "expected_source": expected_source,
                "sources": sources,
                "monotonic": bool(bars.index.is_monotonic_increasing) if not bars.empty else False,
                "unique": bool(bars.index.is_unique) if not bars.empty else False,
                "valid": bool(valid),
            }
        )

    boundary_start = pd.Timestamp("2021-12-31 00:00:00")
    boundary_end = pd.Timestamp("2022-01-02 23:59:59")
    boundary_frame = loader.fetch_data_by_date_range(boundary_start, boundary_end)
    ordinary = boundary_frame.loc[boundary_frame["source"] == "ordinary_kline"]
    trade = boundary_frame.loc[boundary_frame["source"] == "trade_bar"]
    price_gap = np.nan
    volume_ratio = np.nan
    if not ordinary.empty and not trade.empty:
        price_gap = float(trade["open"].iloc[0] / ordinary["close"].iloc[-1] - 1.0)
        ordinary_volume = float(ordinary["volume"].tail(720).median())
        trade_volume = float(trade["volume"].head(720).median())
        if ordinary_volume > 0:
            volume_ratio = trade_volume / ordinary_volume
    boundary = {
        "ordinary_rows": int(len(ordinary)),
        "trade_rows": int(len(trade)),
        "price_gap_pct": price_gap,
        "volume_median_ratio_trade_over_kline": volume_ratio,
        "note": "volume is used through rolling relative features; a source-boundary ratio is reported rather than silently rescaled",
    }
    if ordinary.empty or trade.empty or not np.isfinite(price_gap) or abs(price_gap) > 0.03:
        status = "BLOCKED"
    return StateDataPreflight(status=status, samples=tuple(rows), boundary=boundary)
