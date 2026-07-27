#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CoinBacktest data adapters and replay loops for the martingale engine."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader
from src.data_feed.okx_tick_loader import OKXTickLoader
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.research_common.progress import ProgressReporter

from .engine import MartingaleEngine


def _clean_bar_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise RuntimeError("No bar data loaded for requested range")
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        if col not in out.columns:
            raise RuntimeError(f"bar data missing required column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if out.empty:
        raise RuntimeError("Bar data became empty after OHLC validation")
    return out


def load_bar_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.data_source == "trade_bar":
        loader = OKXTradeBarLoader(
            symbol=args.symbol,
            timeframe=args.trade_bar_timeframe,
            data_dir=args.data_dir,
            trades_url_template=args.trades_url_template,
        )
        df = loader.fetch_data_by_date_range(
            args.start_date,
            args.end_date,
            chunksize=args.chunksize,
            force_rebuild=bool(args.force_rebuild),
            build_missing=not bool(args.cache_only),
        )
        return _clean_bar_frame(df)

    if args.data_source == "range_bar":
        loader = OKXRangeBarLoader(
            symbol=args.symbol,
            range_pct=float(args.range_pct),
            data_dir=args.data_dir,
            trades_url_template=args.trades_url_template,
        )
        if args.cache_only:
            df = loader.load_local_data(args.start_date, args.end_date)
        else:
            df = loader.fetch_data_by_date_range(
                args.start_date,
                args.end_date,
                chunksize=args.chunksize,
                force_rebuild=bool(args.force_rebuild),
            )
        return _clean_bar_frame(df)

    raise ValueError(f"load_bar_data does not support {args.data_source!r}")


def run_bar_replay(
    bars: pd.DataFrame,
    engines: list[MartingaleEngine],
    *,
    source: str,
    progress_enabled: bool,
) -> None:
    opens = bars["open"].to_numpy(dtype="float64", copy=False)
    highs = bars["high"].to_numpy(dtype="float64", copy=False)
    lows = bars["low"].to_numpy(dtype="float64", copy=False)
    closes = bars["close"].to_numpy(dtype="float64", copy=False)
    timestamps = bars.index.to_numpy()
    total = len(bars)
    every = max(1, total // 100)
    progress = ProgressReporter(
        label=f"[backtest] {source}",
        total=total,
        every=every,
        enabled=progress_enabled,
    )

    previous_day: date | None = None
    previous_ts: Any = None
    previous_close: float | None = None
    for i in range(total):
        ts = pd.Timestamp(timestamps[i])
        current_day = ts.date()
        if previous_day is not None and current_day != previous_day and previous_close is not None:
            for engine in engines:
                engine.snapshot(previous_ts, previous_close)
        for engine in engines:
            engine.process_bar(
                ts,
                opens[i],
                highs[i],
                lows[i],
                closes[i],
                source=source,
            )
        previous_day = current_day
        previous_ts = ts
        previous_close = float(closes[i])
        progress.update(i + 1)

    if previous_ts is not None and previous_close is not None:
        for engine in engines:
            engine.snapshot(previous_ts, previous_close)
    progress.close()


def _date_range(start_date: str, end_date: str) -> Iterator[date]:
    current = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    while current <= end:
        yield current
        current += timedelta(days=1)


def iter_raw_trade_chunks(args: argparse.Namespace) -> Iterator[tuple[date, pd.DataFrame]]:
    loader = OKXTickLoader(
        symbol=args.symbol,
        data_dir=args.data_dir,
        trades_url_template=args.trades_url_template,
    )
    if not args.cache_only:
        yield from loader.iter_trades(
            args.start_date,
            args.end_date,
            chunksize=args.chunksize,
            trades_url_template=args.trades_url_template,
        )
        return

    for day in _date_range(args.start_date, args.end_date):
        found = loader.find_local_trade_file(day, template=args.trades_url_template)
        if found is None:
            raise RuntimeError(f"cache-only raw trade ZIP missing for UTC day {day}")
        for chunk in loader.read_zip(found, chunksize=args.chunksize):
            yield day, chunk


def run_raw_trade_replay(
    args: argparse.Namespace,
    engines: list[MartingaleEngine],
    *,
    progress_enabled: bool,
) -> None:
    total_days = (pd.Timestamp(args.end_date).date() - pd.Timestamp(args.start_date).date()).days + 1
    progress = ProgressReporter(
        label="[backtest] raw_trade days",
        total=total_days,
        every=1,
        enabled=progress_enabled,
    )
    completed_days = 0
    current_day: date | None = None
    last_ts: pd.Timestamp | None = None
    last_price: float | None = None

    for day, chunk in iter_raw_trade_chunks(args):
        if current_day is not None and day != current_day:
            if last_ts is not None and last_price is not None:
                for engine in engines:
                    engine.snapshot(last_ts, last_price)
            completed_days += 1
            progress.update(completed_days)
        current_day = day

        prices = pd.to_numeric(chunk["price"], errors="coerce").to_numpy(dtype="float64")
        timestamps = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce").to_numpy()
        valid = np.isfinite(prices) & (prices > 0) & ~pd.isna(timestamps)
        if not np.any(valid):
            continue
        prices = prices[valid]
        timestamps = timestamps[valid]
        for engine in engines:
            engine.process_tick_chunk(timestamps, prices)
        last_ts = pd.Timestamp(timestamps[-1])
        last_price = float(prices[-1])
        del chunk, prices, timestamps, valid

    if current_day is not None:
        if last_ts is not None and last_price is not None:
            for engine in engines:
                engine.snapshot(last_ts, last_price)
        completed_days += 1
        progress.update(min(completed_days, total_days))
    progress.close()


