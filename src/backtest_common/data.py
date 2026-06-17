#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data loading and tick aggregation helpers shared by backtest scripts."""

from __future__ import annotations

from typing import Any

import pandas as pd


def load_ohlcv_data(symbol: str, start_date: str, end_date: str, timeframe: str = "5m") -> pd.DataFrame:
    """Load OHLCV candles through the existing OKXDataLoader and validate columns."""
    from src.data_feed.okx_loader import OKXDataLoader

    loader = OKXDataLoader(symbol=symbol, timeframe=timeframe)
    df = loader.fetch_data_by_date_range(start_date, end_date)
    if df.empty:
        raise RuntimeError(f"No data loaded for {symbol} {timeframe} {start_date} -> {end_date}")
    df = df.sort_index().copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close", "volume"])


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    elif "ts_ms" in out.columns:
        ts = pd.to_datetime(pd.to_numeric(out["ts_ms"], errors="coerce"), unit="ms", utc=True, errors="coerce")
    else:
        raise RuntimeError("tick chunk missing timestamp/ts_ms")
    out = out.loc[ts.notna()].copy()
    out.index = ts.loc[ts.notna()]
    return out.sort_index()


def aggregate_trades_to_seconds(chunk: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    if chunk.empty:
        return pd.DataFrame()
    df = _ensure_utc_index(chunk)
    for col in ["price", "size"]:
        if col not in df.columns:
            raise RuntimeError(f"tick chunk missing column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "side" not in df.columns:
        raise RuntimeError("tick chunk missing column: side")

    df = df.dropna(subset=["price", "size"])
    if df.empty:
        return pd.DataFrame()

    side = df["side"].astype(str).str.lower()
    notional = df["price"] * df["size"] * cfg.contract_value
    df["buy_notional"] = notional.where(side == "buy", 0.0)
    df["sell_notional"] = notional.where(side == "sell", 0.0)
    df["buy_contracts"] = df["size"].where(side == "buy", 0.0)
    df["sell_contracts"] = df["size"].where(side == "sell", 0.0)
    df["notional"] = notional
    df["trades_count"] = 1

    rule = f"{int(cfg.bar_seconds)}s"
    bars = df.resample(rule).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        buy_notional=("buy_notional", "sum"),
        sell_notional=("sell_notional", "sum"),
        buy_contracts=("buy_contracts", "sum"),
        sell_contracts=("sell_contracts", "sum"),
        volume_notional=("notional", "sum"),
        trades_count=("trades_count", "sum"),
    )
    return bars.dropna(subset=["open", "high", "low", "close"])


def merge_second_bars(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts).sort_index()
    merged = raw.groupby(level=0).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        buy_notional=("buy_notional", "sum"),
        sell_notional=("sell_notional", "sum"),
        buy_contracts=("buy_contracts", "sum"),
        sell_contracts=("sell_contracts", "sum"),
        volume_notional=("volume_notional", "sum"),
        trades_count=("trades_count", "sum"),
    )
    if merged.empty:
        return merged
    full_index = pd.date_range(merged.index.min(), merged.index.max(), freq="1s", tz="UTC")
    merged = merged.reindex(full_index)
    merged["close"] = merged["close"].ffill()
    for col in ["open", "high", "low"]:
        merged[col] = merged[col].fillna(merged["close"])
    for col in ["buy_notional", "sell_notional", "buy_contracts", "sell_contracts", "volume_notional", "trades_count"]:
        merged[col] = merged[col].fillna(0.0)
    merged["cvd_notional"] = merged["buy_notional"] - merged["sell_notional"]
    merged["signed_contracts"] = merged["buy_contracts"] - merged["sell_contracts"]
    return merged.dropna(subset=["open", "high", "low", "close"])


def load_second_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    cfg: Any,
    *,
    chunksize: int,
    trades_url_template: str,
    data_dir: str | None,
) -> pd.DataFrame:
    from src.data_feed.okx_tick_loader import OKXTickLoader

    loader = OKXTickLoader(symbol=symbol, data_dir=data_dir, trades_url_template=trades_url_template)
    parts: list[pd.DataFrame] = []
    for day, chunk in loader.iter_trades(start_date, end_date, chunksize=chunksize):
        bars = aggregate_trades_to_seconds(chunk, cfg)
        if not bars.empty:
            parts.append(bars)
        print(f"loaded tick chunk day={day} rows={len(chunk)} bars={len(bars)}")
        del chunk, bars
    out = merge_second_bars(parts)
    if out.empty:
        raise RuntimeError(f"No tick data loaded for {symbol} {start_date} -> {end_date}")
    return out
