#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cache-only rich OKX trade-bar loading for Flow-Impact research."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.research_common.flow_impact import regularize_trade_bar_axis


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    text = str(timeframe).strip()
    if len(text) < 2 or not text[:-1].isdigit():
        raise ValueError(f"invalid timeframe: {timeframe!r}")
    amount = int(text[:-1])
    unit = text[-1].lower()
    aliases = {"s": "s", "m": "min", "h": "h", "d": "D"}
    if unit not in aliases or amount <= 0:
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    return pd.Timedelta(amount, unit=aliases[unit])


def inclusive_end(value: str | pd.Timestamp, bar_delta: pd.Timedelta) -> pd.Timestamp:
    if isinstance(value, str) and len(value.strip()) == 10:
        return pd.Timestamp(value) + pd.Timedelta(days=1) - bar_delta
    return pd.Timestamp(value)


def load_rich_trade_bars(
    *,
    project_root: Path,
    symbol: str,
    timeframe: str,
    warmup_start_date: str,
    end_date: str,
    data_dir: str | Path | None = None,
    db_name: str = "okx_trade_bars.db",
) -> pd.DataFrame:
    """Load only existing trade-bar cache; never download or rebuild data."""
    bar_delta = timeframe_delta(timeframe)
    resolved_data_dir = Path(data_dir) if data_dir else Path(project_root) / "data"
    db_path = resolved_data_dir / db_name
    if not db_path.exists():
        raise FileNotFoundError(
            f"Local trade-bar DB not found: {db_path}. This research is cache-only and will not download/build missing data."
        )
    loader = OKXTradeBarLoader(
        symbol=symbol,
        timeframe=timeframe,
        data_dir=resolved_data_dir,
        db_name=db_name,
    )
    load_end = inclusive_end(end_date, bar_delta)
    print(
        f"[load] local OKX trade bars {warmup_start_date} -> {load_end} timeframe={timeframe}",
        flush=True,
    )
    bars = loader.fetch_data_by_date_range(
        warmup_start_date,
        load_end,
        build_missing=False,
        force_rebuild=False,
        cvd_mode="range",
    )
    if bars.empty:
        raise RuntimeError("Local OKX trade-bar query returned no rows")
    bars = bars.loc[
        (bars.index >= pd.Timestamp(warmup_start_date))
        & (bars.index <= load_end)
    ].copy()
    return regularize_trade_bar_axis(bars, bar_delta=bar_delta)
