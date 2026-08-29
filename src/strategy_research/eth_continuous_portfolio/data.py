from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader

from .config import ContinuousPortfolioConfig


FLOW_SUM_COLS = (
    "volume",
    "notional",
    "buy_volume",
    "sell_volume",
    "buy_notional",
    "sell_notional",
    "delta_volume",
    "delta_notional",
    "trades_count",
    "buy_trades_count",
    "sell_trades_count",
    "large_buy_trades_count",
    "large_sell_trades_count",
)


def clean_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        idx = pd.to_datetime(out.index, errors="coerce")
    elif "timestamp" in out.columns:
        idx = pd.to_datetime(out["timestamp"], errors="coerce")
    else:
        idx = pd.to_datetime(out.index, errors="coerce")
    out.index = pd.DatetimeIndex(idx)
    out = out[~out.index.isna()].sort_index()
    return out[~out.index.duplicated(keep="last")]


def _agg_map(columns: Iterable[str]) -> dict[str, str]:
    cols = set(columns)
    agg: dict[str, str] = {}
    for c, f in (("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")):
        if c in cols:
            agg[c] = f
    for c in FLOW_SUM_COLS:
        if c in cols:
            agg[c] = "sum"
    if "max_trade_notional" in cols:
        agg["max_trade_notional"] = "max"
    if "max_trade_size" in cols:
        agg["max_trade_size"] = "max"
    return agg


def resample_causal(df: pd.DataFrame, rule: str, timezone_offset_hours: int) -> pd.DataFrame:
    """Aggregate closed bars and expose each row only at its available time."""
    bars = clean_index(df)
    if bars.empty:
        return bars
    agg = _agg_map(bars.columns)
    if rule.upper() in {"1D", "24H"}:
        shifted = bars.copy()
        shifted.index = shifted.index - pd.Timedelta(hours=timezone_offset_hours)
        grouped = shifted.resample("1D", label="left", closed="left")
        out = grouped.agg(agg)
        counts = grouped["close"].count()
        out.index = out.index + pd.Timedelta(hours=timezone_offset_hours)
        counts.index = counts.index + pd.Timedelta(hours=timezone_offset_hours)
        expected = 1440
        delta = pd.Timedelta(days=1)
    else:
        grouped = bars.resample(rule, label="left", closed="left")
        out = grouped.agg(agg)
        counts = grouped["close"].count()
        minutes = pd.Timedelta(rule).total_seconds() / 60.0
        expected = int(minutes) if minutes >= 1 and float(minutes).is_integer() else None
        delta = pd.Timedelta(rule)
    if expected is not None:
        out = out.loc[counts >= expected]
    required = [c for c in ("open", "high", "low", "close") if c in out.columns]
    out = out.dropna(subset=required)
    out["available_time"] = out.index + delta
    return out


@dataclass
class ContinuousPortfolioData:
    cfg: ContinuousPortfolioConfig
    one_minute: pd.DataFrame
    cache: dict[str, pd.DataFrame] = field(default_factory=dict)

    def bars(self, rule: str) -> pd.DataFrame:
        key = rule.upper()
        if key not in self.cache:
            self.cache[key] = resample_causal(self.one_minute, rule, self.cfg.timezone_offset_hours)
        return self.cache[key]


def load_data(cfg: ContinuousPortfolioConfig) -> ContinuousPortfolioData:
    loader = OKXTradeBarLoader(symbol=cfg.symbol, timeframe="1m")
    frame = loader.load_local_data(start_date=cfg.warmup_start, end_date=cfg.research_end)
    frame = clean_index(frame)
    if frame.empty:
        raise RuntimeError("local 1m trade-bar cache is empty for R02 window")
    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"1m trade-bar cache missing required columns: {missing}")
    return ContinuousPortfolioData(cfg=cfg, one_minute=frame)
