from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader

from .config import SourceLockedConfig


def clean_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        idx = pd.to_datetime(out.index, errors="coerce")
    elif "timestamp" in out.columns:
        idx = pd.to_datetime(out["timestamp"], errors="coerce")
    else:
        idx = pd.to_datetime(out.index, errors="coerce")
    out.index = pd.DatetimeIndex(idx)
    out = out[~out.index.isna()].sort_index()
    return out[~out.index.duplicated(keep="last")]


def resample_daily_causal(one_minute: pd.DataFrame, timezone_offset_hours: int) -> pd.DataFrame:
    """Build complete +8 daily bars; a day is visible only at the next 08:00 boundary."""
    bars = clean_index(one_minute)
    shifted = bars[["open", "high", "low", "close"]].copy()
    shifted.index = shifted.index - pd.Timedelta(hours=timezone_offset_hours)
    grouped = shifted.resample("1D", label="left", closed="left")
    out = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    counts = grouped["close"].count()
    out = out.loc[counts >= 1440].dropna()
    out.index = out.index + pd.Timedelta(hours=timezone_offset_hours)
    out["available_time"] = out.index + pd.Timedelta(days=1)
    return out


@dataclass
class SourceLockedData:
    cfg: SourceLockedConfig
    one_minute: pd.DataFrame
    cache: dict[str, pd.DataFrame] = field(default_factory=dict)

    def daily(self) -> pd.DataFrame:
        if "1D" not in self.cache:
            self.cache["1D"] = resample_daily_causal(self.one_minute, self.cfg.timezone_offset_hours)
        return self.cache["1D"]


def load_data(cfg: SourceLockedConfig) -> SourceLockedData:
    loader = OKXTradeBarLoader(symbol=cfg.symbol, timeframe="1m")
    frame = loader.load_local_data(start_date=cfg.warmup_start, end_date=cfg.research_end)
    frame = clean_index(frame)
    if frame.empty:
        raise RuntimeError("local 1m trade-bar cache is empty for R03 window")
    missing = sorted({"open", "high", "low", "close"} - set(frame.columns))
    if missing:
        raise RuntimeError(f"1m trade-bar cache missing required columns: {missing}")
    return SourceLockedData(cfg=cfg, one_minute=frame)
