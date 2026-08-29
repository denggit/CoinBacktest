from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader

from .config import TournamentConfig


FLOW_SUM_COLS = [
    "volume",
    "trades_count",
    "buy_volume",
    "sell_volume",
    "notional",
    "buy_notional",
    "sell_notional",
    "buy_trades_count",
    "sell_trades_count",
    "delta_volume",
    "delta_notional",
    "large_buy_notional",
    "large_sell_notional",
    "large_buy_trades_count",
    "large_sell_trades_count",
    "large_delta_notional",
    "large_trades_count",
]


@dataclass
class TournamentData:
    cfg: TournamentConfig
    one_minute: pd.DataFrame
    cache: dict[str, pd.DataFrame] = field(default_factory=dict)

    def bars(self, rule: str) -> pd.DataFrame:
        key = f"bars:{rule}"
        if key not in self.cache:
            self.cache[key] = resample_trade_bars(self.one_minute, rule, self.cfg.timezone_offset_hours)
        return self.cache[key]

    def quarter_hour_opening_imbalance(self) -> pd.DataFrame:
        key = "quarter_hour_opening_imbalance"
        if key not in self.cache:
            self.cache[key] = load_quarter_hour_opening_imbalance(self.cfg)
        return self.cache[key]

    def absorption_features(self) -> pd.DataFrame:
        key = "absorption_features"
        if key not in self.cache:
            self.cache[key] = load_absorption_features(self.cfg)
        return self.cache[key]


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
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
    out = out[~out.index.duplicated(keep="last")]
    return out


def load_base_data(cfg: TournamentConfig) -> TournamentData:
    loader = OKXTradeBarLoader(symbol=cfg.symbol, timeframe="1m")
    df = loader.load_local_data(start_date=cfg.warmup_start, end_date=cfg.research_end)
    if df.empty:
        raise RuntimeError("local 1m trade-bar cache is empty for requested tournament window")
    df = _clean_index(df)
    required = {"open", "high", "low", "close", "notional", "buy_notional", "sell_notional", "delta_notional"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"1m trade-bar cache missing required columns: {missing}")
    return TournamentData(cfg=cfg, one_minute=df)


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


def _expected_minutes(rule: str) -> int | None:
    td = pd.Timedelta(rule)
    minutes = td.total_seconds() / 60.0
    if minutes >= 1 and float(minutes).is_integer():
        return int(minutes)
    return None


def resample_trade_bars(df: pd.DataFrame, rule: str, timezone_offset_hours: int = 8) -> pd.DataFrame:
    """Causally aggregate 1m bars; daily bars are anchored at local 08:00 for UTC days."""
    bars = _clean_index(df)
    if bars.empty:
        return bars
    agg = _agg_map(bars.columns)
    if rule in {"1D", "1d", "24h", "24H"}:
        shifted = bars.copy()
        shifted.index = shifted.index - pd.Timedelta(hours=timezone_offset_hours)
        grouped = shifted.resample("1D", label="left", closed="left")
        out = grouped.agg(agg)
        counts = grouped["close"].count()
        out.index = out.index + pd.Timedelta(hours=timezone_offset_hours)
        counts.index = counts.index + pd.Timedelta(hours=timezone_offset_hours)
        expected = 1440
    else:
        grouped = bars.resample(rule, label="left", closed="left")
        out = grouped.agg(agg)
        counts = grouped["close"].count()
        expected = _expected_minutes(rule)
    if expected is not None:
        out = out.loc[counts >= expected]
    out = out.dropna(subset=[c for c in ("open", "high", "low", "close") if c in out.columns])
    delta = pd.Timedelta(rule)
    out["available_time"] = out.index + delta
    if {"buy_notional", "sell_notional"}.issubset(out.columns):
        denom = out["buy_notional"] + out["sell_notional"]
        out["flow_imbalance"] = (out["buy_notional"] - out["sell_notional"]) / denom.replace(0.0, np.nan)
    if "delta_notional" in out.columns:
        out["cvd_notional"] = out["delta_notional"].cumsum()
    return out


def _month_ranges(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    current = start.to_period("M").to_timestamp()
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    while current <= end:
        nxt = current + pd.offsets.MonthBegin(1)
        lo = max(start, current)
        hi = min(end, nxt - pd.Timedelta(microseconds=1))
        ranges.append((lo, hi))
        current = nxt
    return ranges


def load_quarter_hour_opening_imbalance(cfg: TournamentConfig) -> pd.DataFrame:
    """Load only the first 10 seconds at 00/15/30/45-minute marks from cached 5s bars."""
    loader = OKXTradeBarLoader(symbol=cfg.symbol, timeframe="5s")
    start = pd.Timestamp(cfg.research_start) - pd.Timedelta(days=31)
    end = pd.Timestamp(cfg.research_end)
    parts: list[pd.DataFrame] = []
    cols = ["buy_notional", "sell_notional", "notional", "open", "close"]
    for lo, hi in _month_ranges(start, end):
        frame = loader.load_local_data(start_date=lo, end_date=hi)
        if frame.empty:
            continue
        frame = _clean_index(frame)
        frame = frame[[c for c in cols if c in frame.columns]].copy()
        mask = frame.index.minute.isin([0, 15, 30, 45]) & (frame.index.second < 10)
        sub = frame.loc[mask]
        if sub.empty:
            continue
        bucket = sub.index.floor("15min")
        agg = sub.assign(_bucket=bucket).groupby("_bucket", sort=True).agg(
            buy_notional=("buy_notional", "sum"),
            sell_notional=("sell_notional", "sum"),
            bars=("buy_notional", "size"),
        )
        agg = agg[agg["bars"] >= 2]
        parts.append(agg)
    if not parts:
        return pd.DataFrame(columns=["buy_notional", "sell_notional", "bars", "imbalance", "available_time"])
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    denom = out["buy_notional"] + out["sell_notional"]
    out["imbalance"] = (out["buy_notional"] - out["sell_notional"]) / denom.replace(0.0, np.nan)
    # Two completed 5s bars => information is available 10 seconds after the quarter-hour.
    out["available_time"] = out.index + pd.Timedelta(seconds=10)
    return out


def _detach_ambiguous_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    index_names = {name for name in out.index.names if name is not None}
    if index_names.intersection(out.columns):
        out = out.reset_index(drop=True)
    return out


def load_absorption_features(cfg: TournamentConfig) -> pd.DataFrame:
    bars_loader = OKXRangeBarLoader(symbol=cfg.symbol, range_pct=cfg.footprint_range_pct)
    fp_loader = OKXRangeFootprintLoader(
        symbol=cfg.symbol,
        range_pct=cfg.footprint_range_pct,
        price_step=cfg.footprint_price_step,
    )
    start = pd.Timestamp(cfg.warmup_start)
    end = pd.Timestamp(cfg.research_end)
    bar_parts: list[pd.DataFrame] = []
    footprint_summary_parts: list[pd.DataFrame] = []
    bar_cols = ["bar_id", "start_ts", "end_ts", "open", "high", "low", "close", "notional", "delta_notional"]
    fp_cols = ["bar_id", "end_ts", "price_bucket", "buy_notional", "sell_notional", "delta_notional"]
    for lo, hi in _month_ranges(start, end):
        rb = bars_loader.load_local_data(start_date=lo, end_date=hi, columns=bar_cols)
        rb = _detach_ambiguous_index(rb)
        if rb.empty:
            continue
        rb["end_ts"] = pd.to_datetime(rb["end_ts"], errors="coerce")
        rb = rb.dropna(subset=["end_ts", "bar_id"]).drop_duplicates("bar_id", keep="last")
        bar_parts.append(rb)
        bid_min, bid_max = int(rb["bar_id"].min()), int(rb["bar_id"].max())
        fp = fp_loader.load_local_data(
            bar_id_min=bid_min,
            bar_id_max=bid_max,
            columns=fp_cols,
        )
        if fp.empty:
            continue
        # Aggregate the potentially huge price-bucket footprint immediately for
        # this month, then release raw rows. Multi-year raw footprint is never
        # materialized in memory.
        fp = fp.drop_duplicates(["bar_id", "price_bucket"], keep="last")
        ranges = rb.set_index("bar_id")[["low", "high"]]
        fp = fp.join(ranges, on="bar_id", how="inner")
        width = (fp["high"] - fp["low"]).replace(0.0, np.nan)
        pos = (fp["price_bucket"] - fp["low"]) / width
        fp["zone"] = np.where(pos <= 1.0 / 3.0, "lower", np.where(pos >= 2.0 / 3.0, "upper", "middle"))
        fp["zone_notional"] = fp["buy_notional"] + fp["sell_notional"]
        grouped = fp.groupby(["bar_id", "zone"], sort=False).agg(
            delta_notional=("delta_notional", "sum"),
            zone_notional=("zone_notional", "sum"),
        )
        grouped["delta_ratio"] = grouped["delta_notional"] / grouped["zone_notional"].replace(0.0, np.nan)
        piv = grouped["delta_ratio"].unstack("zone")
        footprint_summary_parts.append(
            piv.rename(columns={"lower": "lower_delta_ratio", "upper": "upper_delta_ratio"})
            .reindex(columns=["lower_delta_ratio", "upper_delta_ratio"])
            .reset_index()
        )
        del fp, grouped, piv
    if not bar_parts:
        return pd.DataFrame()
    bars = pd.concat(bar_parts, ignore_index=True).drop_duplicates("bar_id", keep="last").sort_values("end_ts")
    if footprint_summary_parts:
        fps = pd.concat(footprint_summary_parts, ignore_index=True).drop_duplicates("bar_id", keep="last")
        bars = bars.merge(fps, on="bar_id", how="left")
    else:
        bars["lower_delta_ratio"] = np.nan
        bars["upper_delta_ratio"] = np.nan
    width = (bars["high"] - bars["low"]).replace(0.0, np.nan)
    bars["close_pos"] = (bars["close"] - bars["low"]) / width
    bars["total_delta_ratio"] = bars["delta_notional"] / bars["notional"].replace(0.0, np.nan)
    bars["available_time"] = bars["end_ts"]
    return bars.set_index("end_ts", drop=False).sort_index()
