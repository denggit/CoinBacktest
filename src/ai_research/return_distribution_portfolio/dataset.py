from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.research_common.progress import ProgressReporter

from .config import ReturnDistributionConfig


_REQUIRED_1M = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "notional",
    "trades_count",
    "buy_notional",
    "sell_notional",
    "delta_notional",
    "large_buy_notional",
    "large_sell_notional",
    "large_delta_notional",
)

_SUM_COLUMNS = (
    "volume",
    "notional",
    "trades_count",
    "buy_notional",
    "sell_notional",
    "delta_notional",
    "large_buy_notional",
    "large_sell_notional",
    "large_delta_notional",
)


@dataclass(frozen=True)
class YearShard:
    year: int
    frame: pd.DataFrame


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def _efficiency(close: pd.Series, window: int) -> pd.Series:
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window, min_periods=window).sum()
    return net / path.replace(0.0, np.nan)


def _future_rolling(series: pd.Series, window: int, op: str) -> pd.Series:
    rev = series.iloc[::-1]
    roll = rev.rolling(window, min_periods=window)
    if op == "max":
        out = roll.max()
    elif op == "min":
        out = roll.min()
    elif op == "std":
        out = roll.std(ddof=0)
    else:
        raise ValueError(f"unsupported future rolling op: {op}")
    return out.iloc[::-1]


def _resample_5m(raw: pd.DataFrame, minutes: int) -> pd.DataFrame:
    rule = f"{minutes}min"
    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    agg.update({col: "sum" for col in _SUM_COLUMNS if col in raw.columns})
    work = raw.copy()
    work["__minute_count"] = work["close"].notna().astype("int8")
    agg["__minute_count"] = "sum"
    bars = work.resample(rule, label="left", closed="left").agg(agg)
    bars = bars.loc[bars["__minute_count"] == minutes].drop(columns=["__minute_count"])
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    # Critical causal contract: the left-labeled bar becomes visible only after
    # its full interval has completed.
    bars.index = bars.index + pd.Timedelta(minutes=minutes)
    bars.index.name = "decision_time"
    return bars


def build_causal_features(raw_1m: pd.DataFrame, config: ReturnDistributionConfig) -> pd.DataFrame:
    bars = _resample_5m(raw_1m, config.decision_minutes)
    out = bars.copy()
    close = out["close"].astype(float)
    ret1 = np.log(close).diff()
    out["ret_5m"] = ret1

    windows = {
        "15m": 3,
        "30m": 6,
        "1h": 12,
        "2h": 24,
        "6h": 72,
        "24h": 288,
        "72h": 864,
    }
    for name, window in windows.items():
        out[f"ret_{name}"] = np.log(close / close.shift(window))
        out[f"rv_{name}"] = ret1.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(window)
        out[f"eff_{name}"] = _efficiency(close, window)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_1h_pct"] = tr.rolling(12, min_periods=12).mean() / close
    out["atr_6h_pct"] = tr.rolling(72, min_periods=72).mean() / close

    for name, window in (("6h", 72), ("24h", 288), ("72h", 864)):
        hh = out["high"].rolling(window, min_periods=window).max()
        ll = out["low"].rolling(window, min_periods=window).min()
        span = (hh - ll).replace(0.0, np.nan)
        out[f"range_pos_{name}"] = (close - ll) / span
        out[f"drawdown_{name}"] = close / hh - 1.0

    notional = out.get("notional", pd.Series(np.nan, index=out.index)).astype(float)
    delta = out.get("delta_notional", pd.Series(np.nan, index=out.index)).astype(float)
    large_delta = out.get("large_delta_notional", pd.Series(np.nan, index=out.index)).astype(float)
    for name, window in (("5m", 1), ("30m", 6), ("2h", 24), ("6h", 72)):
        n = notional.rolling(window, min_periods=window).sum()
        d = delta.rolling(window, min_periods=window).sum()
        ld = large_delta.rolling(window, min_periods=window).sum()
        out[f"flow_imb_{name}"] = d / n.replace(0.0, np.nan)
        out[f"large_flow_imb_{name}"] = ld / n.replace(0.0, np.nan)

    if "buy_notional" in out.columns and "sell_notional" in out.columns:
        denom = (out["buy_notional"] + out["sell_notional"]).replace(0.0, np.nan)
        out["taker_buy_ratio_5m"] = out["buy_notional"] / denom
    if "notional" in out.columns:
        out["notional_z_2h"] = _zscore(np.log1p(out["notional"].astype(float)), 24)
        out["notional_z_24h"] = _zscore(np.log1p(out["notional"].astype(float)), 288)
    if "trades_count" in out.columns:
        out["trade_count_z_2h"] = _zscore(np.log1p(out["trades_count"].astype(float)), 24)
    out["impact_efficiency_5m"] = out["ret_5m"] / out["flow_imb_5m"].abs().replace(0.0, np.nan)

    # Time is context, not a hard session filter.
    minute_of_day = out.index.hour * 60 + out.index.minute
    angle = 2.0 * np.pi * minute_of_day / 1440.0
    out["tod_sin"] = np.sin(angle)
    out["tod_cos"] = np.cos(angle)
    dow_angle = 2.0 * np.pi * out.index.dayofweek / 7.0
    out["dow_sin"] = np.sin(dow_angle)
    out["dow_cos"] = np.cos(dow_angle)

    return out


def build_future_targets(raw_1m: pd.DataFrame, decisions: pd.DatetimeIndex, config: ReturnDistributionConfig) -> pd.DataFrame:
    minute = raw_1m.loc[:, ["open", "high", "low", "close"]].copy().sort_index()
    minute = minute.loc[~minute.index.duplicated(keep="last")]
    if not minute.empty:
        full_index = pd.date_range(minute.index.min(), minute.index.max(), freq="1min")
        minute = minute.reindex(full_index)
    result = pd.DataFrame(index=decisions)
    result.index.name = "decision_time"

    entry_open = minute["open"].reindex(decisions)
    result["execution_price"] = entry_open

    log_ret_1m = np.log(minute["close"]).diff()
    for horizon in config.horizons_minutes:
        exit_time = decisions + pd.Timedelta(minutes=horizon)
        exit_open = minute["open"].reindex(exit_time)
        exit_open.index = decisions
        future_high = _future_rolling(minute["high"].astype(float), horizon, "max").reindex(decisions)
        future_low = _future_rolling(minute["low"].astype(float), horizon, "min").reindex(decisions)
        future_rv = _future_rolling(log_ret_1m.astype(float), horizon, "std").reindex(decisions) * np.sqrt(horizon)

        entry = entry_open.astype(float)
        ret = exit_open.astype(float) / entry - 1.0
        result[f"ret_h{horizon}"] = ret
        result[f"mfe_long_h{horizon}"] = future_high / entry - 1.0
        result[f"mae_long_h{horizon}"] = 1.0 - future_low / entry
        result[f"mfe_short_h{horizon}"] = 1.0 - future_low / entry
        result[f"mae_short_h{horizon}"] = future_high / entry - 1.0
        result[f"future_rv_h{horizon}"] = future_rv

    return result


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded_prefixes = ("ret_h", "mfe_", "mae_", "future_rv_h")
    excluded = {"year", "execution_price", "open", "high", "low", "close", "volume", "notional", "trades_count", "buy_notional", "sell_notional", "delta_notional", "large_buy_notional", "large_sell_notional", "large_delta_notional"}
    return [
        c
        for c in frame.columns
        if c not in excluded and not c.startswith(excluded_prefixes) and pd.api.types.is_numeric_dtype(frame[c])
    ]


def shard_path(config: ReturnDistributionConfig, year: int) -> Path:
    return config.cache_path / f"price_flow_distribution_{year}.pkl.gz"


def build_year_shard(
    year: int,
    config: ReturnDistributionConfig,
    *,
    data_dir: str | None = None,
    force: bool = False,
    progress: bool = True,
) -> YearShard:
    config.validate()
    path = shard_path(config, year)
    if path.exists() and not force:
        return YearShard(year=year, frame=pd.read_pickle(path, compression="gzip"))

    research_start = pd.Timestamp(config.research_start)
    research_end = pd.Timestamp(config.research_end)
    year_start = max(pd.Timestamp(f"{year}-01-01"), research_start)
    year_end = min(pd.Timestamp(f"{year}-12-31 23:59:59"), research_end)
    if year_end < year_start:
        raise ValueError(f"year {year} lies outside research window")

    max_h = max(config.horizons_minutes)
    load_start = year_start - pd.Timedelta(days=config.shard_context_days)
    load_end = year_end + pd.Timedelta(minutes=max_h + 5)
    load_start = max(load_start, pd.Timestamp(config.warmup_start))

    reporter = ProgressReporter(label=f"[RDP][{year}]", total=4, every=1, enabled=progress)
    loader = OKXTradeBarLoader(symbol=config.symbol, timeframe="1m", data_dir=data_dir)
    raw = loader.fetch_data_by_date_range(load_start, load_end, build_missing=False, cvd_mode="range")
    reporter.update(1)
    if raw.empty:
        raise RuntimeError(f"no local 1m trade bars for {load_start} -> {load_end}")
    missing = [c for c in _REQUIRED_1M if c not in raw.columns]
    if missing:
        raise RuntimeError(f"1m trade bars missing required columns: {missing}")
    raw = raw.loc[:, list(_REQUIRED_1M)].sort_index()

    features = build_causal_features(raw, config)
    reporter.update(2)
    decisions = features.index[(features.index >= year_start) & (features.index <= year_end)]
    targets = build_future_targets(raw, decisions, config)
    reporter.update(3)
    frame = features.reindex(decisions).join(targets, how="left")
    frame.insert(0, "year", year)
    frame = frame.replace([np.inf, -np.inf], np.nan)

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(path, compression="gzip", protocol=5)
    reporter.update(4, force=True)
    reporter.close()
    return YearShard(year=year, frame=frame)


def load_or_build_shards(
    config: ReturnDistributionConfig,
    *,
    data_dir: str | None = None,
    force: bool = False,
    progress: bool = True,
) -> dict[int, pd.DataFrame]:
    start_year = pd.Timestamp(config.research_start).year
    end_year = pd.Timestamp(config.research_end).year
    return {
        year: build_year_shard(year, config, data_dir=data_dir, force=force, progress=progress).frame
        for year in range(start_year, end_year + 1)
    }
