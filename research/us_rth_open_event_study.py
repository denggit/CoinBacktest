#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH US RTH Opening Event Study
==============================

Discovery-only research lab for ETH behavior around the US equity regular
trading-hours open.  The primary OHLCV axis is built from OKX raw trades via
``OKXTradeBarLoader``; ordinary exchange K-lines are intentionally not used.
The script builds daily RTH outcome labels, open-window feature snapshots, and
causal closed-bar / next-open event-study labels.

Important boundaries
--------------------
1. Full-session max-up/max-down values are labels only.  They are never used to
   generate event conditions.
2. Event rows are generated only from bars that have already closed.  Execution
   labels use next-bar open through ``src.research_common.event_study``.
3. Trade-bar order-flow features come from the primary trade bars by default;
   optional higher-timeframe trade-bar / range-bar / footprint context is aligned
   onto the primary closed-bar axis without looking past the signal row.
4. This is not a final strategy.  Candidate rows are phenomena that still need a
   formal backtest, fee/slippage/delay stress, parameter-neighbourhood checks,
   walk-forward checks, and live execution review.

Example
-------
python research/us_rth_open_event_study.py --symbol ETH-USDT-SWAP --primary-timeframe 1m --start-date 2023-01-01 --end-date 2026-06-30 --warmup-start-date 2022-01-01 --out-dir data/reports/research/us_rth_open_event_study
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # noqa: E402
    from config.loader import TIMEZONE  # type: ignore
except Exception:  # pragma: no cover
    TIMEZONE = "+8"

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.event_study import (  # noqa: E402
    CostConfig,
    EventStudyConfig,
    fixed_threshold_labels,
    qcut_labels,
    run_event_study,
    summarize_many,
    top_winner_dependency,
)
from src.research_common.progress import ProgressReporter  # noqa: E402

try:  # Optional heavy loaders are imported lazily if available.
    from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader  # noqa: E402
    from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader  # noqa: E402
except Exception:  # pragma: no cover
    OKXRangeBarLoader = None  # type: ignore
    OKXRangeFootprintLoader = None  # type: ignore

SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1H", "4H"}
TRADE_BAR_CONTEXT_COLUMNS = [
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
    "cvd_volume",
    "cvd_notional",
    "taker_buy_ratio",
    "avg_trade_size",
    "vwap",
    "large_buy_notional",
    "large_sell_notional",
    "large_buy_trades_count",
    "large_sell_trades_count",
    "large_delta_notional",
    "large_trades_count",
    "max_trade_notional",
    "max_trade_size",
]
US_EQUITY_HOLIDAYS = {
    # NYSE full-day holidays, enough for the project's default 2023-2026 research window.
    "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30", "2022-06-20", "2022-07-04", "2022-09-05", "2022-11-24", "2022-12-26",
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29", "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25",
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
}


@dataclass(frozen=True)
class SessionObservation:
    session_date: str
    session_id: int
    open_pos: int
    close_pos: int
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    open_price: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ETH US RTH opening-regime event study with causal next-open labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--primary-timeframe", default="1m", choices=sorted(SUPPORTED_TIMEFRAMES))
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--out-dir", default="data/reports/research/us_rth_open_event_study")

    p.add_argument("--ny-open", default="09:30")
    p.add_argument("--ny-close", default="16:00")
    p.add_argument("--open-windows-min", default="5,15,30,60,120")
    p.add_argument("--horizons", default="5,15,30,60,120,240,390", help="Forward horizons in primary bars.")
    p.add_argument("--mfe-mae-horizon", type=int, default=120)
    p.add_argument("--candidate-horizon", type=int, default=60)

    p.add_argument("--timestamp-offset-hours", type=float, default=None, help="Naive DB timestamp offset vs UTC. Default parses config.loader.TIMEZONE, usually +8.")
    p.add_argument("--exclude-weekends", action="store_true", default=True)
    p.add_argument("--include-known-us-equity-holidays", action="store_true", default=False)
    p.add_argument("--min-rth-bars", type=int, default=300)

    p.add_argument("--include-trade-bars", action="store_true", default=False, help="Compatibility switch. Primary OHLCV is always trade_bar; this only loads extra trade-bar context when --tradebar-timeframe differs from --primary-timeframe.")
    p.add_argument("--tradebar-timeframe", default=None, help="Optional additional trade-bar context timeframe. Defaults to primary timeframe, so no second load is needed.")
    p.add_argument("--include-range-bars", action="store_true", default=False)
    p.add_argument("--range-pcts", default="0.0015,0.0020,0.0025")
    p.add_argument("--include-footprint", action="store_true", default=False)
    p.add_argument("--footprint-range-pct", type=float, default=0.0020)
    p.add_argument("--footprint-price-step", type=float, default=1.0)
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--force-rebuild-context", action="store_true", default=False)
    p.add_argument("--no-build-missing-trade-bars", action="store_true", default=False)

    p.add_argument("--big-move-pcts", default="0.005,0.010,0.015,0.020")
    p.add_argument("--open-move-thresholds", default="0.0010,0.0020,0.0030,0.0050")
    p.add_argument("--fakeout-thresholds", default="0.0020,0.0030,0.0050")
    p.add_argument("--min-count", type=int, default=60)
    p.add_argument("--max-top5-winner-share", type=float, default=0.50)

    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage-pct", type=float, default=0.0)
    p.add_argument("--exit-slippage-pct", type=float, default=0.0)
    p.add_argument("--progress-every", type=int, default=2000)
    p.add_argument("--save-feature-sample", type=int, default=5000)
    return p.parse_args(argv)


def _parse_number_list(text: str, *, name: str, value_type: type = float) -> list:
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = value_type(part)
        if isinstance(value, (int, float)) and value <= 0:
            raise ValueError(f"{name} must contain positive values")
        values.append(value)
    if not values:
        raise ValueError(f"{name} must not be empty")
    return sorted(set(values))


def _parse_hhmm(value: str) -> tuple[int, int]:
    h, m = str(value).split(":", 1)
    return int(h), int(m)


def _timeframe_minutes(timeframe: str) -> int:
    tf = str(timeframe)
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("H"):
        return int(tf[:-1]) * 60
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _timeframe_timedelta(timeframe: str) -> pd.Timedelta:
    return pd.Timedelta(minutes=_timeframe_minutes(timeframe))


def _parse_timezone_offset_hours(value: str | None) -> float:
    text = str(value or "+0").strip()
    if not text:
        return 0.0
    sign = -1.0 if text.startswith("-") else 1.0
    text = text.lstrip("+-")
    try:
        return sign * float(text)
    except ValueError:
        return 0.0


def _attach_ny_time(index: pd.DatetimeIndex, *, offset_hours: float, tz_name: str = "America/New_York") -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is None:
        utc = (idx - pd.Timedelta(hours=float(offset_hours))).tz_localize("UTC")
    else:
        utc = idx.tz_convert("UTC")
    ny = utc.tz_convert(ZoneInfo(tz_name))
    return pd.DataFrame(
        {
            "utc_time": utc,
            "ny_time": ny,
            "ny_date": [x.date().isoformat() for x in ny],
            "ny_weekday": [int(x.weekday()) for x in ny],
            "ny_minutes": [int(x.hour) * 60 + int(x.minute) for x in ny],
        },
        index=index,
    )


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    out = pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce").replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _load_primary_trade_bars(args: argparse.Namespace) -> pd.DataFrame:
    """Load the research primary OHLCV axis from trade-aggregated bars only.

    Ordinary OKX K-lines are intentionally not used in this study.  The trade bar
    loader is cache-first and can build genuinely missing raw-trade coverage when
    allowed by ``--no-build-missing-trade-bars`` / ``--force-rebuild-context``.
    """
    loader = OKXTradeBarLoader(symbol=args.symbol, timeframe=args.primary_timeframe)
    df = loader.fetch_data_by_date_range(
        args.warmup_start_date,
        args.end_date,
        chunksize=int(args.chunksize),
        force_rebuild=bool(args.force_rebuild_context),
        build_missing=not bool(args.no_build_missing_trade_bars),
        cvd_mode="range",
    )
    if df.empty:
        raise RuntimeError(
            f"No primary trade_bar data loaded for {args.symbol} {args.primary_timeframe} "
            f"{args.warmup_start_date} -> {args.end_date}"
        )
    df = df.sort_index().copy()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")].sort_index()
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Primary trade bars missing columns: {missing}")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=required)



def _add_trade_bar_orderflow_features(df: pd.DataFrame, *, source: pd.DataFrame) -> pd.DataFrame:
    """Add tb_* order-flow columns from trade-bar source without reloading data."""
    out = df.copy()
    for col in TRADE_BAR_CONTEXT_COLUMNS:
        if col in source.columns:
            out[f"tb_{col}"] = pd.to_numeric(source[col], errors="coerce")
    for w in (5, 15, 30, 60):
        for col in ["tb_delta_notional", "tb_large_delta_notional", "tb_notional", "tb_trades_count"]:
            if col in out.columns:
                out[f"{col}_sum_{w}"] = pd.to_numeric(out[col], errors="coerce").rolling(w, min_periods=max(2, w // 3)).sum()
    if "tb_delta_notional" in out.columns and "tb_notional" in out.columns:
        out["tb_delta_ratio"] = _safe_divide(out["tb_delta_notional"], out["tb_notional"].abs())
    return out

def build_primary_features(bars: pd.DataFrame, *, offset_hours: float) -> pd.DataFrame:
    df = bars.copy().sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ret_1"] = df["close"].pct_change()
    df["logret_1"] = np.log(df["close"] / df["close"].shift(1)).replace([np.inf, -np.inf], np.nan)
    df["tr"] = _true_range(df)
    for w in (5, 15, 30, 60, 120, 240, 390, 720, 1440):
        if len(df) >= w:
            df[f"ret_{w}"] = df["close"].pct_change(w)
            roll_high = df["high"].rolling(w, min_periods=max(3, w // 3)).max()
            roll_low = df["low"].rolling(w, min_periods=max(3, w // 3)).min()
            df[f"range_pct_{w}"] = _safe_divide(roll_high - roll_low, df["close"])
            df[f"rv_{w}"] = df["logret_1"].rolling(w, min_periods=max(3, w // 3)).std() * math.sqrt(w)
            df[f"volume_sum_{w}"] = df["volume"].rolling(w, min_periods=max(3, w // 3)).sum()
    for w in (20, 50, 100, 200):
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False, min_periods=w).mean()
        df[f"dist_ema_{w}"] = df["close"] / df[f"ema_{w}"] - 1.0
    df["atr_60"] = df["tr"].rolling(60, min_periods=20).mean()
    df["atr_pct_60"] = _safe_divide(df["atr_60"], df["close"])
    vol_med = df["volume"].shift(1).rolling(1440, min_periods=240).median()
    df["volume_ratio_1d"] = _safe_divide(df["volume"], vol_med)
    df["volume_spike_1d"] = df["volume_ratio_1d"] >= 1.5
    df["atr_pct_rel_1d"] = _safe_divide(df["atr_pct_60"], df["atr_pct_60"].shift(1).rolling(1440, min_periods=240).median())
    bar_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    df["close_pos"] = _safe_divide(df["close"] - df["low"], bar_range)
    df["upper_wick_frac"] = _safe_divide(df["high"] - df[["open", "close"]].max(axis=1), bar_range)
    df["lower_wick_frac"] = _safe_divide(df[["open", "close"]].min(axis=1) - df["low"], bar_range)
    df = _add_trade_bar_orderflow_features(df, source=bars)
    ny = _attach_ny_time(df.index, offset_hours=offset_hours)
    return pd.concat([df, ny], axis=1)


def _causal_join_same_or_higher_tf(primary: pd.DataFrame, ctx: pd.DataFrame, *, ctx_timeframe: str, prefix: str) -> pd.DataFrame:
    if ctx.empty:
        return primary
    p = primary.copy().sort_index()
    c = ctx.copy().sort_index()
    c = c.add_prefix(prefix)
    # Same-timeframe bars close together with the primary row; higher-timeframe bars become visible at bar_start + tf_delta.
    if _timeframe_minutes(ctx_timeframe) <= 1:
        return p.join(c, how="left")
    c.index = c.index + _timeframe_timedelta(ctx_timeframe)
    return pd.merge_asof(p.sort_index(), c.sort_index(), left_index=True, right_index=True, direction="backward")


def attach_trade_bar_context(features: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Optionally attach a different-timeframe trade-bar context.

    The primary frame is already trade_bar, so when the context timeframe equals
    the primary timeframe this function deliberately does nothing to avoid a
    duplicate DB scan and duplicate feature names.
    """
    ctx_timeframe = args.tradebar_timeframe or args.primary_timeframe
    if str(ctx_timeframe) == str(args.primary_timeframe):
        print("[context] primary bars are already trade_bar; skip duplicate trade-bar context load", flush=True)
        return features
    print(f"[context] loading extra trade bars {args.symbol} {ctx_timeframe}", flush=True)
    loader = OKXTradeBarLoader(symbol=args.symbol, timeframe=ctx_timeframe)
    df = loader.fetch_data_by_date_range(
        args.warmup_start_date,
        args.end_date,
        chunksize=int(args.chunksize),
        force_rebuild=bool(args.force_rebuild_context),
        build_missing=not bool(args.no_build_missing_trade_bars),
        cvd_mode="range",
    )
    if df.empty:
        print("[context] extra trade bars empty; continuing without extra trade context", flush=True)
        return features
    cols = [c for c in TRADE_BAR_CONTEXT_COLUMNS if c in df.columns]
    ctx = df[cols].copy()
    out = _causal_join_same_or_higher_tf(features, ctx, ctx_timeframe=ctx_timeframe, prefix="tbctx_")
    for w in (5, 15, 30, 60):
        for col in ["tbctx_delta_notional", "tbctx_large_delta_notional", "tbctx_notional", "tbctx_trades_count"]:
            if col in out.columns:
                out[f"{col}_sum_{w}"] = pd.to_numeric(out[col], errors="coerce").rolling(w, min_periods=max(2, w // 3)).sum()
    if "tbctx_delta_notional" in out.columns and "tbctx_notional" in out.columns:
        out["tbctx_delta_ratio"] = _safe_divide(out["tbctx_delta_notional"], out["tbctx_notional"].abs())
    return out


def _resample_event_bars_to_primary_context(frame: pd.DataFrame, primary_index: pd.DatetimeIndex, *, prefix: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=primary_index)
    df = frame.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = "end_ts" if "end_ts" in df.columns else "timestamp"
        df.index = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_index()
    # Put each event bar into the primary bar bucket in which it closed.
    bucket = df.index.floor("min")
    df = df.assign(_bucket=bucket)
    agg_map: dict[str, tuple[str, str]] = {}
    for col in ["bar_id", "direction", "delta_notional", "large_delta_notional", "notional", "trades_count", "duration_seconds", "max_trade_notional"]:
        if col not in df.columns:
            continue
        if col == "bar_id":
            agg_map[f"{prefix}count"] = (col, "count")
        elif col == "duration_seconds":
            agg_map[f"{prefix}duration_median"] = (col, "median")
        else:
            agg_map[f"{prefix}{col}_sum"] = (col, "sum")
    if not agg_map:
        return pd.DataFrame(index=primary_index)
    out = df.groupby("_bucket").agg(**agg_map).sort_index()
    out = out.reindex(primary_index).fillna(0.0)
    for w in (5, 15, 30, 60):
        for col in list(out.columns):
            out[f"{col}_{w}m"] = out[col].rolling(w, min_periods=max(2, w // 3)).sum()
    return out


def attach_range_bar_context(features: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if OKXRangeBarLoader is None:
        print("[context] range bar loader unavailable; skipping", flush=True)
        return features
    out = features.copy()
    for range_pct in _parse_number_list(args.range_pcts, name="range_pcts", value_type=float):
        print(f"[context] loading range bars range_pct={range_pct}", flush=True)
        loader = OKXRangeBarLoader(symbol=args.symbol, range_pct=float(range_pct))  # type: ignore[operator]
        try:
            df = loader.fetch_data_by_date_range(
                args.warmup_start_date,
                args.end_date,
                chunksize=int(args.chunksize),
                force_rebuild=bool(args.force_rebuild_context),
                cvd_mode="range",
            )
        except Exception as exc:
            print(f"[context] range bars failed range_pct={range_pct}: {exc}; continuing", flush=True)
            continue
        code = str(range_pct).replace(".", "p")
        ctx = _resample_event_bars_to_primary_context(df, out.index, prefix=f"rb{code}_")
        out = out.join(ctx, how="left")
    return out


def attach_footprint_context(features: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if OKXRangeFootprintLoader is None:
        print("[context] footprint loader unavailable; skipping", flush=True)
        return features
    print(f"[context] loading footprint range_pct={args.footprint_range_pct} step={args.footprint_price_step}", flush=True)
    loader = OKXRangeFootprintLoader(  # type: ignore[operator]
        symbol=args.symbol,
        range_pct=float(args.footprint_range_pct),
        price_step=float(args.footprint_price_step),
    )
    try:
        fp = loader.fetch_data_by_date_range(
            args.warmup_start_date,
            args.end_date,
            chunksize=int(args.chunksize),
            force_rebuild=bool(args.force_rebuild_context),
        )
    except Exception as exc:
        print(f"[context] footprint failed: {exc}; continuing", flush=True)
        return features
    if fp.empty:
        print("[context] footprint empty; continuing", flush=True)
        return features
    df = fp.copy()
    for col in ["delta_notional", "large_delta_notional", "notional", "trades_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    grouped = df.groupby(["bar_id", "end_ts"], dropna=False).agg(
        fp_bucket_count=("price_bucket", "count"),
        fp_delta_notional_sum=("delta_notional", "sum"),
        fp_large_delta_notional_sum=("large_delta_notional", "sum"),
        fp_max_abs_bucket_delta=("delta_notional", lambda s: float(pd.to_numeric(s, errors="coerce").abs().max() or 0.0)),
        fp_notional_sum=("notional", "sum"),
    ).reset_index()
    grouped["fp_delta_ratio"] = _safe_divide(grouped["fp_delta_notional_sum"], grouped["fp_notional_sum"].abs())
    grouped = grouped.set_index(pd.to_datetime(grouped["end_ts"], errors="coerce")).sort_index()
    ctx = _resample_event_bars_to_primary_context(grouped, features.index, prefix="fp_")
    return features.join(ctx, how="left")


def _session_mask(features: pd.DataFrame, *, open_minute: int, close_minute: int, exclude_weekends: bool, include_holidays: bool) -> pd.Series:
    m = (features["ny_minutes"] >= open_minute) & (features["ny_minutes"] < close_minute)
    if exclude_weekends:
        m &= features["ny_weekday"] < 5
    if not include_holidays:
        m &= ~features["ny_date"].astype(str).isin(US_EQUITY_HOLIDAYS)
    return m.fillna(False).astype(bool)


def _find_sessions(features: pd.DataFrame, args: argparse.Namespace) -> list[SessionObservation]:
    open_h, open_m = _parse_hhmm(args.ny_open)
    close_h, close_m = _parse_hhmm(args.ny_close)
    open_minute = open_h * 60 + open_m
    close_minute = close_h * 60 + close_m
    rth_mask = _session_mask(
        features,
        open_minute=open_minute,
        close_minute=close_minute,
        exclude_weekends=bool(args.exclude_weekends),
        include_holidays=bool(args.include_known_us_equity_holidays),
    )
    sessions: list[SessionObservation] = []
    pos_by_ts = pd.Series(np.arange(len(features), dtype=int), index=features.index)
    for session_id, (session_date, part) in enumerate(features.loc[rth_mask].groupby("ny_date", sort=True), start=1):
        if len(part) < int(args.min_rth_bars):
            continue
        open_pos = int(pos_by_ts.loc[part.index[0]])
        close_pos = int(pos_by_ts.loc[part.index[-1]])
        open_price = float(part["open"].iloc[0])
        if not np.isfinite(open_price) or open_price <= 0:
            continue
        sessions.append(
            SessionObservation(
                session_date=str(session_date),
                session_id=session_id,
                open_pos=open_pos,
                close_pos=close_pos,
                open_time=part.index[0],
                close_time=part.index[-1],
                open_price=open_price,
            )
        )
    return sessions


def _session_feature_snapshot(features: pd.DataFrame, pos: int, prefix: str) -> dict[str, object]:
    cols = [
        "ret_15", "ret_30", "ret_60", "ret_240", "ret_1440", "range_pct_60", "range_pct_240", "range_pct_1440",
        "rv_60", "rv_240", "rv_1440", "atr_pct_60", "atr_pct_rel_1d", "volume_ratio_1d",
        "dist_ema_20", "dist_ema_50", "dist_ema_100", "dist_ema_200", "close_pos", "upper_wick_frac", "lower_wick_frac",
        "tb_delta_notional", "tb_delta_notional_sum_15", "tb_delta_notional_sum_60", "tb_large_delta_notional", "tb_large_delta_notional_sum_15", "tb_large_delta_notional_sum_60",
        "tb_taker_buy_ratio", "tb_delta_ratio", "tb_max_trade_notional", "tb_large_trades_count", "tb_large_trades_count_sum_15", "tb_large_trades_count_sum_60",
    ]
    row = features.iloc[pos]
    out: dict[str, object] = {}
    for col in cols:
        if col in features.columns:
            out[f"{prefix}{col}"] = row.get(col)
    # Include optional context columns without enumerating every range_pct code.
    for col in features.columns:
        if col.startswith(("rb", "fp_")) and any(token in col for token in ("_15m", "_30m", "_60m", "delta", "count")):
            out[f"{prefix}{col}"] = row.get(col)
    return out


def build_daily_labels_and_observations(features: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    tf_min = _timeframe_minutes(args.primary_timeframe)
    open_windows = _parse_number_list(args.open_windows_min, name="open_windows_min", value_type=int)
    sessions = _find_sessions(features, args)
    if not sessions:
        raise RuntimeError("No US RTH sessions found. Check timestamp offset/timeframe/date range.")

    daily_rows: list[dict[str, object]] = []
    obs_rows: list[dict[str, object]] = []
    reporter = ProgressReporter(label="[us-rth] sessions", total=len(sessions), every=max(1, len(sessions) // 20))
    for done, sess in enumerate(sessions, start=1):
        part = features.iloc[sess.open_pos : sess.close_pos + 1]
        highs = pd.to_numeric(part["high"], errors="coerce")
        lows = pd.to_numeric(part["low"], errors="coerce")
        closes = pd.to_numeric(part["close"], errors="coerce")
        open_price = float(sess.open_price)
        max_high = float(highs.max())
        min_low = float(lows.min())
        high_time = highs.idxmax()
        low_time = lows.idxmin()
        max_up = max_high / open_price - 1.0
        max_down = min_low / open_price - 1.0
        close_ret = float(closes.iloc[-1] / open_price - 1.0)
        first_extreme_side = "UP" if pd.Timestamp(high_time) < pd.Timestamp(low_time) else "DOWN"
        abs_down = abs(max_down)
        if max_up >= 0.010 and max_up >= abs_down * 1.5:
            label = "UP_EXPANSION"
        elif abs_down >= 0.010 and abs_down >= max_up * 1.5:
            label = "DOWN_EXPANSION"
        elif max_up >= 0.008 and abs_down >= 0.008 and abs(close_ret) <= 0.004:
            label = "TWO_SIDED_SWEEP"
        elif first_extreme_side == "UP" and close_ret < 0 and max_up >= 0.005:
            label = "UP_FAKEOUT_DOWN_CLOSE"
        elif first_extreme_side == "DOWN" and close_ret > 0 and abs_down >= 0.005:
            label = "DOWN_FAKEOUT_UP_CLOSE"
        else:
            label = "CHOP_OR_SMALL"
        pre_pos = max(0, sess.open_pos - 1)
        daily = {
            "session_date": sess.session_date,
            "session_id": sess.session_id,
            "open_time": sess.open_time,
            "close_time": sess.close_time,
            "open_price": open_price,
            "max_high": max_high,
            "min_low": min_low,
            "max_up_pct": max_up,
            "max_down_pct": max_down,
            "abs_max_down_pct": abs_down,
            "rth_close_return_pct": close_ret,
            "rth_range_pct": (max_high - min_low) / open_price,
            "time_to_max_up_min": (pd.Timestamp(high_time) - pd.Timestamp(sess.open_time)).total_seconds() / 60.0,
            "time_to_max_down_min": (pd.Timestamp(low_time) - pd.Timestamp(sess.open_time)).total_seconds() / 60.0,
            "first_extreme_side": first_extreme_side,
            "dominant_side": "UP" if max_up > abs_down else "DOWN" if abs_down > max_up else "BALANCED",
            "daily_event_group": label,
            **_session_feature_snapshot(features, pre_pos, "preopen_"),
        }
        daily_rows.append(daily)

        for w_min in open_windows:
            bars_needed = int(math.ceil(float(w_min) / float(tf_min)))
            sig_pos = sess.open_pos + bars_needed - 1
            if sig_pos >= sess.close_pos:
                continue
            sig = features.iloc[sig_pos]
            obs_part = features.iloc[sess.open_pos : sig_pos + 1]
            obs_high = float(pd.to_numeric(obs_part["high"], errors="coerce").max())
            obs_low = float(pd.to_numeric(obs_part["low"], errors="coerce").min())
            obs_close = float(sig["close"])
            obs_ret = obs_close / open_price - 1.0
            obs_up = obs_high / open_price - 1.0
            obs_down = obs_low / open_price - 1.0
            obs_range = (obs_high - obs_low) / open_price
            obs_close_pos = (obs_close - obs_low) / (obs_high - obs_low) if obs_high > obs_low else np.nan
            obs = {
                "session_date": sess.session_date,
                "session_id": sess.session_id,
                "open_window_min": int(w_min),
                "signal_time": features.index[sig_pos],
                "ny_signal_time": sig.get("ny_time"),
                "open_time": sess.open_time,
                "open_price": open_price,
                "open_window_return": obs_ret,
                "open_window_up_pct": obs_up,
                "open_window_down_pct": obs_down,
                "open_window_abs_down_pct": abs(obs_down),
                "open_window_range_pct": obs_range,
                "open_window_close_pos": obs_close_pos,
                "daily_event_group": label,
                "full_rth_max_up_pct_label": max_up,
                "full_rth_max_down_pct_label": max_down,
                **_session_feature_snapshot(features, sig_pos, "sig_"),
                **_session_feature_snapshot(features, pre_pos, "preopen_"),
            }
            obs_rows.append(obs)
        reporter.update(done)
    reporter.close()
    daily_df = pd.DataFrame(daily_rows)
    obs_df = pd.DataFrame(obs_rows)
    return daily_df, obs_df


def _direction_value(series: pd.Series | object) -> float:
    try:
        return float(series)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def build_event_table(observations: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame()
    open_thresholds = _parse_number_list(args.open_move_thresholds, name="open_move_thresholds", value_type=float)
    fakeout_thresholds = _parse_number_list(args.fakeout_thresholds, name="fakeout_thresholds", value_type=float)
    rows: list[dict[str, object]] = []
    reporter = ProgressReporter(label="[us-rth] event candidates", total=len(observations), every=max(1, len(observations) // 20))
    for done, row in enumerate(observations.itertuples(index=False), start=1):
        base = row._asdict()
        w = int(base["open_window_min"])
        open_ret = _direction_value(base.get("open_window_return"))
        up = _direction_value(base.get("open_window_up_pct"))
        down_abs = abs(_direction_value(base.get("open_window_down_pct")))
        close_pos = _direction_value(base.get("open_window_close_pos"))
        vol_ratio = _direction_value(base.get("sig_volume_ratio_1d"))
        atr_rel = _direction_value(base.get("preopen_atr_pct_rel_1d"))
        sig_tb_delta = _direction_value(base.get("sig_tb_delta_notional_sum_15"))
        sig_large_delta = _direction_value(base.get("sig_tb_large_delta_notional_sum_15"))

        def add_event(name: str, side: int, family: str, trigger: str) -> None:
            item = dict(base)
            item["event_name"] = f"open{w}m_{name}_{'LONG' if side > 0 else 'SHORT'}"
            item["event_family"] = family
            item["trigger"] = trigger
            item["side"] = int(side)
            rows.append(item)

        # Baselines for every observation.  These detect whether the time slot itself has directional bias.
        add_event("baseline", 1, "baseline", "slot_long")
        add_event("baseline", -1, "baseline", "slot_short")

        for thr in open_thresholds:
            tag = f"ret{int(thr * 10000):04d}bp"
            if np.isfinite(open_ret) and open_ret >= thr:
                add_event(f"continuation_{tag}", 1, "open_continuation", f"open_ret>={thr}")
            if np.isfinite(open_ret) and open_ret <= -thr:
                add_event(f"continuation_{tag}", -1, "open_continuation", f"open_ret<=-{thr}")
            if np.isfinite(open_ret) and np.isfinite(vol_ratio) and vol_ratio >= 1.5 and open_ret >= thr:
                add_event(f"vol_confirm_cont_{tag}", 1, "volume_confirmed_continuation", f"open_ret>={thr}&vol_ratio>=1.5")
            if np.isfinite(open_ret) and np.isfinite(vol_ratio) and vol_ratio >= 1.5 and open_ret <= -thr:
                add_event(f"vol_confirm_cont_{tag}", -1, "volume_confirmed_continuation", f"open_ret<=-{thr}&vol_ratio>=1.5")

        for thr in fakeout_thresholds:
            tag = f"sweep{int(thr * 10000):04d}bp"
            if np.isfinite(up) and np.isfinite(open_ret) and np.isfinite(close_pos) and up >= thr and open_ret <= thr * 0.25 and close_pos <= 0.45:
                add_event(f"up_fakeout_{tag}", -1, "opening_fakeout", f"up>={thr}&weak_close")
            if np.isfinite(down_abs) and np.isfinite(open_ret) and np.isfinite(close_pos) and down_abs >= thr and open_ret >= -thr * 0.25 and close_pos >= 0.55:
                add_event(f"down_fakeout_{tag}", 1, "opening_reclaim", f"down>={thr}&strong_close")

        if np.isfinite(atr_rel) and atr_rel <= 0.85:
            if np.isfinite(open_ret) and open_ret > 0:
                add_event("preopen_compression_up", 1, "compression_breakout", "preopen_atr_rel<=0.85&open_ret>0")
            if np.isfinite(open_ret) and open_ret < 0:
                add_event("preopen_compression_down", -1, "compression_breakout", "preopen_atr_rel<=0.85&open_ret<0")

        if np.isfinite(sig_tb_delta):
            if sig_tb_delta > 0 and np.isfinite(open_ret) and open_ret >= 0:
                add_event("tb_delta_confirm", 1, "tradebar_orderflow_confirm", "tb_delta_15m>0&open_ret>=0")
            if sig_tb_delta < 0 and np.isfinite(open_ret) and open_ret <= 0:
                add_event("tb_delta_confirm", -1, "tradebar_orderflow_confirm", "tb_delta_15m<0&open_ret<=0")
        if np.isfinite(sig_large_delta):
            if sig_large_delta > 0 and np.isfinite(open_ret) and open_ret >= 0:
                add_event("large_trade_confirm", 1, "large_trade_orderflow_confirm", "large_delta_15m>0&open_ret>=0")
            if sig_large_delta < 0 and np.isfinite(open_ret) and open_ret <= 0:
                add_event("large_trade_confirm", -1, "large_trade_orderflow_confirm", "large_delta_15m<0&open_ret<=0")

        # Generic range/footprint context hooks: if optional columns exist, use directional 15m sums.
        for key, family in (("rb", "rangebar_confirm"), ("fp_", "footprint_confirm")):
            directional_cols = [c for c in observations.columns if c.startswith(key) and "delta" in c and "15m" in c]
            for c in directional_cols[:8]:  # cap to avoid exploding event names if many ranges are enabled.
                v = _direction_value(base.get(c))
                if not np.isfinite(v) or abs(v) <= 0:
                    continue
                short_col = c.replace("sig_", "")[-48:].replace(".", "p")
                if v > 0 and np.isfinite(open_ret) and open_ret >= 0:
                    add_event(f"{short_col}_confirm", 1, family, f"{c}>0&open_ret>=0")
                elif v < 0 and np.isfinite(open_ret) and open_ret <= 0:
                    add_event(f"{short_col}_confirm", -1, family, f"{c}<0&open_ret<=0")
        reporter.update(done)
    reporter.close()
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events["open_return_bucket"] = fixed_threshold_labels(
        events["open_window_return"],
        thresholds=(-0.005, -0.003, -0.001, 0.001, 0.003, 0.005),
        labels=("RET_STRONG_DOWN", "RET_MED_DOWN", "RET_SMALL_DOWN", "RET_FLAT", "RET_SMALL_UP", "RET_MED_UP", "RET_STRONG_UP"),
    )
    events["preopen_atr_rel_bucket"] = fixed_threshold_labels(
        events.get("preopen_atr_pct_rel_1d", pd.Series(index=events.index, dtype=float)),
        thresholds=(0.70, 0.85, 1.00, 1.20),
        labels=("ATR_COMPRESS_STRONG", "ATR_COMPRESS", "ATR_NORMAL_LOW", "ATR_NORMAL_HIGH", "ATR_EXPANDED"),
    )
    events["preopen_ret_240_q"] = qcut_labels(events.get("preopen_ret_240", pd.Series(index=events.index, dtype=float)), q=4, prefix="PRE240_Q")
    return events


def build_daily_feature_profiles(daily: pd.DataFrame, big_move_pcts: Iterable[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if daily.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, object]] = []
    feature_cols = [
        c
        for c in daily.columns
        if c.startswith("preopen_") and pd.api.types.is_numeric_dtype(pd.to_numeric(daily[c], errors="coerce"))
    ]
    groups = ["daily_event_group", "dominant_side", "first_extreme_side"]
    for group_col in groups:
        if group_col not in daily.columns:
            continue
        for key, part in daily.groupby(group_col, dropna=False):
            row: dict[str, object] = {"group_col": group_col, "group_value": key, "count": int(len(part))}
            row["avg_max_up_pct"] = float(pd.to_numeric(part["max_up_pct"], errors="coerce").mean())
            row["avg_abs_max_down_pct"] = float(pd.to_numeric(part["abs_max_down_pct"], errors="coerce").mean())
            row["avg_rth_close_return_pct"] = float(pd.to_numeric(part["rth_close_return_pct"], errors="coerce").mean())
            for col in feature_cols[:80]:
                row[f"mean_{col}"] = float(pd.to_numeric(part[col], errors="coerce").mean())
                row[f"median_{col}"] = float(pd.to_numeric(part[col], errors="coerce").median())
            rows.append(row)
    profile = pd.DataFrame(rows)

    threshold_rows: list[dict[str, object]] = []
    for pct in big_move_pcts:
        up_mask = pd.to_numeric(daily["max_up_pct"], errors="coerce") >= float(pct)
        down_mask = pd.to_numeric(daily["abs_max_down_pct"], errors="coerce") >= float(pct)
        for label, mask in ((f"UP_GE_{pct:.4f}", up_mask), (f"DOWN_GE_{pct:.4f}", down_mask)):
            true_part = daily.loc[mask]
            false_part = daily.loc[~mask]
            for col in feature_cols[:120]:
                true_mean = float(pd.to_numeric(true_part[col], errors="coerce").mean()) if not true_part.empty else np.nan
                false_mean = float(pd.to_numeric(false_part[col], errors="coerce").mean()) if not false_part.empty else np.nan
                threshold_rows.append(
                    {
                        "label": label,
                        "feature": col,
                        "true_count": int(mask.sum()),
                        "false_count": int((~mask).sum()),
                        "true_mean": true_mean,
                        "false_mean": false_mean,
                        "mean_diff": true_mean - false_mean if np.isfinite(true_mean) and np.isfinite(false_mean) else np.nan,
                    }
                )
    contrasts = pd.DataFrame(threshold_rows)
    if not contrasts.empty:
        contrasts["abs_mean_diff"] = pd.to_numeric(contrasts["mean_diff"], errors="coerce").abs()
        contrasts = contrasts.sort_values(["label", "abs_mean_diff"], ascending=[True, False])
    return profile, contrasts


def _metric_float(value: object) -> float:
    if isinstance(value, str) and value == "inf":
        return float("inf")
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def build_candidate_rank(events: pd.DataFrame, stats: pd.DataFrame, yearly: pd.DataFrame, *, return_col: str, min_count: int, max_top5_share: float) -> pd.DataFrame:
    if events.empty or stats.empty:
        return pd.DataFrame()
    stat = stats[stats["metric"] == return_col].copy()
    if stat.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, row in stat.iterrows():
        event_name = row.get("event_name")
        sub = events.loc[events["event_name"] == event_name, return_col]
        yearly_sub = yearly[(yearly.get("event_name") == event_name) & (yearly.get("metric") == return_col)].copy()
        eligible = yearly_sub[yearly_sub.get("eligible", False).astype(bool)] if not yearly_sub.empty and "eligible" in yearly_sub.columns else yearly_sub
        positive_years = int((pd.to_numeric(eligible.get("mean", pd.Series(dtype=float)), errors="coerce") > 0).sum()) if not eligible.empty else 0
        tested_years = int(len(eligible))
        yearly_positive_rate = positive_years / tested_years if tested_years > 0 else np.nan
        top5 = top_winner_dependency(sub, top_n=5)
        count = int(row.get("count", 0) or 0)
        mean = _metric_float(row.get("mean"))
        median = _metric_float(row.get("median"))
        pf = _metric_float(row.get("profit_factor"))
        win_rate = _metric_float(row.get("win_rate"))
        candidate = bool(
            count >= int(min_count)
            and np.isfinite(mean)
            and mean > 0
            and np.isfinite(median)
            and median > -0.001
            and np.isfinite(pf)
            and pf > 1.08
            and (not np.isfinite(top5) or top5 <= float(max_top5_share))
            and (not np.isfinite(yearly_positive_rate) or yearly_positive_rate >= 0.50)
        )
        rows.append(
            {
                "event_name": event_name,
                "event_family": row.get("event_family"),
                "open_window_min": row.get("open_window_min"),
                "side_name": row.get("side_name"),
                "metric": return_col,
                "count": count,
                "mean": mean,
                "median": median,
                "win_rate": win_rate,
                "profit_factor": pf,
                "payoff_ratio": _metric_float(row.get("payoff_ratio")),
                "top5_winner_share_recalc": top5,
                "tested_years": tested_years,
                "positive_years": positive_years,
                "yearly_positive_rate": yearly_positive_rate,
                "candidate_flag": candidate,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank_score"] = (
        pd.to_numeric(out["mean"], errors="coerce").fillna(-999.0) * 1000.0
        + (pd.to_numeric(out["profit_factor"], errors="coerce").replace(np.inf, 10.0).fillna(0.0) - 1.0)
        + (pd.to_numeric(out["win_rate"], errors="coerce").fillna(0.0) - 0.5)
        + pd.to_numeric(out["yearly_positive_rate"], errors="coerce").fillna(0.0)
    )
    return out.sort_values(["candidate_flag", "rank_score", "count"], ascending=[False, False, False])


def write_markdown_brief(out_dir: Path, meta: dict[str, object], candidate_rank: pd.DataFrame) -> None:
    lines = [
        "# ETH US RTH Opening Event Study",
        "",
        "This is a discovery-only event study. Full RTH max-up/max-down labels are not used as entry rules.",
        "",
        "## Meta",
        "",
    ]
    for k, v in meta.items():
        if k == "best_candidates":
            continue
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Top candidate phenomena")
    lines.append("")
    if candidate_rank.empty:
        lines.append("No candidate phenomena passed the conservative rank filter.")
    else:
        cols = ["event_name", "count", "mean", "median", "win_rate", "profit_factor", "yearly_positive_rate", "candidate_flag"]
        lines.append("```text")
        lines.append(candidate_rank[cols].head(20).to_string(index=False))
        lines.append("```")
    lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append("Only promote rows that survive formal strategy replay, next-open audit, fee/slippage/delay stress, parameter-neighbourhood checks, and yearly stability checks.")
    (out_dir / "20_research_brief.md").write_text("\n".join(lines), encoding="utf-8")


def run_research(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    offset_hours = float(args.timestamp_offset_hours) if args.timestamp_offset_hours is not None else _parse_timezone_offset_hours(TIMEZONE)
    horizons = _parse_number_list(args.horizons, name="horizons", value_type=int)
    if int(args.candidate_horizon) not in horizons:
        horizons = sorted(set([*horizons, int(args.candidate_horizon)]))

    if args.tradebar_timeframe is None:
        args.tradebar_timeframe = args.primary_timeframe

    print(f"[1/7] load primary trade bars {args.symbol} {args.primary_timeframe} {args.warmup_start_date}->{args.end_date}", flush=True)
    bars = _load_primary_trade_bars(args)
    print(f"      rows={len(bars):,} range={bars.index[0]} -> {bars.index[-1]}", flush=True)

    print("[2/7] build primary closed-bar features + NY session clock", flush=True)
    features = build_primary_features(bars, offset_hours=offset_hours)

    if bool(args.include_trade_bars) or str(args.tradebar_timeframe) != str(args.primary_timeframe):
        features = attach_trade_bar_context(features, args)
    if bool(args.include_range_bars):
        features = attach_range_bar_context(features, args)
    if bool(args.include_footprint):
        features = attach_footprint_context(features, args)

    print("[3/7] build daily RTH labels and open-window observations", flush=True)
    daily, observations = build_daily_labels_and_observations(features, args)
    eval_start = pd.Timestamp(args.start_date)
    eval_end = pd.Timestamp(args.end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    daily = daily.loc[(pd.to_datetime(daily["open_time"]) >= eval_start) & (pd.to_datetime(daily["open_time"]) <= eval_end)].copy()
    observations = observations.loc[(pd.to_datetime(observations["signal_time"]) >= eval_start) & (pd.to_datetime(observations["signal_time"]) <= eval_end)].copy()
    print(f"      sessions={len(daily):,} observations={len(observations):,}", flush=True)

    print("[4/7] build event rows from causal observations", flush=True)
    events = build_event_table(observations, args)
    if events.empty:
        raise RuntimeError("No events generated from observations.")
    print(f"      events={len(events):,} event_names={events['event_name'].nunique():,}", flush=True)

    print("[5/7] run event-study core: next-open returns, MFE/MAE, causal audit", flush=True)
    eval_bars = bars.loc[eval_start:eval_end].copy()
    cfg = EventStudyConfig(
        horizons=tuple(int(h) for h in horizons),
        mfe_mae_horizon=int(args.mfe_mae_horizon),
        cost=CostConfig(
            entry_fee_rate=float(args.entry_fee_rate),
            exit_fee_rate=float(args.exit_fee_rate),
            entry_slippage_pct=float(args.entry_slippage_pct),
            exit_slippage_pct=float(args.exit_slippage_pct),
        ),
        min_count=int(args.min_count),
        progress_every=int(args.progress_every),
    )
    result = run_event_study(eval_bars, events, cfg)
    result.write(out_dir)

    print("[6/7] write daily labels, grouped stats, feature profiles, candidate rank", flush=True)
    daily.to_csv(out_dir / "00_daily_rth_labels.csv", index=False)
    observations.to_csv(out_dir / "00_open_window_observations.csv", index=False)
    big_move_pcts = _parse_number_list(args.big_move_pcts, name="big_move_pcts", value_type=float)
    profile, contrasts = build_daily_feature_profiles(daily, big_move_pcts)
    profile.to_csv(out_dir / "06_daily_group_feature_profile.csv", index=False)
    contrasts.to_csv(out_dir / "07_daily_big_move_feature_contrast.csv", index=False)

    return_cols = [f"next_open_ret_h{int(h)}_net" for h in horizons]
    primary_return_col = f"next_open_ret_h{int(args.candidate_horizon)}_net"
    event_name_stats = summarize_many(result.events, return_cols, group_cols=["event_name", "event_family", "open_window_min", "side_name"], min_count=int(args.min_count))
    event_family_stats = summarize_many(result.events, return_cols, group_cols=["event_family", "open_window_min", "side_name"], min_count=int(args.min_count))
    event_yearly_stats = summarize_many(result.events, return_cols, group_cols=["event_name", "year"], min_count=max(10, int(args.min_count) // 3))
    bin_stats = summarize_many(
        result.events,
        [primary_return_col],
        group_cols=["event_family", "open_window_min", "open_return_bucket", "preopen_atr_rel_bucket", "preopen_ret_240_q", "side_name"],
        min_count=max(10, int(args.min_count) // 3),
    )
    candidate_rank = build_candidate_rank(
        result.events,
        event_name_stats,
        event_yearly_stats,
        return_col=primary_return_col,
        min_count=int(args.min_count),
        max_top5_share=float(args.max_top5_winner_share),
    )
    event_name_stats.to_csv(out_dir / "11_event_name_stats.csv", index=False)
    event_family_stats.to_csv(out_dir / "12_event_family_stats.csv", index=False)
    event_yearly_stats.to_csv(out_dir / "13_event_name_yearly_stats.csv", index=False)
    bin_stats.to_csv(out_dir / "14_event_feature_bin_stats.csv", index=False)
    candidate_rank.to_csv(out_dir / "15_candidate_rank.csv", index=False)

    if int(args.save_feature_sample) > 0:
        sample_cols = [c for c in features.columns if not c in {"ny_time", "utc_time"}]
        features.loc[eval_start:eval_end, sample_cols].tail(int(args.save_feature_sample)).to_csv(out_dir / "16_feature_tail_sample.csv")

    best = candidate_rank.head(20).copy() if not candidate_rank.empty else pd.DataFrame()
    meta: dict[str, object] = {
        "symbol": args.symbol,
        "primary_timeframe": args.primary_timeframe,
        "primary_data_source": "OKXTradeBarLoader",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "timestamp_offset_hours": offset_hours,
        "ny_open": args.ny_open,
        "ny_close": args.ny_close,
        "open_windows_min": _parse_number_list(args.open_windows_min, name="open_windows_min", value_type=int),
        "horizons": horizons,
        "candidate_horizon": int(args.candidate_horizon),
        "round_trip_cost_pct": float(cfg.cost.round_trip_cost_pct),
        "primary_rows": int(len(bars)),
        "eval_rows": int(len(eval_bars)),
        "sessions": int(len(daily)),
        "observations": int(len(observations)),
        "events": int(len(result.events)),
        "event_names": int(result.events["event_name"].nunique()) if not result.events.empty else 0,
        "causal_fail_count": int(result.causal_audit["causal_fail_flag"].sum()) if not result.causal_audit.empty else 0,
        "include_trade_bars": bool(args.include_trade_bars),
        "tradebar_timeframe": args.tradebar_timeframe,
        "include_range_bars": bool(args.include_range_bars),
        "include_footprint": bool(args.include_footprint),
        "best_candidates": best.to_dict(orient="records"),
        "notes": "Discovery-only event study. Full RTH max-up/max-down are labels, not entry features.",
    }
    with (out_dir / "19_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    write_markdown_brief(out_dir, meta, candidate_rank)

    print("[7/7] done", flush=True)
    if not best.empty:
        cols = ["event_name", "count", "mean", "median", "win_rate", "profit_factor", "candidate_flag"]
        print("\nTop candidate phenomena:", flush=True)
        print(best[cols].head(10).to_string(index=False), flush=True)
    print(f"\nReport written to: {out_dir}", flush=True)
    return meta


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_research(args)


if __name__ == "__main__":
    main()
