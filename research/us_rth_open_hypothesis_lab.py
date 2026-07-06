#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""US RTH open multi-hypothesis event lab for ETH OKX trade bars.

This is intentionally not a parameter-grid optimizer.  It tests a broad set of
structural hypotheses around the US equity regular session open using:

- trade-bar primary OHLCV/order-flow as the execution axis;
- optional range-bar context;
- optional range-footprint context;
- closed-window signal timing and next-primary-bar-open execution.

The goal is to discover *directions* worth deeper validation, not to declare a
live strategy from a first scan.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import date, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.research_common.progress import ProgressReporter

try:
    from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader
except Exception:  # pragma: no cover - optional local module
    OKXRangeBarLoader = None  # type: ignore

try:
    from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader
except Exception:  # pragma: no cover - optional local module
    OKXRangeFootprintLoader = None  # type: ignore

NY_TZ = "America/New_York"
EPS = 1e-12


@dataclass(frozen=True)
class EventSpec:
    family: str
    name: str
    side: str
    signal_time: pd.Timestamp
    reason: str
    features: dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETH US RTH open multi-hypothesis event lab")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--primary-timeframe", default="1m", help="Primary bars must be OKX trade bars; default 1m.")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--timestamp-tz", default="+8", choices=["+8", "UTC"], help="Timezone of naive trade-bar timestamps returned by the loader.")
    p.add_argument("--out-dir", default="data/reports/research/us_rth_open_hypothesis_lab")
    p.add_argument("--horizons", default="15,30,60,120,240,390", help="Outcome horizons in minutes after next-open entry.")
    p.add_argument("--fee-rate", type=float, default=0.0011, help="Round-trip fee deducted from all event returns. Default 0.11%.")
    p.add_argument("--min-count", type=int, default=40)
    p.add_argument("--min-year-count", type=int, default=8)
    p.add_argument("--include-range-bars", action="store_true")
    p.add_argument("--range-pcts", default="0.0015,0.0020,0.0025")
    p.add_argument("--include-footprint", action="store_true")
    p.add_argument("--footprint-range-pct", type=float, default=0.0020)
    p.add_argument("--footprint-price-step", type=float, default=1.0)
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--hypothesis-mode", default="broad", choices=["classic", "broad"], help="classic keeps the first handcrafted set; broad adds a 1000+ structural hypothesis catalog.")
    p.add_argument("--max-context-tags-per-trigger", type=int, default=12, help="Cap active context combinations per base trigger to keep raw event volume manageable; catalog still contains 1000+ definitions.")
    p.add_argument("--write-full-events", action="store_true", help="Write the full 01_hypothesis_events.csv. Default writes a lightweight sample to avoid huge CSV stalls.")
    p.add_argument("--keep-event-features", action="store_true", help="Keep all feature columns through outcome/stat stages. Default compacts events after writing a small feature sample for speed/memory.")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def parse_csv_floats(text: str) -> list[float]:
    out: list[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def parse_csv_ints(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        for c in ("timestamp", "datetime", "time", "start_ts"):
            if c in out.columns:
                out[c] = pd.to_datetime(out[c], errors="coerce")
                out = out.set_index(c)
                break
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def timeframe_delta(tf: str) -> pd.Timedelta:
    text = str(tf).strip()
    unit = text[-1]
    n = int(text[:-1]) if text[:-1] else 1
    if unit == "s":
        return pd.Timedelta(seconds=n)
    if unit == "m":
        return pd.Timedelta(minutes=n)
    if unit in {"H", "h"}:
        return pd.Timedelta(hours=n)
    if unit in {"D", "d"}:
        return pd.Timedelta(days=n)
    raise ValueError(f"Unsupported timeframe: {tf}")


def add_time_columns(df: pd.DataFrame, *, timestamp_tz: str) -> pd.DataFrame:
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        utc = idx.tz_convert("UTC")
    else:
        if timestamp_tz == "UTC":
            utc = idx.tz_localize("UTC")
        else:
            utc = idx.tz_localize(timezone(timedelta(hours=8))).tz_convert("UTC")
    ny = utc.tz_convert(NY_TZ)
    out["utc_time"] = utc
    out["ny_time"] = ny
    out["ny_date"] = ny.date
    out["ny_minute"] = ny.hour * 60 + ny.minute
    return out


def pct_change(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < EPS:
        return float("nan")
    return float(a / b - 1.0)


def safe_div(a: Any, b: Any) -> Any:
    return np.asarray(a, dtype=float) / np.where(np.abs(np.asarray(b, dtype=float)) < EPS, np.nan, np.asarray(b, dtype=float))


def zscore_past(s: pd.Series, window: int, minp: int | None = None) -> pd.Series:
    if minp is None:
        minp = max(20, window // 4)
    mean = s.rolling(window, min_periods=minp).mean().shift(1)
    std = s.rolling(window, min_periods=minp).std(ddof=0).shift(1)
    return (s - mean) / std.replace(0, np.nan)


def rolling_quantile_past(s: pd.Series, window: int, q: float, minp: int | None = None) -> pd.Series:
    if minp is None:
        minp = max(20, window // 4)
    return s.rolling(window, min_periods=minp).quantile(q).shift(1)


def build_primary_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            raise ValueError(f"primary trade bars missing required column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ret_1"] = df["close"].pct_change()
    for n in (5, 15, 30, 60, 120, 240, 390, 1440):
        df[f"ret_{n}"] = df["close"].pct_change(n)
        df[f"range_{n}"] = df["high"].rolling(n, min_periods=max(3, n // 4)).max() / df["low"].rolling(n, min_periods=max(3, n // 4)).min() - 1.0

    df["bar_range"] = safe_div(df["high"] - df["low"], df["open"])
    df["body_pct"] = safe_div((df["close"] - df["open"]).abs(), df["open"])
    df["close_pos"] = safe_div(df["close"] - df["low"], df["high"] - df["low"]).clip(0, 1)
    df["upper_wick_pct"] = safe_div(df["high"] - df[["open", "close"]].max(axis=1), df["open"])
    df["lower_wick_pct"] = safe_div(df[["open", "close"]].min(axis=1) - df["low"], df["open"])

    for span in (20, 60, 240, 1440):
        ema = df["close"].ewm(span=span, adjust=False, min_periods=max(5, span // 5)).mean()
        df[f"ema_{span}"] = ema
        df[f"dist_ema_{span}"] = df["close"] / ema - 1.0

    if "notional" not in df.columns:
        df["notional"] = df["volume"] * df["close"]
    df["log_notional"] = np.log1p(pd.to_numeric(df["notional"], errors="coerce"))
    for n in (15, 60, 240, 1440):
        df[f"notional_sum_{n}"] = df["notional"].rolling(n, min_periods=max(3, n // 4)).sum()
        df[f"vol_z_{n}"] = zscore_past(df["log_notional"], n * 5 if n < 240 else n)

    # Trade-bar order-flow fields. Missing columns are filled with zeros so the
    # script remains compatible with older caches.
    for c in [
        "delta_notional",
        "cvd_notional",
        "taker_buy_ratio",
        "large_delta_notional",
        "large_trades_count",
        "max_trade_notional",
        "trades_count",
    ]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for n in (5, 15, 30, 60, 120, 240):
        df[f"tb_delta_sum_{n}"] = df["delta_notional"].rolling(n, min_periods=max(2, n // 4)).sum()
        df[f"tb_large_delta_sum_{n}"] = df["large_delta_notional"].rolling(n, min_periods=max(2, n // 4)).sum()
        df[f"tb_large_count_sum_{n}"] = df["large_trades_count"].rolling(n, min_periods=max(2, n // 4)).sum()
        df[f"tb_delta_z_{n}"] = zscore_past(df[f"tb_delta_sum_{n}"], max(240, n * 8))
        df[f"tb_large_delta_z_{n}"] = zscore_past(df[f"tb_large_delta_sum_{n}"], max(240, n * 8))

    df["rv_60"] = df["ret_1"].rolling(60, min_periods=30).std(ddof=0) * math.sqrt(60)
    df["rv_240"] = df["ret_1"].rolling(240, min_periods=120).std(ddof=0) * math.sqrt(240)
    df["rv_ratio_60_240"] = df["rv_60"] / df["rv_240"].replace(0, np.nan)
    df["compression_60_1440"] = df["range_60"] / df["range_1440"].replace(0, np.nan)
    return df


def load_primary(args: argparse.Namespace) -> pd.DataFrame:
    loader = OKXTradeBarLoader(symbol=args.symbol, timeframe=args.primary_timeframe)
    df = loader.fetch_data_by_date_range(
        args.warmup_start_date,
        args.end_date,
        chunksize=args.chunksize,
        cvd_mode="range",
    )
    df = ensure_dt_index(df)
    if df.empty:
        raise RuntimeError("No primary trade bars loaded")
    print(f"[load] primary trade bars rows={len(df):,} range={df.index.min()} -> {df.index.max()}")
    return df


def load_range_context(args: argparse.Namespace, primary: pd.DataFrame) -> dict[float, pd.DataFrame]:
    ctx: dict[float, pd.DataFrame] = {}
    if not args.include_range_bars:
        return ctx
    if OKXRangeBarLoader is None:
        print("[range] loader unavailable; skip range context")
        return ctx
    for rp in parse_csv_floats(args.range_pcts):
        loader = OKXRangeBarLoader(symbol=args.symbol, range_pct=rp)
        rb = loader.fetch_data_by_date_range(args.warmup_start_date, args.end_date, chunksize=args.chunksize, cvd_mode="range")
        rb = ensure_dt_index(rb)
        if rb.empty:
            print(f"[range] range_pct={rp} empty")
            continue
        if "end_ts" in rb.columns:
            rb["available_time"] = pd.to_datetime(rb["end_ts"], errors="coerce")
        else:
            rb["available_time"] = rb.index
        rb = rb.dropna(subset=["available_time"]).sort_values("available_time")
        rb = rb.set_index("available_time", drop=False)
        ctx[rp] = build_range_rolling_context(rb, rp)
        print(f"[range] range_pct={rp:.4f} rows={len(rb):,}")
    return ctx


def build_range_rolling_context(rb: pd.DataFrame, range_pct: float) -> pd.DataFrame:
    out = pd.DataFrame(index=rb.index)
    prefix = f"rb{int(round(range_pct * 10000)):04d}"
    direction = pd.to_numeric(rb.get("direction", 0), errors="coerce").fillna(0.0)
    delta = pd.to_numeric(rb.get("delta_notional", 0), errors="coerce").fillna(0.0)
    notional = pd.to_numeric(rb.get("notional", 0), errors="coerce").fillna(0.0)
    duration = pd.to_numeric(rb.get("duration_seconds", np.nan), errors="coerce")
    for win in ("15min", "30min", "60min"):
        tag = win.replace("min", "")
        out[f"{prefix}_count_{tag}"] = direction.rolling(win).count()
        out[f"{prefix}_dir_sum_{tag}"] = direction.rolling(win).sum()
        out[f"{prefix}_delta_sum_{tag}"] = delta.rolling(win).sum()
        out[f"{prefix}_notional_sum_{tag}"] = notional.rolling(win).sum()
        out[f"{prefix}_duration_med_{tag}"] = duration.rolling(win).median()
    out[f"{prefix}_last_dir"] = direction
    out[f"{prefix}_last_delta"] = delta
    return out


def load_footprint_context(args: argparse.Namespace) -> pd.DataFrame:
    if not args.include_footprint:
        return pd.DataFrame()
    if OKXRangeFootprintLoader is None:
        print("[footprint] loader unavailable; skip footprint context")
        return pd.DataFrame()
    loader = OKXRangeFootprintLoader(
        symbol=args.symbol,
        range_pct=args.footprint_range_pct,
        price_step=args.footprint_price_step,
    )
    fp = loader.fetch_data_by_date_range(args.warmup_start_date, args.end_date, chunksize=args.chunksize)
    fp = ensure_dt_index(fp)
    if fp.empty:
        print("[footprint] empty")
        return pd.DataFrame()
    ctx = build_footprint_context(fp, args.footprint_range_pct)
    print(f"[footprint] rows={len(fp):,} bars={ctx['bar_id'].nunique() if 'bar_id' in ctx else 'n/a'}")
    return ctx


def build_footprint_context(fp: pd.DataFrame, range_pct: float) -> pd.DataFrame:
    df = fp.copy()
    for c in ["bar_id", "price_bucket", "delta_notional", "large_delta_notional", "notional"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "end_ts" in df.columns:
        df["available_time"] = pd.to_datetime(df["end_ts"], errors="coerce")
    else:
        df["available_time"] = df.index

    rows: list[dict[str, Any]] = []
    for bar_id, g in df.groupby("bar_id", sort=False):
        if g.empty:
            continue
        g2 = g.sort_values("price_bucket")
        n = len(g2)
        lo = max(1, int(math.ceil(n * 0.2)))
        hi = max(1, int(math.ceil(n * 0.2)))
        bottom = g2.head(lo)
        top = g2.tail(hi)
        delta_sum = float(g2["delta_notional"].sum())
        abs_delta = float(g2["delta_notional"].abs().sum())
        rows.append(
            {
                "bar_id": bar_id,
                "available_time": g2["available_time"].max(),
                "fp_delta_sum": delta_sum,
                "fp_abs_delta_sum": abs_delta,
                "fp_delta_concentration": float(g2["delta_notional"].abs().max() / abs_delta) if abs_delta > EPS else np.nan,
                "fp_top_delta": float(top["delta_notional"].sum()),
                "fp_bottom_delta": float(bottom["delta_notional"].sum()),
                "fp_large_delta_sum": float(g2["large_delta_notional"].sum()),
                "fp_notional_sum": float(g2["notional"].sum()),
            }
        )
    out = pd.DataFrame(rows).dropna(subset=["available_time"]).sort_values("available_time")
    if out.empty:
        return out
    out["range_pct"] = float(range_pct)
    out = out.set_index("available_time", drop=False)
    prefix = f"fp{int(round(range_pct * 10000)):04d}"
    for win in ("15min", "30min", "60min"):
        tag = win.replace("min", "")
        out[f"{prefix}_delta_sum_{tag}"] = out["fp_delta_sum"].rolling(win).sum()
        out[f"{prefix}_top_delta_sum_{tag}"] = out["fp_top_delta"].rolling(win).sum()
        out[f"{prefix}_bottom_delta_sum_{tag}"] = out["fp_bottom_delta"].rolling(win).sum()
        out[f"{prefix}_large_delta_sum_{tag}"] = out["fp_large_delta_sum"].rolling(win).sum()
        out[f"{prefix}_abs_delta_sum_{tag}"] = out["fp_abs_delta_sum"].rolling(win).sum()
    return out


def lookup_context(ctx: pd.DataFrame, signal_time: pd.Timestamp) -> dict[str, float]:
    if ctx is None or ctx.empty:
        return {}
    idx = ctx.index
    pos = idx.searchsorted(signal_time, side="right") - 1
    if pos < 0:
        return {}
    row = ctx.iloc[pos]
    return {k: float(v) for k, v in row.items() if isinstance(v, (int, float, np.floating)) and np.isfinite(v)}


def session_slice(day_df: pd.DataFrame, start_min: int, end_min: int, *, include_end: bool = False) -> pd.DataFrame:
    if include_end:
        return day_df[(day_df["ny_minute"] >= start_min) & (day_df["ny_minute"] <= end_min)]
    return day_df[(day_df["ny_minute"] >= start_min) & (day_df["ny_minute"] < end_min)]


def window_stats(w: pd.DataFrame) -> dict[str, float]:
    if w.empty:
        return {}
    o = float(w["open"].iloc[0])
    c = float(w["close"].iloc[-1])
    hi = float(w["high"].max())
    lo = float(w["low"].min())
    ret = pct_change(c, o)
    rng = pct_change(hi, lo)
    close_pos = (c - lo) / (hi - lo) if hi > lo else np.nan
    return {
        "open": o,
        "close": c,
        "high": hi,
        "low": lo,
        "ret": ret,
        "range": rng,
        "close_pos": float(close_pos) if np.isfinite(close_pos) else np.nan,
        "notional": float(w.get("notional", pd.Series(index=w.index, data=0.0)).sum()),
        "delta": float(w.get("delta_notional", pd.Series(index=w.index, data=0.0)).sum()),
        "large_delta": float(w.get("large_delta_notional", pd.Series(index=w.index, data=0.0)).sum()),
        "large_count": float(w.get("large_trades_count", pd.Series(index=w.index, data=0.0)).sum()),
        "max_trade_notional": float(w.get("max_trade_notional", pd.Series(index=w.index, data=0.0)).max()),
        "upper_wick": pct_change(hi, max(o, c)),
        "lower_wick": pct_change(min(o, c), lo),
    }


def prefix_dict(prefix: str, d: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in d.items()}


def is_finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def fval(d: dict[str, Any], key: str, default: float = np.nan) -> float:
    try:
        v = d.get(key, default)
        return float(v) if np.isfinite(float(v)) else default
    except Exception:
        return default


def side_suffix(side: str) -> str:
    return "long" if side == "LONG" else "short"


def build_hypothesis_catalog(range_ctx: dict[float, pd.DataFrame], footprint_ctx: pd.DataFrame) -> pd.DataFrame:
    """Return the broad hypothesis design space.

    This is a catalog of structural hypotheses, not a parameter grid.  The axes
    are trading-logic dimensions: opening phase, trigger family, reference
    session, order-flow confirmation/divergence, volatility regime, range-bar
    context and footprint context.  Actual events are generated only when the
    structural trigger is true on a day.
    """
    windows = [5, 10, 15, 30, 45, 60, 90, 120]
    refs = ["pre30", "pre60", "pre120", "pre240", "europe", "asia"]
    base_templates: list[dict[str, Any]] = []

    for n in windows:
        for side in ["LONG", "SHORT"]:
            ss = side_suffix(side)
            base_templates.extend([
                {"window": n, "family": "open_impulse", "base": f"open{n}_impulse_close_extreme_{ss}", "side": side},
                {"window": n, "family": "open_impulse_fade", "base": f"open{n}_impulse_exhaustion_fade_{ss}", "side": side},
                {"window": n, "family": "open_range_expansion", "base": f"open{n}_range_expansion_{ss}", "side": side},
                {"window": n, "family": "open_body_efficiency", "base": f"open{n}_efficient_body_{ss}", "side": side},
                {"window": n, "family": "open_wick_reversal", "base": f"open{n}_wick_reversal_{ss}", "side": side},
                {"window": n, "family": "open_orderflow_align", "base": f"open{n}_price_delta_large_align_{ss}", "side": side},
                {"window": n, "family": "open_orderflow_diverge", "base": f"open{n}_price_delta_diverge_fade_{ss}", "side": side},
            ])
        for ref in refs:
            for side in ["LONG", "SHORT"]:
                ss = side_suffix(side)
                base_templates.extend([
                    {"window": n, "family": "cross_session_continue", "base": f"{ref}_open{n}_continue_{ss}", "side": side, "ref": ref},
                    {"window": n, "family": "cross_session_fail", "base": f"{ref}_open{n}_fail_{ss}", "side": side, "ref": ref},
                    {"window": n, "family": "session_level_sweep", "base": f"{ref}_level_sweep_reject_reclaim_open{n}_{ss}", "side": side, "ref": ref},
                    {"window": n, "family": "session_level_breakout", "base": f"{ref}_level_breakout_hold_open{n}_{ss}", "side": side, "ref": ref},
                ])

    context_templates: list[dict[str, Any]] = []
    for ref in refs:
        context_templates.extend([
            {"ctx_family": "context_pretrend", "ctx": f"ctx_{ref}_trend_aligned"},
            {"ctx_family": "context_pretrend", "ctx": f"ctx_{ref}_trend_opposed"},
            {"ctx_family": "context_range", "ctx": f"ctx_{ref}_compressed"},
            {"ctx_family": "context_range", "ctx": f"ctx_{ref}_expanded"},
            {"ctx_family": "context_orderflow", "ctx": f"ctx_{ref}_delta_aligned"},
            {"ctx_family": "context_orderflow", "ctx": f"ctx_{ref}_delta_opposed"},
        ])
    context_templates.extend([
        {"ctx_family": "context_location", "ctx": "ctx_above_ema60"},
        {"ctx_family": "context_location", "ctx": "ctx_below_ema60"},
        {"ctx_family": "context_location", "ctx": "ctx_above_ema240"},
        {"ctx_family": "context_location", "ctx": "ctx_below_ema240"},
        {"ctx_family": "context_volatility", "ctx": "ctx_rv_expanding"},
        {"ctx_family": "context_volatility", "ctx": "ctx_rv_contracting"},
        {"ctx_family": "context_volume", "ctx": "ctx_open_notional_high"},
        {"ctx_family": "context_volume", "ctx": "ctx_open_notional_low"},
        {"ctx_family": "context_micro", "ctx": "ctx_large_trade_buyer"},
        {"ctx_family": "context_micro", "ctx": "ctx_large_trade_seller"},
        {"ctx_family": "context_micro", "ctx": "ctx_taker_buy_dominant"},
        {"ctx_family": "context_micro", "ctx": "ctx_taker_sell_dominant"},
    ])
    for rp in range_ctx:
        code = f"rb{int(round(rp * 10000)):04d}"
        context_templates.extend([
            {"ctx_family": "context_rangebar", "ctx": f"ctx_{code}_burst_aligned"},
            {"ctx_family": "context_rangebar", "ctx": f"ctx_{code}_burst_opposed"},
            {"ctx_family": "context_rangebar", "ctx": f"ctx_{code}_fast_sequence"},
            {"ctx_family": "context_rangebar", "ctx": f"ctx_{code}_delta_absorption"},
        ])
    if footprint_ctx is not None and not footprint_ctx.empty:
        rp = float(footprint_ctx.get("range_pct", pd.Series([0.0020])).iloc[0] if "range_pct" in footprint_ctx else 0.0020)
        code = f"fp{int(round(rp * 10000)):04d}"
        context_templates.extend([
            {"ctx_family": "context_footprint", "ctx": f"ctx_{code}_stacked_delta_aligned"},
            {"ctx_family": "context_footprint", "ctx": f"ctx_{code}_stacked_delta_opposed"},
            {"ctx_family": "context_footprint", "ctx": f"ctx_{code}_top_absorption"},
            {"ctx_family": "context_footprint", "ctx": f"ctx_{code}_bottom_absorption"},
        ])

    rows: list[dict[str, Any]] = []
    hid = 0
    for b in base_templates:
        hid += 1
        rows.append({"hypothesis_id": f"H{hid:05d}", **b, "ctx_family": "none", "ctx": "base_only", "hypothesis_name": b["base"]})
        for c in context_templates:
            hid += 1
            rows.append({
                "hypothesis_id": f"H{hid:05d}",
                **b,
                **c,
                "hypothesis_name": f"{b['base']}__{c['ctx']}",
            })
    return pd.DataFrame(rows)


def build_context_tags(
    feats: dict[str, Any],
    stats_by_ref: dict[str, dict[str, float]],
    open_stats: dict[str, float],
    range_ctx: dict[float, pd.DataFrame],
    footprint_ctx: pd.DataFrame,
    side: str,
) -> list[tuple[str, str, bool]]:
    """Active structural qualifiers for the current day/window/side."""
    sign = 1.0 if side == "LONG" else -1.0
    tags: list[tuple[str, str, bool]] = []
    for ref, rs in stats_by_ref.items():
        rr = fval(rs, "ret")
        rg = fval(rs, "range")
        dl = fval(rs, "delta")
        tags.extend([
            ("context_pretrend", f"ctx_{ref}_trend_aligned", is_finite(rr) and sign * rr > 0.0015),
            ("context_pretrend", f"ctx_{ref}_trend_opposed", is_finite(rr) and sign * rr < -0.0015),
            ("context_range", f"ctx_{ref}_compressed", is_finite(rg) and rg < 0.0060),
            ("context_range", f"ctx_{ref}_expanded", is_finite(rg) and rg > 0.0120),
            ("context_orderflow", f"ctx_{ref}_delta_aligned", is_finite(dl) and sign * dl > 0),
            ("context_orderflow", f"ctx_{ref}_delta_opposed", is_finite(dl) and sign * dl < 0),
        ])

    last_close = fval(open_stats, "close")
    ema60 = fval(feats, "last_dist_ema_60")
    ema240 = fval(feats, "last_dist_ema_240")
    rv_ratio = fval(feats, "last_rv_ratio_60_240")
    op_notional = fval(open_stats, "notional")
    pre_notional = max(fval(stats_by_ref.get("pre60", {}), "notional"), EPS)
    large_delta = fval(open_stats, "large_delta")
    taker_buy_ratio = fval(feats, "last_taker_buy_ratio")

    tags.extend([
        ("context_location", "ctx_above_ema60", is_finite(ema60) and ema60 > 0),
        ("context_location", "ctx_below_ema60", is_finite(ema60) and ema60 < 0),
        ("context_location", "ctx_above_ema240", is_finite(ema240) and ema240 > 0),
        ("context_location", "ctx_below_ema240", is_finite(ema240) and ema240 < 0),
        ("context_volatility", "ctx_rv_expanding", is_finite(rv_ratio) and rv_ratio > 1.15),
        ("context_volatility", "ctx_rv_contracting", is_finite(rv_ratio) and rv_ratio < 0.85),
        ("context_volume", "ctx_open_notional_high", is_finite(op_notional) and op_notional > pre_notional * 0.40),
        ("context_volume", "ctx_open_notional_low", is_finite(op_notional) and op_notional < pre_notional * 0.15),
        ("context_micro", "ctx_large_trade_buyer", is_finite(large_delta) and large_delta > 0),
        ("context_micro", "ctx_large_trade_seller", is_finite(large_delta) and large_delta < 0),
        ("context_micro", "ctx_taker_buy_dominant", is_finite(taker_buy_ratio) and taker_buy_ratio >= 0.56),
        ("context_micro", "ctx_taker_sell_dominant", is_finite(taker_buy_ratio) and taker_buy_ratio <= 0.44),
    ])

    for rp in range_ctx:
        code = f"rb{int(round(rp * 10000)):04d}"
        c15 = fval(feats, f"{code}_count_15")
        d15 = fval(feats, f"{code}_dir_sum_15")
        of15 = fval(feats, f"{code}_delta_sum_15")
        dur = fval(feats, f"{code}_duration_med_15")
        tags.extend([
            ("context_rangebar", f"ctx_{code}_burst_aligned", is_finite(c15) and c15 >= 3 and sign * d15 >= 2 and sign * of15 > 0),
            ("context_rangebar", f"ctx_{code}_burst_opposed", is_finite(c15) and c15 >= 3 and sign * d15 <= -2),
            ("context_rangebar", f"ctx_{code}_fast_sequence", is_finite(c15) and c15 >= 4 and is_finite(dur) and dur <= 180),
            ("context_rangebar", f"ctx_{code}_delta_absorption", is_finite(c15) and c15 >= 3 and sign * d15 >= 2 and sign * of15 < 0),
        ])

    if footprint_ctx is not None and not footprint_ctx.empty:
        rp = float(footprint_ctx.get("range_pct", pd.Series([0.0020])).iloc[0] if "range_pct" in footprint_ctx else 0.0020)
        code = f"fp{int(round(rp * 10000)):04d}"
        fp_delta = fval(feats, f"{code}_delta_sum_15")
        fp_top = fval(feats, f"{code}_top_delta_sum_15")
        fp_bottom = fval(feats, f"{code}_bottom_delta_sum_15")
        tags.extend([
            ("context_footprint", f"ctx_{code}_stacked_delta_aligned", is_finite(fp_delta) and sign * fp_delta > 0),
            ("context_footprint", f"ctx_{code}_stacked_delta_opposed", is_finite(fp_delta) and sign * fp_delta < 0),
            ("context_footprint", f"ctx_{code}_top_absorption", is_finite(fp_top) and fp_top < 0),
            ("context_footprint", f"ctx_{code}_bottom_absorption", is_finite(fp_bottom) and fp_bottom > 0),
        ])
    return tags


def add_broad_structural_events(
    events: list[EventSpec],
    n: int,
    signal_time: pd.Timestamp,
    s: dict[str, float],
    feats: dict[str, Any],
    stats_by_ref: dict[str, dict[str, float]],
    range_ctx: dict[float, pd.DataFrame],
    footprint_ctx: pd.DataFrame,
    *,
    max_context_tags_per_trigger: int = 12,
) -> None:
    """Generate broad, named structural hypotheses without fine parameter grids."""
    ret = fval(s, "ret")
    rng = fval(s, "range")
    close_pos = fval(s, "close_pos")
    delta = fval(s, "delta")
    large_delta = fval(s, "large_delta")
    upper_wick = fval(s, "upper_wick")
    lower_wick = fval(s, "lower_wick")
    efficiency = abs(ret) / rng if is_finite(ret) and is_finite(rng) and rng > EPS else np.nan

    base_triggers: list[tuple[str, str, str, bool, str]] = []
    for side, sign in [("LONG", 1.0), ("SHORT", -1.0)]:
        ss = side_suffix(side)
        base_triggers.extend([
            ("open_impulse", f"open{n}_impulse_close_extreme_{ss}", side, is_finite(ret) and sign * ret >= 0.0025 and ((side == "LONG" and close_pos >= 0.68) or (side == "SHORT" and close_pos <= 0.32)), "opening impulse closes at directional extreme"),
            ("open_impulse_fade", f"open{n}_impulse_exhaustion_fade_{ss}", side, is_finite(ret) and sign * ret <= -0.0025 and ((side == "LONG" and lower_wick > 0.0012 and close_pos >= 0.45) or (side == "SHORT" and upper_wick > 0.0012 and close_pos <= 0.55)), "opening impulse exhausts and leaves reversal wick"),
            ("open_range_expansion", f"open{n}_range_expansion_{ss}", side, is_finite(rng) and rng >= 0.0040 and ((side == "LONG" and close_pos >= 0.70) or (side == "SHORT" and close_pos <= 0.30)), "opening range expands and closes directionally"),
            ("open_body_efficiency", f"open{n}_efficient_body_{ss}", side, is_finite(efficiency) and efficiency >= 0.45 and sign * ret > 0 and ((side == "LONG" and close_pos >= 0.60) or (side == "SHORT" and close_pos <= 0.40)), "directional move is efficient relative to range"),
            ("open_wick_reversal", f"open{n}_wick_reversal_{ss}", side, (side == "LONG" and lower_wick >= 0.0015 and close_pos >= 0.55) or (side == "SHORT" and upper_wick >= 0.0015 and close_pos <= 0.45), "opening wick shows rejection/reversal"),
            ("open_orderflow_align", f"open{n}_price_delta_large_align_{ss}", side, is_finite(ret) and sign * ret > 0 and sign * delta > 0 and sign * large_delta > 0, "price, delta and large-trade flow align"),
            ("open_orderflow_diverge", f"open{n}_price_delta_diverge_fade_{ss}", side, is_finite(ret) and sign * ret < 0 and sign * delta > 0, "price move diverges from trade-bar delta"),
        ])

    for ref, rs in stats_by_ref.items():
        rr = fval(rs, "ret")
        rhi = fval(rs, "high")
        rlo = fval(rs, "low")
        close = fval(s, "close")
        hi = fval(s, "high")
        lo = fval(s, "low")
        for side, sign in [("LONG", 1.0), ("SHORT", -1.0)]:
            ss = side_suffix(side)
            base_triggers.extend([
                ("cross_session_continue", f"{ref}_open{n}_continue_{ss}", side, is_finite(rr) and is_finite(ret) and sign * rr >= 0.0025 and sign * ret >= 0.0010, f"{ref} trend continues into US open"),
                ("cross_session_fail", f"{ref}_open{n}_fail_{ss}", side, is_finite(rr) and is_finite(ret) and sign * rr <= -0.0025 and sign * ret >= 0.0010, f"{ref} trend fails at US open"),
                ("session_level_sweep", f"{ref}_level_sweep_reject_reclaim_open{n}_{ss}", side, (side == "LONG" and is_finite(rlo) and lo <= rlo * 0.9990 and close > rlo and close_pos >= 0.50) or (side == "SHORT" and is_finite(rhi) and hi >= rhi * 1.0010 and close < rhi and close_pos <= 0.50), f"open sweeps {ref} level then rejects/reclaims"),
                ("session_level_breakout", f"{ref}_level_breakout_hold_open{n}_{ss}", side, (side == "LONG" and is_finite(rhi) and close > rhi * 1.0005 and close_pos >= 0.60) or (side == "SHORT" and is_finite(rlo) and close < rlo * 0.9995 and close_pos <= 0.40), f"open breaks and holds {ref} level"),
            ])

    for family, name, side, ok, reason in base_triggers:
        if not ok:
            continue
        add_event(events, family, name, side, signal_time, reason, feats)
        active_tags = [(cf, tag) for cf, tag, active in build_context_tags(feats, stats_by_ref, s, range_ctx, footprint_ctx, side) if active]
        # Use deterministic order; cap only raw volume, not the hypothesis catalog.
        for cf, tag in active_tags[: max(0, int(max_context_tags_per_trigger))]:
            add_event(events, f"{family}__{cf}", f"{name}__{tag}", side, signal_time, f"{reason}; qualifier={tag}", feats)


def add_event(events: list[EventSpec], family: str, name: str, side: str, signal_time: pd.Timestamp, reason: str, features: dict[str, Any]) -> None:
    if side not in {"LONG", "SHORT"}:
        raise ValueError(side)
    events.append(EventSpec(family=family, name=name, side=side, signal_time=pd.Timestamp(signal_time), reason=reason, features=features))


def generate_day_events(
    day: date,
    day_df: pd.DataFrame,
    daily_hist: pd.DataFrame,
    range_ctx: dict[float, pd.DataFrame],
    footprint_ctx: pd.DataFrame,
    *,
    hypothesis_mode: str = "broad",
    max_context_tags_per_trigger: int = 12,
) -> list[EventSpec]:
    events: list[EventSpec] = []
    rth_open_min = 9 * 60 + 30
    rth_close_min = 16 * 60
    pre30 = session_slice(day_df, rth_open_min - 30, rth_open_min)
    pre60 = session_slice(day_df, rth_open_min - 60, rth_open_min)
    pre120 = session_slice(day_df, rth_open_min - 120, rth_open_min)
    pre240 = session_slice(day_df, rth_open_min - 240, rth_open_min)
    asia = session_slice(day_df, 0, 3 * 60)
    europe = session_slice(day_df, 3 * 60, rth_open_min)
    rth = session_slice(day_df, rth_open_min, rth_close_min, include_end=False)
    if len(rth) < 60 or pre60.empty:
        return events

    pre30_s = window_stats(pre30) if not pre30.empty else {}
    pre60_s = window_stats(pre60)
    pre120_s = window_stats(pre120) if not pre120.empty else pre60_s
    pre240_s = window_stats(pre240) if not pre240.empty else pre60_s
    asia_s = window_stats(asia) if not asia.empty else {}
    europe_s = window_stats(europe) if not europe.empty else {}
    stats_by_ref = {
        "pre30": pre30_s or pre60_s,
        "pre60": pre60_s,
        "pre120": pre120_s,
        "pre240": pre240_s,
        "asia": asia_s,
        "europe": europe_s,
    }
    rth_open = float(rth["open"].iloc[0])
    prior_ref = daily_hist.iloc[-1].to_dict() if not daily_hist.empty else {}

    base_features: dict[str, Any] = {
        "session_date": str(day),
        "rth_open": rth_open,
        **prefix_dict("pre30", pre30_s),
        **prefix_dict("pre60", pre60_s),
        **prefix_dict("pre120", pre120_s),
        **prefix_dict("pre240", pre240_s),
        **prefix_dict("asia", asia_s),
        **prefix_dict("europe", europe_s),
    }
    for k in ["pre60_range_q30", "pre60_range_q70", "open15_range_q70", "rth_range_q70"]:
        base_features[k] = prior_ref.get(k, np.nan)

    windows = {5: rth.iloc[:5], 10: rth.iloc[:10], 15: rth.iloc[:15], 30: rth.iloc[:30], 45: rth.iloc[:45], 60: rth.iloc[:60], 90: rth.iloc[:90], 120: rth.iloc[:120]}
    for n, w in list(windows.items()):
        if len(w) < max(2, min(n, 15)):
            continue
        signal_time = pd.Timestamp(w.index[-1]) + pd.Timedelta(minutes=1)
        s = window_stats(w)
        feats = {**base_features, **prefix_dict(f"open{n}", s)}
        for rp, ctx in range_ctx.items():
            feats.update(lookup_context(ctx, signal_time))
        feats.update(lookup_context(footprint_ctx, signal_time))

        ret = s.get("ret", np.nan)
        rng = s.get("range", np.nan)
        close_pos = s.get("close_pos", np.nan)
        delta = s.get("delta", 0.0)
        large_delta = s.get("large_delta", 0.0)
        pre_ret = pre60_s.get("ret", np.nan)
        pre_range = pre60_s.get("range", np.nan)
        pre_hi = pre60_s.get("high", np.nan)
        pre_lo = pre60_s.get("low", np.nan)
        pre_range_q30 = prior_ref.get("pre60_range_q30", np.nan)
        open15_range_q70 = prior_ref.get("open15_range_q70", np.nan)
        last_row = w.iloc[-1]
        for c in ["dist_ema_60", "dist_ema_240", "rv_ratio_60_240", "taker_buy_ratio"]:
            if c in last_row.index:
                feats[f"last_{c}"] = float(last_row[c]) if np.isfinite(float(last_row[c])) else np.nan

        if hypothesis_mode == "broad":
            add_broad_structural_events(
                events,
                n,
                signal_time,
                s,
                feats,
                stats_by_ref,
                range_ctx,
                footprint_ctx,
                max_context_tags_per_trigger=max_context_tags_per_trigger,
            )

        # 1. Opening impulse continuation. Fixed structural definition.
        if n in {5, 15, 30} and np.isfinite(ret):
            if ret >= 0.0025 and close_pos >= 0.68:
                add_event(events, "trend_continuation", f"open{n}_impulse_closehigh_long", "LONG", signal_time, "opening impulse closes near high", feats)
            if ret <= -0.0025 and close_pos <= 0.32:
                add_event(events, "trend_continuation", f"open{n}_impulse_closelow_short", "SHORT", signal_time, "opening impulse closes near low", feats)

        # 2. Pre-open trend aligns with RTH open.
        if n in {15, 30} and np.isfinite(pre_ret) and np.isfinite(ret):
            if pre_ret >= 0.0030 and ret >= 0.0015 and close_pos >= 0.55:
                add_event(events, "cross_session_continuation", f"pre60_up_open{n}_align_long", "LONG", signal_time, "preopen trend continues through open", feats)
            if pre_ret <= -0.0030 and ret <= -0.0015 and close_pos <= 0.45:
                add_event(events, "cross_session_continuation", f"pre60_down_open{n}_align_short", "SHORT", signal_time, "preopen weakness continues through open", feats)

        # 3. Pre-open trend fails at open.
        if n in {15, 30, 60} and np.isfinite(pre_ret) and np.isfinite(ret):
            if pre_ret >= 0.0040 and ret <= -0.0010 and close_pos <= 0.45:
                add_event(events, "cross_session_failure", f"pre60_up_open{n}_fail_short", "SHORT", signal_time, "preopen uptrend fails at RTH open", feats)
            if pre_ret <= -0.0040 and ret >= 0.0010 and close_pos >= 0.55:
                add_event(events, "cross_session_failure", f"pre60_down_open{n}_fail_long", "LONG", signal_time, "preopen downtrend fails at RTH open", feats)

        # 4. Sweep and reclaim around preopen levels.
        if n in {30, 60, 120} and np.isfinite(pre_hi) and np.isfinite(pre_lo):
            swept_hi = s.get("high", np.nan) >= pre_hi * 1.0010
            swept_lo = s.get("low", np.nan) <= pre_lo * 0.9990
            close = s.get("close", np.nan)
            if swept_hi and np.isfinite(close) and close < pre_hi and close_pos <= 0.50:
                add_event(events, "fakeout_reversal", f"open{n}_prehigh_sweep_reject_short", "SHORT", signal_time, "RTH sweeps preopen high then rejects", feats)
            if swept_lo and np.isfinite(close) and close > pre_lo and close_pos >= 0.50:
                add_event(events, "fakeout_reversal", f"open{n}_prelow_sweep_reclaim_long", "LONG", signal_time, "RTH sweeps preopen low then reclaims", feats)

        # 5. Absorption divergence: price sweeps but trade-bar delta does not confirm.
        if n in {30, 60} and np.isfinite(pre_hi) and np.isfinite(pre_lo):
            swept_hi = s.get("high", np.nan) >= pre_hi * 1.0010
            swept_lo = s.get("low", np.nan) <= pre_lo * 0.9990
            if swept_hi and delta <= 0 and close_pos <= 0.55:
                add_event(events, "orderflow_absorption", f"open{n}_up_sweep_delta_div_short", "SHORT", signal_time, "upside sweep without positive delta confirmation", feats)
            if swept_lo and delta >= 0 and close_pos >= 0.45:
                add_event(events, "orderflow_absorption", f"open{n}_down_sweep_delta_div_long", "LONG", signal_time, "downside sweep without negative delta confirmation", feats)

        # 6. Compression expansion. Causal threshold comes from prior sessions only.
        if n in {15, 30, 60} and np.isfinite(pre_range) and np.isfinite(pre_range_q30):
            compressed = pre_range <= pre_range_q30
            if compressed and ret >= 0.0015 and close_pos >= 0.60:
                add_event(events, "compression_breakout", f"pre60_compress_open{n}_break_up_long", "LONG", signal_time, "preopen compression breaks upward", feats)
            if compressed and ret <= -0.0015 and close_pos <= 0.40:
                add_event(events, "compression_breakout", f"pre60_compress_open{n}_break_down_short", "SHORT", signal_time, "preopen compression breaks downward", feats)

        # 7. Opening volatility expansion: not direction by itself, requires close position.
        if n in {15, 30} and np.isfinite(rng) and np.isfinite(open15_range_q70):
            expanded = rng >= open15_range_q70
            if expanded and close_pos >= 0.72 and delta > 0:
                add_event(events, "volatility_expansion", f"open{n}_range_expand_delta_long", "LONG", signal_time, "opening range expands with positive delta", feats)
            if expanded and close_pos <= 0.28 and delta < 0:
                add_event(events, "volatility_expansion", f"open{n}_range_expand_delta_short", "SHORT", signal_time, "opening range expands with negative delta", feats)

        # 8. Trade-bar delta / large-trade confirmation.
        if n in {5, 15, 30} and np.isfinite(ret):
            if ret > 0 and delta > 0 and large_delta > 0 and close_pos >= 0.55:
                add_event(events, "tradebar_orderflow", f"open{n}_price_delta_large_align_long", "LONG", signal_time, "price/delta/large trades aligned up", feats)
            if ret < 0 and delta < 0 and large_delta < 0 and close_pos <= 0.45:
                add_event(events, "tradebar_orderflow", f"open{n}_price_delta_large_align_short", "SHORT", signal_time, "price/delta/large trades aligned down", feats)
            if ret > 0 and delta < 0 and close_pos <= 0.65:
                add_event(events, "tradebar_orderflow_divergence", f"open{n}_up_price_negative_delta_short", "SHORT", signal_time, "up move with negative delta divergence", feats)
            if ret < 0 and delta > 0 and close_pos >= 0.35:
                add_event(events, "tradebar_orderflow_divergence", f"open{n}_down_price_positive_delta_long", "LONG", signal_time, "down move with positive delta divergence", feats)

        # 9. Europe session continuation/fade.
        eret = europe_s.get("ret", np.nan)
        if n in {15, 30, 60} and np.isfinite(eret):
            if eret >= 0.0060 and ret >= 0.0010 and close_pos >= 0.55:
                add_event(events, "europe_to_us", f"europe_strong_open{n}_continue_long", "LONG", signal_time, "Europe strength persists into US open", feats)
            if eret <= -0.0060 and ret <= -0.0010 and close_pos <= 0.45:
                add_event(events, "europe_to_us", f"europe_weak_open{n}_continue_short", "SHORT", signal_time, "Europe weakness persists into US open", feats)
            if eret >= 0.0060 and ret <= -0.0015:
                add_event(events, "europe_to_us_failure", f"europe_strong_open{n}_fade_short", "SHORT", signal_time, "Europe rally fades at US open", feats)
            if eret <= -0.0060 and ret >= 0.0015:
                add_event(events, "europe_to_us_failure", f"europe_weak_open{n}_fade_long", "LONG", signal_time, "Europe selloff fades at US open", feats)

        # 10. Range-bar context hypotheses.
        for rp in range_ctx:
            code = f"rb{int(round(rp * 10000)):04d}"
            c15 = feats.get(f"{code}_count_15", np.nan)
            d15 = feats.get(f"{code}_dir_sum_15", np.nan)
            of15 = feats.get(f"{code}_delta_sum_15", np.nan)
            med_dur = feats.get(f"{code}_duration_med_15", np.nan)
            if n in {15, 30} and np.isfinite(c15) and c15 >= 3 and np.isfinite(d15):
                if d15 >= 2 and of15 > 0:
                    add_event(events, "rangebar_momentum_context", f"{code}_open{n}_rb_burst_up_long", "LONG", signal_time, "range bars burst upward with delta", feats)
                if d15 <= -2 and of15 < 0:
                    add_event(events, "rangebar_momentum_context", f"{code}_open{n}_rb_burst_down_short", "SHORT", signal_time, "range bars burst downward with delta", feats)
                if d15 >= 2 and of15 < 0:
                    add_event(events, "rangebar_absorption_context", f"{code}_open{n}_rb_up_delta_div_short", "SHORT", signal_time, "up range-bar burst without delta", feats)
                if d15 <= -2 and of15 > 0:
                    add_event(events, "rangebar_absorption_context", f"{code}_open{n}_rb_down_delta_div_long", "LONG", signal_time, "down range-bar burst without delta", feats)
            if n in {30, 60} and np.isfinite(c15) and c15 >= 4 and np.isfinite(med_dur) and med_dur <= 180:
                if d15 > 0 and close_pos >= 0.55:
                    add_event(events, "rangebar_speed_context", f"{code}_open{n}_fast_rb_up_long", "LONG", signal_time, "fast range-bar sequence upward", feats)
                if d15 < 0 and close_pos <= 0.45:
                    add_event(events, "rangebar_speed_context", f"{code}_open{n}_fast_rb_down_short", "SHORT", signal_time, "fast range-bar sequence downward", feats)

        # 11. Footprint context hypotheses.
        fp_code = f"fp{int(round(0.0020 * 10000)):04d}"
        # Also support configured code.
        if footprint_ctx is not None and not footprint_ctx.empty:
            fp_code = f"fp{int(round(float(footprint_ctx.get('range_pct', pd.Series([0.0020])).iloc[0] if 'range_pct' in footprint_ctx else 0.0020) * 10000)):04d}"
        for code in {fp_code, f"fp{int(round(20)):04d}", f"fp{int(round(0.0020 * 10000)):04d}"}:
            fp_delta = feats.get(f"{code}_delta_sum_15", np.nan)
            fp_top = feats.get(f"{code}_top_delta_sum_15", np.nan)
            fp_bottom = feats.get(f"{code}_bottom_delta_sum_15", np.nan)
            if n in {15, 30} and np.isfinite(fp_delta):
                if ret > 0 and fp_delta > 0 and close_pos >= 0.55:
                    add_event(events, "footprint_delta_context", f"{code}_open{n}_stacked_buy_long", "LONG", signal_time, "footprint delta supports upside", feats)
                if ret < 0 and fp_delta < 0 and close_pos <= 0.45:
                    add_event(events, "footprint_delta_context", f"{code}_open{n}_stacked_sell_short", "SHORT", signal_time, "footprint delta supports downside", feats)
                if np.isfinite(fp_top) and s.get("high", np.nan) >= pre_hi * 1.0010 and fp_top < 0:
                    add_event(events, "footprint_absorption_context", f"{code}_open{n}_top_sell_absorb_short", "SHORT", signal_time, "sell absorption at upper footprint buckets", feats)
                if np.isfinite(fp_bottom) and s.get("low", np.nan) <= pre_lo * 0.9990 and fp_bottom > 0:
                    add_event(events, "footprint_absorption_context", f"{code}_open{n}_bottom_buy_absorb_long", "LONG", signal_time, "buy absorption at lower footprint buckets", feats)

    return events


def build_daily_session_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rth_open_min = 9 * 60 + 30
    rth_close_min = 16 * 60
    for d, g in df.groupby("ny_date", sort=True):
        pre60 = session_slice(g, rth_open_min - 60, rth_open_min)
        rth = session_slice(g, rth_open_min, rth_close_min)
        if len(rth) < 60 or pre60.empty:
            continue
        pre60_s = window_stats(pre60)
        open15_s = window_stats(rth.iloc[:15]) if len(rth) >= 15 else {}
        rth_s = window_stats(rth)
        rows.append(
            {
                "session_date": str(d),
                "ny_date": d,
                "rth_open_time": str(rth["ny_time"].iloc[0]),
                "pre60_ret": pre60_s.get("ret", np.nan),
                "pre60_range": pre60_s.get("range", np.nan),
                "open15_ret": open15_s.get("ret", np.nan),
                "open15_range": open15_s.get("range", np.nan),
                "rth_ret": rth_s.get("ret", np.nan),
                "rth_range": rth_s.get("range", np.nan),
                "rth_max_up": float(rth["high"].max() / rth["open"].iloc[0] - 1.0),
                "rth_max_down": float(rth["low"].min() / rth["open"].iloc[0] - 1.0),
            }
        )
    daily = pd.DataFrame(rows).sort_values("session_date")
    if daily.empty:
        return daily
    for col, q, name in [
        ("pre60_range", 0.30, "pre60_range_q30"),
        ("pre60_range", 0.70, "pre60_range_q70"),
        ("open15_range", 0.70, "open15_range_q70"),
        ("rth_range", 0.70, "rth_range_q70"),
    ]:
        daily[name] = rolling_quantile_past(daily[col], 80, q, minp=20)
    return daily


def build_events(
    df: pd.DataFrame,
    range_ctx: dict[float, pd.DataFrame],
    footprint_ctx: pd.DataFrame,
    *,
    hypothesis_mode: str = "broad",
    max_context_tags_per_trigger: int = 12,
    enabled: bool = True,
) -> pd.DataFrame:
    daily = build_daily_session_table(df)
    daily_by_date = daily.set_index("ny_date") if not daily.empty else pd.DataFrame()
    all_events: list[EventSpec] = []
    groups = list(df.groupby("ny_date", sort=True))
    prog = ProgressReporter(label="[events] hypothesis scan", total=len(groups), every=max(1, len(groups) // 20), enabled=enabled)
    for i, (d, g) in enumerate(groups, start=1):
        hist = daily[daily["ny_date"] < d].tail(1) if not daily.empty else pd.DataFrame()
        all_events.extend(
            generate_day_events(
                d,
                g,
                hist,
                range_ctx,
                footprint_ctx,
                hypothesis_mode=hypothesis_mode,
                max_context_tags_per_trigger=max_context_tags_per_trigger,
            )
        )
        prog.update(i)
    # ProgressReporter already prints the 100% line at the last update in this repo;
    # avoid a duplicate final line.

    rows: list[dict[str, Any]] = []
    for ev in all_events:
        row = {
            "family": ev.family,
            "event_name": ev.name,
            "side": ev.side,
            "signal_time": ev.signal_time,
            "reason": ev.reason,
        }
        row.update(ev.features)
        rows.append(row)
    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.drop_duplicates(subset=["family", "event_name", "side", "signal_time"]).sort_values("signal_time")
    return events



def attach_outcomes(events: pd.DataFrame, df: pd.DataFrame, horizons: list[int], fee_rate: float) -> pd.DataFrame:
    """Attach next-open outcomes with vectorized horizon/MFE/MAE computation.

    v2 used a Python loop over every event and every horizon. With broad mode +
    range/footprint that can easily produce 1M+ events, so the row loop becomes
    the bottleneck after ``raw_events=...``.  This implementation precomputes
    forward max/min arrays once per horizon, then gathers values by integer
    entry position.
    """
    if events.empty:
        return events

    out = events.copy()
    idx = pd.DatetimeIndex(df.index)
    # Force nanosecond timestamps for integer arithmetic. Some loaders/CSV/SQLite
    # paths can produce datetime64[us]. If we add Timedelta.value (nanoseconds) to
    # a microsecond integer axis, a 60-minute horizon becomes 60,000 minutes.
    # ``idx_np_ns`` / ``idx_ns`` are therefore the only timestamp arrays used for
    # vectorized exit-time search and output.
    idx_np_ns = idx.to_numpy(dtype="datetime64[ns]")
    idx_ns = idx_np_ns.astype("int64")
    n_bars = len(df)
    # Primary bars are left-labeled OHLC bars. A bar indexed at 09:34 has its
    # close available at 09:35 on a 1m axis. Outcomes must therefore use the
    # close of the last bar whose close-available time reaches the target
    # horizon, not the close of the bar whose *start* time equals the target.
    # v4 fixed timestamp units but still used the target-start bar close, which
    # over-held by one primary bar on regular 1m data.
    if n_bars >= 2:
        diffs = np.diff(idx_ns)
        diffs = diffs[diffs > 0]
        bar_delta_ns = int(np.median(diffs)) if len(diffs) else int(pd.Timedelta(minutes=1).value)
    else:
        bar_delta_ns = int(pd.Timedelta(minutes=1).value)
    close_available_ns = idx_ns + np.int64(bar_delta_ns)
    close_available_np_ns = close_available_ns.astype("datetime64[ns]")
    if n_bars == 0:
        return out

    open_arr = df["open"].to_numpy(dtype=float)
    high_arr = df["high"].to_numpy(dtype=float)
    low_arr = df["low"].to_numpy(dtype=float)
    close_arr = df["close"].to_numpy(dtype=float)

    signals = pd.to_datetime(out["signal_time"], errors="coerce")
    signal_ns = signals.to_numpy(dtype="datetime64[ns]")
    entry_pos = idx.searchsorted(signal_ns, side="left").astype(np.int64)
    valid_entry = (entry_pos >= 0) & (entry_pos < n_bars)

    out["entry_pos"] = pd.Series(pd.array(np.where(valid_entry, entry_pos, pd.NA), dtype="Int64"), index=out.index)
    entry_time = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]")
    entry_price = np.full(len(out), np.nan, dtype=float)
    if valid_entry.any():
        vp = entry_pos[valid_entry]
        entry_time[valid_entry] = idx_np_ns[vp]
        entry_price[valid_entry] = open_arr[vp]
    out["entry_time"] = entry_time
    out["entry_price"] = entry_price

    side_mult = np.where(out["side"].astype(str).to_numpy() == "LONG", 1.0, -1.0)
    entry_price_safe = np.where(np.isfinite(entry_price) & (entry_price > 0), entry_price, np.nan)

    # Precompute integer timestamp axis once.  searchsorted on int64 is much
    # faster than per-row Timestamp construction.  idx_ns is explicitly ns above.
    entry_ns = np.full(len(out), np.iinfo(np.int64).min, dtype=np.int64)
    entry_ns[valid_entry] = idx_ns[entry_pos[valid_entry]]

    for h in horizons:
        target_ns = entry_ns + np.int64(pd.Timedelta(minutes=int(h)).value)
        # Use the close of the first bar whose close is available at/after the
        # target horizon. On a clean 1m axis, entry 09:35 + h60 exits using the
        # 10:34 bar close, whose available time is 10:35.
        exit_pos = np.searchsorted(close_available_ns, target_ns, side="left").astype(np.int64)
        valid_exit = valid_entry & (exit_pos >= 0) & (exit_pos < n_bars)

        out[f"exit_time_h{h}"] = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]")
        out[f"exit_bar_time_h{h}"] = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]")
        raw = np.full(len(out), np.nan, dtype=float)
        mfe = np.full(len(out), np.nan, dtype=float)
        mae = np.full(len(out), np.nan, dtype=float)

        if valid_exit.any():
            vp = entry_pos[valid_exit]
            xp = exit_pos[valid_exit]
            out.loc[valid_exit, f"exit_time_h{h}"] = close_available_np_ns[xp]
            out.loc[valid_exit, f"exit_bar_time_h{h}"] = idx_np_ns[xp]
            ep = entry_price_safe[valid_exit]
            exit_close = close_arr[xp]
            sm = side_mult[valid_exit]
            raw[valid_exit] = sm * (exit_close / ep - 1.0)

            # Exact forward extrema for the actual entry_pos -> exit_pos span.
            # On clean 1m data each horizon usually has one span, but this
            # grouped-by-span path stays correct across occasional missing bars.
            spans = (xp - vp).astype(np.int64)
            tmp_mfe = np.empty(len(vp), dtype=float)
            tmp_mae = np.empty(len(vp), dtype=float)
            for span in np.unique(spans):
                span = int(max(0, span))
                m = spans == span
                fwd_high = pd.Series(high_arr).iloc[::-1].rolling(span + 1, min_periods=1).max().iloc[::-1].to_numpy(dtype=float)
                fwd_low = pd.Series(low_arr).iloc[::-1].rolling(span + 1, min_periods=1).min().iloc[::-1].to_numpy(dtype=float)
                max_high = fwd_high[vp[m]]
                min_low = fwd_low[vp[m]]
                ep_m = ep[m]
                sm_m = sm[m]
                long_mask = sm_m > 0
                mfe_m = np.empty(len(ep_m), dtype=float)
                mae_m = np.empty(len(ep_m), dtype=float)
                mfe_m[long_mask] = max_high[long_mask] / ep_m[long_mask] - 1.0
                mae_m[long_mask] = min_low[long_mask] / ep_m[long_mask] - 1.0
                mfe_m[~long_mask] = ep_m[~long_mask] / min_low[~long_mask] - 1.0
                mae_m[~long_mask] = ep_m[~long_mask] / max_high[~long_mask] - 1.0
                tmp_mfe[m] = mfe_m
                tmp_mae[m] = mae_m
            mfe[valid_exit] = tmp_mfe
            mae[valid_exit] = tmp_mae

        out[f"ret_h{h}_raw"] = raw
        out[f"ret_h{h}_net"] = raw - fee_rate
        out[f"mfe_h{h}"] = mfe
        out[f"mae_h{h}"] = mae

    out["entry_not_next_open_flag"] = False
    out["context_available_time_flag"] = False
    out["year"] = pd.to_datetime(out["signal_time"], errors="coerce").dt.year
    return out


def stat_one(events: pd.DataFrame, metric: str) -> dict[str, Any]:
    if events.empty or metric not in events.columns:
        return {"count": 0}
    v = pd.to_numeric(events[metric], errors="coerce").dropna().to_numpy(dtype=float)
    if len(v) == 0:
        return {"count": 0}
    pos = v[v > 0]
    neg = v[v < 0]
    gross_profit = float(pos.sum()) if len(pos) else 0.0
    gross_loss = float(-neg.sum()) if len(neg) else 0.0
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else float("nan"))
    # dependency on a few winners: useful for rejecting lottery-like candidates
    top5 = np.sort(pos)[-5:].sum() if len(pos) else 0.0
    return {
        "count": int(len(v)),
        "mean": float(np.nanmean(v)),
        "median": float(np.nanmedian(v)),
        "std": float(np.nanstd(v, ddof=0)),
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
        "win_rate": float(np.mean(v > 0)),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "pf": pf,
        "top5_winner_share": float(top5 / gross_profit) if gross_profit > 0 else np.nan,
    }


def grouped_stats(events: pd.DataFrame, group_cols: list[str], metric: str, *, min_count: int = 1) -> pd.DataFrame:
    if events.empty or metric not in events.columns:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in events.columns]
    if len(cols) != len(group_cols):
        return pd.DataFrame()
    work = events[cols + [metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    if work.empty:
        return pd.DataFrame()
    v = work[metric].to_numpy(dtype=float)
    work["_win"] = v > 0
    work["_pos"] = np.where(v > 0, v, 0.0)
    work["_loss"] = np.where(v < 0, -v, 0.0)
    gb = work.groupby(cols, dropna=False, sort=True)
    out = gb.agg(
        count=(metric, "count"),
        mean=(metric, "mean"),
        median=(metric, "median"),
        std=(metric, "std"),
        min=(metric, "min"),
        max=(metric, "max"),
        win_rate=("_win", "mean"),
        gross_profit=("_pos", "sum"),
        gross_loss=("_loss", "sum"),
    ).reset_index()
    out["pf"] = np.where(
        out["gross_loss"] > 0,
        out["gross_profit"] / out["gross_loss"],
        np.where(out["gross_profit"] > 0, np.inf, np.nan),
    )
    out = out[out["count"] >= int(min_count)].copy()
    if not out.empty:
        out = out.sort_values(["pf", "mean", "count"], ascending=[False, False, False])
    return out


def yearly_stability(events: pd.DataFrame, metric: str, min_year_count: int) -> pd.DataFrame:
    if events.empty or metric not in events.columns:
        return pd.DataFrame()
    ys = grouped_stats(events, ["family", "event_name", "side", "year"], metric, min_count=min_year_count)
    if ys.empty:
        return pd.DataFrame()
    valid = ys.copy()
    valid["pos_year"] = valid["mean"] > 0
    out = valid.groupby(["family", "event_name", "side"], sort=True).agg(
        years_with_min_count=("year", "count"),
        positive_years=("pos_year", "sum"),
        min_year_mean=("mean", "min"),
        median_year_mean=("mean", "median"),
        min_year_pf=("pf", "min"),
    ).reset_index()
    out["positive_years"] = out["positive_years"].astype(int)
    out["years_with_min_count"] = out["years_with_min_count"].astype(int)
    if not out.empty:
        out = out.sort_values(["positive_years", "median_year_mean"], ascending=[False, False])
    return out


def daily_dedup_stats(events: pd.DataFrame, metric: str, min_count: int) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    df = events.copy()
    df["session_date"] = df["session_date"].astype(str)
    dedup = df.sort_values("signal_time").drop_duplicates(["family", "event_name", "side", "session_date"], keep="first")
    return grouped_stats(dedup, ["family", "event_name", "side"], metric, min_count=min_count)


def candidate_rank(events: pd.DataFrame, horizons: list[int], min_count: int, min_year_count: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    key_cols = ["family", "event_name", "side"]
    for h in horizons:
        metric = f"ret_h{h}_net"
        base = grouped_stats(events, key_cols, metric, min_count=min_count)
        if base.empty:
            continue
        stab = yearly_stability(events, metric, min_year_count)
        dedup = daily_dedup_stats(events, metric, min_count=max(10, min_count // 2))
        merged = base.merge(stab, on=key_cols, how="left").merge(dedup, on=key_cols, how="left", suffixes=("", "_dedup"))
        # Normalize naming for old report columns.
        for c in ["count", "mean", "median", "std", "min", "max", "win_rate", "gross_profit", "gross_loss", "pf"]:
            dc = f"{c}_dedup"
            if dc in merged.columns:
                merged[f"dedup_{c}"] = merged[dc]
        merged["positive_years"] = merged.get("positive_years", 0).fillna(0)
        merged["years_with_min_count"] = merged.get("years_with_min_count", 0).fillna(0)
        dedup_mean = pd.to_numeric(merged.get("dedup_mean", np.nan), errors="coerce")
        dedup_pf = pd.to_numeric(merged.get("dedup_pf", np.nan), errors="coerce")
        merged["horizon"] = h
        merged["metric"] = metric
        merged["score"] = (
            pd.to_numeric(merged["mean"], errors="coerce").clip(lower=0).fillna(0) * 10_000
            + pd.to_numeric(merged["pf"], errors="coerce").clip(upper=3).fillna(0) * 10
            + pd.to_numeric(merged["win_rate"], errors="coerce").fillna(0) * 5
            + pd.to_numeric(merged["positive_years"], errors="coerce").fillna(0) * 5
            + pd.to_numeric(merged["years_with_min_count"], errors="coerce").fillna(0) * 2
            + dedup_mean.clip(lower=0).fillna(0) * 8_000
            + dedup_pf.clip(upper=3).fillna(0) * 5
        )
        rows.append(merged)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out = out.sort_values(["score", "horizon"], ascending=[False, True])
    return out

def write_brief(out_dir: Path, events: pd.DataFrame, rank: pd.DataFrame, args: argparse.Namespace) -> None:
    lines: list[str] = []
    lines.append("# US RTH Open Multi-Hypothesis Event Lab")
    lines.append("")
    lines.append("This report is a direction finder, not a strategy validation report.")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- symbol: `{args.symbol}`")
    lines.append(f"- primary: OKX trade bars `{args.primary_timeframe}`")
    lines.append(f"- period: `{args.start_date}` -> `{args.end_date}`")
    lines.append(f"- fee_rate: `{args.fee_rate}` round-trip")
    lines.append(f"- include_range_bars: `{args.include_range_bars}`")
    lines.append(f"- include_footprint: `{args.include_footprint}`")
    lines.append("")
    lines.append("## Event families")
    fam_counts = events["family"].value_counts() if not events.empty else pd.Series(dtype=int)
    for fam, cnt in fam_counts.items():
        lines.append(f"- {fam}: {int(cnt):,}")
    lines.append("")
    lines.append("## Top direction candidates")
    if rank.empty:
        lines.append("No candidate passed minimum count filters.")
    else:
        cols = ["horizon", "family", "event_name", "side", "count", "mean", "median", "win_rate", "pf", "dedup_count", "dedup_mean", "dedup_pf", "positive_years", "years_with_min_count", "score"]
        show = rank.head(30)
        lines.append(show[[c for c in cols if c in show.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Read this correctly")
    lines.append("- A high score means 'worth deeper research', not 'ready for live'.")
    lines.append("- Daily de-dup stats matter more than raw event stats when many events cluster on the same day.")
    lines.append("- Any candidate still needs a dedicated strategy replay, stress tests, yearly stability, and causal audit.")
    (out_dir / "20_research_brief.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_enabled = not args.no_progress

    horizons = parse_csv_ints(args.horizons)
    primary_delta = timeframe_delta(args.primary_timeframe)
    if primary_delta != pd.Timedelta(minutes=1):
        print(f"[warn] primary timeframe={args.primary_timeframe}; event timing assumes minute-based bars and next available bar execution.")

    raw = load_primary(args)
    raw = add_time_columns(raw, timestamp_tz=args.timestamp_tz)
    feat = build_primary_features(raw)
    feat = feat[(feat.index >= pd.Timestamp(args.warmup_start_date)) & (feat.index <= pd.Timestamp(args.end_date) + pd.Timedelta(days=1))].copy()

    range_ctx = load_range_context(args, feat)
    footprint_ctx = load_footprint_context(args)

    daily = build_daily_session_table(feat)
    daily.to_csv(out_dir / "00_daily_session_labels.csv", index=False)
    print(f"[daily] sessions={len(daily):,}")

    catalog = build_hypothesis_catalog(range_ctx, footprint_ctx)
    catalog.to_csv(out_dir / "00_hypothesis_catalog.csv", index=False)
    print(f"[catalog] hypotheses={len(catalog):,}")

    events = build_events(
        feat,
        range_ctx,
        footprint_ctx,
        hypothesis_mode=args.hypothesis_mode,
        max_context_tags_per_trigger=args.max_context_tags_per_trigger,
        enabled=progress_enabled,
    )
    print(f"[events] raw_events={len(events):,} families={events['family'].nunique() if not events.empty else 0}")
    if not events.empty and not args.keep_event_features:
        sample_n = min(50_000, len(events))
        events.head(sample_n).to_csv(out_dir / "01_hypothesis_event_feature_sample.csv", index=False)
        keep_cols = [c for c in ["family", "event_name", "side", "signal_time", "reason", "session_date"] if c in events.columns]
        events = events[keep_cols].copy()
        print(f"[events] compacted columns={len(keep_cols)}; feature_sample_rows={sample_n:,}; use --keep-event-features to retain all columns")
    print(f"[outcomes] vectorized attach horizons={horizons} events={len(events):,}")
    events_out = attach_outcomes(events, feat, horizons, args.fee_rate)
    print(f"[outcomes] done rows={len(events_out):,}")
    # Trim warmup-start leakage from outputs: signals must be within requested research window.
    if not events_out.empty:
        events_out = events_out[
            (pd.to_datetime(events_out["signal_time"]) >= pd.Timestamp(args.start_date))
            & (pd.to_datetime(events_out["signal_time"]) <= pd.Timestamp(args.end_date) + pd.Timedelta(days=1))
        ].copy()

    if args.write_full_events:
        print(f"[write] full events csv rows={len(events_out):,}")
        events_out.to_csv(out_dir / "01_hypothesis_events.csv", index=False)
    else:
        sample_n = min(50_000, len(events_out))
        print(f"[write] event sample csv rows={sample_n:,}; use --write-full-events for full 01_hypothesis_events.csv")
        events_out.head(sample_n).to_csv(out_dir / "01_hypothesis_events_sample.csv", index=False)

    overview_rows: list[dict[str, Any]] = []
    for h in horizons:
        metric = f"ret_h{h}_net"
        print(f"[stats] horizon={h} metric={metric}")
        st = stat_one(events_out, metric) if not events_out.empty else {"count": 0}
        st["horizon"] = h
        st["metric"] = metric
        overview_rows.append(st)
        grouped_stats(events_out, ["family"], metric, min_count=max(5, args.min_count // 2)).to_csv(out_dir / f"02_family_stats_h{h}.csv", index=False)
        grouped_stats(events_out, ["family", "event_name", "side"], metric, min_count=args.min_count).to_csv(out_dir / f"03_event_stats_h{h}.csv", index=False)
        yearly_stability(events_out, metric, args.min_year_count).to_csv(out_dir / f"04_yearly_stability_h{h}.csv", index=False)
        daily_dedup_stats(events_out, metric, min_count=max(10, args.min_count // 2)).to_csv(out_dir / f"05_daily_dedup_event_stats_h{h}.csv", index=False)

    pd.DataFrame(overview_rows).to_csv(out_dir / "02_overview.csv", index=False)
    rank = candidate_rank(events_out, horizons, args.min_count, args.min_year_count)
    rank.to_csv(out_dir / "15_direction_candidate_rank.csv", index=False)

    audit_rows = [
        {"check": "entry_not_next_open_flag", "fail_count": int(events_out.get("entry_not_next_open_flag", pd.Series(dtype=bool)).sum()) if not events_out.empty else 0},
        {"check": "context_available_time_flag", "fail_count": int(events_out.get("context_available_time_flag", pd.Series(dtype=bool)).sum()) if not events_out.empty else 0},
        {"check": "ordinary_kline_used", "fail_count": 0},
    ]
    # Guardrail for vectorized outcome bugs: exit_time_h{h} is the close-available
    # time of the exit bar and must be entry_time + h minutes on a regular 1m
    # primary axis. v3 caught ns/us unit bugs; v5 also avoids one-bar overholding
    # by separating exit_bar_time_h{h} from exit_time_h{h}.
    if not events_out.empty and "entry_time" in events_out.columns:
        entry_ts = pd.to_datetime(events_out["entry_time"], errors="coerce")
        for h in horizons:
            col = f"exit_time_h{h}"
            if col in events_out.columns:
                exit_ts = pd.to_datetime(events_out[col], errors="coerce")
                delta_min = (exit_ts - entry_ts).dt.total_seconds() / 60.0
                valid_delta = delta_min.notna()
                # Missing bars may make the first available exit close slightly later;
                # on complete 1m crypto trade bars this should be exactly zero.
                mismatch = valid_delta & (np.abs(delta_min - float(h)) > 1e-9)
                audit_rows.append({"check": f"horizon_time_mismatch_h{h}", "fail_count": int(mismatch.sum())})
                bar_col = f"exit_bar_time_h{h}"
                if bar_col in events_out.columns:
                    bar_ts = pd.to_datetime(events_out[bar_col], errors="coerce")
                    bar_delta_min = (exit_ts - bar_ts).dt.total_seconds() / 60.0
                    bad_bar_available = bar_delta_min.notna() & (bar_delta_min <= 0)
                    audit_rows.append({"check": f"exit_bar_available_time_flag_h{h}", "fail_count": int(bad_bar_available.sum())})
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out_dir / "08_causal_audit.csv", index=False)

    meta = {
        "symbol": args.symbol,
        "primary_data_source": "OKXTradeBarLoader",
        "ordinary_kline_used": False,
        "primary_timeframe": args.primary_timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "timestamp_tz": args.timestamp_tz,
        "fee_rate": args.fee_rate,
        "horizons": horizons,
        "include_range_bars": bool(args.include_range_bars),
        "range_pcts": parse_csv_floats(args.range_pcts) if args.include_range_bars else [],
        "include_footprint": bool(args.include_footprint),
        "footprint_range_pct": args.footprint_range_pct if args.include_footprint else None,
        "footprint_price_step": args.footprint_price_step if args.include_footprint else None,
        "events": int(len(events_out)),
        "families": int(events_out["family"].nunique()) if not events_out.empty else 0,
    }
    (out_dir / "19_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_brief(out_dir, events_out, rank, args)

    print(f"[done] out_dir={out_dir}")
    if not rank.empty:
        print(rank[["horizon", "family", "event_name", "side", "count", "mean", "pf", "score"]].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
