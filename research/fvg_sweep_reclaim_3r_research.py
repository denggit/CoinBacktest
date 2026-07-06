#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4H Ordinary-Kline Trend + HTF FVG Touch + 15m Sweep + New FVG 3R Research
===========================================================================

Deterministic ICT-style research replay, defaulting to BOTH long and short.

Long structure
--------------
1. 4H ordinary OKX kline is in uptrend.
2. A previous bullish 4H FVG exists and is available only after the 3rd 4H candle closes.
3. 15m trade bar trades back into that 4H FVG.
4. 15m sweeps prior liquidity below, then closes back above the swept level.
5. Within 2-3 closed 15m bars, price impulsively moves up and creates a new bullish 15m FVG.
6. Enter long on retrace into the new 15m FVG using 1m trade bars / 15m trade bars / range bars.
7. Stop is below sweep low, target is fixed RR.

Short structure is the exact mirror:
1. 4H ordinary OKX kline is in downtrend.
2. A previous bearish 4H FVG exists.
3. 15m trade bar trades back into that bearish 4H FVG.
4. 15m sweeps prior liquidity above, then closes back below the swept level.
5. Within 2-3 closed 15m bars, price impulsively moves down and creates a new bearish 15m FVG.
6. Enter short on retrace into the new 15m FVG.
7. Stop is above sweep high, target is fixed RR.

Data policy
-----------
- 4H context: ordinary OKX kline via OKXDataLoader, per user request.
- 15m / 1m: trade-derived bars via OKXTradeBarLoader.
- Optional entry axis: trade-derived range bars via OKXRangeBarLoader.
- 4H/15m context uses bar_available_time = bar_start_time + timeframe.
- Entry bars must start at or after the new 15m FVG available time to avoid straddling-bar leakage.
- Same-bar TP+SL is conservative by default: STOP first.

Example
-------
python research/fvg_sweep_reclaim_3r_research.py --symbol ETH-USDT-SWAP --start-date 2023-01-01 --end-date 2026-06-30 --warmup-start-date 2022-01-01 --entry-axis 1m,range --range-pct 0.0020 --out-dir data/reports/research/fvg_sweep_reclaim_3r_both
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

warnings.simplefilter("ignore", PerformanceWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader, range_code  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402


SUPPORTED_ENTRY_AXES = {"15m", "1m", "range"}
SUPPORTED_SIDES = {"long", "short", "both"}
SIDE_SIGN = {"long": 1, "short": -1}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Research 4H ordinary-kline FVG touch + 15m sweep + new FVG 3R entries, default both sides.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--out-dir", default="data/reports/research/fvg_sweep_reclaim_3r_both")

    # Timeframes / entry axis.
    p.add_argument("--htf-timeframe", default="4H", help="HTF ordinary kline timeframe. Keep 4H for this research.")
    p.add_argument("--signal-timeframe", default="15m", help="Signal trade-bar timeframe.")
    p.add_argument("--ltf-timeframe", default="1m", help="Lower-timeframe trade-bar entry axis.")
    p.add_argument("--entry-axis", default="1m,range", help="Comma-separated from: 1m,15m,range")
    p.add_argument("--range-pct", type=float, default=0.0020)
    p.add_argument("--side", choices=sorted(SUPPORTED_SIDES), default="both", help="long, short, or both. Default is both.")

    # HTF trend and HTF FVG.
    p.add_argument("--htf-ema-fast", type=int, default=20)
    p.add_argument("--htf-ema-slow", type=int, default=50)
    p.add_argument("--htf-slope-lookback", type=int, default=3)
    p.add_argument("--htf-fvg-min-width-pct", type=float, default=0.0010)
    p.add_argument("--htf-fvg-max-age-hours", type=float, default=24 * 30)
    p.add_argument("--htf-fvg-expire-on-invalidation", action="store_true", default=True)
    p.add_argument("--allow-fvg-full-fill", action="store_true", help="Keep HTF FVG active even after deep invalidation.")

    # 15m sweep / impulse / new FVG.
    p.add_argument("--sweep-lookbacks", default="8,16,32", help="Prior 15m high/low lookbacks for liquidity sweep.")
    p.add_argument("--sweep-buffer-pct", type=float, default=0.0000)
    p.add_argument("--min-sweep-wick-frac", type=float, default=0.25)
    p.add_argument("--confirm-bars", default="2,3", help="New 15m FVG must form sweep+N bars after sweep, N in this list.")
    p.add_argument("--min-impulse-ret-pct", type=float, default=0.0020)
    p.add_argument("--new-fvg-min-width-pct", type=float, default=0.0005)

    # Entry / risk.
    p.add_argument("--entry-levels", default="top,mid,bottom", help="FVG entry levels: top,mid,bottom")
    p.add_argument("--entry-wait-minutes", type=int, default=720)
    p.add_argument("--max-hold-minutes", type=int, default=4320)
    p.add_argument("--stop-buffer-pct", type=float, default=0.0003)
    p.add_argument("--rr", type=float, default=3.0)
    p.add_argument("--fee-rate", type=float, default=0.0011, help="Round-trip fee deducted from returns.")
    p.add_argument("--slippage-pct", type=float, default=0.0002, help="One-way conservative slippage on entry and exit.")
    p.add_argument("--same-bar-policy", choices=["conservative", "target", "unknown"], default="conservative")

    # Reporting / speed.
    p.add_argument("--min-count", type=int, default=30)
    p.add_argument("--write-full-path-audit", action="store_true")
    p.add_argument("--event-sample-size", type=int, default=100000)
    p.add_argument("--progress-every", type=int, default=250)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def parse_csv_ints(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v <= 0:
            raise ValueError(f"expected positive int, got {v}")
        out.append(v)
    if not out:
        raise ValueError("empty int list")
    return sorted(set(out))


def parse_csv_tokens(text: str, valid: set[str] | None = None) -> list[str]:
    out = [p.strip().lower() for p in str(text).split(",") if p.strip()]
    if valid is not None:
        bad = sorted(set(out) - set(valid))
        if bad:
            raise ValueError(f"bad values={bad}; valid={sorted(valid)}")
    if not out:
        raise ValueError("empty token list")
    return list(dict.fromkeys(out))


def requested_sides(side_text: str) -> list[str]:
    s = str(side_text).lower().strip()
    if s == "both":
        return ["long", "short"]
    if s in {"long", "short"}:
        return [s]
    raise ValueError(f"bad side={side_text}")


def timeframe_delta(tf: str) -> pd.Timedelta:
    s = str(tf).strip()
    aliases = {
        "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
        "1H": "1h", "4H": "4h", "1D": "1D",
    }
    return pd.Timedelta(aliases.get(s, s))


def profit_factor(x: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    if arr.size == 0:
        return float("nan")
    gp = float(arr[arr > 0].sum())
    gl = float(-arr[arr <= 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else float("nan")
    return gp / gl


def top_winner_share(x: pd.Series | np.ndarray, n: int = 5) -> float:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    wins = np.sort(arr[arr > 0])[::-1]
    if wins.size == 0:
        return float("nan")
    total = float(wins.sum())
    return float(wins[: int(n)].sum() / total) if total > 0 else float("nan")


def summarize_returns(x: pd.Series | np.ndarray, *, min_count: int = 0) -> dict[str, object]:
    s = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if s.empty:
        return {
            "count": 0,
            "eligible": False,
            "mean": np.nan,
            "median": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "top5_winner_share": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p95": np.nan,
        }
    return {
        "count": int(len(s)),
        "eligible": bool(len(s) >= int(min_count)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "win_rate": float((s > 0).mean()),
        "profit_factor": profit_factor(s),
        "top5_winner_share": top_winner_share(s),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
    }


def write_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


# ---------------------------------------------------------------------------
# Loaders / normalization
# ---------------------------------------------------------------------------


def normalize_ohlcv_bars(df: pd.DataFrame, *, timeframe: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy().sort_index()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].copy()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in out.columns:
            raise RuntimeError(f"OHLCV bars missing required column: {c}")
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out["bar_start_time"] = out.index
    out["bar_available_time"] = out.index + timeframe_delta(timeframe)
    return out


def normalize_trade_bars(df: pd.DataFrame, *, timeframe: str) -> pd.DataFrame:
    out = normalize_ohlcv_bars(df, timeframe=timeframe)
    for c in [
        "notional", "delta_notional", "cvd_notional", "taker_buy_ratio", "large_delta_notional",
        "large_trades_count", "max_trade_notional", "trades_count",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_htf_kline_bars(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    print(f"[load] OKXDataLoader ordinary kline {symbol} {timeframe} {start}->{end}", flush=True)
    df = OKXDataLoader(symbol=symbol, timeframe=timeframe).fetch_data_by_date_range(start, end)
    out = normalize_ohlcv_bars(df, timeframe=timeframe)
    if out.empty:
        raise RuntimeError(f"No ordinary kline bars loaded for {symbol} {timeframe} {start}->{end}")
    print(f"       rows={len(out):,} range={out.index[0]} -> {out.index[-1]}", flush=True)
    return out


def load_trade_bars(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    print(f"[load] OKXTradeBarLoader {symbol} {timeframe} {start}->{end}", flush=True)
    df = OKXTradeBarLoader(symbol=symbol, timeframe=timeframe).fetch_data_by_date_range(start, end)
    out = normalize_trade_bars(df, timeframe=timeframe)
    if out.empty:
        raise RuntimeError(f"No trade bars loaded for {symbol} {timeframe} {start}->{end}")
    print(f"       rows={len(out):,} range={out.index[0]} -> {out.index[-1]}", flush=True)
    return out


def load_range_bars(symbol: str, range_pct: float, start: str, end: str) -> pd.DataFrame:
    print(f"[load] OKXRangeBarLoader {symbol} range={range_code(range_pct)} {start}->{end}", flush=True)
    df = OKXRangeBarLoader(symbol=symbol, range_pct=float(range_pct)).fetch_data_by_date_range(start, end, cvd_mode="range")
    if df.empty:
        raise RuntimeError("No range bars loaded. Build range bars first or check DB path.")
    out = df.reset_index(drop=True).copy()
    for c in ["start_ts", "end_ts"]:
        out[c] = pd.to_datetime(out[c], errors="coerce")
    out = out.dropna(subset=["start_ts", "end_ts"])
    out["bar_id"] = pd.to_numeric(out["bar_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["bar_id"])
    out["bar_id"] = out["bar_id"].astype("int64")
    for c in ["open", "high", "low", "close", "direction", "delta_notional", "notional", "taker_buy_ratio", "large_delta_notional"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.sort_values(["end_ts", "bar_id"]).reset_index(drop=True)
    out["bar_available_time"] = out["end_ts"]
    print(f"       rows={len(out):,} range={out['end_ts'].min()} -> {out['end_ts'].max()}", flush=True)
    return out


# ---------------------------------------------------------------------------
# HTF trend/FVG and 15m setup construction
# ---------------------------------------------------------------------------


def build_htf_features(htf: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = htf.copy().sort_index()
    fast = int(args.htf_ema_fast)
    slow = int(args.htf_ema_slow)
    slope_lb = int(args.htf_slope_lookback)
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False, min_periods=fast).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False, min_periods=slow).mean()
    df["ema_fast_slope"] = df["ema_fast"] / df["ema_fast"].shift(slope_lb) - 1.0
    df["uptrend"] = (df["close"] > df["ema_fast"]) & (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast_slope"] > 0)
    df["downtrend"] = (df["close"] < df["ema_fast"]) & (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast_slope"] < 0)

    # Bullish FVG: candle i high < candle i+2 low. Formed when candle i+2 closes.
    df["bull_fvg_lower"] = df["high"].shift(2)
    df["bull_fvg_upper"] = df["low"]
    df["bull_fvg_width_pct"] = df["bull_fvg_upper"] / df["bull_fvg_lower"] - 1.0
    df["bull_fvg"] = (
        df["bull_fvg_lower"].notna()
        & df["bull_fvg_upper"].notna()
        & (df["bull_fvg_upper"] > df["bull_fvg_lower"])
        & (df["bull_fvg_width_pct"] >= float(args.htf_fvg_min_width_pct))
        & df["uptrend"].fillna(False)
    )

    # Bearish FVG: candle i low > candle i+2 high.  Zone is [high(i+2), low(i)].
    df["bear_fvg_lower"] = df["high"]
    df["bear_fvg_upper"] = df["low"].shift(2)
    df["bear_fvg_width_pct"] = df["bear_fvg_upper"] / df["bear_fvg_lower"] - 1.0
    df["bear_fvg"] = (
        df["bear_fvg_lower"].notna()
        & df["bear_fvg_upper"].notna()
        & (df["bear_fvg_upper"] > df["bear_fvg_lower"])
        & (df["bear_fvg_width_pct"] >= float(args.htf_fvg_min_width_pct))
        & df["downtrend"].fillna(False)
    )
    return df


def build_htf_fvg_zones(htf: pd.DataFrame, sides: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    zone_id = 0
    for ts, row in htf.iterrows():
        for side in sides:
            prefix = "bull" if side == "long" else "bear"
            if not bool(row.get(f"{prefix}_fvg", False)):
                continue
            zone_id += 1
            rows.append(
                {
                    "htf_fvg_id": zone_id,
                    "side": side,
                    "htf_fvg_formed_bar_start": row["bar_start_time"],
                    "htf_fvg_available_time": row["bar_available_time"],
                    "htf_fvg_lower": float(row[f"{prefix}_fvg_lower"]),
                    "htf_fvg_upper": float(row[f"{prefix}_fvg_upper"]),
                    "htf_fvg_width_pct": float(row[f"{prefix}_fvg_width_pct"]),
                    "htf_ema_fast": float(row.get("ema_fast", np.nan)),
                    "htf_ema_slow": float(row.get("ema_slow", np.nan)),
                    "htf_ema_fast_slope": float(row.get("ema_fast_slope", np.nan)),
                    "source_i0_time": htf["bar_start_time"].shift(2).loc[ts] if ts in htf.index else pd.NaT,
                    "source_i2_time": row["bar_start_time"],
                }
            )
    return pd.DataFrame(rows)


def attach_active_htf_fvg_for_side(signal: pd.DataFrame, zones: pd.DataFrame, args: argparse.Namespace, side: str) -> pd.DataFrame:
    """Attach latest available active 4H FVG touched by each 15m bar for one side."""
    df = signal.copy().sort_values("bar_available_time").reset_index(drop=False).rename(columns={"index": "timestamp"})
    side_zones = zones[zones.get("side", pd.Series(dtype=object)).astype(str) == side].copy() if not zones.empty else pd.DataFrame()
    out_keys = ["htf_fvg_id", "htf_fvg_lower", "htf_fvg_upper", "htf_fvg_width_pct", "htf_fvg_available_time", "htf_fvg_age_hours"]
    if side_zones.empty:
        for c in out_keys:
            df[c] = pd.NaT if c == "htf_fvg_available_time" else np.nan
        df["setup_side"] = side
        return df.set_index("timestamp", drop=True).sort_index()

    zones_sorted = side_zones.sort_values("htf_fvg_available_time").reset_index(drop=True)
    z_ptr = 0
    active: list[dict[str, object]] = []
    max_age = pd.Timedelta(hours=float(args.htf_fvg_max_age_hours))
    out_cols = {k: [] for k in out_keys}
    lows = df["low"].to_numpy(float)
    highs = df["high"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    times = pd.to_datetime(df["bar_available_time"]).to_numpy(dtype="datetime64[ns]")

    for i in range(len(df)):
        now = pd.Timestamp(times[i])
        while z_ptr < len(zones_sorted) and pd.Timestamp(zones_sorted.loc[z_ptr, "htf_fvg_available_time"]) <= now:
            active.append(zones_sorted.loc[z_ptr].to_dict())
            z_ptr += 1
        if active:
            kept: list[dict[str, object]] = []
            for z in active:
                z_avail = pd.Timestamp(z["htf_fvg_available_time"])
                if now - z_avail > max_age:
                    continue
                lower = float(z["htf_fvg_lower"])
                upper = float(z["htf_fvg_upper"])
                if bool(args.htf_fvg_expire_on_invalidation) and not bool(args.allow_fvg_full_fill):
                    if side == "long" and closes[i] < lower:
                        continue
                    if side == "short" and closes[i] > upper:
                        continue
                kept.append(z)
            active = kept
        touched: dict[str, object] | None = None
        if active and np.isfinite(lows[i]) and np.isfinite(highs[i]):
            for z in reversed(active):
                lower = float(z["htf_fvg_lower"])
                upper = float(z["htf_fvg_upper"])
                if highs[i] >= lower and lows[i] <= upper:
                    touched = z
                    break
        if touched is None:
            for k in out_cols:
                out_cols[k].append(pd.NaT if k == "htf_fvg_available_time" else np.nan)
        else:
            out_cols["htf_fvg_id"].append(int(touched["htf_fvg_id"]))
            out_cols["htf_fvg_lower"].append(float(touched["htf_fvg_lower"]))
            out_cols["htf_fvg_upper"].append(float(touched["htf_fvg_upper"]))
            out_cols["htf_fvg_width_pct"].append(float(touched["htf_fvg_width_pct"]))
            out_cols["htf_fvg_available_time"].append(pd.Timestamp(touched["htf_fvg_available_time"]))
            age_h = (now - pd.Timestamp(touched["htf_fvg_available_time"])).total_seconds() / 3600.0
            out_cols["htf_fvg_age_hours"].append(float(age_h))
    for c, values in out_cols.items():
        df[c] = values
    df["setup_side"] = side
    return df.set_index("timestamp", drop=True).sort_index()


def add_15m_sweep_and_fvg_features(sig: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = sig.copy().sort_index()
    low = df["low"]
    high = df["high"]
    open_ = df["open"]
    close = df["close"]
    body_low = np.minimum(open_, close)
    body_high = np.maximum(open_, close)
    rng = (high - low).replace(0.0, np.nan)
    df["lower_wick_frac"] = (body_low - low) / rng
    df["upper_wick_frac"] = (high - body_high) / rng
    df["bar_ret"] = close / open_ - 1.0
    df["ret_3"] = close / close.shift(3) - 1.0
    df["ret_8"] = close / close.shift(8) - 1.0
    df["tr"] = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    df["atr_20_pct"] = df["tr"].rolling(20, min_periods=10).mean() / close
    if "delta_notional" in df.columns:
        denom = pd.to_numeric(df.get("notional", pd.Series(index=df.index)), errors="coerce").replace(0, np.nan)
        df["delta_ratio"] = df["delta_notional"] / denom
        df["delta_sum_4"] = df["delta_notional"].rolling(4, min_periods=2).sum()
    if "taker_buy_ratio" in df.columns:
        df["taker_buy_ratio"] = pd.to_numeric(df["taker_buy_ratio"], errors="coerce")

    # Bullish 15m FVG formed on current bar: high[t-2] < low[t].
    df["new_bull_fvg_lower"] = high.shift(2)
    df["new_bull_fvg_upper"] = low
    df["new_bull_fvg_width_pct"] = df["new_bull_fvg_upper"] / df["new_bull_fvg_lower"] - 1.0
    df["new_bull_fvg"] = (
        df["new_bull_fvg_lower"].notna()
        & (df["new_bull_fvg_upper"] > df["new_bull_fvg_lower"])
        & (df["new_bull_fvg_width_pct"] >= float(args.new_fvg_min_width_pct))
    )

    # Bearish 15m FVG formed on current bar: low[t-2] > high[t].
    df["new_bear_fvg_lower"] = high
    df["new_bear_fvg_upper"] = low.shift(2)
    df["new_bear_fvg_width_pct"] = df["new_bear_fvg_upper"] / df["new_bear_fvg_lower"] - 1.0
    df["new_bear_fvg"] = (
        df["new_bear_fvg_lower"].notna()
        & (df["new_bear_fvg_upper"] > df["new_bear_fvg_lower"])
        & (df["new_bear_fvg_width_pct"] >= float(args.new_fvg_min_width_pct))
    )

    for lb in parse_csv_ints(args.sweep_lookbacks):
        prior_low = low.shift(1).rolling(lb, min_periods=max(3, lb // 2)).min()
        prior_high = high.shift(1).rolling(lb, min_periods=max(3, lb // 2)).max()
        df[f"prior_low_{lb}"] = prior_low
        df[f"prior_high_{lb}"] = prior_high
        df[f"sweep_reclaim_long_{lb}"] = (
            prior_low.notna()
            & (low < prior_low * (1.0 - float(args.sweep_buffer_pct)))
            & (close > prior_low)
            & (df["lower_wick_frac"] >= float(args.min_sweep_wick_frac))
        )
        df[f"sweep_reclaim_short_{lb}"] = (
            prior_high.notna()
            & (high > prior_high * (1.0 + float(args.sweep_buffer_pct)))
            & (close < prior_high)
            & (df["upper_wick_frac"] >= float(args.min_sweep_wick_frac))
        )
    return df


def build_setups(sig: pd.DataFrame, args: argparse.Namespace, *, side: str) -> pd.DataFrame:
    df = sig.copy().sort_index().reset_index(drop=True)
    if "bar_start_time" not in df.columns:
        raise RuntimeError("signal frame missing bar_start_time")
    df["signal_bar_start"] = pd.to_datetime(df["bar_start_time"], errors="coerce")
    df["has_htf_fvg_touch"] = df["htf_fvg_id"].notna()
    confirm_bars = parse_csv_ints(args.confirm_bars)
    sweep_lbs = parse_csv_ints(args.sweep_lookbacks)
    rows: list[dict[str, object]] = []
    prog = ProgressReporter(
        label=f"[setups] {side} sweep+new-fvg scan",
        total=len(df),
        every=max(1, int(args.progress_every)),
        enabled=not bool(args.no_progress),
    )
    fvg_prefix = "bull" if side == "long" else "bear"
    for i in range(len(df)):
        prog.update(i + 1)
        if not bool(df.loc[i, "has_htf_fvg_touch"]):
            continue
        for lb in sweep_lbs:
            sweep_col = f"sweep_reclaim_{side}_{lb}"
            if not bool(df.loc[i, sweep_col]):
                continue
            for n in confirm_bars:
                j = i + n
                if j >= len(df):
                    continue
                if not bool(df.loc[j, f"new_{fvg_prefix}_fvg"]):
                    continue
                sweep_close = float(df.loc[i, "close"])
                impulse_close = float(df.loc[j, "close"])
                if sweep_close <= 0 or impulse_close <= 0:
                    continue
                if side == "long":
                    if impulse_close / sweep_close - 1.0 < float(args.min_impulse_ret_pct):
                        continue
                    if float(df.loc[j, "close"]) <= float(df.loc[i, f"prior_low_{lb}"]):
                        continue
                    sweep_level = float(df.loc[i, f"prior_low_{lb}"])
                    sweep_extreme = float(df.loc[i, "low"])
                    sweep_wick = float(df.loc[i, "lower_wick_frac"])
                else:
                    if sweep_close / impulse_close - 1.0 < float(args.min_impulse_ret_pct):
                        continue
                    if float(df.loc[j, "close"]) >= float(df.loc[i, f"prior_high_{lb}"]):
                        continue
                    sweep_level = float(df.loc[i, f"prior_high_{lb}"])
                    sweep_extreme = float(df.loc[i, "high"])
                    sweep_wick = float(df.loc[i, "upper_wick_frac"])
                lower = float(df.loc[j, f"new_{fvg_prefix}_fvg_lower"])
                upper = float(df.loc[j, f"new_{fvg_prefix}_fvg_upper"])
                if not (np.isfinite(lower) and np.isfinite(upper) and upper > lower):
                    continue
                setup_id = len(rows) + 1
                rows.append(
                    {
                        "setup_id": setup_id,
                        "side": side,
                        "sweep_bar_pos": int(i),
                        "confirm_bar_pos": int(j),
                        "sweep_lookback": int(lb),
                        "confirm_bars_after_sweep": int(n),
                        "sweep_bar_start": df.loc[i, "signal_bar_start"],
                        "sweep_signal_time": df.loc[i, "bar_available_time"],
                        "confirm_bar_start": df.loc[j, "signal_bar_start"],
                        "new_fvg_available_time": df.loc[j, "bar_available_time"],
                        "htf_fvg_id": int(df.loc[i, "htf_fvg_id"]),
                        "htf_fvg_available_time": df.loc[i, "htf_fvg_available_time"],
                        "htf_fvg_lower": float(df.loc[i, "htf_fvg_lower"]),
                        "htf_fvg_upper": float(df.loc[i, "htf_fvg_upper"]),
                        "htf_fvg_width_pct": float(df.loc[i, "htf_fvg_width_pct"]),
                        "htf_fvg_age_hours": float(df.loc[i, "htf_fvg_age_hours"]),
                        "sweep_level": sweep_level,
                        "sweep_extreme": sweep_extreme,
                        "sweep_low": float(df.loc[i, "low"]),
                        "sweep_high": float(df.loc[i, "high"]),
                        "sweep_close": float(df.loc[i, "close"]),
                        "sweep_wick_frac": sweep_wick,
                        "impulse_close": impulse_close,
                        "impulse_ret_from_sweep_close": float(SIDE_SIGN[side] * (impulse_close / sweep_close - 1.0)),
                        "new_fvg_lower": lower,
                        "new_fvg_upper": upper,
                        "new_fvg_mid": float((lower + upper) / 2.0),
                        "new_fvg_width_pct": float(df.loc[j, f"new_{fvg_prefix}_fvg_width_pct"]),
                        "sig_ret_3": float(df.loc[i, "ret_3"]) if pd.notna(df.loc[i, "ret_3"]) else np.nan,
                        "sig_ret_8": float(df.loc[i, "ret_8"]) if pd.notna(df.loc[i, "ret_8"]) else np.nan,
                        "sig_atr_20_pct": float(df.loc[i, "atr_20_pct"]) if pd.notna(df.loc[i, "atr_20_pct"]) else np.nan,
                        "sig_delta_ratio": float(df.loc[i, "delta_ratio"]) if "delta_ratio" in df.columns and pd.notna(df.loc[i, "delta_ratio"]) else np.nan,
                        "sig_taker_buy_ratio": float(df.loc[i, "taker_buy_ratio"]) if "taker_buy_ratio" in df.columns and pd.notna(df.loc[i, "taker_buy_ratio"]) else np.nan,
                    }
                )
    prog.close()
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["new_fvg_available_time", "setup_id"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Entry / TP-SL replay
# ---------------------------------------------------------------------------


def make_entry_price(row: pd.Series, level: str) -> float:
    if level == "top":
        return float(row["new_fvg_upper"])
    if level == "mid":
        return float(row["new_fvg_mid"])
    if level == "bottom":
        return float(row["new_fvg_lower"])
    raise ValueError(f"bad entry level: {level}")


def prepare_entry_axis(name: str, trade_15m: pd.DataFrame, ltf: pd.DataFrame | None, range_bars: pd.DataFrame | None) -> pd.DataFrame:
    if name == "15m":
        axis = trade_15m.copy()
        axis["axis_time"] = axis["bar_start_time"]
        axis["axis_available_time"] = axis["bar_available_time"]
        return axis
    if name == "1m":
        if ltf is None or ltf.empty:
            raise RuntimeError("1m entry axis requested but ltf bars are empty")
        axis = ltf.copy()
        axis["axis_time"] = axis["bar_start_time"]
        axis["axis_available_time"] = axis["bar_available_time"]
        return axis
    if name == "range":
        if range_bars is None or range_bars.empty:
            raise RuntimeError("range entry axis requested but range bars are empty")
        axis = range_bars.copy()
        axis["axis_time"] = axis["start_ts"]
        axis["axis_available_time"] = axis["bar_available_time"]
        return axis
    raise ValueError(f"bad entry axis: {name}")


def replay_setups_on_axis(setups: pd.DataFrame, axis: pd.DataFrame, *, axis_name: str, entry_level: str, args: argparse.Namespace) -> pd.DataFrame:
    if setups.empty or axis.empty:
        return pd.DataFrame()
    ax = axis.sort_values("axis_available_time").reset_index(drop=True).copy()
    for c in ["open", "high", "low", "close"]:
        ax[c] = pd.to_numeric(ax[c], errors="coerce")
    axis_times = pd.to_datetime(ax["axis_time"]).to_numpy(dtype="datetime64[ns]")
    lows = ax["low"].to_numpy(float)
    highs = ax["high"].to_numpy(float)
    closes = ax["close"].to_numpy(float)

    rows: list[dict[str, object]] = []
    prog = ProgressReporter(
        label=f"[replay] axis={axis_name} level={entry_level}",
        total=len(setups),
        every=max(1, int(args.progress_every)),
        enabled=not bool(args.no_progress),
    )
    entry_wait = pd.Timedelta(minutes=int(args.entry_wait_minutes))
    max_hold = pd.Timedelta(minutes=int(args.max_hold_minutes))
    rr = float(args.rr)
    fee_rate = float(args.fee_rate)
    slip = float(args.slippage_pct)

    for n, (_, s) in enumerate(setups.iterrows(), start=1):
        prog.update(n)
        side = str(s["side"])
        start_time = pd.Timestamp(s["new_fvg_available_time"])
        deadline = start_time + entry_wait
        start_pos = int(np.searchsorted(axis_times, np.datetime64(start_time), side="left"))
        deadline_pos = int(np.searchsorted(axis_times, np.datetime64(deadline), side="right"))
        if start_pos >= len(ax) or deadline_pos <= start_pos:
            continue
        entry_limit = make_entry_price(s, entry_level)
        if side == "long":
            fill_mask = lows[start_pos:deadline_pos] <= entry_limit
        else:
            fill_mask = highs[start_pos:deadline_pos] >= entry_limit
        if not fill_mask.any():
            rows.append({**s.to_dict(), "entry_axis": axis_name, "entry_level_name": entry_level, "filled": False, "exit_reason": "NO_FILL"})
            continue
        entry_pos = start_pos + int(np.argmax(fill_mask))
        entry_time = pd.Timestamp(ax.loc[entry_pos, "axis_time"])
        entry_available_time = pd.Timestamp(ax.loc[entry_pos, "axis_available_time"])

        if side == "long":
            entry_price = float(entry_limit * (1.0 + slip))
            raw_stop = float(s["sweep_low"] * (1.0 - float(args.stop_buffer_pct)))
            stop_price = float(raw_stop * (1.0 - slip))
            if not np.isfinite(entry_price) or not np.isfinite(stop_price) or stop_price >= entry_price:
                rows.append({**s.to_dict(), "entry_axis": axis_name, "entry_level_name": entry_level, "filled": False, "exit_reason": "BAD_RISK"})
                continue
            risk = entry_price - stop_price
            target_price = float(entry_price + rr * risk)
        else:
            entry_price = float(entry_limit * (1.0 - slip))
            raw_stop = float(s["sweep_high"] * (1.0 + float(args.stop_buffer_pct)))
            stop_price = float(raw_stop * (1.0 + slip))
            if not np.isfinite(entry_price) or not np.isfinite(stop_price) or stop_price <= entry_price:
                rows.append({**s.to_dict(), "entry_axis": axis_name, "entry_level_name": entry_level, "filled": False, "exit_reason": "BAD_RISK"})
                continue
            risk = stop_price - entry_price
            target_price = float(entry_price - rr * risk)
            if target_price <= 0:
                rows.append({**s.to_dict(), "entry_axis": axis_name, "entry_level_name": entry_level, "filled": False, "exit_reason": "BAD_TARGET"})
                continue

        max_exit_time = entry_time + max_hold
        end_pos = int(np.searchsorted(axis_times, np.datetime64(max_exit_time), side="right"))
        end_pos = min(end_pos, len(ax))
        if end_pos <= entry_pos:
            continue
        exit_reason = "TIMEOUT"
        exit_pos = end_pos - 1
        same_bar_both = False
        entry_bar_target_ambiguous = False

        for pos in range(entry_pos, end_pos):
            if side == "long":
                hit_stop = lows[pos] <= stop_price
                hit_target = highs[pos] >= target_price
            else:
                hit_stop = highs[pos] >= stop_price
                hit_target = lows[pos] <= target_price
            if pos == entry_pos:
                if hit_stop:
                    exit_reason = "STOP"
                    exit_pos = pos
                    same_bar_both = bool(hit_target)
                    break
                if hit_target:
                    entry_bar_target_ambiguous = True
                    continue
                continue
            if hit_stop and hit_target:
                same_bar_both = True
                if args.same_bar_policy == "target":
                    exit_reason = "TARGET"
                elif args.same_bar_policy == "unknown":
                    exit_reason = "BOTH_UNKNOWN"
                else:
                    exit_reason = "STOP"
                exit_pos = pos
                break
            if hit_stop:
                exit_reason = "STOP"
                exit_pos = pos
                break
            if hit_target:
                exit_reason = "TARGET"
                exit_pos = pos
                break

        if exit_reason == "TARGET":
            exit_price = float(target_price * (1.0 - slip)) if side == "long" else float(target_price * (1.0 + slip))
        elif exit_reason == "STOP":
            exit_price = float(stop_price * (1.0 - slip)) if side == "long" else float(stop_price * (1.0 + slip))
        elif exit_reason == "BOTH_UNKNOWN":
            exit_price = float(closes[exit_pos])
        else:
            exit_price = float(closes[exit_pos] * (1.0 - slip)) if side == "long" else float(closes[exit_pos] * (1.0 + slip))

        gross_ret = float(SIDE_SIGN[side] * (exit_price / entry_price - 1.0))
        net_ret = gross_ret - fee_rate
        high_path = highs[entry_pos:end_pos]
        low_path = lows[entry_pos:end_pos]
        if side == "long":
            mfe = float(np.nanmax(high_path) / entry_price - 1.0) if high_path.size else np.nan
            mae = float(np.nanmin(low_path) / entry_price - 1.0) if low_path.size else np.nan
        else:
            mfe = float(1.0 - np.nanmin(low_path) / entry_price) if low_path.size else np.nan
            mae = float(1.0 - np.nanmax(high_path) / entry_price) if high_path.size else np.nan
        rows.append(
            {
                **s.to_dict(),
                "entry_axis": axis_name,
                "entry_level_name": entry_level,
                "filled": True,
                "entry_time": entry_time,
                "entry_available_time": entry_available_time,
                "entry_price": entry_price,
                "entry_limit_price": entry_limit,
                "stop_price": stop_price,
                "target_price": target_price,
                "risk_pct": risk / entry_price,
                "exit_bar_start": pd.Timestamp(ax.loc[exit_pos, "axis_time"]),
                "exit_time": pd.Timestamp(ax.loc[exit_pos, "axis_available_time"]),
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "same_bar_both_hit_flag": bool(same_bar_both),
                "entry_bar_target_ambiguous_flag": bool(entry_bar_target_ambiguous),
                "gross_return": gross_ret,
                "net_return": net_ret,
                "mfe": mfe,
                "mae": mae,
                "r_multiple_gross": gross_ret / (risk / entry_price) if risk > 0 else np.nan,
                "bars_held_on_axis": int(exit_pos - entry_pos + 1),
                "minutes_to_fill": (entry_time - start_time).total_seconds() / 60.0,
                "minutes_held": (pd.Timestamp(ax.loc[exit_pos, "axis_available_time"]) - entry_time).total_seconds() / 60.0,
            }
        )
    prog.close()
    out = pd.DataFrame(rows)
    if not out.empty:
        out["variant_name"] = (
            out["side"].astype(str) + "_" + out["entry_axis"].astype(str) + "_" + out["entry_level_name"].astype(str)
            + "_sw" + out["sweep_lookback"].astype(str) + "_cf" + out["confirm_bars_after_sweep"].astype(str)
        )
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize_trades(trades: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    filled = trades[trades.get("filled", False).astype(bool)].copy()
    if filled.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    groups = ["side", "variant_name", "entry_axis", "entry_level_name", "sweep_lookback", "confirm_bars_after_sweep"]
    for key, g in filled.groupby(groups, dropna=False):
        row = dict(zip(groups, key, strict=False))
        ret = pd.to_numeric(g["net_return"], errors="coerce")
        row.update(summarize_returns(ret, min_count=int(args.min_count)))
        row["target_rate"] = float((g["exit_reason"] == "TARGET").mean())
        row["stop_rate"] = float((g["exit_reason"] == "STOP").mean())
        row["timeout_rate"] = float((g["exit_reason"] == "TIMEOUT").mean())
        row["same_bar_both_rate"] = float(pd.to_numeric(g["same_bar_both_hit_flag"], errors="coerce").fillna(0).mean())
        row["avg_risk_pct"] = float(pd.to_numeric(g["risk_pct"], errors="coerce").mean())
        row["median_minutes_to_fill"] = float(pd.to_numeric(g["minutes_to_fill"], errors="coerce").median())
        row["median_minutes_held"] = float(pd.to_numeric(g["minutes_held"], errors="coerce").median())
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["eligible", "profit_factor", "mean", "count"], ascending=[False, False, False, False]).reset_index(drop=True)


def combined_side_summary(trades: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    filled = trades[trades.get("filled", False).astype(bool)].copy()
    if filled.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for key_name, group_cols in {
        "combined_all": [],
        "by_side": ["side"],
        "by_axis_side": ["side", "entry_axis"],
    }.items():
        if group_cols:
            iterable = filled.groupby(group_cols, dropna=False)
        else:
            iterable = [("combined", filled)]
        for key, g in iterable:
            row: dict[str, object] = {"summary_level": key_name}
            if group_cols:
                key_tuple = key if isinstance(key, tuple) else (key,)
                row.update(dict(zip(group_cols, key_tuple, strict=False)))
            row.update(summarize_returns(g["net_return"], min_count=int(args.min_count)))
            row["target_rate"] = float((g["exit_reason"] == "TARGET").mean())
            row["stop_rate"] = float((g["exit_reason"] == "STOP").mean())
            rows.append(row)
    return pd.DataFrame(rows)


def yearly_summary(trades: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if trades.empty or "entry_time" not in trades.columns:
        return pd.DataFrame()
    filled = trades[trades.get("filled", False).astype(bool)].copy()
    if filled.empty:
        return pd.DataFrame()
    filled["year"] = pd.to_datetime(filled["entry_time"], errors="coerce").dt.year
    rows: list[dict[str, object]] = []
    for (side, variant, year), g in filled.groupby(["side", "variant_name", "year"], dropna=False):
        row = {"side": side, "variant_name": variant, "year": year}
        row.update(summarize_returns(g["net_return"], min_count=0))
        row["target_rate"] = float((g["exit_reason"] == "TARGET").mean())
        row["stop_rate"] = float((g["exit_reason"] == "STOP").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["side", "variant_name", "year"]).reset_index(drop=True)


def monthly_summary(trades: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if trades.empty or "entry_time" not in trades.columns:
        return pd.DataFrame()
    filled = trades[trades.get("filled", False).astype(bool)].copy()
    if filled.empty:
        return pd.DataFrame()
    filled["month"] = pd.to_datetime(filled["entry_time"], errors="coerce").dt.to_period("M").astype(str)
    rows: list[dict[str, object]] = []
    for (side, variant, month), g in filled.groupby(["side", "variant_name", "month"], dropna=False):
        row = {"side": side, "variant_name": variant, "month": month}
        row.update(summarize_returns(g["net_return"], min_count=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["side", "variant_name", "month"]).reset_index(drop=True)


def build_funnel(setups: pd.DataFrame, trades: pd.DataFrame, zones: pd.DataFrame, sig_by_side: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append({"side": "combined", "stage": "htf_fvg_zones", "count": int(len(zones))})
    rows.append({"side": "combined", "stage": "sweep_plus_new_fvg_setups", "count": int(len(setups))})
    for side in ["long", "short"]:
        zc = int((zones.get("side", pd.Series(dtype=object)).astype(str) == side).sum()) if not zones.empty else 0
        sc = int((setups.get("side", pd.Series(dtype=object)).astype(str) == side).sum()) if not setups.empty else 0
        tc = 0
        if side in sig_by_side and not sig_by_side[side].empty and "htf_fvg_id" in sig_by_side[side].columns:
            tc = int(sig_by_side[side]["htf_fvg_id"].notna().sum())
        rows.extend([
            {"side": side, "stage": "htf_fvg_zones", "count": zc},
            {"side": side, "stage": "15m_htf_fvg_touch_bars", "count": tc},
            {"side": side, "stage": "sweep_plus_new_fvg_setups", "count": sc},
        ])
    if not trades.empty:
        rows.append({"side": "combined", "stage": "entry_attempts", "count": int(len(trades))})
        rows.append({"side": "combined", "stage": "filled_trades", "count": int(trades.get("filled", False).astype(bool).sum())})
        rows.append({"side": "combined", "stage": "target_hits", "count": int((trades.get("exit_reason", "") == "TARGET").sum())})
        rows.append({"side": "combined", "stage": "stop_hits", "count": int((trades.get("exit_reason", "") == "STOP").sum())})
        for side, g in trades.groupby("side", dropna=False):
            rows.append({"side": side, "stage": "entry_attempts", "count": int(len(g))})
            rows.append({"side": side, "stage": "filled_trades", "count": int(g.get("filled", False).astype(bool).sum())})
            rows.append({"side": side, "stage": "target_hits", "count": int((g.get("exit_reason", "") == "TARGET").sum())})
            rows.append({"side": side, "stage": "stop_hits", "count": int((g.get("exit_reason", "") == "STOP").sum())})
    return pd.DataFrame(rows)


def build_audit(args: argparse.Namespace, setups: pd.DataFrame, trades: pd.DataFrame, meta: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append({"check": "ordinary_kline_used_for_htf_4h", "value": 1})
    rows.append({"check": "ordinary_kline_used_for_signal_or_entry", "value": 0})
    if not setups.empty:
        rows.append({"check": "htf_context_after_sweep_signal", "value": int((pd.to_datetime(setups["htf_fvg_available_time"]) > pd.to_datetime(setups["sweep_signal_time"])).sum())})
        rows.append({"check": "new_fvg_available_before_sweep_signal", "value": int((pd.to_datetime(setups["new_fvg_available_time"]) <= pd.to_datetime(setups["sweep_signal_time"])).sum())})
        long_bad = setups[setups["side"] == "long"]
        short_bad = setups[setups["side"] == "short"]
        bad_long = int((pd.to_numeric(long_bad["sweep_low"], errors="coerce") >= pd.to_numeric(long_bad["new_fvg_upper"], errors="coerce")).sum()) if not long_bad.empty else 0
        bad_short = int((pd.to_numeric(short_bad["sweep_high"], errors="coerce") <= pd.to_numeric(short_bad["new_fvg_lower"], errors="coerce")).sum()) if not short_bad.empty else 0
        rows.append({"check": "bad_long_stop_above_entry_zone", "value": bad_long})
        rows.append({"check": "bad_short_stop_below_entry_zone", "value": bad_short})
    if not trades.empty and "entry_time" in trades.columns:
        filled = trades[trades.get("filled", False).astype(bool)].copy()
        rows.append({"check": "filled_entry_before_new_fvg_available", "value": int((pd.to_datetime(filled["entry_time"]) < pd.to_datetime(filled["new_fvg_available_time"])).sum()) if not filled.empty else 0})
        rows.append({"check": "exit_before_entry", "value": int((pd.to_datetime(filled["exit_time"]) < pd.to_datetime(filled["entry_time"])).sum()) if not filled.empty else 0})
        rows.append({"check": "same_bar_both_hit", "value": int(pd.to_numeric(filled.get("same_bar_both_hit_flag", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not filled.empty else 0})
        rows.append({"check": "entry_bar_target_ambiguous", "value": int(pd.to_numeric(filled.get("entry_bar_target_ambiguous_flag", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not filled.empty else 0})
    for k, v in meta.items():
        rows.append({"check": str(k), "value": v})
    return pd.DataFrame(rows)


def make_brief(args: argparse.Namespace, funnel: pd.DataFrame, summary: pd.DataFrame, side_summary: pd.DataFrame, audit: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# FVG Sweep Reclaim 3R Research Brief")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- symbol: `{args.symbol}`")
    lines.append(f"- side: `{args.side}`")
    lines.append(f"- range: `{args.start_date}` -> `{args.end_date}`, warmup `{args.warmup_start_date}`")
    lines.append(f"- data: HTF `{args.htf_timeframe}` uses `OKXDataLoader` ordinary kline; signal `{args.signal_timeframe}` / LTF `{args.ltf_timeframe}` use `OKXTradeBarLoader`; optional range uses `OKXRangeBarLoader`.")
    lines.append(f"- entry_axis: `{args.entry_axis}`; RR={args.rr}:1; fee={args.fee_rate:.4%}; slippage one-way={args.slippage_pct:.4%}")
    lines.append("")
    lines.append("## Funnel")
    lines.append("No funnel output." if funnel.empty else funnel.to_markdown(index=False))
    lines.append("")
    lines.append("## Audit")
    lines.append("No audit output." if audit.empty else audit.head(80).to_markdown(index=False))
    lines.append("")
    lines.append("## Side / combined summary")
    if side_summary.empty:
        lines.append("No side summary generated.")
    else:
        cols = ["summary_level", "side", "entry_axis", "count", "mean", "median", "win_rate", "profit_factor", "target_rate", "stop_rate", "top5_winner_share"]
        lines.append(side_summary[[c for c in cols if c in side_summary.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Top variants")
    if summary.empty:
        lines.append("No filled trade summary generated.")
    else:
        cols = ["side", "variant_name", "count", "mean", "median", "win_rate", "profit_factor", "target_rate", "stop_rate", "top5_winner_share", "avg_risk_pct"]
        lines.append(summary[[c for c in cols if c in summary.columns]].head(30).to_markdown(index=False))
    lines.append("")
    lines.append("## Notes")
    lines.append("- This is a formal research replay, not a live strategy.")
    lines.append("- Same-bar TP+SL is conservative by default: STOP first.")
    lines.append("- Long/short are mirrored; check each side separately before considering combined results.")
    lines.append("- A candidate is not usable unless yearly stability, delay/cost stress, and overlap de-dup pass later.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sides = requested_sides(args.side)
    entry_axes = parse_csv_tokens(args.entry_axis, SUPPORTED_ENTRY_AXES)
    entry_levels = parse_csv_tokens(args.entry_levels, {"top", "mid", "bottom"})

    htf = load_htf_kline_bars(args.symbol, args.htf_timeframe, args.warmup_start_date, args.end_date)
    sig = load_trade_bars(args.symbol, args.signal_timeframe, args.warmup_start_date, args.end_date)
    ltf = None
    if "1m" in entry_axes:
        ltf = load_trade_bars(args.symbol, args.ltf_timeframe, args.warmup_start_date, args.end_date)
    range_bars = None
    if "range" in entry_axes:
        range_bars = load_range_bars(args.symbol, float(args.range_pct), args.warmup_start_date, args.end_date)

    print("[features] build 4H ordinary-kline trend/FVG", flush=True)
    htf_feat = build_htf_features(htf, args)
    zones = build_htf_fvg_zones(htf_feat, sides)
    print(f"[htf_fvg] zones={len(zones):,} sides={sides}", flush=True)

    print("[features] build 15m sweep/new-FVG features", flush=True)
    sig_base = add_15m_sweep_and_fvg_features(sig, args)

    sig_by_side: dict[str, pd.DataFrame] = {}
    setups_parts: list[pd.DataFrame] = []
    for side in sides:
        print(f"[features] attach active 4H FVG to 15m side={side}", flush=True)
        sig_side = attach_active_htf_fvg_for_side(sig_base, zones, args, side)
        sig_by_side[side] = sig_side
        print(f"[setups] build {side} sweep + new 15m FVG setups", flush=True)
        part = build_setups(sig_side, args, side=side)
        if not part.empty:
            setups_parts.append(part)
    setups = pd.concat(setups_parts, ignore_index=True) if setups_parts else pd.DataFrame()
    if not setups.empty:
        setups = setups[pd.to_datetime(setups["new_fvg_available_time"]) >= pd.Timestamp(args.start_date)].copy()
        setups = setups[pd.to_datetime(setups["new_fvg_available_time"]) <= pd.Timestamp(args.end_date) + pd.Timedelta(days=1)].copy()
        setups = setups.sort_values(["new_fvg_available_time", "side", "setup_id"]).reset_index(drop=True)
    print(f"[setups] rows={len(setups):,}", flush=True)

    all_trades: list[pd.DataFrame] = []
    axis_cache: dict[str, pd.DataFrame] = {}
    for axis_name in entry_axes:
        axis_cache[axis_name] = prepare_entry_axis(axis_name, sig, ltf, range_bars)
        for level in entry_levels:
            trades_part = replay_setups_on_axis(setups, axis_cache[axis_name], axis_name=axis_name, entry_level=level, args=args)
            if not trades_part.empty:
                all_trades.append(trades_part)
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    # Output core tables.
    write_csv(zones, out_dir / "00_htf_fvg_zones.csv")
    signal_samples: list[pd.DataFrame] = []
    for side, frame in sig_by_side.items():
        ss = frame.tail(min(len(frame), max(1, int(args.event_sample_size) // max(1, len(sig_by_side))))).copy()
        ss["sample_side"] = side
        signal_samples.append(ss.reset_index())
    write_csv(pd.concat(signal_samples, ignore_index=True) if signal_samples else pd.DataFrame(), out_dir / "01_signal_feature_sample.csv")
    write_csv(setups, out_dir / "02_sweep_new_fvg_setups.csv")
    if not trades.empty:
        sample = trades if bool(args.write_full_path_audit) else trades.head(int(args.event_sample_size))
        write_csv(sample, out_dir / "03_trades_sample.csv")
        if bool(args.write_full_path_audit):
            write_csv(trades, out_dir / "03_trades_full.csv")
    else:
        write_csv(pd.DataFrame(), out_dir / "03_trades_sample.csv")

    summary = summarize_trades(trades, args)
    side_summary = combined_side_summary(trades, args)
    yearly = yearly_summary(trades, args)
    monthly = monthly_summary(trades, args)
    funnel = build_funnel(setups, trades, zones, sig_by_side)
    meta = {
        "ordinary_kline_used_for_htf_4h": 1,
        "ordinary_kline_used_for_signal_or_entry": 0,
        "side_arg": args.side,
        "sides": ",".join(sides),
        "htf_rows": int(len(htf)),
        "signal_rows": int(len(sig)),
        "ltf_rows": int(len(ltf)) if ltf is not None else 0,
        "range_rows": int(len(range_bars)) if range_bars is not None else 0,
        "htf_fvg_zones": int(len(zones)),
        "setups": int(len(setups)),
        "trade_attempt_rows": int(len(trades)),
        "filled_trades": int(trades.get("filled", pd.Series(dtype=bool)).astype(bool).sum()) if not trades.empty else 0,
    }
    audit = build_audit(args, setups, trades, meta)

    write_csv(funnel, out_dir / "04_funnel.csv")
    write_csv(summary, out_dir / "05_variant_summary.csv")
    write_csv(yearly, out_dir / "06_yearly_summary.csv")
    write_csv(monthly, out_dir / "07_monthly_summary.csv")
    write_csv(audit, out_dir / "08_causal_audit.csv")
    write_csv(side_summary, out_dir / "10_side_combined_summary.csv")

    config = vars(args).copy()
    config.update(meta)
    (out_dir / "09_meta.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "20_research_brief.md").write_text(make_brief(args, funnel, summary, side_summary, audit), encoding="utf-8")
    print(f"[done] wrote reports to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
