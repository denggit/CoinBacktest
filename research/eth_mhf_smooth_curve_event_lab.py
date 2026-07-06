#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH MHF Smooth-Curve Event Lab
==============================

Research-only lab for finding a follower-friendly MHF layer on ETH perpetual.

Purpose
-------
This script studies two different things in one report folder:

1. MHF event families on a 1m trade-bar execution axis:
   - impulse exhaustion reversal
   - micro sweep reclaim / reject
   - VWAP extension fade with order-flow exhaustion
   - compression breakout pullback
   - order-flow imbalance continuation
   - session liquidity raid

2. A standalone EMA20/EMA50 armed crossover strategy:
   - long arm: price crosses from below to above EMA50 on a closed 1m bar
   - long entry: after the arm, any later closed bar with close >= EMA50*(1+entry_buffer_pct)
   - long exit: three consecutive closed bars below EMA20 OR one closed bar below EMA50
   - short side is symmetric
   - all entries/exits execute on the next bar open

Important boundaries
--------------------
1. Closed-bar signals only; no same-bar entry.
2. All event-study labels and EMA trades execute at next-bar open.
3. Rolling levels use prior windows via shift(1) wherever they are structural context.
4. Range-bar and footprint context are merged by their closed end_ts with merge_asof(direction='backward').
5. Future returns, MFE/MAE, and TP/SL first-touch labels are evaluation labels only.
6. This script is research only. Positive candidates still need full replay audit,
   stress, neighbourhood, and live execution review before promotion.

Example
-------
python research/eth_mhf_smooth_curve_event_lab.py --symbol ETH-USDT-SWAP --timeframe 1m --start-date 2023-01-01 --end-date 2026-06-30 --warmup-start-date 2022-01-01 --out-dir data/reports/research/eth_mhf_smooth_curve_event_lab
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader  # noqa: E402
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader  # noqa: E402
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.event_study import (  # noqa: E402
    CostConfig,
    EventStudyConfig,
    EventStudyResult,
    audit_context_available_times,
    audit_next_open_entries,
    first_touch_outcome,
    profit_factor,
    run_event_study,
    summarize_many,
    summarize_returns,
    top_winner_dependency,
)
from src.research_common.progress import ProgressReporter, progress_iter  # noqa: E402


SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1H", "4H", "1D"}
SESSION_WINDOWS = {
    # Local timestamp style follows config.loader.TIMEZONE, normally +8 in this project.
    "asia_am": ((8, 0), (12, 0)),
    "eu_open": ((15, 0), (18, 0)),
    "us_premarket": ((20, 0), (22, 30)),
    "us_open": ((21, 30), (23, 30)),
    "late_us": ((0, 0), (3, 0)),
}


# ---------------------------------------------------------------------------
# Args / parsing
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ETH MHF smooth-curve event lab using 1m OKX trade bars plus optional range/footprint context.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", choices=sorted(SUPPORTED_TIMEFRAMES), default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--out-dir", default="data/reports/research/eth_mhf_smooth_curve_event_lab")

    # Common feature windows.
    p.add_argument("--atr-window", type=int, default=42)
    p.add_argument("--volume-window", type=int, default=120)
    p.add_argument("--delta-window", type=int, default=120)
    p.add_argument("--vwap-windows", default="60,240")
    p.add_argument("--sweep-windows", default="20,60,120")
    p.add_argument("--compression-windows", default="30,60,120")

    # Event thresholds. Keep a compact but useful default grid; expand locally if needed.
    p.add_argument("--volume-ratio-thresholds", default="1.2,1.5,2.0")
    p.add_argument("--delta-z-thresholds", default="1.0,1.5,2.0")
    p.add_argument("--large-delta-z-thresholds", default="0.8,1.2")
    p.add_argument("--impulse-pcts", default="0.003,0.005,0.008")
    p.add_argument("--vwap-extension-pcts", default="0.003,0.005,0.008")
    p.add_argument("--sweep-break-pcts", default="0.0,0.0005,0.0010")
    p.add_argument("--reclaim-buffer-pcts", default="0.0,0.0003")
    p.add_argument("--compression-range-pcts", default="0.003,0.005,0.008")
    p.add_argument("--breakout-buffer-pcts", default="0.0005,0.0010")
    p.add_argument("--pullback-max-bars", type=int, default=12)
    p.add_argument("--pullback-hold-buffer-pct", type=float, default=0.0002)
    p.add_argument("--continuation-buy-ratio", type=float, default=0.62)
    p.add_argument("--continuation-sell-ratio", type=float, default=0.38)
    p.add_argument("--close-pos-long-min", type=float, default=0.60)
    p.add_argument("--close-pos-short-max", type=float, default=0.40)
    p.add_argument("--max-events-per-family", type=int, default=250_000)

    # EMA20/50 strategy requested by user.
    p.add_argument("--ema-fast", type=int, default=20)
    p.add_argument("--ema-slow", type=int, default=50)
    p.add_argument("--ema-entry-buffer-pct", type=float, default=0.0020)
    p.add_argument("--ema-exit-fast-consecutive", type=int, default=3)
    p.add_argument("--ema-max-hold-bars", type=int, default=24 * 60, help="0 disables max-hold exit for EMA strategy.")

    # Event-study labels / stress.
    p.add_argument("--horizons", default="3,6,12,24,48,96")
    p.add_argument("--candidate-horizon", type=int, default=24)
    p.add_argument("--mfe-mae-horizon", type=int, default=96)
    p.add_argument("--min-count", type=int, default=100)
    p.add_argument("--min-active-days-ratio", type=float, default=0.40)
    p.add_argument("--max-days-without-trade", type=int, default=5)
    p.add_argument("--min-profit-factor", type=float, default=1.15)
    p.add_argument("--min-win-rate", type=float, default=0.52)
    p.add_argument("--max-top5-winner-share", type=float, default=0.45)

    # Costs: default project convention is 0.11% round trip. Slippage is added on top by default.
    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage-pct", type=float, default=0.00020)
    p.add_argument("--exit-slippage-pct", type=float, default=0.00020)
    p.add_argument("--stress-cost-multipliers", default="1.0,1.5,2.0")
    p.add_argument("--stress-delay-bars", default="1,2,3")

    # TP/SL first-touch grid for short-horizon MHF behaviour.
    p.add_argument("--touch-target-pcts", default="0.002,0.003,0.004,0.006,0.008")
    p.add_argument("--touch-stop-pcts", default="0.0015,0.002,0.003,0.004,0.006")
    p.add_argument("--touch-horizon", type=int, default=48)

    # Optional micro-structure context.
    p.add_argument("--range-pcts", default="0.0015,0.0020,0.0025")
    p.add_argument("--footprint-price-step", type=float, default=1.0)
    p.add_argument("--no-range-context", action="store_true")
    p.add_argument("--no-footprint", action="store_true")
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--build-missing", action="store_true", help="Allow loaders to build missing trade/range/footprint cache from raw OKX trades.")
    p.add_argument("--force-rebuild", action="store_true")

    # Output / runtime.
    p.add_argument("--progress-every", type=int, default=25_000)
    p.add_argument("--fast-mode", action="store_true", default=True, help="Use vectorized event labels/stress/touch grid for large MHF research.")
    p.add_argument("--slow-event-study", dest="fast_mode", action="store_false", help="Use the reusable generic runner. Safer for debugging, slower on large event sets.")
    p.add_argument("--touch-scope", choices=["event", "family"], default="family", help="TP/SL grid aggregation scope. family is much faster and enough for first-pass discovery.")
    p.add_argument("--stress-scope", choices=["family", "event"], default="family", help="Stress aggregation scope. family is faster; event is heavier but more detailed.")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--save-feature-sample", type=int, default=5_000)
    p.add_argument("--save-event-sample", type=int, default=100_000)
    p.add_argument("--save-trade-sample", type=int, default=100_000)

    # Winner/loser filter mining. This is intentionally train-first to reduce
    # full-sample data snooping: filters are discovered on train and then
    # evaluated on valid/test/year splits.
    p.add_argument("--no-filter-mining", action="store_true")
    p.add_argument("--filter-train-end", default="2024-12-31")
    p.add_argument("--filter-valid-end", default="2025-12-31")
    p.add_argument("--filter-quantiles", default="0.10,0.20,0.30,0.40,0.60,0.70,0.80,0.90")
    p.add_argument("--filter-min-count", type=int, default=500)
    p.add_argument("--filter-min-split-count", type=int, default=80)
    p.add_argument("--filter-top-per-group", type=int, default=24)
    p.add_argument("--filter-max-candidates", type=int, default=2000)
    p.add_argument("--filter-max-combos", type=int, default=5000)
    p.add_argument("--filter-stability-top-n", type=int, default=300)
    p.add_argument("--filter-tp-sl-top-n", type=int, default=100)
    p.add_argument("--no-filter-dedupe", action="store_true", help="Do not deduplicate event_family/side/signal_time before filter mining.")
    p.add_argument("--filter-max-rows-per-group", type=int, default=80_000, help="Deterministically cap rows per event_family/side for filter mining only.")

    # EMA armed-entry path study. This does not use the old EMA exit as label;
    # it asks whether the armed entry has a tradable TP/SL/max-hold path.
    p.add_argument("--ema-path-horizons", default="20,30,48,60")
    p.add_argument("--ema-path-target-pcts", default="0.003,0.004,0.005,0.006")
    p.add_argument("--ema-path-stop-pcts", default="0.0015,0.002,0.0025,0.003,0.004")

    # Passive-entry lab: treat MHF/EMA events as setups, then wait for a
    # better limit entry instead of taker next-open chasing. This is a
    # research-only execution model; it intentionally does not modify any
    # existing live/backtest strategy.
    p.add_argument("--no-passive-entry-lab", action="store_true")
    p.add_argument("--passive-top-filters", type=int, default=40)
    p.add_argument("--passive-max-setups-per-spec", type=int, default=20_000)
    p.add_argument("--passive-entry-offset-pcts", default="0.0005,0.0010,0.0015,0.0020")
    p.add_argument("--passive-fill-windows", default="1,3,5,8")
    p.add_argument("--passive-target-pcts", default="0.003,0.004,0.005")
    p.add_argument("--passive-stop-pcts", default="0.0025,0.0035,0.0050,0.0060")
    p.add_argument("--passive-horizons", default="20,30,48,60")
    p.add_argument("--passive-entry-maker-fee-rate", type=float, default=0.00030)
    p.add_argument("--passive-tp-maker-fee-rate", type=float, default=0.00030)
    p.add_argument("--passive-stop-taker-fee-rate", type=float, default=0.00055)
    p.add_argument("--passive-timeout-taker-fee-rate", type=float, default=0.00055)
    p.add_argument("--passive-stop-slippage-pct", type=float, default=0.00020)
    p.add_argument("--passive-timeout-slippage-pct", type=float, default=0.00020)
    p.add_argument("--passive-include-curated-setups", action="store_true", default=True)
    p.add_argument("--no-passive-curated-setups", dest="passive_include_curated_setups", action="store_false")
    p.add_argument("--passive-trade-sample", type=int, default=100_000)
    return p.parse_args(argv)


def _parse_number_list(text: str, *, cast=float, name: str = "values", allow_zero: bool = False) -> list[Any]:
    out: list[Any] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = cast(part)
        if allow_zero:
            if value < 0:
                raise ValueError(f"{name} must not contain negative values")
        elif value <= 0:
            raise ValueError(f"{name} must contain positive values")
        out.append(value)
    if not out:
        raise ValueError(f"{name} must not be empty")
    return sorted(set(out))


def _parse_int_list(text: str, *, name: str, allow_zero: bool = False) -> list[int]:
    return [int(x) for x in _parse_number_list(text, cast=int, name=name, allow_zero=allow_zero)]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _safe_divide(a: Any, b: Any) -> pd.Series:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce").replace(0.0, np.nan)
    return (aa / bb).replace([np.inf, -np.inf], np.nan)


def _true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return pd.to_numeric(tr, errors="coerce")


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    min_periods = min(int(window), max(10, int(window) // 3))
    roll = x.shift(1).rolling(int(window), min_periods=min_periods)
    return ((x - roll.mean()) / roll.std(ddof=0).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _side_name(side: int) -> str:
    return "LONG" if int(side) == 1 else "SHORT" if int(side) == -1 else "FLAT"


def _to_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _active_days_metrics(signal_times: pd.Series | pd.DatetimeIndex, *, start_date: str, end_date: str) -> dict[str, object]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    all_days = pd.date_range(start, end, freq="D")
    if len(all_days) == 0:
        return {"active_days": 0, "total_days": 0, "active_days_ratio": np.nan, "max_days_without_trade": np.nan}
    ts = pd.to_datetime(signal_times, errors="coerce").dropna()
    active = pd.DatetimeIndex(ts).normalize().unique()
    active_set = set(active)
    max_gap = 0
    cur_gap = 0
    for day in all_days:
        if day in active_set:
            cur_gap = 0
        else:
            cur_gap += 1
            max_gap = max(max_gap, cur_gap)
    return {
        "active_days": int(len(active)),
        "total_days": int(len(all_days)),
        "active_days_ratio": float(len(active) / len(all_days)),
        "max_days_without_trade": int(max_gap),
    }


def _summarize_equity(returns: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(returns, errors="coerce").dropna()
    if x.empty:
        return {
            "total_return": np.nan,
            "max_drawdown": np.nan,
            "profit_factor": np.nan,
            "win_rate": np.nan,
            "avg_return": np.nan,
            "median_return": np.nan,
            "max_win": np.nan,
            "max_loss": np.nan,
        }
    equity = (1.0 + x).cumprod()
    dd = equity / equity.cummax() - 1.0
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(dd.min()),
        "profit_factor": float(profit_factor(x)) if math.isfinite(float(profit_factor(x))) else np.inf,
        "win_rate": float((x > 0).mean()),
        "avg_return": float(x.mean()),
        "median_return": float(x.median()),
        "max_win": float(x.max()),
        "max_loss": float(x.min()),
    }


def _cost_from_args(args: argparse.Namespace, *, multiplier: float = 1.0) -> CostConfig:
    return CostConfig(
        entry_fee_rate=float(args.entry_fee_rate) * multiplier,
        exit_fee_rate=float(args.exit_fee_rate) * multiplier,
        entry_slippage_pct=float(args.entry_slippage_pct) * multiplier,
        exit_slippage_pct=float(args.exit_slippage_pct) * multiplier,
    )


# ---------------------------------------------------------------------------
# Loading / features
# ---------------------------------------------------------------------------


def load_trade_bars(args: argparse.Namespace) -> pd.DataFrame:
    print(f"[load] OKXTradeBarLoader {args.symbol} {args.timeframe} {args.warmup_start_date}->{args.end_date}", flush=True)
    loader = OKXTradeBarLoader(symbol=args.symbol, timeframe=args.timeframe)
    df = loader.fetch_data_by_date_range(
        args.warmup_start_date,
        args.end_date,
        chunksize=int(args.chunksize),
        force_rebuild=bool(args.force_rebuild),
        build_missing=bool(args.build_missing or args.force_rebuild),
    )
    if df.empty:
        raise RuntimeError(
            f"No trade bars loaded for {args.symbol} {args.timeframe}. "
            "If local cache is missing, rerun with --build-missing after raw OKX trades are available."
        )
    out = df.copy().sort_index()
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"trade bars missing required columns: {missing}")
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=required)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    print(f"       rows={len(out):,} range={out.index[0]} -> {out.index[-1]}", flush=True)
    return out


def build_tradebar_features(bars: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    print("[features] trade-bar features", flush=True)
    df = bars.copy().sort_index()
    for col in df.columns:
        if col not in {"timestamp"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["bar_ret"] = df["close"] / df["open"] - 1.0
    df["ret_1"] = df["close"].pct_change()
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)
    df["tr"] = _true_range(df)
    df["atr"] = df["tr"].rolling(int(args.atr_window), min_periods=int(args.atr_window)).mean()
    df["atr_pct"] = _safe_divide(df["atr"], df["close"])

    bar_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    df["close_pos"] = _safe_divide(df["close"] - df["low"], bar_range)
    df["upper_wick_frac"] = _safe_divide(df["high"] - df[["open", "close"]].max(axis=1), bar_range)
    df["lower_wick_frac"] = _safe_divide(df[["open", "close"]].min(axis=1) - df["low"], bar_range)

    vol_base = df["volume"].shift(1).rolling(int(args.volume_window), min_periods=max(10, int(args.volume_window) // 3)).median()
    df["volume_ratio"] = _safe_divide(df["volume"], vol_base)
    if "notional" in df.columns:
        notional_base = pd.to_numeric(df["notional"], errors="coerce").shift(1).rolling(
            int(args.volume_window), min_periods=max(10, int(args.volume_window) // 3)
        ).median()
        df["notional_ratio"] = _safe_divide(df["notional"], notional_base)
    else:
        df["notional_ratio"] = df["volume_ratio"]

    if "trades_count" in df.columns:
        trade_base = pd.to_numeric(df["trades_count"], errors="coerce").shift(1).rolling(
            int(args.volume_window), min_periods=max(10, int(args.volume_window) // 3)
        ).median()
        df["trades_count_ratio"] = _safe_divide(df["trades_count"], trade_base)
    else:
        df["trades_count_ratio"] = np.nan

    if "buy_volume" in df.columns and "sell_volume" in df.columns:
        df["buy_volume"] = pd.to_numeric(df["buy_volume"], errors="coerce")
        df["sell_volume"] = pd.to_numeric(df["sell_volume"], errors="coerce")
        df["taker_buy_ratio_calc"] = _safe_divide(df["buy_volume"], df["buy_volume"] + df["sell_volume"])
    if "taker_buy_ratio" not in df.columns:
        df["taker_buy_ratio"] = df.get("taker_buy_ratio_calc", np.nan)

    if "delta_notional" not in df.columns:
        if "buy_notional" in df.columns and "sell_notional" in df.columns:
            df["delta_notional"] = pd.to_numeric(df["buy_notional"], errors="coerce") - pd.to_numeric(df["sell_notional"], errors="coerce")
        else:
            df["delta_notional"] = np.nan
    df["delta_notional_z"] = _rolling_z(df["delta_notional"], int(args.delta_window))
    if "buy_notional" in df.columns and "sell_notional" in df.columns:
        df["delta_notional_ratio"] = _safe_divide(df["delta_notional"], pd.to_numeric(df["buy_notional"], errors="coerce") + pd.to_numeric(df["sell_notional"], errors="coerce"))
    else:
        df["delta_notional_ratio"] = np.nan

    if "large_delta_notional" not in df.columns:
        if "large_buy_notional" in df.columns and "large_sell_notional" in df.columns:
            df["large_delta_notional"] = pd.to_numeric(df["large_buy_notional"], errors="coerce") - pd.to_numeric(df["large_sell_notional"], errors="coerce")
        else:
            df["large_delta_notional"] = np.nan
    df["large_delta_notional_z"] = _rolling_z(df["large_delta_notional"], int(args.delta_window))
    if "large_buy_notional" in df.columns and "large_sell_notional" in df.columns:
        df["large_delta_notional_ratio"] = _safe_divide(
            df["large_delta_notional"],
            pd.to_numeric(df["large_buy_notional"], errors="coerce") + pd.to_numeric(df["large_sell_notional"], errors="coerce"),
        )
    else:
        df["large_delta_notional_ratio"] = np.nan

    # VWAPs. rolling_vwap_* uses prior/current closed bars only and is therefore visible at signal close.
    if "vwap" not in df.columns or pd.to_numeric(df["vwap"], errors="coerce").isna().all():
        df["vwap"] = _safe_divide(df["close"] * df["volume"], df["volume"])
    for win in _parse_int_list(args.vwap_windows, name="vwap_windows"):
        vol_sum = df["volume"].rolling(win, min_periods=max(10, win // 3)).sum()
        pv_sum = (df["close"] * df["volume"]).rolling(win, min_periods=max(10, win // 3)).sum()
        df[f"rolling_vwap_{win}"] = _safe_divide(pv_sum, vol_sum)
        df[f"vwap_dist_{win}"] = df["close"] / df[f"rolling_vwap_{win}"] - 1.0

    # Daily/session VWAP on local timestamp date. This only uses closed bars up to the signal bar.
    day_key = df.index.normalize()
    pv = df["close"] * df["volume"]
    df["session_pv_cum"] = pv.groupby(day_key).cumsum()
    df["session_volume_cum"] = df["volume"].groupby(day_key).cumsum()
    df["session_vwap"] = _safe_divide(df["session_pv_cum"], df["session_volume_cum"])
    df["session_vwap_dist"] = df["close"] / df["session_vwap"] - 1.0

    df["ema_fast"] = df["close"].ewm(span=int(args.ema_fast), adjust=False, min_periods=int(args.ema_fast)).mean()
    df["ema_slow"] = df["close"].ewm(span=int(args.ema_slow), adjust=False, min_periods=int(args.ema_slow)).mean()
    df["ema_gap_pct"] = df["ema_fast"] / df["ema_slow"] - 1.0
    df["close_to_ema_fast_pct"] = df["close"] / df["ema_fast"] - 1.0
    df["close_to_ema_slow_pct"] = df["close"] / df["ema_slow"] - 1.0
    for _ema_slope_win in (3, 5, 10, 20):
        df[f"ema_slow_slope_{_ema_slope_win}"] = df["ema_slow"] / df["ema_slow"].shift(_ema_slope_win) - 1.0
        df[f"ema_fast_slope_{_ema_slope_win}"] = df["ema_fast"] / df["ema_fast"].shift(_ema_slope_win) - 1.0
    df["ema_trend_side"] = np.select(
        [
            (df["ema_fast"] > df["ema_slow"]) & (df["close"] > df["ema_fast"]),
            (df["ema_fast"] < df["ema_slow"]) & (df["close"] < df["ema_fast"]),
        ],
        [1, -1],
        default=0,
    ).astype(int)

    for win in sorted(set(_parse_int_list(args.sweep_windows, name="sweep_windows") + _parse_int_list(args.compression_windows, name="compression_windows"))):
        df[f"prior_high_{win}"] = df["high"].shift(1).rolling(win, min_periods=win).max()
        df[f"prior_low_{win}"] = df["low"].shift(1).rolling(win, min_periods=win).min()
        df[f"prior_range_pct_{win}"] = df[f"prior_high_{win}"] / df[f"prior_low_{win}"] - 1.0
        df[f"range_pos_{win}"] = _safe_divide(df["close"] - df[f"prior_low_{win}"], df[f"prior_high_{win}"] - df[f"prior_low_{win}"])

    df["minute_of_day"] = df.index.hour * 60 + df.index.minute
    for name, ((sh, sm), (eh, em)) in SESSION_WINDOWS.items():
        start = sh * 60 + sm
        end = eh * 60 + em
        if start <= end:
            df[f"session_{name}"] = df["minute_of_day"].between(start, end, inclusive="left")
        else:
            df[f"session_{name}"] = (df["minute_of_day"] >= start) | (df["minute_of_day"] < end)
    df["session_any_active"] = df[[f"session_{name}" for name in SESSION_WINDOWS]].any(axis=1)

    return df


def _range_context_one(args: argparse.Namespace, range_pct: float) -> pd.DataFrame:
    loader = OKXRangeBarLoader(symbol=args.symbol, range_pct=float(range_pct))
    if bool(args.build_missing or args.force_rebuild):
        rb = loader.fetch_data_by_date_range(
            args.warmup_start_date,
            args.end_date,
            chunksize=int(args.chunksize),
            force_rebuild=bool(args.force_rebuild),
        )
    else:
        rb = loader.load_local_data(start_date=args.warmup_start_date, end_date=args.end_date)
    if rb.empty:
        return pd.DataFrame()

    # OKXRangeBarLoader finalizes with end_ts as index. Multiple range bars can
    # close on the same trade timestamp, so the index is not guaranteed unique.
    # Context joins below must be driven by explicit closed-bar available_time,
    # not by the loader return index.
    ctx = rb.reset_index(drop=True).copy()
    ctx["rb_available_time"] = pd.to_datetime(ctx["end_ts"], errors="coerce")
    ctx["rb_duration_seconds"] = (pd.to_datetime(ctx["end_ts"]) - pd.to_datetime(ctx["start_ts"])).dt.total_seconds()
    ctx["rb_return"] = ctx["close"] / ctx["open"] - 1.0
    ctx["rb_direction"] = np.sign(ctx["rb_return"]).astype(float)
    ctx["rb_delta_ratio"] = _safe_divide(ctx["delta_notional"], ctx["notional"])
    ctx["rb_large_delta_ratio"] = _safe_divide(ctx.get("large_delta_notional", np.nan), ctx.get("large_buy_notional", 0.0) + ctx.get("large_sell_notional", 0.0))
    ctx["rb_volume_z20"] = _rolling_z(ctx["volume"], 20)
    keep = [
        "bar_id",
        "start_ts",
        "end_ts",
        "rb_available_time",
        "rb_duration_seconds",
        "rb_return",
        "rb_direction",
        "rb_delta_ratio",
        "rb_large_delta_ratio",
        "rb_volume_z20",
        "trades_count",
        "max_trade_notional",
    ]
    out = ctx[[c for c in keep if c in ctx.columns]].copy()
    out = out.rename(
        columns={
            "bar_id": "rb_bar_id",
            "start_ts": "rb_start_ts",
            "end_ts": "rb_end_ts",
            "trades_count": "rb_trades_count",
            "max_trade_notional": "rb_max_trade_notional",
        }
    )
    out = out.dropna(subset=["rb_available_time"])
    if "rb_bar_id" in out.columns:
        out = out.drop_duplicates(subset=["rb_bar_id"], keep="last")
    code = f"r{int(round(float(range_pct) * 10_000)):04d}"
    return out.add_prefix(f"{code}_").reset_index(drop=True)


def _footprint_context_one(args: argparse.Namespace, range_pct: float) -> pd.DataFrame:
    loader = OKXRangeFootprintLoader(symbol=args.symbol, range_pct=float(range_pct), price_step=float(args.footprint_price_step))
    if bool(args.build_missing or args.force_rebuild):
        fp = loader.fetch_data_by_date_range(
            args.warmup_start_date,
            args.end_date,
            chunksize=int(args.chunksize),
            force_rebuild=bool(args.force_rebuild),
        )
    else:
        fp = loader.load_local_data(start_date=args.warmup_start_date, end_date=args.end_date)
    if fp.empty:
        return pd.DataFrame()

    # Footprint rows are price buckets inside a range bar. Collapse to one
    # closed range-bar context row per bar_id before any asof join. Keep bar_id
    # as a real column; groupby leaves it as an index by default.
    fp = fp.reset_index(drop=True).copy()
    grouped = fp.groupby("bar_id", sort=True)
    agg = grouped.agg(
        fp_start_ts=("start_ts", "first"),
        fp_end_ts=("end_ts", "first"),
        fp_bucket_count=("price_bucket", "count"),
        fp_total_notional=("notional", "sum"),
        fp_total_volume=("volume", "sum"),
        fp_total_delta_notional=("delta_notional", "sum"),
        fp_abs_delta_notional=("delta_notional", lambda x: float(np.nansum(np.abs(x)))),
        fp_max_bucket_delta_notional=("delta_notional", "max"),
        fp_min_bucket_delta_notional=("delta_notional", "min"),
        fp_large_delta_notional=("large_delta_notional", "sum"),
        fp_large_trades_count=("large_trades_count", "sum"),
        fp_max_trade_notional=("max_trade_notional", "max"),
    ).reset_index()
    agg["fp_available_time"] = pd.to_datetime(agg["fp_end_ts"], errors="coerce")
    agg["fp_delta_ratio"] = _safe_divide(agg["fp_total_delta_notional"], agg["fp_total_notional"])
    agg["fp_absorption_proxy"] = _safe_divide(agg["fp_abs_delta_notional"], agg["fp_total_notional"])
    agg["fp_large_delta_ratio"] = _safe_divide(agg["fp_large_delta_notional"], agg["fp_total_notional"])
    agg = agg.rename(columns={"bar_id": "fp_bar_id"})
    agg = agg.dropna(subset=["fp_available_time"]).drop_duplicates(subset=["fp_bar_id"], keep="last")
    code = f"r{int(round(float(range_pct) * 10_000)):04d}"
    return agg.add_prefix(f"{code}_").reset_index(drop=True)


def _attach_one_context_frame(
    base: pd.DataFrame,
    ctx: pd.DataFrame,
    *,
    available_col: str,
    label: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Causally attach one closed context frame to the 1m trade-bar axis.

    The right side is explicitly de-duplicated by available_time. Range bars can
    close multiple bars at the same timestamp; for a 1m signal all of those are
    already visible, and the latest row is the deterministic context snapshot.
    """
    if ctx.empty or available_col not in ctx.columns:
        return base, []
    right = ctx.copy()
    right[available_col] = pd.to_datetime(right[available_col], errors="coerce")
    right = right.dropna(subset=[available_col]).sort_values(available_col)
    if right.empty:
        return base, []

    before = len(right)
    right = right.drop_duplicates(subset=[available_col], keep="last")
    dropped = before - len(right)
    right = right.set_index(available_col, drop=False).sort_index()

    left = base.sort_index()
    merged = pd.merge_asof(left, right, left_index=True, right_index=True, direction="backward")
    new_cols = [c for c in merged.columns if c not in base.columns]
    if dropped:
        print(f"[context] {label} available_time duplicates collapsed={dropped:,}", flush=True)
    print(f"[context] {label} rows={before:,} asof_rows={len(right):,} cols_added={len(new_cols):,}", flush=True)
    return merged, [c for c in new_cols if c.endswith("_available_time")]


def attach_range_and_footprint_context(features: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    if bool(args.no_range_context) and bool(args.no_footprint):
        return features, []

    base = features.sort_index().copy()
    context_cols: list[str] = []
    for range_pct in _parse_number_list(args.range_pcts, cast=float, name="range_pcts"):
        code = f"r{int(round(float(range_pct) * 10_000)):04d}"
        if not bool(args.no_range_context):
            print(f"[context] range bars {code}", flush=True)
            rb = _range_context_one(args, range_pct)
            if not rb.empty:
                base, added = _attach_one_context_frame(
                    base,
                    rb,
                    available_col=f"{code}_rb_available_time",
                    label=f"range bars {code}",
                )
                context_cols.extend(added)
            else:
                print(f"[context] range bars {code} empty", flush=True)
        if not bool(args.no_footprint):
            print(f"[context] range footprint {code} step={args.footprint_price_step}", flush=True)
            fp = _footprint_context_one(args, range_pct)
            if not fp.empty:
                base, added = _attach_one_context_frame(
                    base,
                    fp,
                    available_col=f"{code}_fp_available_time",
                    label=f"range footprint {code}",
                )
                context_cols.extend(added)
            else:
                print(f"[context] range footprint {code} empty", flush=True)
    return base, sorted(set(context_cols))


# ---------------------------------------------------------------------------
# Event family builders
# ---------------------------------------------------------------------------


def _make_events(df: pd.DataFrame, mask: pd.Series, *, event_name: str, side: int, extra_cols: Sequence[str]) -> pd.DataFrame:
    m = mask.fillna(False).astype(bool)
    if not m.any():
        return pd.DataFrame()
    cols = [c for c in extra_cols if c in df.columns]
    out = df.loc[m, cols].copy()
    out["signal_time"] = out.index
    out["side"] = int(side)
    out["side_name"] = _side_name(side)
    out["event_name"] = event_name
    out["event_family"] = event_name.split("|")[0]
    return out.reset_index(drop=True)


def _cap_family_events(events: pd.DataFrame, *, max_events: int, family: str) -> pd.DataFrame:
    if max_events <= 0 or len(events) <= max_events:
        return events
    # Deterministic thinning keeps chronology coverage without random instability.
    step = max(1, int(math.ceil(len(events) / max_events)))
    out = events.iloc[::step].copy()
    print(f"[events] capped {family}: {len(events):,} -> {len(out):,} rows using step={step}", flush=True)
    return out


def build_impulse_exhaustion_events(df: pd.DataFrame, args: argparse.Namespace, extra_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    vols = _parse_number_list(args.volume_ratio_thresholds, name="volume_ratio_thresholds")
    dzs = _parse_number_list(args.delta_z_thresholds, name="delta_z_thresholds")
    ldzs = _parse_number_list(args.large_delta_z_thresholds, name="large_delta_z_thresholds")
    impulse_pcts = _parse_number_list(args.impulse_pcts, name="impulse_pcts")
    for vol_thr in vols:
        for dz in dzs:
            for ext in impulse_pcts:
                buy_impulse = (df["volume_ratio"] >= vol_thr) & (df["delta_notional_z"] >= dz) & (df["ret_5"] >= ext)
                sell_impulse = (df["volume_ratio"] >= vol_thr) & (df["delta_notional_z"] <= -dz) & (df["ret_5"] <= -ext)
                short_fail = buy_impulse & ((df["close_pos"] <= float(args.close_pos_short_max)) | (df["bar_ret"] < 0))
                long_fail = sell_impulse & ((df["close_pos"] >= float(args.close_pos_long_min)) | (df["bar_ret"] > 0))
                rows.append(_make_events(df, short_fail, event_name=f"impulse_exhaustion|short|vol{vol_thr:g}|dz{dz:g}|ret5{ext:g}", side=-1, extra_cols=extra_cols))
                rows.append(_make_events(df, long_fail, event_name=f"impulse_exhaustion|long|vol{vol_thr:g}|dz{dz:g}|ret5{ext:g}", side=1, extra_cols=extra_cols))
    for vol_thr in vols:
        for ldz in ldzs:
            large_buy_fail = (df["volume_ratio"] >= vol_thr) & (df["large_delta_notional_z"] >= ldz) & (df["close_pos"] <= float(args.close_pos_short_max))
            large_sell_fail = (df["volume_ratio"] >= vol_thr) & (df["large_delta_notional_z"] <= -ldz) & (df["close_pos"] >= float(args.close_pos_long_min))
            rows.append(_make_events(df, large_buy_fail, event_name=f"impulse_exhaustion_large|short|vol{vol_thr:g}|ldz{ldz:g}", side=-1, extra_cols=extra_cols))
            rows.append(_make_events(df, large_sell_fail, event_name=f"impulse_exhaustion_large|long|vol{vol_thr:g}|ldz{ldz:g}", side=1, extra_cols=extra_cols))
    out = pd.concat([x for x in rows if not x.empty], ignore_index=True) if rows else pd.DataFrame()
    return _cap_family_events(out, max_events=int(args.max_events_per_family), family="impulse_exhaustion")


def build_micro_sweep_events(df: pd.DataFrame, args: argparse.Namespace, extra_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    windows = _parse_int_list(args.sweep_windows, name="sweep_windows")
    breaks = _parse_number_list(args.sweep_break_pcts, name="sweep_break_pcts", allow_zero=True)
    reclaims = _parse_number_list(args.reclaim_buffer_pcts, name="reclaim_buffer_pcts", allow_zero=True)
    dzs = _parse_number_list(args.delta_z_thresholds, name="delta_z_thresholds")
    for win in windows:
        prior_low = df[f"prior_low_{win}"]
        prior_high = df[f"prior_high_{win}"]
        for br in breaks:
            for rec in reclaims:
                base_long = (df["low"] <= prior_low * (1.0 - br)) & (df["close"] >= prior_low * (1.0 + rec)) & (df["close_pos"] >= float(args.close_pos_long_min))
                base_short = (df["high"] >= prior_high * (1.0 + br)) & (df["close"] <= prior_high * (1.0 - rec)) & (df["close_pos"] <= float(args.close_pos_short_max))
                rows.append(_make_events(df, base_long, event_name=f"micro_sweep_reclaim|long|w{win}|br{br:g}|rec{rec:g}|noflow", side=1, extra_cols=extra_cols))
                rows.append(_make_events(df, base_short, event_name=f"micro_sweep_reclaim|short|w{win}|br{br:g}|rec{rec:g}|noflow", side=-1, extra_cols=extra_cols))
                for dz in dzs:
                    long_abs = base_long & (df["delta_notional_z"] <= -dz)
                    short_abs = base_short & (df["delta_notional_z"] >= dz)
                    rows.append(_make_events(df, long_abs, event_name=f"micro_sweep_reclaim|long|w{win}|br{br:g}|rec{rec:g}|dz{dz:g}", side=1, extra_cols=extra_cols))
                    rows.append(_make_events(df, short_abs, event_name=f"micro_sweep_reclaim|short|w{win}|br{br:g}|rec{rec:g}|dz{dz:g}", side=-1, extra_cols=extra_cols))
                # Session-gated raid variant; same core event but only inside active liquidity windows.
                long_session = base_long & df["session_any_active"].astype(bool)
                short_session = base_short & df["session_any_active"].astype(bool)
                rows.append(_make_events(df, long_session, event_name=f"session_liquidity_raid|long|w{win}|br{br:g}|rec{rec:g}", side=1, extra_cols=extra_cols))
                rows.append(_make_events(df, short_session, event_name=f"session_liquidity_raid|short|w{win}|br{br:g}|rec{rec:g}", side=-1, extra_cols=extra_cols))
    out = pd.concat([x for x in rows if not x.empty], ignore_index=True) if rows else pd.DataFrame()
    return _cap_family_events(out, max_events=int(args.max_events_per_family), family="micro_sweep")


def build_vwap_extension_fade_events(df: pd.DataFrame, args: argparse.Namespace, extra_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    exts = _parse_number_list(args.vwap_extension_pcts, name="vwap_extension_pcts")
    dzs = _parse_number_list(args.delta_z_thresholds, name="delta_z_thresholds")
    vwap_windows = _parse_int_list(args.vwap_windows, name="vwap_windows")
    for win in vwap_windows:
        dist = df[f"vwap_dist_{win}"]
        for ext in exts:
            over = dist >= ext
            under = dist <= -ext
            for dz in dzs:
                short_mask = over & (df["delta_notional_z"] >= dz) & ((df["close_pos"] <= float(args.close_pos_short_max)) | (df["bar_ret"] < 0))
                long_mask = under & (df["delta_notional_z"] <= -dz) & ((df["close_pos"] >= float(args.close_pos_long_min)) | (df["bar_ret"] > 0))
                rows.append(_make_events(df, short_mask, event_name=f"vwap_extension_fade|short|w{win}|ext{ext:g}|dz{dz:g}", side=-1, extra_cols=extra_cols))
                rows.append(_make_events(df, long_mask, event_name=f"vwap_extension_fade|long|w{win}|ext{ext:g}|dz{dz:g}", side=1, extra_cols=extra_cols))
            # No-flow version: extension + weak close only, to check whether orderflow really improves it.
            rows.append(_make_events(df, over & (df["close_pos"] <= float(args.close_pos_short_max)), event_name=f"vwap_extension_fade|short|w{win}|ext{ext:g}|noflow", side=-1, extra_cols=extra_cols))
            rows.append(_make_events(df, under & (df["close_pos"] >= float(args.close_pos_long_min)), event_name=f"vwap_extension_fade|long|w{win}|ext{ext:g}|noflow", side=1, extra_cols=extra_cols))
    out = pd.concat([x for x in rows if not x.empty], ignore_index=True) if rows else pd.DataFrame()
    return _cap_family_events(out, max_events=int(args.max_events_per_family), family="vwap_extension_fade")


def _pullback_events_for_window(
    df: pd.DataFrame,
    *,
    window: int,
    compression_range_pct: float,
    breakout_buffer_pct: float,
    pullback_max_bars: int,
    hold_buffer_pct: float,
    extra_cols: Sequence[str],
) -> pd.DataFrame:
    prior_high = pd.to_numeric(df[f"prior_high_{window}"], errors="coerce").to_numpy(dtype=float)
    prior_low = pd.to_numeric(df[f"prior_low_{window}"], errors="coerce").to_numpy(dtype=float)
    prior_range = pd.to_numeric(df[f"prior_range_pct_{window}"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    n = len(df)

    long_mask = np.zeros(n, dtype=bool)
    short_mask = np.zeros(n, dtype=bool)
    long_level = np.full(n, np.nan, dtype=float)
    short_level = np.full(n, np.nan, dtype=float)
    long_age = np.full(n, np.nan, dtype=float)
    short_age = np.full(n, np.nan, dtype=float)

    armed_long_level = np.nan
    armed_short_level = np.nan
    armed_long_age = 0
    armed_short_age = 0
    for i in range(n):
        if np.isfinite(armed_long_level):
            armed_long_age += 1
            if armed_long_age > pullback_max_bars or close[i] < armed_long_level * (1.0 - hold_buffer_pct * 2.0):
                armed_long_level = np.nan
                armed_long_age = 0
        if np.isfinite(armed_short_level):
            armed_short_age += 1
            if armed_short_age > pullback_max_bars or close[i] > armed_short_level * (1.0 + hold_buffer_pct * 2.0):
                armed_short_level = np.nan
                armed_short_age = 0

        if np.isfinite(armed_long_level):
            touched = low[i] <= armed_long_level * (1.0 + hold_buffer_pct)
            held = close[i] >= armed_long_level * (1.0 + hold_buffer_pct)
            if touched and held:
                long_mask[i] = True
                long_level[i] = armed_long_level
                long_age[i] = armed_long_age
                armed_long_level = np.nan
                armed_long_age = 0
        if np.isfinite(armed_short_level):
            touched = high[i] >= armed_short_level * (1.0 - hold_buffer_pct)
            held = close[i] <= armed_short_level * (1.0 - hold_buffer_pct)
            if touched and held:
                short_mask[i] = True
                short_level[i] = armed_short_level
                short_age[i] = armed_short_age
                armed_short_level = np.nan
                armed_short_age = 0

        if np.isfinite(prior_range[i]) and prior_range[i] <= compression_range_pct:
            if np.isfinite(prior_high[i]) and close[i] >= prior_high[i] * (1.0 + breakout_buffer_pct):
                armed_long_level = prior_high[i]
                armed_long_age = 0
            if np.isfinite(prior_low[i]) and close[i] <= prior_low[i] * (1.0 - breakout_buffer_pct):
                armed_short_level = prior_low[i]
                armed_short_age = 0

    tmp = df.copy()
    tmp["pullback_breakout_level_long"] = long_level
    tmp["pullback_breakout_level_short"] = short_level
    tmp["pullback_age_long"] = long_age
    tmp["pullback_age_short"] = short_age
    cols = list(extra_cols) + ["pullback_breakout_level_long", "pullback_breakout_level_short", "pullback_age_long", "pullback_age_short"]
    a = _make_events(
        tmp,
        pd.Series(long_mask, index=df.index),
        event_name=f"compression_breakout_pullback|long|w{window}|rng{compression_range_pct:g}|bo{breakout_buffer_pct:g}",
        side=1,
        extra_cols=cols,
    )
    b = _make_events(
        tmp,
        pd.Series(short_mask, index=df.index),
        event_name=f"compression_breakout_pullback|short|w{window}|rng{compression_range_pct:g}|bo{breakout_buffer_pct:g}",
        side=-1,
        extra_cols=cols,
    )
    return pd.concat([a, b], ignore_index=True) if not a.empty or not b.empty else pd.DataFrame()


def build_compression_breakout_pullback_events(df: pd.DataFrame, args: argparse.Namespace, extra_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    windows = _parse_int_list(args.compression_windows, name="compression_windows")
    ranges = _parse_number_list(args.compression_range_pcts, name="compression_range_pcts")
    breakouts = _parse_number_list(args.breakout_buffer_pcts, name="breakout_buffer_pcts")
    total = len(windows) * len(ranges) * len(breakouts)
    progress = ProgressReporter("[events] compression_pullback", total=total, every=1, enabled=not bool(args.no_progress))
    done = 0
    for win in windows:
        for rng in ranges:
            for bo in breakouts:
                rows.append(
                    _pullback_events_for_window(
                        df,
                        window=win,
                        compression_range_pct=float(rng),
                        breakout_buffer_pct=float(bo),
                        pullback_max_bars=int(args.pullback_max_bars),
                        hold_buffer_pct=float(args.pullback_hold_buffer_pct),
                        extra_cols=extra_cols,
                    )
                )
                done += 1
                progress.update(done)
    progress.close()
    out = pd.concat([x for x in rows if not x.empty], ignore_index=True) if rows else pd.DataFrame()
    return _cap_family_events(out, max_events=int(args.max_events_per_family), family="compression_breakout_pullback")


def build_orderflow_continuation_events(df: pd.DataFrame, args: argparse.Namespace, extra_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    vols = _parse_number_list(args.volume_ratio_thresholds, name="volume_ratio_thresholds")
    dzs = _parse_number_list(args.delta_z_thresholds, name="delta_z_thresholds")
    for vol_thr in vols:
        for dz in dzs:
            long_mask = (
                (df["volume_ratio"] >= vol_thr)
                & (df["delta_notional_z"] >= dz)
                & (df["taker_buy_ratio"] >= float(args.continuation_buy_ratio))
                & (df["ema_trend_side"] >= 1)
                & (df["bar_ret"] > 0)
                & (df["close_pos"] >= 0.55)
            )
            short_mask = (
                (df["volume_ratio"] >= vol_thr)
                & (df["delta_notional_z"] <= -dz)
                & (df["taker_buy_ratio"] <= float(args.continuation_sell_ratio))
                & (df["ema_trend_side"] <= -1)
                & (df["bar_ret"] < 0)
                & (df["close_pos"] <= 0.45)
            )
            rows.append(_make_events(df, long_mask, event_name=f"orderflow_continuation|long|vol{vol_thr:g}|dz{dz:g}", side=1, extra_cols=extra_cols))
            rows.append(_make_events(df, short_mask, event_name=f"orderflow_continuation|short|vol{vol_thr:g}|dz{dz:g}", side=-1, extra_cols=extra_cols))
    out = pd.concat([x for x in rows if not x.empty], ignore_index=True) if rows else pd.DataFrame()
    return _cap_family_events(out, max_events=int(args.max_events_per_family), family="orderflow_continuation")


def build_all_events(features: pd.DataFrame, args: argparse.Namespace, context_available_cols: Sequence[str]) -> pd.DataFrame:
    print("[events] build MHF event families", flush=True)
    base_extra = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bar_ret",
        "ret_3",
        "ret_5",
        "ret_10",
        "atr_pct",
        "volume_ratio",
        "notional_ratio",
        "trades_count_ratio",
        "taker_buy_ratio",
        "delta_notional_z",
        "delta_notional_ratio",
        "large_delta_notional_z",
        "large_delta_notional_ratio",
        "close_pos",
        "upper_wick_frac",
        "lower_wick_frac",
        "session_vwap_dist",
        "ema_fast",
        "ema_slow",
        "ema_gap_pct",
        "close_to_ema_fast_pct",
        "close_to_ema_slow_pct",
        "ema_fast_slope_3",
        "ema_fast_slope_5",
        "ema_fast_slope_10",
        "ema_fast_slope_20",
        "ema_slow_slope_3",
        "ema_slow_slope_5",
        "ema_slow_slope_10",
        "ema_slow_slope_20",
        "ema_trend_side",
        "session_any_active",
    ]
    for win in _parse_int_list(args.vwap_windows, name="vwap_windows"):
        base_extra.extend([f"rolling_vwap_{win}", f"vwap_dist_{win}"])
    for win in sorted(set(_parse_int_list(args.sweep_windows, name="sweep_windows") + _parse_int_list(args.compression_windows, name="compression_windows"))):
        base_extra.extend([f"prior_high_{win}", f"prior_low_{win}", f"prior_range_pct_{win}", f"range_pos_{win}"])
    # Include compact context features but avoid copying every raw range/footprint column when many are present.
    context_extra = [
        c
        for c in features.columns
        if c.startswith("r")
        and (
            c.endswith("_available_time")
            or any(key in c for key in ["rb_return", "rb_direction", "rb_delta_ratio", "rb_large_delta_ratio", "rb_duration", "rb_volume_z", "fp_delta_ratio", "fp_absorption_proxy", "fp_large_delta_ratio", "fp_bucket_count", "fp_max_trade"])
        )
    ]
    extra_cols = sorted(set(base_extra + list(context_available_cols) + context_extra))

    builders = [
        ("impulse_exhaustion", build_impulse_exhaustion_events),
        ("micro_sweep", build_micro_sweep_events),
        ("vwap_extension_fade", build_vwap_extension_fade_events),
        ("compression_breakout_pullback", build_compression_breakout_pullback_events),
        ("orderflow_continuation", build_orderflow_continuation_events),
    ]
    parts: list[pd.DataFrame] = []
    for name, builder in builders:
        print(f"[events] {name}", flush=True)
        part = builder(features, args, extra_cols)
        if not part.empty:
            print(f"         rows={len(part):,}", flush=True)
            parts.append(part)
        else:
            print("         rows=0", flush=True)
    if not parts:
        return pd.DataFrame()
    events = pd.concat(parts, ignore_index=True)
    events["event_id"] = np.arange(len(events), dtype=np.int64)
    events = events.sort_values(["signal_time", "event_name", "side"]).reset_index(drop=True)
    print(f"[events] total rows={len(events):,} unique_events={events['event_name'].nunique():,}", flush=True)
    return events



# ---------------------------------------------------------------------------
# Fast vectorized event-study helpers
# ---------------------------------------------------------------------------


def _forward_rolling_extreme(values: np.ndarray, window: int, *, kind: str) -> np.ndarray:
    """Forward-looking rolling max/min over values[i:i+window].

    This is used only for evaluation labels after a closed signal bar. It does
    not feed back into signal generation.
    """
    w = max(1, int(window))
    ser = pd.Series(values)
    rev = ser.iloc[::-1]
    if kind == "max":
        out = rev.rolling(w, min_periods=1).max().iloc[::-1]
    elif kind == "min":
        out = rev.rolling(w, min_periods=1).min().iloc[::-1]
    else:
        raise ValueError("kind must be 'max' or 'min'")
    return out.to_numpy(dtype=float)


def _coerce_event_positions(bars: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
    signal_times = pd.to_datetime(events["signal_time"], errors="coerce")
    return bars.index.get_indexer(pd.DatetimeIndex(signal_times))


def _attach_fast_event_labels(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizons: Sequence[int],
    mfe_mae_horizon: int,
    entry_delay_bars: int,
    cost: CostConfig,
) -> pd.DataFrame:
    """Attach next-open returns and MFE/MAE without per-event Python loops."""
    frame = bars.sort_index()
    out = events.copy()
    pos = _coerce_event_positions(frame, out)
    side = pd.to_numeric(out["side"], errors="coerce").fillna(0).astype(float).to_numpy()
    n = len(frame)
    valid = pos >= 0
    out["signal_bar_pos"] = pos
    out["signal_on_bar_index_flag"] = valid

    open_arr = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    close_arr = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    high_arr = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    low_arr = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    delay = max(1, int(entry_delay_bars))

    for h0 in horizons:
        h = int(h0)
        ctc_col = f"close_to_close_ret_h{h}"
        gross_col = f"next_open_ret_h{h}_gross"
        net_col = f"next_open_ret_h{h}_net"
        ctc = np.full(len(out), np.nan, dtype=float)
        gross = np.full(len(out), np.nan, dtype=float)
        fut_pos = pos + h
        entry_pos = pos + delay
        ok = valid & (fut_pos >= 0) & (fut_pos < n) & (entry_pos >= 0) & (entry_pos < n)
        ok &= np.isfinite(close_arr[pos.clip(0, max(0, n - 1))]) & np.isfinite(close_arr[fut_pos.clip(0, max(0, n - 1))])
        ok &= np.isfinite(open_arr[entry_pos.clip(0, max(0, n - 1))]) & (open_arr[entry_pos.clip(0, max(0, n - 1))] > 0)
        if ok.any():
            p = pos[ok]
            fp = fut_pos[ok]
            ep = entry_pos[ok]
            ctc[ok] = (close_arr[fp] / close_arr[p] - 1.0) * side[ok]
            gross[ok] = (close_arr[fp] / open_arr[ep] - 1.0) * side[ok]
        out[ctc_col] = ctc
        out[gross_col] = gross
        out[net_col] = gross - float(cost.round_trip_cost_pct)

    h = int(mfe_mae_horizon)
    path_window = max(1, h - delay + 1)
    shifted_high = pd.Series(high_arr).shift(-delay).to_numpy(dtype=float)
    shifted_low = pd.Series(low_arr).shift(-delay).to_numpy(dtype=float)
    fwd_high = _forward_rolling_extreme(shifted_high, path_window, kind="max")
    fwd_low = _forward_rolling_extreme(shifted_low, path_window, kind="min")
    entry_open = pd.Series(open_arr).shift(-delay).to_numpy(dtype=float)
    mfe = np.full(len(out), np.nan, dtype=float)
    mae = np.full(len(out), np.nan, dtype=float)
    entry_pos = pos + delay
    ok = valid & (entry_pos >= 0) & (entry_pos < n) & (pos + h < n)
    ok &= np.isfinite(entry_open[pos.clip(0, max(0, n - 1))]) & (entry_open[pos.clip(0, max(0, n - 1))] > 0)
    if ok.any():
        p = pos[ok]
        entry = entry_open[p]
        long_ok = ok.copy()
        long_ok[ok] = side[ok] > 0
        short_ok = ok.copy()
        short_ok[ok] = side[ok] < 0
        if long_ok.any():
            pp = pos[long_ok]
            ee = entry_open[pp]
            mfe[long_ok] = fwd_high[pp] / ee - 1.0
            mae[long_ok] = fwd_low[pp] / ee - 1.0
        if short_ok.any():
            pp = pos[short_ok]
            ee = entry_open[pp]
            mfe[short_ok] = ee / fwd_low[pp] - 1.0
            mae[short_ok] = ee / fwd_high[pp] - 1.0
    out[f"mfe_h{h}"] = mfe
    out[f"mae_h{h}"] = mae

    out["entry_bar_pos"] = pos + delay
    in_entry_range = (out["entry_bar_pos"] >= 0) & (out["entry_bar_pos"] < n)
    out["entry_time"] = pd.NaT
    out["entry_price"] = pd.NA
    if in_entry_range.any():
        ep = out.loc[in_entry_range, "entry_bar_pos"].astype(int).to_numpy()
        out.loc[in_entry_range, "entry_time"] = frame.index[ep]
        out.loc[in_entry_range, "entry_price"] = open_arr[ep]
    out["side_name"] = out["side"].map({1: "LONG", -1: "SHORT"}).fillna("FLAT")
    out["year"] = pd.to_datetime(out["signal_time"]).dt.year
    out["round_trip_cost_pct"] = float(cost.round_trip_cost_pct)
    return out


def run_event_study_fast(bars: pd.DataFrame, events: pd.DataFrame, cfg: EventStudyConfig) -> EventStudyResult:
    """Fast equivalent for this lab's large first-pass event studies.

    It preserves the closed-bar/next-open convention while avoiding the generic
    runner's per-event MFE/MAE loop. Use --slow-event-study to fall back.
    """
    frame = bars.sort_index()
    ev = events.copy()
    ev["signal_time"] = pd.to_datetime(ev["signal_time"], errors="coerce")
    ev = ev.dropna(subset=["signal_time"]).copy()
    ev["side"] = pd.to_numeric(ev["side"], errors="coerce").fillna(0).astype(int)
    ev = ev[ev["side"] != 0].sort_values("signal_time").reset_index(drop=True)
    enriched = _attach_fast_event_labels(
        frame,
        ev,
        horizons=tuple(int(h) for h in cfg.horizons),
        mfe_mae_horizon=int(cfg.mfe_mae_horizon),
        entry_delay_bars=int(cfg.entry_delay_bars),
        cost=cfg.cost,
    )

    next_open_audit = audit_next_open_entries(enriched, signal_time_col="signal_time", entry_time_col="entry_time")
    context_audit = audit_context_available_times(
        enriched,
        signal_time_col="signal_time",
        context_available_time_cols=tuple(cfg.context_available_time_cols),
    )
    causal_audit = pd.concat(
        [
            next_open_audit.drop(columns=["signal_time"], errors="ignore"),
            context_audit.drop(columns=["signal_time"], errors="ignore"),
            enriched[["signal_on_bar_index_flag"]],
        ],
        axis=1,
    )
    causal_audit["causal_fail_flag"] = (
        (~causal_audit["signal_on_bar_index_flag"].astype(bool))
        | causal_audit.get("entry_not_after_signal_flag", False).astype(bool)
        | causal_audit.get("context_available_time_flag", False).astype(bool)
    )

    return_cols = [f"next_open_ret_h{int(h)}_net" for h in cfg.horizons]
    overview = summarize_many(enriched, return_cols, min_count=cfg.min_count)
    yearly = summarize_many(enriched, return_cols, group_cols=["year"], min_count=cfg.min_count)
    side_stats = summarize_many(enriched, return_cols, group_cols=["side_name"], min_count=cfg.min_count)
    meta: dict[str, object] = {
        "event_count": int(len(enriched)),
        "valid_signal_count": int(enriched["signal_on_bar_index_flag"].sum()) if not enriched.empty else 0,
        "entry_assumption": cfg.entry_assumption,
        "entry_delay_bars": int(cfg.entry_delay_bars),
        "horizons": tuple(int(h) for h in cfg.horizons),
        "mfe_mae_horizon": int(cfg.mfe_mae_horizon),
        "round_trip_cost_pct": float(cfg.cost.round_trip_cost_pct),
        "causal_fail_count": int(causal_audit["causal_fail_flag"].sum()) if not causal_audit.empty else 0,
        "fast_mode": True,
    }
    return EventStudyResult(
        events=enriched,
        overview=overview,
        yearly=yearly,
        side_stats=side_stats,
        horizon_stats=overview.copy(),
        causal_audit=causal_audit,
        meta=meta,
    )


def _fast_net_return_for_events(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon: int,
    delay: int,
    cost: CostConfig,
) -> np.ndarray:
    frame = bars.sort_index()
    pos = _coerce_event_positions(frame, events)
    side = pd.to_numeric(events["side"], errors="coerce").fillna(0).astype(float).to_numpy()
    open_arr = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    close_arr = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    n = len(frame)
    ret = np.full(len(events), np.nan, dtype=float)
    fut = pos + int(horizon)
    ent = pos + int(delay)
    ok = (pos >= 0) & (fut >= 0) & (fut < n) & (ent >= 0) & (ent < n)
    ok &= np.isfinite(open_arr[ent.clip(0, max(0, n - 1))]) & (open_arr[ent.clip(0, max(0, n - 1))] > 0)
    ok &= np.isfinite(close_arr[fut.clip(0, max(0, n - 1))])
    if ok.any():
        ret[ok] = (close_arr[fut[ok]] / open_arr[ent[ok]] - 1.0) * side[ok] - float(cost.round_trip_cost_pct)
    return ret


def _first_touch_outcome_sparse(
    bars: pd.DataFrame,
    signal_pos: np.ndarray,
    side: np.ndarray,
    *,
    target_pct: float,
    stop_pct: float,
    horizon: int,
    entry_delay_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First-touch labels only for event rows, not the full bar axis."""
    frame = bars.sort_index()
    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    opens = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    n = len(frame)
    result = np.full(len(signal_pos), "NO_EVENT", dtype=object)
    touch_bars = np.full(len(signal_pos), np.nan, dtype=float)
    both = np.zeros(len(signal_pos), dtype=bool)
    delay = int(entry_delay_bars)
    h = int(horizon)
    for j, (i0, direction0) in enumerate(zip(signal_pos, side)):
        i = int(i0)
        direction = int(direction0)
        entry_pos = i + delay
        if direction == 0 or i < 0 or entry_pos >= n:
            continue
        entry = opens[entry_pos]
        if not np.isfinite(entry) or entry <= 0:
            result[j] = "NO_ENTRY"
            continue
        if direction == 1:
            target = entry * (1.0 + float(target_pct))
            stop = entry * (1.0 - float(stop_pct))
        else:
            target = entry * (1.0 - float(target_pct))
            stop = entry * (1.0 + float(stop_pct))
        res = "TIMEOUT"
        end = min(n, i + h + 1)
        for k in range(entry_pos, end):
            if direction == 1:
                hit_target = highs[k] >= target
                hit_stop = lows[k] <= stop
            else:
                hit_target = lows[k] <= target
                hit_stop = highs[k] >= stop
            if hit_target and hit_stop:
                both[j] = True
                res = "STOP"  # conservative OHLC path policy
                touch_bars[j] = k - i
                break
            if hit_stop:
                res = "STOP"
                touch_bars[j] = k - i
                break
            if hit_target:
                res = "TARGET"
                touch_bars[j] = k - i
                break
        result[j] = res
    return result, touch_bars, both


# ---------------------------------------------------------------------------
# Winner/loser mining and EMA path study
# ---------------------------------------------------------------------------


def _candidate_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Columns eligible for winner/loser and filter mining.

    Raw absolute prices are intentionally excluded; relative location, volatility,
    order-flow, range, footprint, and EMA shape features are kept.
    """
    exclude_exact = {
        "event_id",
        "signal_time",
        "entry_time",
        "entry_price",
        "signal_bar_pos",
        "entry_bar_pos",
        "year",
        "side",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema_fast",
        "ema_slow",
        "session_pv_cum",
        "session_volume_cum",
        "session_vwap",
    }
    exclude_contains = (
        "_ret_h",
        "next_open_ret_",
        "close_to_close_ret_",
        "mfe_h",
        "mae_h",
        "available_time",
        "_time",
        "_id",
        "flag",
        "eligible",
        "count",
    )
    hints = (
        "atr_pct",
        "ret_",
        "bar_ret",
        "range_pos",
        "prior_range_pct",
        "vwap_dist",
        "volume_ratio",
        "notional_ratio",
        "trades_count_ratio",
        "taker_buy_ratio",
        "delta_notional",
        "large_delta",
        "close_pos",
        "wick_frac",
        "session_",
        "ema_gap_pct",
        "close_to_ema",
        "ema_fast_slope",
        "ema_slow_slope",
        "rb_",
        "fp_",
    )
    cols: list[str] = []
    for col in frame.columns:
        if col in exclude_exact:
            continue
        if any(x in col for x in exclude_contains):
            continue
        if not any(h in col for h in hints):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_bool_dtype(frame[col]):
            x = pd.to_numeric(frame[col], errors="coerce")
            if x.notna().sum() >= 50 and x.nunique(dropna=True) >= 4:
                cols.append(col)
    return sorted(set(cols))


def _prefix_stats(stats: dict[str, object], prefix: str) -> dict[str, object]:
    return {f"{prefix}_{k}": v for k, v in stats.items()}


def _quick_return_stats(values: pd.Series | np.ndarray, *, min_count: int) -> dict[str, object]:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    count = int(len(x))
    if count == 0:
        return {
            "count": 0,
            "eligible": False,
            "mean": np.nan,
            "median": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_win": np.nan,
            "max_loss": np.nan,
        }
    wins = x[x > 0]
    losses = x[x < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    if gross_loss == 0.0:
        pf = np.inf if gross_profit > 0 else np.nan
    else:
        pf = gross_profit / gross_loss
    return {
        "count": count,
        "eligible": bool(count >= int(min_count)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "win_rate": float((x > 0).mean()),
        "profit_factor": float(pf) if np.isfinite(pf) else np.inf,
        "max_win": float(x.max()),
        "max_loss": float(x.min()),
    }


def _split_masks(frame: pd.DataFrame, args: argparse.Namespace) -> dict[str, pd.Series]:
    ts = pd.to_datetime(frame["signal_time"], errors="coerce")
    train_end = pd.Timestamp(args.filter_train_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    valid_end = pd.Timestamp(args.filter_valid_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return {
        "all": ts.notna(),
        "train": ts.notna() & (ts <= train_end),
        "valid": ts.notna() & (ts > train_end) & (ts <= valid_end),
        "test": ts.notna() & (ts > valid_end),
    }


def _filter_mask(frame: pd.DataFrame, feature: str, op: str, threshold: float) -> pd.Series:
    x = pd.to_numeric(frame[feature], errors="coerce")
    if op == ">=":
        return x >= float(threshold)
    if op == "<=":
        return x <= float(threshold)
    raise ValueError(f"unsupported op: {op}")


def _eval_filtered_subset(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    return_col: str,
    args: argparse.Namespace,
    min_count: int,
    split_masks: dict[str, pd.Series] | None = None,
    with_active: bool = False,
) -> dict[str, object]:
    out: dict[str, object] = {}
    masks = split_masks if split_masks is not None else _split_masks(frame, args)
    for split, sm in masks.items():
        part = frame.loc[(mask & sm).fillna(False)]
        out.update(_prefix_stats(_quick_return_stats(part[return_col], min_count=min_count), split))
    all_part = frame.loc[mask.fillna(False)]
    if with_active:
        active = _active_days_metrics(all_part["signal_time"], start_date=args.start_date, end_date=args.end_date) if not all_part.empty else {
            "active_days": 0,
            "total_days": 0,
            "active_days_ratio": np.nan,
            "max_days_without_trade": np.nan,
        }
        out.update({f"all_{k}": v for k, v in active.items()})
    if not all_part.empty:
        tmp = all_part.copy()
        tmp["_year"] = pd.to_datetime(tmp["signal_time"], errors="coerce").dt.year
        ystats = []
        for _, yp in tmp.groupby("_year", sort=True):
            y = summarize_returns(yp[return_col], name=return_col, min_count=max(20, min_count // 4))
            ystats.append(float(y.get("mean", np.nan)) > 0 if pd.notna(y.get("mean", np.nan)) else False)
        out["positive_years"] = int(sum(ystats))
        out["year_rows"] = int(len(ystats))
    else:
        out["positive_years"] = 0
        out["year_rows"] = 0
    return out


def _winner_loser_feature_diff(events: pd.DataFrame, args: argparse.Namespace, *, return_col: str, label: str) -> pd.DataFrame:
    if events.empty or return_col not in events.columns:
        return pd.DataFrame()
    cols = _candidate_feature_columns(events)
    rows: list[dict[str, object]] = []
    min_side = max(30, int(args.filter_min_count) // 5)
    for (family, side_name), part in events.groupby(["event_family", "side_name"], sort=False):
        y = pd.to_numeric(part[return_col], errors="coerce")
        win_mask = y > 0
        loss_mask = y <= 0
        if int(win_mask.sum()) < min_side or int(loss_mask.sum()) < min_side:
            continue
        for col in cols:
            x = pd.to_numeric(part[col], errors="coerce")
            xw = x[win_mask].dropna()
            xl = x[loss_mask].dropna()
            if len(xw) < min_side or len(xl) < min_side:
                continue
            pooled = float(pd.concat([xw, xl]).std(ddof=0))
            mean_win = float(xw.mean())
            mean_loss = float(xl.mean())
            effect = (mean_win - mean_loss) / pooled if pooled and math.isfinite(pooled) else np.nan
            rows.append(
                {
                    "label": label,
                    "event_family": family,
                    "side_name": side_name,
                    "feature": col,
                    "winner_count": int(len(xw)),
                    "loser_count": int(len(xl)),
                    "winner_mean": mean_win,
                    "loser_mean": mean_loss,
                    "winner_median": float(xw.median()),
                    "loser_median": float(xl.median()),
                    "mean_diff": mean_win - mean_loss,
                    "effect_z": float(effect) if pd.notna(effect) else np.nan,
                    "abs_effect_z": float(abs(effect)) if pd.notna(effect) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["abs_effect_z", "winner_count"], ascending=[False, False])
    return out


def _build_univariate_filter_candidates(events: pd.DataFrame, args: argparse.Namespace, *, return_col: str, label: str) -> pd.DataFrame:
    if events.empty or return_col not in events.columns:
        return pd.DataFrame()
    cols = _candidate_feature_columns(events)
    quantiles = [q for q in _parse_number_list(args.filter_quantiles, name="filter_quantiles") if 0.0 < float(q) < 1.0]
    min_count = int(args.filter_min_count)
    min_split = int(args.filter_min_split_count)
    rows: list[dict[str, object]] = []
    groups = list(events.groupby(["event_family", "side_name"], sort=False))
    progress = ProgressReporter("[filter] univariate train", total=max(1, len(groups) * len(cols)), every=max(1, len(cols)), enabled=not bool(args.no_progress))
    done = 0
    split_masks = _split_masks(events, args)
    train_mask_all = split_masks["train"]
    for (family, side_name), group_part in groups:
        group_idx = group_part.index
        train_group_idx = group_idx[train_mask_all.loc[group_idx].fillna(False)]
        if len(train_group_idx) < min_count:
            done += len(cols)
            progress.update(done)
            continue
        group_mask = events.index.isin(group_idx)
        train_group = events.loc[train_group_idx]
        base_train = _quick_return_stats(train_group[return_col], min_count=min_count)
        for col in cols:
            x_train = pd.to_numeric(train_group[col], errors="coerce").dropna()
            if len(x_train) < min_count or x_train.nunique(dropna=True) < 4:
                done += 1
                progress.update(done)
                continue
            thresholds = sorted(set(float(v) for v in x_train.quantile(quantiles).dropna().to_numpy()))
            for thr in thresholds:
                for op in (">=", "<="):
                    cond = _filter_mask(events, col, op, thr)
                    full_mask = pd.Series(group_mask, index=events.index) & cond
                    train_count = int((full_mask & train_mask_all).sum())
                    if train_count < min_count:
                        continue
                    eval_stats = _eval_filtered_subset(events, full_mask, return_col=return_col, args=args, min_count=min_split, split_masks=split_masks, with_active=False)
                    rows.append(
                        {
                            "label": label,
                            "filter_type": "univariate",
                            "event_family": family,
                            "side_name": side_name,
                            "feature": col,
                            "op": op,
                            "threshold": float(thr),
                            "filter_expr": f"{col} {op} {thr:.12g}",
                            "base_train_count": base_train.get("count", np.nan),
                            "base_train_mean": base_train.get("mean", np.nan),
                            "base_train_profit_factor": base_train.get("profit_factor", np.nan),
                            **eval_stats,
                        }
                    )
            done += 1
            progress.update(done)
    progress.close()
    out = pd.DataFrame(rows)
    if not out.empty:
        out["train_pf_num"] = pd.to_numeric(out["train_profit_factor"].replace("inf", np.inf), errors="coerce")
        out["valid_pf_num"] = pd.to_numeric(out["valid_profit_factor"].replace("inf", np.inf), errors="coerce")
        out["test_pf_num"] = pd.to_numeric(out["test_profit_factor"].replace("inf", np.inf), errors="coerce")
        out["score"] = out["train_pf_num"].clip(upper=5).fillna(0) * np.log1p(pd.to_numeric(out["train_count"], errors="coerce").fillna(0))
        out = out.sort_values(["test_pf_num", "valid_pf_num", "train_pf_num", "train_count"], ascending=[False, False, False, False])
        if int(args.filter_max_candidates) > 0 and len(out) > int(args.filter_max_candidates):
            out = out.head(int(args.filter_max_candidates)).copy()
    return out


def _filter_candidate_mask(events: pd.DataFrame, row: pd.Series) -> pd.Series:
    group_mask = (events["event_family"].astype(str) == str(row["event_family"])) & (events["side_name"].astype(str) == str(row["side_name"]))
    if str(row.get("filter_type", "")) == "combo2":
        m1 = _filter_mask(events, str(row["feature_1"]), str(row["op_1"]), float(row["threshold_1"]))
        m2 = _filter_mask(events, str(row["feature_2"]), str(row["op_2"]), float(row["threshold_2"]))
        return group_mask & m1 & m2
    return group_mask & _filter_mask(events, str(row["feature"]), str(row["op"]), float(row["threshold"]))


def _build_combo_filter_candidates(events: pd.DataFrame, uni: pd.DataFrame, args: argparse.Namespace, *, return_col: str, label: str) -> pd.DataFrame:
    if events.empty or uni.empty:
        return pd.DataFrame()
    min_count = int(args.filter_min_count)
    min_split = int(args.filter_min_split_count)
    split_masks = _split_masks(events, args)
    train_mask_all = split_masks["train"]
    rows: list[dict[str, object]] = []
    max_total = int(args.filter_max_combos)
    groups = list(uni.groupby(["event_family", "side_name"], sort=False))
    progress = ProgressReporter("[filter] combo2 train", total=max(1, len(groups)), every=1, enabled=not bool(args.no_progress))
    done = 0
    for (family, side_name), upart in groups:
        up = upart.sort_values(["train_pf_num", "valid_pf_num", "train_count"], ascending=[False, False, False]).head(int(args.filter_top_per_group)).copy()
        records = up.to_dict("records")
        if len(records) < 2:
            done += 1
            progress.update(done)
            continue
        group_mask = (events["event_family"].astype(str) == str(family)) & (events["side_name"].astype(str) == str(side_name))
        base_train = _quick_return_stats(events.loc[group_mask & train_mask_all, return_col], min_count=min_count)
        pre_masks: list[pd.Series] = []
        for r in records:
            pre_masks.append(_filter_mask(events, str(r["feature"]), str(r["op"]), float(r["threshold"])))
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                if len(rows) >= max_total:
                    break
                r1 = records[i]
                r2 = records[j]
                if str(r1["feature"]) == str(r2["feature"]):
                    continue
                full_mask = group_mask & pre_masks[i] & pre_masks[j]
                train_count = int((full_mask & train_mask_all).sum())
                if train_count < min_count:
                    continue
                eval_stats = _eval_filtered_subset(events, full_mask, return_col=return_col, args=args, min_count=min_split, split_masks=split_masks, with_active=False)
                rows.append(
                    {
                        "label": label,
                        "filter_type": "combo2",
                        "event_family": family,
                        "side_name": side_name,
                        "feature_1": str(r1["feature"]),
                        "op_1": str(r1["op"]),
                        "threshold_1": float(r1["threshold"]),
                        "feature_2": str(r2["feature"]),
                        "op_2": str(r2["op"]),
                        "threshold_2": float(r2["threshold"]),
                        "filter_expr": f"{r1['feature']} {r1['op']} {float(r1['threshold']):.12g} AND {r2['feature']} {r2['op']} {float(r2['threshold']):.12g}",
                        "base_train_count": base_train.get("count", np.nan),
                        "base_train_mean": base_train.get("mean", np.nan),
                        "base_train_profit_factor": base_train.get("profit_factor", np.nan),
                        **eval_stats,
                    }
                )
            if len(rows) >= max_total:
                break
        done += 1
        progress.update(done)
        if len(rows) >= max_total:
            break
    progress.close()
    out = pd.DataFrame(rows)
    if not out.empty:
        out["train_pf_num"] = pd.to_numeric(out["train_profit_factor"].replace("inf", np.inf), errors="coerce")
        out["valid_pf_num"] = pd.to_numeric(out["valid_profit_factor"].replace("inf", np.inf), errors="coerce")
        out["test_pf_num"] = pd.to_numeric(out["test_profit_factor"].replace("inf", np.inf), errors="coerce")
        out = out.sort_values(["test_pf_num", "valid_pf_num", "train_pf_num", "train_count"], ascending=[False, False, False, False])
    return out




def _attach_filter_active_metrics(events: pd.DataFrame, filters: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if events.empty or filters.empty:
        return filters
    rows: list[dict[str, object]] = []
    for _, row in filters.iterrows():
        mask = _filter_candidate_mask(events, row)
        part = events.loc[mask.fillna(False)]
        active = _active_days_metrics(part["signal_time"], start_date=args.start_date, end_date=args.end_date) if not part.empty else {
            "active_days": 0,
            "total_days": 0,
            "active_days_ratio": np.nan,
            "max_days_without_trade": np.nan,
        }
        rows.append({f"all_{k}": v for k, v in active.items()})
    active_df = pd.DataFrame(rows, index=filters.index)
    out = filters.copy()
    for col in active_df.columns:
        out[col] = active_df[col]
    return out

def _filter_candidate_gate(filters: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if filters.empty:
        return pd.DataFrame()
    out = filters.copy()
    for prefix in ("all", "train", "valid", "test"):
        if f"{prefix}_profit_factor" in out.columns:
            out[f"{prefix}_pf_num"] = pd.to_numeric(out[f"{prefix}_profit_factor"].replace("inf", np.inf), errors="coerce")
        if f"{prefix}_win_rate" in out.columns:
            out[f"{prefix}_win_rate_num"] = pd.to_numeric(out[f"{prefix}_win_rate"], errors="coerce")
    min_count = int(args.filter_min_count)
    min_split = int(args.filter_min_split_count)
    out["gate_count"] = pd.to_numeric(out.get("all_count", 0), errors="coerce") >= min_count
    out["gate_train_pf"] = out.get("train_pf_num", pd.Series(index=out.index, dtype=float)) >= float(args.min_profit_factor)
    out["gate_valid_pf"] = out.get("valid_pf_num", pd.Series(index=out.index, dtype=float)) >= 1.0
    out["gate_test_pf"] = out.get("test_pf_num", pd.Series(index=out.index, dtype=float)) >= 1.0
    out["gate_train_count"] = pd.to_numeric(out.get("train_count", 0), errors="coerce") >= min_count
    out["gate_valid_count"] = pd.to_numeric(out.get("valid_count", 0), errors="coerce") >= min_split
    out["gate_test_count"] = pd.to_numeric(out.get("test_count", 0), errors="coerce") >= min_split
    out["gate_active_days"] = pd.to_numeric(out.get("all_active_days_ratio", np.nan), errors="coerce") >= float(args.min_active_days_ratio)
    out["gate_no_trade_gap"] = pd.to_numeric(out.get("all_max_days_without_trade", np.nan), errors="coerce") <= int(args.max_days_without_trade)
    out["gate_positive_years"] = pd.to_numeric(out.get("positive_years", 0), errors="coerce") >= 3
    gates = [c for c in out.columns if c.startswith("gate_")]
    out["candidate_pass"] = out[gates].all(axis=1) if gates else False
    sort_cols = [c for c in ["candidate_pass", "test_pf_num", "valid_pf_num", "train_pf_num", "all_count"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] + [False] * (len(sort_cols) - 1))
    return out


def _filter_stability_long(events: pd.DataFrame, filters: pd.DataFrame, args: argparse.Namespace, *, return_col: str) -> pd.DataFrame:
    if events.empty or filters.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    top = filters.head(int(args.filter_stability_top_n)).copy()
    ts = pd.to_datetime(events["signal_time"], errors="coerce")
    split_masks = _split_masks(events, args)
    years = sorted(y for y in ts.dt.year.dropna().unique())
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        mask = _filter_candidate_mask(events, row)
        base_meta = {
            "rank": rank,
            "label": row.get("label", ""),
            "filter_type": row.get("filter_type", ""),
            "event_family": row.get("event_family", ""),
            "side_name": row.get("side_name", ""),
            "filter_expr": row.get("filter_expr", ""),
        }
        for split, sm in split_masks.items():
            part = events.loc[(mask & sm).fillna(False)]
            rows.append({**base_meta, "period": split, **summarize_returns(part[return_col], name=return_col, min_count=int(args.filter_min_split_count))})
        for y in years:
            part = events.loc[(mask & (ts.dt.year == y)).fillna(False)]
            rows.append({**base_meta, "period": f"year_{int(y)}", **summarize_returns(part[return_col], name=return_col, min_count=max(20, int(args.filter_min_split_count) // 2))})
    return pd.DataFrame(rows)




def _prepare_events_for_filter_mining(events: pd.DataFrame, args: argparse.Namespace, *, label: str) -> pd.DataFrame:
    if events.empty:
        return events
    out = events.copy()
    before = len(out)
    if not bool(getattr(args, "no_filter_dedupe", False)) and {"event_family", "side_name", "signal_time"}.issubset(out.columns):
        out = out.sort_values(["event_family", "side_name", "signal_time", "event_name"], kind="mergesort")
        out = out.drop_duplicates(["event_family", "side_name", "signal_time"], keep="last").reset_index(drop=True)
    cap = int(getattr(args, "filter_max_rows_per_group", 0) or 0)
    if cap > 0 and {"event_family", "side_name"}.issubset(out.columns):
        parts: list[pd.DataFrame] = []
        for _, part in out.groupby(["event_family", "side_name"], sort=False):
            if len(part) > cap:
                step = max(1, int(math.ceil(len(part) / cap)))
                parts.append(part.iloc[::step].copy())
            else:
                parts.append(part)
        out = pd.concat(parts, ignore_index=True) if parts else out.iloc[0:0].copy()
    after = len(out)
    if after != before:
        print(f"[filter] {label} mining rows {before:,} -> {after:,} after dedupe/cap", flush=True)
    return out

def run_filter_mining(events: pd.DataFrame, args: argparse.Namespace, *, return_col: str, label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Winner/loser separation plus train-first univariate/combo filter mining."""
    if events.empty or return_col not in events.columns:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty
    events = _prepare_events_for_filter_mining(events, args, label=label)
    if events.empty or return_col not in events.columns:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty
    print(f"[filter] {label} winner/loser diff", flush=True)
    diff = _winner_loser_feature_diff(events, args, return_col=return_col, label=label)
    print(f"[filter] {label} univariate", flush=True)
    uni = _build_univariate_filter_candidates(events, args, return_col=return_col, label=label)
    print(f"[filter] {label} combo2", flush=True)
    combo = _build_combo_filter_candidates(events, uni, args, return_col=return_col, label=label)
    combined = pd.concat([x for x in [uni, combo] if not x.empty], ignore_index=True) if (not uni.empty or not combo.empty) else pd.DataFrame()
    if not combined.empty:
        combined = combined.head(int(args.filter_max_candidates)).copy()
        combined = _attach_filter_active_metrics(events, combined, args)
    gate = _filter_candidate_gate(combined, args)
    stability = _filter_stability_long(events, gate, args, return_col=return_col)
    return diff, uni, combo, gate, stability




def _run_filtered_tp_sl_grid(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    filters: pd.DataFrame,
    args: argparse.Namespace,
    *,
    label: str,
    use_ema_grid: bool = False,
) -> pd.DataFrame:
    """TP/SL path grid after discovered filters, limited to top candidates."""
    if events.empty or filters.empty:
        return pd.DataFrame()
    top_n = int(args.filter_tp_sl_top_n)
    if top_n <= 0:
        return pd.DataFrame()
    top = filters.head(top_n).copy()
    if use_ema_grid:
        targets = _parse_number_list(args.ema_path_target_pcts, name="ema_path_target_pcts")
        stops = _parse_number_list(args.ema_path_stop_pcts, name="ema_path_stop_pcts")
        horizons = _parse_int_list(args.ema_path_horizons, name="ema_path_horizons")
    else:
        targets = _parse_number_list(args.touch_target_pcts, name="touch_target_pcts")
        stops = _parse_number_list(args.touch_stop_pcts, name="touch_stop_pcts")
        horizons = [int(args.touch_horizon)]
    total = len(top) * len(targets) * len(stops) * len(horizons)
    progress = ProgressReporter(f"[filter-touch] {label}", total=max(1, total), every=max(1, total // 50), enabled=not bool(args.no_progress))
    rows: list[dict[str, object]] = []
    done = 0
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        mask = _filter_candidate_mask(events, row)
        part = events.loc[mask.fillna(False)].copy()
        if part.empty:
            done += len(targets) * len(stops) * len(horizons)
            progress.update(done)
            continue
        pos = _coerce_event_positions(bars, part)
        side = pd.to_numeric(part["side"], errors="coerce").fillna(0).astype(int).to_numpy()
        ok = pos >= 0
        pos = pos[ok]
        side = side[ok]
        if len(pos) == 0:
            done += len(targets) * len(stops) * len(horizons)
            progress.update(done)
            continue
        for horizon in horizons:
            for target in targets:
                for stop in stops:
                    result, touch_bars, both = _first_touch_outcome_sparse(
                        bars,
                        pos,
                        side,
                        target_pct=float(target),
                        stop_pct=float(stop),
                        horizon=int(horizon),
                        entry_delay_bars=1,
                    )
                    count = int(len(result))
                    if count:
                        target_rate = float((result == "TARGET").mean())
                        stop_rate = float((result == "STOP").mean())
                        timeout_rate = float((result == "TIMEOUT").mean())
                        both_rate = float(both.mean())
                        avg_touch_bars = float(np.nanmean(touch_bars)) if np.isfinite(touch_bars).any() else np.nan
                        expectancy_proxy = target_rate * float(target) - stop_rate * float(stop) - float(_cost_from_args(args).round_trip_cost_pct)
                    else:
                        target_rate = stop_rate = timeout_rate = both_rate = avg_touch_bars = expectancy_proxy = np.nan
                    rows.append(
                        {
                            "label": label,
                            "rank": rank,
                            "filter_type": row.get("filter_type", ""),
                            "event_family": row.get("event_family", ""),
                            "side_name": row.get("side_name", ""),
                            "filter_expr": row.get("filter_expr", ""),
                            "horizon": int(horizon),
                            "target_pct": float(target),
                            "stop_pct": float(stop),
                            "count": count,
                            "target_rate": target_rate,
                            "stop_rate": stop_rate,
                            "timeout_rate": timeout_rate,
                            "same_bar_both_hit_rate": both_rate,
                            "avg_touch_bars": avg_touch_bars,
                            "expectancy_proxy_net": expectancy_proxy,
                        }
                    )
                    done += 1
                    progress.update(done)
    progress.close()
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["expectancy_proxy_net", "target_rate", "same_bar_both_hit_rate"], ascending=[False, False, True])
    return out

def build_ema_armed_entry_events(features: pd.DataFrame, args: argparse.Namespace, extra_cols: Sequence[str]) -> pd.DataFrame:
    """EMA armed-entry candidates only; no old EMA exit logic is used here."""
    df = features.sort_index()
    idx = df.index
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    ema_slow = pd.to_numeric(df["ema_slow"], errors="coerce").to_numpy(dtype=float)
    n = len(df)
    entry_buffer = float(args.ema_entry_buffer_pct)
    armed_long = False
    armed_short = False
    long_age = 0
    short_age = 0
    rows: list[dict[str, object]] = []
    cols = [c for c in extra_cols if c in df.columns]
    for i in range(1, n):
        if not all(np.isfinite(x) for x in (close[i], close[i - 1], ema_slow[i], ema_slow[i - 1])):
            continue
        cross_up = close[i - 1] <= ema_slow[i - 1] and close[i] > ema_slow[i]
        cross_down = close[i - 1] >= ema_slow[i - 1] and close[i] < ema_slow[i]
        if cross_up:
            armed_long = True
            armed_short = False
            long_age = 0
        elif cross_down:
            armed_short = True
            armed_long = False
            short_age = 0
        if armed_long:
            long_age += 1
            if close[i] < ema_slow[i]:
                armed_long = False
                long_age = 0
            elif close[i] >= ema_slow[i] * (1.0 + entry_buffer):
                row = {c: df.iloc[i][c] for c in cols}
                row.update(
                    {
                        "signal_time": idx[i],
                        "side": 1,
                        "side_name": "LONG",
                        "event_name": f"ema20_50_armed_entry|long|buffer{entry_buffer:g}",
                        "event_family": "ema20_50_armed_entry",
                        "ema_armed_age": int(long_age),
                    }
                )
                rows.append(row)
                armed_long = False
                long_age = 0
        if armed_short:
            short_age += 1
            if close[i] > ema_slow[i]:
                armed_short = False
                short_age = 0
            elif close[i] <= ema_slow[i] * (1.0 - entry_buffer):
                row = {c: df.iloc[i][c] for c in cols}
                row.update(
                    {
                        "signal_time": idx[i],
                        "side": -1,
                        "side_name": "SHORT",
                        "event_name": f"ema20_50_armed_entry|short|buffer{entry_buffer:g}",
                        "event_family": "ema20_50_armed_entry",
                        "ema_armed_age": int(short_age),
                    }
                )
                rows.append(row)
                armed_short = False
                short_age = 0
    out = pd.DataFrame(rows)
    if not out.empty:
        out["event_id"] = np.arange(len(out), dtype=np.int64)
        out = out.sort_values(["signal_time", "event_name", "side"]).reset_index(drop=True)
    print(f"[ema-path] armed entry events={len(out):,}", flush=True)
    return out


def _ema_path_summary(bars: pd.DataFrame, ema_events: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ema_events.empty:
        return pd.DataFrame(), pd.DataFrame()
    horizons = tuple(_parse_int_list(args.ema_path_horizons, name="ema_path_horizons"))
    cfg = EventStudyConfig(
        horizons=horizons,
        mfe_mae_horizon=max(horizons),
        entry_delay_bars=1,
        cost=_cost_from_args(args),
        context_available_time_cols=tuple(c for c in ema_events.columns if c.endswith("_available_time")),
        min_count=int(args.min_count),
        progress_every=int(args.progress_every) if not bool(args.no_progress) else 0,
    )
    result = run_event_study_fast(bars, ema_events, cfg)
    enriched = result.events
    rows: list[dict[str, object]] = []
    for h in horizons:
        col = f"next_open_ret_h{int(h)}_net"
        for keys, part in enriched.groupby(["event_family", "side_name"], sort=False):
            stats = summarize_returns(part[col], name=col, min_count=int(args.min_count))
            active = _active_days_metrics(part["signal_time"], start_date=args.start_date, end_date=args.end_date)
            rows.append({"event_family": keys[0], "side_name": keys[1], "horizon": int(h), **stats, **active})
    return pd.DataFrame(rows), enriched


def _run_ema_tp_sl_grid(bars: pd.DataFrame, ema_events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if ema_events.empty:
        return pd.DataFrame()
    targets = _parse_number_list(args.ema_path_target_pcts, name="ema_path_target_pcts")
    stops = _parse_number_list(args.ema_path_stop_pcts, name="ema_path_stop_pcts")
    horizons = _parse_int_list(args.ema_path_horizons, name="ema_path_horizons")
    groups = list(ema_events.groupby("side_name", sort=False))
    total = len(groups) * len(targets) * len(stops) * len(horizons)
    progress = ProgressReporter("[ema-path] TP/SL grid", total=max(1, total), every=max(1, total // 50), enabled=not bool(args.no_progress))
    rows: list[dict[str, object]] = []
    done = 0
    for side_name, part in groups:
        pos = _coerce_event_positions(bars, part)
        side = pd.to_numeric(part["side"], errors="coerce").fillna(0).astype(int).to_numpy()
        ok = pos >= 0
        pos = pos[ok]
        side = side[ok]
        for horizon in horizons:
            for target in targets:
                for stop in stops:
                    result, touch_bars, both = _first_touch_outcome_sparse(
                        bars,
                        pos,
                        side,
                        target_pct=float(target),
                        stop_pct=float(stop),
                        horizon=int(horizon),
                        entry_delay_bars=1,
                    )
                    count = int(len(result))
                    if count:
                        target_rate = float((result == "TARGET").mean())
                        stop_rate = float((result == "STOP").mean())
                        timeout_rate = float((result == "TIMEOUT").mean())
                        both_rate = float(both.mean())
                        avg_touch_bars = float(np.nanmean(touch_bars)) if np.isfinite(touch_bars).any() else np.nan
                        expectancy_proxy = target_rate * float(target) - stop_rate * float(stop) - float(_cost_from_args(args).round_trip_cost_pct)
                    else:
                        target_rate = stop_rate = timeout_rate = both_rate = avg_touch_bars = expectancy_proxy = np.nan
                    rows.append(
                        {
                            "event_family": "ema20_50_armed_entry",
                            "side_name": side_name,
                            "horizon": int(horizon),
                            "target_pct": float(target),
                            "stop_pct": float(stop),
                            "count": count,
                            "target_rate": target_rate,
                            "stop_rate": stop_rate,
                            "timeout_rate": timeout_rate,
                            "same_bar_both_hit_rate": both_rate,
                            "avg_touch_bars": avg_touch_bars,
                            "expectancy_proxy_net": expectancy_proxy,
                        }
                    )
                    done += 1
                    progress.update(done)
    progress.close()
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["expectancy_proxy_net", "target_rate", "same_bar_both_hit_rate"], ascending=[False, False, True])
    return out


# ---------------------------------------------------------------------------
# Passive limit-entry setup lab
# ---------------------------------------------------------------------------


def _safe_numeric_col(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _rank_filter_candidates_for_passive(filters: pd.DataFrame, *, label: str, args: argparse.Namespace) -> pd.DataFrame:
    if filters.empty:
        return pd.DataFrame()
    out = filters.copy()
    out["label"] = out.get("label", label)
    # Prefer filters that do not die on valid/test, but do not require pass=true
    # because the whole point of this lab is to check whether better passive
    # entry can rescue thin gross-edge setups.
    for c in ("test_pf_num", "valid_pf_num", "train_pf_num"):
        if c not in out.columns and c.replace("_num", "") in out.columns:
            out[c] = pd.to_numeric(out[c.replace("_num", "")].replace("inf", np.inf), errors="coerce")
    if "all_mean" not in out.columns:
        out["all_mean"] = pd.to_numeric(out.get("all_mean", np.nan), errors="coerce")
    if "all_count" not in out.columns:
        out["all_count"] = pd.to_numeric(out.get("all_count", 0), errors="coerce")
    for c in ("test_pf_num", "valid_pf_num", "train_pf_num", "all_mean", "all_count"):
        if c not in out.columns:
            out[c] = np.nan
    out["_passive_rank_score"] = (
        out["test_pf_num"].clip(upper=5).fillna(0) * 100.0
        + out["valid_pf_num"].clip(upper=5).fillna(0) * 30.0
        + out["train_pf_num"].clip(upper=5).fillna(0) * 10.0
        + out["all_mean"].fillna(-1.0) * 10_000.0
        + np.log1p(out["all_count"].fillna(0))
    )
    out = out.sort_values(["_passive_rank_score", "all_count"], ascending=[False, False]).head(int(args.passive_top_filters)).copy()
    out["passive_spec_source"] = label
    return out.reset_index(drop=True)


def _curated_passive_specs(label: str) -> pd.DataFrame:
    # These are not claims of edge. They are the narrow directions suggested by
    # the v4 report: extreme VWAP fade / impulse exhaustion / sweep-reclaim,
    # tested again under passive entry. Keeping this tiny avoids wasting time.
    if label != "mhf_event":
        return pd.DataFrame()
    rows = [
        {"label": label, "passive_spec_source": "curated", "filter_type": "all", "event_family": "vwap_extension_fade", "side_name": "LONG", "filter_expr": "ALL vwap_extension_fade LONG"},
        {"label": label, "passive_spec_source": "curated", "filter_type": "all", "event_family": "impulse_exhaustion", "side_name": "LONG", "filter_expr": "ALL impulse_exhaustion LONG"},
        {"label": label, "passive_spec_source": "curated", "filter_type": "all", "event_family": "micro_sweep_reclaim", "side_name": "LONG", "filter_expr": "ALL micro_sweep_reclaim LONG"},
        {"label": label, "passive_spec_source": "curated", "filter_type": "all", "event_family": "micro_sweep_reclaim", "side_name": "SHORT", "filter_expr": "ALL micro_sweep_reclaim SHORT"},
    ]
    return pd.DataFrame(rows)


def _candidate_mask_for_passive(events: pd.DataFrame, row: pd.Series) -> pd.Series:
    group_mask = (events["event_family"].astype(str) == str(row.get("event_family", ""))) & (events["side_name"].astype(str) == str(row.get("side_name", "")))
    ftype = str(row.get("filter_type", ""))
    if ftype in {"", "all", "ALL", "nan"}:
        return group_mask
    return _filter_candidate_mask(events, row)


def _cap_passive_setups(part: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    cap = int(args.passive_max_setups_per_spec)
    if cap <= 0 or len(part) <= cap:
        return part
    step = int(math.ceil(len(part) / cap))
    return part.iloc[::step].head(cap).copy()


def _simulate_passive_for_spec_grid(
    bars: pd.DataFrame,
    setups: pd.DataFrame,
    *,
    side_value: int,
    offset_pct: float,
    fill_window: int,
    target_pct: float,
    stop_pct: float,
    horizon: int,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if setups.empty:
        return pd.DataFrame()
    frame = bars.sort_index()
    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    opens = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)

    if "signal_bar_pos" in setups.columns:
        sig_pos = pd.to_numeric(setups["signal_bar_pos"], errors="coerce").fillna(-1).astype(int).to_numpy()
    else:
        sig_pos = _coerce_event_positions(frame, setups)
    sides = pd.to_numeric(setups.get("side", side_value), errors="coerce").fillna(side_value).astype(int).to_numpy()
    signal_times = pd.to_datetime(setups["signal_time"], errors="coerce").to_numpy()

    entry_maker_fee = float(args.passive_entry_maker_fee_rate)
    tp_maker_fee = float(args.passive_tp_maker_fee_rate)
    stop_cost = entry_maker_fee + float(args.passive_stop_taker_fee_rate) + float(args.passive_stop_slippage_pct)
    target_cost = entry_maker_fee + tp_maker_fee
    timeout_cost = entry_maker_fee + float(args.passive_timeout_taker_fee_rate) + float(args.passive_timeout_slippage_pct)

    rows: list[dict[str, object]] = []
    fw = max(1, int(fill_window))
    h = max(1, int(horizon))
    off = float(offset_pct)
    for j, i0 in enumerate(sig_pos):
        i = int(i0)
        side = int(sides[j]) if j < len(sides) else int(side_value)
        if side == 0 or i < 0 or i + 1 >= n or not np.isfinite(closes[i]) or closes[i] <= 0:
            continue
        if side == 1:
            limit_px = closes[i] * (1.0 - off)
        else:
            limit_px = closes[i] * (1.0 + off)
        if not np.isfinite(limit_px) or limit_px <= 0:
            continue

        fill_pos = -1
        fill_end = min(n - 1, i + fw)
        for k in range(i + 1, fill_end + 1):
            if side == 1:
                filled = np.isfinite(lows[k]) and lows[k] <= limit_px
            else:
                filled = np.isfinite(highs[k]) and highs[k] >= limit_px
            if filled:
                fill_pos = k
                break
        if fill_pos < 0:
            continue

        if side == 1:
            target_px = limit_px * (1.0 + float(target_pct))
            stop_px = limit_px * (1.0 - float(stop_pct))
        else:
            target_px = limit_px * (1.0 - float(target_pct))
            stop_px = limit_px * (1.0 + float(stop_pct))

        result = "TIMEOUT"
        exit_pos = min(n - 1, i + h)
        exit_px = closes[exit_pos]
        both_hit = False
        touch_bars = np.nan
        scan_end = min(n - 1, i + h)
        for k in range(fill_pos, scan_end + 1):
            if side == 1:
                hit_target = np.isfinite(highs[k]) and highs[k] >= target_px
                hit_stop = np.isfinite(lows[k]) and lows[k] <= stop_px
            else:
                hit_target = np.isfinite(lows[k]) and lows[k] <= target_px
                hit_stop = np.isfinite(highs[k]) and highs[k] >= stop_px
            if hit_target and hit_stop:
                both_hit = True
                result = "STOP"  # conservative OHLC path policy
                exit_pos = k
                exit_px = stop_px
                touch_bars = k - i
                break
            if hit_stop:
                result = "STOP"
                exit_pos = k
                exit_px = stop_px
                touch_bars = k - i
                break
            if hit_target:
                result = "TARGET"
                exit_pos = k
                exit_px = target_px
                touch_bars = k - i
                break
        if not np.isfinite(exit_px) or exit_px <= 0:
            continue
        if result == "TARGET":
            gross = float(target_pct)
            net = gross - target_cost
        elif result == "STOP":
            gross = -float(stop_pct)
            net = gross - stop_cost
        else:
            gross = (exit_px / limit_px - 1.0) * side
            net = gross - timeout_cost
        rows.append(
            {
                "signal_time": signal_times[j] if j < len(signal_times) else pd.NaT,
                "signal_bar_pos": int(i),
                "side": int(side),
                "side_name": _side_name(side),
                "signal_close": float(closes[i]),
                "entry_time": idx[fill_pos],
                "entry_bar_pos": int(fill_pos),
                "entry_price": float(limit_px),
                "entry_offset_pct": float(off),
                "fill_window": int(fw),
                "fill_bars_after_signal": int(fill_pos - i),
                "exit_time": idx[exit_pos],
                "exit_bar_pos": int(exit_pos),
                "exit_price": float(exit_px),
                "exit_result": result,
                "target_pct": float(target_pct),
                "stop_pct": float(stop_pct),
                "horizon": int(h),
                "bars_held_after_fill": int(exit_pos - fill_pos),
                "touch_bars_after_signal": float(touch_bars) if pd.notna(touch_bars) else np.nan,
                "same_bar_both_hit": bool(both_hit),
                "gross_return": float(gross),
                "net_return": float(net),
                "entry_maker_fee": float(entry_maker_fee),
                "target_exit_maker_fee": float(tp_maker_fee),
                "stop_exit_taker_fee": float(args.passive_stop_taker_fee_rate),
                "timeout_exit_taker_fee": float(args.passive_timeout_taker_fee_rate),
                "stop_slippage_pct": float(args.passive_stop_slippage_pct),
                "timeout_slippage_pct": float(args.passive_timeout_slippage_pct),
            }
        )
    return pd.DataFrame(rows)


def _passive_summary_row(trades: pd.DataFrame, *, setup_count: int, spec_meta: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    row = dict(spec_meta)
    row["setup_count"] = int(setup_count)
    row["trade_count"] = int(len(trades))
    row["fill_rate"] = float(len(trades) / setup_count) if setup_count else np.nan
    if trades.empty:
        row.update({
            "mean": np.nan,
            "median": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "total_return": np.nan,
            "max_drawdown": np.nan,
            "target_rate": np.nan,
            "stop_rate": np.nan,
            "timeout_rate": np.nan,
            "same_bar_both_hit_rate": np.nan,
            "avg_fill_bars": np.nan,
            "avg_bars_held_after_fill": np.nan,
            "active_days": 0,
            "active_days_ratio": np.nan,
            "max_days_without_trade": np.nan,
            "positive_years": 0,
        })
        return row
    stats = _summarize_equity(trades["net_return"])
    active = _active_days_metrics(trades["entry_time"], start_date=args.start_date, end_date=args.end_date)
    years = pd.to_datetime(trades["entry_time"], errors="coerce").dt.year
    positive_years = 0
    year_rows = 0
    for _, yp in trades.assign(_year=years).dropna(subset=["_year"]).groupby("_year", sort=True):
        year_rows += 1
        if pd.to_numeric(yp["net_return"], errors="coerce").mean() > 0:
            positive_years += 1
    row.update(stats)
    row.update(active)
    row["target_rate"] = float((trades["exit_result"] == "TARGET").mean())
    row["stop_rate"] = float((trades["exit_result"] == "STOP").mean())
    row["timeout_rate"] = float((trades["exit_result"] == "TIMEOUT").mean())
    row["same_bar_both_hit_rate"] = float(pd.Series(trades["same_bar_both_hit"]).astype(bool).mean())
    row["avg_fill_bars"] = float(pd.to_numeric(trades["fill_bars_after_signal"], errors="coerce").mean())
    row["avg_bars_held_after_fill"] = float(pd.to_numeric(trades["bars_held_after_fill"], errors="coerce").mean())
    row["positive_years"] = int(positive_years)
    row["year_rows"] = int(year_rows)
    row["candidate_pass"] = bool(
        int(len(trades)) >= int(args.filter_min_count)
        and float(row.get("profit_factor", 0) if pd.notna(row.get("profit_factor", np.nan)) else 0) >= float(args.min_profit_factor)
        and float(row.get("active_days_ratio", 0) if pd.notna(row.get("active_days_ratio", np.nan)) else 0) >= float(args.min_active_days_ratio)
        and int(row.get("max_days_without_trade", 999999) if pd.notna(row.get("max_days_without_trade", np.nan)) else 999999) <= int(args.max_days_without_trade)
        and int(positive_years) >= 3
    )
    return row


def _passive_yearly_rows(trades: pd.DataFrame, spec_meta: dict[str, object], *, setup_count: int) -> list[dict[str, object]]:
    if trades.empty:
        return []
    tmp = trades.copy()
    tmp["year"] = pd.to_datetime(tmp["entry_time"], errors="coerce").dt.year
    rows: list[dict[str, object]] = []
    for year, part in tmp.dropna(subset=["year"]).groupby("year", sort=True):
        stats = _quick_return_stats(part["net_return"], min_count=20)
        rows.append({**spec_meta, "setup_count": int(setup_count), "year": int(year), **stats})
    return rows


def _run_passive_entry_lab_one(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    filters: pd.DataFrame,
    args: argparse.Namespace,
    *,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    specs = _rank_filter_candidates_for_passive(filters, label=label, args=args) if not filters.empty else pd.DataFrame()
    if bool(args.passive_include_curated_setups):
        specs = pd.concat([specs, _curated_passive_specs(label)], ignore_index=True) if not specs.empty else _curated_passive_specs(label)
    if specs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # Deduplicate exact same spec to avoid duplicate expensive grid work.
    dedupe_cols = [c for c in ["label", "event_family", "side_name", "filter_type", "filter_expr", "feature", "op", "threshold", "feature_1", "op_1", "threshold_1", "feature_2", "op_2", "threshold_2"] if c in specs.columns]
    specs = specs.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)

    offsets = _parse_number_list(args.passive_entry_offset_pcts, name="passive_entry_offset_pcts")
    fill_windows = _parse_int_list(args.passive_fill_windows, name="passive_fill_windows")
    targets = _parse_number_list(args.passive_target_pcts, name="passive_target_pcts")
    stops = _parse_number_list(args.passive_stop_pcts, name="passive_stop_pcts")
    horizons = _parse_int_list(args.passive_horizons, name="passive_horizons")
    total = max(1, len(specs) * len(offsets) * len(fill_windows) * len(targets) * len(stops) * len(horizons))
    progress = ProgressReporter(f"[passive] {label} grid", total=total, every=max(1, total // 100), enabled=not bool(args.no_progress))
    done = 0
    summary_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    trade_samples: list[pd.DataFrame] = []
    spec_rows: list[dict[str, object]] = []
    for spec_rank, (_, row) in enumerate(specs.iterrows(), start=1):
        mask = _candidate_mask_for_passive(events, row)
        part = events.loc[mask.fillna(False)].copy()
        setup_count_full = int(len(part))
        part = _cap_passive_setups(part, args)
        setup_count_used = int(len(part))
        side_name = str(row.get("side_name", ""))
        side_value = 1 if side_name == "LONG" else -1 if side_name == "SHORT" else int(pd.to_numeric(part.get("side", pd.Series([0])).iloc[0], errors="coerce") if not part.empty else 0)
        spec_meta_base = {
            "label": label,
            "spec_rank": int(spec_rank),
            "passive_spec_source": row.get("passive_spec_source", label),
            "filter_type": row.get("filter_type", ""),
            "event_family": row.get("event_family", ""),
            "side_name": side_name,
            "filter_expr": row.get("filter_expr", ""),
            "setup_count_full": setup_count_full,
            "setup_count_used": setup_count_used,
            "source_all_count": row.get("all_count", np.nan),
            "source_all_mean": row.get("all_mean", np.nan),
            "source_train_pf": row.get("train_pf_num", row.get("train_profit_factor", np.nan)),
            "source_valid_pf": row.get("valid_pf_num", row.get("valid_profit_factor", np.nan)),
            "source_test_pf": row.get("test_pf_num", row.get("test_profit_factor", np.nan)),
        }
        spec_rows.append(dict(spec_meta_base))
        if part.empty or side_value == 0:
            done += len(offsets) * len(fill_windows) * len(targets) * len(stops) * len(horizons)
            progress.update(done)
            continue
        for offset in offsets:
            for fill_window in fill_windows:
                for target in targets:
                    for stop in stops:
                        for horizon in horizons:
                            grid_meta = {
                                **spec_meta_base,
                                "entry_offset_pct": float(offset),
                                "fill_window": int(fill_window),
                                "target_pct": float(target),
                                "stop_pct": float(stop),
                                "horizon": int(horizon),
                            }
                            trades = _simulate_passive_for_spec_grid(
                                bars,
                                part,
                                side_value=side_value,
                                offset_pct=float(offset),
                                fill_window=int(fill_window),
                                target_pct=float(target),
                                stop_pct=float(stop),
                                horizon=int(horizon),
                                args=args,
                            )
                            summary_rows.append(_passive_summary_row(trades, setup_count=setup_count_used, spec_meta=grid_meta, args=args))
                            yearly_rows.extend(_passive_yearly_rows(trades, grid_meta, setup_count=setup_count_used))
                            if not trades.empty and sum(len(x) for x in trade_samples) < int(args.passive_trade_sample):
                                sample = trades.head(max(0, int(args.passive_trade_sample) - sum(len(x) for x in trade_samples))).copy()
                                for k, v in grid_meta.items():
                                    sample[k] = v
                                trade_samples.append(sample)
                            done += 1
                            progress.update(done)
    progress.close()
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        sort_cols = [c for c in ["candidate_pass", "profit_factor", "mean", "trade_count", "fill_rate"] if c in summary.columns]
        summary = summary.sort_values(sort_cols, ascending=[False, False, False, False, False][: len(sort_cols)])
    yearly = pd.DataFrame(yearly_rows)
    trades_out = pd.concat(trade_samples, ignore_index=True) if trade_samples else pd.DataFrame()
    specs_out = pd.DataFrame(spec_rows)
    return specs_out, summary, yearly, trades_out


def run_passive_entry_lab(
    bars: pd.DataFrame,
    events_enriched: pd.DataFrame,
    filter_gate: pd.DataFrame,
    ema_path_events: pd.DataFrame,
    ema_filter_gate: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if bool(args.no_passive_entry_lab):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    print("[passive] setup + limit-entry lab", flush=True)
    specs_all: list[pd.DataFrame] = []
    summary_all: list[pd.DataFrame] = []
    yearly_all: list[pd.DataFrame] = []
    trades_all: list[pd.DataFrame] = []
    if not events_enriched.empty:
        specs, summary, yearly, trades = _run_passive_entry_lab_one(bars, events_enriched, filter_gate, args, label="mhf_event")
        specs_all.append(specs); summary_all.append(summary); yearly_all.append(yearly); trades_all.append(trades)
    if not ema_path_events.empty:
        specs, summary, yearly, trades = _run_passive_entry_lab_one(bars, ema_path_events, ema_filter_gate, args, label="ema_armed_entry")
        specs_all.append(specs); summary_all.append(summary); yearly_all.append(yearly); trades_all.append(trades)
    specs_out = pd.concat([x for x in specs_all if not x.empty], ignore_index=True) if specs_all else pd.DataFrame()
    summary_out = pd.concat([x for x in summary_all if not x.empty], ignore_index=True) if summary_all else pd.DataFrame()
    yearly_out = pd.concat([x for x in yearly_all if not x.empty], ignore_index=True) if yearly_all else pd.DataFrame()
    trades_out = pd.concat([x for x in trades_all if not x.empty], ignore_index=True) if trades_all else pd.DataFrame()
    if not summary_out.empty:
        sort_cols = [c for c in ["candidate_pass", "profit_factor", "mean", "trade_count", "fill_rate"] if c in summary_out.columns]
        summary_out = summary_out.sort_values(sort_cols, ascending=[False, False, False, False, False][: len(sort_cols)])
    return specs_out, summary_out, yearly_out, trades_out

# ---------------------------------------------------------------------------
# EMA armed crossover strategy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeCost:
    entry_fee_rate: float
    exit_fee_rate: float
    entry_slippage_pct: float
    exit_slippage_pct: float

    @property
    def round_trip_cost_pct(self) -> float:
        return float(self.entry_fee_rate + self.exit_fee_rate + self.entry_slippage_pct + self.exit_slippage_pct)


def simulate_ema_armed_strategy(features: pd.DataFrame, args: argparse.Namespace, *, cost_multiplier: float = 1.0, entry_delay_bars: int = 1) -> pd.DataFrame:
    df = features.sort_index().copy()
    idx = df.index
    open_ = pd.to_numeric(df["open"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    ema_fast = pd.to_numeric(df["ema_fast"], errors="coerce").to_numpy(dtype=float)
    ema_slow = pd.to_numeric(df["ema_slow"], errors="coerce").to_numpy(dtype=float)
    n = len(df)

    fee_entry = float(args.entry_fee_rate) * cost_multiplier
    fee_exit = float(args.exit_fee_rate) * cost_multiplier
    slippage_entry = float(args.entry_slippage_pct) * cost_multiplier
    slippage_exit = float(args.exit_slippage_pct) * cost_multiplier
    rt_cost = fee_entry + fee_exit + slippage_entry + slippage_exit

    entry_buffer = float(args.ema_entry_buffer_pct)
    exit_fast_n = int(args.ema_exit_fast_consecutive)
    max_hold = int(args.ema_max_hold_bars)
    delay = int(entry_delay_bars)

    pos = 0
    armed_long = False
    armed_short = False
    below_fast_count = 0
    above_fast_count = 0
    pending_entry_side = 0
    pending_entry_signal_i = -1
    pending_exit_reason = ""
    pending_exit_signal_i = -1
    entry_i = -1
    entry_signal_i = -1
    entry_price = np.nan
    entry_side = 0
    mae = 0.0
    mfe = 0.0
    trades: list[dict[str, object]] = []

    progress = ProgressReporter("[simulate] ema20_50_armed", total=n, every=max(1, int(args.progress_every)), enabled=not bool(args.no_progress))
    for i in range(1, n):
        # Execute pending orders at the current open. Orders were scheduled from a prior closed signal bar.
        if pending_exit_signal_i >= 0 and i >= pending_exit_signal_i + delay and pos != 0:
            exit_price = open_[i]
            if np.isfinite(exit_price) and exit_price > 0 and np.isfinite(entry_price) and entry_price > 0:
                if entry_side == 1:
                    gross = exit_price / entry_price - 1.0
                else:
                    gross = entry_price / exit_price - 1.0
                net = gross - rt_cost
                trades.append(
                    {
                        "engine": "ema20_50_armed",
                        "side": entry_side,
                        "side_name": _side_name(entry_side),
                        "entry_signal_time": idx[entry_signal_i],
                        "entry_time": idx[entry_i],
                        "entry_price": entry_price,
                        "exit_signal_time": idx[pending_exit_signal_i],
                        "exit_time": idx[i],
                        "exit_price": exit_price,
                        "exit_reason": pending_exit_reason,
                        "bars_held": int(i - entry_i),
                        "gross_return": float(gross),
                        "net_return": float(net),
                        "mfe": float(mfe),
                        "mae": float(mae),
                        "round_trip_cost_pct": float(rt_cost),
                    }
                )
            pos = 0
            entry_side = 0
            entry_i = -1
            entry_signal_i = -1
            entry_price = np.nan
            mae = 0.0
            mfe = 0.0
            below_fast_count = 0
            above_fast_count = 0
            pending_exit_signal_i = -1
            pending_exit_reason = ""

        if pending_entry_side != 0 and i >= pending_entry_signal_i + delay and pos == 0:
            px = open_[i]
            if np.isfinite(px) and px > 0:
                pos = pending_entry_side
                entry_side = pending_entry_side
                entry_i = i
                entry_signal_i = pending_entry_signal_i
                entry_price = px
                mae = 0.0
                mfe = 0.0
                below_fast_count = 0
                above_fast_count = 0
            pending_entry_side = 0
            pending_entry_signal_i = -1

        if pos != 0 and np.isfinite(entry_price) and entry_price > 0:
            if pos == 1:
                mfe = max(mfe, high[i] / entry_price - 1.0 if np.isfinite(high[i]) else mfe)
                mae = min(mae, low[i] / entry_price - 1.0 if np.isfinite(low[i]) else mae)
            else:
                mfe = max(mfe, entry_price / low[i] - 1.0 if np.isfinite(low[i]) and low[i] > 0 else mfe)
                mae = min(mae, entry_price / high[i] - 1.0 if np.isfinite(high[i]) and high[i] > 0 else mae)

        # Signal evaluation on the just-closed current bar i.
        if not all(np.isfinite(x) for x in (close[i], close[i - 1], ema_fast[i], ema_slow[i], ema_slow[i - 1])):
            progress.update(i)
            continue

        cross_up = close[i - 1] <= ema_slow[i - 1] and close[i] > ema_slow[i]
        cross_down = close[i - 1] >= ema_slow[i - 1] and close[i] < ema_slow[i]
        if pos == 0 and pending_entry_side == 0:
            if cross_up:
                armed_long = True
                armed_short = False
            elif cross_down:
                armed_short = True
                armed_long = False
            # Arm stays alive after cross until entry or invalidation back through EMA50.
            if armed_long and close[i] < ema_slow[i]:
                armed_long = False
            if armed_short and close[i] > ema_slow[i]:
                armed_short = False
            if armed_long and close[i] >= ema_slow[i] * (1.0 + entry_buffer):
                pending_entry_side = 1
                pending_entry_signal_i = i
                armed_long = False
            elif armed_short and close[i] <= ema_slow[i] * (1.0 - entry_buffer):
                pending_entry_side = -1
                pending_entry_signal_i = i
                armed_short = False

        if pos == 1 and pending_exit_signal_i < 0:
            below_fast_count = below_fast_count + 1 if close[i] < ema_fast[i] else 0
            reason = ""
            if close[i] < ema_slow[i]:
                reason = "close_below_ema50"
            elif below_fast_count >= exit_fast_n:
                reason = f"{exit_fast_n}_closes_below_ema20"
            elif max_hold > 0 and entry_i >= 0 and i - entry_i >= max_hold:
                reason = "max_hold"
            if reason:
                pending_exit_signal_i = i
                pending_exit_reason = reason
        elif pos == -1 and pending_exit_signal_i < 0:
            above_fast_count = above_fast_count + 1 if close[i] > ema_fast[i] else 0
            reason = ""
            if close[i] > ema_slow[i]:
                reason = "close_above_ema50"
            elif above_fast_count >= exit_fast_n:
                reason = f"{exit_fast_n}_closes_above_ema20"
            elif max_hold > 0 and entry_i >= 0 and i - entry_i >= max_hold:
                reason = "max_hold"
            if reason:
                pending_exit_signal_i = i
                pending_exit_reason = reason
        progress.update(i)
    progress.close()
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _event_summary(events: pd.DataFrame, args: argparse.Namespace, *, return_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if events.empty or return_col not in events.columns:
        return pd.DataFrame()
    for name, part in events.groupby("event_name", sort=False):
        stats = summarize_returns(part[return_col], name=return_col, min_count=int(args.min_count))
        active = _active_days_metrics(part["signal_time"], start_date=args.start_date, end_date=args.end_date)
        row = {
            "event_name": name,
            "event_family": part["event_family"].iloc[0] if "event_family" in part.columns and len(part) else "",
            "side_name": part["side_name"].iloc[0] if "side_name" in part.columns and len(part) else "",
            **stats,
            **active,
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["eligible", "profit_factor", "mean", "count"], ascending=[False, False, False, False])
    return out


def _family_summary(events: pd.DataFrame, args: argparse.Namespace, *, return_col: str) -> pd.DataFrame:
    if events.empty or return_col not in events.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, part in events.groupby(["event_family", "side_name"], sort=False):
        stats = summarize_returns(part[return_col], name=return_col, min_count=int(args.min_count))
        active = _active_days_metrics(part["signal_time"], start_date=args.start_date, end_date=args.end_date)
        rows.append({"event_family": keys[0], "side_name": keys[1], **stats, **active})
    return pd.DataFrame(rows).sort_values(["profit_factor", "mean"], ascending=[False, False])


def _yearly_event_summary(events: pd.DataFrame, *, return_col: str, min_count: int) -> pd.DataFrame:
    if events.empty or return_col not in events.columns:
        return pd.DataFrame()
    frame = events.copy()
    frame["year"] = pd.to_datetime(frame["signal_time"]).dt.year
    return summarize_many(frame, [return_col], group_cols=["event_name", "year"], min_count=min_count)


def _condition_breakdown(events: pd.DataFrame, args: argparse.Namespace, *, return_col: str) -> pd.DataFrame:
    if events.empty or return_col not in events.columns:
        return pd.DataFrame()
    bucket_cols = [
        "volume_ratio",
        "delta_notional_z",
        "large_delta_notional_z",
        "taker_buy_ratio",
        "atr_pct",
        "session_vwap_dist",
    ]
    bucket_cols.extend([c for c in events.columns if c.endswith("fp_delta_ratio") or c.endswith("fp_absorption_proxy") or c.endswith("rb_delta_ratio")])
    rows: list[dict[str, object]] = []
    for event_family, part in events.groupby("event_family", sort=False):
        for col in bucket_cols:
            if col not in part.columns:
                continue
            x = pd.to_numeric(part[col], errors="coerce")
            if x.notna().sum() < max(50, int(args.min_count)) or x.nunique(dropna=True) < 4:
                continue
            try:
                bucket = pd.qcut(x, q=4, duplicates="drop")
            except ValueError:
                continue
            tmp = part.assign(_bucket=bucket.astype(str))
            for b, bp in tmp.groupby("_bucket", dropna=False):
                stats = summarize_returns(bp[return_col], name=return_col, min_count=max(20, int(args.min_count) // 2))
                rows.append({"event_family": event_family, "condition_col": col, "bucket": b, **stats})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["event_family", "condition_col", "bucket"])
    return out


def _run_tp_sl_grid(bars: pd.DataFrame, events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    targets = _parse_number_list(args.touch_target_pcts, name="touch_target_pcts")
    stops = _parse_number_list(args.touch_stop_pcts, name="touch_stop_pcts")
    scope_col = "event_family" if str(getattr(args, "touch_scope", "family")) == "family" else "event_name"
    groups = list(events.groupby(scope_col, sort=False))
    total = len(groups) * len(targets) * len(stops)
    progress = ProgressReporter("[touch] event TP/SL grid", total=total, every=max(1, total // 50), enabled=not bool(args.no_progress))
    rows: list[dict[str, object]] = []
    done = 0
    frame = bars.sort_index()
    for group_name, part in groups:
        pos = _coerce_event_positions(frame, part)
        side = pd.to_numeric(part["side"], errors="coerce").fillna(0).astype(int).to_numpy()
        ok = pos >= 0
        pos = pos[ok]
        side = side[ok]
        if len(pos) == 0:
            done += len(targets) * len(stops)
            progress.update(done)
            continue
        meta = {
            "scope": scope_col,
            "name": str(group_name),
            "event_family": part["event_family"].iloc[0] if "event_family" in part.columns and len(part) else str(group_name),
            "side_name": part["side_name"].iloc[0] if "side_name" in part.columns and len(part) else "MIXED",
        }
        for target in targets:
            for stop in stops:
                result, touch_bars, both = _first_touch_outcome_sparse(
                    frame,
                    pos,
                    side,
                    target_pct=float(target),
                    stop_pct=float(stop),
                    horizon=int(args.touch_horizon),
                    entry_delay_bars=1,
                )
                count = int(len(result))
                if count:
                    target_rate = float((result == "TARGET").mean())
                    stop_rate = float((result == "STOP").mean())
                    timeout_rate = float((result == "TIMEOUT").mean())
                    both_rate = float(both.mean())
                    avg_touch_bars = float(np.nanmean(touch_bars)) if np.isfinite(touch_bars).any() else np.nan
                    expectancy_proxy = target_rate * float(target) - stop_rate * float(stop) - float(_cost_from_args(args).round_trip_cost_pct)
                else:
                    target_rate = stop_rate = timeout_rate = both_rate = avg_touch_bars = expectancy_proxy = np.nan
                rows.append(
                    {
                        **meta,
                        "target_pct": float(target),
                        "stop_pct": float(stop),
                        "count": count,
                        "target_rate": target_rate,
                        "stop_rate": stop_rate,
                        "timeout_rate": timeout_rate,
                        "same_bar_both_hit_rate": both_rate,
                        "avg_touch_bars": avg_touch_bars,
                        "expectancy_proxy_net": expectancy_proxy,
                    }
                )
                done += 1
                progress.update(done)
    progress.close()
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["expectancy_proxy_net", "target_rate", "same_bar_both_hit_rate"], ascending=[False, False, True])
    return out

def _run_event_stress(bars: pd.DataFrame, events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    cost_multipliers = _parse_number_list(args.stress_cost_multipliers, name="stress_cost_multipliers")
    delays = _parse_int_list(args.stress_delay_bars, name="stress_delay_bars")
    candidate_h = int(args.candidate_horizon)
    scope_col = "event_name" if str(getattr(args, "stress_scope", "family")) == "event" else "event_family"
    total = len(cost_multipliers) * len(delays)
    progress = ProgressReporter("[stress] event fast", total=total, every=1, enabled=not bool(args.no_progress))
    done = 0
    for cost_mult in cost_multipliers:
        cost = _cost_from_args(args, multiplier=float(cost_mult))
        for delay in delays:
            col = f"stress_ret_h{candidate_h}_d{int(delay)}_c{float(cost_mult):g}"
            tmp = events[[scope_col]].copy()
            tmp[col] = _fast_net_return_for_events(
                bars,
                events,
                horizon=candidate_h,
                delay=int(delay),
                cost=cost,
            )
            for name, part in tmp.groupby(scope_col, sort=False):
                stats = summarize_returns(part[col], name=col, min_count=int(args.min_count))
                rows.append({"scope": scope_col, "name": name, "cost_mult": float(cost_mult), "delay_bars": int(delay), **stats})
            done += 1
            progress.update(done)
    progress.close()
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["name", "cost_mult", "delay_bars"])
    return out

def _candidate_gate(event_summary: pd.DataFrame, yearly: pd.DataFrame, args: argparse.Namespace, *, return_col: str) -> pd.DataFrame:
    if event_summary.empty:
        return pd.DataFrame()
    yearly_positive = pd.DataFrame()
    if not yearly.empty and "event_name" in yearly.columns:
        y = yearly.copy()
        y["mean_num"] = pd.to_numeric(y["mean"], errors="coerce")
        y["positive_year"] = y["mean_num"] > 0
        yearly_positive = y.groupby("event_name", as_index=False).agg(
            positive_years=("positive_year", "sum"),
            yearly_rows=("year", "count"),
        )
    out = event_summary.copy()
    if not yearly_positive.empty:
        out = out.merge(yearly_positive, on="event_name", how="left")
    else:
        out["positive_years"] = 0
        out["yearly_rows"] = 0

    out["pf_num"] = pd.to_numeric(out["profit_factor"].replace("inf", np.inf), errors="coerce")
    out["win_rate_num"] = pd.to_numeric(out["win_rate"], errors="coerce")
    out["top5_num"] = pd.to_numeric(out.get("top5_winner_share", np.nan), errors="coerce")
    out["gate_count"] = pd.to_numeric(out["count"], errors="coerce") >= int(args.min_count)
    out["gate_pf"] = out["pf_num"] >= float(args.min_profit_factor)
    out["gate_win_rate"] = out["win_rate_num"] >= float(args.min_win_rate)
    out["gate_active_days"] = pd.to_numeric(out["active_days_ratio"], errors="coerce") >= float(args.min_active_days_ratio)
    out["gate_no_trade_gap"] = pd.to_numeric(out["max_days_without_trade"], errors="coerce") <= int(args.max_days_without_trade)
    out["gate_top_winner"] = out["top5_num"].isna() | (out["top5_num"] <= float(args.max_top5_winner_share))
    out["gate_positive_years"] = pd.to_numeric(out["positive_years"], errors="coerce").fillna(0) >= 3
    gates = [c for c in out.columns if c.startswith("gate_")]
    out["candidate_pass"] = out[gates].all(axis=1)
    return out.sort_values(["candidate_pass", "pf_num", "mean"], ascending=[False, False, False])


def _ema_reports(features: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cost_multipliers = _parse_number_list(args.stress_cost_multipliers, name="stress_cost_multipliers")
    delays = _parse_int_list(args.stress_delay_bars, name="stress_delay_bars")
    trades_all: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    for cm in cost_multipliers:
        for delay in delays:
            trades = simulate_ema_armed_strategy(features, args, cost_multiplier=float(cm), entry_delay_bars=int(delay))
            if not trades.empty:
                trades["cost_mult"] = float(cm)
                trades["delay_bars"] = int(delay)
                trades_all.append(trades)
                ret = pd.to_numeric(trades["net_return"], errors="coerce")
                active = _active_days_metrics(trades["entry_signal_time"], start_date=args.start_date, end_date=args.end_date)
                row = {
                    "engine": "ema20_50_armed",
                    "cost_mult": float(cm),
                    "delay_bars": int(delay),
                    "trades": int(len(trades)),
                    **_summarize_equity(ret),
                    **active,
                    "avg_bars_held": float(pd.to_numeric(trades["bars_held"], errors="coerce").mean()),
                    "median_bars_held": float(pd.to_numeric(trades["bars_held"], errors="coerce").median()),
                    "avg_mfe": float(pd.to_numeric(trades["mfe"], errors="coerce").mean()),
                    "avg_mae": float(pd.to_numeric(trades["mae"], errors="coerce").mean()),
                }
                summary_rows.append(row)
                tmp = trades.copy()
                tmp["year"] = pd.to_datetime(tmp["entry_signal_time"]).dt.year
                for year, part in tmp.groupby("year", sort=True):
                    yearly_rows.append(
                        {
                            "engine": "ema20_50_armed",
                            "cost_mult": float(cm),
                            "delay_bars": int(delay),
                            "year": int(year),
                            "trades": int(len(part)),
                            **_summarize_equity(part["net_return"]),
                        }
                    )
            else:
                summary_rows.append(
                    {
                        "engine": "ema20_50_armed",
                        "cost_mult": float(cm),
                        "delay_bars": int(delay),
                        "trades": 0,
                    }
                )
    trades_df = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
    return pd.DataFrame(summary_rows), trades_df, pd.DataFrame(yearly_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# V5 memory-safe filter mining overrides
# ---------------------------------------------------------------------------
# The first v4 implementation evaluated every filter against the full mining
# DataFrame with pandas boolean Series and DataFrame.loc copies. On 400k+ MHF
# events and ~900 scans this can be killed by Windows without a traceback.
# These overrides keep the same report schema but evaluate train-first filters
# group-locally with numpy arrays, and prepare a downcast/minimal mining frame.


def _as_float_array_fast(values: Any) -> np.ndarray:
    out = pd.to_numeric(values, errors="coerce")
    if hasattr(out, "to_numpy"):
        return out.to_numpy(dtype=np.float64, copy=False)
    return np.asarray(out, dtype=np.float64)


def _quick_return_stats_array(values: np.ndarray, mask: np.ndarray | None = None, *, min_count: int = 1) -> dict[str, object]:
    if mask is not None:
        y = values[mask]
    else:
        y = values
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    count = int(y.size)
    if count == 0:
        return {
            "count": 0,
            "eligible": False,
            "mean": np.nan,
            "median": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_win": np.nan,
            "max_loss": np.nan,
        }
    wins = y[y > 0]
    losses = y[y < 0]
    gross_profit = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(-losses.sum()) if losses.size else 0.0
    pf = (np.inf if gross_profit > 0 else np.nan) if gross_loss == 0.0 else gross_profit / gross_loss
    return {
        "count": count,
        "eligible": bool(count >= int(min_count)),
        "mean": float(y.mean()),
        "median": float(np.median(y)),
        "win_rate": float((y > 0).mean()),
        "profit_factor": float(pf) if np.isfinite(pf) else np.inf,
        "max_win": float(y.max()),
        "max_loss": float(y.min()),
    }


def _split_codes_from_signal_time(signal_time: pd.Series, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    ts = pd.to_datetime(signal_time, errors="coerce")
    train_end = pd.Timestamp(args.filter_train_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    valid_end = pd.Timestamp(args.filter_valid_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    codes = np.full(len(ts), -1, dtype=np.int8)
    ok = ts.notna().to_numpy()
    train = (ts <= train_end).to_numpy() & ok
    valid = (ts > train_end).to_numpy() & (ts <= valid_end).to_numpy() & ok
    test = (ts > valid_end).to_numpy() & ok
    codes[train] = 0
    codes[valid] = 1
    codes[test] = 2
    years = ts.dt.year.fillna(0).astype(np.int16).to_numpy()
    return codes, years


def _eval_filter_arrays(
    y: np.ndarray,
    cond: np.ndarray,
    split_codes: np.ndarray,
    years: np.ndarray,
    *,
    min_count: int,
    min_split_count: int,
) -> dict[str, object]:
    cond = np.asarray(cond, dtype=bool)
    out: dict[str, object] = {}
    split_defs = {
        "all": cond & (split_codes >= 0),
        "train": cond & (split_codes == 0),
        "valid": cond & (split_codes == 1),
        "test": cond & (split_codes == 2),
    }
    for name, m in split_defs.items():
        mc = int(min_count if name in {"all", "train"} else min_split_count)
        out.update(_prefix_stats(_quick_return_stats_array(y, m, min_count=mc), name))
    positive_years = 0
    year_rows = 0
    for yy in np.unique(years[(cond & (years > 0))]):
        m = cond & (years == yy)
        if int(m.sum()) < max(20, int(min_split_count) // 2):
            continue
        st = _quick_return_stats_array(y, m, min_count=max(20, int(min_split_count) // 2))
        if pd.notna(st.get("mean", np.nan)):
            year_rows += 1
            positive_years += int(float(st.get("mean", np.nan)) > 0)
    out["positive_years"] = int(positive_years)
    out["year_rows"] = int(year_rows)
    return out


def _prepare_events_for_filter_mining(events: pd.DataFrame, args: argparse.Namespace, *, label: str) -> pd.DataFrame:
    if events.empty:
        return events
    before = len(events)
    feature_cols = _candidate_feature_columns(events)
    return_cols = [c for c in events.columns if c.startswith("next_open_ret_h") and c.endswith("_net")]
    meta_cols = [
        "event_family",
        "side_name",
        "event_name",
        "signal_time",
        "side",
        "signal_bar_pos",
        "entry_bar_pos",
    ]
    keep_cols = [c for c in meta_cols + return_cols + feature_cols if c in events.columns]
    out = events.loc[:, keep_cols].copy()
    if not bool(getattr(args, "no_filter_dedupe", False)) and {"event_family", "side_name", "signal_time"}.issubset(out.columns):
        sort_cols = [c for c in ["event_family", "side_name", "signal_time", "event_name"] if c in out.columns]
        out = out.sort_values(sort_cols, kind="mergesort")
        out = out.drop_duplicates(["event_family", "side_name", "signal_time"], keep="last").reset_index(drop=True)
    cap = int(getattr(args, "filter_max_rows_per_group", 0) or 0)
    if cap > 0 and {"event_family", "side_name"}.issubset(out.columns):
        parts: list[pd.DataFrame] = []
        for _, part in out.groupby(["event_family", "side_name"], sort=False):
            if len(part) > cap:
                step = max(1, int(math.ceil(len(part) / cap)))
                parts.append(part.iloc[::step].copy())
            else:
                parts.append(part)
        out = pd.concat(parts, ignore_index=True) if parts else out.iloc[0:0].copy()
    # Downcast only the mining copy. The original enriched events remain unchanged
    # for reports and TP/SL replay.
    for col in feature_cols + return_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    if "side" in out.columns:
        out["side"] = pd.to_numeric(out["side"], errors="coerce").fillna(0).astype("int8")
    for col in ("event_family", "side_name", "event_name"):
        if col in out.columns:
            out[col] = out[col].astype("category")
    # The loop above intentionally mutates many numeric columns to downcast the
    # mining copy. On pandas this can leave the frame highly fragmented; adding
    # split/year columns afterwards then emits PerformanceWarning and slows the
    # next group scans. Consolidate once before appending helper columns.
    out = out.copy()
    codes, years = _split_codes_from_signal_time(out["signal_time"], args)
    split_cols = pd.DataFrame(
        {
            "_filter_split": np.asarray(codes, dtype=np.int8),
            "_filter_year": np.asarray(years, dtype=np.int16),
        },
        index=out.index,
    )
    out = pd.concat([out, split_cols], axis=1, copy=False)
    after = len(out)
    if after != before or len(keep_cols) != len(events.columns):
        print(
            f"[filter] {label} mining rows {before:,} -> {after:,}; cols {len(events.columns):,} -> {len(out.columns):,} after dedupe/cap/minimal/downcast",
            flush=True,
        )
    return out


def _winner_loser_feature_diff(events: pd.DataFrame, args: argparse.Namespace, *, return_col: str, label: str) -> pd.DataFrame:
    if events.empty or return_col not in events.columns:
        return pd.DataFrame()
    cols = _candidate_feature_columns(events)
    rows: list[dict[str, object]] = []
    min_side = max(30, int(args.filter_min_count) // 5)
    y_all = _as_float_array_fast(events[return_col])
    groups = events.groupby(["event_family", "side_name"], sort=False).indices
    for (family, side_name), idx in groups.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        y = y_all[idx_arr]
        finite_y = np.isfinite(y)
        win_mask = finite_y & (y > 0)
        loss_mask = finite_y & (y <= 0)
        if int(win_mask.sum()) < min_side or int(loss_mask.sum()) < min_side:
            continue
        for col in cols:
            x = _as_float_array_fast(events[col].iloc[idx_arr])
            xw = x[win_mask]
            xl = x[loss_mask]
            xw = xw[np.isfinite(xw)]
            xl = xl[np.isfinite(xl)]
            if xw.size < min_side or xl.size < min_side:
                continue
            pooled_arr = np.concatenate([xw, xl])
            pooled = float(np.nanstd(pooled_arr))
            mean_win = float(np.nanmean(xw))
            mean_loss = float(np.nanmean(xl))
            effect = (mean_win - mean_loss) / pooled if pooled and math.isfinite(pooled) else np.nan
            rows.append(
                {
                    "label": label,
                    "event_family": family,
                    "side_name": side_name,
                    "feature": col,
                    "winner_count": int(xw.size),
                    "loser_count": int(xl.size),
                    "winner_mean": mean_win,
                    "loser_mean": mean_loss,
                    "winner_median": float(np.nanmedian(xw)),
                    "loser_median": float(np.nanmedian(xl)),
                    "mean_diff": mean_win - mean_loss,
                    "effect_z": float(effect) if pd.notna(effect) else np.nan,
                    "abs_effect_z": float(abs(effect)) if pd.notna(effect) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["abs_effect_z", "winner_count"], ascending=[False, False])
    return out


def _build_univariate_filter_candidates(events: pd.DataFrame, args: argparse.Namespace, *, return_col: str, label: str) -> pd.DataFrame:
    if events.empty or return_col not in events.columns:
        return pd.DataFrame()
    cols = _candidate_feature_columns(events)
    quantiles = [q for q in _parse_number_list(args.filter_quantiles, name="filter_quantiles") if 0.0 < float(q) < 1.0]
    min_count = int(args.filter_min_count)
    min_split = int(args.filter_min_split_count)
    rows: list[dict[str, object]] = []
    y_all = _as_float_array_fast(events[return_col])
    split_all = events.get("_filter_split")
    year_all = events.get("_filter_year")
    if split_all is None or year_all is None:
        split_codes, years = _split_codes_from_signal_time(events["signal_time"], args)
    else:
        split_codes = split_all.to_numpy(dtype=np.int8, copy=False)
        years = year_all.to_numpy(dtype=np.int16, copy=False)
    groups = events.groupby(["event_family", "side_name"], sort=False).indices
    progress = ProgressReporter("[filter] univariate train", total=max(1, len(groups) * len(cols)), every=max(1, len(cols)), enabled=not bool(args.no_progress))
    done = 0
    for (family, side_name), idx in groups.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        y = y_all[idx_arr]
        sp = split_codes[idx_arr]
        yr = years[idx_arr]
        train_base = sp == 0
        if int(train_base.sum()) < min_count:
            done += len(cols)
            progress.update(done)
            continue
        base_train = _quick_return_stats_array(y, train_base, min_count=min_count)
        for col in cols:
            x = _as_float_array_fast(events[col].iloc[idx_arr])
            x_train = x[train_base & np.isfinite(x)]
            if x_train.size < min_count:
                done += 1
                progress.update(done)
                continue
            # avoid expensive nunique on pandas; a small sample is enough to skip flat cols
            if np.unique(x_train[: min(x_train.size, 5000)]).size < 4 and np.unique(x_train).size < 4:
                done += 1
                progress.update(done)
                continue
            thresholds = sorted(set(float(v) for v in np.nanquantile(x_train, quantiles) if np.isfinite(v)))
            for thr in thresholds:
                ge = x >= float(thr)
                le = x <= float(thr)
                for op, cond in ((">=", ge), ("<=", le)):
                    if int((cond & train_base).sum()) < min_count:
                        continue
                    eval_stats = _eval_filter_arrays(y, cond, sp, yr, min_count=min_count, min_split_count=min_split)
                    rows.append(
                        {
                            "label": label,
                            "filter_type": "univariate",
                            "event_family": family,
                            "side_name": side_name,
                            "feature": col,
                            "op": op,
                            "threshold": float(thr),
                            "filter_expr": f"{col} {op} {thr:.12g}",
                            "base_train_count": base_train.get("count", np.nan),
                            "base_train_mean": base_train.get("mean", np.nan),
                            "base_train_profit_factor": base_train.get("profit_factor", np.nan),
                            **eval_stats,
                        }
                    )
            done += 1
            progress.update(done)
    progress.close()
    out = pd.DataFrame(rows)
    if not out.empty:
        out["train_pf_num"] = pd.to_numeric(out["train_profit_factor"].replace("inf", np.inf), errors="coerce")
        out["valid_pf_num"] = pd.to_numeric(out["valid_profit_factor"].replace("inf", np.inf), errors="coerce")
        out["test_pf_num"] = pd.to_numeric(out["test_profit_factor"].replace("inf", np.inf), errors="coerce")
        out["score"] = out["train_pf_num"].clip(upper=5).fillna(0) * np.log1p(pd.to_numeric(out["train_count"], errors="coerce").fillna(0))
        out = out.sort_values(["test_pf_num", "valid_pf_num", "train_pf_num", "train_count"], ascending=[False, False, False, False])
        if int(args.filter_max_candidates) > 0 and len(out) > int(args.filter_max_candidates):
            out = out.head(int(args.filter_max_candidates)).copy()
    return out


def _build_combo_filter_candidates(events: pd.DataFrame, uni: pd.DataFrame, args: argparse.Namespace, *, return_col: str, label: str) -> pd.DataFrame:
    if events.empty or uni.empty:
        return pd.DataFrame()
    min_count = int(args.filter_min_count)
    min_split = int(args.filter_min_split_count)
    max_total = int(args.filter_max_combos)
    y_all = _as_float_array_fast(events[return_col])
    split_codes = events["_filter_split"].to_numpy(dtype=np.int8, copy=False) if "_filter_split" in events.columns else _split_codes_from_signal_time(events["signal_time"], args)[0]
    years = events["_filter_year"].to_numpy(dtype=np.int16, copy=False) if "_filter_year" in events.columns else _split_codes_from_signal_time(events["signal_time"], args)[1]
    group_indices = events.groupby(["event_family", "side_name"], sort=False).indices
    rows: list[dict[str, object]] = []
    groups = list(uni.groupby(["event_family", "side_name"], sort=False))
    progress = ProgressReporter("[filter] combo2 train", total=max(1, len(groups)), every=1, enabled=not bool(args.no_progress))
    done = 0
    for (family, side_name), upart in groups:
        idx_arr = np.asarray(group_indices.get((family, side_name), []), dtype=np.int64)
        if idx_arr.size == 0:
            done += 1
            progress.update(done)
            continue
        up = upart.sort_values(["train_pf_num", "valid_pf_num", "train_count"], ascending=[False, False, False]).head(int(args.filter_top_per_group)).copy()
        records = up.to_dict("records")
        if len(records) < 2:
            done += 1
            progress.update(done)
            continue
        y = y_all[idx_arr]
        sp = split_codes[idx_arr]
        yr = years[idx_arr]
        train_base = sp == 0
        base_train = _quick_return_stats_array(y, train_base, min_count=min_count)
        pre_masks: list[np.ndarray] = []
        for r in records:
            x = _as_float_array_fast(events[str(r["feature"])].iloc[idx_arr])
            if str(r["op"]) == ">=":
                pre_masks.append(x >= float(r["threshold"]))
            else:
                pre_masks.append(x <= float(r["threshold"]))
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                if len(rows) >= max_total:
                    break
                r1 = records[i]
                r2 = records[j]
                if str(r1["feature"]) == str(r2["feature"]):
                    continue
                cond = pre_masks[i] & pre_masks[j]
                if int((cond & train_base).sum()) < min_count:
                    continue
                eval_stats = _eval_filter_arrays(y, cond, sp, yr, min_count=min_count, min_split_count=min_split)
                rows.append(
                    {
                        "label": label,
                        "filter_type": "combo2",
                        "event_family": family,
                        "side_name": side_name,
                        "feature_1": str(r1["feature"]),
                        "op_1": str(r1["op"]),
                        "threshold_1": float(r1["threshold"]),
                        "feature_2": str(r2["feature"]),
                        "op_2": str(r2["op"]),
                        "threshold_2": float(r2["threshold"]),
                        "filter_expr": f"{r1['feature']} {r1['op']} {float(r1['threshold']):.12g} AND {r2['feature']} {r2['op']} {float(r2['threshold']):.12g}",
                        "base_train_count": base_train.get("count", np.nan),
                        "base_train_mean": base_train.get("mean", np.nan),
                        "base_train_profit_factor": base_train.get("profit_factor", np.nan),
                        **eval_stats,
                    }
                )
            if len(rows) >= max_total:
                break
        done += 1
        progress.update(done)
        if len(rows) >= max_total:
            break
    progress.close()
    out = pd.DataFrame(rows)
    if not out.empty:
        out["train_pf_num"] = pd.to_numeric(out["train_profit_factor"].replace("inf", np.inf), errors="coerce")
        out["valid_pf_num"] = pd.to_numeric(out["valid_profit_factor"].replace("inf", np.inf), errors="coerce")
        out["test_pf_num"] = pd.to_numeric(out["test_profit_factor"].replace("inf", np.inf), errors="coerce")
        out = out.sort_values(["test_pf_num", "valid_pf_num", "train_pf_num", "train_count"], ascending=[False, False, False, False])
    return out



def _filter_expr_cond_local(group: pd.DataFrame, row: pd.Series) -> np.ndarray:
    if str(row.get("filter_type", "")) == "combo2":
        x1 = _as_float_array_fast(group[str(row["feature_1"])])
        x2 = _as_float_array_fast(group[str(row["feature_2"])])
        m1 = x1 >= float(row["threshold_1"]) if str(row["op_1"]) == ">=" else x1 <= float(row["threshold_1"])
        m2 = x2 >= float(row["threshold_2"]) if str(row["op_2"]) == ">=" else x2 <= float(row["threshold_2"])
        return m1 & m2
    x = _as_float_array_fast(group[str(row["feature"])])
    return x >= float(row["threshold"]) if str(row["op"]) == ">=" else x <= float(row["threshold"])


def _attach_filter_active_metrics(events: pd.DataFrame, filters: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if events.empty or filters.empty:
        return filters
    out = filters.copy()
    active_rows: list[dict[str, object]] = []
    group_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for _, row in out.iterrows():
        key = (str(row.get("event_family", "")), str(row.get("side_name", "")))
        group = group_cache.get(key)
        if group is None:
            group = events[(events["event_family"].astype(str) == key[0]) & (events["side_name"].astype(str) == key[1])]
            group_cache[key] = group
        if group.empty:
            active_rows.append({"all_active_days": 0, "all_total_days": 0, "all_active_days_ratio": np.nan, "all_max_days_without_trade": np.nan})
            continue
        cond = _filter_expr_cond_local(group, row)
        times = group.loc[cond, "signal_time"]
        active = _active_days_metrics(times, start_date=args.start_date, end_date=args.end_date) if len(times) else {
            "active_days": 0,
            "total_days": 0,
            "active_days_ratio": np.nan,
            "max_days_without_trade": np.nan,
        }
        active_rows.append({f"all_{k}": v for k, v in active.items()})
    active_df = pd.DataFrame(active_rows, index=out.index)
    for col in active_df.columns:
        out[col] = active_df[col]
    return out


def run_filter_mining(events: pd.DataFrame, args: argparse.Namespace, *, return_col: str, label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Memory-safe winner/loser separation plus train-first filter mining."""
    if events.empty or return_col not in events.columns:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty
    events = _prepare_events_for_filter_mining(events, args, label=label)
    if events.empty or return_col not in events.columns:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty
    import gc
    print(f"[filter] {label} winner/loser diff", flush=True)
    diff = _winner_loser_feature_diff(events, args, return_col=return_col, label=label)
    gc.collect()
    print(f"[filter] {label} univariate", flush=True)
    uni = _build_univariate_filter_candidates(events, args, return_col=return_col, label=label)
    gc.collect()
    print(f"[filter] {label} combo2", flush=True)
    combo = _build_combo_filter_candidates(events, uni, args, return_col=return_col, label=label)
    gc.collect()
    combined = pd.concat([x for x in [uni, combo] if not x.empty], ignore_index=True) if (not uni.empty or not combo.empty) else pd.DataFrame()
    if not combined.empty:
        combined = combined.head(int(args.filter_max_candidates)).copy()
        combined = _attach_filter_active_metrics(events, combined, args)
    gate = _filter_candidate_gate(combined, args)
    stability = _filter_stability_long(events, gate, args, return_col=return_col)
    gc.collect()
    return diff, uni, combo, gate, stability


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = _to_path(args.out_dir)
    print("[run] eth_mhf_smooth_curve_event_lab", flush=True)
    print(f"[args] out_dir={out_dir}", flush=True)

    bars_all = load_trade_bars(args)
    features_all = build_tradebar_features(bars_all, args)
    features_all, context_available_cols = attach_range_and_footprint_context(features_all, args)

    start_ts = pd.Timestamp(args.start_date)
    end_ts = pd.Timestamp(args.end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    bars = bars_all.loc[(bars_all.index >= start_ts) & (bars_all.index <= end_ts)].copy()
    features = features_all.loc[(features_all.index >= start_ts) & (features_all.index <= end_ts)].copy()
    if bars.empty or features.empty:
        raise RuntimeError(f"No data after slicing to {args.start_date}->{args.end_date}")
    print(f"[slice] bars={len(bars):,} features={len(features):,} range={features.index[0]}->{features.index[-1]}", flush=True)

    events_raw = build_all_events(features, args, context_available_cols)
    wl_diff = pd.DataFrame()
    uni_filters = pd.DataFrame()
    combo_filters = pd.DataFrame()
    filter_gate = pd.DataFrame()
    filter_stability = pd.DataFrame()
    filtered_touch = pd.DataFrame()
    if events_raw.empty:
        print("[warn] no MHF events generated; EMA strategy will still run", flush=True)
        event_result = None
        events_enriched = pd.DataFrame()
        event_summary = pd.DataFrame()
        family_summary = pd.DataFrame()
        yearly_summary = pd.DataFrame()
        condition_summary = pd.DataFrame()
        touch_summary = pd.DataFrame()
        stress_summary = pd.DataFrame()
        candidate_gate = pd.DataFrame()
    else:
        horizons = tuple(_parse_int_list(args.horizons, name="horizons"))
        cfg = EventStudyConfig(
            horizons=horizons,
            mfe_mae_horizon=int(args.mfe_mae_horizon),
            entry_delay_bars=1,
            cost=_cost_from_args(args),
            context_available_time_cols=tuple(c for c in events_raw.columns if c.endswith("_available_time")),
            min_count=int(args.min_count),
            progress_every=int(args.progress_every) if not bool(args.no_progress) else 0,
        )
        print("[event-study] base labels" + (" fast" if bool(args.fast_mode) else " slow"), flush=True)
        event_result = run_event_study_fast(bars, events_raw, cfg) if bool(args.fast_mode) else run_event_study(bars, events_raw, cfg)
        events_enriched = event_result.events
        candidate_h = int(args.candidate_horizon)
        return_col = f"next_open_ret_h{candidate_h}_net"
        if return_col not in events_enriched.columns:
            # Fallback to the largest configured horizon if candidate_h was not in --horizons.
            candidate_h = max(horizons)
            return_col = f"next_open_ret_h{candidate_h}_net"
        print(f"[summary] event return_col={return_col}", flush=True)
        event_summary = _event_summary(events_enriched, args, return_col=return_col)
        family_summary = _family_summary(events_enriched, args, return_col=return_col)
        yearly_summary = _yearly_event_summary(events_enriched, return_col=return_col, min_count=max(20, int(args.min_count) // 2))
        condition_summary = _condition_breakdown(events_enriched, args, return_col=return_col)
        candidate_gate = _candidate_gate(event_summary, yearly_summary, args, return_col=return_col)
        touch_summary = _run_tp_sl_grid(bars, events_enriched, args)
        stress_summary = _run_event_stress(bars, events_raw, args)
        if not bool(args.no_filter_mining):
            wl_diff, uni_filters, combo_filters, filter_gate, filter_stability = run_filter_mining(
                events_enriched, args, return_col=return_col, label="mhf_event"
            )
            filtered_touch = _run_filtered_tp_sl_grid(bars, events_enriched, filter_gate, args, label="mhf_event", use_ema_grid=False)

    print("[ema] simulate EMA20/50 armed crossover strategy", flush=True)
    ema_summary, ema_trades, ema_yearly = _ema_reports(features, args)

    print("[ema-path] armed entry path study", flush=True)
    ema_path_summary = pd.DataFrame()
    ema_path_events = pd.DataFrame()
    ema_path_touch = pd.DataFrame()
    ema_wl_diff = pd.DataFrame()
    ema_uni_filters = pd.DataFrame()
    ema_combo_filters = pd.DataFrame()
    ema_filter_gate = pd.DataFrame()
    ema_filter_stability = pd.DataFrame()
    ema_filtered_touch = pd.DataFrame()
    ema_extra_cols = sorted(set(_candidate_feature_columns(features) + ["open", "high", "low", "close", "volume"] + list(context_available_cols)))
    ema_entry_raw = build_ema_armed_entry_events(features, args, ema_extra_cols)
    if not ema_entry_raw.empty:
        ema_path_summary, ema_path_events = _ema_path_summary(bars, ema_entry_raw, args)
        ema_path_touch = _run_ema_tp_sl_grid(bars, ema_path_events, args)
        ema_horizons = _parse_int_list(args.ema_path_horizons, name="ema_path_horizons")
        ema_candidate_h = 48 if 48 in ema_horizons else max(ema_horizons)
        ema_return_col = f"next_open_ret_h{ema_candidate_h}_net"
        if not bool(args.no_filter_mining) and ema_return_col in ema_path_events.columns:
            ema_wl_diff, ema_uni_filters, ema_combo_filters, ema_filter_gate, ema_filter_stability = run_filter_mining(
                ema_path_events, args, return_col=ema_return_col, label="ema_armed_entry"
            )
            ema_filtered_touch = _run_filtered_tp_sl_grid(bars, ema_path_events, ema_filter_gate, args, label="ema_armed_entry", use_ema_grid=True)

    passive_specs = pd.DataFrame()
    passive_summary = pd.DataFrame()
    passive_yearly = pd.DataFrame()
    passive_trades = pd.DataFrame()
    if not bool(args.no_passive_entry_lab):
        passive_specs, passive_summary, passive_yearly, passive_trades = run_passive_entry_lab(
            bars,
            events_enriched,
            filter_gate,
            ema_path_events,
            ema_filter_gate,
            args,
        )

    print("[write] reports", flush=True)
    # Core event outputs.
    if not events_enriched.empty:
        event_summary.to_csv(out_dir / "01_event_summary.csv", index=False)
        family_summary.to_csv(out_dir / "02_event_family_summary.csv", index=False)
        yearly_summary.to_csv(out_dir / "03_event_yearly.csv", index=False)
        condition_summary.to_csv(out_dir / "04_micro_condition_breakdown.csv", index=False)
        touch_summary.to_csv(out_dir / "05_tp_sl_first_touch_grid.csv", index=False)
        stress_summary.to_csv(out_dir / "06_event_stress_summary.csv", index=False)
        candidate_gate.to_csv(out_dir / "07_candidate_gate.csv", index=False)
        sample_n = int(args.save_event_sample)
        if sample_n > 0:
            events_enriched.head(sample_n).to_csv(out_dir / "08_events_enriched_sample.csv", index=False)
        if event_result is not None:
            event_result.causal_audit.to_csv(out_dir / "09_causal_audit.csv", index=False)
        wl_diff.to_csv(out_dir / "31_winner_loser_feature_diff.csv", index=False)
        uni_filters.to_csv(out_dir / "32_univariate_filter_scan.csv", index=False)
        combo_filters.to_csv(out_dir / "33_combo_filter_scan.csv", index=False)
        filter_gate.to_csv(out_dir / "34_filter_candidate_gate.csv", index=False)
        filtered_touch.to_csv(out_dir / "35_filtered_tp_sl_grid.csv", index=False)
        filter_stability.to_csv(out_dir / "36_filter_train_test_stability.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "01_event_summary.csv", index=False)

    # EMA outputs.
    ema_summary.to_csv(out_dir / "20_ema20_50_armed_summary.csv", index=False)
    ema_yearly.to_csv(out_dir / "21_ema20_50_armed_yearly.csv", index=False)
    if not ema_trades.empty and int(args.save_trade_sample) > 0:
        ema_trades.head(int(args.save_trade_sample)).to_csv(out_dir / "22_ema20_50_armed_trades_sample.csv", index=False)

    # EMA armed-entry path-study outputs.
    ema_path_summary.to_csv(out_dir / "23_ema20_50_armed_entry_path_summary.csv", index=False)
    ema_path_touch.to_csv(out_dir / "24_ema20_50_armed_entry_tp_sl_grid.csv", index=False)
    if not ema_path_events.empty and int(args.save_event_sample) > 0:
        ema_path_events.head(int(args.save_event_sample)).to_csv(out_dir / "25_ema20_50_armed_entry_events_sample.csv", index=False)
    ema_wl_diff.to_csv(out_dir / "26_ema20_50_armed_entry_winner_loser_diff.csv", index=False)
    ema_uni_filters.to_csv(out_dir / "27_ema20_50_armed_entry_univariate_filter_scan.csv", index=False)
    ema_combo_filters.to_csv(out_dir / "28_ema20_50_armed_entry_combo_filter_scan.csv", index=False)
    ema_filter_gate.to_csv(out_dir / "29_ema20_50_armed_entry_filter_candidate_gate.csv", index=False)
    ema_filter_stability.to_csv(out_dir / "30_ema20_50_armed_entry_filter_stability.csv", index=False)
    ema_filtered_touch.to_csv(out_dir / "31_ema20_50_armed_entry_filtered_tp_sl_grid.csv", index=False)

    # Passive limit-entry setup lab outputs. These answer whether the thin
    # gross edge survives if entries are improved and entry cost is maker-like.
    passive_specs.to_csv(out_dir / "40_passive_entry_specs.csv", index=False)
    passive_summary.to_csv(out_dir / "41_passive_entry_summary.csv", index=False)
    passive_yearly.to_csv(out_dir / "42_passive_entry_yearly.csv", index=False)
    if not passive_trades.empty and int(args.passive_trade_sample) > 0:
        passive_trades.head(int(args.passive_trade_sample)).to_csv(out_dir / "43_passive_entry_trades_sample.csv", index=False)

    if int(args.save_feature_sample) > 0:
        features.tail(int(args.save_feature_sample)).to_csv(out_dir / "30_feature_tail_sample.csv")

    meta = {
        "script": "research/eth_mhf_smooth_curve_event_lab.py",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "rows": int(len(bars)),
        "feature_rows": int(len(features)),
        "events_raw": int(len(events_raw)) if not events_raw.empty else 0,
        "event_names": int(events_raw["event_name"].nunique()) if not events_raw.empty else 0,
        "context_available_cols": list(context_available_cols),
        "cost": {
            "entry_fee_rate": float(args.entry_fee_rate),
            "exit_fee_rate": float(args.exit_fee_rate),
            "entry_slippage_pct": float(args.entry_slippage_pct),
            "exit_slippage_pct": float(args.exit_slippage_pct),
        },
        "passive_entry_lab": {
            "enabled": not bool(args.no_passive_entry_lab),
            "top_filters": int(args.passive_top_filters),
            "max_setups_per_spec": int(args.passive_max_setups_per_spec),
            "entry_offset_pcts": args.passive_entry_offset_pcts,
            "fill_windows": args.passive_fill_windows,
            "target_pcts": args.passive_target_pcts,
            "stop_pcts": args.passive_stop_pcts,
            "horizons": args.passive_horizons,
            "entry_fee": "maker limit",
            "tp_exit_fee": "maker limit",
            "stop_timeout_exit_fee": "taker + slippage",
            "same_bar_tp_sl_policy": "STOP conservative",
        },
        "ema_rule": {
            "ema_fast": int(args.ema_fast),
            "ema_slow": int(args.ema_slow),
            "entry_buffer_pct": float(args.ema_entry_buffer_pct),
            "long_entry": "after upward close cross of EMA50, any later closed bar close >= EMA50*(1+buffer), entry next open",
            "short_entry": "after downward close cross of EMA50, any later closed bar close <= EMA50*(1-buffer), entry next open",
            "long_exit": "3 closed bars below EMA20 or 1 closed bar below EMA50, exit next open",
            "short_exit": "3 closed bars above EMA20 or 1 closed bar above EMA50, exit next open",
        },
        "outputs": sorted(p.name for p in out_dir.glob("*.csv")),
    }
    with (out_dir / "99_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    # Tiny console brief.
    if not event_summary.empty:
        print("[top] event candidates", flush=True)
        cols = ["event_name", "count", "mean", "win_rate", "profit_factor", "active_days_ratio", "max_days_without_trade"]
        print(event_summary[[c for c in cols if c in event_summary.columns]].head(10).to_string(index=False), flush=True)
    if not ema_summary.empty:
        print("[top] ema summary", flush=True)
        print(ema_summary.head(10).to_string(index=False), flush=True)
    if not passive_summary.empty:
        print("[top] passive entry", flush=True)
        cols = ["label", "event_family", "side_name", "filter_expr", "entry_offset_pct", "fill_window", "target_pct", "stop_pct", "horizon", "trade_count", "fill_rate", "mean", "win_rate", "profit_factor", "candidate_pass"]
        print(passive_summary[[c for c in cols if c in passive_summary.columns]].head(10).to_string(index=False), flush=True)
    print(f"[done] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
