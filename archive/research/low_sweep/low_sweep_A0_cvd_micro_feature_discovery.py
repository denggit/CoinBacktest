#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A0 Low Sweep CVD / microstructure feature discovery research.

Research-only.  This script does not change formal backtests, portfolio code, or
live strategy behavior.  It keeps the current MF anchor fixed:

    A0_fp_abs_delta_high + single_swing + next_open + time48 + no_stop

Then it extracts a wide set of signal-time-visible microstructure features from
OKX trade bars around each parent trade:

- local CVD / delta / volume windows before and including the closed signal bar;
- spike-bar absorption proxies;
- price-vs-CVD divergence around prior lows and the previous swing low;
- preceding down-leg features;
- optional post-signal confirmation features, explicitly marked as requiring a
  delayed entry and never mixed into pre-entry rule tables;
- long-form CVD path rows around each event so the user can visually inspect the
  actual CVD sequence.

Causality convention:
- columns prefixed ``pre_`` or ``signal_`` are known when the signal bar has
  closed and may be used for the normal next-open entry research;
- columns prefixed ``post_signal_`` require waiting after the signal and are only
  valid for delayed-confirmation research.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.low_sweep_a_upgrade_research import (  # noqa: E402
    UpgradeVariant,
    _event_positions,
    build_candidate_layer_masks,
    build_market_cache,
    build_support_mask,
    parse_args as _upgrade_parse_args,
    parse_stop_specs,
    simulate_upgrade_variant,
    summarize_trades,
    write_csv,
)
from research.low_sweep_panic_reversal_strategy_backtest_probe import load_trade_bars  # noqa: E402
from backtest.mf.low_sweep.low_sweep_V1_a0_footprint_backtest import prepare_events_and_context  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402

SCRIPT_NAME = "low_sweep_A0_cvd_micro_feature_discovery"
SCRIPT_VERSION = "v2_coinbacktest_signature_fix"
DEFAULT_OUT_DIR = "data/reports/research/low_sweep/A0_cvd_micro_feature_discovery"


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def _parse_int_list(raw: str) -> list[int]:
    vals: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return sorted(set(v for v in vals if v > 0))


def _parse_float_list(raw: str) -> list[float]:
    vals: list[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return sorted(set(v for v in vals if 0 < v < 1))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--feature-windows", default="1,3,5,10,15,30,60")
    p.add_argument("--post-signal-bars", default="1,2,3,5,10")
    p.add_argument("--path-offset-left", type=int, default=60)
    p.add_argument("--path-offset-right", type=int, default=10)
    p.add_argument("--rule-quantiles", default="0.10,0.20,0.30,0.50")
    p.add_argument("--min-rule-selected", type=int, default=5)
    p.add_argument("--big-win-pct", type=float, default=0.020)
    p.add_argument("--big-loss-pct", type=float, default=-0.010)
    p.add_argument("--max-rule-features", type=int, default=240)
    p.add_argument("--save-cvd-path", type=int, default=1, help="Save long-form CVD/order-flow path around every parent trade.")
    p.add_argument("--save-post-signal", type=int, default=1, help="Include post-signal delayed-confirmation features in the full feature matrix.")
    known, rest = p.parse_known_args(argv)

    defaults = [
        "--out-dir",
        DEFAULT_OUT_DIR,
        "--candidate-layers",
        "A0_fp_abs_delta_high",
        "--support-modes",
        "single_swing",
        "--entry-modes",
        "next_open",
        "--exit-modes",
        "time48",
        "--upgrade-stop-specs",
        "no_stop",
        "--context-sources",
        "trade_bar,footprint",
        # Keep micro attachment off by default. This script extracts rich 1m
        # CVD/delta features from the primary trade bars. Users can still pass
        # --micro-timeframes 5s,10s --micro-load-mode local/auto explicitly.
        "--micro-timeframes",
        "",
        "--save-trades",
        "0",
        "--save-events",
        "4000",
    ]
    args = _upgrade_parse_args(defaults + list(rest))
    for k, v in vars(known).items():
        setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _as_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_div(n: float, d: float, default: float = np.nan) -> float:
    if not math.isfinite(n) or not math.isfinite(d) or abs(d) <= 1e-12:
        return default
    return float(n / d)


def _slice(arr: np.ndarray, start: int, end_inclusive: int) -> np.ndarray:
    if arr.size == 0:
        return np.asarray([], dtype=float)
    start = max(0, int(start))
    end_inclusive = min(int(end_inclusive), len(arr) - 1)
    if start > end_inclusive:
        return np.asarray([], dtype=float)
    return arr[start : end_inclusive + 1]


def _nan_sum(arr: np.ndarray) -> float:
    return float(np.nansum(arr)) if arr.size else np.nan


def _nan_mean(arr: np.ndarray) -> float:
    return float(np.nanmean(arr)) if arr.size and np.isfinite(arr).any() else np.nan


def _nan_min(arr: np.ndarray) -> float:
    return float(np.nanmin(arr)) if arr.size and np.isfinite(arr).any() else np.nan


def _nan_max(arr: np.ndarray) -> float:
    return float(np.nanmax(arr)) if arr.size and np.isfinite(arr).any() else np.nan


def _last_finite(arr: np.ndarray) -> float:
    if not arr.size:
        return np.nan
    finite = arr[np.isfinite(arr)]
    return float(finite[-1]) if finite.size else np.nan


def _profit_factor(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if vals.empty:
        return np.nan
    gains = vals[vals > 0].sum()
    losses = -vals[vals < 0].sum()
    if losses <= 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def _auc_score(x: pd.Series, y: pd.Series) -> float:
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    m = xv.notna() & yv.notna()
    xv = xv[m]
    yv = yv[m].astype(int)
    n_pos = int((yv == 1).sum())
    n_neg = int((yv == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = xv.rank(method="average")
    pos_rank_sum = float(ranks[yv == 1].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _max_drawdown_from_returns(x: pd.Series, starting_equity: float = 1.0) -> float:
    vals = pd.to_numeric(x, errors="coerce").fillna(0.0)
    if vals.empty:
        return np.nan
    equity = float(starting_equity) * (1.0 + vals).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


# ---------------------------------------------------------------------------
# Bar arrays and feature extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BarArrays:
    index: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    notional: np.ndarray
    buy_notional: np.ndarray
    sell_notional: np.ndarray
    delta_notional: np.ndarray
    cvd_notional: np.ndarray
    buy_volume: np.ndarray
    sell_volume: np.ndarray
    delta_volume: np.ndarray
    cvd_volume: np.ndarray
    taker_buy_ratio: np.ndarray
    large_buy_notional: np.ndarray
    large_sell_notional: np.ndarray
    large_delta_notional: np.ndarray
    large_trades_count: np.ndarray
    trades_count: np.ndarray


def _col(frame: pd.DataFrame, name: str, fallback: float = np.nan) -> np.ndarray:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    return np.full(len(frame), float(fallback), dtype=float)


def build_bar_arrays(bars: pd.DataFrame) -> BarArrays:
    frame = bars.sort_index()
    buy_notional = _col(frame, "buy_notional")
    sell_notional = _col(frame, "sell_notional")
    delta_notional = _col(frame, "delta_notional")
    if not np.isfinite(delta_notional).any():
        delta_notional = buy_notional - sell_notional
    notional = _col(frame, "notional")
    if not np.isfinite(notional).any():
        notional = buy_notional + sell_notional
    cvd_notional = _col(frame, "cvd_notional")
    if not np.isfinite(cvd_notional).any():
        cvd_notional = np.nancumsum(np.nan_to_num(delta_notional, nan=0.0))

    buy_volume = _col(frame, "buy_volume")
    sell_volume = _col(frame, "sell_volume")
    delta_volume = _col(frame, "delta_volume")
    if not np.isfinite(delta_volume).any():
        delta_volume = buy_volume - sell_volume
    cvd_volume = _col(frame, "cvd_volume")
    if not np.isfinite(cvd_volume).any():
        cvd_volume = np.nancumsum(np.nan_to_num(delta_volume, nan=0.0))

    return BarArrays(
        index=pd.DatetimeIndex(frame.index),
        open=_col(frame, "open"),
        high=_col(frame, "high"),
        low=_col(frame, "low"),
        close=_col(frame, "close"),
        volume=_col(frame, "volume"),
        notional=notional,
        buy_notional=buy_notional,
        sell_notional=sell_notional,
        delta_notional=delta_notional,
        cvd_notional=cvd_notional,
        buy_volume=buy_volume,
        sell_volume=sell_volume,
        delta_volume=delta_volume,
        cvd_volume=cvd_volume,
        taker_buy_ratio=_col(frame, "taker_buy_ratio"),
        large_buy_notional=_col(frame, "large_buy_notional"),
        large_sell_notional=_col(frame, "large_sell_notional"),
        large_delta_notional=_col(frame, "large_delta_notional"),
        large_trades_count=_col(frame, "large_trades_count"),
        trades_count=_col(frame, "trades_count"),
    )


def _range_frac(open_: float, high: float, low: float, close: float) -> tuple[float, float, float]:
    rng = high - low
    if not math.isfinite(rng) or rng <= 0:
        return np.nan, np.nan, np.nan
    close_pos = (close - low) / rng
    lower_wick = max(0.0, min(open_, close) - low) / rng
    upper_wick = max(0.0, high - max(open_, close)) / rng
    return float(close_pos), float(lower_wick), float(upper_wick)


def _window_stats(arr: BarArrays, signal_pos: int, w: int) -> dict[str, float]:
    end = int(signal_pos)
    start = max(0, end - int(w) + 1)
    prev_start = max(0, start - int(w))
    prev_end = start - 1

    close_win = _slice(arr.close, start, end)
    high_win = _slice(arr.high, start, end)
    low_win = _slice(arr.low, start, end)
    vol_win = _slice(arr.volume, start, end)
    notional_win = _slice(arr.notional, start, end)
    buy_notional_win = _slice(arr.buy_notional, start, end)
    sell_notional_win = _slice(arr.sell_notional, start, end)
    delta_notional_win = _slice(arr.delta_notional, start, end)
    delta_volume_win = _slice(arr.delta_volume, start, end)
    large_buy_win = _slice(arr.large_buy_notional, start, end)
    large_sell_win = _slice(arr.large_sell_notional, start, end)
    large_delta_win = _slice(arr.large_delta_notional, start, end)

    prev_delta_win = _slice(arr.delta_notional, prev_start, prev_end) if prev_end >= prev_start else np.asarray([], dtype=float)
    prev_notional_win = _slice(arr.notional, prev_start, prev_end) if prev_end >= prev_start else np.asarray([], dtype=float)
    prev_low_win = _slice(arr.low, prev_start, prev_end) if prev_end >= prev_start else np.asarray([], dtype=float)
    prev_cvd_win = _slice(arr.cvd_notional, prev_start, prev_end) if prev_end >= prev_start else np.asarray([], dtype=float)

    sum_notional = _nan_sum(notional_win)
    sum_buy = _nan_sum(buy_notional_win)
    sum_sell = _nan_sum(sell_notional_win)
    sum_delta = _nan_sum(delta_notional_win)
    sum_large_buy = _nan_sum(large_buy_win)
    sum_large_sell = _nan_sum(large_sell_win)
    sum_large_delta = _nan_sum(large_delta_win)
    first_close = float(close_win[0]) if close_win.size and math.isfinite(float(close_win[0])) else np.nan
    last_close = _last_finite(close_win)
    min_low = _nan_min(low_win)
    max_high = _nan_max(high_win)

    # Prior-window low/CVD divergence.  Uses only bars before and including the
    # signal bar.  If the signal makes a lower/equal low while CVD is above the
    # prior-low CVD, that is a local CVD higher-low divergence proxy.
    prior_low = _nan_min(prev_low_win)
    prior_low_cvd = np.nan
    if prev_low_win.size and np.isfinite(prev_low_win).any():
        rel = int(np.nanargmin(prev_low_win))
        if rel < prev_cvd_win.size:
            prior_low_cvd = float(prev_cvd_win[rel])
    signal_low = float(arr.low[end])
    signal_cvd = float(arr.cvd_notional[end])

    return {
        f"pre_w{w}_close_ret": _safe_div(last_close, first_close, np.nan) - 1.0 if math.isfinite(first_close) and first_close > 0 else np.nan,
        f"pre_w{w}_max_high_ret_from_first_close": _safe_div(max_high, first_close, np.nan) - 1.0 if math.isfinite(first_close) and first_close > 0 else np.nan,
        f"pre_w{w}_min_low_ret_from_first_close": _safe_div(min_low, first_close, np.nan) - 1.0 if math.isfinite(first_close) and first_close > 0 else np.nan,
        f"pre_w{w}_range_pct": _safe_div(max_high, min_low, np.nan) - 1.0 if math.isfinite(min_low) and min_low > 0 else np.nan,
        f"pre_w{w}_volume_sum": _nan_sum(vol_win),
        f"pre_w{w}_notional_sum": sum_notional,
        f"pre_w{w}_delta_notional_sum": sum_delta,
        f"pre_w{w}_delta_volume_sum": _nan_sum(delta_volume_win),
        f"pre_w{w}_delta_pressure": _safe_div(sum_delta, sum_notional),
        f"pre_w{w}_buy_notional_share": _safe_div(sum_buy, sum_buy + sum_sell),
        f"pre_w{w}_sell_notional_share": _safe_div(sum_sell, sum_buy + sum_sell),
        f"pre_w{w}_large_notional_share": _safe_div(sum_large_buy + sum_large_sell, sum_notional),
        f"pre_w{w}_large_delta_pressure": _safe_div(sum_large_delta, sum_large_buy + sum_large_sell),
        f"pre_w{w}_cvd_notional_change": sum_delta,
        f"pre_w{w}_cvd_volume_change": _nan_sum(delta_volume_win),
        f"pre_w{w}_delta_accel_vs_prev": sum_delta - _nan_sum(prev_delta_win),
        f"pre_w{w}_delta_pressure_accel_vs_prev": _safe_div(sum_delta, sum_notional) - _safe_div(_nan_sum(prev_delta_win), _nan_sum(prev_notional_win)),
        f"pre_w{w}_sell_exhaustion_vs_prev": float(sum_delta > _nan_sum(prev_delta_win)) if prev_delta_win.size else np.nan,
        f"pre_w{w}_price_lower_low_cvd_higher_low": float(signal_low <= prior_low and signal_cvd > prior_low_cvd) if math.isfinite(prior_low) and math.isfinite(prior_low_cvd) else np.nan,
        f"pre_w{w}_signal_low_vs_prior_window_low_pct": _safe_div(signal_low, prior_low) - 1.0 if math.isfinite(prior_low) and prior_low > 0 else np.nan,
        f"pre_w{w}_signal_cvd_vs_prior_low_cvd": signal_cvd - prior_low_cvd if math.isfinite(prior_low_cvd) else np.nan,
    }


def _preceding_downleg_stats(arr: BarArrays, signal_pos: int, lookback: int = 60) -> dict[str, float]:
    start = max(0, int(signal_pos) - int(lookback))
    end = int(signal_pos)
    highs = _slice(arr.high, start, end)
    if not highs.size or not np.isfinite(highs).any():
        return {}
    rel_high = int(np.nanargmax(highs))
    high_pos = start + rel_high
    leg_low = float(arr.low[end])
    leg_high = float(arr.high[high_pos])
    delta_leg = _nan_sum(_slice(arr.delta_notional, high_pos, end))
    notional_leg = _nan_sum(_slice(arr.notional, high_pos, end))
    volume_leg = _nan_sum(_slice(arr.volume, high_pos, end))
    return {
        "pre_downleg_lookback60_bars_from_high": int(end - high_pos),
        "pre_downleg_lookback60_price_ret_high_to_signal_low": _safe_div(leg_low, leg_high) - 1.0 if math.isfinite(leg_high) and leg_high > 0 else np.nan,
        "pre_downleg_lookback60_delta_notional_sum": delta_leg,
        "pre_downleg_lookback60_delta_pressure": _safe_div(delta_leg, notional_leg),
        "pre_downleg_lookback60_volume_sum": volume_leg,
        "pre_downleg_lookback60_cvd_per_price_drop": _safe_div(delta_leg, abs(_safe_div(leg_low, leg_high, np.nan) - 1.0)),
    }


def _swing_compare_stats(arr: BarArrays, event: pd.Series, signal_pos: int) -> dict[str, float]:
    swing_age = _as_float(event.get("swing_age", np.nan))
    swing_level = _as_float(event.get("swing_level", np.nan))
    if not math.isfinite(swing_age) or swing_age < 0:
        return {}
    swing_pos = int(signal_pos) - int(round(swing_age))
    if swing_pos < 0 or swing_pos >= len(arr.index):
        return {}
    sig = int(signal_pos)
    delta_between = _nan_sum(_slice(arr.delta_notional, swing_pos, sig))
    notional_between = _nan_sum(_slice(arr.notional, swing_pos, sig))
    signal_low = float(arr.low[sig])
    swing_low = float(arr.low[swing_pos])
    if math.isfinite(swing_level) and swing_level > 0:
        swing_low_ref = swing_level
    else:
        swing_low_ref = swing_low
    return {
        "swing_ref_pos": int(swing_pos),
        "swing_ref_time": arr.index[swing_pos],
        "swing_ref_low": swing_low,
        "swing_signal_low_vs_swing_level_pct": _safe_div(signal_low, swing_low_ref) - 1.0 if math.isfinite(swing_low_ref) and swing_low_ref > 0 else np.nan,
        "swing_signal_close_vs_swing_level_pct": _safe_div(float(arr.close[sig]), swing_low_ref) - 1.0 if math.isfinite(swing_low_ref) and swing_low_ref > 0 else np.nan,
        "swing_signal_cvd_minus_swing_cvd": float(arr.cvd_notional[sig] - arr.cvd_notional[swing_pos]),
        "swing_signal_delta_notional_sum": delta_between,
        "swing_signal_delta_pressure": _safe_div(delta_between, notional_between),
        "swing_price_lower_low_cvd_higher_low": float(signal_low <= swing_low_ref and arr.cvd_notional[sig] > arr.cvd_notional[swing_pos]) if math.isfinite(swing_low_ref) else np.nan,
        "swing_bars_between": int(sig - swing_pos),
    }


def _post_signal_stats(arr: BarArrays, signal_pos: int, bars_list: Sequence[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    sig = int(signal_pos)
    signal_close = float(arr.close[sig])
    signal_low = float(arr.low[sig])
    for k in bars_list:
        end = min(len(arr.index) - 1, sig + int(k))
        if end <= sig:
            continue
        sl = slice(sig + 1, end + 1)
        close_end = float(arr.close[end])
        high_max = _nan_max(arr.high[sl])
        low_min = _nan_min(arr.low[sl])
        delta_sum = _nan_sum(arr.delta_notional[sl])
        notional_sum = _nan_sum(arr.notional[sl])
        out.update({
            f"post_signal_{k}_close_ret_requires_delay": _safe_div(close_end, signal_close) - 1.0 if math.isfinite(signal_close) and signal_close > 0 else np.nan,
            f"post_signal_{k}_max_high_ret_requires_delay": _safe_div(high_max, signal_close) - 1.0 if math.isfinite(signal_close) and signal_close > 0 else np.nan,
            f"post_signal_{k}_min_low_ret_requires_delay": _safe_div(low_min, signal_close) - 1.0 if math.isfinite(signal_close) and signal_close > 0 else np.nan,
            f"post_signal_{k}_delta_notional_sum_requires_delay": delta_sum,
            f"post_signal_{k}_delta_pressure_requires_delay": _safe_div(delta_sum, notional_sum),
            f"post_signal_{k}_cvd_reclaim_positive_requires_delay": float(delta_sum > 0),
            f"post_signal_{k}_no_new_low_requires_delay": float(low_min >= signal_low),
            f"post_signal_{k}_close_reclaim_signal_open_requires_delay": float(close_end >= float(arr.open[sig])),
        })
    return out


def extract_feature_row(
    arr: BarArrays,
    event: pd.Series,
    trade: pd.Series,
    signal_pos: int,
    feature_windows: Sequence[int],
    post_signal_bars: Sequence[int],
    include_post_signal: bool,
) -> dict[str, object]:
    sig = int(signal_pos)
    open_ = float(arr.open[sig])
    high = float(arr.high[sig])
    low = float(arr.low[sig])
    close = float(arr.close[sig])
    close_pos, lower_wick, upper_wick = _range_frac(open_, high, low, close)
    notional = float(arr.notional[sig])
    delta_notional = float(arr.delta_notional[sig])
    large_delta = float(arr.large_delta_notional[sig])
    large_total = float(arr.large_buy_notional[sig] + arr.large_sell_notional[sig])

    row: dict[str, object] = {
        "signal_time": event.get("signal_time"),
        "entry_time": trade.get("entry_time"),
        "exit_time": trade.get("exit_time"),
        "signal_pos": int(sig),
        "entry_pos": int(trade.get("entry_pos", sig + 1)),
        "exit_pos": int(trade.get("exit_pos", sig + 48)),
        "net_return_on_equity": _as_float(trade.get("net_return_on_equity", np.nan)),
        "mae_on_equity": _as_float(trade.get("mae_on_equity", np.nan)),
        "mfe_on_equity": _as_float(trade.get("mfe_on_equity", np.nan)),
        "bars_held": _as_float(trade.get("bars_held", np.nan)),
        "is_win": bool(_as_float(trade.get("net_return_on_equity", np.nan)) > 0),
        "is_loss": bool(_as_float(trade.get("net_return_on_equity", np.nan)) < 0),
        "is_big_win": False,  # filled after args are known in build matrix
        "is_big_loss": False,
        "has_overlap_signal": bool(trade.get("has_overlap_signal", False)),
        "overlap_signal_count": int(_as_float(trade.get("overlap_signal_count", 0), 0.0)),
        "first_overlap_signal_time": trade.get("first_overlap_signal_time", pd.NaT),
        "first_overlap_bars_after_signal": _as_float(trade.get("first_overlap_bars_after_signal", np.nan)),
        "event_name": event.get("event_name", ""),
        "session_bucket": event.get("session_bucket", "NA"),
        "swing_age": _as_float(event.get("swing_age", np.nan)),
        "swing_level": _as_float(event.get("swing_level", np.nan)),
        "atr_pct": _as_float(event.get("atr_pct", np.nan)),
        "down_spike_pct": _as_float(event.get("down_spike_pct", np.nan)),
        "large_trade_share": _as_float(event.get("large_trade_share", np.nan)),
        "event_close_pos_in_bar": _as_float(event.get("close_pos_in_bar", np.nan)),
        "signal_open": open_,
        "signal_high": high,
        "signal_low": low,
        "signal_close": close,
        "signal_body_ret": _safe_div(close, open_) - 1.0 if math.isfinite(open_) and open_ > 0 else np.nan,
        "signal_range_pct": _safe_div(high, low) - 1.0 if math.isfinite(low) and low > 0 else np.nan,
        "signal_close_pos_in_bar": close_pos,
        "signal_lower_wick_frac": lower_wick,
        "signal_upper_wick_frac": upper_wick,
        "signal_volume": float(arr.volume[sig]),
        "signal_notional": notional,
        "signal_buy_notional": float(arr.buy_notional[sig]),
        "signal_sell_notional": float(arr.sell_notional[sig]),
        "signal_delta_notional": delta_notional,
        "signal_delta_volume": float(arr.delta_volume[sig]),
        "signal_cvd_notional": float(arr.cvd_notional[sig]),
        "signal_cvd_volume": float(arr.cvd_volume[sig]),
        "signal_delta_pressure": _safe_div(delta_notional, notional),
        "signal_buy_notional_share": _safe_div(float(arr.buy_notional[sig]), float(arr.buy_notional[sig] + arr.sell_notional[sig])),
        "signal_taker_buy_ratio": float(arr.taker_buy_ratio[sig]),
        "signal_large_delta_pressure": _safe_div(large_delta, large_total),
        "signal_large_notional_share": _safe_div(large_total, notional),
        "signal_spike_absorption_proxy": close_pos - max(0.0, -_safe_div(delta_notional, notional, 0.0)) if math.isfinite(close_pos) else np.nan,
    }

    # Keep known footprint/context columns from the existing event builder.  They
    # are signal-time context from existing code, not newly aligned here.
    for col in event.index:
        if str(col).startswith(("fp_", "range_", "micro_")) and col not in row:
            row[str(col)] = event.get(col)

    for w in feature_windows:
        row.update(_window_stats(arr, sig, int(w)))
    row.update(_preceding_downleg_stats(arr, sig, lookback=max(60, max(feature_windows) if feature_windows else 60)))
    row.update(_swing_compare_stats(arr, event, sig))
    if include_post_signal:
        row.update(_post_signal_stats(arr, sig, post_signal_bars))
    return row


# ---------------------------------------------------------------------------
# Trade selection and labels
# ---------------------------------------------------------------------------


def build_baseline_trades_and_events(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    print("[select] A0_fp_abs_delta_high + single_swing", flush=True)
    layer_masks = build_candidate_layer_masks(events, args)
    layer_mask = layer_masks.get("A0_fp_abs_delta_high", pd.Series(False, index=events.index)).fillna(False).astype(bool)
    support_mask = build_support_mask(events, "single_swing", args).fillna(False).astype(bool)
    selected = events.loc[layer_mask & support_mask].copy().sort_values("signal_time").reset_index(drop=True)
    selected, positions = _event_positions(bars, selected, max_horizon=72)
    print(f"[select] selected_events={len(selected):,} valid_positions={len(positions):,}", flush=True)

    stop = parse_stop_specs("no_stop")[0]
    variant = UpgradeVariant(
        variant_name="A0_fp_abs_delta_high__single_swing__next_open__time48__no_stop",
        candidate_layer="A0_fp_abs_delta_high",
        support_mode="single_swing",
        entry_mode="next_open",
        exit_mode="time48",
        stop_spec=stop,
    )
    market = build_market_cache(bars, args)
    # CoinBacktest research.low_sweep_a_upgrade_research.simulate_upgrade_variant
    # does not accept cost_mult.  Cost stress is implemented in the formal MF
    # wrapper by calling simulate_upgrade_trade directly.  This feature
    # discovery script only needs the baseline cost-1x parent trades, so keep
    # the project-native signature here.
    trades, counters = simulate_upgrade_variant(bars, selected, variant, args, market=market)
    if trades.empty:
        print(f"[simulate] no valid baseline trades counters={counters}", flush=True)
        return trades, selected, positions
    trades = trades.sort_values("signal_time").reset_index(drop=True)
    print(f"[simulate] baseline trades={len(trades):,} skipped_overlap={counters.get('skipped_overlap', np.nan)}", flush=True)
    return trades, selected, positions


def attach_overlap_labels(trades: pd.DataFrame, selected: pd.DataFrame, positions: np.ndarray) -> pd.DataFrame:
    if trades.empty or selected.empty:
        return trades
    out = trades.copy()
    event_pos = pd.Series(positions, index=pd.to_datetime(selected["signal_time"]))
    event_times = pd.to_datetime(selected["signal_time"]).to_numpy()
    pos_arr = np.asarray(positions, dtype=int)
    counts: list[int] = []
    first_times: list[object] = []
    first_positions: list[float] = []
    first_bars_after: list[float] = []
    for _, tr in out.iterrows():
        sig_pos = int(tr.get("signal_pos", -1))
        exit_pos = int(tr.get("exit_pos", -1))
        mask = (pos_arr > sig_pos) & (pos_arr <= exit_pos)
        c = int(mask.sum())
        counts.append(c)
        if c:
            i = int(np.flatnonzero(mask)[0])
            first_times.append(event_times[i])
            first_positions.append(float(pos_arr[i]))
            first_bars_after.append(float(pos_arr[i] - sig_pos))
        else:
            first_times.append(pd.NaT)
            first_positions.append(np.nan)
            first_bars_after.append(np.nan)
    out["has_overlap_signal"] = [c > 0 for c in counts]
    out["overlap_signal_count"] = counts
    out["first_overlap_signal_time"] = first_times
    out["first_overlap_signal_pos"] = first_positions
    out["first_overlap_bars_after_signal"] = first_bars_after
    return out


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------


def _numeric_feature_columns(df: pd.DataFrame, *, include_post: bool, max_features: int) -> list[str]:
    skip = {
        "signal_pos",
        "entry_pos",
        "exit_pos",
        "net_return_on_equity",
        "mae_on_equity",
        "mfe_on_equity",
        "bars_held",
        "is_win",
        "is_loss",
        "is_big_win",
        "is_big_loss",
        "has_overlap_signal",
        "overlap_signal_count",
    }
    cols: list[str] = []
    for c in df.columns:
        if c in skip:
            continue
        if not include_post and str(c).startswith("post_signal_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            valid = pd.to_numeric(df[c], errors="coerce").notna().sum()
            if valid >= max(8, min(20, len(df) // 5)):
                cols.append(c)
    # Keep output size bounded but deterministic.  Prefer explicit CVD/delta and
    # signal/pre/swing/downleg fields before inherited broad context fields.
    def priority(name: str) -> tuple[int, str]:
        prefixes = ("signal_", "pre_", "swing_", "pre_downleg_", "atr_", "down_spike", "large_trade", "fp_", "range_", "micro_", "post_signal_")
        for i, p in enumerate(prefixes):
            if name.startswith(p):
                return (i, name)
        return (99, name)
    cols = sorted(cols, key=priority)
    return cols[: max(1, int(max_features))]


def build_feature_diff(df: pd.DataFrame, target_col: str, feature_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if df.empty or target_col not in df.columns:
        return pd.DataFrame()
    y = df[target_col].astype(bool)
    base_rate = float(y.mean()) if len(y) else np.nan
    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        pos = x[y].dropna()
        neg = x[~y].dropna()
        if len(pos) < 2 or len(neg) < 2:
            continue
        rows.append({
            "target": target_col,
            "base_rate": base_rate,
            "feature": col,
            "pos_count": int(len(pos)),
            "neg_count": int(len(neg)),
            "pos_mean": float(pos.mean()),
            "neg_mean": float(neg.mean()),
            "mean_diff_pos_minus_neg": float(pos.mean() - neg.mean()),
            "pos_median": float(pos.median()),
            "neg_median": float(neg.median()),
            "median_diff_pos_minus_neg": float(pos.median() - neg.median()),
            "auc_high_predicts_target": _auc_score(x, y.astype(int)),
            "auc_best_direction": max(_auc_score(x, y.astype(int)), 1.0 - _auc_score(x, y.astype(int))) if math.isfinite(_auc_score(x, y.astype(int))) else np.nan,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["auc_best_direction", "target", "feature"], ascending=[False, True, True]).reset_index(drop=True)


def build_correlations(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y = pd.to_numeric(df.get("net_return_on_equity", pd.Series(dtype=float)), errors="coerce")
    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        m = x.notna() & y.notna()
        if int(m.sum()) < 8:
            continue
        pearson = float(x[m].corr(y[m], method="pearson")) if x[m].nunique(dropna=True) > 1 else np.nan
        spearman = float(x[m].corr(y[m], method="spearman")) if x[m].nunique(dropna=True) > 1 else np.nan
        rows.append({
            "feature": col,
            "valid_count": int(m.sum()),
            "pearson_net_return": pearson,
            "spearman_net_return": spearman,
            "abs_spearman": abs(spearman) if math.isfinite(spearman) else np.nan,
            "auc_is_win": _auc_score(x, df["is_win"].astype(int)) if "is_win" in df else np.nan,
            "auc_is_big_win": _auc_score(x, df["is_big_win"].astype(int)) if "is_big_win" in df else np.nan,
            "auc_has_overlap_signal": _auc_score(x, df["has_overlap_signal"].astype(int)) if "has_overlap_signal" in df else np.nan,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["abs_spearman", "feature"], ascending=[False, True]).reset_index(drop=True)


def build_single_feature_rules(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    targets: Sequence[str],
    quantiles: Sequence[float],
    min_selected: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_total = len(df)
    if n_total == 0:
        return pd.DataFrame()
    for target in targets:
        if target not in df.columns:
            continue
        y = df[target].astype(bool)
        base_rate = float(y.mean()) if n_total else np.nan
        for col in feature_cols:
            x = pd.to_numeric(df[col], errors="coerce")
            valid = x.notna()
            if int(valid.sum()) < max(int(min_selected), 8):
                continue
            for q in quantiles:
                thresholds = []
                lo = float(x[valid].quantile(q))
                hi = float(x[valid].quantile(1.0 - q))
                thresholds.append(("low", "<=", lo, x <= lo))
                thresholds.append(("high", ">=", hi, x >= hi))
                for side, op, threshold, mask in thresholds:
                    sel = valid & mask
                    c = int(sel.sum())
                    if c < int(min_selected):
                        continue
                    hits = int(y[sel].sum())
                    precision = float(hits / c) if c else np.nan
                    recall = float(hits / max(1, int(y.sum()))) if int(y.sum()) else np.nan
                    ret = pd.to_numeric(df.loc[sel, "net_return_on_equity"], errors="coerce")
                    rows.append({
                        "target": target,
                        "feature": col,
                        "side": side,
                        "operator": op,
                        "threshold": threshold,
                        "quantile_tail": float(q),
                        "selected": c,
                        "target_hits": hits,
                        "base_rate": base_rate,
                        "precision": precision,
                        "recall": recall,
                        "lift_vs_base": precision / base_rate if base_rate and math.isfinite(base_rate) and base_rate > 0 else np.nan,
                        "mean_return_selected": float(ret.mean()) if not ret.empty else np.nan,
                        "median_return_selected": float(ret.median()) if not ret.empty else np.nan,
                        "pf_selected": _profit_factor(ret),
                        "max_dd_selected_in_original_order": _max_drawdown_from_returns(ret),
                    })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["target", "lift_vs_base", "precision", "selected"], ascending=[True, False, False, False]).reset_index(drop=True)


def build_session_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "session_bucket" not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for session, grp in df.groupby("session_bucket", dropna=False):
        ret = pd.to_numeric(grp["net_return_on_equity"], errors="coerce")
        rows.append({
            "session_bucket": session,
            "trades": int(len(grp)),
            "return_total": float((1.0 + ret.fillna(0.0)).prod() - 1.0),
            "mean_return": float(ret.mean()),
            "median_return": float(ret.median()),
            "win_rate": float((ret > 0).mean()),
            "profit_factor": _profit_factor(ret),
            "has_overlap_rate": float(grp.get("has_overlap_signal", pd.Series(False, index=grp.index)).astype(bool).mean()),
            "big_win_rate": float(grp.get("is_big_win", pd.Series(False, index=grp.index)).astype(bool).mean()),
            "big_loss_rate": float(grp.get("is_big_loss", pd.Series(False, index=grp.index)).astype(bool).mean()),
        })
    return pd.DataFrame(rows).sort_values("session_bucket").reset_index(drop=True)


def build_cvd_path_rows(
    arr: BarArrays,
    feature_df: pd.DataFrame,
    left: int,
    right: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if feature_df.empty:
        return pd.DataFrame()
    for i, row in feature_df.reset_index(drop=True).iterrows():
        sig = int(row["signal_pos"])
        start = max(0, sig - int(left))
        end = min(len(arr.index) - 1, sig + int(right))
        signal_close = float(arr.close[sig])
        signal_cvd = float(arr.cvd_notional[sig])
        for pos in range(start, end + 1):
            notional = float(arr.notional[pos])
            delta = float(arr.delta_notional[pos])
            close_pos, lower_wick, upper_wick = _range_frac(float(arr.open[pos]), float(arr.high[pos]), float(arr.low[pos]), float(arr.close[pos]))
            rows.append({
                "parent_row": int(i),
                "signal_time": row.get("signal_time"),
                "bar_time": arr.index[pos],
                "offset_bars": int(pos - sig),
                "open": float(arr.open[pos]),
                "high": float(arr.high[pos]),
                "low": float(arr.low[pos]),
                "close": float(arr.close[pos]),
                "close_ret_vs_signal_close": _safe_div(float(arr.close[pos]), signal_close) - 1.0 if signal_close > 0 else np.nan,
                "low_ret_vs_signal_close": _safe_div(float(arr.low[pos]), signal_close) - 1.0 if signal_close > 0 else np.nan,
                "volume": float(arr.volume[pos]),
                "notional": notional,
                "buy_notional": float(arr.buy_notional[pos]),
                "sell_notional": float(arr.sell_notional[pos]),
                "delta_notional": delta,
                "delta_pressure": _safe_div(delta, notional),
                "cvd_notional": float(arr.cvd_notional[pos]),
                "cvd_change_vs_signal": float(arr.cvd_notional[pos] - signal_cvd),
                "large_delta_notional": float(arr.large_delta_notional[pos]),
                "large_delta_pressure": _safe_div(float(arr.large_delta_notional[pos]), float(arr.large_buy_notional[pos] + arr.large_sell_notional[pos])),
                "close_pos_in_bar": close_pos,
                "lower_wick_frac": lower_wick,
                "upper_wick_frac": upper_wick,
                "net_return_on_equity": row.get("net_return_on_equity"),
                "is_win": row.get("is_win"),
                "is_big_win": row.get("is_big_win"),
                "has_overlap_signal": row.get("has_overlap_signal"),
            })
    return pd.DataFrame(rows)


def build_meta(args: argparse.Namespace, bars: pd.DataFrame, events: pd.DataFrame, trades: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"key": "script", "value": SCRIPT_NAME},
        {"key": "version", "value": SCRIPT_VERSION},
        {"key": "symbol", "value": args.symbol},
        {"key": "timeframe", "value": args.timeframe},
        {"key": "date_range", "value": f"{args.start_date}->{args.end_date}"},
        {"key": "warmup_start_date", "value": args.warmup_start_date},
        {"key": "bars_rows", "value": len(bars)},
        {"key": "events_rows", "value": len(events)},
        {"key": "baseline_trades", "value": len(trades)},
        {"key": "feature_rows", "value": len(feature_df)},
        {"key": "anchor", "value": "A0_fp_abs_delta_high + single_swing + next_open + time48 + no_stop"},
        {"key": "causality", "value": "signal_/pre_ fields are visible after signal bar close; post_signal_* requires delayed confirmation"},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {SCRIPT_NAME} {SCRIPT_VERSION}", flush=True)
    print("[scope] research-only feature discovery; no portfolio/formal/live changes", flush=True)
    print(f"[args] out_dir={out_dir}", flush=True)

    feature_windows = _parse_int_list(args.feature_windows)
    post_signal_bars = _parse_int_list(args.post_signal_bars)
    quantiles = _parse_float_list(args.rule_quantiles)

    print(f"[load] trade bars {args.symbol} {args.timeframe} {args.warmup_start_date}->{args.end_date}", flush=True)
    bars = load_trade_bars(args)
    arr = build_bar_arrays(bars)

    print("[events] build existing Low Sweep V1 A0 events/context", flush=True)
    events = prepare_events_and_context(bars, args)

    trades, selected_events, selected_positions = build_baseline_trades_and_events(bars, events, args)
    trades = attach_overlap_labels(trades, selected_events, selected_positions)
    if trades.empty:
        write_csv(build_meta(args, bars, events, trades, pd.DataFrame()), out_dir / "99_meta.csv", "meta")
        return 0

    # Map parent trade signal_time back to the selected event row and signal pos.
    event_map = selected_events.copy()
    event_map["signal_time"] = pd.to_datetime(event_map["signal_time"])
    pos_map = pd.Series(selected_positions, index=event_map["signal_time"])
    event_by_time = {pd.Timestamp(r["signal_time"]): r for _, r in event_map.iterrows()}

    rows: list[dict[str, object]] = []
    progress = ProgressReporter(
        label="[features] A0 parent CVD/micro extraction",
        total=len(trades),
        every=max(1, int(getattr(args, "progress_every", 1000))),
        enabled=not bool(getattr(args, "no_progress", False)),
    )
    for i, tr in trades.iterrows():
        st = pd.Timestamp(tr["signal_time"])
        ev = event_by_time.get(st)
        if ev is None:
            continue
        sig_pos = int(pos_map.loc[st]) if st in pos_map.index else int(tr.get("signal_pos", -1))
        if sig_pos < 0:
            continue
        row = extract_feature_row(
            arr,
            ev,
            tr,
            signal_pos=sig_pos,
            feature_windows=feature_windows,
            post_signal_bars=post_signal_bars,
            include_post_signal=bool(int(args.save_post_signal)),
        )
        rows.append(row)
        progress.update(i + 1)
    progress.close()

    feature_df = pd.DataFrame(rows).sort_values("signal_time").reset_index(drop=True)
    if not feature_df.empty:
        ret = pd.to_numeric(feature_df["net_return_on_equity"], errors="coerce")
        feature_df["is_big_win"] = ret >= float(args.big_win_pct)
        feature_df["is_big_loss"] = ret <= float(args.big_loss_pct)

    pre_feature_cols = _numeric_feature_columns(feature_df, include_post=False, max_features=int(args.max_rule_features))
    post_feature_cols = [c for c in _numeric_feature_columns(feature_df, include_post=True, max_features=int(args.max_rule_features) * 2) if str(c).startswith("post_signal_")]
    all_feature_cols = _numeric_feature_columns(feature_df, include_post=True, max_features=int(args.max_rule_features) * 2)

    baseline_summary = pd.DataFrame([summarize_trades(trades, args, extra={"variant_name": "A0_time48_baseline"})])
    feature_diff_targets = ["is_win", "is_big_win", "is_big_loss", "has_overlap_signal"]
    pre_diffs = pd.concat([build_feature_diff(feature_df, t, pre_feature_cols) for t in feature_diff_targets], ignore_index=True) if pre_feature_cols else pd.DataFrame()
    post_diffs = pd.concat([build_feature_diff(feature_df, t, post_feature_cols) for t in feature_diff_targets], ignore_index=True) if post_feature_cols else pd.DataFrame()
    correlations = build_correlations(feature_df, all_feature_cols)
    pre_rules = build_single_feature_rules(feature_df, pre_feature_cols, feature_diff_targets, quantiles, int(args.min_rule_selected))
    post_rules = build_single_feature_rules(feature_df, post_feature_cols, feature_diff_targets, quantiles, int(args.min_rule_selected)) if post_feature_cols else pd.DataFrame()
    session_breakdown = build_session_breakdown(feature_df)
    meta = build_meta(args, bars, events, trades, feature_df)

    path_rows = pd.DataFrame()
    if bool(int(args.save_cvd_path)):
        print("[path] building long-form CVD path rows", flush=True)
        path_rows = build_cvd_path_rows(arr, feature_df, int(args.path_offset_left), int(args.path_offset_right))

    # Keep the familiar filename as requested in earlier Low Sweep reports.
    sample_cols = [
        "signal_time", "entry_time", "exit_time", "net_return_on_equity", "mae_on_equity", "mfe_on_equity",
        "is_win", "is_big_win", "is_big_loss", "has_overlap_signal", "overlap_signal_count",
        "session_bucket", "atr_pct", "down_spike_pct", "large_trade_share",
        "signal_delta_pressure", "signal_close_pos_in_bar", "signal_spike_absorption_proxy",
        "pre_w3_delta_pressure", "pre_w5_delta_pressure", "pre_w10_delta_pressure",
        "pre_w5_price_lower_low_cvd_higher_low", "swing_price_lower_low_cvd_higher_low",
        "swing_signal_delta_pressure", "pre_downleg_lookback60_delta_pressure",
    ]
    sample_cols = [c for c in sample_cols if c in feature_df.columns]
    trade_sample = feature_df[sample_cols].copy() if sample_cols else feature_df.copy()

    write_csv(baseline_summary, out_dir / "00_baseline_summary.csv", "baseline_summary")
    write_csv(feature_df, out_dir / "01_parent_preentry_cvd_feature_matrix.csv", "feature_matrix")
    write_csv(pre_diffs, out_dir / "02_preentry_feature_diff_by_target.csv", "preentry_feature_diff")
    write_csv(correlations, out_dir / "03_feature_correlation_auc.csv", "feature_correlation_auc")
    write_csv(pre_rules, out_dir / "04_preentry_single_feature_rules.csv", "preentry_single_feature_rules")
    write_csv(post_diffs, out_dir / "05_post_signal_delayed_feature_diff.csv", "post_signal_feature_diff")
    write_csv(post_rules, out_dir / "06_post_signal_delayed_rules.csv", "post_signal_delayed_rules")
    write_csv(session_breakdown, out_dir / "07_session_breakdown.csv", "session_breakdown")
    write_csv(trade_sample, out_dir / "trade_sample.csv", "trade_sample")
    write_csv(path_rows, out_dir / "08_cvd_path_around_signal.csv", "cvd_path")
    write_csv(meta, out_dir / "99_meta.csv", "meta")

    print("[done] research report written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
