#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""01 basic research for causal panic selloff -> rejection -> recovery episodes.

Research questions
------------------
1. Is the orange observation node, yellow exhaustion node, or green recovery
   signal the better *tradable* entry stage after next-bar execution and costs?
2. Which fixed, causal market environments improve green-signal expectancy?
3. When green signals occur near each other, does a finite, exposure-capped
   scale-in plan improve the path versus one full entry?

This file deliberately does not declare a production strategy.  It is a first
research pass with a fixed feature set, train/holdout reporting, structural
stops, conservative same-bar handling, and no ordinary time exit.

Timing rules
------------
- Every node is generated from a closed bar by ``detect_panic_episodes``.
- Entry is always the next bar open (default delay=1).
- Environment features use only information available by the node bar.
- Diagnostics such as "start was the final episode low" are explicitly marked
  as forward-looking and are never allowed in filter masks.
- Scale-in adds only after another closed green signal and executes next open.
- If target and stop are touched in the same bar, stop wins.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.liquidity.panic_selloff_rejection_recovery_long.common.panic_episode import (  # noqa: E402
    PanicEpisodeConfig,
    detect_panic_episodes,
)
from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

try:  # Optional acceleration; deterministic Python fallback remains available.
    from numba import njit
except Exception:  # pragma: no cover - environments without numba
    njit = None


SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1H", "4H", "1D"}
STAGE_ORDER = {"start": 0, "exhaustion": 1, "signal": 2}


@dataclass(frozen=True)
class FixedFilter:
    name: str
    family: str
    description: str
    predicate: Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class ScaleScheme:
    name: str
    weights: tuple[float, ...]
    add_only_below_avg: bool = False

    @property
    def max_entries(self) -> int:
        return len(self.weights)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    mode: str  # "r" or "reference"
    value: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="01 panic recovery environment + clustered scale-in research",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", choices=sorted(SUPPORTED_TIMEFRAMES), default="1m")
    p.add_argument("--data-source", choices=["trade_bar", "ohlcv"], default="trade_bar")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--train-end-date", default="2024-12-31 23:59:59")
    p.add_argument(
        "--out-dir",
        default="data/reports/research/liquidity/panic_selloff_rejection_recovery_long/01_environment_and_cluster_scale_in",
    )

    # Shared episode detector. Defaults match analyze_tool v1.
    p.add_argument("--baseline-window", type=int, default=60)
    p.add_argument("--selloff-window", type=int, default=5)
    p.add_argument("--min-red-bars", type=int, default=3)
    p.add_argument("--observe-drop-pct", type=float, default=0.0045)
    p.add_argument("--observe-drop-vol-mult", type=float, default=2.5)
    p.add_argument("--observe-volume-ratio", type=float, default=1.10)
    p.add_argument("--panic-drop-pct", type=float, default=0.0075)
    p.add_argument("--panic-volume-ratio", type=float, default=1.35)
    p.add_argument("--stabilization-bars", type=int, default=2)
    p.add_argument("--min-rebound-from-low-pct", type=float, default=0.0020)
    p.add_argument("--pressure-decay-ratio", type=float, default=0.68)
    p.add_argument("--reclaim-fraction", type=float, default=0.35)
    p.add_argument("--breakout-lookback", type=int, default=2)
    p.add_argument("--max-episode-bars", type=int, default=30)
    p.add_argument("--cooldown-bars", type=int, default=8)

    # Event outcome / filter research.
    p.add_argument("--horizons", default="5,15,30,60,120,240")
    p.add_argument("--candidate-horizon", type=int, default=60)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--min-filter-train", type=int, default=80)
    p.add_argument("--min-filter-holdout", type=int, default=35)
    p.add_argument("--top-atomic-for-pairs", type=int, default=12)
    p.add_argument("--top-scale-filters", type=int, default=3)

    # Cost convention: fee 0.11% round trip + conservative slippage.
    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage-pct", type=float, default=0.00020)
    p.add_argument("--exit-slippage-pct", type=float, default=0.00020)
    p.add_argument("--cost-multipliers", default="1.0,2.0")

    # Clustered green-signal scale-in. Total scheme weight is capped at 1.0.
    p.add_argument("--cluster-gap-bars", default="15,30,60")
    p.add_argument("--stop-buffer-pct", type=float, default=0.0005)
    p.add_argument("--target-r-list", default="0.75,1.0,1.5")
    p.add_argument("--save-trade-sample", type=int, default=30000)

    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def _parse_list(text: str, *, cast: Callable[[str], Any], name: str) -> list[Any]:
    values: list[Any] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = cast(token)
        if float(value) <= 0:
            raise ValueError(f"{name} must contain positive values")
        values.append(value)
    if not values:
        raise ValueError(f"{name} must not be empty")
    return sorted(set(values))


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    out = pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce").replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)


def load_bars(args: argparse.Namespace) -> pd.DataFrame:
    print(
        f"[load] source={args.data_source} {args.symbol} {args.timeframe} "
        f"{args.warmup_start_date}->{args.end_date}",
        flush=True,
    )
    if args.data_source == "trade_bar":
        df = OKXTradeBarLoader(symbol=args.symbol, timeframe=args.timeframe).fetch_data_by_date_range(
            args.warmup_start_date,
            args.end_date,
        )
    else:
        df = OKXDataLoader(symbol=args.symbol, timeframe=args.timeframe).fetch_data_by_date_range(
            args.warmup_start_date,
            args.end_date,
        )
    if df.empty:
        raise RuntimeError("No local bars loaded")
    out = df.copy().sort_index()
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")]
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"Loaded bars missing required columns: {missing}")
    for col in out.columns:
        if col in required or col in {
            "delta_notional",
            "delta_volume",
            "notional",
            "taker_buy_ratio",
            "large_delta_notional",
            "trades_count",
        }:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=required)
    print(f"       rows={len(out):,} range={out.index[0]} -> {out.index[-1]}", flush=True)
    return out


def build_context_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Build causal context available at each closed bar.

    ``pre_*`` columns are explicitly shifted so they describe the market before
    the current node bar. Current closed-bar episode features are attached from
    the shared detector separately.
    """
    out = bars.copy().sort_index()
    prev_close = out["close"].shift(1)
    ret = out["close"].pct_change()
    for window in (15, 60, 240):
        out[f"pre_ret_{window}"] = prev_close / out["close"].shift(window + 1) - 1.0
        abs_path = ret.abs().shift(1).rolling(window, min_periods=max(5, window // 3)).sum()
        out[f"pre_efficiency_{window}"] = _safe_divide(out[f"pre_ret_{window}"].abs(), abs_path)

    tr = _true_range(out)
    out["pre_atr_pct_60"] = _safe_divide(
        tr.shift(1).rolling(60, min_periods=20).mean(),
        prev_close,
    )
    atr_regime_base = out["pre_atr_pct_60"].shift(1).rolling(1440, min_periods=240).median()
    out["pre_atr_regime_ratio"] = _safe_divide(out["pre_atr_pct_60"], atr_regime_base)

    prior_high_240 = out["high"].shift(1).rolling(240, min_periods=80).max()
    prior_low_240 = out["low"].shift(1).rolling(240, min_periods=80).min()
    out["pre_range_pos_240"] = _safe_divide(prev_close - prior_low_240, prior_high_240 - prior_low_240).clip(0.0, 1.0)
    out["pre_drawdown_240"] = _safe_divide(prev_close, prior_high_240) - 1.0

    volume_base = out["volume"].shift(1).rolling(240, min_periods=80).median()
    out["pre_volume_ratio"] = _safe_divide(out["volume"].shift(1), volume_base)
    out["hour"] = out.index.hour
    out["weekday"] = out.index.dayofweek
    out["session"] = pd.cut(
        out["hour"],
        bins=[-1, 7, 15, 23],
        labels=["S0_00_07", "S1_08_15", "S2_16_23"],
    ).astype("object")
    return out


def episode_config(args: argparse.Namespace, horizons: tuple[int, ...]) -> PanicEpisodeConfig:
    return PanicEpisodeConfig(
        baseline_window=int(args.baseline_window),
        selloff_window=int(args.selloff_window),
        min_red_bars=int(args.min_red_bars),
        observe_drop_pct=float(args.observe_drop_pct),
        observe_drop_vol_mult=float(args.observe_drop_vol_mult),
        observe_volume_ratio=float(args.observe_volume_ratio),
        panic_drop_pct=float(args.panic_drop_pct),
        panic_volume_ratio=float(args.panic_volume_ratio),
        stabilization_bars=int(args.stabilization_bars),
        min_rebound_from_low_pct=float(args.min_rebound_from_low_pct),
        pressure_decay_ratio=float(args.pressure_decay_ratio),
        reclaim_fraction=float(args.reclaim_fraction),
        breakout_lookback=int(args.breakout_lookback),
        max_episode_bars=int(args.max_episode_bars),
        cooldown_bars=int(args.cooldown_bars),
        outcome_horizons=horizons,
    )


def _at(frame: pd.DataFrame, ts: pd.Timestamp, col: str, default: Any = np.nan) -> Any:
    if col not in frame.columns or ts not in frame.index:
        return default
    value = frame.at[ts, col]
    return value


def build_stage_events(
    bars: pd.DataFrame,
    context: pd.DataFrame,
    args: argparse.Namespace,
    horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[episodes] detecting causal multi-bar episodes", flush=True)
    result = detect_panic_episodes(bars, episode_config(args, horizons))
    features = result.feature_frame
    rows: list[dict[str, Any]] = []
    for ep in result.episodes:
        node_by_kind = {node.kind: node for node in ep.nodes}
        start = node_by_kind.get("start")
        acceleration = node_by_kind.get("acceleration")
        exhaustion = node_by_kind.get("exhaustion")
        signal = node_by_kind.get("signal")

        start_ts = pd.Timestamp(ep.start_time)
        end_ts = pd.Timestamp(ep.end_time)
        window = features.loc[start_ts:end_ts]
        max_volume = float(window["volume_ratio"].max()) if not window.empty else np.nan
        max_pressure = float(window["sell_pressure_score"].max()) if not window.empty else np.nan
        min_flow = float(window["flow_pressure"].min()) if "flow_pressure" in window.columns and not window.empty else np.nan
        start_low = float(features.at[start_ts, "low"]) if start_ts in features.index else np.nan
        start_to_final_low = start_low / float(ep.episode_low) - 1.0 if ep.episode_low > 0 and np.isfinite(start_low) else np.nan
        start_was_near_final_low = bool(np.isfinite(start_to_final_low) and start_to_final_low <= 0.0010)

        shared = {
            "episode_id": int(ep.episode_id),
            "episode_status": ep.status,
            "episode_start_time": start_ts,
            "episode_end_time": end_ts,
            "reference_price": float(ep.reference_price),
            "episode_low": float(ep.episode_low),
            "signal_time": pd.Timestamp(ep.signal_time) if ep.signal_time is not None else pd.NaT,
            "signal_price": float(ep.signal_price) if ep.signal_price is not None else np.nan,
            "episode_bars": int(ep.fields.get("bars", len(window))),
            "episode_drop": float(ep.episode_low / ep.reference_price - 1.0),
            "recovery_to_signal": float(ep.fields.get("recovery_to_signal", np.nan)),
            "episode_max_volume_ratio": max_volume,
            "episode_max_pressure": max_pressure,
            "episode_min_flow_pressure": min_flow,
            # Forward episode diagnostics; never filter inputs.
            "diagnostic_start_to_final_low": start_to_final_low,
            "diagnostic_start_was_near_final_low": start_was_near_final_low,
            "has_green_signal": ep.signal_time is not None,
        }

        for stage, node in (("start", start), ("exhaustion", exhaustion), ("signal", signal)):
            if node is None:
                continue
            ts = pd.Timestamp(node.timestamp)
            rec = {
                **shared,
                "stage": stage,
                "stage_order": STAGE_ORDER[stage],
                "event_time": ts,
                "node_price": float(node.price) if node.price is not None else np.nan,
                "node_label": node.label,
            }
            for key, value in node.fields.items():
                if key not in rec:
                    rec[key] = value
            for col in (
                "pre_ret_15",
                "pre_ret_60",
                "pre_ret_240",
                "pre_efficiency_60",
                "pre_efficiency_240",
                "pre_atr_pct_60",
                "pre_atr_regime_ratio",
                "pre_range_pos_240",
                "pre_drawdown_240",
                "pre_volume_ratio",
                "hour",
                "weekday",
                "session",
            ):
                rec[col] = _at(context, ts, col)
            rec["stage_close"] = _at(bars, ts, "close")
            rec["stage_low"] = _at(bars, ts, "low")
            rec["stage_high"] = _at(bars, ts, "high")
            rows.append(rec)

    events = pd.DataFrame(rows)
    if events.empty:
        return events, features
    events["event_time"] = pd.to_datetime(events["event_time"])
    events["episode_start_time"] = pd.to_datetime(events["episode_start_time"])
    events["episode_end_time"] = pd.to_datetime(events["episode_end_time"])
    events = events.sort_values(["event_time", "episode_id", "stage_order"]).reset_index(drop=True)
    research_start = pd.Timestamp(args.start_date)
    research_end = pd.Timestamp(args.end_date) + pd.Timedelta(days=1)
    events = events[(events["event_time"] >= research_start) & (events["event_time"] < research_end)].copy()

    signals = events[events["stage"] == "signal"].copy().sort_values("event_time")
    if not signals.empty:
        signals["prev_signal_gap_bars"] = _timestamp_gap_in_bars(
            signals["event_time"],
            bars.index,
        )
        events = events.merge(
            signals[["episode_id", "prev_signal_gap_bars"]],
            on="episode_id",
            how="left",
            suffixes=("", "_signal"),
        )
    else:
        events["prev_signal_gap_bars"] = np.nan
    return events.reset_index(drop=True), features


def _timestamp_gap_in_bars(times: pd.Series, bar_index: pd.DatetimeIndex) -> np.ndarray:
    pos = bar_index.get_indexer(pd.DatetimeIndex(times))
    gaps = np.full(len(pos), np.nan, dtype=float)
    if len(pos) > 1:
        delta = pos[1:] - pos[:-1]
        gaps[1:] = np.where((pos[1:] >= 0) & (pos[:-1] >= 0), delta, np.nan)
    return gaps


def attach_next_open_outcomes(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    args: argparse.Namespace,
    horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return events.copy(), pd.DataFrame()
    out = events.copy()
    idx = bars.index
    event_pos = idx.get_indexer(pd.DatetimeIndex(out["event_time"]))
    entry_pos = event_pos + int(args.entry_delay_bars)
    valid_entry = (event_pos >= 0) & (entry_pos >= 0) & (entry_pos < len(bars))
    out["event_pos"] = event_pos
    out["entry_pos"] = entry_pos
    out["expected_entry_time"] = pd.NaT
    out.loc[valid_entry, "expected_entry_time"] = idx[entry_pos[valid_entry]].to_numpy()
    out["entry_open_raw"] = np.nan
    out.loc[valid_entry, "entry_open_raw"] = bars["open"].to_numpy(dtype=float)[entry_pos[valid_entry]]

    entry_cost_mult = (1.0 + float(args.entry_slippage_pct)) * (1.0 + float(args.entry_fee_rate))
    exit_cost_mult = (1.0 - float(args.exit_slippage_pct)) * (1.0 - float(args.exit_fee_rate))
    out["entry_cash_price"] = out["entry_open_raw"] * entry_cost_mult

    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)
    opens = bars["open"].to_numpy(dtype=float)
    for horizon in horizons:
        net = np.full(len(out), np.nan, dtype=float)
        gross = np.full(len(out), np.nan, dtype=float)
        mfe = np.full(len(out), np.nan, dtype=float)
        mae = np.full(len(out), np.nan, dtype=float)
        end_pos = entry_pos + int(horizon)
        valid = valid_entry & (end_pos < len(bars))
        valid_indices = np.flatnonzero(valid)
        for i in valid_indices:
            ep = int(entry_pos[i])
            hp = int(end_pos[i])
            raw_entry = opens[ep]
            exit_cash = closes[hp] * exit_cost_mult
            entry_cash = raw_entry * entry_cost_mult
            gross[i] = closes[hp] / raw_entry - 1.0
            net[i] = exit_cash / entry_cash - 1.0
            path_high = float(np.max(highs[ep : hp + 1]))
            path_low = float(np.min(lows[ep : hp + 1]))
            mfe[i] = path_high / raw_entry - 1.0
            mae[i] = path_low / raw_entry - 1.0
        out[f"ret_h{horizon}_gross"] = gross
        out[f"ret_h{horizon}_net"] = net
        out[f"mfe_h{horizon}"] = mfe
        out[f"mae_h{horizon}"] = mae

    audit = pd.DataFrame(
        {
            "episode_id": out["episode_id"],
            "stage": out["stage"],
            "event_time": out["event_time"],
            "entry_time": out["expected_entry_time"],
            "entry_is_after_event": pd.to_datetime(out["expected_entry_time"]) > pd.to_datetime(out["event_time"]),
            "entry_delay_bars": int(args.entry_delay_bars),
            "event_index_found": event_pos >= 0,
            "entry_index_valid": valid_entry,
            "forward_diagnostic_excluded_from_filters": True,
        }
    )
    return out, audit


def _profit_factor(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return np.nan
    profit = float(x[x > 0].sum())
    loss = float(-x[x < 0].sum())
    if loss <= 0:
        return np.inf if profit > 0 else np.nan
    return profit / loss


def _max_drawdown(returns: pd.Series) -> float:
    x = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return np.nan
    equity = np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def _summary_row(part: pd.DataFrame, return_col: str) -> dict[str, Any]:
    x = pd.to_numeric(part.get(return_col, pd.Series(dtype=float)), errors="coerce").dropna()
    if x.empty:
        return {
            "count": 0,
            "mean_net": np.nan,
            "median_net": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "total_compound": np.nan,
            "max_drawdown": np.nan,
        }
    return {
        "count": int(len(x)),
        "mean_net": float(x.mean()),
        "median_net": float(x.median()),
        "win_rate": float((x > 0).mean()),
        "profit_factor": _profit_factor(x),
        "total_compound": float(np.prod(1.0 + x.to_numpy(dtype=float)) - 1.0),
        "max_drawdown": _max_drawdown(x),
    }


def summarize_stage_entries(events: pd.DataFrame, horizons: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    for stage, part in events.groupby("stage", sort=False):
        for horizon in horizons:
            col = f"ret_h{horizon}_net"
            row = {"stage": stage, "horizon": horizon, **_summary_row(part, col)}
            if f"mfe_h{horizon}" in part.columns:
                row["median_mfe"] = float(pd.to_numeric(part[f"mfe_h{horizon}"], errors="coerce").median())
                row["median_mae"] = float(pd.to_numeric(part[f"mae_h{horizon}"], errors="coerce").median())
            rows.append(row)
            for year, yp in part.groupby(pd.to_datetime(part["event_time"]).dt.year):
                yearly_rows.append({"stage": stage, "horizon": horizon, "year": int(year), **_summary_row(yp, col)})
    overview = pd.DataFrame(rows).sort_values(["horizon", "stage"]).reset_index(drop=True)
    yearly = pd.DataFrame(yearly_rows).sort_values(["stage", "horizon", "year"]).reset_index(drop=True)
    return overview, yearly


def build_fixed_filters(signals: pd.DataFrame) -> list[FixedFilter]:
    """Predefined causal filters only; no full-sample qcut thresholds."""
    return [
        FixedFilter("pre60_down_05", "trend", "signal前60 bars跌幅 <= -0.5%", lambda d: d["pre_ret_60"] <= -0.005),
        FixedFilter("pre60_down_10", "trend", "signal前60 bars跌幅 <= -1.0%", lambda d: d["pre_ret_60"] <= -0.010),
        FixedFilter("pre240_not_deep_bear", "trend", "signal前240 bars跌幅 > -4%", lambda d: d["pre_ret_240"] > -0.040),
        FixedFilter("range240_low20", "location", "信号前价格处于240 bars区间底部20%", lambda d: d["pre_range_pos_240"] <= 0.20),
        FixedFilter("range240_low35", "location", "信号前价格处于240 bars区间底部35%", lambda d: d["pre_range_pos_240"] <= 0.35),
        FixedFilter("drawdown240_ge_15", "location", "距240 bars高点回撤至少1.5%", lambda d: d["pre_drawdown_240"] <= -0.015),
        FixedFilter("vol_regime_high", "volatility", "历史波动率高于长期中位数1.2倍", lambda d: d["pre_atr_regime_ratio"] >= 1.20),
        FixedFilter("vol_regime_not_extreme", "volatility", "历史波动率低于长期中位数2.5倍", lambda d: d["pre_atr_regime_ratio"] < 2.50),
        FixedFilter("episode_drop_deep10", "severity", "episode最低点跌幅至少1.0%", lambda d: d["episode_drop"] <= -0.010),
        FixedFilter("episode_drop_deep15", "severity", "episode最低点跌幅至少1.5%", lambda d: d["episode_drop"] <= -0.015),
        FixedFilter("episode_volume_ge_18", "severity", "episode最大成交量倍数 >= 1.8", lambda d: d["episode_max_volume_ratio"] >= 1.80),
        FixedFilter("episode_pressure_ge_8", "severity", "episode最大卖压评分 >= 8", lambda d: d["episode_max_pressure"] >= 8.0),
        FixedFilter("flow_capitulation", "orderflow", "episode内flow pressure <= -0.20", lambda d: d["episode_min_flow_pressure"] <= -0.20),
        FixedFilter("signal_fast_from_low_3", "timing", "低点后3 bars内出现绿灯", lambda d: pd.to_numeric(d["bars_from_low"], errors="coerce") <= 3),
        FixedFilter("signal_fast_from_low_5", "timing", "低点后5 bars内出现绿灯", lambda d: pd.to_numeric(d["bars_from_low"], errors="coerce") <= 5),
        FixedFilter("recovery_not_consumed50", "timing", "绿灯前收复不超过完整跌幅50%", lambda d: _recovery_consumed(d) <= 0.50),
        FixedFilter("recovery_not_consumed70", "timing", "绿灯前收复不超过完整跌幅70%", lambda d: _recovery_consumed(d) <= 0.70),
        FixedFilter("repeat_green_30", "clustering", "距离上一绿灯不超过30 bars", lambda d: d["prev_signal_gap_bars"] <= 30),
        FixedFilter("repeat_green_60", "clustering", "距离上一绿灯不超过60 bars", lambda d: d["prev_signal_gap_bars"] <= 60),
        FixedFilter("quiet_before_120", "clustering", "上一绿灯距离超过120 bars或不存在", lambda d: d["prev_signal_gap_bars"].isna() | (d["prev_signal_gap_bars"] > 120)),
        FixedFilter("session_00_07", "session", "UTC+8 00:00-07:59", lambda d: d["session"] == "S0_00_07"),
        FixedFilter("session_08_15", "session", "UTC+8 08:00-15:59", lambda d: d["session"] == "S1_08_15"),
        FixedFilter("session_16_23", "session", "UTC+8 16:00-23:59", lambda d: d["session"] == "S2_16_23"),
    ]


def _recovery_consumed(df: pd.DataFrame) -> pd.Series:
    recovery = pd.to_numeric(df["signal_price"], errors="coerce") - pd.to_numeric(df["episode_low"], errors="coerce")
    full = pd.to_numeric(df["reference_price"], errors="coerce") - pd.to_numeric(df["episode_low"], errors="coerce")
    return _safe_divide(recovery, full)


def add_filter_columns(signals: pd.DataFrame, specs: list[FixedFilter]) -> pd.DataFrame:
    out = signals.copy()
    out["recovery_consumed_fraction"] = _recovery_consumed(out)
    for spec in specs:
        mask = spec.predicate(out)
        out[f"filter__{spec.name}"] = mask.fillna(False).astype(bool)
    return out


def _split_stats(part: pd.DataFrame, return_col: str, train_end: pd.Timestamp) -> dict[str, Any]:
    train = part[pd.to_datetime(part["event_time"]) <= train_end]
    holdout = part[pd.to_datetime(part["event_time"]) > train_end]
    all_row = _summary_row(part, return_col)
    train_row = _summary_row(train, return_col)
    holdout_row = _summary_row(holdout, return_col)
    years = []
    for year, yp in part.groupby(pd.to_datetime(part["event_time"]).dt.year):
        yr = _summary_row(yp, return_col)
        years.append((int(year), int(yr["count"]), float(yr["mean_net"]) if np.isfinite(yr["mean_net"]) else np.nan))
    valid_years = [x for x in years if x[1] >= 10 and np.isfinite(x[2])]
    return {
        **{f"all_{k}": v for k, v in all_row.items()},
        **{f"train_{k}": v for k, v in train_row.items()},
        **{f"holdout_{k}": v for k, v in holdout_row.items()},
        "positive_years": int(sum(mean > 0 for _, _, mean in valid_years)),
        "valid_years": int(len(valid_years)),
        "worst_year_mean": float(min((mean for _, _, mean in valid_years), default=np.nan)),
        "year_detail": ";".join(f"{y}:{n}:{m:.6f}" for y, n, m in years if np.isfinite(m)),
    }


def evaluate_environment_filters(
    signals: pd.DataFrame,
    specs: list[FixedFilter],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return_col = f"ret_h{int(args.candidate_horizon)}_net"
    if return_col not in signals.columns:
        raise RuntimeError(f"candidate horizon missing: {return_col}")
    train_end = pd.Timestamp(args.train_end_date)
    atomic_rows: list[dict[str, Any]] = []
    for spec in specs:
        part = signals[signals[f"filter__{spec.name}"]]
        row = {
            "filter_name": spec.name,
            "family": spec.family,
            "description": spec.description,
            **_split_stats(part, return_col, train_end),
        }
        row["train_score"] = _selection_score(row, prefix="train")
        row["holdout_pass"] = _holdout_pass(row, args)
        atomic_rows.append(row)
    atomic = pd.DataFrame(atomic_rows)
    if not atomic.empty:
        atomic = atomic.sort_values(["holdout_pass", "train_score", "train_count"], ascending=[False, False, False]).reset_index(drop=True)

    eligible_atomic = atomic[
        (atomic["train_count"] >= int(args.min_filter_train))
        & np.isfinite(pd.to_numeric(atomic["train_score"], errors="coerce"))
    ].head(int(args.top_atomic_for_pairs))
    spec_map = {spec.name: spec for spec in specs}
    pair_rows: list[dict[str, Any]] = []
    names = eligible_atomic["filter_name"].tolist()
    for left, right in itertools.combinations(names, 2):
        if spec_map[left].family == spec_map[right].family:
            continue
        mask = signals[f"filter__{left}"] & signals[f"filter__{right}"]
        part = signals[mask]
        row = {
            "filter_name": f"{left}&{right}",
            "left_filter": left,
            "right_filter": right,
            "family": f"{spec_map[left].family}+{spec_map[right].family}",
            "description": f"{spec_map[left].description} AND {spec_map[right].description}",
            **_split_stats(part, return_col, train_end),
        }
        row["train_score"] = _selection_score(row, prefix="train")
        row["holdout_pass"] = _holdout_pass(row, args)
        pair_rows.append(row)
    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        pairs = pairs.sort_values(["holdout_pass", "train_score", "train_count"], ascending=[False, False, False]).reset_index(drop=True)

    candidates = build_scale_filter_candidates(atomic, pairs, args)
    return atomic, pairs, candidates


def _selection_score(row: dict[str, Any] | pd.Series, *, prefix: str) -> float:
    count = float(row.get(f"{prefix}_count", 0) or 0)
    mean = float(row.get(f"{prefix}_mean_net", np.nan))
    pf = float(row.get(f"{prefix}_profit_factor", np.nan))
    if count <= 0 or not np.isfinite(mean) or not np.isfinite(pf):
        return np.nan
    pf_component = min(max(pf - 1.0, -1.0), 2.0)
    return float(mean * math.sqrt(count) + 0.0005 * pf_component * math.sqrt(count))


def _holdout_pass(row: dict[str, Any] | pd.Series, args: argparse.Namespace) -> bool:
    return bool(
        float(row.get("train_count", 0) or 0) >= int(args.min_filter_train)
        and float(row.get("holdout_count", 0) or 0) >= int(args.min_filter_holdout)
        and float(row.get("train_mean_net", -np.inf)) > 0
        and float(row.get("holdout_mean_net", -np.inf)) > 0
        and float(row.get("train_profit_factor", 0)) > 1.0
        and float(row.get("holdout_profit_factor", 0)) > 1.0
    )


def build_scale_filter_candidates(atomic: pd.DataFrame, pairs: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = [
        {
            "candidate_name": "ALL_GREEN",
            "filter_expression": "ALL",
            "source": "baseline",
            "holdout_pass": True,
            "train_score": np.nan,
        }
    ]
    combined = pd.concat(
        [
            atomic.assign(source="atomic") if not atomic.empty else pd.DataFrame(),
            pairs.assign(source="pair") if not pairs.empty else pd.DataFrame(),
        ],
        ignore_index=True,
        sort=False,
    )
    if not combined.empty:
        passed = combined[combined["holdout_pass"] == True].copy()  # noqa: E712
        if passed.empty:
            passed = combined[
                (combined["train_count"] >= int(args.min_filter_train))
                & (combined["holdout_count"] >= int(args.min_filter_holdout))
            ].copy()
            passed["source"] = passed["source"].astype(str) + "_exploratory_no_holdout_pass"
        passed = passed.sort_values(["holdout_pass", "train_score"], ascending=[False, False]).head(int(args.top_scale_filters))
        for _, row in passed.iterrows():
            rows.append(
                {
                    "candidate_name": str(row["filter_name"]),
                    "filter_expression": str(row["filter_name"]),
                    "source": str(row["source"]),
                    "holdout_pass": bool(row["holdout_pass"]),
                    "train_score": float(row["train_score"]) if np.isfinite(row["train_score"]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def candidate_mask(signals: pd.DataFrame, expression: str) -> pd.Series:
    if expression == "ALL":
        return pd.Series(True, index=signals.index)
    parts = expression.split("&")
    mask = pd.Series(True, index=signals.index)
    for part in parts:
        col = f"filter__{part}"
        if col not in signals.columns:
            raise KeyError(f"Missing filter column for candidate: {col}")
        mask &= signals[col]
    return mask


def scale_schemes() -> list[ScaleScheme]:
    return [
        ScaleScheme("single_full", (1.0,), False),
        ScaleScheme("cluster_50_50", (0.5, 0.5), False),
        ScaleScheme("cluster_50_50_below_avg", (0.5, 0.5), True),
        ScaleScheme("cluster_thirds", (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), False),
        ScaleScheme("cluster_thirds_below_avg", (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), True),
    ]


def target_specs(args: argparse.Namespace) -> list[TargetSpec]:
    targets = [TargetSpec(f"target_{r:g}R", "r", float(r)) for r in _parse_list(args.target_r_list, cast=float, name="target_r_list")]
    targets.append(TargetSpec("target_reference", "reference", 0.0))
    return targets


def _build_signal_arrays(signals: pd.DataFrame, bars: pd.DataFrame) -> dict[str, np.ndarray]:
    frame = signals.sort_values("event_time").reset_index(drop=True)
    pos = bars.index.get_indexer(pd.DatetimeIndex(frame["event_time"]))
    return {
        "signal_pos": pos.astype(np.int64),
        "episode_low": pd.to_numeric(frame["episode_low"], errors="coerce").to_numpy(dtype=float),
        "reference": pd.to_numeric(frame["reference_price"], errors="coerce").to_numpy(dtype=float),
        "episode_id": pd.to_numeric(frame["episode_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64),
    }


def _simulate_cluster_python(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    signal_pos: np.ndarray,
    episode_low: np.ndarray,
    reference: np.ndarray,
    initial_eligible: np.ndarray,
    weights: np.ndarray,
    max_entries: int,
    gap_bars: int,
    delay_bars: int,
    stop_buffer_pct: float,
    target_mode: int,
    target_value: float,
    add_only_below_avg: int,
    entry_fee: float,
    exit_fee: float,
    entry_slippage: float,
    exit_slippage: float,
) -> tuple[np.ndarray, ...]:
    n_sig = len(signal_pos)
    max_trades = n_sig
    entry_signal_idx = np.full(max_trades, -1, dtype=np.int64)
    entry_bar = np.full(max_trades, -1, dtype=np.int64)
    exit_bar = np.full(max_trades, -1, dtype=np.int64)
    entry_count = np.zeros(max_trades, dtype=np.int64)
    filled_weight_arr = np.zeros(max_trades, dtype=np.float64)
    avg_entry_raw_arr = np.full(max_trades, np.nan, dtype=np.float64)
    stop_arr = np.full(max_trades, np.nan, dtype=np.float64)
    target_arr = np.full(max_trades, np.nan, dtype=np.float64)
    ret_max_arr = np.full(max_trades, np.nan, dtype=np.float64)
    ret_deployed_arr = np.full(max_trades, np.nan, dtype=np.float64)
    hold_arr = np.zeros(max_trades, dtype=np.int64)
    reason_arr = np.zeros(max_trades, dtype=np.int64)  # 1 target, 2 stop, 3 eod

    trade_n = 0
    k = 0
    last_exit = -1
    while k < n_sig:
        while k < n_sig and (initial_eligible[k] == 0 or signal_pos[k] + delay_bars <= last_exit):
            k += 1
        if k >= n_sig:
            break
        ep = signal_pos[k] + delay_bars
        if signal_pos[k] < 0 or ep < 0 or ep >= len(opens):
            k += 1
            continue

        legs = 1
        filled_weight = weights[0]
        raw_cost_sum = weights[0] * opens[ep]
        cash_cost_sum = weights[0] * opens[ep] * (1.0 + entry_slippage) * (1.0 + entry_fee)
        ref_sum = weights[0] * reference[k]
        stop = episode_low[k] * (1.0 - stop_buffer_pct)
        last_added_signal_pos = signal_pos[k]
        add_idx = k + 1
        cluster_open = True
        exit_p = len(opens) - 1
        exit_reason = 3
        target = np.nan

        p = ep
        while p < len(opens):
            while add_idx < n_sig and signal_pos[add_idx] + delay_bars < p:
                add_idx += 1
            if add_idx < n_sig and signal_pos[add_idx] + delay_bars == p and cluster_open:
                gap = signal_pos[add_idx] - last_added_signal_pos
                if gap <= gap_bars and legs < max_entries:
                    new_price = opens[p]
                    current_avg = raw_cost_sum / max(filled_weight, 1e-12)
                    if add_only_below_avg == 0 or new_price <= current_avg:
                        w = weights[legs]
                        filled_weight += w
                        raw_cost_sum += w * new_price
                        cash_cost_sum += w * new_price * (1.0 + entry_slippage) * (1.0 + entry_fee)
                        ref_sum += w * reference[add_idx]
                        if episode_low[add_idx] * (1.0 - stop_buffer_pct) < stop:
                            stop = episode_low[add_idx] * (1.0 - stop_buffer_pct)
                        legs += 1
                        last_added_signal_pos = signal_pos[add_idx]
                else:
                    cluster_open = False
                add_idx += 1

            avg_raw = raw_cost_sum / max(filled_weight, 1e-12)
            if target_mode == 0:
                risk = avg_raw - stop
                target = avg_raw + target_value * risk
            else:
                target = ref_sum / max(filled_weight, 1e-12)
                if target <= avg_raw:
                    target = avg_raw + 0.25 * max(avg_raw - stop, avg_raw * 1e-6)

            stop_hit = lows[p] <= stop
            target_hit = highs[p] >= target
            if stop_hit:
                exit_p = p
                exit_reason = 2
                break
            if target_hit:
                exit_p = p
                exit_reason = 1
                break
            p += 1

        avg_cash = cash_cost_sum / max(filled_weight, 1e-12)
        if exit_reason == 1:
            exit_market = target
        elif exit_reason == 2:
            exit_market = stop
        else:
            exit_market = closes[exit_p]
        exit_cash = exit_market * (1.0 - exit_slippage) * (1.0 - exit_fee)
        ret_deployed = exit_cash / avg_cash - 1.0
        ret_max = filled_weight * ret_deployed

        entry_signal_idx[trade_n] = k
        entry_bar[trade_n] = ep
        exit_bar[trade_n] = exit_p
        entry_count[trade_n] = legs
        filled_weight_arr[trade_n] = filled_weight
        avg_entry_raw_arr[trade_n] = raw_cost_sum / max(filled_weight, 1e-12)
        stop_arr[trade_n] = stop
        target_arr[trade_n] = target
        ret_max_arr[trade_n] = ret_max
        ret_deployed_arr[trade_n] = ret_deployed
        hold_arr[trade_n] = exit_p - ep + 1
        reason_arr[trade_n] = exit_reason
        trade_n += 1
        last_exit = exit_p
        while k < n_sig and signal_pos[k] <= last_exit:
            k += 1

    return (
        entry_signal_idx[:trade_n],
        entry_bar[:trade_n],
        exit_bar[:trade_n],
        entry_count[:trade_n],
        filled_weight_arr[:trade_n],
        avg_entry_raw_arr[:trade_n],
        stop_arr[:trade_n],
        target_arr[:trade_n],
        ret_max_arr[:trade_n],
        ret_deployed_arr[:trade_n],
        hold_arr[:trade_n],
        reason_arr[:trade_n],
    )


if njit is not None:  # Compile once per local run; cache avoids repeat compile.
    _simulate_cluster_fast = njit(cache=False)(_simulate_cluster_python)
else:  # pragma: no cover
    _simulate_cluster_fast = _simulate_cluster_python


def simulate_cluster_variants(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    candidates: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    signals = signals.sort_values("event_time").reset_index(drop=True)
    arrays = _build_signal_arrays(signals, bars)
    valid_signal = arrays["signal_pos"] >= 0
    opens = bars["open"].to_numpy(dtype=float)
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)

    schemes = scale_schemes()
    targets = target_specs(args)
    gaps = _parse_list(args.cluster_gap_bars, cast=int, name="cluster_gap_bars")
    cost_multipliers = _parse_list(args.cost_multipliers, cast=float, name="cost_multipliers")
    total_jobs = len(candidates) * len(schemes) * len(gaps) * len(targets) * len(cost_multipliers)
    progress = ProgressReporter(
        label="[cluster-scale] variants",
        total=total_jobs,
        every=max(1, int(args.progress_every)),
        enabled=not bool(args.no_progress),
    )
    trade_parts: list[pd.DataFrame] = []
    done = 0
    reason_map = {1: "target", 2: "structural_stop", 3: "end_of_data"}
    for _, candidate in candidates.iterrows():
        mask = candidate_mask(signals, str(candidate["filter_expression"])).to_numpy(dtype=bool) & valid_signal
        for scheme in schemes:
            weights = np.asarray(scheme.weights, dtype=float)
            if float(weights.sum()) > 1.0000001:
                raise ValueError(f"scheme exceeds total position cap: {scheme}")
            for gap in gaps:
                for target in targets:
                    target_mode = 0 if target.mode == "r" else 1
                    for cost_mult in cost_multipliers:
                        result = _simulate_cluster_fast(
                            opens,
                            highs,
                            lows,
                            closes,
                            arrays["signal_pos"],
                            arrays["episode_low"],
                            arrays["reference"],
                            mask.astype(np.int8),
                            weights,
                            int(scheme.max_entries),
                            int(gap),
                            int(args.entry_delay_bars),
                            float(args.stop_buffer_pct),
                            int(target_mode),
                            float(target.value),
                            int(scheme.add_only_below_avg),
                            float(args.entry_fee_rate) * float(cost_mult),
                            float(args.exit_fee_rate) * float(cost_mult),
                            float(args.entry_slippage_pct) * float(cost_mult),
                            float(args.exit_slippage_pct) * float(cost_mult),
                        )
                        if len(result[0]) > 0:
                            part = pd.DataFrame(
                                {
                                    "entry_signal_idx": result[0],
                                    "entry_bar_pos": result[1],
                                    "exit_bar_pos": result[2],
                                    "entry_count": result[3],
                                    "filled_weight": result[4],
                                    "avg_entry_raw": result[5],
                                    "stop_price": result[6],
                                    "target_price": result[7],
                                    "net_return_on_max_capital": result[8],
                                    "net_return_on_deployed_capital": result[9],
                                    "hold_bars": result[10],
                                    "exit_reason_code": result[11],
                                }
                            )
                            part["entry_time"] = bars.index[part["entry_bar_pos"].to_numpy(dtype=int)].to_numpy()
                            part["exit_time"] = bars.index[part["exit_bar_pos"].to_numpy(dtype=int)].to_numpy()
                            part["entry_episode_id"] = signals.iloc[part["entry_signal_idx"].to_numpy(dtype=int)]["episode_id"].to_numpy()
                            part["year"] = pd.to_datetime(part["entry_time"]).dt.year
                            part["exit_reason"] = part["exit_reason_code"].map(reason_map)
                            part["candidate_name"] = str(candidate["candidate_name"])
                            part["candidate_source"] = str(candidate["source"])
                            part["scheme"] = scheme.name
                            part["max_entries"] = scheme.max_entries
                            part["add_only_below_avg"] = scheme.add_only_below_avg
                            part["cluster_gap_bars"] = int(gap)
                            part["target_name"] = target.name
                            part["target_mode"] = target.mode
                            part["target_value"] = float(target.value)
                            part["cost_mult"] = float(cost_mult)
                            trade_parts.append(part)
                        done += 1
                        if done < total_jobs:
                            progress.update(done)
    progress.close()
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    if trades.empty:
        return trades, pd.DataFrame(), pd.DataFrame()
    group_cols = [
        "candidate_name",
        "candidate_source",
        "scheme",
        "max_entries",
        "add_only_below_avg",
        "cluster_gap_bars",
        "target_name",
        "target_mode",
        "target_value",
        "cost_mult",
    ]
    summary = summarize_cluster_trades(trades, group_cols)
    yearly = summarize_cluster_trades(trades, [*group_cols, "year"])
    return trades, summary, yearly


def summarize_cluster_trades(trades: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, part in trades.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        ret = pd.to_numeric(part["net_return_on_max_capital"], errors="coerce").dropna()
        deployed = pd.to_numeric(part["net_return_on_deployed_capital"], errors="coerce").dropna()
        row.update(
            {
                "trades": int(len(part)),
                "mean_net_on_max": float(ret.mean()) if not ret.empty else np.nan,
                "median_net_on_max": float(ret.median()) if not ret.empty else np.nan,
                "win_rate_on_max": float((ret > 0).mean()) if not ret.empty else np.nan,
                "profit_factor_on_max": _profit_factor(ret),
                "total_compound_on_max": float(np.prod(1.0 + ret.to_numpy()) - 1.0) if not ret.empty else np.nan,
                "max_drawdown_on_max": _max_drawdown(ret),
                "mean_net_deployed": float(deployed.mean()) if not deployed.empty else np.nan,
                "avg_entries": float(part["entry_count"].mean()),
                "add_rate": float((part["entry_count"] > 1).mean()),
                "avg_filled_weight": float(part["filled_weight"].mean()),
                "median_hold_bars": float(part["hold_bars"].median()),
                "p90_hold_bars": float(part["hold_bars"].quantile(0.90)),
                "target_exit_rate": float((part["exit_reason"] == "target").mean()),
                "stop_exit_rate": float((part["exit_reason"] == "structural_stop").mean()),
                "end_of_data_rate": float((part["exit_reason"] == "end_of_data").mean()),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [c for c in ("cost_mult", "mean_net_on_max", "trades") if c in out.columns]
    asc = [True, False, False][: len(sort_cols)]
    return out.sort_values(sort_cols, ascending=asc).reset_index(drop=True)


def build_cluster_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    base_keys = ["candidate_name", "cluster_gap_bars", "target_name", "cost_mult"]
    base = summary[summary["scheme"] == "single_full"]
    base = base[base_keys + [
        "mean_net_on_max",
        "profit_factor_on_max",
        "total_compound_on_max",
        "max_drawdown_on_max",
        "trades",
    ]].rename(
        columns={
            "mean_net_on_max": "baseline_mean_net",
            "profit_factor_on_max": "baseline_pf",
            "total_compound_on_max": "baseline_total_compound",
            "max_drawdown_on_max": "baseline_max_drawdown",
            "trades": "baseline_trades",
        }
    )
    out = summary.merge(base, on=base_keys, how="left")
    out["delta_mean_vs_single"] = out["mean_net_on_max"] - out["baseline_mean_net"]
    out["delta_pf_vs_single"] = out["profit_factor_on_max"] - out["baseline_pf"]
    out["delta_drawdown_vs_single"] = out["max_drawdown_on_max"] - out["baseline_max_drawdown"]
    out["scale_in_candidate"] = (
        (out["scheme"] != "single_full")
        & (out["trades"] >= 30)
        & (out["mean_net_on_max"] > 0)
        & (out["profit_factor_on_max"] > 1.0)
        & (out["delta_mean_vs_single"] > 0)
        & (out["delta_drawdown_vs_single"] >= -0.02)
    )
    return out.sort_values(["scale_in_candidate", "delta_mean_vs_single", "mean_net_on_max"], ascending=[False, False, False]).reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_research_summary(
    out_dir: Path,
    stage_overview: pd.DataFrame,
    atomic: pd.DataFrame,
    pairs: pd.DataFrame,
    cluster_compare: pd.DataFrame,
) -> None:
    lines = [
        "# 01 Panic Recovery Research Summary",
        "",
        "这是基础研究，不是可实盘结论。所有入场均为信号后下一根 open。",
        "",
        "## Stage comparison",
    ]
    if stage_overview.empty:
        lines.append("No stage events.")
    else:
        best = stage_overview.sort_values("mean_net", ascending=False).head(10)
        for _, r in best.iterrows():
            lines.append(
                f"- {r['stage']} h={int(r['horizon'])}: n={int(r['count'])}, "
                f"mean={r['mean_net']:.4%}, win={r['win_rate']:.2%}, PF={r['profit_factor']:.3f}"
            )
    lines.extend(["", "## Environment filters with holdout pass"])
    passed = pd.concat(
        [
            atomic[atomic.get("holdout_pass", False) == True] if not atomic.empty else pd.DataFrame(),  # noqa: E712
            pairs[pairs.get("holdout_pass", False) == True] if not pairs.empty else pd.DataFrame(),  # noqa: E712
        ],
        ignore_index=True,
        sort=False,
    )
    if passed.empty:
        lines.append("- None. Do not promote an environment filter from this run.")
    else:
        for _, r in passed.sort_values("train_score", ascending=False).head(10).iterrows():
            lines.append(
                f"- {r['filter_name']}: train n={int(r['train_count'])}, mean={r['train_mean_net']:.4%}, "
                f"holdout n={int(r['holdout_count'])}, mean={r['holdout_mean_net']:.4%}"
            )
    lines.extend(["", "## Cluster scale-in candidates"])
    candidates = cluster_compare[cluster_compare.get("scale_in_candidate", False) == True] if not cluster_compare.empty else pd.DataFrame()  # noqa: E712
    if candidates.empty:
        lines.append("- None. Repeated green signals did not improve the single-entry baseline under the basic rules.")
    else:
        for _, r in candidates.head(10).iterrows():
            lines.append(
                f"- {r['candidate_name']} / {r['scheme']} / gap={int(r['cluster_gap_bars'])} / {r['target_name']} / "
                f"cost={r['cost_mult']:.1f}x: n={int(r['trades'])}, mean={r['mean_net_on_max']:.4%}, "
                f"PF={r['profit_factor_on_max']:.3f}, add_rate={r['add_rate']:.2%}"
            )
    lines.extend(
        [
            "",
            "## Important cautions",
            "- `diagnostic_start_was_near_final_low` uses the final episode low and is visual/diagnostic only.",
            "- Pair filters are selected on the train segment and must be judged on holdout columns.",
            "- Scale-in exposure is capped at 100%; no unlimited martingale or doubling.",
            "- No ordinary time exit is used. Remaining open positions are closed only at end of data and reported separately.",
        ]
    )
    (out_dir / "12_RESEARCH_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_research(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = tuple(int(x) for x in _parse_list(args.horizons, cast=int, name="horizons"))
    if int(args.candidate_horizon) not in horizons:
        raise ValueError("candidate_horizon must be included in horizons")

    bars_all = load_bars(args)
    context_all = build_context_features(bars_all)
    stage_events, detector_features = build_stage_events(bars_all, context_all, args, horizons)
    if stage_events.empty:
        raise RuntimeError("No panic episode stages detected")
    stage_events, causal_audit = attach_next_open_outcomes(stage_events, bars_all, args, horizons)
    stage_overview, stage_yearly = summarize_stage_entries(stage_events, horizons)

    signals = stage_events[stage_events["stage"] == "signal"].copy().sort_values("event_time").reset_index(drop=True)
    specs = build_fixed_filters(signals)
    signals = add_filter_columns(signals, specs)
    atomic, pairs, scale_candidates = evaluate_environment_filters(signals, specs, args)

    trades, cluster_summary, cluster_yearly = simulate_cluster_variants(
        bars_all,
        signals,
        scale_candidates,
        args,
    )
    cluster_compare = build_cluster_comparison(cluster_summary)

    write_csv(stage_events, out_dir / "01_stage_events_with_outcomes.csv")
    write_csv(stage_overview, out_dir / "02_stage_entry_overview.csv")
    write_csv(stage_yearly, out_dir / "03_stage_entry_yearly.csv")
    write_csv(atomic, out_dir / "04_green_environment_atomic_train_holdout.csv")
    write_csv(pairs, out_dir / "05_green_environment_pairs_train_holdout.csv")
    write_csv(scale_candidates, out_dir / "06_scale_filter_candidates.csv")
    write_csv(cluster_summary, out_dir / "07_cluster_scale_in_summary.csv")
    write_csv(cluster_compare, out_dir / "08_cluster_scale_in_vs_single.csv")
    write_csv(cluster_yearly, out_dir / "09_cluster_scale_in_yearly.csv")
    trade_out = trades
    if int(args.save_trade_sample) > 0 and len(trades) > int(args.save_trade_sample):
        trade_out = trades.sort_values(["candidate_name", "scheme", "entry_time"]).head(int(args.save_trade_sample))
    write_csv(trade_out, out_dir / "10_cluster_scale_in_trades_sample.csv")
    write_csv(causal_audit, out_dir / "11_causal_audit.csv")
    write_research_summary(out_dir, stage_overview, atomic, pairs, cluster_compare)

    meta = {
        "script": "01_environment_and_cluster_scale_in_research.py",
        "research_family": "liquidity/panic_selloff_rejection_recovery_long",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "data_source": args.data_source,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "train_end_date": args.train_end_date,
        "bar_rows": int(len(bars_all)),
        "detector_feature_rows": int(len(detector_features)),
        "episode_count": int(stage_events["episode_id"].nunique()),
        "green_signal_count": int(len(signals)),
        "stage_event_count": int(len(stage_events)),
        "atomic_filters": int(len(atomic)),
        "pair_filters": int(len(pairs)),
        "scale_candidate_filters": scale_candidates.to_dict(orient="records"),
        "cluster_trade_rows": int(len(trades)),
        "cost_convention": {
            "round_trip_fee": float(args.entry_fee_rate + args.exit_fee_rate),
            "round_trip_slippage": float(args.entry_slippage_pct + args.exit_slippage_pct),
            "cost_multipliers": _parse_list(args.cost_multipliers, cast=float, name="cost_multipliers"),
        },
        "causal_guards": [
            "shared detector is a forward state machine using current/past closed bars only",
            "all stage entries execute next bar open",
            "pre-context features are shifted",
            "final episode low diagnostics are excluded from filter definitions",
            "same-bar stop and target collision is resolved stop-first",
            "scale-in total weight <= 1.0 and no unlimited averaging",
            "no ordinary time exit; end-of-data exits are separately reported",
        ],
        "params": vars(args),
    }
    write_json(out_dir / "00_manifest.json", meta)
    finalize_research_report(
        out_dir,
        title="01 Panic Selloff Rejection Recovery Environment and Cluster Scale-In",
        print_log=True,
    )
    print(f"[done] reports -> {out_dir}", flush=True)
    return meta


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
