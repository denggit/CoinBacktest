#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH 1m panic-wick candidate backtest V1.1: exit upgrade ladder.

Research-only candidate backtest. This script keeps the V1 shortlisted entry
set fixed and tests a small, named set of exit-upgrade hypotheses. It is not a
parameter grid search. The goal is to decide whether a condition should be
added only when it improves multiple dimensions versus the frozen structural
baseline.

Frozen baseline
---------------
- entry: strict_flow / strict_reclaim / priority_union
- exit: multi-sweep deeper-fail + higher-low trail

V1.1 exit upgrades
------------------
- fixed 1.0% take-profit as a control;
- decaying take-profit, e.g. 1.0% -> 0.8% -> 0.6% -> 0.4% -> 0.2%;
- the same decaying take-profit with 6h / 12h max-hold caps as safety exits;
- all versions retain the structural failure logic from the baseline.

Causal policy
-------------
- 1m bars are left-labeled by bar start time.
- Event bars are known only after they close.
- signal_time = event_bar_start + 1 minute.
- entry_time = next bar open, equal to signal_time for 1m left-labeled bars.
- Structural exit decisions use only closed 1m bars and execute at the next 1m open.
- Take-profit is modeled as a resting limit exit when the 1m high reaches the target.
- If structural failure and take-profit are both observed in the same 1m bar, the
  conservative structural/failure exit wins and the trade is flagged ambiguous.
- No multi-timeframe context is used in this script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "eth_1m_panic_wick_candidate_backtest_v1_1_exit_upgrade"
SCRIPT_VERSION = "1.1.0"
EXPERIMENT_ID = "ETH_MF_1M_PANIC_WICK_CANDIDATE_BACKTEST_V1_1_EXIT"
EDGE_ID = "RESEARCH_ONLY_ETH_MF_1M_PANIC_WICK_CANDIDATE_BT_V1_1_EXIT"
TITLE = "ETH 1m Panic Wick Structural Candidate Backtest V1.1 Exit Upgrade"
DEFAULT_OUT_DIR = "data/reports/research/eth_1m_panic_wick_candidate_backtest_v1_1_exit_upgrade"
BAR_DELTA = pd.Timedelta(minutes=1)

ENTRY_POLICIES: tuple[str, ...] = (
    "strict_flow",
    "strict_reclaim",
    "priority_union",
)

PRIMARY_ENTRY_EVENT = {
    "strict_flow": "panic_downtrend_neg_delta_absorption_long",
    "strict_reclaim": "panic_downtrend_lower_reclaim_long",
}

BASELINE_EXIT_MODE = "baseline_multi_sweep_higher_low_trail"

EXIT_MODES: tuple[str, ...] = (
    BASELINE_EXIT_MODE,
    "fixed_tp_1p0_multi_sweep",
    "decay_tp_1p0_to_0p2_multi_sweep",
    "decay_tp_0p8_to_0p2_multi_sweep",
    "decay_tp_1p0_to_0p2_multi_sweep_cap_360",
    "decay_tp_0p8_to_0p2_multi_sweep_cap_360",
    "decay_tp_1p0_to_0p2_multi_sweep_cap_720",
)

TIME_BENCHMARK_MODES: set[str] = set()


@dataclass(frozen=True)
class PanicThresholds:
    """Fixed V2 thresholds; not swept in candidate backtest."""

    wick_share_min: float = 0.50
    wick_atr_min: float = 0.55
    volume_ratio_min: float = 2.0
    reclaim_close_pos: float = 0.66
    soft_reclaim_close_pos: float = 0.55
    prior_flush_30_min: float = -0.005
    prior_flush_120_min: float = -0.010
    delta_absorption_max: float = -0.10
    taker_buy_absorption_max: float = 0.45
    stop_buffer_pct: float = 0.0015


@dataclass(frozen=True)
class ExitStateConfig:
    """Fixed structural exit controls. These are named hypotheses, not a grid."""

    sweep_fail_reclaim_bars_2: int = 2
    sweep_fail_reclaim_bars_3: int = 3
    dwell_below_low_bars: int = 3
    deeper_sweep_buffer_pct: float = 0.0015
    trail_confirm_left_bars: int = 1
    event_high_fail_closes: int = 1


@dataclass(frozen=True)
class TakeProfitConfig:
    """Named TP hypothesis. Percent values are decimal returns."""

    initial_pct: float
    min_pct: float
    decay_per_hour_pct: float = 0.0
    max_hold_bars: int | None = None
    fixed: bool = False



def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Candidate backtest V1.1 for ETH 1m panic-wick entries with exit-upgrade ladder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--no-build-missing", action="store_true")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--round-trip-cost-pct", type=float, default=0.0011)
    p.add_argument("--cost-multipliers", default="1.0,1.5,2.0,3.0")
    p.add_argument("--entry-delay-bars-list", default="0,1,2")
    p.add_argument("--min-trades", type=int, default=80)
    p.add_argument("--event-sample-size", type=int, default=5000)
    p.add_argument("--progress-every", type=int, default=2500)
    p.add_argument("--write-full-trades", action="store_true")

    # V2 fixed vocabulary thresholds. Keep defaults unless reproducing prior reports.
    p.add_argument("--wick-share-min", type=float, default=0.50)
    p.add_argument("--wick-atr-min", type=float, default=0.55)
    p.add_argument("--volume-ratio-min", type=float, default=2.0)
    p.add_argument("--reclaim-close-pos", type=float, default=0.66)
    p.add_argument("--soft-reclaim-close-pos", type=float, default=0.55)
    p.add_argument("--prior-flush-30-min", type=float, default=-0.005)
    p.add_argument("--prior-flush-120-min", type=float, default=-0.010)
    p.add_argument("--delta-absorption-max", type=float, default=-0.10)
    p.add_argument("--taker-buy-absorption-max", type=float, default=0.45)
    p.add_argument("--stop-buffer-pct", type=float, default=0.0015)
    return p.parse_args(argv)


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    vals: list[int] = []
    for part in str(raw).split(","):
        text = part.strip()
        if text:
            vals.append(int(text))
    out = tuple(dict.fromkeys(vals))
    if not out:
        raise ValueError("integer csv must not be empty")
    return out


def _parse_csv_floats(raw: str) -> tuple[float, ...]:
    vals: list[float] = []
    for part in str(raw).split(","):
        text = part.strip()
        if text:
            vals.append(float(text))
    out = tuple(dict.fromkeys(vals))
    if not out:
        raise ValueError("float csv must not be empty")
    return out


def _thresholds_from_args(args: argparse.Namespace) -> PanicThresholds:
    return PanicThresholds(
        wick_share_min=float(args.wick_share_min),
        wick_atr_min=float(args.wick_atr_min),
        volume_ratio_min=float(args.volume_ratio_min),
        reclaim_close_pos=float(args.reclaim_close_pos),
        soft_reclaim_close_pos=float(args.soft_reclaim_close_pos),
        prior_flush_30_min=float(args.prior_flush_30_min),
        prior_flush_120_min=float(args.prior_flush_120_min),
        delta_absorption_max=float(args.delta_absorption_max),
        taker_buy_absorption_max=float(args.taker_buy_absorption_max),
        stop_buffer_pct=float(args.stop_buffer_pct),
    )


def _research_window_mask(index: pd.DatetimeIndex, start_date: str, end_date: str) -> pd.Series:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if end_ts == end_ts.normalize() and len(str(end_date).strip()) <= 10:
        end_exclusive = end_ts + pd.Timedelta(days=1)
    else:
        end_exclusive = end_ts + BAR_DELTA
    return pd.Series((index >= start_ts) & (index < end_exclusive), index=index)


def _safe_divide(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> pd.Series:
    aa = pd.Series(a) if not isinstance(a, pd.Series) else pd.to_numeric(a, errors="coerce")
    bb = pd.Series(b) if not isinstance(b, pd.Series) else pd.to_numeric(b, errors="coerce")
    return (aa / bb.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


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


def _profit_factor(ret: pd.Series) -> float:
    x = pd.to_numeric(ret, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    gp = float(x[x > 0].sum())
    gl = float(-x[x <= 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else float("nan")
    return gp / gl


def _top5_winner_share(ret: pd.Series) -> float:
    x = pd.to_numeric(ret, errors="coerce").dropna()
    wins = x[x > 0].sort_values(ascending=False)
    gross = float(wins.sum())
    if gross <= 0 or wins.empty:
        return float("nan")
    return float(wins.head(5).sum()) / gross


def _max_days_without_event(times: pd.Series) -> float:
    ts = pd.to_datetime(times, errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return float("nan")
    return float(ts.diff().dropna().max() / pd.Timedelta(days=1))


def _events_per_month(times: pd.Series) -> float:
    ts = pd.to_datetime(times, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    span_days = max(1.0, float((ts.max() - ts.min()) / pd.Timedelta(days=1)))
    return float(len(ts) / span_days * 30.4375)


def _max_consecutive_losses(ret: pd.Series) -> int:
    losses = pd.to_numeric(ret, errors="coerce").fillna(0.0).to_numpy() <= 0
    best = cur = 0
    for is_loss in losses:
        if bool(is_loss):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _equity_stats(ret: pd.Series) -> dict[str, float]:
    r = pd.to_numeric(ret, errors="coerce").dropna()
    if r.empty:
        return {"total_return": np.nan, "max_drawdown": np.nan, "equity_end": np.nan}
    equity = (1.0 + r).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(dd.min()),
        "equity_end": float(equity.iloc[-1]),
    }


def session_label(index: pd.DatetimeIndex) -> np.ndarray:
    h = index.hour
    return np.select(
        [h < 8, (h >= 8) & (h < 16), h >= 16],
        ["asia_early", "asia_london", "eu_us"],
        default="unknown",
    )


def load_trade_bars(args: argparse.Namespace) -> pd.DataFrame:
    print(
        f"[load] OKXTradeBarLoader {args.symbol} {args.timeframe} "
        f"{args.warmup_start_date}->{args.end_date}",
        flush=True,
    )
    loader = OKXTradeBarLoader(
        symbol=args.symbol,
        timeframe=args.timeframe,
        data_dir=args.data_dir,
        db_name=args.db_name,
    )
    df = loader.fetch_data_by_date_range(
        args.warmup_start_date,
        args.end_date,
        chunksize=int(args.chunksize),
        force_rebuild=bool(args.force_rebuild),
        build_missing=not bool(args.no_build_missing),
    )
    if df.empty:
        raise RuntimeError(f"No trade bars loaded for {args.symbol} {args.timeframe}")
    out = df.copy().sort_index()
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"Loaded trade bars missing required columns: {missing}")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    print(f"       rows={len(out):,} range={out.index.min()} -> {out.index.max()}", flush=True)
    return out


def build_features(bars: pd.DataFrame, th: PanicThresholds) -> pd.DataFrame:
    print("[features] building V2 panic-wick entry features", flush=True)
    df = bars.copy().sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    body_low = df[["open", "close"]].min(axis=1)
    body_high = df[["open", "close"]].max(axis=1)
    out = pd.DataFrame(index=df.index)
    out[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]]
    out["range_pct"] = _safe_divide(rng, df["close"]).to_numpy()
    out["close_pos"] = _safe_divide(df["close"] - df["low"], rng).to_numpy()
    out["lower_wick"] = (body_low - df["low"]).clip(lower=0.0)
    out["upper_wick"] = (df["high"] - body_high).clip(lower=0.0)
    out["lower_wick_share"] = _safe_divide(out["lower_wick"], rng).to_numpy()
    out["upper_wick_share"] = _safe_divide(out["upper_wick"], rng).to_numpy()
    out["tr"] = _true_range(df)
    out["atr"] = out["tr"].rolling(60, min_periods=30).mean()
    out["atr_pct"] = _safe_divide(out["atr"], df["close"]).to_numpy()
    out["lower_wick_atr"] = _safe_divide(out["lower_wick"], out["atr"]).to_numpy()
    vol_base = df["volume"].shift(1).rolling(240, min_periods=60).median()
    out["volume_ratio"] = _safe_divide(df["volume"], vol_base).to_numpy()
    out["volume_climax"] = out["volume_ratio"] >= th.volume_ratio_min
    out["ret_30"] = df["close"].pct_change(30)
    out["ret_120"] = df["close"].pct_change(120)
    out["ema_240"] = df["close"].ewm(span=240, adjust=False, min_periods=240).mean()
    out["ema240_slope_60"] = out["ema_240"] / out["ema_240"].shift(60) - 1.0

    delta = pd.to_numeric(df.get("delta_notional", np.nan), errors="coerce")
    notional = pd.to_numeric(df.get("notional", np.nan), errors="coerce").abs()
    out["delta_ratio"] = _safe_divide(delta, notional).to_numpy()
    if "taker_buy_ratio" in df.columns:
        out["taker_buy_ratio"] = pd.to_numeric(df["taker_buy_ratio"], errors="coerce")
    else:
        buy_volume = pd.to_numeric(df.get("buy_volume", np.nan), errors="coerce")
        out["taker_buy_ratio"] = _safe_divide(buy_volume, df["volume"]).to_numpy()

    out["long_lower_wick"] = (out["lower_wick_share"] >= th.wick_share_min) & (out["lower_wick_atr"] >= th.wick_atr_min)
    out["long_upper_wick"] = out["upper_wick_share"] >= th.wick_share_min
    out["lower_volume_shadow"] = out["long_lower_wick"] & out["volume_climax"]
    out["two_sided_shadow"] = out["lower_volume_shadow"] & out["long_upper_wick"]
    out["session"] = session_label(out.index)
    out["vol_regime"] = pd.cut(
        out["atr_pct"],
        bins=[-np.inf, 0.0015, 0.0030, 0.0050, np.inf],
        labels=["very_low_vol", "low_mid_vol", "mid_high_vol", "extreme_vol"],
    ).astype("object").fillna("NA")
    out["trend_regime"] = np.select(
        [
            (df["close"] > out["ema_240"]) & (out["ema240_slope_60"] > 0.0005),
            (df["close"] < out["ema_240"]) & (out["ema240_slope_60"] < -0.0005),
        ],
        ["uptrend", "downtrend"],
        default="range_or_transition",
    )
    high_vol = out["vol_regime"].isin(["mid_high_vol", "extreme_vol"])
    prior_flush = (out["ret_30"] <= th.prior_flush_30_min) | (out["ret_120"] <= th.prior_flush_120_min)
    out["panic_context"] = high_vol & prior_flush
    out["panic_downtrend_context"] = out["panic_context"] & (out["trend_regime"] == "downtrend")
    out["flow_absorption"] = (out["delta_ratio"] <= th.delta_absorption_max) | (out["taker_buy_ratio"] <= th.taker_buy_absorption_max)
    return out


def _event_frame_from_mask(features: pd.DataFrame, *, mask: pd.Series, event_name: str, family: str, structure: str) -> pd.DataFrame:
    bool_mask = mask.fillna(False).to_numpy(dtype=bool)
    idx = features.index[bool_mask]
    if len(idx) == 0:
        return pd.DataFrame()
    pos_map = pd.Series(np.arange(len(features), dtype=int), index=features.index)
    event_pos = pos_map.reindex(idx).to_numpy(dtype=int)
    signal_pos = event_pos
    valid = signal_pos + 1 < len(features)
    if not valid.any():
        return pd.DataFrame()
    idx = idx[valid]
    f = features.loc[idx]
    event_pos = event_pos[valid]
    signal_pos = signal_pos[valid]
    signal_bar_time = features.index[signal_pos]
    return pd.DataFrame(
        {
            "event_name": event_name,
            "family": family,
            "direction": "LONG",
            "side": 1,
            "structure": structure,
            "event_bar_time": idx,
            "event_bar_pos": event_pos,
            "signal_bar_time": signal_bar_time,
            "signal_bar_pos": signal_pos,
            "signal_time": signal_bar_time + BAR_DELTA,
            "signal_available_time": signal_bar_time + BAR_DELTA,
            "entry_bar_pos": signal_pos + 1,
            "session": f["session"].to_numpy(),
            "vol_regime": f["vol_regime"].astype(str).to_numpy(),
            "trend_regime": f["trend_regime"].astype(str).to_numpy(),
            "close_pos": f["close_pos"].to_numpy(dtype=float),
            "volume_ratio": f["volume_ratio"].to_numpy(dtype=float),
            "lower_wick_share": f["lower_wick_share"].to_numpy(dtype=float),
            "lower_wick_atr": f["lower_wick_atr"].to_numpy(dtype=float),
            "ret_30": f["ret_30"].to_numpy(dtype=float),
            "ret_120": f["ret_120"].to_numpy(dtype=float),
            "delta_ratio": f["delta_ratio"].to_numpy(dtype=float),
            "taker_buy_ratio": f["taker_buy_ratio"].to_numpy(dtype=float),
            "panic_context": f["panic_context"].astype(bool).to_numpy(),
            "panic_downtrend_context": f["panic_downtrend_context"].astype(bool).to_numpy(),
            "flow_absorption": f["flow_absorption"].astype(bool).to_numpy(),
            "event_low": f["low"].to_numpy(dtype=float),
            "event_high": f["high"].to_numpy(dtype=float),
            "event_close": f["close"].to_numpy(dtype=float),
        }
    )


def build_entry_events(features: pd.DataFrame, th: PanicThresholds) -> pd.DataFrame:
    print("[events] building strict_flow and strict_reclaim entry events", flush=True)
    f = features
    lower = f["lower_volume_shadow"] & ~f["two_sided_shadow"]
    panic_downtrend = lower & f["panic_downtrend_context"]
    reclaim = f["close_pos"] >= th.reclaim_close_pos
    soft_reclaim = f["close_pos"] >= th.soft_reclaim_close_pos
    negative_flow = f["flow_absorption"]
    specs: list[tuple[str, str, str, pd.Series]] = [
        (
            "panic_downtrend_lower_reclaim_long",
            "panic_downtrend_reclaim",
            "downtrend_high_vol_prior_flush_lower_wick_close_reclaim",
            panic_downtrend & reclaim,
        ),
        (
            "panic_downtrend_neg_delta_absorption_long",
            "panic_downtrend_flow_absorption",
            "downtrend_panic_lower_wick_soft_reclaim_negative_delta_or_low_taker_buy",
            panic_downtrend & soft_reclaim & negative_flow,
        ),
    ]
    frames = [_event_frame_from_mask(f, mask=mask, event_name=name, family=family, structure=structure) for name, family, structure, mask in specs]
    out = pd.concat([x for x in frames if not x.empty], ignore_index=True) if any(not x.empty for x in frames) else pd.DataFrame()
    if out.empty:
        return out
    out = out.sort_values(["signal_bar_time", "event_name"]).reset_index(drop=True)
    out["event_id"] = np.arange(len(out), dtype=int)
    out["entry_time"] = pd.NaT
    out["entry_price"] = np.nan
    return out


def make_policy_signals(events: pd.DataFrame, policy: str) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    if policy == "strict_flow":
        out = events.loc[events["event_name"] == PRIMARY_ENTRY_EVENT["strict_flow"]].copy()
        out["policy"] = policy
        out["source_event_names"] = out["event_name"]
        return out
    if policy == "strict_reclaim":
        out = events.loc[events["event_name"] == PRIMARY_ENTRY_EVENT["strict_reclaim"]].copy()
        out["policy"] = policy
        out["source_event_names"] = out["event_name"]
        return out
    if policy == "priority_union":
        p = events.copy()
        priority = {
            "panic_downtrend_neg_delta_absorption_long": 0,
            "panic_downtrend_lower_reclaim_long": 1,
        }
        p["priority"] = p["event_name"].map(priority).fillna(99).astype(int)
        grouped = p.groupby("event_bar_pos", observed=False)["event_name"].agg(lambda x: "+".join(sorted(set(x))))
        out = p.sort_values(["event_bar_pos", "priority"]).drop_duplicates("event_bar_pos", keep="first").copy()
        out["policy"] = policy
        out["source_event_names"] = out["event_bar_pos"].map(grouped)
        return out.drop(columns=["priority"])
    raise ValueError(f"Unknown entry policy: {policy}")


def _time_stop_bars(exit_mode: str) -> int | None:
    # Kept for compatibility with V1 report builders. V1.1 does not use a pure
    # time-stop as a candidate exit. Max-hold caps are safety exits inside TP modes.
    return None


def _take_profit_config(exit_mode: str) -> TakeProfitConfig | None:
    if exit_mode == "fixed_tp_1p0_multi_sweep":
        return TakeProfitConfig(initial_pct=0.010, min_pct=0.010, fixed=True)
    if exit_mode == "decay_tp_1p0_to_0p2_multi_sweep":
        return TakeProfitConfig(initial_pct=0.010, min_pct=0.002, decay_per_hour_pct=0.002)
    if exit_mode == "decay_tp_0p8_to_0p2_multi_sweep":
        return TakeProfitConfig(initial_pct=0.008, min_pct=0.002, decay_per_hour_pct=0.002)
    if exit_mode == "decay_tp_1p0_to_0p2_multi_sweep_cap_360":
        return TakeProfitConfig(initial_pct=0.010, min_pct=0.002, decay_per_hour_pct=0.002, max_hold_bars=360)
    if exit_mode == "decay_tp_0p8_to_0p2_multi_sweep_cap_360":
        return TakeProfitConfig(initial_pct=0.008, min_pct=0.002, decay_per_hour_pct=0.002, max_hold_bars=360)
    if exit_mode == "decay_tp_1p0_to_0p2_multi_sweep_cap_720":
        return TakeProfitConfig(initial_pct=0.010, min_pct=0.002, decay_per_hour_pct=0.002, max_hold_bars=720)
    return None


def _tp_target_pct(tp: TakeProfitConfig, holding_bars: int) -> float:
    if tp.fixed:
        return float(tp.initial_pct)
    # First 60 bars keep the initial target, then reduce by one step per hour.
    hour_steps = max(0, int((max(1, holding_bars) - 1) // 60))
    return float(max(tp.min_pct, tp.initial_pct - tp.decay_per_hour_pct * hour_steps))


def _exit_class(exit_mode: str) -> str:
    if exit_mode == BASELINE_EXIT_MODE:
        return "baseline_structural"
    if exit_mode in TIME_BENCHMARK_MODES:
        return "time_benchmark"
    return "structural_upgrade"


def _calc_mfe_mae(high_arr: np.ndarray, low_arr: np.ndarray, entry_price: float, start_pos: int, end_pos: int) -> tuple[float, float]:
    if end_pos < start_pos:
        return (float("nan"), float("nan"))
    hi = high_arr[start_pos : end_pos + 1]
    lo = low_arr[start_pos : end_pos + 1]
    if hi.size == 0 or not np.isfinite(entry_price) or entry_price <= 0:
        return (float("nan"), float("nan"))
    mfe = float(np.nanmax(hi / entry_price - 1.0))
    mae = float(np.nanmin(lo / entry_price - 1.0))
    return mfe, mae


def _confirmed_higher_low(low_arr: np.ndarray, j: int, event_low: float) -> float | None:
    # At close of bar j, bar j-1 can be confirmed as a 3-bar pivot low.
    if j < 2:
        return None
    pivot_pos = j - 1
    left = low_arr[pivot_pos - 1]
    pivot = low_arr[pivot_pos]
    right = low_arr[j]
    if not (np.isfinite(left) and np.isfinite(pivot) and np.isfinite(right)):
        return None
    if pivot <= left and pivot <= right and pivot > event_low:
        return float(pivot)
    return None


def _simulate_one_path(
    frame: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    policy: str,
    exit_mode: str,
    entry_delay_bars: int,
    round_trip_cost_pct: float,
    cfg: ExitStateConfig,
    progress_every: int,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    idx = frame.index
    open_arr = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    high_arr = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    low_arr = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    close_arr = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    n = len(frame)
    sig = signals.sort_values(["entry_bar_pos", "event_name"]).copy()
    trades: list[dict[str, object]] = []
    next_free_pos = -1
    tp_cfg = _take_profit_config(exit_mode)
    for k, row in enumerate(sig.itertuples(index=False), start=1):
        base_entry_pos = int(getattr(row, "entry_bar_pos"))
        entry_pos = base_entry_pos + int(entry_delay_bars)
        if entry_pos <= next_free_pos:
            continue
        if entry_pos < 0 or entry_pos >= n - 1:
            continue
        entry_price = float(open_arr[entry_pos])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        event_low = float(getattr(row, "event_low"))
        event_high = float(getattr(row, "event_high"))
        stop_buffer = float(getattr(row, "event_low")) * float(cfg.deeper_sweep_buffer_pct)
        exit_pos = n - 1
        exit_price = float(close_arr[-1])
        exit_reason = "end_of_data"
        decision_bar_pos = n - 1
        exit_execution = "end_of_data_close"

        swept_active = False
        bars_since_sweep = 0
        sweep_count = 0
        prev_below = False
        min_sweep_low = math.inf
        consec_close_below_low = 0
        event_high_reclaimed = False
        trail_stop = float("nan")
        trail_updates = 0
        max_favorable_close = entry_price
        tp_hit_count_seen = 0
        ambiguous_tp_structural_same_bar = False
        max_hold_cap_hit = False
        tp_target_at_exit = float("nan")

        # Closed bar j can trigger structural exit at next bar open j+1.
        # Resting TP can fill inside bar j if high reaches the target. If the same
        # bar also has structural failure, the structural exit wins conservatively.
        for j in range(entry_pos, n - 1):
            c = float(close_arr[j])
            l = float(low_arr[j])
            h = float(high_arr[j])
            holding_bars_so_far = int(j - entry_pos + 1)
            if np.isfinite(c):
                max_favorable_close = max(max_favorable_close, c)
            if np.isfinite(l) and l < event_low:
                if not prev_below:
                    sweep_count += 1
                prev_below = True
                swept_active = True
                bars_since_sweep += 1
                min_sweep_low = min(min_sweep_low, l)
            else:
                prev_below = False
                if swept_active:
                    bars_since_sweep += 1

            if np.isfinite(c) and c >= event_low:
                swept_active = False
                bars_since_sweep = 0
                consec_close_below_low = 0
            elif np.isfinite(c) and c < event_low:
                consec_close_below_low += 1
            else:
                consec_close_below_low = 0

            if np.isfinite(c) and c >= event_high:
                event_high_reclaimed = True
                if not np.isfinite(trail_stop):
                    trail_stop = event_low

            pivot = _confirmed_higher_low(low_arr, j, event_low)
            if event_high_reclaimed and pivot is not None:
                if not np.isfinite(trail_stop) or pivot > trail_stop:
                    trail_stop = float(pivot)
                    trail_updates += 1

            target_pct = float("nan")
            target_price = float("nan")
            tp_hit_this_bar = False
            if tp_cfg is not None:
                target_pct = _tp_target_pct(tp_cfg, holding_bars_so_far)
                target_price = entry_price * (1.0 + target_pct)
                tp_hit_this_bar = bool(np.isfinite(h) and np.isfinite(target_price) and h >= target_price)
                if tp_hit_this_bar:
                    tp_hit_count_seen += 1

            reason: str | None = None
            # Baseline structural failure retained for every V1.1 mode.
            deeper_threshold = event_low - max(stop_buffer, abs(event_low) * float(cfg.deeper_sweep_buffer_pct))
            if sweep_count >= 2 and np.isfinite(l) and l < deeper_threshold and np.isfinite(c) and c < event_low:
                reason = "multi_sweep_deeper_fail"
            elif event_high_reclaimed and np.isfinite(trail_stop) and np.isfinite(c) and c < trail_stop:
                reason = "higher_low_trail_break"

            if reason is not None:
                if tp_hit_this_bar:
                    ambiguous_tp_structural_same_bar = True
                exit_pos = j + 1
                exit_price = float(open_arr[exit_pos])
                exit_reason = reason
                decision_bar_pos = j
                exit_execution = "next_open_after_closed_bar_structural_exit"
                tp_target_at_exit = target_pct
                break

            if tp_hit_this_bar:
                exit_pos = j
                exit_price = float(target_price)
                exit_reason = "take_profit_hit"
                decision_bar_pos = j
                exit_execution = "intrabar_resting_limit_tp"
                tp_target_at_exit = target_pct
                break

            if tp_cfg is not None and tp_cfg.max_hold_bars is not None and holding_bars_so_far >= int(tp_cfg.max_hold_bars):
                exit_pos = j + 1
                exit_price = float(open_arr[exit_pos])
                exit_reason = f"max_hold_cap_{int(tp_cfg.max_hold_bars)}"
                decision_bar_pos = j
                exit_execution = "next_open_after_max_hold_cap"
                max_hold_cap_hit = True
                tp_target_at_exit = target_pct
                break

        if not np.isfinite(exit_price) or exit_price <= 0:
            continue
        gross = float(exit_price / entry_price - 1.0)
        net = gross - float(round_trip_cost_pct)
        mfe, mae = _calc_mfe_mae(high_arr, low_arr, entry_price, entry_pos, exit_pos)
        trades.append(
            {
                "policy": policy,
                "exit_mode": exit_mode,
                "exit_class": _exit_class(exit_mode),
                "entry_delay_bars": int(entry_delay_bars),
                "source_event_names": getattr(row, "source_event_names", getattr(row, "event_name")),
                "event_name": getattr(row, "event_name"),
                "event_id": getattr(row, "event_id"),
                "event_bar_time": getattr(row, "event_bar_time"),
                "signal_bar_time": getattr(row, "signal_bar_time"),
                "signal_time": getattr(row, "signal_time"),
                "expected_entry_time": idx[base_entry_pos] if 0 <= base_entry_pos < n else pd.NaT,
                "entry_time": idx[entry_pos],
                "entry_bar_pos": entry_pos,
                "entry_price": entry_price,
                "exit_decision_bar_time": idx[decision_bar_pos] if 0 <= decision_bar_pos < n else pd.NaT,
                "exit_time": idx[exit_pos] if 0 <= exit_pos < n else pd.NaT,
                "exit_bar_pos": int(exit_pos),
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "exit_execution": exit_execution,
                "gross_return": gross,
                "net_return": net,
                "holding_bars": int(exit_pos - entry_pos + 1),
                "mfe": mfe,
                "mae": mae,
                "event_low": event_low,
                "event_high": event_high,
                "event_close": getattr(row, "event_close"),
                "session": getattr(row, "session"),
                "vol_regime": getattr(row, "vol_regime"),
                "trend_regime": getattr(row, "trend_regime"),
                "delta_ratio": getattr(row, "delta_ratio"),
                "taker_buy_ratio": getattr(row, "taker_buy_ratio"),
                "sweep_count": int(sweep_count),
                "min_sweep_low": float(min_sweep_low) if np.isfinite(min_sweep_low) else np.nan,
                "event_high_reclaimed": bool(event_high_reclaimed),
                "trail_updates": int(trail_updates),
                "final_trail_stop": float(trail_stop) if np.isfinite(trail_stop) else np.nan,
                "tp_enabled": bool(tp_cfg is not None),
                "tp_initial_pct": float(tp_cfg.initial_pct) if tp_cfg is not None else np.nan,
                "tp_min_pct": float(tp_cfg.min_pct) if tp_cfg is not None else np.nan,
                "tp_decay_per_hour_pct": float(tp_cfg.decay_per_hour_pct) if tp_cfg is not None else np.nan,
                "tp_target_at_exit": float(tp_target_at_exit) if np.isfinite(tp_target_at_exit) else np.nan,
                "tp_hit_count_seen": int(tp_hit_count_seen),
                "max_hold_cap_bars": int(tp_cfg.max_hold_bars) if tp_cfg is not None and tp_cfg.max_hold_bars is not None else np.nan,
                "max_hold_cap_hit": bool(max_hold_cap_hit),
                "ambiguous_tp_structural_same_bar": bool(ambiguous_tp_structural_same_bar),
            }
        )
        next_free_pos = int(exit_pos)
    return pd.DataFrame(trades)

def run_backtests(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    delays: tuple[int, ...],
    round_trip_cost_pct: float,
    progress_every: int,
) -> pd.DataFrame:
    print("[backtest] running V1.1 exit-upgrade state machines", flush=True)
    cfg = ExitStateConfig()
    frames: list[pd.DataFrame] = []
    total = len(ENTRY_POLICIES) * len(EXIT_MODES) * len(delays)
    progress = ProgressReporter(label="[backtest] exit upgrade", total=total, every=1)
    done = 0
    for policy in ENTRY_POLICIES:
        policy_events = make_policy_signals(events, policy)
        for exit_mode in EXIT_MODES:
            for delay in delays:
                frames.append(
                    _simulate_one_path(
                        bars,
                        policy_events,
                        policy=policy,
                        exit_mode=exit_mode,
                        entry_delay_bars=int(delay),
                        round_trip_cost_pct=float(round_trip_cost_pct),
                        cfg=cfg,
                        progress_every=int(progress_every),
                    )
                )
                done += 1
                progress.update(done)
    progress.close()
    non_empty = [x for x in frames if not x.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


def summarize_return(x: pd.Series) -> dict[str, float | int]:
    r = pd.to_numeric(x, errors="coerce").dropna()
    base: dict[str, float | int] = {
        "count": int(len(r)),
        "mean_net": float(r.mean()) if len(r) else np.nan,
        "median_net": float(r.median()) if len(r) else np.nan,
        "win_rate": float((r > 0).mean()) if len(r) else np.nan,
        "profit_factor": _profit_factor(r),
        "top5_winner_share": _top5_winner_share(r),
        "max_consecutive_losses": _max_consecutive_losses(r),
    }
    base.update(_equity_stats(r))
    return base


def build_trade_summary(trades: pd.DataFrame, *, min_trades: int) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, part in trades.groupby(["policy", "exit_mode", "entry_delay_bars"], observed=False):
        policy, exit_mode, delay = keys
        yearly = part.assign(year=pd.to_datetime(part["entry_time"]).dt.year).groupby("year", observed=False)["net_return"].mean()
        s = summarize_return(part["net_return"])
        rows.append(
            {
                "policy": policy,
                "exit_mode": exit_mode,
                "exit_class": _exit_class(str(exit_mode)),
                "entry_delay_bars": int(delay),
                **s,
                "eligible": bool(s["count"] >= int(min_trades)),
                "trades_per_month": _events_per_month(part["entry_time"]),
                "max_days_without_trade": _max_days_without_event(part["entry_time"]),
                "positive_years": int((yearly > 0).sum()),
                "year_count": int(yearly.notna().sum()),
                "avg_holding_bars": float(pd.to_numeric(part["holding_bars"], errors="coerce").mean()),
                "median_holding_bars": float(pd.to_numeric(part["holding_bars"], errors="coerce").median()),
                "mean_mfe": float(pd.to_numeric(part["mfe"], errors="coerce").mean()),
                "mean_mae": float(pd.to_numeric(part["mae"], errors="coerce").mean()),
                "event_high_reclaim_share": float(pd.to_numeric(part["event_high_reclaimed"], errors="coerce").mean()),
                "avg_sweep_count": float(pd.to_numeric(part["sweep_count"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["exit_class", "mean_net"], ascending=[True, False], na_position="last")


def build_yearly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    tmp = trades.copy()
    tmp["year"] = pd.to_datetime(tmp["entry_time"]).dt.year
    rows: list[dict[str, object]] = []
    for keys, part in tmp.groupby(["policy", "exit_mode", "entry_delay_bars", "year"], observed=False):
        policy, exit_mode, delay, year = keys
        rows.append({"policy": policy, "exit_mode": exit_mode, "entry_delay_bars": int(delay), "year": int(year), **summarize_return(part["net_return"])})
    return pd.DataFrame(rows)


def build_monthly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    tmp = trades.copy()
    et = pd.to_datetime(tmp["entry_time"])
    tmp["month"] = et.dt.to_period("M").astype(str)
    rows: list[dict[str, object]] = []
    for keys, part in tmp.groupby(["policy", "exit_mode", "entry_delay_bars", "month"], observed=False):
        policy, exit_mode, delay, month = keys
        rows.append({"policy": policy, "exit_mode": exit_mode, "entry_delay_bars": int(delay), "month": month, **summarize_return(part["net_return"])})
    return pd.DataFrame(rows)


def build_exit_reason_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, part in trades.groupby(["policy", "exit_mode", "entry_delay_bars"], observed=False):
        policy, exit_mode, delay = keys
        total = len(part)
        for reason, rpart in part.groupby("exit_reason", observed=False):
            rows.append(
                {
                    "policy": policy,
                    "exit_mode": exit_mode,
                    "entry_delay_bars": int(delay),
                    "exit_reason": reason,
                    "count": int(len(rpart)),
                    "share": float(len(rpart) / total) if total else np.nan,
                    **summarize_return(rpart["net_return"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["policy", "exit_mode", "entry_delay_bars", "count"], ascending=[True, True, True, False])


def build_cost_stress(trades: pd.DataFrame, *, cost_multipliers: tuple[float, ...], round_trip_cost_pct: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, part in trades.groupby(["policy", "exit_mode", "entry_delay_bars"], observed=False):
        policy, exit_mode, delay = keys
        gross = pd.to_numeric(part["gross_return"], errors="coerce")
        for mult in cost_multipliers:
            net = gross - float(round_trip_cost_pct) * float(mult)
            rows.append({"policy": policy, "exit_mode": exit_mode, "entry_delay_bars": int(delay), "cost_multiplier": float(mult), **summarize_return(net)})
    return pd.DataFrame(rows)


def build_delay_stress(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    cols = [
        "policy",
        "exit_mode",
        "exit_class",
        "entry_delay_bars",
        "count",
        "mean_net",
        "median_net",
        "win_rate",
        "profit_factor",
        "total_return",
        "max_drawdown",
        "positive_years",
        "year_count",
        "avg_holding_bars",
    ]
    return summary[[c for c in cols if c in summary.columns]].copy().sort_values(["policy", "exit_mode", "entry_delay_bars"])


def _lookup_fee2_mean(cost_stress: pd.DataFrame) -> dict[tuple[str, str, int], float]:
    out: dict[tuple[str, str, int], float] = {}
    if cost_stress.empty:
        return out
    tmp = cost_stress.loc[np.isclose(pd.to_numeric(cost_stress["cost_multiplier"], errors="coerce"), 2.0)].copy()
    for row in tmp.itertuples(index=False):
        out[(str(getattr(row, "policy")), str(getattr(row, "exit_mode")), int(getattr(row, "entry_delay_bars")))] = float(getattr(row, "mean_net"))
    return out


def build_baseline_vs_candidate(summary: pd.DataFrame, cost_stress: pd.DataFrame, *, min_trades: int) -> pd.DataFrame:
    """Compare every exit-upgrade row against the frozen V1 baseline.

    A condition is promotable only when it improves several dimensions at once;
    a higher mean alone is not enough.
    """
    if summary.empty:
        return pd.DataFrame()
    fee2 = _lookup_fee2_mean(cost_stress)
    base = summary.loc[summary["exit_mode"] == BASELINE_EXIT_MODE].copy()
    if base.empty:
        return pd.DataFrame()
    base_map = {(str(r.policy), int(r.entry_delay_bars)): r for r in base.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    for row in summary.loc[summary["exit_mode"] != BASELINE_EXIT_MODE].itertuples(index=False):
        key = (str(row.policy), int(row.entry_delay_bars))
        b = base_map.get(key)
        if b is None:
            continue
        cand_fee2 = fee2.get((str(row.policy), str(row.exit_mode), int(row.entry_delay_bars)), np.nan)
        base_fee2 = fee2.get((str(b.policy), str(b.exit_mode), int(b.entry_delay_bars)), np.nan)
        count_ok = int(row.count) >= int(min_trades)
        mean_improved = float(row.mean_net) > float(b.mean_net)
        pf_not_worse = (pd.isna(row.profit_factor) or pd.isna(b.profit_factor)) or float(row.profit_factor) >= float(b.profit_factor) - 0.02
        fee2_improved = (pd.isna(cand_fee2) or pd.isna(base_fee2)) or float(cand_fee2) > float(base_fee2)
        fee2_alive = pd.notna(cand_fee2) and float(cand_fee2) > 0.0
        years_not_worse = int(row.positive_years) >= int(b.positive_years)
        trade_count_not_shrunk = int(row.count) >= max(int(min_trades), int(float(b.count) * 0.70))
        hold_not_exploded = pd.notna(row.avg_holding_bars) and pd.notna(b.avg_holding_bars) and float(row.avg_holding_bars) <= max(float(b.avg_holding_bars) * 1.25, float(b.avg_holding_bars) + 120.0)
        top5_not_worse = (
            pd.isna(row.top5_winner_share)
            or pd.isna(b.top5_winner_share)
            or float(row.top5_winner_share) <= float(b.top5_winner_share) + 0.05
        )
        max_gap_not_worse = (
            pd.isna(row.max_days_without_trade)
            or pd.isna(b.max_days_without_trade)
            or float(row.max_days_without_trade) <= float(b.max_days_without_trade) + 14.0
        )
        promote = bool(
            count_ok
            and mean_improved
            and pf_not_worse
            and fee2_improved
            and fee2_alive
            and years_not_worse
            and trade_count_not_shrunk
            and hold_not_exploded
            and top5_not_worse
            and max_gap_not_worse
        )
        failed = []
        for name, ok in [
            ("count_ok", count_ok),
            ("mean_improved", mean_improved),
            ("pf_not_worse", pf_not_worse),
            ("fee2_improved", fee2_improved),
            ("fee2_alive", fee2_alive),
            ("years_not_worse", years_not_worse),
            ("trade_count_not_shrunk", trade_count_not_shrunk),
            ("hold_not_exploded", hold_not_exploded),
            ("top5_not_worse", top5_not_worse),
            ("max_gap_not_worse", max_gap_not_worse),
        ]:
            if not bool(ok):
                failed.append(name)
        rows.append(
            {
                "policy": row.policy,
                "entry_delay_bars": int(row.entry_delay_bars),
                "candidate_exit_mode": row.exit_mode,
                "baseline_exit_mode": b.exit_mode,
                "candidate_count": int(row.count),
                "baseline_count": int(b.count),
                "candidate_mean_net": float(row.mean_net),
                "baseline_mean_net": float(b.mean_net),
                "delta_mean_net": float(row.mean_net) - float(b.mean_net),
                "candidate_profit_factor": float(row.profit_factor) if pd.notna(row.profit_factor) else np.nan,
                "baseline_profit_factor": float(b.profit_factor) if pd.notna(b.profit_factor) else np.nan,
                "delta_profit_factor": (float(row.profit_factor) - float(b.profit_factor)) if pd.notna(row.profit_factor) and pd.notna(b.profit_factor) else np.nan,
                "candidate_fee2_mean_net": cand_fee2,
                "baseline_fee2_mean_net": base_fee2,
                "delta_fee2_mean_net": (float(cand_fee2) - float(base_fee2)) if pd.notna(cand_fee2) and pd.notna(base_fee2) else np.nan,
                "candidate_positive_years": int(row.positive_years),
                "baseline_positive_years": int(b.positive_years),
                "candidate_avg_holding_bars": float(row.avg_holding_bars),
                "baseline_avg_holding_bars": float(b.avg_holding_bars),
                "candidate_max_days_without_trade": float(row.max_days_without_trade) if pd.notna(row.max_days_without_trade) else np.nan,
                "baseline_max_days_without_trade": float(b.max_days_without_trade) if pd.notna(b.max_days_without_trade) else np.nan,
                "candidate_top5_winner_share": float(row.top5_winner_share) if pd.notna(row.top5_winner_share) else np.nan,
                "baseline_top5_winner_share": float(b.top5_winner_share) if pd.notna(b.top5_winner_share) else np.nan,
                "count_ok": bool(count_ok),
                "mean_improved": bool(mean_improved),
                "pf_not_worse": bool(pf_not_worse),
                "fee2_improved": bool(fee2_improved),
                "fee2_alive": bool(fee2_alive),
                "years_not_worse": bool(years_not_worse),
                "trade_count_not_shrunk": bool(trade_count_not_shrunk),
                "hold_not_exploded": bool(hold_not_exploded),
                "top5_not_worse": bool(top5_not_worse),
                "max_gap_not_worse": bool(max_gap_not_worse),
                "promote_condition": promote,
                "decision": "promote_condition" if promote else "reject_condition",
                "failed_checks": ";".join(failed),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["promote_condition", "delta_mean_net", "candidate_fee2_mean_net"], ascending=[False, False, False])
    return out


def build_trade_counts(events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy in ENTRY_POLICIES:
        signals = make_policy_signals(events, policy)
        count = len(signals)
        for keys, part in trades.loc[trades["policy"] == policy].groupby(["exit_mode", "entry_delay_bars"], observed=False):
            exit_mode, delay = keys
            rows.append(
                {
                    "policy": policy,
                    "exit_mode": exit_mode,
                    "entry_delay_bars": int(delay),
                    "raw_signal_count": int(count),
                    "executed_trade_count": int(len(part)),
                    "skipped_due_overlap_or_invalid": int(count - len(part)),
                    "executed_share": float(len(part) / count) if count else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_causal_audit(events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not events.empty:
        for row in events.itertuples(index=False):
            expected_entry = getattr(row, "signal_bar_time") + BAR_DELTA
            rows.append(
                {
                    "audit_type": "event_entry_plan",
                    "policy": "event_pool",
                    "event_id": getattr(row, "event_id"),
                    "event_name": getattr(row, "event_name"),
                    "signal_bar_time": getattr(row, "signal_bar_time"),
                    "signal_time": getattr(row, "signal_time"),
                    "expected_entry_time": expected_entry,
                    "entry_time": expected_entry,
                    "used_context_timestamp": pd.NaT,
                    "used_context_available_time": pd.NaT,
                    "context_available_time_flag": False,
                    "entry_not_next_open_flag": False,
                    "exit_decision_after_entry_flag": False,
                    "lookahead_flag": False,
                    "audit_notes": "closed 1m event bar -> next open; no MTF context",
                }
            )
    if not trades.empty:
        sample = trades.head(20000)
        for row in sample.itertuples(index=False):
            rows.append(
                {
                    "audit_type": "executed_trade_exit",
                    "policy": getattr(row, "policy"),
                    "event_id": getattr(row, "event_id"),
                    "event_name": getattr(row, "event_name"),
                    "signal_bar_time": getattr(row, "signal_bar_time"),
                    "signal_time": getattr(row, "signal_time"),
                    "expected_entry_time": getattr(row, "expected_entry_time"),
                    "entry_time": getattr(row, "entry_time"),
                    "used_context_timestamp": pd.NaT,
                    "used_context_available_time": pd.NaT,
                    "context_available_time_flag": False,
                    "entry_not_next_open_flag": pd.to_datetime(getattr(row, "entry_time")) < pd.to_datetime(getattr(row, "expected_entry_time")),
                    "exit_decision_after_entry_flag": pd.to_datetime(getattr(row, "exit_decision_bar_time")) >= pd.to_datetime(getattr(row, "entry_time")),
                    "lookahead_flag": False,
                    "audit_notes": "exit uses closed 1m bar decision and exits next open; no MTF context",
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["lookahead_flag"] = out["context_available_time_flag"].astype(bool) | out["entry_not_next_open_flag"].astype(bool)
    return out


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_reports(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    thresholds: PanicThresholds,
    features: pd.DataFrame,
    events: pd.DataFrame,
    trades: pd.DataFrame,
    cost_multipliers: tuple[float, ...],
    delays: tuple[int, ...],
) -> None:
    print("[write] writing candidate backtest reports", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_trade_summary(trades, min_trades=int(args.min_trades))
    yearly = build_yearly(trades)
    monthly = build_monthly(trades)
    exit_reasons = build_exit_reason_summary(trades)
    cost_stress = build_cost_stress(trades, cost_multipliers=cost_multipliers, round_trip_cost_pct=float(args.round_trip_cost_pct))
    delay_stress = build_delay_stress(summary)
    trade_counts = build_trade_counts(events, trades)
    causal = build_causal_audit(events, trades)
    baseline_vs_candidate = build_baseline_vs_candidate(summary, cost_stress, min_trades=int(args.min_trades))
    event_sample_cols = [
        "event_id",
        "event_name",
        "family",
        "event_bar_time",
        "signal_time",
        "session",
        "vol_regime",
        "trend_regime",
        "close_pos",
        "volume_ratio",
        "lower_wick_share",
        "lower_wick_atr",
        "ret_30",
        "ret_120",
        "delta_ratio",
        "taker_buy_ratio",
        "event_low",
        "event_high",
    ]
    event_sample = events[[c for c in event_sample_cols if c in events.columns]].head(int(args.event_sample_size)).copy() if not events.empty else pd.DataFrame()
    trade_sample = trades.head(int(args.event_sample_size)).copy() if not trades.empty else pd.DataFrame()
    structural_summary = summary.loc[summary["exit_class"].isin(["baseline_structural", "structural_upgrade"])].copy() if not summary.empty and "exit_class" in summary else pd.DataFrame()
    benchmark_summary = summary.loc[summary["exit_class"] == "time_benchmark"].copy() if not summary.empty and "exit_class" in summary else pd.DataFrame()
    promoted_conditions = baseline_vs_candidate.loc[baseline_vs_candidate["promote_condition"].astype(bool)].copy() if not baseline_vs_candidate.empty else pd.DataFrame()
    rejected_conditions = baseline_vs_candidate.loc[~baseline_vs_candidate["promote_condition"].astype(bool)].copy() if not baseline_vs_candidate.empty else pd.DataFrame()

    manifest = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "edge_status": "research_only_not_tradable",
        "title": TITLE,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "primary_timeframe": args.timeframe,
        "context_timeframes": [],
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "entry_policies": list(ENTRY_POLICIES),
        "entry_baseline": "priority_union / strict_flow / strict_reclaim from V1",
        "baseline_exit_mode": BASELINE_EXIT_MODE,
        "exit_modes": list(EXIT_MODES),
        "time_benchmark_modes": sorted(TIME_BENCHMARK_MODES),
        "entry_delay_bars_list": list(delays),
        "round_trip_cost_pct": float(args.round_trip_cost_pct),
        "cost_multipliers": list(cost_multipliers),
        "input_rows": int(len(features)),
        "event_count": int(len(events)),
        "trade_count": int(len(trades)),
        "thresholds": thresholds.__dict__,
        "exit_state_config": ExitStateConfig().__dict__,
        "causal_policy": "closed 1m event bar; entry next open; exit decisions use closed 1m bars and execute next open; no MTF context",
        "anti_overfit_policy": "Frozen entry set; fixed named exit-upgrade ladder; compare each condition to same-policy/same-delay baseline; promote only if multiple dimensions improve; no parameter grid search",
        "created_at": pd.Timestamp.now("UTC").isoformat(),
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_csv(summary, out_dir / "01_trade_summary.csv")
    _write_csv(yearly, out_dir / "02_trade_yearly.csv")
    _write_csv(monthly, out_dir / "03_trade_monthly.csv")
    _write_csv(exit_reasons, out_dir / "04_exit_reason_summary.csv")
    _write_csv(cost_stress, out_dir / "05_cost_stress.csv")
    _write_csv(delay_stress, out_dir / "06_delay_stress.csv")
    _write_csv(trade_counts, out_dir / "07_trade_counts.csv")
    _write_csv(causal, out_dir / "08_causal_audit.csv")
    _write_csv(structural_summary, out_dir / "09_structural_candidate_summary.csv")
    _write_csv(benchmark_summary, out_dir / "10_time_benchmark_summary.csv")
    _write_csv(event_sample, out_dir / "11_event_sample.csv")
    _write_csv(trade_sample, out_dir / "12_trade_sample.csv")
    _write_csv(baseline_vs_candidate, out_dir / "13_baseline_vs_candidate.csv")
    _write_csv(promoted_conditions, out_dir / "14_promoted_conditions.csv")
    _write_csv(rejected_conditions, out_dir / "15_rejected_conditions.csv")
    if bool(args.write_full_trades) and not trades.empty:
        _write_csv(trades, out_dir / "16_full_trades.csv")

    notes = f"""# {TITLE}

This is a research-only candidate backtest. It does **not** create a tradable edge.

## Entry policies
- `strict_flow`: `panic_downtrend_neg_delta_absorption_long`
- `strict_reclaim`: `panic_downtrend_lower_reclaim_long`
- `priority_union`: exact-bar deduplicated union; flow absorption has priority over reclaim.

## Frozen baseline
- `{BASELINE_EXIT_MODE}` = multi-sweep deeper-fail + higher-low trail.
- Every V1.1 exit-upgrade row is compared against this baseline at the same policy and same entry delay.

## Exit-upgrade ladder
- `fixed_tp_1p0_multi_sweep`: 1.0% fixed TP + baseline structural failure.
- `decay_tp_1p0_to_0p2_multi_sweep`: 1.0% TP, then -0.2% per hour, floor 0.2%.
- `decay_tp_0p8_to_0p2_multi_sweep`: 0.8% TP, then -0.2% per hour, floor 0.2%.
- `_cap_360` / `_cap_720`: same decay TP plus 6h/12h max-hold safety cap.

## Promotion rule
A condition is promoted only if multiple dimensions improve versus baseline:
mean net, PF not worse, fee2 improves and remains positive, positive years not worse, trade count not overly compressed, holding time not exploded, top-winner dependence not worse, and max trade gap not materially worse.

## Review focus
1. `13_baseline_vs_candidate.csv`: main condition decision table.
2. `14_promoted_conditions.csv`: conditions that pass the multi-dimensional gate.
3. `15_rejected_conditions.csv`: failed checks for every rejected condition.
4. `09_structural_candidate_summary.csv`: raw structural summary.
5. `05_cost_stress.csv`: fee 2x / 3x survival.
6. `06_delay_stress.csv`: delay 1/2 bar sensitivity.
7. `02_trade_yearly.csv` and `03_trade_monthly.csv`: stability, especially 2025.
8. `08_causal_audit.csv`: must have no lookahead flags.

## Conservative OHLC path policy
Take-profit is modeled as a resting limit when high reaches target. If structural failure and TP are both observed in the same 1m bar, structural failure wins and `ambiguous_tp_structural_same_bar=True`.
"""
    (out_dir / "README_RESEARCH.md").write_text(notes, encoding="utf-8")
    finalize_research_report(
        out_dir,
        experiment_id=EXPERIMENT_ID,
        edge_id=EDGE_ID,
        title=TITLE,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    thresholds = _thresholds_from_args(args)
    cost_multipliers = _parse_csv_floats(args.cost_multipliers)
    delays = _parse_csv_ints(args.entry_delay_bars_list)
    print(f"[run] {SCRIPT_NAME} v{SCRIPT_VERSION}", flush=True)
    print(f"[args] out_dir={out_dir}", flush=True)
    print("[scope] V1.1 exit upgrade; frozen entry; baseline-vs-candidate comparison; no parameter grid", flush=True)
    bars_full = load_trade_bars(args)
    features_full = build_features(bars_full, thresholds)
    mask = _research_window_mask(features_full.index, args.start_date, args.end_date)
    features = features_full.loc[mask].copy()
    if features.empty:
        raise RuntimeError("No bars in research window after warmup filtering")
    events = build_entry_events(features, thresholds)
    if events.empty:
        print("[events] no entry events found; reports will be empty", flush=True)
        trades = pd.DataFrame()
    else:
        trades = run_backtests(
            features,
            events,
            delays=delays,
            round_trip_cost_pct=float(args.round_trip_cost_pct),
            progress_every=int(args.progress_every),
        )
    write_reports(
        out_dir=out_dir,
        args=args,
        thresholds=thresholds,
        features=features,
        events=events,
        trades=trades,
        cost_multipliers=cost_multipliers,
        delays=delays,
    )
    print("[done] V1.1 exit-upgrade backtest completed; inspect gpt_review_pack.zip before promotion", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
