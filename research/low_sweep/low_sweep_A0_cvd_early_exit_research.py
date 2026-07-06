#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A0 Low Sweep CVD early-exit research.

Research-only.  This file does not change the formal MF backtest, portfolio
wrapper, or live strategy.  It keeps the current MF anchor fixed:

    A0_fp_abs_delta_high + single_swing + next_open + time48 + no_stop

Then it tests causal post-entry exit states that were suggested by the loss
path diagnostics:

1. Early-fail exit: after N closed bars, if price has not reclaimed, MFE is
   still small, and local CVD/order-flow has not recovered, exit on the next
   open.  This is not a hard price stop; it is a failed-reversal state.
2. Weak-bounce giveback: after a small bounce has occurred, if close gives back
   to the entry/floor area and recent CVD turns negative, exit on the next open.
3. Combined states with time48 fallback.

Causality convention:
- The signal bar is closed before entry.
- Post-entry conditions are evaluated only after the relevant 1m bar has closed.
- All early exits execute on the next bar open.
- No same-bar high/low/CVD decision is used for same-bar execution.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.mf.low_sweep.low_sweep_V1_a0_footprint_backtest import prepare_events_and_context  # noqa: E402
from research.low_sweep_a_upgrade_research import (  # noqa: E402
    UpgradeVariant,
    _entry_cost,
    _event_positions,
    _exit_cost,
    _profit_factor,
    _split_csv_names,
    _timing_stats_from_path,
    build_candidate_layer_masks,
    build_market_cache,
    build_support_mask,
    parse_args as _upgrade_parse_args,
    parse_stop_specs,
    summarize_by_period,
    summarize_trades,
    write_csv,
)
from research.low_sweep_panic_reversal_strategy_backtest_probe import load_trade_bars  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402

SCRIPT_NAME = "low_sweep_A0_cvd_early_exit_research"
SCRIPT_VERSION = "v2_write_csv_signature_fix"
DEFAULT_OUT_DIR = "data/reports/research/low_sweep/A0_cvd_early_exit_research"


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def _parse_float_list(raw: str) -> list[float]:
    vals: list[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return sorted(set(v for v in vals if math.isfinite(v)))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--research-cost-mults", default="1.0,1.5,2.0")
    p.add_argument("--progress-every", type=int, default=250)
    p.add_argument("--save-trade-sample", type=int, default=100000)
    p.add_argument(
        "--early-check-bars",
        default="3,5,10",
        help="Closed bars after entry used for early-fail checks. Exit is next open.",
    )
    p.add_argument(
        "--early-mfe-caps",
        default="0.003,0.005",
        help="Require high-water MFE below this cap for early-fail exit.",
    )
    p.add_argument(
        "--giveback-mfe-triggers",
        default="0.006,0.008,0.010",
        help="Small-bounce MFE triggers for weak-bounce giveback states.",
    )
    p.add_argument(
        "--giveback-floors",
        default="0.000,0.001",
        help="Close-return floor vs entry for weak-bounce giveback. Exit next open when close <= floor and CVD turns weak.",
    )
    p.add_argument(
        "--cvd-windows",
        default="3,5,10",
        help="Recent closed-bar windows for CVD/order-flow recovery checks.",
    )
    p.add_argument(
        "--buy-share-threshold",
        type=float,
        default=0.50,
        help="Recent buy-notional share threshold used by early-fail/giveback states.",
    )
    p.add_argument(
        "--min-trades-for-compare",
        type=int,
        default=30,
    )
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
# Bar arrays and numeric helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlowArrays:
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


def _col(frame: pd.DataFrame, name: str, fallback: float = np.nan) -> np.ndarray:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    return np.full(len(frame), float(fallback), dtype=float)


def build_flow_arrays(bars: pd.DataFrame) -> FlowArrays:
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

    return FlowArrays(
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
    )


def _safe_div(n: float, d: float, default: float = np.nan) -> float:
    if not math.isfinite(n) or not math.isfinite(d) or abs(d) <= 1e-12:
        return default
    return float(n / d)


def _sum_window(arr: np.ndarray, start: int, end_inclusive: int) -> float:
    if arr.size == 0:
        return np.nan
    start = max(0, int(start))
    end_inclusive = min(len(arr) - 1, int(end_inclusive))
    if start > end_inclusive:
        return np.nan
    return float(np.nansum(arr[start : end_inclusive + 1]))


def _max_window(arr: np.ndarray, start: int, end_inclusive: int) -> float:
    start = max(0, int(start))
    end_inclusive = min(len(arr) - 1, int(end_inclusive))
    if start > end_inclusive:
        return np.nan
    win = arr[start : end_inclusive + 1]
    return float(np.nanmax(win)) if win.size and np.isfinite(win).any() else np.nan


def _min_window(arr: np.ndarray, start: int, end_inclusive: int) -> float:
    start = max(0, int(start))
    end_inclusive = min(len(arr) - 1, int(end_inclusive))
    if start > end_inclusive:
        return np.nan
    win = arr[start : end_inclusive + 1]
    return float(np.nanmin(win)) if win.size and np.isfinite(win).any() else np.nan


def recent_flow(arr: FlowArrays, start: int, end: int) -> dict[str, float]:
    delta = _sum_window(arr.delta_notional, start, end)
    buy = _sum_window(arr.buy_notional, start, end)
    sell = _sum_window(arr.sell_notional, start, end)
    notional = buy + sell if math.isfinite(buy) and math.isfinite(sell) else _sum_window(arr.notional, start, end)
    return {
        "delta_notional": float(delta),
        "notional": float(notional),
        "delta_pressure": _safe_div(float(delta), float(notional)),
        "buy_notional_share": _safe_div(float(buy), float(notional)),
        "buy_notional": float(buy),
        "sell_notional": float(sell),
    }


# ---------------------------------------------------------------------------
# Variant specs and simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExitStateSpec:
    variant_name: str
    early_check_bars: int | None = None
    early_mfe_cap: float | None = None
    early_cvd_window: int = 5
    early_require_price_weak: bool = True
    early_require_cvd_weak: bool = True
    early_require_buy_share_weak: bool = True
    giveback_mfe_trigger: float | None = None
    giveback_floor: float = 0.0
    giveback_cvd_window: int = 3
    giveback_require_cvd_weak: bool = True
    giveback_require_close_floor: bool = True
    time_horizon: int = 48


def build_exit_state_specs(args: argparse.Namespace) -> list[ExitStateSpec]:
    checks = [int(x) for x in _split_csv_names(args.early_check_bars)]
    mfe_caps = _parse_float_list(args.early_mfe_caps)
    triggers = _parse_float_list(args.giveback_mfe_triggers)
    floors = _parse_float_list(args.giveback_floors)
    cvd_windows = [int(x) for x in _split_csv_names(args.cvd_windows)]

    specs: list[ExitStateSpec] = [ExitStateSpec(variant_name="baseline_time48")]

    # Early-fail only: failed-reversal state, not a hard stop.
    for chk in checks:
        for cap in mfe_caps:
            for win in cvd_windows:
                specs.append(
                    ExitStateSpec(
                        variant_name=f"early_fail_b{chk}_mfe{int(cap*1000):03d}_cvd{win}_time48",
                        early_check_bars=chk,
                        early_mfe_cap=cap,
                        early_cvd_window=win,
                    )
                )

    # Weak-bounce giveback only: small bounce armed, then CVD/close confirms failure.
    for trig in triggers:
        for floor in floors:
            for win in [3, 5]:
                specs.append(
                    ExitStateSpec(
                        variant_name=f"weak_gb_mfe{int(trig*1000):03d}_floor{int(floor*1000):03d}_cvd{win}_time48",
                        giveback_mfe_trigger=trig,
                        giveback_floor=floor,
                        giveback_cvd_window=win,
                    )
                )

    # Combined focused candidates. Keep this small enough to be readable.
    for chk in [5, 10]:
        for cap in [0.003, 0.005]:
            for trig in [0.006, 0.008, 0.010]:
                specs.append(
                    ExitStateSpec(
                        variant_name=f"combo_ef{chk}_mfe{int(cap*1000):03d}_gb{int(trig*1000):03d}_time48",
                        early_check_bars=chk,
                        early_mfe_cap=cap,
                        early_cvd_window=5,
                        giveback_mfe_trigger=trig,
                        giveback_floor=0.0,
                        giveback_cvd_window=3,
                    )
                )
    # Deduplicate by name while preserving order.
    out: list[ExitStateSpec] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.variant_name in seen:
            continue
        seen.add(spec.variant_name)
        out.append(spec)
    return out


def _exit_reason_group(reason: str) -> str:
    r = str(reason)
    if r.startswith("early_fail"):
        return "early_fail"
    if r.startswith("weak_bounce_giveback"):
        return "weak_bounce_giveback"
    if r.startswith("time_exit"):
        return "time_exit"
    return r.split("_")[0] if r else "unknown"


def simulate_state_trade(
    bars: pd.DataFrame,
    event: dict[str, object],
    signal_pos: int,
    spec: ExitStateSpec,
    args: argparse.Namespace,
    *,
    cost_mult: float,
    flow: FlowArrays,
) -> dict[str, object]:
    opens = flow.open
    highs = flow.high
    lows = flow.low
    closes = flow.close
    idx = flow.index
    n = len(idx)

    signal_pos = int(signal_pos)
    entry_pos = signal_pos + int(args.entry_delay_bars)
    if entry_pos >= n:
        return {"valid": False, "invalid_reason": "no_future_open"}
    entry_price = float(opens[entry_pos])
    if not math.isfinite(entry_price) or entry_price <= 0:
        return {"valid": False, "invalid_reason": "bad_entry_price"}

    planned_exit_pos = signal_pos + int(spec.time_horizon)
    if planned_exit_pos >= n or planned_exit_pos <= entry_pos:
        return {"valid": False, "invalid_reason": "insufficient_future_bars"}

    exit_pos = int(planned_exit_pos)
    exit_price = float(closes[planned_exit_pos])
    exit_reason = f"time_exit_h{int(spec.time_horizon)}"
    high_water = float(entry_price)
    armed_giveback = False
    armed_giveback_pos = np.nan
    mtm_low: list[float] = []
    mtm_high: list[float] = []
    early_condition_seen = False
    giveback_condition_seen = False

    for pos in range(entry_pos, planned_exit_pos + 1):
        high_water = max(high_water, float(highs[pos]))
        low_unreal = float(lows[pos]) / entry_price - 1.0
        high_unreal = float(highs[pos]) / entry_price - 1.0
        close_unreal = float(closes[pos]) / entry_price - 1.0
        mtm_low.append(float(low_unreal))
        mtm_high.append(float(high_unreal))

        # Arm weak-bounce state only after the bar has closed and high_water is known.
        if spec.giveback_mfe_trigger is not None and not armed_giveback:
            if high_water / entry_price - 1.0 >= float(spec.giveback_mfe_trigger):
                armed_giveback = True
                armed_giveback_pos = int(pos)

        # Early fail: one-shot checks at or after the configured closed-bar count.
        # It is not a hard price stop: it requires small MFE + no price reclaim + weak CVD/order-flow.
        if spec.early_check_bars is not None and pos >= entry_pos + int(spec.early_check_bars):
            if not early_condition_seen:
                high_mfe = high_water / entry_price - 1.0
                win = int(spec.early_cvd_window)
                fl = recent_flow(flow, pos - win + 1, pos)
                price_weak = bool(close_unreal <= 0.0) if spec.early_require_price_weak else True
                mfe_weak = bool(high_mfe < float(spec.early_mfe_cap if spec.early_mfe_cap is not None else 1.0))
                cvd_weak = bool(float(fl["delta_notional"]) <= 0.0 and float(fl["delta_pressure"]) <= 0.0) if spec.early_require_cvd_weak else True
                buy_weak = bool(float(fl["buy_notional_share"]) < float(args.buy_share_threshold)) if spec.early_require_buy_share_weak and math.isfinite(float(fl["buy_notional_share"])) else True
                if price_weak and mfe_weak and cvd_weak and buy_weak:
                    early_condition_seen = True
                    next_pos = min(pos + 1, planned_exit_pos)
                    exit_pos = int(next_pos)
                    exit_price = float(opens[next_pos]) if next_pos < n else float(closes[pos])
                    exit_reason = (
                        f"early_fail_b{int(spec.early_check_bars)}_mfe{float(high_mfe):.4f}"
                        f"_delta{float(fl['delta_pressure']):.4f}_next_open"
                    )
                    break

        # Weak-bounce giveback: armed by a small MFE, then exit only if close gives
        # back to the floor area and recent CVD/order-flow confirms loss of bid.
        if armed_giveback and spec.giveback_mfe_trigger is not None:
            # Do not immediately exit on the same bar that first armed the state.
            if math.isfinite(float(armed_giveback_pos)) and pos <= int(armed_giveback_pos):
                continue
            win = int(spec.giveback_cvd_window)
            fl = recent_flow(flow, pos - win + 1, pos)
            floor_hit = bool(close_unreal <= float(spec.giveback_floor)) if spec.giveback_require_close_floor else True
            cvd_weak = bool(float(fl["delta_notional"]) < 0.0 and float(fl["delta_pressure"]) < 0.0) if spec.giveback_require_cvd_weak else True
            buy_weak = bool(float(fl["buy_notional_share"]) < float(args.buy_share_threshold)) if math.isfinite(float(fl["buy_notional_share"])) else True
            if floor_hit and cvd_weak and buy_weak:
                giveback_condition_seen = True
                next_pos = min(pos + 1, planned_exit_pos)
                exit_pos = int(next_pos)
                exit_price = float(opens[next_pos]) if next_pos < n else float(closes[pos])
                exit_reason = (
                    f"weak_bounce_giveback_mfe{float(spec.giveback_mfe_trigger):.3%}"
                    f"_floor{float(spec.giveback_floor):.3%}_delta{float(fl['delta_pressure']):.4f}_next_open"
                )
                break

    gross = float(exit_price) / entry_price - 1.0
    net = gross - _entry_cost(args, cost_mult) - _exit_cost(args, cost_mult)
    mae = float(np.nanmin(mtm_low)) if mtm_low else np.nan
    mfe = float(np.nanmax(mtm_high)) if mtm_high else np.nan
    timing = _timing_stats_from_path(mtm_low, mtm_high)

    # Post-entry diagnostics at fixed checkpoints, all measured from closed bars.
    diag: dict[str, object] = {}
    for b in [1, 3, 5, 10, 12, 18, 24, 36, 48]:
        p = min(entry_pos + b, planned_exit_pos)
        if p < n:
            fl3 = recent_flow(flow, max(entry_pos, p - 3 + 1), p)
            fl5 = recent_flow(flow, max(entry_pos, p - 5 + 1), p)
            diag[f"b{b:02d}_close_return"] = float(closes[p]) / entry_price - 1.0
            diag[f"b{b:02d}_mfe"] = _max_window(highs, entry_pos, p) / entry_price - 1.0
            diag[f"b{b:02d}_mae"] = _min_window(lows, entry_pos, p) / entry_price - 1.0
            diag[f"b{b:02d}_delta_pressure_3"] = fl3["delta_pressure"]
            diag[f"b{b:02d}_buy_share_3"] = fl3["buy_notional_share"]
            diag[f"b{b:02d}_delta_pressure_5"] = fl5["delta_pressure"]
            diag[f"b{b:02d}_buy_share_5"] = fl5["buy_notional_share"]

    signal_flow = recent_flow(flow, signal_pos, signal_pos)
    pre15_flow = recent_flow(flow, max(0, signal_pos - 15 + 1), signal_pos)
    pre120_range = _max_window(highs, max(0, signal_pos - 120 + 1), signal_pos) / _min_window(lows, max(0, signal_pos - 120 + 1), signal_pos) - 1.0

    out: dict[str, object] = {
        "valid": True,
        "variant_name": spec.variant_name,
        "candidate_layer": "A0_fp_abs_delta_high",
        "support_mode": "single_swing",
        "entry_mode": "next_open",
        "exit_mode": "time48_cvd_state_exit",
        "stop_name": "no_stop",
        "stop_mode": "none",
        "cost_mult": float(cost_mult),
        "signal_time": event.get("signal_time"),
        "entry_time": idx[int(entry_pos)],
        "exit_time": idx[int(exit_pos)],
        "signal_pos": int(signal_pos),
        "entry_pos": int(entry_pos),
        "exit_pos": int(exit_pos),
        "entry_delay_bars_actual": int(entry_pos - signal_pos),
        "bars_held": int(exit_pos - entry_pos),
        "entry_reason": "next_open",
        "exit_reason": exit_reason,
        "exit_reason_group": _exit_reason_group(exit_reason),
        "early_condition_seen": bool(early_condition_seen),
        "giveback_condition_seen": bool(giveback_condition_seen),
        "giveback_armed": bool(armed_giveback),
        "giveback_armed_pos": armed_giveback_pos,
        "stop_hit": False,
        "target_hit": False,
        "partial_exit_done": False,
        "partial_exit_pos": np.nan,
        "partial_exit_price": np.nan,
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "target_price": np.nan,
        "stop_price": np.nan,
        "stop_pct": np.nan,
        "gross_return_on_equity": float(gross),
        "net_return_on_equity": float(net),
        "mae_on_equity": float(mae),
        "mfe_on_equity": float(mfe),
        "mae_time_bars": timing["mae_time_bars"],
        "mfe_time_bars": timing["mfe_time_bars"],
        "first_positive_high_bars": timing["first_positive_high_bars"],
        "mae_before_mfe_flag": timing["mae_before_mfe_flag"],
        "signal_close": float(event.get("close", np.nan)),
        "signal_open": float(event.get("open", np.nan)),
        "signal_low": float(event.get("low", np.nan)),
        "signal_high": float(event.get("high", np.nan)),
        "swing_level": float(event.get("swing_level", np.nan)),
        "swing_age": float(event.get("swing_age", np.nan)),
        "down_spike_pct": float(event.get("down_spike_pct", np.nan)),
        "close_pos_in_bar": float(event.get("close_pos_in_bar", np.nan)),
        "large_trade_share": float(event.get("large_trade_share", np.nan)),
        "atr_pct": float(event.get("atr_pct", np.nan)),
        "session_bucket": event.get("session_bucket", "NA"),
        "cluster_touch_count_020": event.get("cluster_touch_count_020", np.nan),
        "cluster_touch_count_030": event.get("cluster_touch_count_030", np.nan),
        "signal_delta_pressure": signal_flow["delta_pressure"],
        "signal_buy_share": signal_flow["buy_notional_share"],
        "pre15_delta_pressure": pre15_flow["delta_pressure"],
        "pre15_buy_share": pre15_flow["buy_notional_share"],
        "pre120_range_pct": pre120_range,
    }
    out.update(diag)
    return out


def select_a0_events(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    layer_masks = {k: v.fillna(False).astype(bool) for k, v in build_candidate_layer_masks(events, args).items()}
    support_mask = build_support_mask(events, "single_swing", args).fillna(False).astype(bool)
    selected = events.loc[layer_masks.get("A0_fp_abs_delta_high", pd.Series(False, index=events.index)) & support_mask].copy()
    selected = selected.sort_values("signal_time").reset_index(drop=True)
    return selected


def simulate_specs(
    bars: pd.DataFrame,
    selected: pd.DataFrame,
    specs: Sequence[ExitStateSpec],
    args: argparse.Namespace,
    *,
    cost_mult: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flow = build_flow_arrays(bars)
    max_horizon = max(int(s.time_horizon) for s in specs)
    ev, positions = _event_positions(bars, selected, max_horizon=max_horizon)
    event_records = ev.to_dict("records")
    total = len(event_records) * len(specs)
    progress = ProgressReporter(
        label=f"[simulate] cost {cost_mult:g}x",
        total=max(1, total),
        every=max(1, int(args.progress_every)),
        enabled=not bool(getattr(args, "no_progress", False)),
    )
    done = 0
    trade_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for spec in specs:
        rows: list[dict[str, object]] = []
        counters = {
            "candidate_events": int(len(event_records)),
            "input_events": int(len(event_records)),
            "invalid_events": 0,
            "skipped_invalid": 0,
            "skipped_overlap": 0,
            "valid_trades": 0,
        }
        last_exit_pos = -1
        for event, signal_pos in zip(event_records, positions):
            signal_pos_i = int(signal_pos)
            if signal_pos_i <= last_exit_pos:
                counters["skipped_overlap"] += 1
                done += 1
                progress.update(done)
                continue
            rec = simulate_state_trade(bars, event, signal_pos_i, spec, args, cost_mult=float(cost_mult), flow=flow)
            if rec.get("valid"):
                rows.append(rec)
                last_exit_pos = int(rec.get("exit_pos", signal_pos_i))
                counters["valid_trades"] += 1
            else:
                counters["invalid_events"] += 1
                counters["skipped_invalid"] += 1
            done += 1
            progress.update(done)
        trades = pd.DataFrame(rows)
        if not trades.empty:
            trade_parts.append(trades)
        rec = summarize_trades(trades, args, counters)
        rec.update(
            {
                "variant_name": spec.variant_name,
                "cost_mult": float(cost_mult),
                "exit_state": "baseline" if spec.variant_name == "baseline_time48" else "custom",
                "early_check_bars": spec.early_check_bars,
                "early_mfe_cap": spec.early_mfe_cap,
                "early_cvd_window": spec.early_cvd_window,
                "giveback_mfe_trigger": spec.giveback_mfe_trigger,
                "giveback_floor": spec.giveback_floor,
                "giveback_cvd_window": spec.giveback_cvd_window,
            }
        )
        summary_rows.append(rec)
    progress.close()
    all_trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return all_trades, summary


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def build_delta_vs_baseline(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for cost, grp in summary.groupby("cost_mult", dropna=False):
        base = grp.loc[grp["variant_name"].eq("baseline_time48")]
        if base.empty:
            continue
        b = base.iloc[0]
        for _, row in grp.iterrows():
            rec = row.to_dict()
            for k in ["return_total", "profit_factor", "win_rate", "max_drawdown", "worst_trade", "best_trade", "trades"]:
                if k in row and k in b:
                    try:
                        rec[f"delta_{k}_vs_baseline"] = float(row[k]) - float(b[k])
                    except Exception:
                        rec[f"delta_{k}_vs_baseline"] = np.nan
            rec["passes_basic_screen"] = bool(
                row.get("variant_name") == "baseline_time48"
                or (
                    float(row.get("trades", 0)) >= float(getattr(row, "min_trades_for_compare", 0) or 0)
                    and float(row.get("return_total", -999.0)) > float(b.get("return_total", 0.0))
                    and float(row.get("profit_factor", -999.0)) >= float(b.get("profit_factor", 0.0)) * 0.90
                    and float(row.get("max_drawdown", -999.0)) >= float(b.get("max_drawdown", -999.0)) - 0.02
                )
            )
            rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["cost_mult", "delta_return_total_vs_baseline", "profit_factor"], ascending=[True, False, False]).reset_index(drop=True)


def build_exit_reason_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, grp in trades.groupby(["cost_mult", "variant_name", "exit_reason_group"], dropna=False):
        cost, name, reason = keys
        x = pd.to_numeric(grp["net_return_on_equity"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "cost_mult": cost,
                "variant_name": name,
                "exit_reason_group": reason,
                "trades": int(len(grp)),
                "return_sum": float(x.sum()),
                "mean_return": float(x.mean()) if not x.empty else np.nan,
                "median_return": float(x.median()) if not x.empty else np.nan,
                "win_rate": float((x > 0).mean()) if not x.empty else np.nan,
                "profit_factor": _profit_factor(x),
                "worst_trade": float(x.min()) if not x.empty else np.nan,
                "best_trade": float(x.max()) if not x.empty else np.nan,
                "avg_bars_held": float(pd.to_numeric(grp["bars_held"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["cost_mult", "variant_name", "exit_reason_group"]).reset_index(drop=True)


def classify_loss_type(row: pd.Series) -> str:
    ret = float(row.get("baseline_return", row.get("net_return_on_equity", np.nan)))
    if not math.isfinite(ret) or ret >= 0:
        return "win_or_flat"
    mfe = float(row.get("baseline_mfe", row.get("mfe_on_equity", np.nan)))
    b10_close = float(row.get("b10_close_return", np.nan))
    b10_dp = float(row.get("b10_delta_pressure_5", np.nan))
    if math.isfinite(mfe) and mfe < 0.003:
        return "no_bounce_down"
    if math.isfinite(mfe) and mfe >= 0.006:
        return "bounce_then_giveback"
    if math.isfinite(b10_close) and b10_close < 0 and math.isfinite(b10_dp) and b10_dp < 0:
        return "weak_bounce_fail"
    return "other_loss"


def build_trade_compare(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    base = trades.loc[trades["variant_name"].eq("baseline_time48")].copy()
    if base.empty:
        return pd.DataFrame()
    keep = [
        "cost_mult",
        "signal_time",
        "entry_time",
        "entry_price",
        "net_return_on_equity",
        "exit_time",
        "exit_price",
        "exit_reason_group",
        "bars_held",
        "mfe_on_equity",
        "mae_on_equity",
        "session_bucket",
    ]
    keep.extend(c for c in base.columns if re.fullmatch(r"b\d{2}_(close_return|mfe|mae|delta_pressure_3|delta_pressure_5|buy_share_3|buy_share_5)", str(c)))
    keep = [c for c in dict.fromkeys(keep) if c in base.columns]
    base = base[keep].copy()
    base = base.rename(columns={
        "net_return_on_equity": "baseline_return",
        "exit_time": "baseline_exit_time",
        "exit_price": "baseline_exit_price",
        "exit_reason_group": "baseline_exit_reason_group",
        "bars_held": "baseline_bars_held",
        "mfe_on_equity": "baseline_mfe",
        "mae_on_equity": "baseline_mae",
    })
    parts = []
    for name, grp in trades.loc[~trades["variant_name"].eq("baseline_time48")].groupby("variant_name", dropna=False):
        cur = grp[["cost_mult", "signal_time", "net_return_on_equity", "exit_time", "exit_reason_group", "bars_held"]].copy()
        cur = cur.rename(columns={
            "net_return_on_equity": "variant_return",
            "exit_time": "variant_exit_time",
            "exit_reason_group": "variant_exit_reason_group",
            "bars_held": "variant_bars_held",
        })
        joined = base.merge(cur, on=["cost_mult", "signal_time"], how="inner")
        joined["variant_name"] = name
        joined["delta_return_vs_baseline"] = joined["variant_return"] - joined["baseline_return"]
        joined["baseline_loss_type"] = joined.apply(classify_loss_type, axis=1)
        joined["saved_baseline_loss"] = (joined["baseline_return"] < 0) & (joined["variant_return"] > joined["baseline_return"])
        joined["killed_baseline_winner"] = (joined["baseline_return"] > 0) & (joined["variant_return"] < joined["baseline_return"])
        parts.append(joined)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_loss_type_before_after(compare: pd.DataFrame) -> pd.DataFrame:
    if compare.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, grp in compare.groupby(["cost_mult", "variant_name", "baseline_loss_type"], dropna=False):
        cost, name, loss_type = keys
        rows.append(
            {
                "cost_mult": cost,
                "variant_name": name,
                "baseline_loss_type": loss_type,
                "trades": int(len(grp)),
                "baseline_return_sum": float(pd.to_numeric(grp["baseline_return"], errors="coerce").sum()),
                "variant_return_sum": float(pd.to_numeric(grp["variant_return"], errors="coerce").sum()),
                "delta_return_sum": float(pd.to_numeric(grp["delta_return_vs_baseline"], errors="coerce").sum()),
                "mean_delta": float(pd.to_numeric(grp["delta_return_vs_baseline"], errors="coerce").mean()),
                "saved_baseline_loss_count": int(grp["saved_baseline_loss"].sum()),
                "killed_baseline_winner_count": int(grp["killed_baseline_winner"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["cost_mult", "variant_name", "baseline_loss_type"]).reset_index(drop=True)


def select_sample(trades: pd.DataFrame, n: int) -> pd.DataFrame:
    if trades.empty or n <= 0:
        return pd.DataFrame()
    cols_first = [
        "variant_name", "cost_mult", "signal_time", "entry_time", "exit_time", "entry_price", "exit_price",
        "net_return_on_equity", "exit_reason_group", "exit_reason", "bars_held", "mfe_on_equity", "mae_on_equity",
        "b05_close_return", "b05_delta_pressure_5", "b05_buy_share_5",
        "b10_close_return", "b10_delta_pressure_5", "b10_buy_share_5",
        "b18_close_return", "b18_delta_pressure_5", "b18_buy_share_5",
        "session_bucket", "atr_pct", "down_spike_pct", "close_pos_in_bar",
    ]
    cols = [c for c in cols_first if c in trades.columns] + [c for c in trades.columns if c not in cols_first]
    return trades.sort_values(["variant_name", "cost_mult", "signal_time"]).head(int(n))[cols]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {SCRIPT_NAME} {SCRIPT_VERSION}", flush=True)
    print("[scope] research-only; no hard stop; exits are closed-bar CVD/price states then next-open execution", flush=True)
    print(f"[args] out_dir={out_dir}", flush=True)

    print(f"[load] trade bars {args.symbol} {args.timeframe} {args.warmup_start_date}->{args.end_date}", flush=True)
    bars = load_trade_bars(args)
    print("[events] build existing Low Sweep V1 A0 events/context", flush=True)
    events = prepare_events_and_context(bars, args)
    selected = select_a0_events(events, args)
    print(f"[select] A0_fp_abs_delta_high + single_swing selected_events={len(selected):,}", flush=True)

    specs = build_exit_state_specs(args)
    print(f"[specs] exit_state_specs={len(specs):,}; no hard stop; time48 fallback retained", flush=True)

    all_trades_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    for cost_mult in _parse_float_list(args.research_cost_mults):
        trades, summary = simulate_specs(bars, selected, specs, args, cost_mult=float(cost_mult))
        all_trades_parts.append(trades)
        summary_parts.append(summary)

    all_trades = pd.concat(all_trades_parts, ignore_index=True) if all_trades_parts else pd.DataFrame()
    summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    delta = build_delta_vs_baseline(summary)
    yearly = summarize_by_period(all_trades, "year") if not all_trades.empty else pd.DataFrame()
    monthly = summarize_by_period(all_trades, "month") if not all_trades.empty else pd.DataFrame()
    exit_reasons = build_exit_reason_summary(all_trades)
    compare = build_trade_compare(all_trades)
    loss_type = build_loss_type_before_after(compare)

    # Worst/best diagnostics.
    worst = all_trades.sort_values("net_return_on_equity", ascending=True).groupby(["cost_mult", "variant_name"], dropna=False).head(10).reset_index(drop=True) if not all_trades.empty else pd.DataFrame()
    best = all_trades.sort_values("net_return_on_equity", ascending=False).groupby(["cost_mult", "variant_name"], dropna=False).head(10).reset_index(drop=True) if not all_trades.empty else pd.DataFrame()

    # Selected top candidates table: keep baseline plus top custom variants for fast manual review.
    selected_summary = pd.DataFrame()
    if not delta.empty:
        selected_summary = delta.loc[(delta["variant_name"].eq("baseline_time48")) | (pd.to_numeric(delta.get("trades"), errors="coerce") >= int(args.min_trades_for_compare))].copy()
        selected_summary = selected_summary.sort_values(["cost_mult", "delta_return_total_vs_baseline", "profit_factor"], ascending=[True, False, False]).groupby("cost_mult", dropna=False).head(25).reset_index(drop=True)

    write_csv(summary, out_dir / "01_summary.csv", "01_summary.csv")
    write_csv(delta, out_dir / "02_delta_vs_baseline.csv", "02_delta_vs_baseline.csv")
    write_csv(yearly, out_dir / "03_yearly.csv", "03_yearly.csv")
    write_csv(monthly, out_dir / "04_monthly.csv", "04_monthly.csv")
    write_csv(exit_reasons, out_dir / "05_exit_reason_summary.csv", "05_exit_reason_summary.csv")
    write_csv(compare, out_dir / "06_trade_compare_vs_baseline.csv", "06_trade_compare_vs_baseline.csv")
    write_csv(loss_type, out_dir / "07_loss_type_before_after.csv", "07_loss_type_before_after.csv")
    write_csv(worst, out_dir / "08_worst_trades_by_variant.csv", "08_worst_trades_by_variant.csv")
    write_csv(best, out_dir / "09_best_trades_by_variant.csv", "09_best_trades_by_variant.csv")
    write_csv(selected_summary, out_dir / "10_selected_candidate_summary.csv", "10_selected_candidate_summary.csv")
    write_csv(select_sample(all_trades, int(args.save_trade_sample)), out_dir / "trade_sample.csv", "trade_sample.csv")

    meta = pd.DataFrame([
        {
            "script": SCRIPT_NAME,
            "version": SCRIPT_VERSION,
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "warmup_start_date": args.warmup_start_date,
            "selected_events": int(len(selected)),
            "exit_specs": int(len(specs)),
            "cost_mults": args.research_cost_mults,
            "causality": "closed-bar state checks; next-bar open execution; no hard stop; time48 fallback",
        }
    ])
    write_csv(meta, out_dir / "99_meta.csv", "99_meta.csv")

    print("[done] wrote reports", flush=True)
    for name in [
        "01_summary.csv",
        "02_delta_vs_baseline.csv",
        "07_loss_type_before_after.csv",
        "10_selected_candidate_summary.csv",
        "trade_sample.csv",
    ]:
        print(f"  - {out_dir / name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
