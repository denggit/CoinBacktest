#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Low Sweep A0 overlap add-on research.

Research question
-----------------
The current Low Sweep MF research/backtest uses a single-position model: when a
new A0_fp_abs_delta_high + single_swing signal appears while the previous A0
trade is still open, the new signal is counted as ``skipped_overlap``.

This script does not promote a new backtest version and does not change the
portfolio wrapper.  It reuses the existing CoinBacktest Low Sweep data/event
pipeline, reconstructs the current parent trades, then asks a diagnostic
question:

    If an overlap A0 signal appears during an active parent trade, what would a
    one-time add-on leg have done?

The default research is intentionally small and interpretable:

- parent exits: time48 and mfe_lock_15_05_time48;
- add-on entry: the overlap signal's existing next_open entry;
- add-on exit diagnostic A: close the add-on together with the parent trade;
- add-on exit diagnostic B: simulate the add-on as its own independent A0 trade;
- add weights: 0.25 / 0.50 / 1.00 of the original MF leg.

Important caveat
----------------
This is a path diagnostic, not a formal portfolio simulator.  The combined rows
use the existing parent trade sequence and estimate the incremental add-on
return/MAE/MFE.  It is meant to answer whether overlap signals look like useful
add confirmations or mostly like averaging down into weak trades.
"""

from __future__ import annotations

import argparse
import json
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

from backtest.mf.low_sweep import low_sweep_V1_a0_footprint_backtest as low_v1  # noqa: E402
from research.low_sweep_a_upgrade_research import (  # noqa: E402
    UpgradeVariant,
    _entry_cost,
    _entry_from_mode,
    _exit_cost,
    build_candidate_layer_masks,
    build_market_cache,
    build_support_mask,
    parse_stop_specs,
    simulate_upgrade_trade,
    summarize_by_period,
    summarize_trades,
    write_csv,
)
from src.research_common.progress import ProgressReporter  # noqa: E402

SCRIPT_NAME = "low_sweep_A0_overlap_add_research"
SCRIPT_VERSION = "v3_verified_progress_signature_fix"
DEFAULT_OUT_DIR = "data/reports/research/low_sweep/A0_overlap_add_research"
PARENT_TIME48 = "parent_time48"
PARENT_MFE_LOCK = "parent_mfe_lock_15_05_time48"


@dataclass(frozen=True)
class ParentSpec:
    name: str
    exit_mode: str


@dataclass(frozen=True)
class AddLeg:
    signal_idx: int
    signal_pos: int
    signal_time: pd.Timestamp
    entry_pos: int
    entry_time: pd.Timestamp
    entry_price: float
    exit_pos: int
    exit_time: pd.Timestamp
    exit_price: float
    net_return: float
    gross_return: float
    own_exit_pos: int | None
    own_exit_time: pd.Timestamp | None
    own_exit_price: float | None
    own_net_return: float | None
    own_exit_reason: str
    parent_bars_since_entry: int
    parent_bars_to_exit: int
    parent_unreal_close_ret: float
    parent_mfe_before_add: float
    parent_mae_before_add: float
    parent_armed_before_add: bool
    add_signal_down_spike_pct: float
    add_signal_atr_pct: float
    add_signal_large_trade_share: float
    add_signal_close_pos_in_bar: float
    add_signal_session_bucket: object


# ---------------------------------------------------------------------------
# Args / common helpers
# ---------------------------------------------------------------------------


def _split_csv(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for part in _split_csv(raw):
        val = float(part)
        if math.isfinite(val):
            out.append(val)
    return sorted(set(out))


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for part in _split_csv(raw):
        val = int(part)
        out.append(val)
    return sorted(set(out))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    own.add_argument("--parent-exit-modes", default="time48,mfe_lock_15_05_time48")
    own.add_argument("--add-weights", default="0.25,0.5,1.0")
    own.add_argument(
        "--max-add-counts",
        default="1,999",
        help="Per-parent max add legs. 999 means all eligible overlap signals during the parent trade.",
    )
    own.add_argument("--cost-mults", default="1.0,1.5,2.0")
    own.add_argument("--save-trades", type=int, default=200000)
    own.add_argument("--save-events", type=int, default=5000)
    own.add_argument("--save-trade-sample", type=int, default=200000)
    own.add_argument("--min-bars-after-parent-entry", type=int, default=1)
    own.add_argument("--min-bars-before-parent-exit", type=int, default=1)
    own.add_argument("--progress-every", type=int, default=100)
    own.add_argument("--no-progress", action="store_true")
    known, rest = own.parse_known_args(argv)

    defaults = [
        "--out-dir",
        str(known.out_dir),
        "--candidate-layers",
        "A0_fp_abs_delta_high",
        "--support-modes",
        "single_swing",
        "--entry-modes",
        "next_open",
        "--exit-modes",
        "time48,mfe_lock_15_05_time48",
        "--upgrade-stop-specs",
        "no_stop",
        "--context-sources",
        "trade_bar,footprint",
        "--micro-timeframes",
        "",
        "--micro-load-mode",
        "local",
        "--save-trades",
        str(int(known.save_trades)),
        "--save-events",
        str(int(known.save_events)),
        "--skip-full-report",
    ]
    args = low_v1.parse_args(defaults + list(rest))
    for k, v in vars(known).items():
        setattr(args, k, v)
    return args


def _no_stop(args: argparse.Namespace):
    stops = {s.name: s for s in parse_stop_specs("no_stop")}
    return stops["no_stop"]


def _parent_specs(raw: str) -> list[ParentSpec]:
    out: list[ParentSpec] = []
    for mode in _split_csv(raw):
        if mode == "time48":
            out.append(ParentSpec(PARENT_TIME48, "time48"))
        elif mode == "mfe_lock_15_05_time48":
            out.append(ParentSpec(PARENT_MFE_LOCK, "mfe_lock_15_05_time48"))
        else:
            safe = str(mode).replace("__", "_").replace(",", "_")
            out.append(ParentSpec(f"parent_{safe}", mode))
    return out


def _variant(parent: ParentSpec, stop_spec: object, *, entry_mode: str = "next_open") -> UpgradeVariant:
    return UpgradeVariant(
        variant_name=parent.name,
        candidate_layer="A0_fp_abs_delta_high",
        support_mode="single_swing",
        entry_mode=entry_mode,
        exit_mode=parent.exit_mode,
        stop_spec=stop_spec,
    )


def _select_a0_events(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    masks = build_candidate_layer_masks(events, args)
    layer = masks.get("A0_fp_abs_delta_high", pd.Series(False, index=events.index)).fillna(False).astype(bool)
    support = build_support_mask(events, "single_swing", args).fillna(False).astype(bool)
    out = events.loc[layer & support].copy()
    if "signal_time" in out.columns:
        out = out.sort_values("signal_time").reset_index(drop=True)
    return out


def _event_positions(bars: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if events.empty:
        return events.copy(), np.asarray([], dtype=int)
    times = pd.DatetimeIndex(pd.to_datetime(events["signal_time"], errors="coerce"))
    pos = bars.index.get_indexer(times)
    valid = (pos >= 0) & ((pos + 2) < len(bars))
    return events.loc[valid].copy().reset_index(drop=True), pos[valid]


def _safe_float(x: object, default: float = np.nan) -> float:
    try:
        val = float(x)
    except Exception:
        return default
    return val if math.isfinite(val) else default


def _equity_and_dd(x: pd.Series, start: float = 1.0) -> tuple[pd.Series, pd.Series]:
    vals = pd.to_numeric(x, errors="coerce").fillna(0.0)
    eq = float(start) * (1.0 + vals).cumprod()
    dd = eq / eq.cummax() - 1.0
    return eq, dd


def _profit_factor(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    gp = float(vals[vals > 0].sum())
    gl = float(-vals[vals < 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else float("nan")
    return gp / gl


def _max_consecutive_losses(x: pd.Series) -> int:
    cur = best = 0
    for val in pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float):
        if val < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def ensure_summary_compatible(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults: dict[str, object] = {
        "stop_hit": False,
        "target_hit": False,
        "partial_exit_done": False,
        "partial_exit_pos": np.nan,
        "partial_exit_price": np.nan,
        "mae_time_bars": np.nan,
        "mfe_time_bars": np.nan,
        "first_positive_high_bars": np.nan,
        "mae_before_mfe_flag": False,
        "bars_held": np.nan,
        "mae_on_equity": np.nan,
        "mfe_on_equity": np.nan,
    }
    for col, val in defaults.items():
        if col not in out.columns:
            out[col] = val
    return out


# ---------------------------------------------------------------------------
# Parent/overlap reconstruction
# ---------------------------------------------------------------------------


def simulate_parent_sequence(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    positions: np.ndarray,
    parent: ParentSpec,
    args: argparse.Namespace,
    market,
    *,
    cost_mult: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate the current single-position parent sequence and overlap list."""

    variant = _variant(parent, _no_stop(args))
    event_records = events.to_dict("records")
    parent_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    active_parent_idx: int | None = None
    last_exit_pos = -1
    skipped_invalid = 0

    progress = None
    if not bool(getattr(args, "no_progress", False)):
        progress = ProgressReporter(label=f"[parent] {parent.name}", total=len(event_records), every=max(1, int(args.progress_every)), enabled=not bool(getattr(args, "no_progress", False)))

    for i, (event, signal_pos) in enumerate(zip(event_records, positions)):
        signal_pos = int(signal_pos)
        if signal_pos <= last_exit_pos and active_parent_idx is not None:
            active = parent_rows[active_parent_idx]
            overlap_rows.append(
                {
                    "parent_name": parent.name,
                    "parent_exit_mode": parent.exit_mode,
                    "overlap_event_idx": int(i),
                    "overlap_signal_pos": int(signal_pos),
                    "overlap_signal_time": event.get("signal_time"),
                    "parent_trade_idx": int(active_parent_idx),
                    "parent_signal_time": active.get("signal_time"),
                    "parent_entry_time": active.get("entry_time"),
                    "parent_exit_time": active.get("exit_time"),
                    "parent_signal_pos": active.get("signal_pos"),
                    "parent_entry_pos": active.get("entry_pos"),
                    "parent_exit_pos": active.get("exit_pos"),
                    "parent_entry_price": active.get("entry_price"),
                    "parent_exit_price": active.get("exit_price"),
                    "parent_net_return": active.get("net_return_on_equity"),
                    "parent_exit_reason": active.get("exit_reason"),
                    **{f"add_signal_{k}": event.get(k) for k in [
                        "down_spike_pct",
                        "close_pos_in_bar",
                        "large_trade_share",
                        "atr_pct",
                        "session_bucket",
                        "fp_abs_delta_pressure",
                        "fp_delta_pressure",
                        "swing_age",
                        "cluster_touch_count_020",
                    ]},
                }
            )
            if progress:
                progress.update(i + 1)
            continue

        rec = simulate_upgrade_trade(bars, event, signal_pos, variant, args, cost_mult=cost_mult, market=market)
        if not rec.get("valid"):
            skipped_invalid += 1
            if progress:
                progress.update(i + 1)
            continue
        rec = dict(rec)
        rec["parent_name"] = parent.name
        rec["parent_exit_mode"] = parent.exit_mode
        rec["parent_trade_idx"] = int(len(parent_rows))
        rec["cost_mult"] = float(cost_mult)
        rec["skipped_invalid_before_or_at_parent"] = int(skipped_invalid)
        parent_rows.append(rec)
        active_parent_idx = int(len(parent_rows) - 1)
        last_exit_pos = int(rec.get("exit_pos", signal_pos))
        if progress:
            progress.update(i + 1)
    if progress:
        progress.close()

    parents = ensure_summary_compatible(pd.DataFrame(parent_rows))
    overlaps = pd.DataFrame(overlap_rows)
    return parents, overlaps


# ---------------------------------------------------------------------------
# Add-on diagnostics
# ---------------------------------------------------------------------------


def _parent_state_before_pos(parent_row: pd.Series | dict[str, object], market, pos: int) -> dict[str, object]:
    entry_pos = int(parent_row.get("entry_pos"))
    entry_price = float(parent_row.get("entry_price"))
    start = max(entry_pos, 0)
    stop = min(int(pos), len(market.index) - 1)
    if stop < start:
        return {
            "parent_mfe_before_add": np.nan,
            "parent_mae_before_add": np.nan,
            "parent_unreal_close_ret": np.nan,
            "parent_armed_before_add": False,
        }
    highs = market.high[start : stop + 1]
    lows = market.low[start : stop + 1]
    high_ret = highs / entry_price - 1.0
    low_ret = lows / entry_price - 1.0
    close_ret = float(market.close[stop] / entry_price - 1.0)
    mfe = float(np.nanmax(high_ret)) if high_ret.size else np.nan
    mae = float(np.nanmin(low_ret)) if low_ret.size else np.nan
    return {
        "parent_mfe_before_add": mfe,
        "parent_mae_before_add": mae,
        "parent_unreal_close_ret": close_ret,
        "parent_armed_before_add": bool(math.isfinite(mfe) and mfe >= 0.015),
    }


def _simulate_add_leg(
    bars: pd.DataFrame,
    add_event: dict[str, object],
    add_signal_pos: int,
    parent_row: pd.Series,
    parent: ParentSpec,
    args: argparse.Namespace,
    market,
    *,
    cost_mult: float,
) -> AddLeg | None:
    ok, entry_pos, entry_price, reason = _entry_from_mode(bars, add_event, int(add_signal_pos), "next_open", args, market)
    if not ok or entry_pos is None or entry_price is None:
        return None

    parent_exit_pos = int(parent_row["exit_pos"])
    parent_exit_price = float(parent_row["exit_price"])
    if int(entry_pos) >= parent_exit_pos:
        return None
    if int(entry_pos) - int(parent_row["entry_pos"]) < int(args.min_bars_after_parent_entry):
        return None
    if parent_exit_pos - int(entry_pos) < int(args.min_bars_before_parent_exit):
        return None

    gross = float(parent_exit_price / float(entry_price) - 1.0)
    net = float(gross - _entry_cost(args, cost_mult) - _exit_cost(args, cost_mult))

    own_variant = _variant(parent, _no_stop(args))
    own_rec = simulate_upgrade_trade(bars, add_event, int(add_signal_pos), own_variant, args, cost_mult=cost_mult, market=market)
    own_exit_pos: int | None = None
    own_exit_time: pd.Timestamp | None = None
    own_exit_price: float | None = None
    own_net: float | None = None
    own_exit_reason = ""
    if own_rec.get("valid"):
        own_exit_pos = int(own_rec.get("exit_pos"))
        own_exit_time = pd.Timestamp(own_rec.get("exit_time"))
        own_exit_price = float(own_rec.get("exit_price"))
        own_net = float(own_rec.get("net_return_on_equity"))
        own_exit_reason = str(own_rec.get("exit_reason", ""))

    state = _parent_state_before_pos(parent_row, market, int(add_signal_pos))
    event_idx = int(add_event.get("_event_idx", -1))
    return AddLeg(
        signal_idx=event_idx,
        signal_pos=int(add_signal_pos),
        signal_time=pd.Timestamp(add_event.get("signal_time")),
        entry_pos=int(entry_pos),
        entry_time=pd.Timestamp(market.index[int(entry_pos)]),
        entry_price=float(entry_price),
        exit_pos=parent_exit_pos,
        exit_time=pd.Timestamp(parent_row["exit_time"]),
        exit_price=parent_exit_price,
        net_return=net,
        gross_return=gross,
        own_exit_pos=own_exit_pos,
        own_exit_time=own_exit_time,
        own_exit_price=own_exit_price,
        own_net_return=own_net,
        own_exit_reason=own_exit_reason,
        parent_bars_since_entry=int(add_signal_pos - int(parent_row["entry_pos"])),
        parent_bars_to_exit=int(parent_exit_pos - add_signal_pos),
        parent_unreal_close_ret=float(state["parent_unreal_close_ret"]),
        parent_mfe_before_add=float(state["parent_mfe_before_add"]),
        parent_mae_before_add=float(state["parent_mae_before_add"]),
        parent_armed_before_add=bool(state["parent_armed_before_add"]),
        add_signal_down_spike_pct=_safe_float(add_event.get("down_spike_pct")),
        add_signal_atr_pct=_safe_float(add_event.get("atr_pct")),
        add_signal_large_trade_share=_safe_float(add_event.get("large_trade_share")),
        add_signal_close_pos_in_bar=_safe_float(add_event.get("close_pos_in_bar")),
        add_signal_session_bucket=add_event.get("session_bucket"),
    )


def _combined_path_metrics(parent_row: pd.Series, add_legs: Sequence[AddLeg], market, weight: float) -> dict[str, object]:
    entry_pos = int(parent_row["entry_pos"])
    exit_pos = int(parent_row["exit_pos"])
    entry_price = float(parent_row["entry_price"])
    lows: list[float] = []
    highs: list[float] = []
    for pos in range(entry_pos, exit_pos + 1):
        low_ret = float(market.low[pos] / entry_price - 1.0)
        high_ret = float(market.high[pos] / entry_price - 1.0)
        for leg in add_legs:
            if pos >= int(leg.entry_pos):
                low_ret += float(weight) * float(market.low[pos] / float(leg.entry_price) - 1.0)
                high_ret += float(weight) * float(market.high[pos] / float(leg.entry_price) - 1.0)
        lows.append(low_ret)
        highs.append(high_ret)
    low_arr = np.asarray(lows, dtype=float)
    high_arr = np.asarray(highs, dtype=float)
    mae = float(np.nanmin(low_arr)) if low_arr.size else np.nan
    mfe = float(np.nanmax(high_arr)) if high_arr.size else np.nan
    mae_t = int(np.nanargmin(low_arr)) if low_arr.size and np.isfinite(low_arr).any() else np.nan
    mfe_t = int(np.nanargmax(high_arr)) if high_arr.size and np.isfinite(high_arr).any() else np.nan
    first_pos = np.flatnonzero(high_arr > 0) if high_arr.size else np.asarray([])
    return {
        "mae_on_equity": mae,
        "mfe_on_equity": mfe,
        "mae_time_bars": mae_t,
        "mfe_time_bars": mfe_t,
        "first_positive_high_bars": int(first_pos[0]) if first_pos.size else np.nan,
        "mae_before_mfe_flag": bool(math.isfinite(float(mae_t)) and math.isfinite(float(mfe_t)) and int(mae_t) <= int(mfe_t)) if not (isinstance(mae_t, float) and math.isnan(mae_t)) else False,
    }


def _leg_to_row(parent: ParentSpec, parent_row: pd.Series, leg: AddLeg) -> dict[str, object]:
    return {
        "parent_name": parent.name,
        "parent_exit_mode": parent.exit_mode,
        "parent_trade_idx": int(parent_row["parent_trade_idx"]),
        "parent_signal_time": parent_row.get("signal_time"),
        "parent_entry_time": parent_row.get("entry_time"),
        "parent_exit_time": parent_row.get("exit_time"),
        "parent_entry_price": parent_row.get("entry_price"),
        "parent_exit_price": parent_row.get("exit_price"),
        "parent_net_return": parent_row.get("net_return_on_equity"),
        "parent_exit_reason": parent_row.get("exit_reason"),
        "add_signal_idx": leg.signal_idx,
        "add_signal_pos": leg.signal_pos,
        "add_signal_time": leg.signal_time,
        "add_entry_pos": leg.entry_pos,
        "add_entry_time": leg.entry_time,
        "add_entry_price": leg.entry_price,
        "add_exit_pos_same_parent": leg.exit_pos,
        "add_exit_time_same_parent": leg.exit_time,
        "add_exit_price_same_parent": leg.exit_price,
        "add_gross_return_same_parent": leg.gross_return,
        "add_net_return_same_parent": leg.net_return,
        "add_own_exit_pos": leg.own_exit_pos,
        "add_own_exit_time": leg.own_exit_time,
        "add_own_exit_price": leg.own_exit_price,
        "add_own_net_return": leg.own_net_return,
        "add_own_exit_reason": leg.own_exit_reason,
        "parent_bars_since_entry": leg.parent_bars_since_entry,
        "parent_bars_to_exit": leg.parent_bars_to_exit,
        "parent_unreal_close_ret_at_add_signal": leg.parent_unreal_close_ret,
        "parent_mfe_before_add": leg.parent_mfe_before_add,
        "parent_mae_before_add": leg.parent_mae_before_add,
        "parent_armed_before_add": leg.parent_armed_before_add,
        "add_signal_down_spike_pct": leg.add_signal_down_spike_pct,
        "add_signal_atr_pct": leg.add_signal_atr_pct,
        "add_signal_large_trade_share": leg.add_signal_large_trade_share,
        "add_signal_close_pos_in_bar": leg.add_signal_close_pos_in_bar,
        "add_signal_session_bucket": leg.add_signal_session_bucket,
        "add_same_parent_would_win": bool(leg.net_return > 0),
        "add_own_would_win": bool(leg.own_net_return is not None and leg.own_net_return > 0),
    }


def build_add_diagnostics(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    positions: np.ndarray,
    parents: pd.DataFrame,
    parent: ParentSpec,
    args: argparse.Namespace,
    market,
    *,
    cost_mult: float,
) -> tuple[pd.DataFrame, dict[int, list[AddLeg]]]:
    if parents.empty:
        return pd.DataFrame(), {}
    ev = events.copy().reset_index(drop=True)
    ev["_event_idx"] = np.arange(len(ev), dtype=int)
    event_records = ev.to_dict("records")
    by_parent: dict[int, list[AddLeg]] = {int(x): [] for x in parents["parent_trade_idx"].tolist()}
    leg_rows: list[dict[str, object]] = []

    # Pointer-based scan because both parent rows and event positions are sorted.
    for _, prow in parents.iterrows():
        pidx = int(prow["parent_trade_idx"])
        start_pos = int(prow["signal_pos"]) + 1
        end_pos = int(prow["exit_pos"])
        mask = (positions >= start_pos) & (positions <= end_pos)
        overlap_indices = np.flatnonzero(mask)
        for oi in overlap_indices:
            add_event = event_records[int(oi)]
            # Skip the parent signal itself and only consider overlap signals that
            # would have been skipped by the current single-position model.
            if int(add_event.get("_event_idx", -1)) == int(prow.get("parent_trade_idx", -999999)):
                continue
            leg = _simulate_add_leg(
                bars,
                add_event,
                int(positions[int(oi)]),
                prow,
                parent,
                args,
                market,
                cost_mult=cost_mult,
            )
            if leg is None:
                continue
            by_parent[pidx].append(leg)
            leg_rows.append(_leg_to_row(parent, prow, leg))
    return pd.DataFrame(leg_rows), by_parent


def build_combined_trade_rows(
    parents: pd.DataFrame,
    add_by_parent: dict[int, list[AddLeg]],
    parent: ParentSpec,
    args: argparse.Namespace,
    market,
    *,
    cost_mult: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    weights = _parse_float_list(args.add_weights)
    max_add_counts = _parse_int_list(args.max_add_counts)

    # Baseline rows are included in the same table for simple delta comparisons.
    for _, prow in parents.iterrows():
        base = dict(prow)
        base["variant_name"] = f"{parent.name}__no_add"
        base["add_scheme"] = "no_add"
        base["add_weight"] = 0.0
        base["max_add_count"] = 0
        base["add_count"] = 0
        base["add_return_sum_unweighted"] = 0.0
        base["add_return_sum_weighted"] = 0.0
        base["combined_return_delta_vs_parent"] = 0.0
        base["cost_mult"] = float(cost_mult)
        rows.append(base)

    for _, prow in parents.iterrows():
        pidx = int(prow["parent_trade_idx"])
        legs_all = sorted(add_by_parent.get(pidx, []), key=lambda x: x.entry_pos)
        for max_count in max_add_counts:
            selected = legs_all[: int(max_count)] if int(max_count) < 999 else legs_all
            for weight in weights:
                add_net = float(sum(float(leg.net_return) for leg in selected))
                add_weighted = float(weight * add_net)
                combined_net = float(prow["net_return_on_equity"]) + add_weighted
                base = dict(prow)
                scheme = "add_all_same_parent_exit" if int(max_count) >= 999 else f"add_first{int(max_count)}_same_parent_exit"
                base["variant_name"] = f"{parent.name}__{scheme}__w{int(round(weight * 100)):03d}"
                base["add_scheme"] = scheme
                base["add_weight"] = float(weight)
                base["max_add_count"] = int(max_count)
                base["add_count"] = int(len(selected))
                base["add_return_sum_unweighted"] = add_net
                base["add_return_sum_weighted"] = add_weighted
                base["combined_return_delta_vs_parent"] = add_weighted
                base["net_return_on_equity"] = combined_net
                base["gross_return_on_equity"] = float(prow.get("gross_return_on_equity", np.nan)) + add_weighted
                base["exit_reason"] = f"{prow.get('exit_reason', '')}|{scheme}_w{weight:g}"
                base["cost_mult"] = float(cost_mult)
                metrics = _combined_path_metrics(prow, selected, market, float(weight))
                base.update(metrics)
                rows.append(base)
    return ensure_summary_compatible(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def summarize_variant_table(trades: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if trades.empty:
        return pd.DataFrame()
    for name, grp in trades.groupby("variant_name", dropna=False):
        rec = summarize_trades(ensure_summary_compatible(grp), args, extra={})
        rec["variant_name"] = name
        for col in ["parent_name", "parent_exit_mode", "add_scheme", "add_weight", "max_add_count", "cost_mult"]:
            if col in grp.columns:
                vals = grp[col].dropna().unique()
                rec[col] = vals[0] if len(vals) == 1 else "mixed"
        rec["total_add_count"] = int(pd.to_numeric(grp.get("add_count", 0), errors="coerce").fillna(0).sum())
        rec["parents_with_add"] = int((pd.to_numeric(grp.get("add_count", 0), errors="coerce").fillna(0) > 0).sum())
        rec["mean_add_return_sum_weighted"] = float(pd.to_numeric(grp.get("add_return_sum_weighted", np.nan), errors="coerce").mean())
        rec["sum_add_return_weighted"] = float(pd.to_numeric(grp.get("add_return_sum_weighted", np.nan), errors="coerce").sum())
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["parent_name", "cost_mult", "return_total", "profit_factor"], ascending=[True, True, False, False]).reset_index(drop=True)
    return out


def delta_vs_baseline(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    keys = ["parent_name", "cost_mult"]
    for key, grp in summary.groupby(keys, dropna=False):
        base = grp.loc[grp["add_scheme"].eq("no_add")]
        if base.empty:
            continue
        b = base.iloc[0]
        for _, row in grp.iterrows():
            rec = row.to_dict()
            for col in ["return_total", "profit_factor", "win_rate", "max_drawdown", "trades", "worst_trade", "best_trade"]:
                if col in row and col in b:
                    try:
                        rec[f"delta_{col}_vs_no_add"] = float(row[col]) - float(b[col])
                    except Exception:
                        rec[f"delta_{col}_vs_no_add"] = np.nan
            rows.append(rec)
    return pd.DataFrame(rows).reset_index(drop=True)


def summarize_add_legs(add_legs: pd.DataFrame) -> pd.DataFrame:
    if add_legs.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for parent_name, grp in add_legs.groupby("parent_name", dropna=False):
        same = pd.to_numeric(grp["add_net_return_same_parent"], errors="coerce")
        own = pd.to_numeric(grp["add_own_net_return"], errors="coerce")
        parent_unreal = pd.to_numeric(grp["parent_unreal_close_ret_at_add_signal"], errors="coerce")
        rows.append(
            {
                "parent_name": parent_name,
                "overlap_add_legs": int(len(grp)),
                "same_parent_exit_sum": float(same.sum()),
                "same_parent_exit_mean": float(same.mean()),
                "same_parent_exit_median": float(same.median()),
                "same_parent_exit_win_rate": float((same > 0).mean()),
                "same_parent_exit_pf": _profit_factor(same),
                "own_exit_sum": float(own.sum()),
                "own_exit_mean": float(own.mean()),
                "own_exit_median": float(own.median()),
                "own_exit_win_rate": float((own > 0).mean()),
                "own_exit_pf": _profit_factor(own),
                "parent_unreal_at_add_mean": float(parent_unreal.mean()),
                "parent_unreal_at_add_median": float(parent_unreal.median()),
                "parent_losing_at_add_rate": float((parent_unreal < 0).mean()),
                "parent_armed_before_add_rate": float(pd.Series(grp.get("parent_armed_before_add", False)).astype(bool).mean()),
                "add_signal_atr_pct_median": float(pd.to_numeric(grp.get("add_signal_atr_pct"), errors="coerce").median()),
                "add_signal_down_spike_pct_median": float(pd.to_numeric(grp.get("add_signal_down_spike_pct"), errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows).sort_values("parent_name").reset_index(drop=True)


def breakdown_add_context(add_legs: pd.DataFrame) -> pd.DataFrame:
    if add_legs.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    tmp = add_legs.copy()
    tmp["parent_state_at_add"] = np.where(pd.to_numeric(tmp["parent_unreal_close_ret_at_add_signal"], errors="coerce") >= 0, "parent_green", "parent_red")
    tmp["parent_armed_bucket"] = np.where(tmp.get("parent_armed_before_add", False).astype(bool), "armed", "not_armed")
    for col in ["parent_state_at_add", "parent_armed_bucket", "add_signal_session_bucket"]:
        if col not in tmp.columns:
            continue
        for (parent_name, bucket), grp in tmp.groupby(["parent_name", col], dropna=False):
            same = pd.to_numeric(grp["add_net_return_same_parent"], errors="coerce")
            own = pd.to_numeric(grp["add_own_net_return"], errors="coerce")
            rows.append(
                {
                    "parent_name": parent_name,
                    "breakdown_col": col,
                    "bucket": bucket,
                    "add_legs": int(len(grp)),
                    "same_parent_exit_sum": float(same.sum()),
                    "same_parent_exit_mean": float(same.mean()),
                    "same_parent_exit_win_rate": float((same > 0).mean()),
                    "same_parent_exit_pf": _profit_factor(same),
                    "own_exit_sum": float(own.sum()),
                    "own_exit_mean": float(own.mean()),
                    "own_exit_win_rate": float((own > 0).mean()),
                    "own_exit_pf": _profit_factor(own),
                }
            )
    return pd.DataFrame(rows).sort_values(["parent_name", "breakdown_col", "bucket"]).reset_index(drop=True)


def write_outputs(
    out_dir: Path,
    parents_all: pd.DataFrame,
    add_legs_all: pd.DataFrame,
    combined_all: pd.DataFrame,
    summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    yearly = summarize_by_period(combined_all, "year") if not combined_all.empty else pd.DataFrame()
    monthly = summarize_by_period(combined_all, "month") if not combined_all.empty else pd.DataFrame()
    add_summary = summarize_add_legs(add_legs_all)
    add_breakdown = breakdown_add_context(add_legs_all)
    delta = delta_vs_baseline(summary)

    write_csv(summary, out_dir / "01_add_scheme_summary.csv", "add_scheme_summary")
    write_csv(delta, out_dir / "02_add_scheme_delta_vs_no_add.csv", "add_scheme_delta")
    write_csv(add_summary, out_dir / "03_overlap_add_leg_summary.csv", "overlap_add_leg_summary")
    write_csv(add_breakdown, out_dir / "04_overlap_add_context_breakdown.csv", "overlap_add_context_breakdown")
    write_csv(yearly, out_dir / "05_yearly.csv", "yearly")
    write_csv(monthly, out_dir / "06_monthly.csv", "monthly")
    write_csv(add_legs_all, out_dir / "07_overlap_add_legs.csv", "overlap_add_legs")
    write_csv(parents_all, out_dir / "08_parent_trades.csv", "parent_trades")

    sample_n = int(getattr(args, "save_trade_sample", 0) or 0)
    if sample_n > 0:
        sample_cols = [
            "variant_name",
            "parent_name",
            "add_scheme",
            "add_weight",
            "max_add_count",
            "add_count",
            "signal_time",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "exit_reason",
            "net_return_on_equity",
            "parent_net_return",
            "add_return_sum_weighted",
            "mae_on_equity",
            "mfe_on_equity",
            "bars_held",
            "down_spike_pct",
            "atr_pct",
            "large_trade_share",
            "close_pos_in_bar",
            "session_bucket",
        ]
        trade_sample = combined_all[[c for c in sample_cols if c in combined_all.columns]].head(sample_n).copy() if not combined_all.empty else pd.DataFrame()
        write_csv(trade_sample, out_dir / "trade_sample.csv", "trade_sample")

    worst_add = add_legs_all.sort_values("add_net_return_same_parent", ascending=True).head(50).copy() if not add_legs_all.empty else pd.DataFrame()
    best_add = add_legs_all.sort_values("add_net_return_same_parent", ascending=False).head(50).copy() if not add_legs_all.empty else pd.DataFrame()
    write_csv(worst_add, out_dir / "09_worst_add_legs.csv", "worst_add_legs")
    write_csv(best_add, out_dir / "10_best_add_legs.csv", "best_add_legs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {SCRIPT_NAME} {SCRIPT_VERSION}", flush=True)
    print("[scope] research only; no formal backtest / no portfolio change / no live logic", flush=True)
    print(f"[args] out_dir={out_dir}", flush=True)

    print(f"[load] trade bars {args.symbol} {args.timeframe} {args.warmup_start_date}->{args.end_date}", flush=True)
    bars = low_v1.load_trade_bars(args)
    print("[events] build existing Low Sweep V1 A0 events/context", flush=True)
    events = low_v1.prepare_events_and_context(bars, args)
    events = _select_a0_events(events, args)
    events, positions = _event_positions(bars, events)
    print(f"[events] selected A0_fp_abs_delta_high + single_swing rows={len(events):,}", flush=True)

    market = build_market_cache(bars, args)
    cost_mults = _parse_float_list(args.cost_mults)
    parents_all: list[pd.DataFrame] = []
    add_legs_all: list[pd.DataFrame] = []
    combined_all: list[pd.DataFrame] = []

    for cost_mult in cost_mults:
        print(f"[simulate] cost {cost_mult:g}x", flush=True)
        for parent in _parent_specs(args.parent_exit_modes):
            parents, _overlaps = simulate_parent_sequence(bars, events, positions, parent, args, market, cost_mult=float(cost_mult))
            if parents.empty:
                continue
            parents["cost_mult"] = float(cost_mult)
            add_legs, by_parent = build_add_diagnostics(bars, events, positions, parents, parent, args, market, cost_mult=float(cost_mult))
            if not add_legs.empty:
                add_legs["cost_mult"] = float(cost_mult)
            combined = build_combined_trade_rows(parents, by_parent, parent, args, market, cost_mult=float(cost_mult))
            parents_all.append(parents)
            add_legs_all.append(add_legs)
            combined_all.append(combined)

    parents_df = pd.concat(parents_all, ignore_index=True) if parents_all else pd.DataFrame()
    add_legs_df = pd.concat(add_legs_all, ignore_index=True) if add_legs_all else pd.DataFrame()
    combined_df = ensure_summary_compatible(pd.concat(combined_all, ignore_index=True)) if combined_all else pd.DataFrame()
    summary = summarize_variant_table(combined_df, args)

    write_outputs(out_dir, parents_df, add_legs_df, combined_df, summary, args)

    meta = {
        "script": SCRIPT_NAME,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "parent_exit_modes": args.parent_exit_modes,
        "add_weights": args.add_weights,
        "max_add_counts": args.max_add_counts,
        "cost_mults": args.cost_mults,
        "selected_events": int(len(events)),
        "parent_trades_rows": int(len(parents_df)),
        "overlap_add_legs_rows": int(len(add_legs_df)),
        "combined_trade_rows": int(len(combined_df)),
        "notes": "Research-only overlap add diagnostic. Combined rows are not a formal portfolio simulator.",
    }
    (out_dir / "99_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_csv(pd.DataFrame([meta]), out_dir / "99_meta.csv", "meta")

    print(f"[done] wrote reports to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
