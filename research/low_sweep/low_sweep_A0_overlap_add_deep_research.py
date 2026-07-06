#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deep diagnostic research for Low Sweep A0 overlap add-on signals.

Scope
-----
Research only.  This script does not modify the formal MF backtest, portfolio
wrapper, or live strategy.  It reuses the existing CoinBacktest Low Sweep V1
A0 event/context pipeline and asks a narrower diagnostic question:

    During the current A0_fp_abs_delta_high + single_swing + next_open + time48
    parent trade, skipped overlap A0 signals historically looked profitable.
    Are those add signals robust enough to deserve a next research round, and
    what observable state separates good from bad add legs?

The script is deliberately more diagnostic than the first overlap-add probe:
- parent is time48 by default;
- add entry can be stressed by next_open_delayN;
- add schemes are split by first/all, weight, down-spike depth, parent drawdown,
  session, ATR rank, large-trade-share rank, and timing inside the parent;
- outputs include scheme summary, deltas, yearly/monthly, add-leg breakdowns,
  contribution concentration, best/worst legs, and a full trade sample.

Causal notes
------------
All event/context fields are produced by the existing Low Sweep no-leakage
pipeline.  The additional rank columns in this script are computed from prior
selected A0 events only: the current add signal is ranked against historical
signals, not future ones.  Some raw/global threshold schemes are still only
research diagnostics and are marked as such in the filter definition table.
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
    write_csv,
)
from src.research_common.progress import ProgressReporter  # noqa: E402

SCRIPT_NAME = "low_sweep_A0_overlap_add_deep_research"
SCRIPT_VERSION = "v1_deep_time48_overlap_add"
DEFAULT_OUT_DIR = "data/reports/research/low_sweep/A0_overlap_add_deep_research"
PARENT_NAME = "parent_time48"
PARENT_EXIT_MODE = "time48"


@dataclass(frozen=True)
class ParentSpec:
    name: str
    exit_mode: str


@dataclass(frozen=True)
class AddLeg:
    parent_trade_idx: int
    add_event_idx: int
    signal_pos: int
    signal_time: pd.Timestamp
    entry_pos: int
    entry_time: pd.Timestamp
    entry_price: float
    exit_pos: int
    exit_time: pd.Timestamp
    exit_price: float
    gross_return: float
    net_return: float
    mae_same_parent: float
    mfe_same_parent: float
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
    add_signal_down_spike_rank: float
    add_signal_atr_pct: float
    add_signal_atr_rank: float
    add_signal_large_trade_share: float
    add_signal_large_trade_share_rank: float
    add_signal_close_pos_in_bar: float
    add_signal_close_pos_rank: float
    add_signal_session_bucket: object
    add_signal_swing_age: float
    add_signal_cluster_touch_count_020: float
    add_entry_delay_bars: int
    cost_mult: float


@dataclass(frozen=True)
class AddScheme:
    name: str
    filter_expr: str
    max_add_count: int
    add_weight: float
    add_entry_delay_bars: int
    notes: str


# ---------------------------------------------------------------------------
# Args / parsing helpers
# ---------------------------------------------------------------------------


def _split_csv(raw: object) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _parse_float_list(raw: object) -> list[float]:
    out: list[float] = []
    for part in _split_csv(raw):
        val = float(part)
        if math.isfinite(val):
            out.append(val)
    return sorted(set(out))


def _parse_int_list(raw: object) -> list[int]:
    out: list[int] = []
    for part in _split_csv(raw):
        out.append(int(part))
    return sorted(set(out))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    own.add_argument("--parent-exit-mode", default=PARENT_EXIT_MODE, choices=["time48"], help="Deep research currently freezes the parent to the current main time48 leg.")
    own.add_argument("--add-weights", default="0.25,0.5,1.0")
    own.add_argument("--cost-mults", default="1.0,1.5,2.0")
    own.add_argument("--add-entry-delays", default="0,1,2", help="0=next_open; N=next_open_delayN for the add leg only.")
    own.add_argument("--rank-lookback-events", type=int, default=200, help="Prior selected A0 events used for causal event-rank features.")
    own.add_argument("--min-bars-after-parent-entry", type=int, default=1)
    own.add_argument("--min-bars-before-parent-exit", type=int, default=1)
    own.add_argument("--progress-every", type=int, default=100)
    own.add_argument("--save-trades", type=int, default=200000)
    own.add_argument("--save-events", type=int, default=5000)
    own.add_argument("--save-trade-sample", type=int, default=200000)
    own.add_argument("--max-output-add-legs", type=int, default=200000)
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
        "next_open,next_open_delay1,next_open_delay2",
        "--exit-modes",
        "time48",
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
    for key, val in vars(known).items():
        setattr(args, key, val)
    return args


def _no_stop_spec():
    return {s.name: s for s in parse_stop_specs("no_stop")}["no_stop"]


def _parent_spec(args: argparse.Namespace) -> ParentSpec:
    # Kept as a function so future research can explicitly branch here without
    # touching the rest of the diagnostic engine.
    if str(args.parent_exit_mode) != "time48":
        raise ValueError("This deep research script currently supports only parent-exit-mode=time48")
    return ParentSpec(name=PARENT_NAME, exit_mode="time48")


def _variant(parent: ParentSpec, *, entry_mode: str = "next_open") -> UpgradeVariant:
    return UpgradeVariant(
        variant_name=parent.name,
        candidate_layer="A0_fp_abs_delta_high",
        support_mode="single_swing",
        entry_mode=entry_mode,
        exit_mode=parent.exit_mode,
        stop_spec=_no_stop_spec(),
    )


def _safe_float(x: object, default: float = np.nan) -> float:
    try:
        val = float(x)
    except Exception:
        return default
    return val if math.isfinite(val) else default


def _profit_factor(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if vals.empty:
        return np.nan
    gross_profit = float(vals[vals > 0].sum())
    gross_loss = float(-vals[vals < 0].sum())
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else np.nan
    return gross_profit / gross_loss


def _equity_and_dd(x: pd.Series, start: float = 1.0) -> tuple[pd.Series, pd.Series]:
    vals = pd.to_numeric(x, errors="coerce").fillna(0.0)
    equity = float(start) * (1.0 + vals).cumprod()
    dd = equity / equity.cummax() - 1.0
    return equity, dd


def _max_consecutive_losses(x: pd.Series) -> int:
    cur = best = 0
    for val in pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float):
        if val < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _top_winner_share(x: pd.Series, n: int = 5) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    wins = vals[vals > 0].sort_values(ascending=False)
    denom = float(wins.sum())
    if denom <= 0:
        return np.nan
    return float(wins.head(int(n)).sum() / denom)


def _ensure_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
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
# Event selection and causal ranks
# ---------------------------------------------------------------------------


def _select_a0_events(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    masks = build_candidate_layer_masks(events, args)
    layer = masks.get("A0_fp_abs_delta_high", pd.Series(False, index=events.index)).fillna(False).astype(bool)
    support = build_support_mask(events, "single_swing", args).fillna(False).astype(bool)
    out = events.loc[layer & support].copy()
    if "signal_time" in out.columns:
        out = out.sort_values("signal_time", kind="stable").reset_index(drop=True)
    out["_event_idx"] = np.arange(len(out), dtype=int)
    return out


def _event_positions(bars: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if events.empty:
        return events.copy(), np.asarray([], dtype=int)
    times = pd.DatetimeIndex(pd.to_datetime(events["signal_time"], errors="coerce"))
    pos = bars.index.get_indexer(times)
    valid = (pos >= 0) & ((pos + 2) < len(bars))
    return events.loc[valid].copy().reset_index(drop=True), pos[valid]


def _percentile_rank_against_past(values: pd.Series, lookback: int) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    lb = max(5, int(lookback))
    for i, val in enumerate(arr):
        if not math.isfinite(float(val)):
            continue
        start = max(0, i - lb)
        hist = arr[start:i]
        hist = hist[np.isfinite(hist)]
        if len(hist) < 10:
            continue
        out[i] = float((hist <= float(val)).mean())
    return pd.Series(out, index=values.index)


def attach_causal_event_ranks(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = events.copy().reset_index(drop=True)
    lookback = int(getattr(args, "rank_lookback_events", 200))
    field_map = {
        "down_spike_pct": "down_spike_rank",
        "atr_pct": "atr_rank",
        "large_trade_share": "large_trade_share_rank",
        "close_pos_in_bar": "close_pos_rank",
    }
    for src, dst in field_map.items():
        if src in out.columns:
            out[dst] = _percentile_rank_against_past(out[src], lookback)
        else:
            out[dst] = np.nan
    return out


# ---------------------------------------------------------------------------
# Parent and add-leg simulation
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
) -> tuple[pd.DataFrame, dict[str, int]]:
    variant = _variant(parent)
    event_records = events.to_dict("records")
    rows: list[dict[str, object]] = []
    counters = {
        "input_events": int(len(event_records)),
        "valid_parent_trades": 0,
        "skipped_overlap": 0,
        "skipped_invalid": 0,
    }
    last_exit_pos = -1

    progress = ProgressReporter(
        label=f"[parent] {parent.name} cost{cost_mult:g}",
        total=max(1, len(event_records)),
        every=max(1, int(args.progress_every)),
        enabled=not bool(getattr(args, "no_progress", False)),
    )
    for i, (event, signal_pos) in enumerate(zip(event_records, positions), start=1):
        signal_pos = int(signal_pos)
        if signal_pos <= last_exit_pos:
            counters["skipped_overlap"] += 1
            progress.update(i)
            continue
        rec = simulate_upgrade_trade(bars, event, signal_pos, variant, args, cost_mult=cost_mult, market=market)
        if not rec.get("valid"):
            counters["skipped_invalid"] += 1
            progress.update(i)
            continue
        row = dict(rec)
        row["parent_name"] = parent.name
        row["parent_exit_mode"] = parent.exit_mode
        row["parent_trade_idx"] = int(len(rows))
        row["parent_event_idx"] = int(event.get("_event_idx", len(rows)))
        row["cost_mult"] = float(cost_mult)
        row["candidate_events"] = int(len(event_records))
        row["input_events"] = int(len(event_records))
        rows.append(row)
        counters["valid_parent_trades"] += 1
        last_exit_pos = int(row.get("exit_pos", signal_pos))
        progress.update(i)
    progress.close()
    return _ensure_summary_columns(pd.DataFrame(rows)), counters


def _parent_state_at_signal(parent_row: pd.Series, market, add_signal_pos: int) -> dict[str, object]:
    entry_pos = int(parent_row["entry_pos"])
    entry_price = float(parent_row["entry_price"])
    stop = min(max(int(add_signal_pos), entry_pos), len(market.index) - 1)
    highs = market.high[entry_pos : stop + 1]
    lows = market.low[entry_pos : stop + 1]
    if len(highs) == 0:
        return {
            "parent_unreal_close_ret": np.nan,
            "parent_mfe_before_add": np.nan,
            "parent_mae_before_add": np.nan,
            "parent_armed_before_add": False,
        }
    high_ret = highs / entry_price - 1.0
    low_ret = lows / entry_price - 1.0
    mfe = float(np.nanmax(high_ret))
    mae = float(np.nanmin(low_ret))
    close_ret = float(market.close[stop] / entry_price - 1.0)
    return {
        "parent_unreal_close_ret": close_ret,
        "parent_mfe_before_add": mfe,
        "parent_mae_before_add": mae,
        "parent_armed_before_add": bool(math.isfinite(mfe) and mfe >= 0.015),
    }


def _path_mae_mfe(entry_pos: int, exit_pos: int, entry_price: float, market) -> tuple[float, float]:
    if int(exit_pos) < int(entry_pos):
        return np.nan, np.nan
    lows = market.low[int(entry_pos) : int(exit_pos) + 1] / float(entry_price) - 1.0
    highs = market.high[int(entry_pos) : int(exit_pos) + 1] / float(entry_price) - 1.0
    mae = float(np.nanmin(lows)) if len(lows) else np.nan
    mfe = float(np.nanmax(highs)) if len(highs) else np.nan
    return mae, mfe


def _entry_mode_for_delay(delay_bars: int) -> str:
    d = int(delay_bars)
    return "next_open" if d <= 0 else f"next_open_delay{d}"


def simulate_add_leg(
    bars: pd.DataFrame,
    add_event: dict[str, object],
    add_signal_pos: int,
    parent_row: pd.Series,
    parent: ParentSpec,
    args: argparse.Namespace,
    market,
    *,
    cost_mult: float,
    add_entry_delay_bars: int,
) -> AddLeg | None:
    entry_mode = _entry_mode_for_delay(add_entry_delay_bars)
    ok, entry_pos, entry_price, _entry_reason = _entry_from_mode(bars, add_event, int(add_signal_pos), entry_mode, args, market)
    if not ok or entry_pos is None or entry_price is None:
        return None

    parent_exit_pos = int(parent_row["exit_pos"])
    if int(entry_pos) >= parent_exit_pos:
        return None
    if int(entry_pos) - int(parent_row["entry_pos"]) < int(args.min_bars_after_parent_entry):
        return None
    if parent_exit_pos - int(entry_pos) < int(args.min_bars_before_parent_exit):
        return None

    parent_exit_price = float(parent_row["exit_price"])
    gross = float(parent_exit_price / float(entry_price) - 1.0)
    net = float(gross - _entry_cost(args, cost_mult) - _exit_cost(args, cost_mult))
    mae, mfe = _path_mae_mfe(int(entry_pos), parent_exit_pos, float(entry_price), market)

    own_variant = _variant(parent, entry_mode=entry_mode)
    own = simulate_upgrade_trade(bars, add_event, int(add_signal_pos), own_variant, args, cost_mult=cost_mult, market=market)
    own_exit_pos: int | None = None
    own_exit_time: pd.Timestamp | None = None
    own_exit_price: float | None = None
    own_net: float | None = None
    own_reason = ""
    if own.get("valid"):
        own_exit_pos = int(own.get("exit_pos"))
        own_exit_time = pd.Timestamp(own.get("exit_time"))
        own_exit_price = float(own.get("exit_price"))
        own_net = float(own.get("net_return_on_equity"))
        own_reason = str(own.get("exit_reason", ""))

    state = _parent_state_at_signal(parent_row, market, int(add_signal_pos))
    return AddLeg(
        parent_trade_idx=int(parent_row["parent_trade_idx"]),
        add_event_idx=int(add_event.get("_event_idx", -1)),
        signal_pos=int(add_signal_pos),
        signal_time=pd.Timestamp(add_event.get("signal_time")),
        entry_pos=int(entry_pos),
        entry_time=pd.Timestamp(market.index[int(entry_pos)]),
        entry_price=float(entry_price),
        exit_pos=parent_exit_pos,
        exit_time=pd.Timestamp(parent_row["exit_time"]),
        exit_price=parent_exit_price,
        gross_return=gross,
        net_return=net,
        mae_same_parent=mae,
        mfe_same_parent=mfe,
        own_exit_pos=own_exit_pos,
        own_exit_time=own_exit_time,
        own_exit_price=own_exit_price,
        own_net_return=own_net,
        own_exit_reason=own_reason,
        parent_bars_since_entry=int(add_signal_pos - int(parent_row["entry_pos"])),
        parent_bars_to_exit=int(parent_exit_pos - int(add_signal_pos)),
        parent_unreal_close_ret=float(state["parent_unreal_close_ret"]),
        parent_mfe_before_add=float(state["parent_mfe_before_add"]),
        parent_mae_before_add=float(state["parent_mae_before_add"]),
        parent_armed_before_add=bool(state["parent_armed_before_add"]),
        add_signal_down_spike_pct=_safe_float(add_event.get("down_spike_pct")),
        add_signal_down_spike_rank=_safe_float(add_event.get("down_spike_rank")),
        add_signal_atr_pct=_safe_float(add_event.get("atr_pct")),
        add_signal_atr_rank=_safe_float(add_event.get("atr_rank")),
        add_signal_large_trade_share=_safe_float(add_event.get("large_trade_share")),
        add_signal_large_trade_share_rank=_safe_float(add_event.get("large_trade_share_rank")),
        add_signal_close_pos_in_bar=_safe_float(add_event.get("close_pos_in_bar")),
        add_signal_close_pos_rank=_safe_float(add_event.get("close_pos_rank")),
        add_signal_session_bucket=add_event.get("session_bucket"),
        add_signal_swing_age=_safe_float(add_event.get("swing_age")),
        add_signal_cluster_touch_count_020=_safe_float(add_event.get("cluster_touch_count_020")),
        add_entry_delay_bars=int(add_entry_delay_bars),
        cost_mult=float(cost_mult),
    )


def _leg_to_row(parent_row: pd.Series, leg: AddLeg) -> dict[str, object]:
    return {
        "parent_trade_idx": leg.parent_trade_idx,
        "parent_signal_time": parent_row.get("signal_time"),
        "parent_entry_time": parent_row.get("entry_time"),
        "parent_exit_time": parent_row.get("exit_time"),
        "parent_entry_price": parent_row.get("entry_price"),
        "parent_exit_price": parent_row.get("exit_price"),
        "parent_net_return": parent_row.get("net_return_on_equity"),
        "parent_exit_reason": parent_row.get("exit_reason"),
        "add_event_idx": leg.add_event_idx,
        "add_signal_pos": leg.signal_pos,
        "add_signal_time": leg.signal_time,
        "add_entry_delay_bars": leg.add_entry_delay_bars,
        "add_entry_pos": leg.entry_pos,
        "add_entry_time": leg.entry_time,
        "add_entry_price": leg.entry_price,
        "add_exit_pos_same_parent": leg.exit_pos,
        "add_exit_time_same_parent": leg.exit_time,
        "add_exit_price_same_parent": leg.exit_price,
        "add_gross_return_same_parent": leg.gross_return,
        "add_net_return_same_parent": leg.net_return,
        "add_mae_same_parent": leg.mae_same_parent,
        "add_mfe_same_parent": leg.mfe_same_parent,
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
        "add_signal_down_spike_rank": leg.add_signal_down_spike_rank,
        "add_signal_atr_pct": leg.add_signal_atr_pct,
        "add_signal_atr_rank": leg.add_signal_atr_rank,
        "add_signal_large_trade_share": leg.add_signal_large_trade_share,
        "add_signal_large_trade_share_rank": leg.add_signal_large_trade_share_rank,
        "add_signal_close_pos_in_bar": leg.add_signal_close_pos_in_bar,
        "add_signal_close_pos_rank": leg.add_signal_close_pos_rank,
        "add_signal_session_bucket": leg.add_signal_session_bucket,
        "add_signal_swing_age": leg.add_signal_swing_age,
        "add_signal_cluster_touch_count_020": leg.add_signal_cluster_touch_count_020,
        "add_same_parent_would_win": bool(leg.net_return > 0),
        "add_own_would_win": bool(leg.own_net_return is not None and leg.own_net_return > 0),
        "cost_mult": leg.cost_mult,
    }


def build_add_legs(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    positions: np.ndarray,
    parents: pd.DataFrame,
    parent: ParentSpec,
    args: argparse.Namespace,
    market,
    *,
    cost_mult: float,
    add_entry_delay_bars: int,
) -> tuple[pd.DataFrame, dict[int, list[AddLeg]]]:
    event_records = events.to_dict("records")
    by_parent: dict[int, list[AddLeg]] = {int(x): [] for x in parents.get("parent_trade_idx", pd.Series(dtype=int)).tolist()}
    rows: list[dict[str, object]] = []

    for _, parent_row in parents.iterrows():
        parent_idx = int(parent_row["parent_trade_idx"])
        parent_signal_pos = int(parent_row["signal_pos"])
        parent_exit_pos = int(parent_row["exit_pos"])
        parent_event_idx = int(parent_row.get("parent_event_idx", -1))
        mask = (positions > parent_signal_pos) & (positions <= parent_exit_pos)
        for event_array_idx in np.flatnonzero(mask):
            add_event = event_records[int(event_array_idx)]
            if int(add_event.get("_event_idx", -2)) == parent_event_idx:
                continue
            leg = simulate_add_leg(
                bars,
                add_event,
                int(positions[int(event_array_idx)]),
                parent_row,
                parent,
                args,
                market,
                cost_mult=cost_mult,
                add_entry_delay_bars=add_entry_delay_bars,
            )
            if leg is None:
                continue
            by_parent[parent_idx].append(leg)
            rows.append(_leg_to_row(parent_row, leg))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["add_entry_delay_bars"] = int(add_entry_delay_bars)
    return out, by_parent


# ---------------------------------------------------------------------------
# Add scheme generation / filtering
# ---------------------------------------------------------------------------


def _filter_definitions() -> dict[str, str]:
    return {
        "any": "No add-leg filter. Diagnostic baseline for overlap add.",
        "ds_rank_ge_50": "Causal event rank: add down_spike_pct >= median of prior selected A0 events.",
        "ds_rank_ge_60": "Causal event rank: add down_spike_pct >= 60th percentile of prior selected A0 events.",
        "ds_rank_ge_70": "Causal event rank: add down_spike_pct >= 70th percentile of prior selected A0 events.",
        "atr_rank_ge_50": "Causal event rank: add signal ATR >= median of prior selected A0 events.",
        "large_mid_20_80": "Causal event rank: large_trade_share in [20%, 80%]; avoids no-big-trade and extreme-dump tails.",
        "session_S0": "Add signal session_bucket starts with S0_00_07.",
        "not_armed": "Parent has not reached +1.5% MFE before add signal.",
        "armed": "Parent has reached +1.5% MFE before add signal.",
        "pu_lte_m020": "Parent close unrealized return at add signal <= -2.0%.",
        "pu_lte_m030": "Parent close unrealized return at add signal <= -3.0%.",
        "pu_lte_m040": "Parent close unrealized return at add signal <= -4.0%.",
        "mae_lte_m030": "Parent MAE before add <= -3.0%.",
        "bars_since_ge_6": "Add signal appears at least 6 bars after parent entry.",
        "bars_since_ge_12": "Add signal appears at least 12 bars after parent entry.",
        "raw_ds_ge_0100": "Raw diagnostic: add down_spike_pct >= 1.00%. Not rolling-ranked.",
        "raw_ds_ge_0120": "Raw diagnostic: add down_spike_pct >= 1.20%. Not rolling-ranked.",
        "raw_ds_ge_0150": "Raw diagnostic: add down_spike_pct >= 1.50%. Not rolling-ranked.",
    }


def _condition_pass(leg: AddLeg, cond: str) -> bool:
    cond = str(cond).strip()
    if not cond or cond == "any":
        return True
    if cond == "ds_rank_ge_50":
        return math.isfinite(leg.add_signal_down_spike_rank) and leg.add_signal_down_spike_rank >= 0.50
    if cond == "ds_rank_ge_60":
        return math.isfinite(leg.add_signal_down_spike_rank) and leg.add_signal_down_spike_rank >= 0.60
    if cond == "ds_rank_ge_70":
        return math.isfinite(leg.add_signal_down_spike_rank) and leg.add_signal_down_spike_rank >= 0.70
    if cond == "atr_rank_ge_50":
        return math.isfinite(leg.add_signal_atr_rank) and leg.add_signal_atr_rank >= 0.50
    if cond == "large_mid_20_80":
        return math.isfinite(leg.add_signal_large_trade_share_rank) and 0.20 <= leg.add_signal_large_trade_share_rank <= 0.80
    if cond == "session_S0":
        return str(leg.add_signal_session_bucket).startswith("S0")
    if cond == "not_armed":
        return not bool(leg.parent_armed_before_add)
    if cond == "armed":
        return bool(leg.parent_armed_before_add)
    if cond == "pu_lte_m020":
        return math.isfinite(leg.parent_unreal_close_ret) and leg.parent_unreal_close_ret <= -0.020
    if cond == "pu_lte_m030":
        return math.isfinite(leg.parent_unreal_close_ret) and leg.parent_unreal_close_ret <= -0.030
    if cond == "pu_lte_m040":
        return math.isfinite(leg.parent_unreal_close_ret) and leg.parent_unreal_close_ret <= -0.040
    if cond == "mae_lte_m030":
        return math.isfinite(leg.parent_mae_before_add) and leg.parent_mae_before_add <= -0.030
    if cond == "bars_since_ge_6":
        return int(leg.parent_bars_since_entry) >= 6
    if cond == "bars_since_ge_12":
        return int(leg.parent_bars_since_entry) >= 12
    if cond == "raw_ds_ge_0100":
        return math.isfinite(leg.add_signal_down_spike_pct) and leg.add_signal_down_spike_pct >= 0.0100
    if cond == "raw_ds_ge_0120":
        return math.isfinite(leg.add_signal_down_spike_pct) and leg.add_signal_down_spike_pct >= 0.0120
    if cond == "raw_ds_ge_0150":
        return math.isfinite(leg.add_signal_down_spike_pct) and leg.add_signal_down_spike_pct >= 0.0150
    raise ValueError(f"Unsupported add filter condition: {cond}")


def leg_pass_filter(leg: AddLeg, expr: str) -> bool:
    parts = [p.strip() for p in str(expr).split("&") if p.strip()]
    return all(_condition_pass(leg, part) for part in parts)


def build_schemes(args: argparse.Namespace, *, add_entry_delay_bars: int) -> list[AddScheme]:
    weights = _parse_float_list(args.add_weights)
    filters_first = [
        ("any", "first_any", "First eligible overlap add."),
        ("ds_rank_ge_50", "first_ds_rank_ge_50", "First add with causal down-spike rank >= 50%."),
        ("ds_rank_ge_60", "first_ds_rank_ge_60", "First add with causal down-spike rank >= 60%."),
        ("ds_rank_ge_70", "first_ds_rank_ge_70", "First add with causal down-spike rank >= 70%."),
        ("raw_ds_ge_0120", "first_raw_ds_ge_0120", "First add with raw down_spike >= 1.20%."),
        ("pu_lte_m020", "first_parent_unreal_lte_m020", "First add when parent close unreal <= -2%."),
        ("pu_lte_m030", "first_parent_unreal_lte_m030", "First add when parent close unreal <= -3%."),
        ("session_S0", "first_session_S0", "First add in S0_00_07 session."),
        ("atr_rank_ge_50", "first_atr_rank_ge_50", "First add with causal ATR rank >= 50%."),
        ("large_mid_20_80", "first_large_mid_20_80", "First add with large-trade-share rank in middle band."),
        ("ds_rank_ge_50&pu_lte_m020", "first_ds_rank50_parent_unreal_m020", "Deep add and parent <= -2%."),
        ("ds_rank_ge_50&session_S0", "first_ds_rank50_session_S0", "Deep add in S0 session."),
        ("ds_rank_ge_50&large_mid_20_80", "first_ds_rank50_large_mid", "Deep add with middle large-trade-share rank."),
        ("ds_rank_ge_50&bars_since_ge_6", "first_ds_rank50_after6", "Deep add at least 6 bars after parent entry."),
        ("ds_rank_ge_50&pu_lte_m020&session_S0", "first_ds_rank50_parent_m020_session_S0", "Deep add, parent <= -2%, S0 session."),
    ]
    filters_all = [
        ("any", "all_any", "All eligible overlap adds."),
        ("ds_rank_ge_50", "all_ds_rank_ge_50", "All add legs with causal down-spike rank >= 50%."),
        ("ds_rank_ge_60", "all_ds_rank_ge_60", "All add legs with causal down-spike rank >= 60%."),
        ("pu_lte_m020", "all_parent_unreal_lte_m020", "All add legs when parent close unreal <= -2%."),
    ]
    schemes: list[AddScheme] = []
    for weight in weights:
        for expr, name, note in filters_first:
            schemes.append(AddScheme(name=f"{name}__w{int(round(weight * 100)):03d}", filter_expr=expr, max_add_count=1, add_weight=float(weight), add_entry_delay_bars=int(add_entry_delay_bars), notes=note))
        for expr, name, note in filters_all:
            schemes.append(AddScheme(name=f"{name}__w{int(round(weight * 100)):03d}", filter_expr=expr, max_add_count=999999, add_weight=float(weight), add_entry_delay_bars=int(add_entry_delay_bars), notes=note))
    return schemes


# ---------------------------------------------------------------------------
# Combined trade rows / metrics
# ---------------------------------------------------------------------------


def select_add_legs(legs: Sequence[AddLeg], scheme: AddScheme) -> list[AddLeg]:
    filtered = [leg for leg in sorted(legs, key=lambda x: x.entry_pos) if leg_pass_filter(leg, scheme.filter_expr)]
    if scheme.max_add_count >= 999999:
        return filtered
    return filtered[: max(0, int(scheme.max_add_count))]


def _combined_path_metrics(parent_row: pd.Series, legs: Sequence[AddLeg], market, weight: float) -> dict[str, object]:
    entry_pos = int(parent_row["entry_pos"])
    exit_pos = int(parent_row["exit_pos"])
    entry_price = float(parent_row["entry_price"])
    lows: list[float] = []
    highs: list[float] = []
    for pos in range(entry_pos, exit_pos + 1):
        low_ret = float(market.low[pos] / entry_price - 1.0)
        high_ret = float(market.high[pos] / entry_price - 1.0)
        for leg in legs:
            if pos >= int(leg.entry_pos):
                low_ret += float(weight) * float(market.low[pos] / float(leg.entry_price) - 1.0)
                high_ret += float(weight) * float(market.high[pos] / float(leg.entry_price) - 1.0)
        lows.append(low_ret)
        highs.append(high_ret)
    low_arr = np.asarray(lows, dtype=float)
    high_arr = np.asarray(highs, dtype=float)
    if low_arr.size == 0:
        return {
            "mae_on_equity": np.nan,
            "mfe_on_equity": np.nan,
            "mae_time_bars": np.nan,
            "mfe_time_bars": np.nan,
            "first_positive_high_bars": np.nan,
            "mae_before_mfe_flag": False,
        }
    mae_i = int(np.nanargmin(low_arr))
    mfe_i = int(np.nanargmax(high_arr))
    first_pos = np.flatnonzero(high_arr > 0)
    return {
        "mae_on_equity": float(np.nanmin(low_arr)),
        "mfe_on_equity": float(np.nanmax(high_arr)),
        "mae_time_bars": mae_i,
        "mfe_time_bars": mfe_i,
        "first_positive_high_bars": int(first_pos[0]) if first_pos.size else np.nan,
        "mae_before_mfe_flag": bool(mae_i <= mfe_i),
    }


def build_combined_rows(
    parents: pd.DataFrame,
    add_by_parent: dict[int, list[AddLeg]],
    parent: ParentSpec,
    args: argparse.Namespace,
    market,
    *,
    cost_mult: float,
    add_entry_delay_bars: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    # Baseline for this cost/delay slice.  Baseline rows are duplicated per add
    # delay so delta tables can compare each stressed add-entry setup cleanly.
    for _, prow in parents.iterrows():
        base = dict(prow)
        base.update(
            {
                "variant_name": f"{parent.name}__d{int(add_entry_delay_bars)}__no_add",
                "parent_name": parent.name,
                "parent_exit_mode": parent.exit_mode,
                "add_entry_delay_bars": int(add_entry_delay_bars),
                "add_scheme": "no_add",
                "add_filter_expr": "",
                "add_weight": 0.0,
                "max_add_count": 0,
                "add_count": 0,
                "add_return_sum_unweighted": 0.0,
                "add_return_sum_weighted": 0.0,
                "combined_return_delta_vs_parent": 0.0,
                "cost_mult": float(cost_mult),
            }
        )
        rows.append(base)

    schemes = build_schemes(args, add_entry_delay_bars=int(add_entry_delay_bars))
    for _, prow in parents.iterrows():
        parent_idx = int(prow["parent_trade_idx"])
        legs = add_by_parent.get(parent_idx, [])
        for scheme in schemes:
            selected = select_add_legs(legs, scheme)
            add_unweighted = float(sum(float(x.net_return) for x in selected))
            add_weighted = float(scheme.add_weight * add_unweighted)
            combined_ret = float(prow["net_return_on_equity"]) + add_weighted
            row = dict(prow)
            row.update(
                {
                    "variant_name": f"{parent.name}__d{int(add_entry_delay_bars)}__{scheme.name}",
                    "parent_name": parent.name,
                    "parent_exit_mode": parent.exit_mode,
                    "add_entry_delay_bars": int(add_entry_delay_bars),
                    "add_scheme": scheme.name,
                    "add_filter_expr": scheme.filter_expr,
                    "add_weight": float(scheme.add_weight),
                    "max_add_count": int(scheme.max_add_count),
                    "add_count": int(len(selected)),
                    "add_event_indices": "|".join(str(x.add_event_idx) for x in selected),
                    "add_entry_times": "|".join(str(x.entry_time) for x in selected),
                    "add_return_sum_unweighted": add_unweighted,
                    "add_return_sum_weighted": add_weighted,
                    "combined_return_delta_vs_parent": add_weighted,
                    "net_return_on_equity": combined_ret,
                    "gross_return_on_equity": float(prow.get("gross_return_on_equity", np.nan)) + add_weighted,
                    "exit_reason": f"{prow.get('exit_reason', '')}|{scheme.name}",
                    "cost_mult": float(cost_mult),
                }
            )
            row.update(_combined_path_metrics(prow, selected, market, float(scheme.add_weight)))
            rows.append(row)
    return _ensure_summary_columns(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Summaries and reports
# ---------------------------------------------------------------------------


def summarize_trade_group(grp: pd.DataFrame) -> dict[str, object]:
    x = pd.to_numeric(grp["net_return_on_equity"], errors="coerce").fillna(0.0)
    equity, dd = _equity_and_dd(x, 1.0)
    wins = x[x > 0]
    losses = x[x < 0]
    rec: dict[str, object] = {
        "trades": int(len(grp)),
        "return_total": float(equity.iloc[-1] - 1.0) if not equity.empty else np.nan,
        "mean_return": float(x.mean()) if not x.empty else np.nan,
        "median_return": float(x.median()) if not x.empty else np.nan,
        "win_rate": float((x > 0).mean()) if not x.empty else np.nan,
        "avg_win": float(wins.mean()) if not wins.empty else np.nan,
        "avg_loss": float(losses.mean()) if not losses.empty else np.nan,
        "profit_factor": _profit_factor(x),
        "max_drawdown": float(dd.min()) if not dd.empty else np.nan,
        "max_consecutive_losses": _max_consecutive_losses(x),
        "worst_trade": float(x.min()) if not x.empty else np.nan,
        "best_trade": float(x.max()) if not x.empty else np.nan,
        "top5_winner_share": _top_winner_share(x, 5),
        "mae_median": float(pd.to_numeric(grp.get("mae_on_equity"), errors="coerce").median()),
        "mfe_median": float(pd.to_numeric(grp.get("mfe_on_equity"), errors="coerce").median()),
        "avg_bars_held": float(pd.to_numeric(grp.get("bars_held"), errors="coerce").mean()),
    }
    for col in ["parent_name", "parent_exit_mode", "add_entry_delay_bars", "add_scheme", "add_filter_expr", "add_weight", "max_add_count", "cost_mult"]:
        if col in grp.columns:
            vals = grp[col].dropna().unique()
            rec[col] = vals[0] if len(vals) == 1 else "mixed"
    rec["parents_with_add"] = int((pd.to_numeric(grp.get("add_count", 0), errors="coerce").fillna(0) > 0).sum())
    rec["total_add_count"] = int(pd.to_numeric(grp.get("add_count", 0), errors="coerce").fillna(0).sum())
    rec["sum_add_return_weighted"] = float(pd.to_numeric(grp.get("add_return_sum_weighted", 0.0), errors="coerce").fillna(0.0).sum())
    rec["mean_add_return_weighted"] = float(pd.to_numeric(grp.get("add_return_sum_weighted", 0.0), errors="coerce").fillna(0.0).mean())
    return rec


def summarize_schemes(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for variant_name, grp in combined.groupby("variant_name", dropna=False):
        rec = summarize_trade_group(grp)
        rec["variant_name"] = variant_name
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["cost_mult", "add_entry_delay_bars", "return_total", "profit_factor"], ascending=[True, True, False, False]).reset_index(drop=True)
    return out


def delta_vs_no_add(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (parent_name, cost_mult, delay), grp in summary.groupby(["parent_name", "cost_mult", "add_entry_delay_bars"], dropna=False):
        base = grp.loc[grp["add_scheme"].eq("no_add")]
        if base.empty:
            continue
        b = base.iloc[0]
        for _, row in grp.iterrows():
            rec = row.to_dict()
            for col in ["return_total", "profit_factor", "win_rate", "max_drawdown", "worst_trade", "best_trade", "top5_winner_share"]:
                try:
                    rec[f"delta_{col}_vs_no_add"] = float(row[col]) - float(b[col])
                except Exception:
                    rec[f"delta_{col}_vs_no_add"] = np.nan
            rows.append(rec)
    return pd.DataFrame(rows).reset_index(drop=True)


def summarize_by_period(combined: pd.DataFrame, period: str) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()
    tmp = combined.copy()
    tmp["exit_time"] = pd.to_datetime(tmp["exit_time"], errors="coerce")
    if period == "year":
        tmp["period"] = tmp["exit_time"].dt.year.astype("Int64").astype(str)
    elif period == "month":
        tmp["period"] = tmp["exit_time"].dt.to_period("M").astype(str)
    else:
        raise ValueError(period)
    rows: list[dict[str, object]] = []
    for (variant_name, p), grp in tmp.groupby(["variant_name", "period"], dropna=False):
        rec = summarize_trade_group(grp)
        rec.update({"variant_name": variant_name, period: p})
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["variant_name", period]).reset_index(drop=True) if rows else pd.DataFrame()


def add_leg_summary(add_legs: pd.DataFrame) -> pd.DataFrame:
    if add_legs.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (cost_mult, delay), grp in add_legs.groupby(["cost_mult", "add_entry_delay_bars"], dropna=False):
        same = pd.to_numeric(grp["add_net_return_same_parent"], errors="coerce")
        own = pd.to_numeric(grp["add_own_net_return"], errors="coerce")
        parent_unreal = pd.to_numeric(grp["parent_unreal_close_ret_at_add_signal"], errors="coerce")
        rows.append(
            {
                "cost_mult": float(cost_mult),
                "add_entry_delay_bars": int(delay),
                "add_legs": int(len(grp)),
                "parents_with_add": int(grp["parent_trade_idx"].nunique()),
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
                "add_signal_down_spike_rank_median": float(pd.to_numeric(grp.get("add_signal_down_spike_rank"), errors="coerce").median()),
                "add_signal_atr_rank_median": float(pd.to_numeric(grp.get("add_signal_atr_rank"), errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["cost_mult", "add_entry_delay_bars"]).reset_index(drop=True)


def context_breakdowns(add_legs: pd.DataFrame) -> pd.DataFrame:
    if add_legs.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    tmp = add_legs.copy()
    tmp["parent_state"] = np.where(pd.to_numeric(tmp["parent_unreal_close_ret_at_add_signal"], errors="coerce") >= 0, "green", "red")
    tmp["parent_armed_bucket"] = np.where(tmp["parent_armed_before_add"].astype(bool), "armed", "not_armed")

    categorical = ["add_signal_session_bucket", "parent_state", "parent_armed_bucket"]
    numeric = [
        "add_signal_down_spike_pct",
        "add_signal_down_spike_rank",
        "add_signal_atr_rank",
        "add_signal_large_trade_share_rank",
        "parent_unreal_close_ret_at_add_signal",
        "parent_mfe_before_add",
        "parent_mae_before_add",
        "parent_bars_since_entry",
        "parent_bars_to_exit",
    ]

    def emit(bucket_col: str, bucket_name: object, grp: pd.DataFrame, *, breakdown_col: str) -> None:
        same = pd.to_numeric(grp["add_net_return_same_parent"], errors="coerce")
        own = pd.to_numeric(grp["add_own_net_return"], errors="coerce")
        rows.append(
            {
                "breakdown_col": breakdown_col,
                "bucket": bucket_name,
                "cost_mult": float(grp["cost_mult"].iloc[0]) if "cost_mult" in grp else np.nan,
                "add_entry_delay_bars": int(grp["add_entry_delay_bars"].iloc[0]) if "add_entry_delay_bars" in grp else np.nan,
                "add_legs": int(len(grp)),
                "parents": int(grp["parent_trade_idx"].nunique()),
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

    for (cost_mult, delay), g0 in tmp.groupby(["cost_mult", "add_entry_delay_bars"], dropna=False):
        for col in categorical:
            if col not in g0.columns:
                continue
            for bucket, grp in g0.groupby(col, dropna=False):
                emit(col, bucket, grp, breakdown_col=col)
        for col in numeric:
            if col not in g0.columns:
                continue
            vals = pd.to_numeric(g0[col], errors="coerce")
            valid = vals.notna()
            if valid.sum() < 4 or vals[valid].nunique() < 2:
                continue
            bucketed = pd.qcut(vals[valid], q=min(4, int(valid.sum())), duplicates="drop")
            gb = g0.loc[valid].assign(_bucket=bucketed.astype(str))
            for bucket, grp in gb.groupby("_bucket", dropna=False):
                emit(col, bucket, grp, breakdown_col=col)
    return pd.DataFrame(rows).sort_values(["cost_mult", "add_entry_delay_bars", "breakdown_col", "bucket"]).reset_index(drop=True) if rows else pd.DataFrame()


def contribution_concentration(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for variant_name, grp in combined.groupby("variant_name", dropna=False):
        if str(grp["add_scheme"].iloc[0]) == "no_add":
            continue
        add = pd.to_numeric(grp.get("add_return_sum_weighted", 0.0), errors="coerce").fillna(0.0)
        pos = add[add > 0].sort_values(ascending=False)
        neg = add[add < 0]
        total_pos = float(pos.sum())
        rec = {
            "variant_name": variant_name,
            "cost_mult": float(grp["cost_mult"].iloc[0]),
            "add_entry_delay_bars": int(grp["add_entry_delay_bars"].iloc[0]),
            "add_scheme": grp["add_scheme"].iloc[0],
            "add_filter_expr": grp["add_filter_expr"].iloc[0],
            "add_weight": float(grp["add_weight"].iloc[0]),
            "parents_with_add": int((pd.to_numeric(grp.get("add_count", 0), errors="coerce").fillna(0) > 0).sum()),
            "total_add_weighted": float(add.sum()),
            "positive_add_sum": total_pos,
            "negative_add_sum": float(neg.sum()),
            "top1_positive_share": float(pos.head(1).sum() / total_pos) if total_pos > 0 else np.nan,
            "top2_positive_share": float(pos.head(2).sum() / total_pos) if total_pos > 0 else np.nan,
            "top3_positive_share": float(pos.head(3).sum() / total_pos) if total_pos > 0 else np.nan,
            "top5_positive_share": float(pos.head(5).sum() / total_pos) if total_pos > 0 else np.nan,
            "worst_add_trade_contribution": float(add.min()) if not add.empty else np.nan,
            "best_add_trade_contribution": float(add.max()) if not add.empty else np.nan,
        }
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["cost_mult", "add_entry_delay_bars", "total_add_weighted"], ascending=[True, True, False]).reset_index(drop=True) if rows else pd.DataFrame()


def filter_definition_table(args: argparse.Namespace) -> pd.DataFrame:
    defs = _filter_definitions()
    rows: list[dict[str, object]] = []
    for delay in _parse_int_list(args.add_entry_delays):
        for scheme in build_schemes(args, add_entry_delay_bars=delay):
            parts = [p.strip() for p in scheme.filter_expr.split("&") if p.strip()]
            rows.append(
                {
                    "add_entry_delay_bars": int(delay),
                    "scheme_name": scheme.name,
                    "filter_expr": scheme.filter_expr,
                    "max_add_count": int(scheme.max_add_count),
                    "add_weight": float(scheme.add_weight),
                    "notes": scheme.notes,
                    "filter_details": " | ".join(defs.get(p, p) for p in parts) if parts else "No filter",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main output writer
# ---------------------------------------------------------------------------


def write_reports(
    out_dir: Path,
    parents: pd.DataFrame,
    add_legs: pd.DataFrame,
    combined: pd.DataFrame,
    summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    delta = delta_vs_no_add(summary)
    yearly = summarize_by_period(combined, "year")
    monthly = summarize_by_period(combined, "month")
    leg_summary = add_leg_summary(add_legs)
    breakdown = context_breakdowns(add_legs)
    concentration = contribution_concentration(combined)
    filters = filter_definition_table(args)

    write_csv(summary, out_dir / "01_scheme_summary.csv", "scheme_summary")
    write_csv(delta, out_dir / "02_scheme_delta_vs_no_add.csv", "scheme_delta_vs_no_add")
    write_csv(yearly, out_dir / "03_yearly.csv", "yearly")
    write_csv(monthly, out_dir / "04_monthly.csv", "monthly")
    write_csv(leg_summary, out_dir / "05_add_leg_summary.csv", "add_leg_summary")
    write_csv(breakdown, out_dir / "06_add_leg_context_breakdown.csv", "add_leg_context_breakdown")
    write_csv(concentration, out_dir / "07_contribution_concentration.csv", "contribution_concentration")
    write_csv(add_legs.head(int(args.max_output_add_legs)), out_dir / "08_overlap_add_legs_full.csv", "overlap_add_legs_full")
    write_csv(parents, out_dir / "09_parent_trades.csv", "parent_trades")

    sample_n = int(getattr(args, "save_trade_sample", 0) or 0)
    if sample_n > 0:
        sample_cols = [
            "variant_name",
            "add_scheme",
            "add_filter_expr",
            "add_weight",
            "add_entry_delay_bars",
            "add_count",
            "add_event_indices",
            "add_entry_times",
            "signal_time",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "exit_reason",
            "net_return_on_equity",
            "add_return_sum_weighted",
            "combined_return_delta_vs_parent",
            "mae_on_equity",
            "mfe_on_equity",
            "down_spike_pct",
            "atr_pct",
            "large_trade_share",
            "close_pos_in_bar",
            "session_bucket",
        ]
        trade_sample = combined[[c for c in sample_cols if c in combined.columns]].head(sample_n).copy() if not combined.empty else pd.DataFrame()
        write_csv(trade_sample, out_dir / "trade_sample.csv", "trade_sample")

    worst_add = add_legs.sort_values("add_net_return_same_parent", ascending=True).head(100).copy() if not add_legs.empty else pd.DataFrame()
    best_add = add_legs.sort_values("add_net_return_same_parent", ascending=False).head(100).copy() if not add_legs.empty else pd.DataFrame()
    write_csv(worst_add, out_dir / "10_worst_add_legs.csv", "worst_add_legs")
    write_csv(best_add, out_dir / "11_best_add_legs.csv", "best_add_legs")
    write_csv(filters, out_dir / "12_filter_definitions.csv", "filter_definitions")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {SCRIPT_NAME} {SCRIPT_VERSION}", flush=True)
    print("[scope] research only; parent=A0_fp_abs_delta_high + single_swing + next_open + time48", flush=True)
    print(f"[args] out_dir={out_dir}", flush=True)

    print(f"[load] trade bars {args.symbol} {args.timeframe} {args.warmup_start_date}->{args.end_date}", flush=True)
    bars = low_v1.load_trade_bars(args)
    print("[events] build existing Low Sweep V1 A0 events/context", flush=True)
    events = low_v1.prepare_events_and_context(bars, args)
    events = _select_a0_events(events, args)
    events = attach_causal_event_ranks(events, args)
    events, positions = _event_positions(bars, events)
    print(f"[events] selected A0_fp_abs_delta_high + single_swing rows={len(events):,}", flush=True)

    parent = _parent_spec(args)
    market = build_market_cache(bars, args)
    parents_all: list[pd.DataFrame] = []
    add_legs_all: list[pd.DataFrame] = []
    combined_all: list[pd.DataFrame] = []

    for cost_mult in _parse_float_list(args.cost_mults):
        print(f"[simulate] parent cost {cost_mult:g}x", flush=True)
        parents, counters = simulate_parent_sequence(bars, events, positions, parent, args, market, cost_mult=float(cost_mult))
        if parents.empty:
            continue
        parents["cost_mult"] = float(cost_mult)
        for key, val in counters.items():
            parents[key] = val
        parents_all.append(parents)

        for delay in _parse_int_list(args.add_entry_delays):
            print(f"[simulate] add legs cost {cost_mult:g}x delay {delay}", flush=True)
            add_legs, by_parent = build_add_legs(
                bars,
                events,
                positions,
                parents,
                parent,
                args,
                market,
                cost_mult=float(cost_mult),
                add_entry_delay_bars=int(delay),
            )
            if not add_legs.empty:
                add_legs["cost_mult"] = float(cost_mult)
                add_legs["add_entry_delay_bars"] = int(delay)
                add_legs_all.append(add_legs)
            combined = build_combined_rows(
                parents,
                by_parent,
                parent,
                args,
                market,
                cost_mult=float(cost_mult),
                add_entry_delay_bars=int(delay),
            )
            combined_all.append(combined)

    parents_df = pd.concat(parents_all, ignore_index=True) if parents_all else pd.DataFrame()
    add_legs_df = pd.concat(add_legs_all, ignore_index=True) if add_legs_all else pd.DataFrame()
    combined_df = _ensure_summary_columns(pd.concat(combined_all, ignore_index=True)) if combined_all else pd.DataFrame()
    summary = summarize_schemes(combined_df)

    write_reports(out_dir, parents_df, add_legs_df, combined_df, summary, args)

    meta = {
        "script": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "parent_exit_mode": args.parent_exit_mode,
        "add_weights": args.add_weights,
        "cost_mults": args.cost_mults,
        "add_entry_delays": args.add_entry_delays,
        "rank_lookback_events": int(args.rank_lookback_events),
        "selected_events": int(len(events)),
        "parent_rows": int(len(parents_df)),
        "add_leg_rows": int(len(add_legs_df)),
        "combined_rows": int(len(combined_df)),
        "notes": "Research-only deep overlap-add diagnostic. Do not treat as formal portfolio backtest without a follow-up replay/audit.",
    }
    (out_dir / "99_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_csv(pd.DataFrame([meta]), out_dir / "99_meta.csv", "meta")

    print(f"[done] wrote reports to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
