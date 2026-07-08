#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Signal and trade runner for ETH MF Low Sweep A0 Footprint."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.edge_lib.mf_low_sweep.config import EDGE_ID, VARIANT_NAME, build_mf_args, primary_variant
from src.edge_lib.mf_low_sweep.events import load_trade_bars
from src.edge_lib.mf_low_sweep.exits import build_market_cache, horizon_from_exit_mode, simulate_time48_trade, summarize_trades
from src.edge_lib.mf_low_sweep.features import (
    build_fixed_candidate_masks,
    prepare_events_and_context,
    safe_num,
)
from src.portfolio_common.allocator import MF_TIME48_LEG, attach_mf_position_metrics


def build_candidate_layer_masks(events: pd.DataFrame, args: Any) -> dict[str, pd.Series]:
    fixed = build_fixed_candidate_masks(events)
    a = fixed.get("A_spike_close_large_share", {}).get("mask", pd.Series(False, index=events.index))
    a = a.fillna(False).astype(bool)
    spike = safe_num(events.get("down_spike_pct", np.nan), events.index)
    fp_abs_delta = safe_num(events.get("fp_max_bucket_abs_delta_pressure", np.nan), events.index)
    fp_abs_high = fp_abs_delta >= float(args.fp_abs_delta_high_threshold)
    return {
        "A0_fp_abs_delta_high": a & (spike >= 0.0100) & fp_abs_high,
    }


def build_support_mask(events: pd.DataFrame, mode: str, args: Any) -> pd.Series:
    if str(mode) == "single_swing":
        return pd.Series(True, index=events.index)
    raise ValueError(f"Unsupported MF low-sweep support_mode={mode}")


def _event_positions(bars: pd.DataFrame, events: pd.DataFrame, max_horizon: int) -> tuple[pd.DataFrame, np.ndarray]:
    if events.empty:
        return events.copy(), np.asarray([], dtype=int)
    times = pd.DatetimeIndex(pd.to_datetime(events["signal_time"]))
    signal_pos = bars.index.get_indexer(times)
    valid = (signal_pos >= 0) & ((signal_pos + int(max_horizon) + 2) < len(bars))
    return events.loc[valid].copy().reset_index(drop=True), signal_pos[valid]


def simulate_variant_set(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    variants: list[Any],
    args: Any,
    *,
    cost_mult: float = 1.0,
    label: str = "[mf] variants",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty or not variants:
        return pd.DataFrame(), pd.DataFrame()
    market = build_market_cache(bars, args)
    layer_masks = build_candidate_layer_masks(events, args)
    all_trades: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for variant in variants:
        layer_mask = layer_masks.get(variant.candidate_layer, pd.Series(False, index=events.index)).fillna(False).astype(bool)
        support_mask = build_support_mask(events, variant.support_mode, args)
        selected = events.loc[layer_mask & support_mask].sort_values("signal_time").reset_index(drop=True)
        max_horizon = horizon_from_exit_mode(variant.exit_mode)
        ev, positions = _event_positions(bars, selected, max_horizon=max_horizon)
        rows: list[dict[str, object]] = []
        skipped_overlap = 0
        skipped_invalid = 0
        last_exit_pos = -1
        total = len(ev)
        print(f"{label} {variant.variant_name}: input_events={total:,}", flush=True)
        for n, (event, signal_pos) in enumerate(zip(ev.to_dict("records"), positions), start=1):
            if int(signal_pos) <= last_exit_pos:
                skipped_overlap += 1
                continue
            rec = simulate_time48_trade(
                bars,
                event,
                int(signal_pos),
                variant,
                args,
                cost_mult=cost_mult,
                market=market,
            )
            if not rec.get("valid"):
                skipped_invalid += 1
                continue
            last_exit_pos = int(rec.get("exit_pos", signal_pos))
            rows.append(rec)
            every = int(getattr(args, "progress_every", 1000) or 1000)
            if every > 0 and n % every == 0:
                print(f"{label} {variant.variant_name}: {n:,}/{total:,}", flush=True)
        trades = pd.DataFrame(rows)
        if not trades.empty:
            all_trades.append(trades)
        summary = summarize_trades(
            trades,
            args,
            extra={
                "variant_name": variant.variant_name,
                "candidate_layer": variant.candidate_layer,
                "support_mode": variant.support_mode,
                "entry_mode": variant.entry_mode,
                "exit_mode": variant.exit_mode,
                "stop_name": variant.stop_spec.name,
                "skipped_overlap": int(skipped_overlap),
                "skipped_invalid": int(skipped_invalid),
                "input_events": int(len(ev)),
            },
        )
        summary_rows.append(summary)

    trade_df = pd.concat(all_trades, ignore_index=True, sort=False) if all_trades else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return trade_df, summary_df


def run_low_sweep_time48_leg(args: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mf_args = build_mf_args(args)
    bars = load_trade_bars(mf_args)
    events = prepare_events_and_context(bars, mf_args)
    trades, summary = simulate_variant_set(
        bars,
        events,
        [primary_variant()],
        mf_args,
        cost_mult=1.0,
        label="[mf] time48",
    )
    if trades.empty:
        return pd.DataFrame(), events, summary
    out = trades.copy()
    out["strategy_leg"] = MF_TIME48_LEG
    out["variant_name"] = VARIANT_NAME
    out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce")
    out["side"] = 1
    out["return_on_sleeve"] = pd.to_numeric(out.get("net_return_on_equity"), errors="coerce")
    out = attach_mf_position_metrics(out, assumed_exposure=1.0)
    return out.dropna(subset=["entry_time", "exit_time", "return_on_sleeve"]), events, summary


__all__ = [
    "EDGE_ID",
    "VARIANT_NAME",
    "build_candidate_layer_masks",
    "build_support_mask",
    "run_low_sweep_time48_leg",
    "simulate_variant_set",
]

