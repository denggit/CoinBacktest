#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Nested causal event definitions for M1 reversal and M2 momentum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import _safe_divide
from .models import LiquidityHuntConfig

def _distance_bps(price: pd.Series | np.ndarray, reference: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(price, dtype=float)
    r = np.asarray(reference, dtype=float)
    return np.abs(_safe_divide(p - r, r)) * 10_000.0


def _deduplicate_by_cooldown(events: pd.DataFrame, cooldown_minutes: int) -> pd.DataFrame:
    if events.empty:
        return events
    keep = np.zeros(len(events), dtype=bool)
    last: dict[tuple[str, int], pd.Timestamp] = {}
    for i, row in enumerate(events.itertuples(index=False)):
        key = (str(row.stage), int(row.side))
        ts = pd.Timestamp(row.signal_time)
        previous = last.get(key)
        if previous is None or ts - previous >= pd.Timedelta(minutes=int(cooldown_minutes)):
            keep[i] = True
            last[key] = ts
    return events.loc[keep].reset_index(drop=True)


def build_events(frame: pd.DataFrame, cfg: LiquidityHuntConfig, *, range_tag: str) -> pd.DataFrame:
    """Build nested M1/M2 event stages without future information."""

    cfg.validate()
    bars = frame.copy().reset_index(drop=True)
    required = {
        "signal_time",
        "open",
        "high",
        "low",
        "close",
        "direction",
        "notional_multiple",
        "buy_ratio",
        "sell_ratio",
        "prior_support_low",
        "prior_resistance_high",
        "prev_low",
        "prev_high",
        "prev_close",
        "prev_direction",
        "prev_notional_multiple",
        "prev_buy_ratio",
        "prev_sell_ratio",
        "prev_prior_support_low",
        "prev_prior_resistance_high",
    }
    missing = sorted(required - set(bars.columns))
    if missing:
        raise ValueError(f"range features missing: {missing}")

    for name in (
        "book_obi_5s",
        "book_obi_5s_min",
        "book_obi_5s_max",
        "book_ask_depth_25bps_ref_ratio",
        "book_bid_depth_25bps_ref_ratio",
        "book_ask_to_bid_depth_25bps",
        "book_bid_to_ask_depth_25bps",
        "book_nearest_large_bid_price",
        "book_nearest_large_ask_price",
        "book_nearest_large_bid_depth_base",
        "book_nearest_large_ask_depth_base",
        "book_top_bid_wall_price",
        "book_top_ask_wall_price",
        "book_top_bid_wall_depth_base",
        "book_top_ask_wall_depth_base",
        "book_estimated_bid_replenished_base_5s",
        "book_estimated_ask_replenished_base_5s",
        "book_bid_replenish_to_consume",
        "book_ask_replenish_to_consume",
        "fp_low_zone_delta_ratio",
        "fp_high_zone_delta_ratio",
    ):
        if name not in bars.columns:
            bars[name] = np.nan

    # Previous book state is known at the previous completed range-bar close.
    book_shift_cols = [c for c in bars.columns if c.startswith("book_")]
    for col in book_shift_cols:
        bars[f"prev_{col}"] = bars[col].shift(1)

    # ---------------------------- Mode 1 ----------------------------
    long_sweep = (
        (bars["prev_direction"] < 0)
        & (bars["prev_low"] < bars["prev_prior_support_low"])
        & (bars["prev_sell_ratio"] >= cfg.attack_sell_ratio)
        & (bars["prev_notional_multiple"] >= cfg.attack_notional_multiple)
    )
    long_reclaim = (
        (bars["direction"] > 0)
        & (bars["close"] > bars["prev_prior_support_low"])
        & (bars["reclaim_to_attack_volume_ratio"] <= cfg.reclaim_volume_ratio_max)
    )
    short_sweep = (
        (bars["prev_direction"] > 0)
        & (bars["prev_high"] > bars["prev_prior_resistance_high"])
        & (bars["prev_buy_ratio"] >= cfg.attack_buy_ratio)
        & (bars["prev_notional_multiple"] >= cfg.attack_notional_multiple)
    )
    short_reclaim = (
        (bars["direction"] < 0)
        & (bars["close"] < bars["prev_prior_resistance_high"])
        & (bars["reclaim_to_attack_volume_ratio"] <= cfg.reclaim_volume_ratio_max)
    )
    long_obi = (bars["prev_book_obi_5s"] <= -cfg.obi_extreme) & (bars["book_obi_5s"] >= cfg.obi_reversal_positive)
    short_obi = (bars["prev_book_obi_5s"] >= cfg.obi_extreme) & (bars["book_obi_5s"] <= -cfg.obi_reversal_positive)

    support = bars["prev_prior_support_low"]
    resistance = bars["prev_prior_resistance_high"]
    bid_price = bars["book_nearest_large_bid_price"].where(
        bars["book_nearest_large_bid_price"].notna(), bars["book_top_bid_wall_price"]
    )
    ask_price = bars["book_nearest_large_ask_price"].where(
        bars["book_nearest_large_ask_price"].notna(), bars["book_top_ask_wall_price"]
    )
    bid_depth = bars["book_nearest_large_bid_depth_base"].where(
        bars["book_nearest_large_bid_depth_base"].notna(), bars["book_top_bid_wall_depth_base"]
    )
    ask_depth = bars["book_nearest_large_ask_depth_base"].where(
        bars["book_nearest_large_ask_depth_base"].notna(), bars["book_top_ask_wall_depth_base"]
    )
    long_rebuild = (
        (_distance_bps(bid_price, support) <= cfg.rebuilt_distance_bps_max)
        & (bid_depth >= cfg.rebuilt_depth_base_min)
    ) | (
        (bars["book_estimated_bid_replenished_base_5s"] >= cfg.rebuilt_depth_base_min)
        & (bars["book_bid_replenish_to_consume"] >= 1.0)
    )
    short_rebuild = (
        (_distance_bps(ask_price, resistance) <= cfg.rebuilt_distance_bps_max)
        & (ask_depth >= cfg.rebuilt_depth_base_min)
    ) | (
        (bars["book_estimated_ask_replenished_base_5s"] >= cfg.rebuilt_depth_base_min)
        & (bars["book_ask_replenish_to_consume"] >= 1.0)
    )

    # Footprint is used as a descriptive mechanism check, not a hard gate in
    # R01.  Hard-gating it would shrink the already short Books sample and make
    # overfitting easier.
    m1_specs = [
        ("M1_FLOW_RECLAIM", 1, long_sweep & long_reclaim),
        ("M1_FLOW_RECLAIM_OBI", 1, long_sweep & long_reclaim & long_obi),
        ("M1_FLOW_RECLAIM_OBI_REBUILD", 1, long_sweep & long_reclaim & long_obi & long_rebuild),
        ("M1_FLOW_RECLAIM", -1, short_sweep & short_reclaim),
        ("M1_FLOW_RECLAIM_OBI", -1, short_sweep & short_reclaim & short_obi),
        ("M1_FLOW_RECLAIM_OBI_REBUILD", -1, short_sweep & short_reclaim & short_obi & short_rebuild),
    ]

    # ---------------------------- Mode 2 ----------------------------
    long_flow = (
        (bars["prev_direction"] > 0)
        & (bars["direction"] > 0)
        & (bars["prev_buy_ratio"] >= cfg.attack_buy_ratio)
        & (bars["buy_ratio"] >= cfg.attack_buy_ratio)
        & (bars["prev_notional_multiple"] >= cfg.attack_notional_multiple)
        & (bars["notional_multiple"] >= cfg.attack_notional_multiple)
    )
    short_flow = (
        (bars["prev_direction"] < 0)
        & (bars["direction"] < 0)
        & (bars["prev_sell_ratio"] >= cfg.attack_sell_ratio)
        & (bars["sell_ratio"] >= cfg.attack_sell_ratio)
        & (bars["prev_notional_multiple"] >= cfg.attack_notional_multiple)
        & (bars["notional_multiple"] >= cfg.attack_notional_multiple)
    )
    long_obi_sustained = (
        (bars["prev_book_obi_5s"] >= cfg.obi_sustained)
        & (bars["book_obi_5s"] >= cfg.obi_sustained)
        & (bars["prev_book_obi_5s_min"] >= cfg.obi_sustained)
        & (bars["book_obi_5s_min"] >= cfg.obi_sustained)
    )
    short_obi_sustained = (
        (bars["prev_book_obi_5s"] <= -cfg.obi_sustained)
        & (bars["book_obi_5s"] <= -cfg.obi_sustained)
        & (bars["prev_book_obi_5s_max"] <= -cfg.obi_sustained)
        & (bars["book_obi_5s_max"] <= -cfg.obi_sustained)
    )
    long_void = (
        (bars["book_ask_depth_25bps_ref_ratio"] <= cfg.void_depth_ratio_max)
        & (bars["book_ask_to_bid_depth_25bps"] <= cfg.void_side_ratio_max)
    )
    short_void = (
        (bars["book_bid_depth_25bps_ref_ratio"] <= cfg.void_depth_ratio_max)
        & (bars["book_bid_to_ask_depth_25bps"] <= cfg.void_side_ratio_max)
    )
    m2_specs = [
        ("M2_TWO_BAR_ATTACK", 1, long_flow),
        ("M2_TWO_BAR_ATTACK_OBI", 1, long_flow & long_obi_sustained),
        ("M2_TWO_BAR_ATTACK_OBI_VOID", 1, long_flow & long_obi_sustained & long_void),
        ("M2_TWO_BAR_ATTACK", -1, short_flow),
        ("M2_TWO_BAR_ATTACK_OBI", -1, short_flow & short_obi_sustained),
        ("M2_TWO_BAR_ATTACK_OBI_VOID", -1, short_flow & short_obi_sustained & short_void),
    ]

    event_parts: list[pd.DataFrame] = []
    for stage, side, mask in [*m1_specs, *m2_specs]:
        part = bars.loc[pd.Series(mask, index=bars.index).fillna(False)].copy()
        if part.empty:
            continue
        part["stage"] = stage
        part["mode"] = "M1" if stage.startswith("M1") else "M2"
        part["side"] = int(side)
        part["side_name"] = "LONG" if side == 1 else "SHORT"
        part["range_tag"] = str(range_tag)
        part["sweep_price"] = np.where(side == 1, part["prev_low"], part["prev_high"])
        part["structure_price"] = np.where(side == 1, part["prev_prior_support_low"], part["prev_prior_resistance_high"])
        part["first_impulse_low"] = part["prev_low"]
        part["first_impulse_high"] = part["prev_high"]
        part["opposite_liquidity_price"] = np.where(
            side == 1,
            part["book_nearest_large_ask_price"].where(part["book_nearest_large_ask_price"].notna(), part["book_top_ask_wall_price"]),
            part["book_nearest_large_bid_price"].where(part["book_nearest_large_bid_price"].notna(), part["book_top_bid_wall_price"]),
        )
        event_parts.append(part)

    if not event_parts:
        return pd.DataFrame()
    events = pd.concat(event_parts, ignore_index=True)
    events = events.sort_values(["signal_time", "stage", "side"], kind="stable").reset_index(drop=True)
    events = _deduplicate_by_cooldown(events, cfg.cooldown_minutes)
    events.insert(0, "event_id", np.arange(len(events), dtype=np.int64))
    events["book_available_time"] = pd.to_datetime(events["book_available_time"], errors="coerce")
    events["book_available_after_signal_flag"] = events["book_available_time"] > pd.to_datetime(events["signal_time"])
    return events
