#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal structure-family classification for R09 Swing Low stop-pool hypotheses.

The classifier never uses sweep outcomes.  Every feature is available no later
than the Swing Low's causal ``initial_available_time``; zone-level confluence is
attached only at the later closed sweep bar.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.research_common.swing_liquidity_atlas.pivots import aggregate_timeframe, normalize_primary_bars
from src.research_common.swing_liquidity_zone_study.outcomes import RangeMinMaxIndex

from .config import StructuredStopPoolConfig

EPS = 1e-12
FAMILY_COLUMNS = (
    "hyp_h1_first_higher_low_after_decline",
    "hyp_h2_bos_pullback_higher_low",
    "hyp_h3_layered_base_higher_low",
    "hyp_h4_strong_displacement_origin",
    "hyp_h5_base_breakout_pullback",
    "hyp_h6_multitimeframe_confluence",
    "hyp_h7_trend_continuation_higher_low",
    "hyp_h8_failed_breakdown_then_higher_low",
)


def _high_pivot_mask(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 3:
        return np.zeros(n, dtype=bool)
    out = np.zeros(n, dtype=bool)
    out[1:-1] = (
        np.isfinite(values[1:-1])
        & (values[1:-1] > values[:-2])
        & (values[1:-1] >= values[2:])
    )
    return out


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and abs(den) > EPS else np.nan


def _period(value: pd.Series) -> pd.Series:
    ts = pd.to_datetime(value, errors="coerce")
    return pd.Series(
        np.select(
            [ts < pd.Timestamp("2025-01-01"), ts < pd.Timestamp("2025-10-01")],
            ["EARLY_2023_2024", "MID_2025Q1_Q3"],
            default="BOOKS_2025Q4_2026H1",
        ),
        index=value.index,
        dtype="object",
    )


def hypothesis_definitions() -> pd.DataFrame:
    rows = [
        (
            "H1",
            FAMILY_COLUMNS[0],
            "Large-decline first Higher Low",
            "Immediate Higher Low after the previous Swing Low made a Lower Low; predecessor decline and rebound both exceed frozen early-period medians for that timeframe.",
            "A first reversal cohort may enter on the Higher Low and place clustered stops below it.",
        ),
        (
            "H2",
            FAMILY_COLUMNS[1],
            "BOS pullback Higher Low",
            "Higher Low after a previous Lower Low, with price breaking the last confirmed Swing High before the Higher Low forms.",
            "A causal break of bearish structure can attract more reversal and breakout longs whose stops sit below the pullback low.",
        ),
        (
            "H3",
            FAMILY_COLUMNS[2],
            "Layered base Higher Low",
            "Current Higher Low sits above two preceding Swing Lows that were within 0.25 prior-HTF ATR of each other.",
            "A visible base plus a higher support layer may create two tiers of long stops.",
        ),
        (
            "H4",
            FAMILY_COLUMNS[3],
            "Strong displacement origin",
            "Swing Low's causal right-confirmation reaction is above the frozen early-period 75th percentile for its timeframe.",
            "A low that launched strong displacement is more memorable and may attract later defended entries and protective stops.",
        ),
        (
            "H5",
            FAMILY_COLUMNS[4],
            "Base breakout pullback",
            "Layered/equal-low base is followed by a break above the prior base high and then a Higher Low pullback.",
            "Base participants and breakout participants may share a concentrated invalidation point below the pullback.",
        ),
        (
            "H6",
            FAMILY_COLUMNS[5],
            "Multi-timeframe confluence",
            "At the same closed sweep bar, the 10bp zone contains active first-swept levels from at least two timeframes.",
            "Independent trading horizons may place stops in the same price area.",
        ),
        (
            "H7",
            FAMILY_COLUMNS[6],
            "Trend-continuation Higher Low",
            "At least two consecutive Higher Lows are present and the intervening upswing makes a Higher High.",
            "Mature trend followers may accumulate at continuation lows, though a sweep may also mark genuine trend failure.",
        ),
        (
            "H8",
            FAMILY_COLUMNS[7],
            "Failed breakdown then Higher Low",
            "Previous Swing Low broke an older low but its causal confirmation close recovered above that older low; current low is then higher.",
            "A prior failed breakdown can create a stronger reversal narrative and a new cluster of stops below the subsequent Higher Low.",
        ),
    ]
    return pd.DataFrame(rows, columns=["hypothesis_id", "feature_column", "name", "causal_definition", "behavioral_rationale"])


def _timeframe_raw_features(levels: pd.DataFrame, htf: pd.DataFrame, timeframe: str, minutes: int, cfg: StructuredStopPoolConfig) -> pd.DataFrame:
    part = levels.loc[levels["source_timeframe_min"].astype(int).eq(int(minutes))].copy()
    if part.empty:
        return part
    part = part.sort_values(["pivot_pos_htf", "level_id"], kind="mergesort").reset_index(drop=True)
    positions = pd.to_numeric(part["pivot_pos_htf"], errors="raise").astype(np.int64).to_numpy()
    prices = pd.to_numeric(part["level_price"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(htf["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(htf["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(htf["close"], errors="coerce").to_numpy(dtype=float)
    prev_close = pd.Series(close).shift(1).to_numpy(dtype=float)
    tr = np.nanmax(
        np.vstack(
            [
                high - low,
                np.abs(high - prev_close),
                np.abs(low - prev_close),
            ]
        ),
        axis=0,
    )
    atr = pd.Series(tr).shift(1).rolling(int(cfg.atr_window_htf), min_periods=max(5, cfg.atr_window_htf // 4)).mean().to_numpy(dtype=float)
    high_positions = np.flatnonzero(_high_pivot_mask(high))
    high_tree = RangeMinMaxIndex(high)

    n = len(part)
    prev_price = np.full(n, np.nan)
    prev2_price = np.full(n, np.nan)
    prev_pos = np.full(n, -1, dtype=np.int64)
    prev2_pos = np.full(n, -1, dtype=np.int64)
    reference_high_price = np.full(n, np.nan)
    reference_high_pos = np.full(n, -1, dtype=np.int64)
    prior_leg_high = np.full(n, np.nan)
    current_leg_high = np.full(n, np.nan)
    decline_atr = np.full(n, np.nan)
    rebound_atr = np.full(n, np.nan)
    current_hl_gap_atr = np.full(n, np.nan)
    pullback_fraction = np.full(n, np.nan)
    equal_low_gap_atr = np.full(n, np.nan)
    bos_before_current = np.zeros(n, dtype=bool)
    higher_high_before_current = np.zeros(n, dtype=bool)
    failed_breakdown_prev = np.zeros(n, dtype=bool)
    consecutive_higher_lows = np.zeros(n, dtype=np.int16)

    run_hl = 0
    for j in range(n):
        p = int(positions[j])
        if j >= 1:
            prev_pos[j] = int(positions[j - 1])
            prev_price[j] = float(prices[j - 1])
        if j >= 2:
            p1 = int(positions[j - 1])
            p2 = int(positions[j - 2])
            l1 = float(prices[j - 1])
            l2 = float(prices[j - 2])
            prev2_pos[j] = p2
            prev2_price[j] = l2
            h_idx = int(np.searchsorted(high_positions, p1, side="left") - 1)
            if h_idx >= 0:
                hp = int(high_positions[h_idx])
                reference_high_pos[j] = hp
                reference_high_price[j] = float(high[hp])
            _, leg_now = high_tree.query(p1 + 1, p - 1)
            _, leg_prior = high_tree.query(p2 + 1, p1 - 1)
            current_leg_high[j] = leg_now
            prior_leg_high[j] = leg_prior
            atr_prev = float(atr[p1]) if 0 <= p1 < len(atr) else np.nan
            atr_current = float(atr[p]) if 0 <= p < len(atr) else np.nan
            decline_atr[j] = _safe_div(reference_high_price[j] - l1, atr_prev)
            rebound_atr[j] = _safe_div(leg_now - l1, atr_prev)
            current_hl_gap_atr[j] = _safe_div(float(prices[j]) - l1, atr_current)
            equal_low_gap_atr[j] = _safe_div(abs(l1 - l2), atr_prev)
            pullback_fraction[j] = _safe_div(leg_now - float(prices[j]), leg_now - l1)
            bos_before_current[j] = bool(np.isfinite(leg_now) and np.isfinite(reference_high_price[j]) and leg_now > reference_high_price[j])
            higher_high_before_current[j] = bool(np.isfinite(leg_now) and np.isfinite(leg_prior) and leg_now > leg_prior)
            confirm_pos = p1 + 1
            failed_breakdown_prev[j] = bool(
                l1 < l2
                and 0 <= confirm_pos < len(close)
                and np.isfinite(close[confirm_pos])
                and close[confirm_pos] > l2
            )
            if prices[j] > prices[j - 1]:
                run_hl = run_hl + 1 if prices[j - 1] > prices[j - 2] else 1
            else:
                run_hl = 0
            consecutive_higher_lows[j] = run_hl
        else:
            run_hl = 0

    out = part.copy()
    out["previous_swing_low_pos_htf"] = prev_pos
    out["previous_swing_low_price"] = prev_price
    out["previous2_swing_low_pos_htf"] = prev2_pos
    out["previous2_swing_low_price"] = prev2_price
    out["reference_swing_high_pos_htf"] = reference_high_pos
    out["reference_swing_high_price"] = reference_high_price
    out["prior_leg_high_price"] = prior_leg_high
    out["current_leg_high_price"] = current_leg_high
    out["predecessor_decline_atr"] = decline_atr
    out["rebound_before_current_atr"] = rebound_atr
    out["higher_low_gap_atr"] = current_hl_gap_atr
    out["pullback_fraction_of_rebound"] = pullback_fraction
    out["prior_two_low_gap_atr"] = equal_low_gap_atr
    out["bos_before_current_low"] = bos_before_current
    out["higher_high_before_current_low"] = higher_high_before_current
    out["failed_breakdown_previous_low"] = failed_breakdown_prev
    out["consecutive_higher_low_count"] = consecutive_higher_lows
    out["is_higher_low"] = np.isfinite(prev_price) & (prices > prev_price)
    out["previous_low_was_lower_low"] = np.isfinite(prev2_price) & np.isfinite(prev_price) & (prev_price < prev2_price)
    out["previous_low_was_higher_low"] = np.isfinite(prev2_price) & np.isfinite(prev_price) & (prev_price > prev2_price)
    out["prior_lows_near_equal"] = np.isfinite(equal_low_gap_atr) & (equal_low_gap_atr <= float(cfg.equal_low_tolerance_atr))
    out["base_breakout_before_current_low"] = out["prior_lows_near_equal"].astype(bool) & higher_high_before_current
    out["structure_available_time"] = pd.to_datetime(out["initial_available_time"], errors="coerce")
    out["formation_period"] = _period(out["structure_available_time"])
    return out


def _threshold_row(frame: pd.DataFrame, timeframe: str, cfg: StructuredStopPoolConfig) -> dict[str, Any]:
    train = frame.loc[pd.to_datetime(frame["structure_available_time"], errors="coerce") < pd.Timestamp(cfg.frozen_train_end)].copy()
    base = train.loc[train["is_higher_low"].astype(bool) & train["previous_low_was_lower_low"].astype(bool)]
    decline = pd.to_numeric(base["predecessor_decline_atr"], errors="coerce").dropna()
    rebound = pd.to_numeric(base["rebound_before_current_atr"], errors="coerce").dropna()
    reaction = pd.to_numeric(train["confirmation_reaction_high_bp"], errors="coerce").dropna()
    return {
        "source_timeframe": timeframe,
        "training_rows": int(len(train)),
        "h1_base_rows": int(len(base)),
        "h1_decline_atr_threshold": float(decline.quantile(cfg.h1_decline_quantile)) if len(decline) else np.nan,
        "h1_rebound_atr_threshold": float(rebound.quantile(cfg.h1_rebound_quantile)) if len(rebound) else np.nan,
        "h4_reaction_high_bp_threshold": float(reaction.quantile(cfg.h4_displacement_quantile)) if len(reaction) else np.nan,
        "frozen_train_end": str(cfg.frozen_train_end),
    }


def build_level_structure_features(
    levels: pd.DataFrame,
    primary: pd.DataFrame,
    config: StructuredStopPoolConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config.validate()
    bars = normalize_primary_bars(primary)
    if levels.empty:
        return pd.DataFrame(), pd.DataFrame()
    required = {"level_id", "source_timeframe", "source_timeframe_min", "pivot_pos_htf", "level_price", "initial_available_time"}
    missing = sorted(required.difference(levels.columns))
    if missing:
        raise ValueError(f"levels missing columns: {missing}")
    parts: list[pd.DataFrame] = []
    thresholds: list[dict[str, Any]] = []
    for timeframe, minutes in cfg.timeframes:
        htf = aggregate_timeframe(bars, minutes=int(minutes))
        raw = _timeframe_raw_features(levels, htf, timeframe, int(minutes), cfg)
        if raw.empty:
            continue
        threshold = _threshold_row(raw, timeframe, cfg)
        thresholds.append(threshold)
        d_thr = float(threshold["h1_decline_atr_threshold"])
        r_thr = float(threshold["h1_rebound_atr_threshold"])
        x_thr = float(threshold["h4_reaction_high_bp_threshold"])
        h1_base = raw["is_higher_low"].astype(bool) & raw["previous_low_was_lower_low"].astype(bool)
        raw[FAMILY_COLUMNS[0]] = (
            h1_base
            & pd.to_numeric(raw["predecessor_decline_atr"], errors="coerce").ge(d_thr)
            & pd.to_numeric(raw["rebound_before_current_atr"], errors="coerce").ge(r_thr)
        ) if np.isfinite(d_thr) and np.isfinite(r_thr) else False
        raw[FAMILY_COLUMNS[1]] = h1_base & raw["bos_before_current_low"].astype(bool)
        raw[FAMILY_COLUMNS[2]] = raw["is_higher_low"].astype(bool) & raw["prior_lows_near_equal"].astype(bool)
        raw[FAMILY_COLUMNS[3]] = pd.to_numeric(raw["confirmation_reaction_high_bp"], errors="coerce").ge(x_thr) if np.isfinite(x_thr) else False
        raw[FAMILY_COLUMNS[4]] = raw[FAMILY_COLUMNS[2]].astype(bool) & raw["base_breakout_before_current_low"].astype(bool)
        raw[FAMILY_COLUMNS[5]] = False  # only knowable at the later sweep zone
        raw[FAMILY_COLUMNS[6]] = (
            raw["is_higher_low"].astype(bool)
            & raw["previous_low_was_higher_low"].astype(bool)
            & raw["higher_high_before_current_low"].astype(bool)
        )
        raw[FAMILY_COLUMNS[7]] = raw["is_higher_low"].astype(bool) & raw["failed_breakdown_previous_low"].astype(bool)
        raw["structured_family_count_at_formation"] = raw.loc[:, FAMILY_COLUMNS].astype(bool).sum(axis=1).astype(np.int16)
        raw["any_structured_family_at_formation"] = raw["structured_family_count_at_formation"].gt(0)
        parts.append(raw)
    if not parts:
        return pd.DataFrame(), pd.DataFrame(thresholds)
    out = pd.concat(parts, ignore_index=True, sort=False)
    out = out.sort_values(["structure_available_time", "source_timeframe_min", "pivot_time", "level_id"], kind="mergesort").reset_index(drop=True)
    if out["level_id"].duplicated().any():
        raise RuntimeError("duplicate level_id after structure classification")
    invalid = pd.to_datetime(out["structure_available_time"], errors="coerce") > pd.to_datetime(out["initial_available_time"], errors="coerce")
    if bool(invalid.any()):
        raise RuntimeError("structure feature available after initial level availability")
    return out, pd.DataFrame(thresholds)


def _parse_ids(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    out: list[int] = []
    for token in str(value).split("|"):
        token = token.strip()
        if token:
            out.append(int(token))
    return out


def attach_zone_hypotheses(zones: pd.DataFrame, level_features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate level families to same-bar zones and attach H6 confluence."""
    if zones.empty:
        return zones.copy()
    if level_features.empty:
        raise ValueError("level_features are empty")
    lookup = level_features.set_index("level_id", drop=False)
    rows: list[dict[str, Any]] = []
    for zone in zones.itertuples(index=False):
        record = zone._asdict()
        ids = _parse_ids(record.get("zone_member_level_ids"))
        members = lookup.loc[lookup.index.intersection(ids)].copy()
        if len(members) != len(set(ids)):
            missing = sorted(set(ids).difference(set(members.index.astype(int))))
            raise KeyError(f"zone references unknown level ids: {missing[:10]}")
        for family in FAMILY_COLUMNS:
            if family == FAMILY_COLUMNS[5]:
                continue
            record[family] = bool(members[family].astype(bool).any()) if len(members) else False
            record[f"{family}_member_count"] = int(members[family].astype(bool).sum()) if len(members) else 0
        multi_tf = int(record.get("zone_timeframe_count", 0)) >= 2
        record[FAMILY_COLUMNS[5]] = bool(multi_tf)
        record[f"{FAMILY_COLUMNS[5]}_member_count"] = int(record.get("zone_member_count", 0)) if multi_tf else 0
        span = float(record.get("zone_formation_span_minutes", np.nan))
        max_tf = float(record.get("zone_max_timeframe_min", np.nan))
        record["independent_multitimeframe_confluence"] = bool(multi_tf and np.isfinite(span) and np.isfinite(max_tf) and span >= max_tf)
        record["zone_structured_family_count"] = int(sum(bool(record.get(f, False)) for f in FAMILY_COLUMNS))
        record["zone_has_any_structured_family"] = bool(record["zone_structured_family_count"] > 0)
        record["zone_member_structure_available_time_max"] = pd.to_datetime(members["structure_available_time"], errors="coerce").max() if len(members) else pd.NaT
        record["zone_member_primary_family_timeframes"] = "|".join(
            sorted(set(members.loc[members.loc[:, FAMILY_COLUMNS].astype(bool).any(axis=1), "source_timeframe"].astype(str)))
        ) if len(members) else ""
        rows.append(record)
    out = pd.DataFrame(rows)
    invalid = pd.to_datetime(out["zone_member_structure_available_time_max"], errors="coerce") > pd.to_datetime(out["event_available_time"], errors="coerce")
    if bool(invalid.any()):
        raise RuntimeError(f"zone used structure unavailable at sweep: {int(invalid.sum())}")
    return out
