#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R06 adaptive risk, protected-structure promotion, add-on, and equity helpers.

R06 freezes the broad R05 candidate family rather than adding more hard entry
filters.  The base universe is the first causal ``n3_4h_or_lt`` episode-reclaim
opportunity.  Quality changes *risk budget* instead of deciding whether a trade
exists.

Key semantics
-------------
* 1m/2m/5m may be used for entry; 1m is never used for trailing structure.
* A 5m/15m ITL/LTL becoming causally knowable does **not** automatically move
  the stop.  R06 can wait for a later higher-high close before promoting that
  low to a protected stop anchor.
* A strong 15m bullish displacement + FVG can act as an independent protected
  anchor from its close onward.
* Stops only ratchet upward.
* Optional add-on is risk-recycling, never averaging down: at most one add-on,
  only after a protected 5m LTL promotion, and total open risk to the common
  stop remains within the setup risk budget.
* No fixed TP is used.  +3/+5/+10% are diagnostics/state milestones only.
* No time stop is used; simulation runs until a structural stop or data end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.core import aggregate_bars, normalize_1m_bars
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex
from src.research_common.swing_liquidity_zone_study.outcomes import RangeMinMaxIndex
from src.research_common.progress import ProgressReporter

EPS = 1e-12


@dataclass(frozen=True)
class R06Config:
    entry_minutes: tuple[int, ...] = (1, 2, 5)
    primary_entry_minutes: int = 2
    stop_buffer_bps: float = 2.0
    market_roundtrip_cost: float = 0.0011
    max_notional_multiple: float = 3.0
    # Risk fractions are fractions of account equity, not margin/leverage.
    risk_schedules: tuple[tuple[str, float, float, float], ...] = (
        ("equal_1pct", 0.0100, 0.0100, 0.0100),
        ("tiered_conservative", 0.0075, 0.0100, 0.0125),
        ("tiered_full", 0.0100, 0.0150, 0.0200),
    )
    cost_scales: tuple[float, ...] = (1.0, 2.0, 3.0)
    major_upgrade_return: float = 0.03

    def validate(self) -> "R06Config":
        if self.primary_entry_minutes not in self.entry_minutes:
            raise ValueError("primary entry timeframe must be inside entry_minutes")
        if 1 not in self.entry_minutes:
            raise ValueError("R06 expects 1m path/entry support")
        if self.stop_buffer_bps < 0:
            raise ValueError("stop buffer cannot be negative")
        if not 0 < self.market_roundtrip_cost < 0.02:
            raise ValueError("market roundtrip cost looks invalid")
        if self.max_notional_multiple <= 0:
            raise ValueError("max notional multiple must be positive")
        if not 0 < self.major_upgrade_return < 0.5:
            raise ValueError("major upgrade return looks invalid")
        for name, b, a, ap in self.risk_schedules:
            if not name or min(b, a, ap) <= 0 or max(b, a, ap) > 0.02 + EPS:
                raise ValueError("risk schedule must be positive and <=2% per setup")
            if not b <= a <= ap:
                raise ValueError("risk schedules must be monotone B<=A<=A+")
        if any(x <= 0 for x in self.cost_scales):
            raise ValueError("cost scales must be positive")
        return self


def _profit_factor(values: Iterable[float]) -> float:
    x = pd.to_numeric(pd.Series(list(values), dtype=float), errors="coerce").dropna()
    if x.empty:
        return np.nan
    gp = float(x.loc[x > 0].sum())
    gl = float(-x.loc[x < 0].sum())
    if gl <= EPS:
        return np.inf if gp > 0 else np.nan
    return gp / gl


def build_adaptive_base_universe(r05_opportunities: pd.DataFrame) -> pd.DataFrame:
    """Freeze the broad R05 N>=3 key-liquidity family and assign causal tiers.

    Important: the tier uses only the *first N>=3 qualifying stage*.  We never
    label an N=3 entry as A merely because the episode later reaches N=4.
    Therefore an A setup here means the qualifying stage itself jumped directly
    to >=4 pools (a fast-consumption observation already known at entry time).
    """
    if r05_opportunities.empty:
        return pd.DataFrame()
    q = r05_opportunities.loc[r05_opportunities["quality_rule"].astype(str).eq("n3_4h_or_lt")].copy()
    if q.empty:
        return q
    pools = pd.to_numeric(q.get("ict_price_pools_cum"), errors="coerce").fillna(0)
    h4 = pd.to_numeric(q.get("ict_htf240_pools_cum"), errors="coerce").fillna(0).ge(1)
    lt = pd.to_numeric(q.get("ict_lt_pools_cum"), errors="coerce").fillna(0).ge(1)
    fast4 = pools.ge(4)
    q["setup_tier"] = np.select(
        [fast4 & h4 & lt, fast4], ["A_plus", "A"], default="B"
    )
    q["fast_n4_at_entry_flag"] = fast4.astype(np.int8)
    q["both_4h_lt_at_entry_flag"] = (h4 & lt).astype(np.int8)
    q["entry_time"] = pd.to_datetime(q["entry_time"], errors="coerce")
    q["signal_available_time"] = pd.to_datetime(q.get("signal_available_time"), errors="coerce")
    # R05 already attached the frozen episode-extreme thesis stop.
    if "stop_episode_extreme" not in q.columns:
        raise KeyError("R06 requires R05 stop_episode_extreme")
    return q.sort_values(["entry_time", "episode_id", "execution_minutes"], kind="stable").reset_index(drop=True)


def _map_time_to_agg_pos(agg: pd.DataFrame, ts: pd.Timestamp, *, use_end: bool) -> int:
    if agg.empty or pd.isna(ts):
        return -1
    arr = pd.DatetimeIndex(pd.to_datetime(agg["bar_end_time"], errors="coerce")) if use_end else pd.DatetimeIndex(agg.index)
    side = "right" if use_end else "left"
    pos = int(arr.searchsorted(pd.Timestamp(ts), side=side)) - (1 if use_end else 0)
    return pos if 0 <= pos < len(agg) else -1


def build_protected_structure_events(
    primary_1m: pd.DataFrame,
    r05_trailing_events: pd.DataFrame,
    *,
    config: R06Config | None = None,
) -> pd.DataFrame:
    """Promote causal ITL/LTL anchors only after later bullish proof.

    For a structural low that becomes causally known at ``activation_time``, we
    freeze a confirmation high using only bars already closed by that time.  The
    low is promoted only after a *later* HTF close exceeds that frozen high.
    Thus "anchor formed" and "stop moves" are separate events.

    15m q95 bullish displacement + FVG is also retained as a direct protected
    anchor because R05 found it useful and its bar low is known at the bar close.
    """
    cfg = (config or R06Config()).validate()
    if r05_trailing_events.empty:
        return pd.DataFrame()
    bars1 = normalize_1m_bars(primary_1m)
    src = r05_trailing_events.copy()
    src["activation_time"] = pd.to_datetime(src["activation_time"], errors="coerce")
    src["anchor_time"] = pd.to_datetime(src["anchor_time"], errors="coerce")
    out: list[dict[str, object]] = []

    for tf in (5, 15):
        agg = aggregate_bars(bars1, tf)
        if agg.empty:
            continue
        high = pd.to_numeric(agg["high"], errors="coerce").to_numpy(dtype=float)
        close = pd.to_numeric(agg["close"], errors="coerce").to_numpy(dtype=float)
        close_idx = SegmentThresholdIndex(close)
        end_times = pd.DatetimeIndex(pd.to_datetime(agg["bar_end_time"], errors="coerce"))
        starts = pd.DatetimeIndex(agg.index)

        structural_types = ["ltl"] if tf == 5 else ["itl", "ltl"]
        part = src.loc[
            pd.to_numeric(src["trail_tf_min"], errors="coerce").eq(tf)
            & src["event_type"].astype(str).isin(structural_types)
        ].copy()
        for row in part.itertuples(index=False):
            anchor_t = pd.Timestamp(row.anchor_time)
            known_t = pd.Timestamp(row.activation_time)
            pivot_pos = int(starts.searchsorted(anchor_t, side="left"))
            if not 0 <= pivot_pos < len(agg):
                continue
            known_pos = int(end_times.searchsorted(known_t, side="right")) - 1
            if not 0 <= known_pos < len(agg) or known_pos < pivot_pos:
                continue
            frozen_high = float(np.nanmax(high[pivot_pos : known_pos + 1]))
            if not np.isfinite(frozen_high):
                continue
            # The confirming HH must occur on a bar that closes strictly after
            # the hierarchy was already knowable.
            search_start = known_pos + 1
            break_pos = close_idx.first_geq(search_start, len(agg) - 1, np.nextafter(frozen_high, np.inf))
            if break_pos < 0:
                continue
            promotion_t = pd.Timestamp(end_times[break_pos])
            out.append({
                "trail_tf_min": tf,
                "event_type": f"protected_{str(row.event_type)}_{tf}m_hh",
                "candidate_activation_time": known_t,
                "anchor_time": anchor_t,
                "promotion_time": promotion_t,
                "anchor_price": float(row.anchor_price),
                "promotion_level": frozen_high,
                "promotion_bar_close": float(close[break_pos]),
                "promotion_reason": "higher_high_close_after_anchor_known",
            })

    shock = src.loc[
        pd.to_numeric(src["trail_tf_min"], errors="coerce").eq(15)
        & src["event_type"].astype(str).eq("bull_shock_q95")
        & pd.to_numeric(src.get("bullish_fvg_flag"), errors="coerce").eq(1)
    ].copy()
    for row in shock.itertuples(index=False):
        out.append({
            "trail_tf_min": 15,
            "event_type": "protected_shock95_fvg_15m",
            "candidate_activation_time": pd.Timestamp(row.activation_time),
            "anchor_time": pd.Timestamp(row.anchor_time),
            "promotion_time": pd.Timestamp(row.activation_time),
            "anchor_price": float(row.anchor_price),
            "promotion_level": np.nan,
            "promotion_bar_close": np.nan,
            "promotion_reason": "causal_15m_q95_bullish_displacement_with_fvg",
        })
    if not out:
        return pd.DataFrame()
    frame = pd.DataFrame(out).dropna(subset=["promotion_time", "anchor_price"])
    return frame.sort_values(["promotion_time", "trail_tf_min", "event_type", "anchor_price"], kind="stable").reset_index(drop=True)


def _event_mask(events: pd.DataFrame, management: str, major_state: bool) -> pd.Series:
    et = events["event_type"].astype(str)
    if management == "r05_immediate_ltl5":
        return et.eq("ltl") & pd.to_numeric(events.get("trail_tf_min"), errors="coerce").eq(5)
    if management == "protected_ltl5":
        return et.eq("protected_ltl_5m_hh")
    if management == "protected_ltl5_or_shock15fvg":
        return et.isin(["protected_ltl_5m_hh", "protected_shock95_fvg_15m"])
    if management == "protected_ltl5_then_itl15_major":
        return et.isin(["protected_itl_15m_hh", "protected_ltl_15m_hh"]) if major_state else et.eq("protected_ltl_5m_hh")
    if management == "protected_ltl5_then_ltl15_major":
        return et.eq("protected_ltl_15m_hh") if major_state else et.eq("protected_ltl_5m_hh")
    raise ValueError(f"unknown R06 management {management}")


def simulate_adaptive_trade_paths(
    opportunities: pd.DataFrame,
    primary_1m: pd.DataFrame,
    r05_trailing_events: pd.DataFrame,
    protected_events: pd.DataFrame,
    *,
    management_variants: Sequence[str] = (
        "r05_immediate_ltl5",
        "protected_ltl5",
        "protected_ltl5_or_shock15fvg",
        "protected_ltl5_then_itl15_major",
        "protected_ltl5_then_ltl15_major",
    ),
    config: R06Config | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Simulate structural exits once per management variant.

    Add-on eligibility is recorded as a candidate on the resulting path, but
    ``none`` vs ``risk_recycle`` sizing is expanded later. This prevents two
    identical price-path simulations for every opportunity.

    Event tables are pre-filtered into compact NumPy arrays per management
    variant. Full-history runs therefore avoid DataFrame ``iloc`` / string-mask
    work inside the event loop.
    """
    cfg = (config or R06Config()).validate()
    if opportunities.empty:
        return pd.DataFrame()
    bars = normalize_1m_bars(primary_1m)
    idx = pd.DatetimeIndex(bars.index)
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    low_idx = SegmentThresholdIndex(low)
    high_idx = SegmentThresholdIndex(high)
    hi_range = RangeMinMaxIndex(high)
    buf = float(cfg.stop_buffer_bps) / 10_000.0

    immediate = r05_trailing_events.copy()
    if not immediate.empty:
        immediate = immediate.loc[
            immediate["event_type"].astype(str).eq("ltl")
            & pd.to_numeric(immediate["trail_tf_min"], errors="coerce").eq(5)
        ].copy()
        immediate["promotion_time"] = pd.to_datetime(immediate["activation_time"], errors="coerce")
        immediate["event_key"] = "r05_immediate_ltl5"
    prot = protected_events.copy()
    if not prot.empty:
        prot["event_key"] = prot["event_type"].astype(str)
    all_events = pd.concat([immediate, prot], ignore_index=True, sort=False) if not immediate.empty else prot
    if all_events.empty:
        return pd.DataFrame()
    all_events["promotion_time"] = pd.to_datetime(all_events["promotion_time"], errors="coerce")
    all_events = all_events.dropna(subset=["promotion_time", "anchor_price"])
    all_events["event_pos_1m"] = idx.searchsorted(pd.DatetimeIndex(all_events["promotion_time"]), side="left")
    all_events = all_events.loc[pd.to_numeric(all_events["event_pos_1m"], errors="coerce").between(0, len(idx) - 1)].copy()
    all_events = all_events.sort_values(["event_pos_1m", "event_key", "anchor_price"], kind="stable").reset_index(drop=True)

    pos_all = pd.to_numeric(all_events["event_pos_1m"], errors="coerce").to_numpy(dtype=np.int64)
    anchor_all = pd.to_numeric(all_events["anchor_price"], errors="coerce").to_numpy(dtype=float)
    key_all = all_events["event_key"].astype(str).to_numpy(dtype=object)

    def _pack(mask: np.ndarray, phase: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ids = np.flatnonzero(mask)
        if phase is None:
            ph = np.zeros(len(ids), dtype=np.int8)
        else:
            ph = np.asarray(phase, dtype=np.int8)[ids]
        return pos_all[ids], anchor_all[ids], key_all[ids], ph

    keys = pd.Series(key_all, dtype="object")
    event_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    event_arrays["r05_immediate_ltl5"] = _pack(keys.eq("r05_immediate_ltl5").to_numpy())
    event_arrays["protected_ltl5"] = _pack(keys.eq("protected_ltl_5m_hh").to_numpy())
    event_arrays["protected_ltl5_or_shock15fvg"] = _pack(keys.isin(["protected_ltl_5m_hh", "protected_shock95_fvg_15m"]).to_numpy())

    mask = keys.isin(["protected_ltl_5m_hh", "protected_itl_15m_hh", "protected_ltl_15m_hh"]).to_numpy()
    phase = np.where(keys.isin(["protected_itl_15m_hh", "protected_ltl_15m_hh"]).to_numpy(), 1, 0)
    event_arrays["protected_ltl5_then_itl15_major"] = _pack(mask, phase)
    mask = keys.isin(["protected_ltl_5m_hh", "protected_ltl_15m_hh"]).to_numpy()
    phase = np.where(keys.eq("protected_ltl_15m_hh").to_numpy(), 1, 0)
    event_arrays["protected_ltl5_then_ltl15_major"] = _pack(mask, phase)
    for management in management_variants:
        if management not in event_arrays:
            raise ValueError(f"unknown R06 management {management}")

    rows: list[dict[str, object]] = []
    total = len(opportunities) * len(management_variants)
    reporter = ProgressReporter("[r06-adaptive-path]", total=total, every=max(1, total // 100), enabled=show_progress)
    done = 0
    for src in opportunities.itertuples(index=False):
        entry_pos = int(src.entry_pos_1m)
        entry = float(src.entry_price)
        initial_stop = float(src.stop_episode_extreme)
        if not 0 <= entry_pos < len(idx) or not np.isfinite(entry) or not np.isfinite(initial_stop) or not 0 < initial_stop < entry:
            continue
        for management in management_variants:
            done += 1
            reporter.update(done)
            positions, anchors, event_keys, phases = event_arrays[management]
            left = int(np.searchsorted(positions, entry_pos, side="right"))
            active_stop = initial_stop
            cursor = entry_pos
            exit_pos = -1
            exit_price = np.nan
            trail_updates = 0
            protected_updates = 0
            major_state = False
            major_state_pos = -1
            addon_pos = -1
            addon_price = np.nan
            addon_stop = np.nan
            addon_reason = "none"
            first_promotion_pos = -1
            last_event = "none"

            for j in range(left, len(positions)):
                pos = int(positions[j])
                if pos <= cursor:
                    continue
                breach = low_idx.first_leq(cursor, pos - 1, active_stop)
                if breach >= 0:
                    exit_pos = int(breach)
                    exit_price = float(min(open_[breach], active_stop)) if np.isfinite(open_[breach]) else active_stop
                    break
                if not major_state:
                    hit3 = high_idx.first_geq(cursor, pos - 1, entry * (1.0 + float(cfg.major_upgrade_return)))
                    if hit3 >= 0:
                        major_state = True
                        major_state_pos = int(hit3)
                # phase 0 = pre-major event, phase 1 = post-major event. For
                # non-stateful strategies phases are all zero and always usable.
                if management.startswith("protected_ltl5_then_"):
                    if (major_state and int(phases[j]) != 1) or ((not major_state) and int(phases[j]) != 0):
                        cursor = pos
                        continue
                candidate = float(anchors[j]) * (1.0 - buf)
                if not np.isfinite(candidate) or candidate <= active_stop:
                    cursor = pos
                    continue
                active_stop = candidate
                trail_updates += 1
                ek = str(event_keys[j])
                if ek.startswith("protected_"):
                    protected_updates += 1
                if first_promotion_pos < 0:
                    first_promotion_pos = pos
                last_event = ek
                if np.isfinite(open_[pos]) and open_[pos] <= active_stop:
                    exit_pos = pos
                    exit_price = float(open_[pos])
                    break
                # Record the first protected 5m LTL candidate that is actually
                # accepted into the active management path. Sizing is deferred.
                if (
                    addon_pos < 0
                    and ek == "protected_ltl_5m_hh"
                    and np.isfinite(open_[pos])
                    and open_[pos] > active_stop
                    and open_[pos] >= entry
                ):
                    addon_pos = pos
                    addon_price = float(open_[pos])
                    addon_stop = float(active_stop)
                    addon_reason = "protected_5m_ltl_after_hh"
                cursor = pos

            if exit_pos < 0:
                breach = low_idx.first_leq(cursor, len(idx) - 1, active_stop)
                if breach >= 0:
                    exit_pos = int(breach)
                    exit_price = float(min(open_[breach], active_stop)) if np.isfinite(open_[breach]) else active_stop
            path_end = exit_pos if exit_pos >= 0 else len(idx) - 1
            _, max_high = hi_range.query(entry_pos, path_end)
            mfe = max_high / entry - 1.0 if np.isfinite(max_high) else np.nan
            rows.append({
                "episode_id": src.episode_id,
                "trade_event_id": src.trade_event_id,
                "stage_id": src.stage_id,
                "execution_minutes": int(src.execution_minutes),
                "setup_tier": src.setup_tier,
                "entry_time": src.entry_time,
                "entry_pos_1m": entry_pos,
                "entry_price": entry,
                "initial_stop_price": initial_stop,
                "initial_risk_return": (entry - initial_stop) / entry,
                "management_variant": management,
                "trail_updates": trail_updates,
                "protected_updates": protected_updates,
                "first_promotion_minutes": int(first_promotion_pos - entry_pos) if first_promotion_pos >= 0 else np.nan,
                "major_state_reached_flag": int(major_state),
                "major_state_minutes": int(major_state_pos - entry_pos) if major_state_pos >= 0 else np.nan,
                "last_trail_event_type": last_event,
                "final_stop_price": active_stop,
                "addon_pos_1m": addon_pos,
                "addon_time": idx[addon_pos] if addon_pos >= 0 else pd.NaT,
                "addon_price": addon_price,
                "addon_stop_price": addon_stop,
                "addon_reason": addon_reason,
                "exit_pos_1m": exit_pos,
                "exit_time": idx[exit_pos] if exit_pos >= 0 else pd.NaT,
                "exit_price": exit_price,
                "holding_minutes": int(exit_pos - entry_pos) if exit_pos >= 0 else np.nan,
                "right_edge_open_flag": int(exit_pos < 0),
                "mfe_until_exit_or_data_end": mfe,
                "reached_3pct_before_exit_flag": int(np.isfinite(max_high) and max_high >= entry * 1.03),
                "reached_5pct_before_exit_flag": int(np.isfinite(max_high) and max_high >= entry * 1.05),
                "reached_10pct_before_exit_flag": int(np.isfinite(max_high) and max_high >= entry * 1.10),
            })
    return pd.DataFrame(rows)

def _risk_budget_for_tier(schedule: tuple[str, float, float, float], tier: str) -> float:
    _, b, a, ap = schedule
    return float(ap if tier == "A_plus" else a if tier == "A" else b)


def attach_risk_sized_trade_returns(
    paths: pd.DataFrame,
    *,
    addon_variants: Sequence[str] = ("none", "risk_recycle_protected_ltl5"),
    config: R06Config | None = None,
) -> pd.DataFrame:
    """Apply fixed risk schedules and expand optional add-on sizing.

    Price-path / exit simulation is shared. ``none`` and risk-recycled add-on
    variants differ only in position sizing, so they must never trigger a second
    event-path simulation.
    """
    cfg = (config or R06Config()).validate()
    if paths.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for p in paths.itertuples(index=False):
        entry = float(p.entry_price)
        stop0 = float(p.initial_stop_price)
        exit_px = float(p.exit_price) if np.isfinite(p.exit_price) else np.nan
        stop_risk = (entry - stop0) / entry
        if not np.isfinite(stop_risk) or stop_risk <= EPS:
            continue
        for addon_variant in addon_variants:
            if addon_variant not in {"none", "risk_recycle_protected_ltl5"}:
                raise ValueError(f"unknown addon variant {addon_variant}")
            for schedule in cfg.risk_schedules:
                budget = _risk_budget_for_tier(schedule, str(p.setup_tier))
                base_notional = min(float(cfg.max_notional_multiple), budget / stop_risk)
                addon_notional = 0.0
                existing_open_risk = np.nan
                addon_unit_risk = np.nan
                total_open_risk_after_addon = np.nan
                if addon_variant != "none" and np.isfinite(p.addon_price) and np.isfinite(p.addon_stop_price):
                    ap = float(p.addon_price)
                    astop = float(p.addon_stop_price)
                    addon_unit_risk = (ap - astop) / ap
                    existing_open_risk = max(0.0, base_notional * (entry - astop) / entry)
                    available = max(0.0, budget - existing_open_risk)
                    if addon_unit_risk > EPS and available > EPS:
                        addon_notional = min(
                            max(0.0, float(cfg.max_notional_multiple) - base_notional),
                            available / addon_unit_risk,
                        )
                    total_open_risk_after_addon = existing_open_risk + max(0.0, addon_notional * max(0.0, addon_unit_risk))
                for cost_scale in cfg.cost_scales:
                    if np.isfinite(exit_px):
                        gross_base = base_notional * (exit_px / entry - 1.0)
                        gross_add = addon_notional * (exit_px / float(p.addon_price) - 1.0) if addon_notional > 0 else 0.0
                        cost = float(cfg.market_roundtrip_cost) * float(cost_scale) * (base_notional + addon_notional)
                        eq_ret = gross_base + gross_add - cost
                    else:
                        eq_ret = np.nan
                    rows.append({
                        **p._asdict(),
                        "addon_variant": addon_variant,
                        "risk_schedule": schedule[0],
                        "risk_budget_fraction": budget,
                        "cost_scale": float(cost_scale),
                        "base_notional_multiple": base_notional,
                        "addon_notional_multiple": addon_notional,
                        "total_notional_multiple": base_notional + addon_notional,
                        "addon_used_flag": int(addon_notional > EPS),
                        "existing_open_risk_at_addon": existing_open_risk,
                        "addon_unit_risk_return": addon_unit_risk,
                        "total_open_risk_after_addon": total_open_risk_after_addon,
                        "strategy_equity_return": eq_ret,
                    })
    return pd.DataFrame(rows)

def select_single_position_trades(sized: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single-ETH-position allocator: overlapping new base signals are skipped.

    One setup may internally add once via the R06 risk-recycling add-on, but a
    distinct episode cannot open another independent base position while the
    current position remains active.
    """
    if sized.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_cols = ["execution_minutes", "management_variant", "addon_variant", "risk_schedule", "cost_scale"]
    executed_parts: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for key, part in sized.groupby(group_cols, dropna=False, sort=True):
        part = part.sort_values(["entry_time", "episode_id"], kind="stable").copy()
        keep = np.zeros(len(part), dtype=bool)
        blocked_until = pd.Timestamp.min
        unresolved = False
        for i, row in enumerate(part.itertuples(index=False)):
            et = pd.Timestamp(row.entry_time)
            if unresolved or et < blocked_until:
                continue
            keep[i] = True
            if pd.isna(row.exit_time):
                unresolved = True
            else:
                blocked_until = pd.Timestamp(row.exit_time)
        ex = part.iloc[np.flatnonzero(keep)].copy()
        executed_parts.append(ex)
        audit.append(dict(zip(group_cols, key)) | {
            "candidate_signals": int(len(part)),
            "executed_positions": int(len(ex)),
            "overlap_skipped": int(len(part) - len(ex)),
            "execution_rate": float(len(ex) / len(part)) if len(part) else np.nan,
            "right_edge_open_positions": int(ex["exit_time"].isna().sum()) if not ex.empty else 0,
        })
    return pd.concat(executed_parts, ignore_index=True, sort=False), pd.DataFrame(audit)


def build_daily_mtm_equity(
    executed: pd.DataFrame,
    primary_1m: pd.DataFrame,
    *,
    initial_equity: float = 1.0,
    market_roundtrip_cost: float = 0.0011,
) -> pd.DataFrame:
    """Build daily mark-to-market equity in O(days + trades).

    ``executed`` must already obey single-position non-overlap semantics.  A
    time-ordered pointer is therefore enough; no per-day DataFrame rescans are
    needed.
    """
    if executed.empty:
        return pd.DataFrame()
    bars = normalize_1m_bars(primary_1m)
    daily = pd.to_numeric(bars["close"], errors="coerce").resample("1D").last().dropna()
    trades = executed.sort_values(["entry_time", "episode_id"], kind="stable").reset_index(drop=True).copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], errors="coerce")
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], errors="coerce")
    trades["addon_time"] = pd.to_datetime(trades.get("addon_time"), errors="coerce")
    first_entry_day = pd.Timestamp(trades["entry_time"].min()).normalize()
    daily = daily.loc[daily.index >= first_entry_day]
    if daily.empty:
        return pd.DataFrame()

    realized_equity = float(initial_equity)
    pointer = 0
    active_idx = -1
    rows: list[dict[str, object]] = []
    n = len(trades)
    for d, px in daily.items():
        day_end = pd.Timestamp(d) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        # If the active trade exited by this day's end, realize it first.
        if active_idx >= 0:
            r = trades.iloc[active_idx]
            if pd.notna(r["exit_time"]) and pd.Timestamp(r["exit_time"]) <= day_end:
                ret = float(r["strategy_equity_return"]) if np.isfinite(r["strategy_equity_return"]) else 0.0
                realized_equity *= max(EPS, 1.0 + ret)
                pointer = active_idx + 1
                active_idx = -1
        # Consume any same-day round trips, then leave at most one open trade.
        while active_idx < 0 and pointer < n:
            r = trades.iloc[pointer]
            if pd.isna(r["entry_time"]) or pd.Timestamp(r["entry_time"]) > day_end:
                break
            if pd.notna(r["exit_time"]) and pd.Timestamp(r["exit_time"]) <= day_end:
                ret = float(r["strategy_equity_return"]) if np.isfinite(r["strategy_equity_return"]) else 0.0
                realized_equity *= max(EPS, 1.0 + ret)
                pointer += 1
                continue
            active_idx = pointer
            break

        mtm = realized_equity
        if active_idx >= 0:
            r = trades.iloc[active_idx]
            entry = float(r["entry_price"])
            base_n = float(r["base_notional_multiple"])
            open_ret = base_n * (float(px) / entry - 1.0)
            cost_in = float(market_roundtrip_cost) * float(r["cost_scale"]) * 0.5 * base_n
            add_n = float(r["addon_notional_multiple"])
            if add_n > 0 and pd.notna(r["addon_time"]) and pd.Timestamp(r["addon_time"]) <= day_end:
                open_ret += add_n * (float(px) / float(r["addon_price"]) - 1.0)
                cost_in += float(market_roundtrip_cost) * float(r["cost_scale"]) * 0.5 * add_n
            mtm = realized_equity * max(EPS, 1.0 + open_ret - cost_in)
        rows.append({"date": pd.Timestamp(d), "equity": mtm})
    return pd.DataFrame(rows)

def _max_drawdown(equity: pd.Series) -> tuple[float, int]:
    x = pd.to_numeric(equity, errors="coerce").dropna()
    if x.empty:
        return np.nan, 0
    peak = x.cummax()
    dd = x / peak - 1.0
    maxdd = float(dd.min())
    in_dd = dd.lt(-EPS).to_numpy(dtype=bool)
    longest = cur = 0
    for flag in in_dd:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)
    return maxdd, longest


def summarize_portfolio(
    executed: pd.DataFrame,
    daily_equity: pd.DataFrame,
    *,
    months: float,
) -> dict[str, object]:
    if executed.empty or daily_equity.empty:
        return {}
    x = pd.to_numeric(executed["strategy_equity_return"], errors="coerce").dropna()
    eq = daily_equity.sort_values("date").copy()
    eqs = pd.to_numeric(eq["equity"], errors="coerce")
    maxdd, dd_days = _max_drawdown(eqs)
    rets_d = eqs.pct_change().fillna(0.0)
    ulcer = float(np.sqrt(np.mean(np.square((eqs / eqs.cummax() - 1.0).clip(upper=0.0)))))
    month_end = eq.set_index("date")["equity"].resample("ME").last().dropna()
    mret = month_end.pct_change().dropna()
    quarter_end = eq.set_index("date")["equity"].resample("QE").last().dropna()
    qret = quarter_end.pct_change().dropna()
    roll90 = eq.set_index("date")["equity"].pct_change(90).dropna()
    logeq = np.log(eqs.clip(lower=EPS).to_numpy(dtype=float))
    if len(logeq) >= 2:
        t = np.arange(len(logeq), dtype=float)
        coef = np.polyfit(t, logeq, 1)
        pred = coef[0] * t + coef[1]
        ss_res = float(np.sum((logeq - pred) ** 2))
        ss_tot = float(np.sum((logeq - logeq.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else np.nan
    else:
        r2 = np.nan
    wins = x[x > 0]
    losses = x[x < 0]
    final_eq = float(eqs.iloc[-1])
    ordered_trades = executed.sort_values("entry_time", kind="stable")
    loss_flags = pd.to_numeric(ordered_trades["strategy_equity_return"], errors="coerce").lt(0).to_numpy(dtype=bool)
    max_loss_streak = cur_loss = 0
    for flag in loss_flags:
        cur_loss = cur_loss + 1 if flag else 0
        max_loss_streak = max(max_loss_streak, cur_loss)
    entry_times = pd.to_datetime(ordered_trades["entry_time"], errors="coerce").dropna().sort_values()
    gaps = entry_times.diff().dt.total_seconds().div(86400.0).dropna()
    total_minutes = max(1.0, (pd.to_datetime(eq["date"], errors="coerce").max() - pd.to_datetime(eq["date"], errors="coerce").min()) / pd.Timedelta(minutes=1))
    exposure_minutes = float(pd.to_numeric(executed["holding_minutes"], errors="coerce").fillna(0).sum())
    # Concentration diagnostics keep the same executed chronology; top returns
    # are zeroed rather than freeing overlap slots.
    ordered = x.sort_values(ascending=False)
    ret_no5 = x.copy()
    ret_no10 = x.copy()
    if len(ordered):
        ret_no5.loc[ordered.index[: min(5, len(ordered))]] = 0.0
        ret_no10.loc[ordered.index[: min(10, len(ordered))]] = 0.0
    eq_no5 = float(np.prod(1.0 + ret_no5.to_numpy(dtype=float)))
    eq_no10 = float(np.prod(1.0 + ret_no10.to_numpy(dtype=float)))
    return {
        "executed_trades": int(len(executed)),
        "trades_per_month": float(len(executed) / months) if months > 0 else np.nan,
        "resolved_trades": int(x.notna().sum()),
        "win_rate": float((x > 0).mean()) if len(x) else np.nan,
        "trade_pf": _profit_factor(x),
        "mean_equity_return_per_trade": float(x.mean()) if len(x) else np.nan,
        "median_equity_return_per_trade": float(x.median()) if len(x) else np.nan,
        "mean_winner": float(wins.mean()) if len(wins) else np.nan,
        "mean_loser": float(losses.mean()) if len(losses) else np.nan,
        "final_equity_multiple": final_eq,
        "total_return": final_eq - 1.0,
        "max_drawdown_daily_mtm": maxdd,
        "longest_drawdown_days": int(dd_days),
        "ulcer_index": ulcer,
        "log_equity_r2": r2,
        "positive_month_rate": float((mret > 0).mean()) if len(mret) else np.nan,
        "median_month_return": float(mret.median()) if len(mret) else np.nan,
        "worst_month_return": float(mret.min()) if len(mret) else np.nan,
        "positive_quarter_rate": float((qret > 0).mean()) if len(qret) else np.nan,
        "rolling90_positive_rate": float((roll90 > 0).mean()) if len(roll90) else np.nan,
        "median_holding_hours": float(pd.to_numeric(executed["holding_minutes"], errors="coerce").median() / 60.0),
        "market_exposure_rate": min(1.0, exposure_minutes / total_minutes),
        "max_days_between_entries": float(gaps.max()) if not gaps.empty else np.nan,
        "max_consecutive_losses": int(max_loss_streak),
        "addon_use_rate": float(pd.to_numeric(executed["addon_used_flag"], errors="coerce").mean()),
        "final_equity_no_top5": eq_no5,
        "final_equity_no_top10": eq_no10,
    }


def r06_causal_audit(
    base: pd.DataFrame,
    protected_events: pd.DataFrame,
    paths: pd.DataFrame,
    sized: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not base.empty:
        rows.append({
            "check": "base_signal_available_not_after_entry",
            "rows": len(base),
            "violations": int((pd.to_datetime(base["signal_available_time"], errors="coerce") > pd.to_datetime(base["entry_time"], errors="coerce")).sum()),
        })
        # A/A+ can only come from what is already known at the first N>=3 stage.
        pools = pd.to_numeric(base["ict_price_pools_cum"], errors="coerce").fillna(0)
        bad = base["setup_tier"].isin(["A", "A_plus"]) & pools.lt(4)
        rows.append({"check": "risk_tier_uses_current_stage_only", "rows": len(base), "violations": int(bad.sum())})
    if not protected_events.empty:
        structural = protected_events["promotion_reason"].astype(str).eq("higher_high_close_after_anchor_known")
        rows.append({
            "check": "protected_structure_promoted_after_candidate_known",
            "rows": int(structural.sum()),
            "violations": int((pd.to_datetime(protected_events.loc[structural, "promotion_time"], errors="coerce") <= pd.to_datetime(protected_events.loc[structural, "candidate_activation_time"], errors="coerce")).sum()),
        })
    if not paths.empty:
        add = pd.to_numeric(paths["addon_pos_1m"], errors="coerce").ge(0)
        rows.append({
            "check": "addon_only_after_entry",
            "rows": int(add.sum()),
            "violations": int((pd.to_datetime(paths.loc[add, "addon_time"], errors="coerce") <= pd.to_datetime(paths.loc[add, "entry_time"], errors="coerce")).sum()),
        })
        rows.append({
            "check": "no_1m_trailing_management",
            "rows": len(paths),
            "violations": int(paths["last_trail_event_type"].astype(str).str.contains("_1m").sum()),
        })
    if sized is not None and not sized.empty:
        # Risk recycling must not create a schedule budget above 2%; sizing can
        # increase notional but not the configured risk budget.
        rows.append({
            "check": "configured_setup_risk_budget_le_2pct",
            "rows": len(sized),
            "violations": int(pd.to_numeric(sized["risk_budget_fraction"], errors="coerce").gt(0.0200001).sum()),
        })
        add = pd.to_numeric(sized.get("addon_used_flag"), errors="coerce").eq(1)
        actual = pd.to_numeric(sized.get("total_open_risk_after_addon"), errors="coerce")
        budget = pd.to_numeric(sized.get("risk_budget_fraction"), errors="coerce")
        rows.append({
            "check": "risk_recycled_addon_stays_within_setup_budget",
            "rows": int(add.sum()),
            "violations": int((add & actual.gt(budget + 1e-9)).sum()),
        })
    return pd.DataFrame(rows)
