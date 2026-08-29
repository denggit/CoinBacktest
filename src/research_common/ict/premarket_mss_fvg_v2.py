#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02 causal primitives for SOXL ICT premarket sweep -> dynamic MSS -> FVG.

R02 fixes two semantic problems found in R01:

1. A sweep no longer freezes whatever short-term pivot happened to be confirmed at
   the instant liquidity was first crossed.  Instead every sweep opens a causal
   *sweep episode*.  As price extends the sweep, the episode tracks the current
   terminal extreme.  The MSS reference is recomputed from the latest causally
   confirmed opposing pivot that existed before that current terminal extreme.
   This supports both a direct V reversal (reference can pre-date the initial
   sweep) and a W/M reversal (a newer pivot formed during the sweep episode can
   replace the old reference).
2. The opposite premarket extreme must still be fresh when the sweep occurs.
   A target that was already traded through earlier in the session is consumed
   liquidity and cannot be used as the take-profit objective for this setup.

The module keeps R01 execution semantics deliberately unchanged where possible:
closed bars only, strict break-bar FVG, conservative same-minute replay, and no
future data in high-timeframe context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .premarket_mss_fvg import (
    EPS,
    NY_TZ,
    PREMARKET_END,
    PREMARKET_START,
    TRADE_END,
    TRADE_START,
    ResearchConfig,
    _add_displacement_features,
    _day_anchor,
    _one_minute_path_between,
    _profit_factor,
    aggregate_closed_bars,
    ensure_ohlc,
    slice_ny_day,
)


@dataclass(frozen=True)
class SweepEpisodeConfig:
    """Structural settings added in R02.

    ``major_swing_min_excursion_range_mult`` is not a PnL-tuned threshold.  It
    defines "obvious 15m swing" geometrically: both sides of the pivot must move
    away by at least one typical completed 15m bar range from the premarket.
    """

    major_swing_min_excursion_range_mult: float = 1.0


def confirmed_pivots_with_excursion(
    frame: pd.DataFrame,
    *,
    left: int,
    right: int,
) -> pd.DataFrame:
    """Return causal pivots plus two-sided price excursion statistics."""

    if left < 1 or right < 1:
        raise ValueError("pivot left/right must be >= 1")
    if frame.empty:
        return pd.DataFrame()

    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(float)
    idx = pd.DatetimeIndex(frame.index)
    available = pd.to_datetime(frame["available_time"])
    rows: list[dict[str, object]] = []

    for pos in range(left, len(frame) - right):
        h = highs[pos]
        l = lows[pos]
        left_h = highs[pos - left : pos]
        right_h = highs[pos + 1 : pos + right + 1]
        left_l = lows[pos - left : pos]
        right_l = lows[pos + 1 : pos + right + 1]

        if np.isfinite(h) and np.isfinite(left_h).all() and np.isfinite(right_h).all():
            neighbor_high = float(max(np.max(left_h), np.max(right_h)))
            if h > neighbor_high:
                left_exc = float(h - np.min(left_l)) if np.isfinite(left_l).all() else np.nan
                right_exc = float(h - np.min(right_l)) if np.isfinite(right_l).all() else np.nan
                two_sided = float(min(left_exc, right_exc)) if np.isfinite(left_exc) and np.isfinite(right_exc) else np.nan
                rows.append(
                    {
                        "pivot_side": "high",
                        "pivot_time": idx[pos],
                        "pivot_price": float(h),
                        "pivot_pos": pos,
                        "confirmation_available_time": pd.Timestamp(available.iloc[pos + right]),
                        "local_prominence_abs": float(h - neighbor_high),
                        "left_excursion_abs": left_exc,
                        "right_excursion_abs": right_exc,
                        "two_sided_excursion_abs": two_sided,
                    }
                )

        if np.isfinite(l) and np.isfinite(left_l).all() and np.isfinite(right_l).all():
            neighbor_low = float(min(np.min(left_l), np.min(right_l)))
            if l < neighbor_low:
                left_exc = float(np.max(left_h) - l) if np.isfinite(left_h).all() else np.nan
                right_exc = float(np.max(right_h) - l) if np.isfinite(right_h).all() else np.nan
                two_sided = float(min(left_exc, right_exc)) if np.isfinite(left_exc) and np.isfinite(right_exc) else np.nan
                rows.append(
                    {
                        "pivot_side": "low",
                        "pivot_time": idx[pos],
                        "pivot_price": float(l),
                        "pivot_pos": pos,
                        "confirmation_available_time": pd.Timestamp(available.iloc[pos + right]),
                        "local_prominence_abs": float(neighbor_low - l),
                        "left_excursion_abs": left_exc,
                        "right_excursion_abs": right_exc,
                        "two_sided_excursion_abs": two_sided,
                    }
                )

    return pd.DataFrame(rows)


def build_premarket_liquidity_levels_v2(
    bars_ny: pd.DataFrame,
    day,
    *,
    pivot_left: int = 2,
    pivot_right: int = 2,
    episode_config: SweepEpisodeConfig = SweepEpisodeConfig(),
) -> pd.DataFrame:
    """Freeze external extremes and only genuinely strong internal 15m swings.

    The strongest internal pivot on each side is still reported even when weak,
    but ``tradable_level`` is false.  This avoids silently forcing every day to
    have a so-called major swing merely because one candidate ranks first.
    """

    pm = slice_ny_day(bars_ny, day, PREMARKET_START, PREMARKET_END)
    if pm.empty:
        return pd.DataFrame()
    pm15 = aggregate_closed_bars(pm, 15)
    if pm15.empty:
        return pd.DataFrame()

    available_at = _day_anchor(day, 8, 30)
    pm_range = float(pm["high"].max() - pm["low"].min())
    median_15m_range = float((pm15["high"] - pm15["low"]).median())
    high_time = pd.Timestamp(pm["high"].idxmax())
    low_time = pd.Timestamp(pm["low"].idxmin())

    rows: list[dict[str, object]] = [
        {
            "ny_date": str(day),
            "liquidity_side": "high",
            "level_type": "premarket_extreme",
            "level_price": float(pm["high"].max()),
            "source_bar_time": high_time,
            "level_available_time": available_at,
            "local_prominence_abs": np.nan,
            "two_sided_excursion_abs": np.nan,
            "excursion_vs_median_15m_range": np.nan,
            "prominence_frac_of_premarket_range": np.nan,
            "liquidity_strength": "external_extreme",
            "tradable_level": True,
            "rejection_reason": "",
        },
        {
            "ny_date": str(day),
            "liquidity_side": "low",
            "level_type": "premarket_extreme",
            "level_price": float(pm["low"].min()),
            "source_bar_time": low_time,
            "level_available_time": available_at,
            "local_prominence_abs": np.nan,
            "two_sided_excursion_abs": np.nan,
            "excursion_vs_median_15m_range": np.nan,
            "prominence_frac_of_premarket_range": np.nan,
            "liquidity_strength": "external_extreme",
            "tradable_level": True,
            "rejection_reason": "",
        },
    ]

    pivots = confirmed_pivots_with_excursion(pm15, left=pivot_left, right=pivot_right)
    if not pivots.empty:
        pivots = pivots.loc[pd.to_datetime(pivots["confirmation_available_time"]) <= available_at].copy()
        if not pivots.empty:
            pivots["excursion_vs_median_15m_range"] = (
                pd.to_numeric(pivots["two_sided_excursion_abs"], errors="coerce") / median_15m_range
                if median_15m_range > EPS
                else np.nan
            )
            pivots["prominence_frac"] = (
                pd.to_numeric(pivots["two_sided_excursion_abs"], errors="coerce") / pm_range
                if pm_range > EPS
                else np.nan
            )

            for side in ("high", "low"):
                candidates = pivots.loc[pivots["pivot_side"] == side].copy()
                if candidates.empty:
                    continue
                candidates = candidates.sort_values(
                    ["two_sided_excursion_abs", "local_prominence_abs", "pivot_time"],
                    ascending=[False, False, True],
                    kind="mergesort",
                )
                best = candidates.iloc[0]
                extreme_time = high_time if side == "high" else low_time
                if pd.Timestamp(best["pivot_time"]) == extreme_time.floor("15min"):
                    continue

                ratio = float(best["excursion_vs_median_15m_range"])
                min_ratio = float(episode_config.major_swing_min_excursion_range_mult)
                tradable = bool(np.isfinite(ratio) and ratio >= min_ratio)
                if not np.isfinite(ratio):
                    strength = "unknown"
                elif ratio >= 1.5:
                    strength = "strong_plus"
                elif ratio >= min_ratio:
                    strength = "strong"
                elif ratio >= 0.5:
                    strength = "normal"
                else:
                    strength = "weak"

                rows.append(
                    {
                        "ny_date": str(day),
                        "liquidity_side": side,
                        "level_type": "major_15m_swing",
                        "level_price": float(best["pivot_price"]),
                        "source_bar_time": pd.Timestamp(best["pivot_time"]),
                        "level_available_time": pd.Timestamp(best["confirmation_available_time"]),
                        "local_prominence_abs": float(best["local_prominence_abs"]),
                        "two_sided_excursion_abs": float(best["two_sided_excursion_abs"]),
                        "excursion_vs_median_15m_range": ratio,
                        "prominence_frac_of_premarket_range": float(best["prominence_frac"]),
                        "liquidity_strength": strength,
                        "tradable_level": tradable,
                        "rejection_reason": "" if tradable else "internal_swing_not_obvious_enough",
                    }
                )

    out = pd.DataFrame(rows)
    out["premarket_high"] = float(pm["high"].max())
    out["premarket_low"] = float(pm["low"].min())
    out["premarket_range"] = pm_range
    out["premarket_range_pct"] = pm_range / float(pm["close"].iloc[-1]) if abs(float(pm["close"].iloc[-1])) > EPS else np.nan
    out["premarket_close"] = float(pm["close"].iloc[-1])
    out["premarket_15m_bars"] = int(len(pm15))
    out["premarket_median_15m_range"] = median_15m_range
    return out.sort_values(["liquidity_side", "level_type", "level_price"]).reset_index(drop=True)


def build_all_premarket_levels_v2(
    bars_ny: pd.DataFrame,
    days: Sequence,
    *,
    pivot_left: int,
    pivot_right: int,
    episode_config: SweepEpisodeConfig = SweepEpisodeConfig(),
) -> pd.DataFrame:
    parts = [
        build_premarket_liquidity_levels_v2(
            bars_ny,
            day,
            pivot_left=pivot_left,
            pivot_right=pivot_right,
            episode_config=episode_config,
        )
        for day in days
    ]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _first_completed_target_touch_before_or_at(
    trade_1m: pd.DataFrame,
    *,
    side: str,
    target: float,
    cutoff_available_time: pd.Timestamp,
) -> pd.Timestamp | pd.NaT:
    idx = pd.DatetimeIndex(trade_1m.index)
    eligible = trade_1m.loc[idx + pd.Timedelta(minutes=1) <= cutoff_available_time]
    if eligible.empty:
        return pd.NaT
    touched = eligible["high"] >= target if side == "LONG" else eligible["low"] <= target
    hits = eligible.loc[touched]
    if hits.empty:
        return pd.NaT
    return pd.Timestamp(hits.index[0]) + pd.Timedelta(minutes=1)


def build_sweep_events_v2(bars_ny: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    """Detect first sweep and explicitly audit target-liquidity freshness."""

    if levels.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for day_text, day_levels in levels.groupby("ny_date", sort=True):
        day = pd.Timestamp(day_text).date()
        trade = slice_ny_day(bars_ny, day, TRADE_START, TRADE_END)
        if trade.empty:
            continue
        highs = pd.to_numeric(trade["high"], errors="coerce").to_numpy(float)
        lows = pd.to_numeric(trade["low"], errors="coerce").to_numpy(float)

        for level in day_levels.to_dict("records"):
            if not bool(level.get("tradable_level", True)):
                continue
            side = str(level["liquidity_side"])
            price = float(level["level_price"])
            mask = highs > price if side == "high" else lows < price
            hit_positions = np.flatnonzero(mask)
            if hit_positions.size == 0:
                continue
            pos = int(hit_positions[0])
            bar_start = pd.Timestamp(trade.index[pos])
            sweep_time = bar_start + pd.Timedelta(minutes=1)
            extreme = float(highs[pos] if side == "high" else lows[pos])
            trade_side = "SHORT" if side == "high" else "LONG"
            target = float(level["premarket_low"] if trade_side == "SHORT" else level["premarket_high"])
            target_touch = _first_completed_target_touch_before_or_at(
                trade,
                side=trade_side,
                target=target,
                cutoff_available_time=sweep_time,
            )
            target_fresh = pd.isna(target_touch)
            same_bar_ambiguous = bool(
                (highs[pos] >= target if trade_side == "LONG" else lows[pos] <= target)
            )
            rows.append(
                {
                    **level,
                    "event_id": f"{day_text}|{side}|{level['level_type']}|{price:.8f}",
                    "trade_side": trade_side,
                    "sweep_bar_start": bar_start,
                    "sweep_time": sweep_time,
                    "sweep_price_extreme_initial": extreme,
                    "sweep_distance_pct": abs(extreme / price - 1.0) if price > EPS else np.nan,
                    "sweep_minute_of_session": int((bar_start.hour * 60 + bar_start.minute) - (8 * 60 + 30)),
                    "sweep_hour_ny": int(bar_start.hour),
                    "target_price": target,
                    "opposite_target_fresh_at_sweep": bool(target_fresh),
                    "opposite_target_first_touch_time": target_touch,
                    "same_bar_target_sweep_ambiguity": same_bar_ambiguous,
                    "setup_eligible_at_sweep": bool(target_fresh),
                    "setup_rejection_reason": "" if target_fresh else "opposite_premarket_liquidity_already_consumed",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["sweep_time", "liquidity_side", "level_price"]).reset_index(drop=True)


def _terminal_extreme(path_1m: pd.DataFrame, *, is_long: bool) -> tuple[pd.Timestamp, float]:
    if path_1m.empty:
        return pd.NaT, np.nan
    if is_long:
        start = pd.Timestamp(pd.to_numeric(path_1m["low"], errors="coerce").idxmin())
        price = float(path_1m.loc[start, "low"])
    else:
        start = pd.Timestamp(pd.to_numeric(path_1m["high"], errors="coerce").idxmax())
        price = float(path_1m.loc[start, "high"])
    return start + pd.Timedelta(minutes=1), price


def _dynamic_reference(
    pivots: pd.DataFrame,
    *,
    side: str,
    terminal_available_time: pd.Timestamp,
    signal_available_time: pd.Timestamp,
) -> Mapping[str, object] | None:
    """Latest opposing pivot before terminal extreme and confirmed by signal.

    The pivot may pre-date the first sweep (V reversal).  If the episode later
    creates a newer pivot before a still-more-extreme terminal print (W/M), that
    newer pivot automatically replaces the older reference.
    """

    if pivots.empty or pd.isna(terminal_available_time):
        return None
    p = pivots.loc[
        (pivots["pivot_side"] == side)
        & (pd.to_datetime(pivots["pivot_time"]) < terminal_available_time)
        & (pd.to_datetime(pivots["confirmation_available_time"]) <= signal_available_time)
    ].copy()
    if p.empty:
        return None
    return p.sort_values(["pivot_time", "confirmation_available_time"], kind="mergesort").iloc[-1].to_dict()


def _build_signal_attempts_for_timeframe_multi_v2(
    bars_ny: pd.DataFrame,
    sweeps: pd.DataFrame,
    *,
    timeframe_minutes: int,
    displacement_body_multipliers: Sequence[float],
    body_window: int,
    min_periods: int,
    close_location_threshold: float,
    pivot_left: int,
    pivot_right: int,
    progress_reporter=None,
) -> pd.DataFrame:
    """Run one execution timeframe for several displacement thresholds in one pass.

    This is deliberately algorithmic-only optimization.  R02 originally re-built
    the same execution bars/pivots once per displacement threshold and repeatedly
    sliced the 1m path for every candidate signal bar.  On multi-year minute data
    that becomes the dominant cost.  Here each day/timeframe is aggregated once,
    the sweep path is advanced monotonically with cumulative extrema/target-touch
    state, and all displacement thresholds share the same causal structural scan.

    Signal semantics are unchanged: every 1m observation included in the state is
    completed by ``signal_time``; terminal-extreme ties keep the first occurrence,
    matching pandas ``idxmin``/``idxmax`` in the slow reference implementation.
    """

    if sweeps.empty:
        return pd.DataFrame()
    tf = int(timeframe_minutes)
    multipliers = tuple(sorted({float(x) for x in displacement_body_multipliers}))
    if not multipliers:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []

    for day_text, day_sweeps in sweeps.groupby("ny_date", sort=True):
        day = pd.Timestamp(day_text).date()
        day_1m = slice_ny_day(bars_ny, day, PREMARKET_START, TRADE_END)
        if day_1m.empty:
            if progress_reporter is not None:
                progress_reporter.step()
            continue
        exec_frame = aggregate_closed_bars(day_1m, tf)
        exec_frame = _add_displacement_features(exec_frame, body_window=body_window, min_periods=min_periods)
        pivots = confirmed_pivots_with_excursion(exec_frame, left=pivot_left, right=pivot_right)
        if exec_frame.empty or pivots.empty:
            if progress_reporter is not None:
                progress_reporter.step()
            continue

        idx = pd.DatetimeIndex(exec_frame.index)
        available = pd.DatetimeIndex(pd.to_datetime(exec_frame["available_time"]))
        available_ns = available.asi8
        highs = pd.to_numeric(exec_frame["high"], errors="coerce").to_numpy(float)
        lows = pd.to_numeric(exec_frame["low"], errors="coerce").to_numpy(float)
        closes = pd.to_numeric(exec_frame["close"], errors="coerce").to_numpy(float)
        opens = pd.to_numeric(exec_frame["open"], errors="coerce").to_numpy(float)
        body_ratio = pd.to_numeric(exec_frame["body_vs_prior_median"], errors="coerce").to_numpy(float)
        close_loc = pd.to_numeric(exec_frame["close_location"], errors="coerce").to_numpy(float)

        one_idx = pd.DatetimeIndex(day_1m.index)
        one_available = one_idx + pd.Timedelta(minutes=1)
        one_available_ns = one_available.asi8
        one_highs = pd.to_numeric(day_1m["high"], errors="coerce").to_numpy(float)
        one_lows = pd.to_numeric(day_1m["low"], errors="coerce").to_numpy(float)

        for sweep in day_sweeps.to_dict("records"):
            if not bool(sweep.get("setup_eligible_at_sweep", True)):
                continue
            sweep_time = pd.Timestamp(sweep["sweep_time"])
            sweep_ns = int(sweep_time.value)
            trade_side = str(sweep["trade_side"])
            is_long = trade_side == "LONG"
            ref_side = "high" if is_long else "low"
            target = float(sweep["target_price"])

            first_pos = max(2, int(np.searchsorted(available_ns, sweep_ns, side="right")))
            if first_pos >= len(exec_frame):
                continue

            path_start = int(np.searchsorted(one_available_ns, sweep_ns, side="left"))
            if path_start >= len(day_1m):
                continue
            scan_j = path_start - 1
            terminal_time = pd.NaT
            terminal_price = np.nan
            target_touched = False
            chosen_by_mult: dict[float, dict[str, object]] = {}

            for pos in range(first_pos, len(exec_frame)):
                signal_time = pd.Timestamp(available[pos])
                if signal_time > _day_anchor(day, 16, 30):
                    break

                end_j = int(np.searchsorted(one_available_ns, int(signal_time.value), side="right") - 1)
                while scan_j < end_j:
                    scan_j += 1
                    if scan_j < path_start:
                        continue
                    if is_long:
                        px = one_lows[scan_j]
                        if np.isfinite(px) and (not np.isfinite(terminal_price) or px < terminal_price):
                            terminal_price = float(px)
                            terminal_time = pd.Timestamp(one_available[scan_j])
                        if np.isfinite(one_highs[scan_j]) and one_highs[scan_j] >= target:
                            target_touched = True
                    else:
                        px = one_highs[scan_j]
                        if np.isfinite(px) and (not np.isfinite(terminal_price) or px > terminal_price):
                            terminal_price = float(px)
                            terminal_time = pd.Timestamp(one_available[scan_j])
                        if np.isfinite(one_lows[scan_j]) and one_lows[scan_j] <= target:
                            target_touched = True

                # Same conservative semantics as the slow path: once completed
                # 1m data through this signal time touched the opposite target,
                # the sweep episode ends before this bar can become a signal.
                if target_touched:
                    break
                if pd.isna(terminal_time) or not np.isfinite(terminal_price):
                    continue

                reference = _dynamic_reference(
                    pivots,
                    side=ref_side,
                    terminal_available_time=pd.Timestamp(terminal_time),
                    signal_available_time=signal_time,
                )
                if reference is None:
                    continue
                ref_price = float(reference["pivot_price"])
                ref_available = pd.Timestamp(reference["confirmation_available_time"])

                if not np.isfinite(body_ratio[pos]):
                    continue
                pending_mults = [m for m in multipliers if m not in chosen_by_mult and body_ratio[pos] >= m]
                if not pending_mults:
                    continue

                if is_long:
                    mss_break = closes[pos] > ref_price
                    fvg = lows[pos] > highs[pos - 2]
                    close_quality = close_loc[pos] >= float(close_location_threshold)
                    entry_limit = lows[pos]
                    fvg_far_edge = highs[pos - 2]
                else:
                    mss_break = closes[pos] < ref_price
                    fvg = highs[pos] < lows[pos - 2]
                    close_quality = close_loc[pos] <= 1.0 - float(close_location_threshold)
                    entry_limit = highs[pos]
                    fvg_far_edge = lows[pos - 2]
                if not (mss_break and fvg and close_quality):
                    continue

                if pd.Timestamp(idx[pos - 2]) < pd.Timestamp(terminal_time):
                    continue

                stop_extreme = float(terminal_price)
                risk_abs = (entry_limit - stop_extreme) if is_long else (stop_extreme - entry_limit)
                reward_abs = (target - entry_limit) if is_long else (entry_limit - target)
                if not np.isfinite(risk_abs) or risk_abs <= EPS or not np.isfinite(reward_abs) or reward_abs <= EPS:
                    continue

                ref_time = pd.Timestamp(reference["pivot_time"])
                reference_source = "post_sweep_dynamic" if ref_time >= pd.Timestamp(sweep["sweep_bar_start"]) else "pre_sweep_v_reference"
                common = {
                    **sweep,
                    "execution_tf": f"{tf}m",
                    "execution_tf_minutes": tf,
                    "mss_model": "dynamic_terminal_extreme_state_machine",
                    "mss_reference_side": ref_side,
                    "mss_reference_time": ref_time,
                    "mss_reference_price": ref_price,
                    "mss_reference_available_time": ref_available,
                    "mss_reference_source": reference_source,
                    "episode_terminal_extreme_time": pd.Timestamp(terminal_time),
                    "episode_terminal_extreme_price": float(terminal_price),
                    "sweep_to_terminal_minutes": float((pd.Timestamp(terminal_time) - sweep_time).total_seconds() / 60.0),
                    "terminal_to_signal_minutes": float((signal_time - pd.Timestamp(terminal_time)).total_seconds() / 60.0),
                    "reference_to_terminal_minutes": float((pd.Timestamp(terminal_time) - ref_time).total_seconds() / 60.0),
                    "signal_bar_start": pd.Timestamp(idx[pos]),
                    "signal_time": signal_time,
                    "signal_open": float(opens[pos]),
                    "signal_high": float(highs[pos]),
                    "signal_low": float(lows[pos]),
                    "signal_close": float(closes[pos]),
                    "signal_body_vs_prior_median": float(body_ratio[pos]),
                    "signal_close_location": float(close_loc[pos]),
                    "fvg_near_edge_entry": float(entry_limit),
                    "fvg_far_edge": float(fvg_far_edge),
                    "fvg_size_abs": abs(float(entry_limit - fvg_far_edge)),
                    "fvg_size_pct": abs(float(entry_limit / fvg_far_edge - 1.0)) if abs(fvg_far_edge) > EPS else np.nan,
                    "stop_price": stop_extreme,
                    "target_price": target,
                    "risk_abs": float(risk_abs),
                    "risk_pct": float(risk_abs / entry_limit),
                    "planned_reward_abs": float(reward_abs),
                    "planned_rr": float(reward_abs / risk_abs),
                    "sweep_to_signal_minutes": float((signal_time - sweep_time).total_seconds() / 60.0),
                    "target_already_touched_before_signal": False,
                    "strict_break_bar_fvg": True,
                }
                for mult in pending_mults:
                    chosen_by_mult[mult] = {**common, "displacement_body_mult": float(mult)}
                if len(chosen_by_mult) == len(multipliers):
                    break

            rows.extend(chosen_by_mult[m] for m in multipliers if m in chosen_by_mult)

        if progress_reporter is not None:
            progress_reporter.step()

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["attempt_id"] = (
        out["event_id"].astype(str)
        + "|tf="
        + out["execution_tf"].astype(str)
        + "|disp="
        + out["displacement_body_mult"].map(lambda x: f"{float(x):.2f}")
        + "|r02"
    )
    return out.sort_values(["signal_time", "attempt_id"], kind="mergesort").reset_index(drop=True)


def build_signal_attempts_for_timeframe_v2(
    bars_ny: pd.DataFrame,
    sweeps: pd.DataFrame,
    *,
    timeframe_minutes: int,
    displacement_body_mult: float,
    body_window: int,
    min_periods: int,
    close_location_threshold: float,
    pivot_left: int,
    pivot_right: int,
) -> pd.DataFrame:
    """Backward-compatible single-threshold wrapper around the optimized scan."""

    return _build_signal_attempts_for_timeframe_multi_v2(
        bars_ny,
        sweeps,
        timeframe_minutes=timeframe_minutes,
        displacement_body_multipliers=(float(displacement_body_mult),),
        body_window=body_window,
        min_periods=min_periods,
        close_location_threshold=close_location_threshold,
        pivot_left=pivot_left,
        pivot_right=pivot_right,
    )


def build_signal_attempts_v2(
    bars_ny: pd.DataFrame,
    sweeps: pd.DataFrame,
    *,
    config: ResearchConfig,
    displacement_body_multipliers: Sequence[float] | None = None,
    progress_enabled: bool = False,
) -> pd.DataFrame:
    multipliers = tuple(displacement_body_multipliers or (config.displacement_body_mult,))
    tfs = tuple(config.execution_timeframes)
    if sweeps.empty or not tfs:
        return pd.DataFrame()

    # Import lazily so this reusable ICT primitive keeps its dependency surface
    # small when progress output is not requested (e.g. unit tests).
    reporter = None
    if progress_enabled:
        from ..progress import ProgressReporter

        total_days = int(sweeps["ny_date"].astype(str).nunique())
        reporter = ProgressReporter(
            label="[research-signals] causal sweep/MSS scan",
            total=max(1, total_days * len(tfs)),
            every=max(1, (total_days * len(tfs)) // 100),
            enabled=True,
        )

    parts: list[pd.DataFrame] = []
    try:
        for tf in tfs:
            part = _build_signal_attempts_for_timeframe_multi_v2(
                bars_ny,
                sweeps,
                timeframe_minutes=tf,
                displacement_body_multipliers=multipliers,
                body_window=config.displacement_body_window,
                min_periods=config.displacement_min_periods,
                close_location_threshold=config.displacement_close_location,
                pivot_left=config.mss_pivot_left,
                pivot_right=config.mss_pivot_right,
                progress_reporter=reporter,
            )
            if not part.empty:
                parts.append(part)
    finally:
        if reporter is not None:
            reporter.close()
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    # Preserve the exact legacy concatenation order: displacement threshold
    # first, then execution timeframe, while each part is ordered by signal.
    mult_order = {float(v): i for i, v in enumerate(multipliers)}
    tf_order = {f"{int(v)}m": i for i, v in enumerate(tfs)}
    out["_mult_order"] = out["displacement_body_mult"].map(mult_order)
    out["_tf_order"] = out["execution_tf"].map(tf_order)
    out = out.sort_values(["_mult_order", "_tf_order", "signal_time", "attempt_id"], kind="mergesort").drop(columns=["_mult_order", "_tf_order"]).reset_index(drop=True)
    # Keep the legacy column position for downstream CSV/replay compatibility.
    if "displacement_body_mult" in out.columns and "mss_model" in out.columns:
        cols = [c for c in out.columns if c != "displacement_body_mult"]
        pos = cols.index("mss_model")
        cols.insert(pos, "displacement_body_mult")
        out = out.loc[:, cols]
    return out

def filter_liquidity_mode_v2(attempts: pd.DataFrame, mode: str) -> pd.DataFrame:
    if attempts.empty:
        return attempts.copy()
    if mode == "extremes_only":
        return attempts.loc[attempts["level_type"] == "premarket_extreme"].copy()
    if mode == "extremes_plus_strong_15m_swing":
        return attempts.loc[attempts["level_type"].isin(["premarket_extreme", "major_15m_swing"])].copy()
    raise ValueError(f"unknown liquidity mode: {mode}")


def build_causal_audit_v2(attempts: pd.DataFrame, lifecycle: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if attempts.empty:
        return pd.DataFrame([{"check": "attempts_non_empty", "passed": False, "violations": 0, "detail": "no signal attempts"}])

    def add(check: str, mask: pd.Series, detail: str) -> None:
        bad = int((~mask.fillna(False)).sum())
        rows.append({"check": check, "passed": bad == 0, "violations": bad, "detail": detail})

    signal = pd.to_datetime(attempts["signal_time"])
    sweep = pd.to_datetime(attempts["sweep_time"])
    level_avail = pd.to_datetime(attempts["level_available_time"])
    ref_avail = pd.to_datetime(attempts["mss_reference_available_time"])
    terminal = pd.to_datetime(attempts["episode_terminal_extreme_time"])
    ref_time = pd.to_datetime(attempts["mss_reference_time"])
    signal_start = pd.to_datetime(attempts["signal_bar_start"])
    tf_delta = pd.to_timedelta(pd.to_numeric(attempts["execution_tf_minutes"]), unit="m")

    add("premarket_level_available_before_sweep", level_avail <= sweep, "frozen liquidity must exist before sweep")
    add("target_fresh_at_sweep", attempts["opposite_target_fresh_at_sweep"].astype(bool), "opposite target liquidity must not be consumed before this sweep")
    add("signal_after_sweep", signal > sweep, "MSS/FVG signal must be after sweep")
    add("terminal_known_by_signal", terminal <= signal, "terminal extreme must already be observable")
    add("dynamic_reference_before_terminal", ref_time < terminal, "MSS reference must structurally precede current terminal extreme")
    add("dynamic_reference_confirmed_by_signal", ref_avail <= signal, "MSS reference must be causally confirmed by signal")
    add("signal_uses_closed_execution_bar", signal == signal_start + tf_delta, "execution bar available only after close")
    add("positive_risk", pd.to_numeric(attempts["risk_abs"], errors="coerce") > 0, "stop must be beyond entry")
    add("positive_reward", pd.to_numeric(attempts["planned_reward_abs"], errors="coerce") > 0, "fresh opposite target must be beyond entry")

    if not lifecycle.empty:
        order_active = pd.to_datetime(lifecycle["order_active_time"])
        signal_life = pd.to_datetime(lifecycle["signal_time"])
        add("order_not_active_before_signal", order_active >= signal_life, "limit cannot exist before signal")
        filled = lifecycle["filled"].fillna(False).astype(bool)
        if filled.any():
            fill_time = pd.to_datetime(lifecycle.loc[filled, "fill_time"])
            active = pd.to_datetime(lifecycle.loc[filled, "order_active_time"])
            add("fill_not_before_order_active", fill_time >= active, "fill search begins only after activation")
    return pd.DataFrame(rows)


def summarize_by_groups(trades: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Group diagnostics without pooling duplicate liquidity-mode replays."""

    cols = [c for c in group_cols if c in trades.columns]
    if trades.empty or not cols:
        return pd.DataFrame()
    filled = trades.loc[trades["filled"].fillna(False).astype(bool)].copy()
    if filled.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    grouper = cols[0] if len(cols) == 1 else cols
    for key, group in filled.groupby(grouper, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        x = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        row = {col: value for col, value in zip(cols, key_tuple)}
        row.update(
            {
                "trades": int(len(group)),
                "win_rate": float((x > 0).mean()) if len(x) else np.nan,
                "mean_net_return": float(x.mean()) if len(x) else np.nan,
                "median_net_return": float(x.median()) if len(x) else np.nan,
                "profit_factor": _profit_factor(x),
                "target_hit_rate": float((group["exit_reason"] == "opposite_premarket_extreme_target").mean()),
                "stop_hit_rate": float(group["exit_reason"].astype(str).str.contains("stop").mean()),
                "median_planned_rr": float(pd.to_numeric(group["planned_rr"], errors="coerce").median()),
                "median_sweep_to_terminal_minutes": float(pd.to_numeric(group["sweep_to_terminal_minutes"], errors="coerce").median()),
                "median_terminal_to_signal_minutes": float(pd.to_numeric(group["terminal_to_signal_minutes"], errors="coerce").median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
