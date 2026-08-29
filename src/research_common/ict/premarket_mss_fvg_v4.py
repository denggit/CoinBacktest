#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ICT causal sweep -> MSS -> displacement discovery primitives.

R05 keeps MSS/FVG causality but does not hard-code a displacement strength gate:

* MSS: a structural event. After liquidity is taken, price closes through the
  latest causally valid opposing short-term pivot.  That pivot may pre-date the
  terminal extreme (direct/V reversal) or form *after* the terminal extreme
  during the developing reversal (for example a new small STH after a low raid).
* Displacement: the *reversal leg* from the terminal extreme through that MSS.
  It is evaluated as a path, not as one special break candle.  The base
  relative-impulse definition asks that the reversal deliver price at least as
  fast as the inbound reference->terminal leg; this directly rejects a sharp
  selloff followed by a slow grind higher (and mirrors for shorts) without a
  PnL-tuned body multiplier or close-location threshold.
* FVG: a three-candle imbalance that may occur anywhere inside the reversal
  displacement leg.  The MSS candle is NOT required to be the FVG third candle.

All execution bars are closed-bar/available-time causal.  FVG selection is
fully known at MSS confirmation; the order only becomes active from signal time
onward in the shared conservative 1m replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .premarket_mss_fvg import (
    EPS,
    TRADE_END,
    aggregate_closed_bars,
    slice_ny_day,
    _day_anchor,
)
from .premarket_mss_fvg_v2 import (
    build_all_premarket_levels_v2,
    build_sweep_events_v2,
    confirmed_pivots_with_excursion,
)


@dataclass(frozen=True)
class ICTDisplacementDiscoveryConfig:
    execution_timeframes: tuple[int, ...] = (1, 2, 5)
    mss_pivot_left: int = 1
    mss_pivot_right: int = 1


def _path_efficiency(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    net = abs(float(x[-1] - x[0]))
    travel = float(np.abs(np.diff(x)).sum())
    return float(net / travel) if travel > EPS else np.nan


def _select_mss_reference(
    pivots: pd.DataFrame,
    *,
    side: str,
    sweep_bar_start: pd.Timestamp,
    terminal_available_time: pd.Timestamp,
    signal_available_time: pd.Timestamp,
) -> Mapping[str, object] | None:
    """Select the latest causally confirmed opposing short-term pivot.

    ICT structure after a liquidity raid can develop in two valid ways:

    * Direct/V reversal: no new opposing short-term pivot forms after the raid,
      so the first meaningful reference may be a pivot that existed before the
      terminal extreme.
    * Post-sweep structure: after the terminal extreme, price can rally/fall,
      form a *new* short-term swing, pull back without taking the terminal
      extreme, and later break that newly formed swing.  That new pivot is a
      valid MSS reference and must replace the older pre-sweep reference.

    The old R02/R04 helper incorrectly required ``pivot_time < terminal_time``
    for every MSS reference, which excluded the second path unless price later
    made another terminal extreme.  This selector gives priority to the latest
    causally confirmed post-sweep pivot, regardless of whether it formed before
    or after the final terminal print; only when none exists do we fall back to
    the latest pivot that preceded the terminal extreme.
    """

    if pivots.empty or pd.isna(terminal_available_time):
        return None
    signal_time = pd.Timestamp(signal_available_time)
    sweep_start = pd.Timestamp(sweep_bar_start)
    terminal_time = pd.Timestamp(terminal_available_time)
    p = pivots.loc[
        (pivots["pivot_side"] == side)
        & (pd.to_datetime(pivots["confirmation_available_time"]) <= signal_time)
    ].copy()
    if p.empty:
        return None

    pivot_times = pd.to_datetime(p["pivot_time"])
    post_sweep = p.loc[pivot_times >= sweep_start].copy()
    if not post_sweep.empty:
        out = post_sweep.sort_values(
            ["pivot_time", "confirmation_available_time"], kind="mergesort"
        ).iloc[-1].to_dict()
        out["reference_relation"] = (
            "post_terminal_dynamic"
            if pd.Timestamp(out["pivot_time"]) >= terminal_time
            else "post_sweep_pre_terminal_dynamic"
        )
        return out

    pre_terminal = p.loc[pivot_times < terminal_time].copy()
    if pre_terminal.empty:
        return None
    out = pre_terminal.sort_values(
        ["pivot_time", "confirmation_available_time"], kind="mergesort"
    ).iloc[-1].to_dict()
    out["reference_relation"] = "pre_sweep_v_reference"
    return out


def _select_inbound_anchor(
    pivots: pd.DataFrame,
    *,
    side: str,
    terminal_available_time: pd.Timestamp,
    signal_available_time: pd.Timestamp,
    fallback_time: pd.Timestamp,
    fallback_price: float,
) -> Mapping[str, object]:
    """Anchor the move *into* the terminal extreme independently of MSS.

    A post-terminal STH/STL can be the MSS reference, but it obviously cannot
    describe the price leg that entered the terminal extreme.  Relative
    displacement therefore uses the latest opposing pivot that occurred before
    the terminal print.  If no such pivot exists, the swept liquidity level at
    the sweep bar is the causal fallback anchor.
    """

    terminal_time = pd.Timestamp(terminal_available_time)
    signal_time = pd.Timestamp(signal_available_time)
    if not pivots.empty:
        p = pivots.loc[
            (pivots["pivot_side"] == side)
            & (pd.to_datetime(pivots["pivot_time"]) < terminal_time)
            & (pd.to_datetime(pivots["confirmation_available_time"]) <= signal_time)
        ].copy()
        if not p.empty:
            row = p.sort_values(
                ["pivot_time", "confirmation_available_time"], kind="mergesort"
            ).iloc[-1].to_dict()
            return {
                "anchor_time": pd.Timestamp(row["pivot_time"]),
                "anchor_price": float(row["pivot_price"]),
                "anchor_source": "pre_terminal_opposing_pivot",
            }
    return {
        "anchor_time": pd.Timestamp(fallback_time),
        "anchor_price": float(fallback_price),
        "anchor_source": "swept_liquidity_fallback",
    }


def _find_latest_fvg_in_leg(
    *,
    is_long: bool,
    highs: np.ndarray,
    lows: np.ndarray,
    available_ns: np.ndarray,
    terminal_time: pd.Timestamp,
    signal_pos: int,
) -> int | None:
    """Latest FVG third-candle position fully known by MSS signal.

    The first candle of the 3-bar FVG may contain the terminal extreme; it only
    needs to have *closed* no earlier than the terminal became observable.  This
    avoids R02's incorrect requirement that the whole three-candle sequence
    start strictly after the terminal print.
    """

    terminal_ns = int(pd.Timestamp(terminal_time).value)
    for k in range(int(signal_pos), 1, -1):
        if int(available_ns[k - 2]) < terminal_ns:
            break
        if is_long:
            if np.isfinite(lows[k]) and np.isfinite(highs[k - 2]) and lows[k] > highs[k - 2]:
                return k
        else:
            if np.isfinite(highs[k]) and np.isfinite(lows[k - 2]) and highs[k] < lows[k - 2]:
                return k
    return None


def _fvg_positions_in_leg(
    *,
    is_long: bool,
    highs: np.ndarray,
    lows: np.ndarray,
    available_ns: np.ndarray,
    terminal_time: pd.Timestamp,
    end_pos: int,
) -> list[int]:
    """All directional 3-bar FVG third-candle positions known by ``end_pos``."""
    terminal_ns = int(pd.Timestamp(terminal_time).value)
    out: list[int] = []
    for k in range(2, int(end_pos) + 1):
        if int(available_ns[k - 2]) < terminal_ns:
            continue
        if is_long:
            if np.isfinite(lows[k]) and np.isfinite(highs[k - 2]) and lows[k] > highs[k - 2]:
                out.append(k)
        else:
            if np.isfinite(highs[k]) and np.isfinite(lows[k - 2]) and highs[k] < lows[k - 2]:
                out.append(k)
    return out


def _displacement_leg_features(
    *,
    is_long: bool,
    tf: int,
    end_pos: int,
    terminal_time: pd.Timestamp,
    terminal_price: float,
    available: pd.DatetimeIndex,
    available_ns: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> dict[str, float]:
    terminal_pos = int(np.searchsorted(available_ns, int(pd.Timestamp(terminal_time).value), side="left"))
    terminal_pos = max(0, min(terminal_pos, int(end_pos)))
    sl = slice(terminal_pos, int(end_pos) + 1)
    sign = 1.0 if is_long else -1.0
    signed_body = sign * (closes[sl] - opens[sl])
    directional = np.clip(signed_body, 0.0, None)
    adverse = np.clip(-signed_body, 0.0, None)
    body_total = float(np.nansum(directional) + np.nansum(adverse))
    directional_body_share = float(np.nansum(directional) / body_total) if body_total > EPS else np.nan
    finite_signed = signed_body[np.isfinite(signed_body)]
    directional_bar_fraction = float((finite_signed > 0).mean()) if finite_signed.size else np.nan

    ranges = highs - lows
    pre_start = max(0, terminal_pos - 20)
    pre_bodies = np.abs(closes[pre_start:terminal_pos] - opens[pre_start:terminal_pos])
    pre_ranges = ranges[pre_start:terminal_pos]
    pre_body_med = float(np.nanmedian(pre_bodies)) if np.isfinite(pre_bodies).any() else np.nan
    pre_range_med = float(np.nanmedian(pre_ranges)) if np.isfinite(pre_ranges).any() else np.nan
    max_dir_body = float(np.nanmax(directional)) if np.isfinite(directional).any() else np.nan
    leg_ranges = ranges[sl]
    max_leg_range = float(np.nanmax(leg_ranges)) if np.isfinite(leg_ranges).any() else np.nan

    end_close = float(closes[int(end_pos)])
    distance = abs(end_close - float(terminal_price))
    minutes = max(float((pd.Timestamp(available[int(end_pos)]) - pd.Timestamp(terminal_time)).total_seconds() / 60.0), float(tf))
    speed = distance / minutes if minutes > 0 else np.nan
    reversal_closes = np.concatenate(([float(terminal_price)], closes[sl]))
    return {
        "leg_bar_count": float(int(end_pos) - terminal_pos + 1),
        "leg_minutes": float(minutes),
        "leg_distance_abs": float(distance),
        "leg_distance_pct": float(distance / abs(float(terminal_price))) if abs(float(terminal_price)) > EPS else np.nan,
        "leg_speed_abs_per_min": float(speed),
        "leg_speed_pct_per_min": float(speed / abs(float(terminal_price))) if abs(float(terminal_price)) > EPS else np.nan,
        "reversal_path_efficiency": _path_efficiency(reversal_closes),
        "directional_body_share": directional_body_share,
        "directional_bar_fraction": directional_bar_fraction,
        "max_directional_body_abs": max_dir_body,
        "max_directional_body_vs_pre20_median": float(max_dir_body / pre_body_med) if np.isfinite(pre_body_med) and pre_body_med > EPS else np.nan,
        "max_leg_range_abs": max_leg_range,
        "max_leg_range_vs_pre20_median": float(max_leg_range / pre_range_med) if np.isfinite(pre_range_med) and pre_range_med > EPS else np.nan,
        "pre20_body_median_abs": pre_body_med,
        "pre20_range_median_abs": pre_range_med,
    }


def _fvg_relation(mss_pos: int, fvg_pos: int) -> str:
    d = int(fvg_pos) - int(mss_pos)
    if d < 0:
        return "before_mss_within_reversal"
    if d == 0:
        return "fvg_third_on_mss_bar"
    if d == 1:
        return "mss_bar_is_fvg_middle"
    if d == 2:
        return "mss_bar_is_fvg_first"
    return "post_mss_continuation_fvg"


def _build_attempts_one_tf(
    bars_ny: pd.DataFrame,
    sweeps: pd.DataFrame,
    *,
    timeframe_minutes: int,
    pivot_left: int,
    pivot_right: int,
    progress_reporter=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Broad causal ICT candidate builder for displacement discovery.

    No displacement-strength threshold is used.  Once liquidity is swept, the
    first causally valid MSS is recorded.  A directional FVG may already exist
    before/on MSS, or finish forming afterwards while the same terminal extreme
    remains intact.  The order signal is only emitted once *both* MSS and FVG
    are known.  If a new terminal extreme prints before an FVG is available,
    the prior MSS state is invalidated and the episode searches again.
    """
    if sweeps.empty:
        return pd.DataFrame(), pd.DataFrame()

    tf = int(timeframe_minutes)
    attempts: list[dict[str, object]] = []
    funnel: list[dict[str, object]] = []

    for day_text, day_sweeps in sweeps.groupby("ny_date", sort=True):
        day = pd.Timestamp(day_text).date()
        day_1m = slice_ny_day(bars_ny, day, pd.Timestamp("04:00").time(), TRADE_END)
        exec_frame = aggregate_closed_bars(day_1m, tf) if not day_1m.empty else pd.DataFrame()
        pivots = confirmed_pivots_with_excursion(exec_frame, left=pivot_left, right=pivot_right) if not exec_frame.empty else pd.DataFrame()
        if exec_frame.empty or pivots.empty:
            if progress_reporter is not None:
                progress_reporter.step()
            continue

        idx = pd.DatetimeIndex(exec_frame.index)
        available = pd.DatetimeIndex(pd.to_datetime(exec_frame["available_time"]))
        # Pandas 3 may preserve microsecond-resolution DatetimeIndex values.
        # Timestamp.value is always nanoseconds, while DatetimeIndex.asi8 follows
        # the index resolution. Normalize the integer search axis explicitly to
        # ns before comparing with Timestamp.value; otherwise us-vs-ns differs by
        # 1000x and every sweep is incorrectly placed beyond the end of the day.
        available_ns = available.as_unit("ns").asi8
        highs = pd.to_numeric(exec_frame["high"], errors="coerce").to_numpy(float)
        lows = pd.to_numeric(exec_frame["low"], errors="coerce").to_numpy(float)
        closes = pd.to_numeric(exec_frame["close"], errors="coerce").to_numpy(float)
        opens = pd.to_numeric(exec_frame["open"], errors="coerce").to_numpy(float)

        one_idx = pd.DatetimeIndex(day_1m.index)
        one_available = one_idx + pd.Timedelta(minutes=1)
        one_available_ns = one_available.as_unit("ns").asi8
        one_highs = pd.to_numeric(day_1m["high"], errors="coerce").to_numpy(float)
        one_lows = pd.to_numeric(day_1m["low"], errors="coerce").to_numpy(float)

        for sweep in day_sweeps.to_dict("records"):
            if not bool(sweep.get("setup_eligible_at_sweep", True)):
                continue
            sweep_time = pd.Timestamp(sweep["sweep_time"])
            sweep_ns = int(sweep_time.value)
            is_long = str(sweep["trade_side"]) == "LONG"
            ref_side = "high" if is_long else "low"
            target = float(sweep["target_price"])

            first_pos = max(0, int(np.searchsorted(available_ns, sweep_ns, side="right")))
            path_start = int(np.searchsorted(one_available_ns, sweep_ns, side="left"))
            if first_pos >= len(exec_frame) or path_start >= len(day_1m):
                continue

            scan_j = path_start - 1
            terminal_time = pd.NaT
            terminal_source_time = pd.NaT
            terminal_price = np.nan
            terminal_version = 0
            target_touched = False
            mss_state: dict[str, object] | None = None
            ever_mss = False
            ever_fvg = False
            emitted = False

            for pos in range(first_pos, len(exec_frame)):
                now = pd.Timestamp(available[pos])
                if now > _day_anchor(day, 16, 30):
                    break

                terminal_changed = False
                end_j = int(np.searchsorted(one_available_ns, int(now.value), side="right") - 1)
                while scan_j < end_j:
                    scan_j += 1
                    if scan_j < path_start:
                        continue
                    if is_long:
                        px = one_lows[scan_j]
                        if np.isfinite(px) and (not np.isfinite(terminal_price) or px < terminal_price):
                            terminal_price = float(px)
                            terminal_time = pd.Timestamp(one_available[scan_j])
                            terminal_source_time = pd.Timestamp(one_idx[scan_j])
                            terminal_version += 1
                            terminal_changed = True
                        if np.isfinite(one_highs[scan_j]) and one_highs[scan_j] >= target:
                            target_touched = True
                    else:
                        px = one_highs[scan_j]
                        if np.isfinite(px) and (not np.isfinite(terminal_price) or px > terminal_price):
                            terminal_price = float(px)
                            terminal_time = pd.Timestamp(one_available[scan_j])
                            terminal_source_time = pd.Timestamp(one_idx[scan_j])
                            terminal_version += 1
                            terminal_changed = True
                        if np.isfinite(one_lows[scan_j]) and one_lows[scan_j] <= target:
                            target_touched = True

                if target_touched:
                    break
                if pd.isna(terminal_time) or not np.isfinite(terminal_price):
                    continue
                if terminal_changed and mss_state is not None and int(mss_state["terminal_version"]) != terminal_version:
                    mss_state = None

                if mss_state is None:
                    reference = _select_mss_reference(
                        pivots,
                        side=ref_side,
                        sweep_bar_start=pd.Timestamp(sweep["sweep_bar_start"]),
                        terminal_available_time=pd.Timestamp(terminal_time),
                        signal_available_time=now,
                    )
                    if reference is None:
                        continue
                    ref_price = float(reference["pivot_price"])
                    mss_break = bool(closes[pos] > ref_price) if is_long else bool(closes[pos] < ref_price)
                    if not mss_break:
                        continue
                    ever_mss = True
                    inbound_anchor = _select_inbound_anchor(
                        pivots,
                        side=ref_side,
                        terminal_available_time=pd.Timestamp(terminal_time),
                        signal_available_time=now,
                        fallback_time=pd.Timestamp(sweep["sweep_bar_start"]),
                        fallback_price=float(sweep["level_price"]),
                    )
                    inbound_anchor_time = pd.Timestamp(inbound_anchor["anchor_time"])
                    inbound_anchor_price = float(inbound_anchor["anchor_price"])
                    inbound_minutes = max(float((pd.Timestamp(terminal_source_time) - inbound_anchor_time).total_seconds() / 60.0), float(tf))
                    inbound_distance = abs(inbound_anchor_price - float(terminal_price))
                    inbound_speed = inbound_distance / inbound_minutes if inbound_minutes > 0 else np.nan
                    mss_leg = _displacement_leg_features(
                        is_long=is_long, tf=tf, end_pos=pos, terminal_time=pd.Timestamp(terminal_time),
                        terminal_price=float(terminal_price), available=available, available_ns=available_ns,
                        opens=opens, highs=highs, lows=lows, closes=closes,
                    )
                    speed_ratio = float(mss_leg["leg_speed_abs_per_min"] / inbound_speed) if np.isfinite(inbound_speed) and inbound_speed > EPS else np.nan
                    mss_overshoot = (float(closes[pos]) - ref_price) if is_long else (ref_price - float(closes[pos]))
                    mss_state = {
                        "terminal_version": terminal_version,
                        "terminal_time": pd.Timestamp(terminal_time),
                        "terminal_source_time": pd.Timestamp(terminal_source_time),
                        "terminal_price": float(terminal_price),
                        "mss_pos": int(pos),
                        "mss_time": now,
                        "mss_reference": reference,
                        "inbound_anchor": inbound_anchor,
                        "inbound_minutes": inbound_minutes,
                        "inbound_distance_abs": inbound_distance,
                        "inbound_speed_abs_per_min": inbound_speed,
                        "displacement_speed_ratio": speed_ratio,
                        "mss_leg": mss_leg,
                        "mss_overshoot_abs": float(mss_overshoot),
                        "mss_overshoot_pct": float(mss_overshoot / abs(ref_price)) if abs(ref_price) > EPS else np.nan,
                    }

                assert mss_state is not None
                if int(mss_state["terminal_version"]) != terminal_version:
                    mss_state = None
                    continue

                fvg_positions = _fvg_positions_in_leg(
                    is_long=is_long,
                    highs=highs,
                    lows=lows,
                    available_ns=available_ns,
                    terminal_time=pd.Timestamp(mss_state["terminal_time"]),
                    end_pos=pos,
                )
                if not fvg_positions:
                    continue
                # If an FVG was already known at MSS, use the latest one known at
                # MSS. Otherwise the first subsequently completed directional FVG
                # triggers the entry setup. This is causal and does not choose a
                # future "best" gap.
                mss_pos = int(mss_state["mss_pos"])
                at_mss = [x for x in fvg_positions if x <= mss_pos]
                if at_mss:
                    fp = int(at_mss[-1])
                else:
                    after = [x for x in fvg_positions if x > mss_pos]
                    if not after:
                        continue
                    fp = int(after[0])
                directional_fvg_count_at_mss = int(len(at_mss))
                directional_fvg_count_to_signal = int(sum(1 for x in fvg_positions if x <= fp))
                selected_fvg_sequence_rank = int([x for x in fvg_positions if x <= fp].index(fp) + 1)
                if fp > pos:
                    continue
                ever_fvg = True

                if is_long:
                    entry = float(lows[fp]); far = float(highs[fp - 2])
                else:
                    entry = float(highs[fp]); far = float(lows[fp - 2])
                stop = float(mss_state["terminal_price"])
                risk = entry - stop if is_long else stop - entry
                reward = target - entry if is_long else entry - target
                if not np.isfinite(risk) or risk <= EPS or not np.isfinite(reward) or reward <= EPS:
                    break

                signal_time = max(pd.Timestamp(mss_state["mss_time"]), pd.Timestamp(available[fp]))
                signal_pos = max(mss_pos, fp)
                signal_leg = _displacement_leg_features(
                    is_long=is_long, tf=tf, end_pos=signal_pos, terminal_time=pd.Timestamp(mss_state["terminal_time"]),
                    terminal_price=stop, available=available, available_ns=available_ns,
                    opens=opens, highs=highs, lows=lows, closes=closes,
                )
                ref = mss_state["mss_reference"]
                ref_price = float(ref["pivot_price"])
                mss_close = float(closes[mss_pos])
                mss_leg_distance = abs(mss_close - stop)
                entry_distance = (entry - stop) if is_long else (stop - entry)
                fvg_depth = float(entry_distance / mss_leg_distance) if mss_leg_distance > EPS else np.nan
                fvg_size = abs(entry - far)
                fvg_relation = _fvg_relation(mss_pos, fp)
                offset = int(fp - mss_pos)

                attempts.append({
                    **sweep,
                    "execution_tf": f"{tf}m", "execution_tf_minutes": tf,
                    "mss_model": "ict_liquidity_then_short_term_structure_break",
                    "mss_reference_side": ref_side,
                    "mss_reference_time": pd.Timestamp(ref["pivot_time"]),
                    "mss_reference_price": ref_price,
                    "mss_reference_available_time": pd.Timestamp(ref["confirmation_available_time"]),
                    "mss_reference_source": str(ref.get("reference_relation", "unknown")),
                    "episode_terminal_extreme_time": pd.Timestamp(mss_state["terminal_time"]),
                    "episode_terminal_extreme_source_time": pd.Timestamp(mss_state["terminal_source_time"]),
                    "episode_terminal_extreme_price": stop,
                    "mss_bar_start": pd.Timestamp(idx[mss_pos]),
                    "mss_time": pd.Timestamp(mss_state["mss_time"]),
                    "mss_close": mss_close,
                    "mss_overshoot_abs": float(mss_state["mss_overshoot_abs"]),
                    "mss_overshoot_pct": float(mss_state["mss_overshoot_pct"]),
                    "signal_bar_start": pd.Timestamp(idx[signal_pos]),
                    "signal_time": signal_time,
                    "signal_open": float(opens[signal_pos]), "signal_high": float(highs[signal_pos]),
                    "signal_low": float(lows[signal_pos]), "signal_close": float(closes[signal_pos]),
                    "fvg_third_bar_start": pd.Timestamp(idx[fp]),
                    "fvg_available_time": pd.Timestamp(available[fp]),
                    "fvg_near_edge_entry": entry, "fvg_far_edge": far,
                    "fvg_size_abs": fvg_size,
                    "fvg_size_pct": float(fvg_size / abs(far)) if abs(far) > EPS else np.nan,
                    "fvg_size_vs_risk": float(fvg_size / risk) if risk > EPS else np.nan,
                    "fvg_relation_to_mss": fvg_relation,
                    "fvg_mss_offset_bars": offset,
                    "directional_fvg_count_at_mss": directional_fvg_count_at_mss,
                    "directional_fvg_count_to_signal": directional_fvg_count_to_signal,
                    "selected_fvg_sequence_rank": selected_fvg_sequence_rank,
                    "mss_to_fvg_minutes": float((pd.Timestamp(available[fp]) - pd.Timestamp(mss_state["mss_time"])).total_seconds() / 60.0),
                    "fvg_entry_depth_vs_mss_leg": fvg_depth,
                    "displacement_model": "ungated_feature_discovery",
                    "displacement_relative_impulse_pass": bool(np.isfinite(mss_state["displacement_speed_ratio"]) and float(mss_state["displacement_speed_ratio"]) >= 1.0),
                    "inbound_anchor_time": pd.Timestamp(mss_state["inbound_anchor"]["anchor_time"]),
                    "inbound_anchor_price": float(mss_state["inbound_anchor"]["anchor_price"]),
                    "inbound_anchor_source": str(mss_state["inbound_anchor"]["anchor_source"]),
                    "inbound_minutes": float(mss_state["inbound_minutes"]),
                    "inbound_distance_abs": float(mss_state["inbound_distance_abs"]),
                    "inbound_speed_abs_per_min": float(mss_state["inbound_speed_abs_per_min"]),
                    "displacement_speed_ratio": float(mss_state["displacement_speed_ratio"]),
                    "mss_leg_bar_count": float(mss_state["mss_leg"]["leg_bar_count"]),
                    "terminal_to_mss_minutes": float(mss_state["mss_leg"]["leg_minutes"]),
                    "mss_outbound_distance_abs": float(mss_state["mss_leg"]["leg_distance_abs"]),
                    "mss_outbound_distance_pct": float(mss_state["mss_leg"]["leg_distance_pct"]),
                    "mss_outbound_speed_abs_per_min": float(mss_state["mss_leg"]["leg_speed_abs_per_min"]),
                    "mss_outbound_speed_pct_per_min": float(mss_state["mss_leg"]["leg_speed_pct_per_min"]),
                    "reversal_path_efficiency": float(mss_state["mss_leg"]["reversal_path_efficiency"]),
                    "directional_body_share": float(mss_state["mss_leg"]["directional_body_share"]),
                    "directional_bar_fraction": float(mss_state["mss_leg"]["directional_bar_fraction"]),
                    "max_directional_body_vs_pre20_median": float(mss_state["mss_leg"]["max_directional_body_vs_pre20_median"]),
                    "max_leg_range_vs_pre20_median": float(mss_state["mss_leg"]["max_leg_range_vs_pre20_median"]),
                    "signal_leg_minutes": float(signal_leg["leg_minutes"]),
                    "signal_leg_distance_pct": float(signal_leg["leg_distance_pct"]),
                    "signal_leg_speed_pct_per_min": float(signal_leg["leg_speed_pct_per_min"]),
                    "stop_price": stop, "target_price": target,
                    "risk_abs": float(risk), "risk_pct": float(risk / entry),
                    "planned_reward_abs": float(reward), "planned_rr": float(reward / risk),
                    "sweep_to_terminal_minutes": float((pd.Timestamp(mss_state["terminal_time"]) - sweep_time).total_seconds() / 60.0),
                    "sweep_to_mss_minutes": float((pd.Timestamp(mss_state["mss_time"]) - sweep_time).total_seconds() / 60.0),
                    "terminal_to_signal_minutes": float((signal_time - pd.Timestamp(mss_state["terminal_time"])).total_seconds() / 60.0),
                    "sweep_to_signal_minutes": float((signal_time - sweep_time).total_seconds() / 60.0),
                    "target_already_touched_before_signal": False,
                    "strict_break_bar_fvg": False,
                })
                emitted = True
                break

            funnel.append({
                "event_id": sweep["event_id"], "ny_date": day_text,
                "trade_side": sweep["trade_side"], "level_type": sweep["level_type"],
                "execution_tf": f"{tf}m", "fresh_sweep": True,
                "mss_found": bool(ever_mss), "fvg_after_mss_or_in_reversal_found": bool(ever_fvg),
                "attempt_emitted": bool(emitted),
            })

        if progress_reporter is not None:
            progress_reporter.step()

    out = pd.DataFrame(attempts)
    if not out.empty:
        out["attempt_id"] = out["event_id"].astype(str) + "|tf=" + out["execution_tf"].astype(str) + "|disp=discovery|r05"
        out = out.sort_values(["signal_time", "attempt_id"], kind="mergesort").reset_index(drop=True)
    return out, pd.DataFrame(funnel)

def build_signal_attempts_v4(
    bars_ny: pd.DataFrame,
    sweeps: pd.DataFrame,
    *,
    config: ICTDisplacementDiscoveryConfig = ICTDisplacementDiscoveryConfig(),
    progress_enabled: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sweeps.empty:
        return pd.DataFrame(), pd.DataFrame()

    reporter = None
    if progress_enabled:
        from ..progress import ProgressReporter
        days = int(sweeps["ny_date"].astype(str).nunique())
        reporter = ProgressReporter(
            label="[research-signals] ICT path MSS/displacement scan",
            total=max(1, days * len(config.execution_timeframes)),
            every=max(1, (days * len(config.execution_timeframes)) // 100),
            enabled=True,
        )

    parts: list[pd.DataFrame] = []
    funnels: list[pd.DataFrame] = []
    try:
        for tf in config.execution_timeframes:
            part, funnel = _build_attempts_one_tf(
                bars_ny,
                sweeps,
                timeframe_minutes=tf,
                pivot_left=config.mss_pivot_left,
                pivot_right=config.mss_pivot_right,
                progress_reporter=reporter,
            )
            if not part.empty:
                parts.append(part)
            if not funnel.empty:
                funnels.append(funnel)
    finally:
        if reporter is not None:
            reporter.close()

    attempts = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    funnel = pd.concat(funnels, ignore_index=True) if funnels else pd.DataFrame()
    return attempts, funnel


def build_causal_audit_v4(attempts: pd.DataFrame) -> pd.DataFrame:
    if attempts.empty:
        return pd.DataFrame([{"check": "attempts_non_empty", "passed": False, "violations": 0, "detail": "no attempts"}])

    rows: list[dict[str, object]] = []

    def add(name: str, mask: pd.Series, detail: str) -> None:
        bad = int((~mask.fillna(False)).sum())
        rows.append({"check": name, "passed": bad == 0, "violations": bad, "detail": detail})

    signal = pd.to_datetime(attempts["signal_time"])
    mss_time = pd.to_datetime(attempts["mss_time"])
    sweep = pd.to_datetime(attempts["sweep_time"])
    ref_avail = pd.to_datetime(attempts["mss_reference_available_time"])
    terminal = pd.to_datetime(attempts["episode_terminal_extreme_time"])
    fvg_avail = pd.to_datetime(attempts["fvg_available_time"])
    bar_start = pd.to_datetime(attempts["signal_bar_start"])
    tf_delta = pd.to_timedelta(pd.to_numeric(attempts["execution_tf_minutes"]), unit="m")
    expected_signal = pd.concat([mss_time.rename("mss"), fvg_avail.rename("fvg")], axis=1).max(axis=1)

    add("mss_after_sweep", mss_time > sweep, "MSS confirmation must be after liquidity sweep")
    add("signal_not_before_mss", signal >= mss_time, "order signal cannot exist before MSS")
    add("terminal_known_by_mss", terminal <= mss_time, "terminal extreme used by MSS must be known by MSS")
    add("reference_confirmed_by_mss", ref_avail <= mss_time, "short-term pivot must be causally confirmed by MSS")
    add("fvg_known_by_signal", fvg_avail <= signal, "selected FVG must be fully formed before order signal")
    add("signal_when_both_known", signal == expected_signal, "signal is max(MSS confirmation, FVG available time)")
    add("closed_execution_bar", signal == bar_start + tf_delta, "signal uses only a completed execution bar")
    add("positive_risk", pd.to_numeric(attempts["risk_abs"], errors="coerce") > 0, "stop beyond entry")
    add("positive_reward", pd.to_numeric(attempts["planned_reward_abs"], errors="coerce") > 0, "target beyond entry")
    return pd.DataFrame(rows)

