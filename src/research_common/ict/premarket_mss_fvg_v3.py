#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ICT-faithful causal sweep -> MSS -> displacement-leg FVG primitives.

R04 deliberately separates three concepts that R02 incorrectly collapsed:

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
class ICTPathConfig:
    execution_timeframes: tuple[int, ...] = (1, 2, 5)
    mss_pivot_left: int = 1
    mss_pivot_right: int = 1
    # Base displacement is deliberately relative, not an optimized numeric body
    # threshold: outbound directional speed must be >= inbound speed.
    require_relative_impulse: bool = True


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


def _build_attempts_one_tf(
    bars_ny: pd.DataFrame,
    sweeps: pd.DataFrame,
    *,
    timeframe_minutes: int,
    pivot_left: int,
    pivot_right: int,
    require_relative_impulse: bool,
    progress_reporter=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sweeps.empty:
        return pd.DataFrame(), pd.DataFrame()

    tf = int(timeframe_minutes)
    attempts: list[dict[str, object]] = []
    funnel: list[dict[str, object]] = []

    for day_text, day_sweeps in sweeps.groupby("ny_date", sort=True):
        day = pd.Timestamp(day_text).date()
        day_1m = slice_ny_day(bars_ny, day, pd.Timestamp("04:00").time(), TRADE_END)
        exec_frame = aggregate_closed_bars(day_1m, tf) if not day_1m.empty else pd.DataFrame()
        pivots = (
            confirmed_pivots_with_excursion(exec_frame, left=pivot_left, right=pivot_right)
            if not exec_frame.empty
            else pd.DataFrame()
        )
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
            target_touched = False
            mss_seen = False
            fvg_seen = False
            displacement_seen = False
            emitted = False

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
                            terminal_source_time = pd.Timestamp(one_idx[scan_j])
                        if np.isfinite(one_highs[scan_j]) and one_highs[scan_j] >= target:
                            target_touched = True
                    else:
                        px = one_highs[scan_j]
                        if np.isfinite(px) and (not np.isfinite(terminal_price) or px > terminal_price):
                            terminal_price = float(px)
                            terminal_time = pd.Timestamp(one_available[scan_j])
                            terminal_source_time = pd.Timestamp(one_idx[scan_j])
                        if np.isfinite(one_lows[scan_j]) and one_lows[scan_j] <= target:
                            target_touched = True

                if target_touched:
                    break
                if pd.isna(terminal_time) or not np.isfinite(terminal_price):
                    continue

                reference = _select_mss_reference(
                    pivots,
                    side=ref_side,
                    sweep_bar_start=pd.Timestamp(sweep["sweep_bar_start"]),
                    terminal_available_time=pd.Timestamp(terminal_time),
                    signal_available_time=signal_time,
                )
                if reference is None:
                    continue
                ref_price = float(reference["pivot_price"])

                mss_break = bool(closes[pos] > ref_price) if is_long else bool(closes[pos] < ref_price)
                if not mss_break:
                    continue
                mss_seen = True

                fvg_pos = _find_latest_fvg_in_leg(
                    is_long=is_long,
                    highs=highs,
                    lows=lows,
                    available_ns=available_ns,
                    terminal_time=pd.Timestamp(terminal_time),
                    signal_pos=pos,
                )
                if fvg_pos is None:
                    # MSS exists but the reversal leg has not delivered a FVG.
                    # Continue only if a later bar still represents the same
                    # first structure break? Once structure is already broken,
                    # the requested entry model is the displacement that caused
                    # the break, so a later unrelated FVG is not retrofitted.
                    break
                fvg_seen = True

                ref_time = pd.Timestamp(reference["pivot_time"])
                inbound_anchor = _select_inbound_anchor(
                    pivots,
                    side=ref_side,
                    terminal_available_time=pd.Timestamp(terminal_time),
                    signal_available_time=signal_time,
                    fallback_time=pd.Timestamp(sweep["sweep_bar_start"]),
                    fallback_price=float(sweep["level_price"]),
                )
                inbound_anchor_time = pd.Timestamp(inbound_anchor["anchor_time"])
                inbound_anchor_price = float(inbound_anchor["anchor_price"])
                inbound_minutes = max(float((pd.Timestamp(terminal_source_time) - inbound_anchor_time).total_seconds() / 60.0), float(tf))
                outbound_minutes = max(float((signal_time - pd.Timestamp(terminal_time)).total_seconds() / 60.0), float(tf))
                inbound_distance = abs(inbound_anchor_price - float(terminal_price))
                outbound_distance = abs(float(closes[pos]) - float(terminal_price))
                inbound_speed = inbound_distance / inbound_minutes if inbound_minutes > 0 else np.nan
                outbound_speed = outbound_distance / outbound_minutes if outbound_minutes > 0 else np.nan
                speed_ratio = outbound_speed / inbound_speed if np.isfinite(inbound_speed) and inbound_speed > EPS else np.nan
                relative_impulse = bool(np.isfinite(speed_ratio) and speed_ratio >= 1.0)
                displacement_seen = relative_impulse
                if require_relative_impulse and not relative_impulse:
                    break

                # Reversal path quality is diagnostic, not a hard extra gate.
                terminal_exec_pos = int(np.searchsorted(available_ns, int(pd.Timestamp(terminal_time).value), side="left"))
                terminal_exec_pos = max(0, min(terminal_exec_pos, pos))
                reversal_closes = np.concatenate(([float(terminal_price)], closes[terminal_exec_pos : pos + 1]))
                reversal_efficiency = _path_efficiency(reversal_closes)

                fp = int(fvg_pos)
                if is_long:
                    entry = float(lows[fp])
                    far = float(highs[fp - 2])
                else:
                    entry = float(highs[fp])
                    far = float(lows[fp - 2])
                stop = float(terminal_price)
                risk = entry - stop if is_long else stop - entry
                reward = target - entry if is_long else entry - target
                if not np.isfinite(risk) or risk <= EPS or not np.isfinite(reward) or reward <= EPS:
                    break

                reference_source = str(reference.get("reference_relation", "unknown"))
                attempts.append(
                    {
                        **sweep,
                        "execution_tf": f"{tf}m",
                        "execution_tf_minutes": tf,
                        "mss_model": "ict_liquidity_then_short_term_structure_break",
                        "mss_reference_side": ref_side,
                        "mss_reference_time": ref_time,
                        "mss_reference_price": ref_price,
                        "mss_reference_available_time": pd.Timestamp(reference["confirmation_available_time"]),
                        "mss_reference_source": reference_source,
                        "episode_terminal_extreme_time": pd.Timestamp(terminal_time),
                        "episode_terminal_extreme_source_time": pd.Timestamp(terminal_source_time),
                        "episode_terminal_extreme_price": stop,
                        "signal_bar_start": pd.Timestamp(idx[pos]),
                        "signal_time": signal_time,
                        "signal_open": float(opens[pos]),
                        "signal_high": float(highs[pos]),
                        "signal_low": float(lows[pos]),
                        "signal_close": float(closes[pos]),
                        "fvg_third_bar_start": pd.Timestamp(idx[fp]),
                        "fvg_available_time": pd.Timestamp(available[fp]),
                        "fvg_near_edge_entry": entry,
                        "fvg_far_edge": far,
                        "fvg_size_abs": abs(entry - far),
                        "fvg_size_pct": abs(entry / far - 1.0) if abs(far) > EPS else np.nan,
                        "fvg_relation_to_mss": "on_mss_bar" if fp == pos else "before_mss_within_displacement_leg",
                        "displacement_model": "relative_leg_speed_ge_inbound",
                        "inbound_anchor_time": inbound_anchor_time,
                        "inbound_anchor_price": inbound_anchor_price,
                        "inbound_anchor_source": str(inbound_anchor["anchor_source"]),
                        "displacement_relative_impulse_pass": relative_impulse,
                        "inbound_minutes": inbound_minutes,
                        "outbound_minutes": outbound_minutes,
                        "inbound_distance_abs": inbound_distance,
                        "outbound_distance_abs": outbound_distance,
                        "inbound_speed_abs_per_min": inbound_speed,
                        "outbound_speed_abs_per_min": outbound_speed,
                        "displacement_speed_ratio": speed_ratio,
                        "reversal_path_efficiency": reversal_efficiency,
                        "stop_price": stop,
                        "target_price": target,
                        "risk_abs": float(risk),
                        "risk_pct": float(risk / entry),
                        "planned_reward_abs": float(reward),
                        "planned_rr": float(reward / risk),
                        "sweep_to_terminal_minutes": float((pd.Timestamp(terminal_time) - sweep_time).total_seconds() / 60.0),
                        "terminal_to_signal_minutes": float((signal_time - pd.Timestamp(terminal_time)).total_seconds() / 60.0),
                        "sweep_to_signal_minutes": float((signal_time - sweep_time).total_seconds() / 60.0),
                        "target_already_touched_before_signal": False,
                        "strict_break_bar_fvg": False,
                    }
                )
                emitted = True
                break

            funnel.append(
                {
                    "event_id": sweep["event_id"],
                    "ny_date": day_text,
                    "trade_side": sweep["trade_side"],
                    "level_type": sweep["level_type"],
                    "execution_tf": f"{tf}m",
                    "fresh_sweep": True,
                    "mss_found": bool(mss_seen),
                    "fvg_in_displacement_leg_found": bool(fvg_seen),
                    "relative_impulse_pass": bool(displacement_seen),
                    "attempt_emitted": bool(emitted),
                }
            )

        if progress_reporter is not None:
            progress_reporter.step()

    out = pd.DataFrame(attempts)
    if not out.empty:
        out["attempt_id"] = (
            out["event_id"].astype(str)
            + "|tf=" + out["execution_tf"].astype(str)
            + "|disp=relative_leg|r04"
        )
        out = out.sort_values(["signal_time", "attempt_id"], kind="mergesort").reset_index(drop=True)
    return out, pd.DataFrame(funnel)


def build_signal_attempts_v3(
    bars_ny: pd.DataFrame,
    sweeps: pd.DataFrame,
    *,
    config: ICTPathConfig = ICTPathConfig(),
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
                require_relative_impulse=config.require_relative_impulse,
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


def build_causal_audit_v3(attempts: pd.DataFrame) -> pd.DataFrame:
    if attempts.empty:
        return pd.DataFrame([{"check": "attempts_non_empty", "passed": False, "violations": 0, "detail": "no attempts"}])

    rows: list[dict[str, object]] = []
    def add(name: str, mask: pd.Series, detail: str) -> None:
        bad = int((~mask.fillna(False)).sum())
        rows.append({"check": name, "passed": bad == 0, "violations": bad, "detail": detail})

    signal = pd.to_datetime(attempts["signal_time"])
    sweep = pd.to_datetime(attempts["sweep_time"])
    ref_avail = pd.to_datetime(attempts["mss_reference_available_time"])
    terminal = pd.to_datetime(attempts["episode_terminal_extreme_time"])
    fvg_avail = pd.to_datetime(attempts["fvg_available_time"])
    bar_start = pd.to_datetime(attempts["signal_bar_start"])
    tf_delta = pd.to_timedelta(pd.to_numeric(attempts["execution_tf_minutes"]), unit="m")

    add("signal_after_sweep", signal > sweep, "MSS signal must be after liquidity sweep")
    add("terminal_known_by_signal", terminal <= signal, "terminal extreme must be known by signal")
    add("reference_confirmed_by_signal", ref_avail <= signal, "short-term pivot must be causally confirmed")
    add("fvg_known_by_signal", fvg_avail <= signal, "selected FVG must already exist by MSS confirmation")
    add("closed_execution_bar", signal == bar_start + tf_delta, "MSS uses only closed execution bars")
    add("relative_displacement", attempts["displacement_relative_impulse_pass"].astype(bool), "base path model requires reversal speed >= inbound speed")
    add("positive_risk", pd.to_numeric(attempts["risk_abs"], errors="coerce") > 0, "stop beyond entry")
    return pd.DataFrame(rows)
