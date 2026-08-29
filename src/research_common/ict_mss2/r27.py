#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R27 ordered, causal completed-trend sweep reversal state machine.

The module intentionally exposes one frozen path definition rather than a
parameter-search surface.  Root rows contain future path labels for analysis,
but state detection reads only root-time geometry and closed 1m bars.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.core import normalize_1m_bars

EPS = 1e-12
STATES = (
    (0, "S0_sweep"),
    (1, "S1_rejected_reclaimed"),
    (2, "S2_new_structure"),
    (3, "S3_meaningful_mss"),
    (4, "S4_displacement"),
    (5, "S5_fvg_retracement"),
    (6, "S6_protected_reversal"),
)


@dataclass(frozen=True)
class R27Config:
    discovery_end: pd.Timestamp = pd.Timestamp("2024-12-31 23:59:59")
    validation_start: pd.Timestamp = pd.Timestamp("2025-01-01 00:00:00")
    validation_end: pd.Timestamp = pd.Timestamp("2025-06-30 23:59:59")
    embargo_end: pd.Timestamp = pd.Timestamp("2025-07-31 23:59:59")
    holdout_start: pd.Timestamp = pd.Timestamp("2025-08-01 00:00:00")
    atr_minutes: int = 60
    reclaim_minutes: int = 30
    max_outside_consecutive: int = 2
    state_horizon_minutes: int = 360
    fvg_wait_minutes: int = 180
    protected_wait_minutes: int = 180
    outcome_horizon_minutes: int = 43_200
    pivot_left: int = 2
    pivot_right: int = 2
    stop_buffer_atr: float = 0.10
    s2_impulse_atr: float = 0.75
    s2_retained_share: float = 0.382
    s2_clearance_atr: float = 0.10
    s3_close_through_atr: float = 0.05
    s4_displacement_atr: float = 1.00
    s4_body_ratio: float = 0.60
    s4_close_through_atr: float = 0.10
    s4_path_efficiency: float = 0.65
    s4_extra_bars: int = 2
    s5_fvg_width_atr: float = 0.10
    protected_stop_buffer_atr: float = 0.10
    market_roundtrip_cost: float = 0.0011
    limit_roundtrip_cost: float = 0.0008
    cost_scales: tuple[float, ...] = (1.0, 2.0, 3.0)

    def validate(self) -> "R27Config":
        if not (self.discovery_end < self.validation_start <= self.validation_end < self.holdout_start):
            raise ValueError("invalid discovery/validation/holdout order")
        if self.embargo_end >= self.holdout_start or self.embargo_end < self.validation_end:
            raise ValueError("embargo must end immediately before holdout")
        positive = (
            self.atr_minutes, self.reclaim_minutes, self.state_horizon_minutes,
            self.fvg_wait_minutes, self.protected_wait_minutes, self.outcome_horizon_minutes, self.pivot_left,
            self.pivot_right,
        )
        if any(int(x) <= 0 for x in positive):
            raise ValueError("all time and pivot windows must be positive")
        if self.max_outside_consecutive < 0 or self.s4_extra_bars < 0:
            raise ValueError("bar counts cannot be negative")
        if any(float(x) <= 0 for x in self.cost_scales):
            raise ValueError("cost scales must be positive")
        return self


ROOT_COLUMNS = (
    "root_event_id", "root_sweep_time", "root_sweep_available_time", "root_side",
    "root_zone_low", "root_zone_high", "root_bar_open", "root_bar_high",
    "root_bar_low", "root_bar_close", "opposite_1_touch_price",
    "direct_reversal_label", "comparison_class", "path_outcome",
    "root_max_swing_tf_min", "root_min_swing_tf_min", "root_lt_count",
    "root_level_count", "root_region_count", "root_oldest_age_days",
    "root_newest_age_days", "root_known_context_count",
    "root_max_known_trend_tf_min", "root_max_known_trend_move_pct",
    "root_native_context_any", "root_nested_context_any", "root_trend_ge5_any",
    "root_trend_ge7_any", "root_sweep_depth_bps", "root_rejection_wick_share",
    "root_reversal_close_location", "root_same_bar_full_reclaim_flag",
    "root_bar_range_pct", "pre_sweep_ret_5m", "pre_sweep_ret_15m",
    "pre_sweep_ret_60m",
)


def prepare_root_universe(
    rows: pd.DataFrame,
    *,
    split: str,
    config: R27Config | None = None,
) -> pd.DataFrame:
    """Select one physical split and discard nonessential R13 future fields."""
    cfg = (config or R27Config()).validate()
    if split not in {"discovery", "validation"}:
        raise ValueError("split must be discovery or validation")
    missing = [c for c in ROOT_COLUMNS[:12] if c not in rows]
    if missing:
        raise ValueError(f"root source missing columns: {missing}")
    keep = [c for c in ROOT_COLUMNS if c in rows]
    q = rows.loc[:, keep].copy()
    q["root_sweep_time"] = pd.to_datetime(q["root_sweep_time"], errors="coerce")
    q["root_sweep_available_time"] = pd.to_datetime(q["root_sweep_available_time"], errors="coerce")
    if split == "discovery":
        mask = q["root_sweep_time"].between(pd.Timestamp("2023-01-01"), cfg.discovery_end)
    else:
        mask = q["root_sweep_time"].between(cfg.validation_start, cfg.validation_end)
    q = q.loc[mask & q["root_side"].isin(["SSL", "BSL"])].copy()
    q["research_split"] = split
    q["year"] = q["root_sweep_time"].dt.year.astype(int)
    return q.sort_values(["root_sweep_time", "root_side"], kind="stable").reset_index(drop=True)


def _atr_before_root(bars: pd.DataFrame, pos: int, minutes: int) -> float:
    start = max(1, pos - int(minutes))
    if pos - start < max(10, int(minutes) // 2):
        return np.nan
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    ix = np.arange(start, pos)
    tr = np.maximum(high[ix] - low[ix], np.maximum(abs(high[ix] - close[ix - 1]), abs(low[ix] - close[ix - 1])))
    return float(np.nanmean(tr)) if len(tr) else np.nan


def _pivots(high: np.ndarray, low: np.ndarray, start: int, end: int, left: int, right: int) -> tuple[list[int], list[int]]:
    ph: list[int] = []
    pl: list[int] = []
    for i in range(max(start, left), min(end, len(high) - right - 1) + 1):
        if high[i] > np.max(high[i-left:i]) + EPS and high[i] > np.max(high[i+1:i+right+1]) + EPS:
            ph.append(i)
        if low[i] < np.min(low[i-left:i]) - EPS and low[i] < np.min(low[i+1:i+right+1]) - EPS:
            pl.append(i)
    return ph, pl


def _excursions(high: np.ndarray, low: np.ndarray, entry: float, direction: int) -> tuple[float, float]:
    if not len(high) or not np.isfinite(entry) or entry <= EPS:
        return np.nan, np.nan
    if direction == 1:
        return float(np.nanmax(high) / entry - 1.0), float(1.0 - np.nanmin(low) / entry)
    return float(entry / np.nanmin(low) - 1.0), float(np.nanmax(high) / entry - 1.0)


def _path_efficiency(close: np.ndarray, start: int, end: int, anchor: float, direction: int) -> float:
    if end < start:
        return np.nan
    vals = np.r_[float(anchor), close[start:end+1]]
    travel = float(np.abs(np.diff(vals)).sum())
    net = float(direction * (vals[-1] - vals[0]))
    return net / travel if travel > EPS else np.nan


def _first_passage(
    high: np.ndarray,
    low: np.ndarray,
    index: pd.DatetimeIndex,
    *,
    start: int,
    end: int,
    entry: float,
    target: float,
    stop: float,
    direction: int,
    target_credit_start: int | None = None,
) -> dict[str, object]:
    end = min(end, len(index) - 1)
    credit = start if target_credit_start is None else int(target_credit_start)
    for i in range(start, end + 1):
        if direction == 1:
            stop_hit = low[i] <= stop + EPS
            target_hit = i >= credit and high[i] >= target - EPS
        else:
            stop_hit = high[i] >= stop - EPS
            target_hit = i >= credit and low[i] <= target + EPS
        if stop_hit:
            return {"outcome": "sl_first", "exit_pos": i, "exit_time": index[i], "exit_price": stop}
        if target_hit:
            return {"outcome": "tp_first", "exit_pos": i, "exit_time": index[i], "exit_price": target}
    return {"outcome": "censored", "exit_pos": end, "exit_time": index[end], "exit_price": np.nan}


def _entry_result(
    base: dict[str, object],
    *,
    high: np.ndarray,
    low: np.ndarray,
    index: pd.DatetimeIndex,
    entry_pos: int,
    entry_price: float,
    direction: int,
    target: float,
    stop: float,
    end: int,
    cost: float,
    target_credit_start: int | None = None,
) -> dict[str, object]:
    row = dict(base)
    row.update({"entry_time": index[entry_pos], "entry_price": entry_price, "target_price": target, "stop_price": stop})
    if direction * (target - entry_price) <= EPS or direction * (entry_price - stop) <= EPS:
        row.update({"entry_status": "invalid_geometry", "outcome": "not_traded"})
        return row
    fp = _first_passage(high, low, index, start=entry_pos, end=end, entry=entry_price, target=target, stop=stop, direction=direction, target_credit_start=target_credit_start)
    row.update(fp)
    exit_pos = int(fp["exit_pos"])
    seg_hi, seg_lo = high[entry_pos:exit_pos+1], low[entry_pos:exit_pos+1]
    mfe, mae = _excursions(seg_hi, seg_lo, entry_price, direction)
    risk = direction * (entry_price - stop) / entry_price
    reward = direction * (target - entry_price) / entry_price
    gross = reward if fp["outcome"] == "tp_first" else (-risk if fp["outcome"] == "sl_first" else np.nan)
    row.update({
        "entry_status": "filled", "mfe_pct": mfe, "mae_pct": mae,
        "structural_risk_pct": risk, "structural_reward_pct": reward,
        "structural_rr": reward / risk if risk > EPS else np.nan,
        "gross_return": gross, "roundtrip_cost": cost,
    })
    for scale in (1.0, 2.0, 3.0):
        row[f"net_return_cost{scale:g}x"] = gross - cost * scale if np.isfinite(gross) else np.nan
    return row


def _detect_states_for_root(
    bars: pd.DataFrame,
    root: object,
    *,
    cfg: R27Config,
    physical_end: pd.Timestamp,
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    idx = bars.index
    op = bars["open"].to_numpy(float); high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float); close = bars["close"].to_numpy(float)
    root_time = pd.Timestamp(root.root_sweep_time)
    root_pos = int(idx.searchsorted(root_time, side="left"))
    if root_pos >= len(idx) or idx[root_pos] != root_time:
        return {}, {"engine_error": "missing_root_bar"}
    direction = 1 if str(root.root_side) == "SSL" else -1
    atr = _atr_before_root(bars, root_pos, cfg.atr_minutes)
    if not np.isfinite(atr) or atr <= EPS:
        return {}, {"engine_error": "invalid_root_atr"}
    sweep_extreme = float(root.root_bar_low if direction == 1 else root.root_bar_high)
    reclaim_level = float(root.root_zone_high if direction == 1 else root.root_zone_low)
    horizon_end = min(len(idx) - 1, root_pos + cfg.state_horizon_minutes, int(idx.searchsorted(physical_end, side="right")) - 1)
    states: dict[int, dict[str, object]] = {
        0: {"available_time": pd.Timestamp(root.root_sweep_available_time), "signal_pos": root_pos, "quality_reason": "root_sweep_closed"}
    }
    diag: dict[str, object] = {
        "engine_error": "", "root_atr": atr, "direction": direction,
        "sweep_extreme": sweep_extreme, "reclaim_level": reclaim_level,
    }

    # S1: strict reclaim before three consecutive outside closes.
    consec = 0; max_consec = 0; outside = 0; max_outside_depth = 0.0; s1 = -1
    reclaim_end = min(horizon_end, root_pos + cfg.reclaim_minutes)
    for i in range(root_pos + 1, reclaim_end + 1):
        outside_now = close[i] <= reclaim_level + EPS if direction == 1 else close[i] >= reclaim_level - EPS
        if outside_now:
            outside += 1; consec += 1; max_consec = max(max_consec, consec)
            max_outside_depth = max(max_outside_depth, direction * (reclaim_level - close[i]) / atr)
            if consec > cfg.max_outside_consecutive:
                break
        else:
            s1 = i; break
    diag.update({
        "s1_outside_closes": outside, "s1_max_consecutive_outside": max_consec,
        "s1_outside_close_share": outside / max(1, (s1 if s1 >= 0 else reclaim_end) - root_pos),
        "s1_max_outside_depth_atr": max_outside_depth,
        "s1_failed_acceptance": int(max_consec > cfg.max_outside_consecutive),
    })
    if s1 < 0:
        return states, diag
    rng = high[s1] - low[s1]
    body_dir = direction * (close[s1] - op[s1])
    states[1] = {
        "available_time": idx[s1] + pd.Timedelta(minutes=1), "signal_pos": s1,
        "reclaim_delay_min": (idx[s1] - root_time).total_seconds() / 60.0 + 1.0,
        "reclaim_close_penetration_atr": direction * (close[s1] - reclaim_level) / atr,
        "reclaim_body_ratio": body_dir / rng if rng > EPS else np.nan,
        "outside_closes": outside, "max_consecutive_outside": max_consec,
        "outside_close_share": diag["s1_outside_close_share"],
        "max_outside_depth_atr": max_outside_depth,
        "reclaim_retention_15m": float(np.mean(direction * (close[s1+1:min(horizon_end, s1+15)+1] - reclaim_level) > 0)) if min(horizon_end, s1+15) >= s1+1 else np.nan,
        "reclaim_retention_30m": float(np.mean(direction * (close[s1+1:min(horizon_end, s1+30)+1] - reclaim_level) > 0)) if min(horizon_end, s1+30) >= s1+1 else np.nan,
        "quality_reason": "reclaim_before_failed_acceptance",
    }

    # S2: post-reclaim reversal impulse pivot followed by retained pullback.
    ph, pl = _pivots(high, low, s1, horizon_end, cfg.pivot_left, cfg.pivot_right)
    impulse_candidates = ph if direction == 1 else pl
    pullback_candidates = pl if direction == 1 else ph
    impulse_pos = pullback_pos = -1; impulse_price = pullback_price = np.nan
    impulse_atr = retention = clearance = pullback_depth = np.nan
    for ip in impulse_candidates:
        if ip < s1:
            continue
        iprice = high[ip] if direction == 1 else low[ip]
        imp = direction * (iprice - sweep_extreme) / atr
        if imp + EPS < cfg.s2_impulse_atr:
            continue
        for pp in pullback_candidates:
            if pp <= ip:
                continue
            pprice = low[pp] if direction == 1 else high[pp]
            retained = direction * (pprice - sweep_extreme) / max(direction * (iprice - sweep_extreme), EPS)
            clear = direction * (pprice - sweep_extreme) / atr
            if retained + EPS >= cfg.s2_retained_share and clear + EPS >= cfg.s2_clearance_atr:
                impulse_pos, pullback_pos = ip, pp
                impulse_price, pullback_price = float(iprice), float(pprice)
                impulse_atr, retention, clearance = float(imp), float(retained), float(clear)
                pullback_depth = direction * (iprice - pprice) / max(direction * (iprice - sweep_extreme), EPS)
                break
        if pullback_pos >= 0:
            break
    if pullback_pos < 0:
        return states, diag
    s2_available = idx[pullback_pos + cfg.pivot_right] + pd.Timedelta(minutes=1)
    s2_av_pos = int(idx.searchsorted(s2_available, side="left"))
    pre2_hi = high[root_pos+1:min(s2_av_pos, len(idx)-1)+1]
    pre2_lo = low[root_pos+1:min(s2_av_pos, len(idx)-1)+1]
    pre2_mfe, pre2_mae = _excursions(pre2_hi, pre2_lo, float(op[root_pos+1]), direction)
    states[2] = {
        "available_time": s2_available, "signal_pos": pullback_pos + cfg.pivot_right,
        "impulse_pivot_time": idx[impulse_pos], "impulse_pivot_price": impulse_price,
        "pullback_pivot_time": idx[pullback_pos], "pullback_pivot_price": pullback_price,
        "impulse_atr": impulse_atr, "pullback_retained_share": retention,
        "pullback_depth_share": pullback_depth, "extreme_clearance_atr": clearance,
        "formation_delay_min": (s2_available - pd.Timestamp(root.root_sweep_available_time)).total_seconds() / 60.0,
        "pre_state_mfe_pct": pre2_mfe, "pre_state_mae_pct": pre2_mae,
        "quality_reason": "impulse_then_retained_pullback",
    }

    # S3: completed close through the meaningful S2 impulse pivot.
    s3 = -1
    for i in range(max(s2_av_pos, pullback_pos + 1), horizon_end + 1):
        if direction * (close[i] - impulse_price) / atr + EPS >= cfg.s3_close_through_atr:
            s3 = i; break
    if s3 < 0:
        return states, diag
    pre3_hi, pre3_lo = high[pullback_pos:s3+1], low[pullback_pos:s3+1]
    pre3_mfe, pre3_mae = _excursions(pre3_hi, pre3_lo, pullback_price, direction)
    rng3 = high[s3] - low[s3]
    body3 = direction * (close[s3] - op[s3])
    states[3] = {
        "available_time": idx[s3] + pd.Timedelta(minutes=1), "signal_pos": s3,
        "mss_level": impulse_price,
        "mss_break_distance_atr": direction * (close[s3] - impulse_price) / atr,
        "mss_directional_body_atr": body3 / atr,
        "mss_body_ratio": body3 / rng3 if rng3 > EPS else np.nan,
        "mss_delay_min": (idx[s3] + pd.Timedelta(minutes=1) - root_time).total_seconds() / 60.0,
        "pre_mss_mfe_pct": pre3_mfe, "pre_mss_mae_pct": pre3_mae,
        "mss_path_efficiency": _path_efficiency(close, pullback_pos, s3, pullback_price, direction),
        "quality_reason": "close_breaks_s2_impulse_pivot",
    }

    # S4: one canonical strong displacement confirmation.
    s4 = -1; s4_eff = np.nan
    for i in range(s3, min(horizon_end, s3 + cfg.s4_extra_bars) + 1):
        rng4 = high[i] - low[i]
        body4 = direction * (close[i] - op[i])
        disp = direction * (close[i] - pullback_price) / atr
        through = direction * (close[i] - impulse_price) / atr
        eff = _path_efficiency(close, pullback_pos, i, pullback_price, direction)
        ratio = body4 / rng4 if rng4 > EPS else np.nan
        if disp + EPS >= cfg.s4_displacement_atr and ratio + EPS >= cfg.s4_body_ratio and through + EPS >= cfg.s4_close_through_atr and eff + EPS >= cfg.s4_path_efficiency:
            s4 = i; s4_eff = eff; break
    if s4 < 0:
        return states, diag
    s4_rng = high[s4] - low[s4]
    s4_body = direction * (close[s4] - op[s4])
    displacement_extreme = float(np.max(high[s3:s4+1]) if direction == 1 else np.min(low[s3:s4+1]))
    states[4] = {
        "available_time": idx[s4] + pd.Timedelta(minutes=1), "signal_pos": s4,
        "displacement_atr": direction * (close[s4] - pullback_price) / atr,
        "displacement_body_ratio": s4_body / s4_rng if s4_rng > EPS else np.nan,
        "displacement_close_through_atr": direction * (close[s4] - impulse_price) / atr,
        "displacement_path_efficiency": s4_eff,
        "displacement_extreme": displacement_extreme,
        "pre_displacement_rethreat_clearance_atr": (
            (float(np.min(low[s1:s4+1])) - sweep_extreme) / atr
            if direction == 1 else (sweep_extreme - float(np.max(high[s1:s4+1]))) / atr
        ),
        "quality_reason": "frozen_strong_displacement_gate",
    }

    # S5: first qualifying displacement-linked FVG and causal proximal fill.
    fvg_pos = -1; fvg_low = fvg_high = np.nan
    for i in range(max(2, s3), s4 + 1):
        if direction == 1 and low[i] > high[i-2] + EPS:
            lo, hi = float(high[i-2]), float(low[i])
        elif direction == -1 and high[i] < low[i-2] - EPS:
            lo, hi = float(high[i]), float(low[i-2])
        else:
            continue
        mid = (lo + hi) / 2.0
        if (hi - lo) / atr + EPS >= cfg.s5_fvg_width_atr and direction * (mid - impulse_price) > EPS:
            fvg_pos, fvg_low, fvg_high = i, lo, hi; break
    if fvg_pos < 0:
        return states, diag
    # The FVG may form on S3 while the canonical displacement gate is confirmed
    # one or two bars later.  S5 cannot arm before both facts are available.
    order_signal_pos = max(fvg_pos, s4)
    order_available = idx[order_signal_pos] + pd.Timedelta(minutes=1)
    order_pos = int(idx.searchsorted(order_available, side="left"))
    proximal = fvg_high if direction == 1 else fvg_low
    target = float(root.opposite_1_touch_price)
    stop = sweep_extreme - direction * cfg.stop_buffer_atr * atr
    wait_end = min(horizon_end, order_pos + cfg.fvg_wait_minutes - 1)
    fill_pos = -1; cancel_reason = "unfilled_expired"
    for i in range(order_pos, wait_end + 1):
        stop_hit = low[i] <= stop + EPS if direction == 1 else high[i] >= stop - EPS
        target_hit = high[i] >= target - EPS if direction == 1 else low[i] <= target + EPS
        fill_hit = low[i] <= proximal + EPS if direction == 1 else high[i] >= proximal - EPS
        # Once the bar reaches the limit, the order is filled.  If that same
        # OHLC bar also spans the stop, downstream first passage assigns the
        # pessimistic stop; target credit starts only on the next bar.
        if fill_hit:
            fill_pos = i; cancel_reason = ""; break
        if stop_hit:
            cancel_reason = "stop_before_fill"; break
        if target_hit:
            cancel_reason = "target_before_fill"; break
    diag.update({
        "s5_order_available_time": order_available, "s5_order_status": "filled" if fill_pos >= 0 else cancel_reason,
        "s5_fvg_low": fvg_low, "s5_fvg_high": fvg_high, "s5_proximal": proximal,
    })
    if fill_pos < 0:
        return states, diag
    width = fvg_high - fvg_low
    if direction == 1:
        deepest = max(fvg_low, min(fvg_high, low[fill_pos])); depth_share = (fvg_high - deepest) / width
        rethreat = (float(np.min(low[s4:fill_pos+1])) - sweep_extreme) / atr
    else:
        deepest = min(fvg_high, max(fvg_low, high[fill_pos])); depth_share = (deepest - fvg_low) / width
        rethreat = (sweep_extreme - float(np.max(high[s4:fill_pos+1]))) / atr
    states[5] = {
        "available_time": order_available, "signal_pos": order_signal_pos,
        "entry_time": idx[fill_pos], "entry_pos": fill_pos, "entry_price": proximal,
        "order_type": "limit", "fvg_formed_time": idx[fvg_pos],
        "fvg_low": fvg_low, "fvg_high": fvg_high, "fvg_width_atr": width / atr,
        "fvg_mid_beyond_mss_atr": direction * (((fvg_low + fvg_high) / 2.0) - impulse_price) / atr,
        "fill_delay_min": (idx[fill_pos] - order_available).total_seconds() / 60.0,
        "fill_depth_share": depth_share, "sweep_rethreat_clearance_atr": rethreat,
        "quality_reason": "displacement_fvg_proximal_fill",
    }

    # S6: a new causal pullback pivot becomes protected after S4 extreme break.
    protected_end = min(len(idx) - 1, fill_pos + cfg.protected_wait_minutes)
    ph2, pl2 = _pivots(high, low, fill_pos, protected_end, cfg.pivot_left, cfg.pivot_right)
    candidates = pl2 if direction == 1 else ph2
    for pp in candidates:
        pprice = float(low[pp] if direction == 1 else high[pp])
        if direction * (pprice - sweep_extreme) <= EPS:
            continue
        pav = idx[pp + cfg.pivot_right] + pd.Timedelta(minutes=1)
        j0 = int(idx.searchsorted(pav, side="left"))
        for j in range(j0, protected_end + 1):
            if direction * (close[j] - displacement_extreme) > EPS:
                states[6] = {
                    "available_time": idx[j] + pd.Timedelta(minutes=1), "signal_pos": j,
                    "protected_pivot_time": idx[pp], "protected_pivot_price": pprice,
                    "protected_clearance_atr": direction * (pprice - sweep_extreme) / atr,
                    "protected_confirmation_delay_min": (idx[j] + pd.Timedelta(minutes=1) - idx[fill_pos]).total_seconds() / 60.0,
                    "protected_stop_price": pprice - direction * cfg.protected_stop_buffer_atr * atr,
                    "quality_reason": "pullback_pivot_then_displacement_extreme_break",
                }
                return states, diag
    return states, diag


def build_sequential_state_rows(
    bars_1m: pd.DataFrame,
    roots: pd.DataFrame,
    *,
    physical_end: pd.Timestamp,
    config: R27Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one row per root/state plus one diagnostic row per root."""
    cfg = (config or R27Config()).validate()
    bars = normalize_1m_bars(bars_1m)
    bars = bars.loc[bars.index <= pd.Timestamp(physical_end).floor("min")]
    if bars.empty or roots.empty:
        return pd.DataFrame(), pd.DataFrame()
    high = bars["high"].to_numpy(float); low = bars["low"].to_numpy(float); idx = bars.index
    records: list[dict[str, object]] = []; diagnostics: list[dict[str, object]] = []
    for root in roots.itertuples(index=False):
        states, diag = _detect_states_for_root(bars, root, cfg=cfg, physical_end=physical_end)
        root_values = root._asdict()
        root_quality = {
            key: value for key, value in root_values.items()
            if key.startswith("root_") and key not in {
                "root_event_id", "root_sweep_time", "root_sweep_available_time", "root_side"
            }
        }
        common = {
            "root_event_id": root.root_event_id, "root_sweep_time": pd.Timestamp(root.root_sweep_time),
            "root_side": root.root_side, "research_split": root.research_split, "year": int(root.year),
            "direct_reversal_label": int(root.direct_reversal_label) if pd.notna(root.direct_reversal_label) else np.nan,
            "comparison_class": getattr(root, "comparison_class", ""), "path_outcome": getattr(root, "path_outcome", ""),
            **root_quality,
        }
        diagnostics.append({**common, **diag, "highest_state_reached": max(states) if states else -1})
        direction = int(diag.get("direction", 1)); atr = float(diag.get("root_atr", np.nan))
        sweep_extreme = float(diag.get("sweep_extreme", np.nan)); target = float(root.opposite_1_touch_price)
        stop = sweep_extreme - direction * cfg.stop_buffer_atr * atr
        root_pos = int(idx.searchsorted(pd.Timestamp(root.root_sweep_time), side="left"))
        eval_end = min(len(idx) - 1, root_pos + cfg.outcome_horizon_minutes)
        for state_id, state_name in STATES:
            base = {**common, "state_id": state_id, "state": state_name, "state_reached": int(state_id in states)}
            if state_id not in states:
                base.update({"entry_status": "state_not_reached", "outcome": "not_traded"})
                records.append(base); continue
            state = dict(states[state_id]); base.update(state)
            available = pd.Timestamp(state["available_time"])
            if state_id == 5:
                entry_pos = int(state["entry_pos"]); entry_price = float(state["entry_price"]); cost = cfg.limit_roundtrip_cost
                credit = entry_pos + 1
            else:
                entry_pos = int(idx.searchsorted(available, side="left")); cost = cfg.market_roundtrip_cost; credit = None
                if entry_pos >= len(idx):
                    base.update({"entry_status": "censored_before_entry", "outcome": "not_traded"}); records.append(base); continue
                entry_price = float(bars.iloc[entry_pos]["open"])
            # A state that became available after either barrier is stale.
            if state_id > 0:
                prior = _first_passage(high, low, idx, start=root_pos + 1, end=max(root_pos, entry_pos - 1), entry=float(bars.iloc[root_pos+1]["open"]), target=target, stop=stop, direction=direction)
                if prior["outcome"] in {"tp_first", "sl_first"}:
                    base.update({"entry_time": idx[entry_pos], "entry_price": entry_price, "target_price": target, "stop_price": stop, "entry_status": "stale_before_entry", "outcome": "not_traded", "stale_barrier": prior["outcome"]})
                    records.append(base); continue
            result = _entry_result(base, high=high, low=low, index=idx, entry_pos=entry_pos, entry_price=entry_price, direction=direction, target=target, stop=stop, end=eval_end, cost=cost, target_credit_start=credit)
            if state_id == 6 and result.get("entry_status") == "filled":
                protected_stop = float(state["protected_stop_price"])
                if direction * (entry_price - protected_stop) <= EPS:
                    result["protected_stop_status"] = "invalid_geometry"
                else:
                    pfp = _first_passage(
                        high, low, idx, start=entry_pos, end=eval_end,
                        entry=entry_price, target=target, stop=protected_stop,
                        direction=direction,
                    )
                    prisk = direction * (entry_price - protected_stop) / entry_price
                    preward = direction * (target - entry_price) / entry_price
                    pgross = preward if pfp["outcome"] == "tp_first" else (-prisk if pfp["outcome"] == "sl_first" else np.nan)
                    result.update({
                        "protected_stop_status": "filled", "protected_stop_outcome": pfp["outcome"],
                        "protected_stop_exit_time": pfp["exit_time"], "protected_stop_exit_price": pfp["exit_price"],
                        "protected_stop_risk_pct": prisk,
                        "protected_stop_rr": preward / prisk if prisk > EPS else np.nan,
                        "protected_stop_gross_return": pgross,
                    })
                    for scale in (1.0, 2.0, 3.0):
                        result[f"protected_stop_net_return_cost{scale:g}x"] = pgross - cost * scale if np.isfinite(pgross) else np.nan
            records.append(result)
    return pd.DataFrame(records), pd.DataFrame(diagnostics)


def _profit_factor(values: Iterable[float]) -> float:
    q = pd.Series(values, dtype=float).dropna()
    wins = float(q[q > 0].sum()); losses = float(-q[q < 0].sum())
    return wins / losses if losses > EPS else (np.inf if wins > EPS else np.nan)


def _summarize_group(part: pd.DataFrame, roots: int, prior_reached: int) -> dict[str, object]:
    reached = part.loc[part["state_reached"].eq(1)]
    filled = reached.loc[reached["entry_status"].eq("filled")]
    row: dict[str, object] = {
        "eligible_roots": int(roots), "reached": len(reached),
        "root_reach_rate": len(reached) / roots if roots else np.nan,
        "conditional_reach_rate": len(reached) / prior_reached if prior_reached else np.nan,
        "direct_delivery_probability": pd.to_numeric(reached.get("direct_reversal_label"), errors="coerce").mean(),
        "filled": len(filled), "invalid_geometry": int(reached["entry_status"].eq("invalid_geometry").sum()),
        "stale_before_entry": int(reached["entry_status"].eq("stale_before_entry").sum()),
        "censored": int(filled["outcome"].eq("censored").sum()),
        "tp_first": int(filled["outcome"].eq("tp_first").sum()), "sl_first": int(filled["outcome"].eq("sl_first").sum()),
        "tp_before_sl_rate": filled["outcome"].eq("tp_first").mean() if len(filled) else np.nan,
        "median_mae_pct": pd.to_numeric(filled.get("mae_pct"), errors="coerce").median(),
        "median_mfe_pct": pd.to_numeric(filled.get("mfe_pct"), errors="coerce").median(),
        "median_structural_risk_pct": pd.to_numeric(filled.get("structural_risk_pct"), errors="coerce").median(),
        "median_structural_rr": pd.to_numeric(filled.get("structural_rr"), errors="coerce").median(),
    }
    for label in ("gross", "cost1x", "cost2x", "cost3x"):
        col = "gross_return" if label == "gross" else f"net_return_{label}"
        vals = pd.to_numeric(filled.get(col), errors="coerce").dropna()
        row[f"expectancy_{label}"] = vals.mean() if len(vals) else np.nan
        row[f"pf_{label}"] = _profit_factor(vals)
    v2 = pd.to_numeric(filled.get("net_return_cost2x"), errors="coerce")
    winners = filled.loc[v2.gt(0)].sort_values("net_return_cost2x", ascending=False, kind="stable")
    for n in (5, 10):
        reduced = filled.drop(index=winners.head(n).index)
        vals = pd.to_numeric(reduced.get("net_return_cost2x"), errors="coerce").dropna()
        row[f"expectancy_cost2x_remove_top{n}"] = vals.mean() if len(vals) else np.nan
        row[f"pf_cost2x_remove_top{n}"] = _profit_factor(vals)
    return row


def summarize_state_progression(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    out: list[dict[str, object]] = []
    group_specs = [("overall", ["research_split", "root_side"]), ("year", ["research_split", "root_side", "year"])]
    for grain, cols in group_specs:
        for key, part in rows.groupby(cols, sort=True, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            meta = dict(zip(cols, key)); roots = int(part["root_event_id"].nunique()); prior = roots
            for state_id, state_name in STATES:
                sp = part.loc[part["state_id"].eq(state_id)]
                summary = _summarize_group(sp, roots, prior)
                out.append({"grain": grain, **meta, "state_id": state_id, "state": state_name, **summary})
                prior = int(summary["reached"])
    return pd.DataFrame(out)


def causal_audit(rows: pd.DataFrame, diagnostics: pd.DataFrame, *, config: R27Config | None = None) -> pd.DataFrame:
    cfg = (config or R27Config()).validate(); checks: list[dict[str, object]] = []
    def add(name: str, violations: int) -> None:
        checks.append({"check": name, "violations": int(violations)})
    if rows.empty:
        return pd.DataFrame([{"check": "nonempty_state_rows", "violations": 1}])
    reached = rows.loc[rows["state_reached"].eq(1)].copy()
    av = pd.to_datetime(reached["available_time"], errors="coerce")
    rt = pd.to_datetime(reached["root_sweep_time"], errors="coerce")
    add("state_available_not_before_root", int((av < rt + pd.Timedelta(minutes=1)).sum()))
    filled = reached.loc[reached["entry_status"].eq("filled")].copy()
    add("entry_not_before_availability", int((pd.to_datetime(filled["entry_time"], errors="coerce") < pd.to_datetime(filled["available_time"], errors="coerce")).sum()))
    order = reached.sort_values(["root_event_id", "state_id"], kind="stable")
    prev = order.groupby("root_event_id")["available_time"].shift(1)
    add("state_availability_order", int((pd.to_datetime(order["available_time"], errors="coerce") < pd.to_datetime(prev, errors="coerce")).sum()))
    add("holdout_absent", int(pd.to_datetime(rows["root_sweep_time"], errors="coerce").ge(cfg.holdout_start).sum()))
    gap_violations = 0
    for _, group in rows.loc[rows["state_reached"].eq(1)].groupby("root_event_id", sort=False):
        ids = set(group["state_id"].astype(int))
        if ids and ids != set(range(max(ids) + 1)):
            gap_violations += 1
    add("states_after_gap", gap_violations)
    add("invalid_root_atr", int(pd.to_numeric(diagnostics.get("root_atr"), errors="coerce").isna().sum()))
    add("target_stop_geometry", int(((filled["structural_risk_pct"] <= 0) | (filled["structural_reward_pct"] <= 0)).sum()))
    return pd.DataFrame(checks)


def freeze_discovery_decision(summary: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Apply the preregistered earliest-stable-divergence gate by side."""
    q = summary.loc[(summary["grain"] == "overall") & (summary["research_split"] == "discovery")].copy()
    years = summary.loc[(summary["grain"] == "year") & (summary["research_split"] == "discovery")].copy()
    audit_ok = int(pd.to_numeric(audit.get("violations"), errors="coerce").fillna(0).sum()) == 0
    decisions: list[dict[str, object]] = []
    for side in ("SSL", "BSL"):
        part = q.loc[q["root_side"].eq(side)].sort_values("state_id")
        base = part.loc[part["state_id"].eq(0), "direct_delivery_probability"]
        base_p = float(base.iloc[0]) if len(base) else np.nan
        chosen: pd.Series | None = None; reasons = ""
        for _, row in part.iterrows():
            sid = int(row["state_id"]); yp = years.loc[(years["root_side"] == side) & (years["state_id"] == sid)]
            prior = part.loc[part["state_id"].eq(max(0, sid - 1)), "direct_delivery_probability"]
            prior_p = float(prior.iloc[0]) if len(prior) else np.nan
            flags = {
                "fills_ge50": int(row["filled"]) >= 50,
                "each_year_ge15": len(yp) >= 2 and bool((yp["filled"] >= 15).all()),
                "direct_uplift_ge10pp": pd.notna(row["direct_delivery_probability"]) and float(row["direct_delivery_probability"]) >= base_p + 0.10 - EPS,
                "not_below_prior": pd.notna(row["direct_delivery_probability"]) and float(row["direct_delivery_probability"]) + EPS >= prior_p,
                "overall_gross_2x_positive": float(row["expectancy_gross"]) > 0 and float(row["expectancy_cost2x"]) > 0,
                "year_2x_positive": len(yp) >= 2 and bool((yp["expectancy_cost2x"] > 0).all()),
                "pf_gate": float(row["pf_cost1x"]) >= 1.40 and float(row["pf_cost2x"]) > 1.0,
                "top5_positive": float(row["expectancy_cost2x_remove_top5"]) > 0,
                "causal_audit": bool(audit_ok),
            }
            if all(flags.values()):
                chosen = row; reasons = ";".join(k for k, v in flags.items() if v); break
        if chosen is None:
            decisions.append({"root_side": side, "decision": "NO_DIVERGENCE", "selected_state_id": np.nan, "selected_state": "", "reason": "no ordered state passed every frozen discovery gate", "audit_ok": audit_ok})
        else:
            decisions.append({"root_side": side, "decision": "FREEZE_FOR_VALIDATION", "selected_state_id": int(chosen["state_id"]), "selected_state": chosen["state"], "reason": reasons, "audit_ok": audit_ok, "discovery_fills": int(chosen["filled"]), "discovery_pf_cost2x": chosen["pf_cost2x"], "discovery_expectancy_cost2x": chosen["expectancy_cost2x"]})
    return pd.DataFrame(decisions)


def validation_decision(freeze: pd.DataFrame, summary: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    q = summary.loc[(summary["grain"] == "overall") & (summary["research_split"] == "validation")]
    audit_ok = int(pd.to_numeric(audit.get("violations"), errors="coerce").fillna(0).sum()) == 0
    out: list[dict[str, object]] = []
    for f in freeze.itertuples(index=False):
        if f.decision != "FREEZE_FOR_VALIDATION":
            out.append({"root_side": f.root_side, "decision": "REJECT_NO_DISCOVERY_DIVERGENCE", "selected_state": "", "audit_ok": audit_ok}); continue
        p = q.loc[(q["root_side"] == f.root_side) & (q["state_id"] == int(f.selected_state_id))]
        if p.empty:
            out.append({"root_side": f.root_side, "decision": "REJECT_MISSING_VALIDATION_ROW", "selected_state": f.selected_state, "audit_ok": audit_ok}); continue
        r = p.iloc[0]
        flags = {
            "fills_ge15": int(r["filled"]) >= 15,
            "gross_2x_positive": float(r["expectancy_gross"]) > 0 and float(r["expectancy_cost2x"]) > 0,
            "pf_gate": float(r["pf_cost1x"]) >= 1.20 and float(r["pf_cost2x"]) > 1.0,
            "top5_positive": float(r["expectancy_cost2x_remove_top5"]) > 0,
            "causal_audit": bool(audit_ok),
        }
        out.append({"root_side": f.root_side, "selected_state": f.selected_state, "decision": "ADVANCE" if all(flags.values()) else "REJECT_VALIDATION", "validation_fills": int(r["filled"]), "validation_pf_cost1x": r["pf_cost1x"], "validation_pf_cost2x": r["pf_cost2x"], "validation_expectancy_cost2x": r["expectancy_cost2x"], "validation_expectancy_cost2x_remove_top5": r["expectancy_cost2x_remove_top5"], "audit_ok": audit_ok, **{f"gate_{k}": int(v) for k, v in flags.items()}})
    return pd.DataFrame(out)


def summarize_state_quality_divergence(rows: pd.DataFrame) -> pd.DataFrame:
    """Continuous reached-state quality: direct delivery versus same-side-first."""
    if rows.empty:
        return pd.DataFrame()
    excluded = {
        "state_id", "state_reached", "direct_reversal_label", "year",
        "entry_price", "target_price", "stop_price", "exit_price",
        "gross_return", "net_return_cost1x", "net_return_cost2x", "net_return_cost3x",
        "structural_risk_pct", "structural_reward_pct", "structural_rr",
        "mfe_pct", "mae_pct", "signal_pos", "entry_pos", "exit_pos",
    }
    numeric = [c for c in rows.select_dtypes(include=[np.number]).columns if c not in excluded]
    out: list[dict[str, object]] = []
    reached = rows.loc[rows["state_reached"].eq(1)]
    for (split, side, sid, state), part in reached.groupby(["research_split", "root_side", "state_id", "state"], sort=True):
        success = part.loc[part["direct_reversal_label"].eq(1)]
        failure = part.loc[part["direct_reversal_label"].eq(0)]
        for feature in numeric:
            a = pd.to_numeric(success[feature], errors="coerce").dropna()
            b = pd.to_numeric(failure[feature], errors="coerce").dropna()
            if not len(a) or not len(b):
                continue
            pooled = pd.concat([a, b]); sd = float(pooled.std(ddof=0)); diff = float(a.mean() - b.mean())
            out.append({
                "research_split": split, "root_side": side, "state_id": sid, "state": state,
                "feature": feature, "success_n": len(a), "failure_n": len(b),
                "success_mean": a.mean(), "failure_mean": b.mean(),
                "success_median": a.median(), "failure_median": b.median(),
                "mean_difference": diff,
                "standardized_mean_difference": diff / sd if sd > EPS else np.nan,
            })
    return pd.DataFrame(out)


def summarize_protected_stop_diagnostic(rows: pd.DataFrame) -> pd.DataFrame:
    q = rows.loc[(rows["state_id"] == 6) & rows.get("protected_stop_status", pd.Series("", index=rows.index)).eq("filled")].copy()
    if q.empty:
        return pd.DataFrame(columns=["research_split", "root_side", "fills", "tp_first", "sl_first", "median_risk_pct", "median_rr", "expectancy_cost1x", "expectancy_cost2x", "expectancy_cost3x", "pf_cost2x"])
    out: list[dict[str, object]] = []
    for (split, side), p in q.groupby(["research_split", "root_side"], sort=True):
        vals2 = pd.to_numeric(p["protected_stop_net_return_cost2x"], errors="coerce").dropna()
        out.append({
            "research_split": split, "root_side": side, "fills": len(p),
            "tp_first": int(p["protected_stop_outcome"].eq("tp_first").sum()),
            "sl_first": int(p["protected_stop_outcome"].eq("sl_first").sum()),
            "median_risk_pct": pd.to_numeric(p["protected_stop_risk_pct"], errors="coerce").median(),
            "median_rr": pd.to_numeric(p["protected_stop_rr"], errors="coerce").median(),
            "expectancy_cost1x": pd.to_numeric(p["protected_stop_net_return_cost1x"], errors="coerce").mean(),
            "expectancy_cost2x": vals2.mean(),
            "expectancy_cost3x": pd.to_numeric(p["protected_stop_net_return_cost3x"], errors="coerce").mean(),
            "pf_cost2x": _profit_factor(vals2),
        })
    return pd.DataFrame(out)


def config_record(config: R27Config | None = None) -> dict[str, object]:
    cfg = (config or R27Config()).validate()
    return {k: (str(v) if isinstance(v, pd.Timestamp) else list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()}
