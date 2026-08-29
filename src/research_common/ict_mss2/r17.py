#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R17 causal trend-pullback reclaim/re-acceleration path atlas.

The module deliberately builds a new continuation mechanism instead of reusing
the completed-trend sweep entry families.  Higher-timeframe structure is made
available only after right-hand pivot confirmation; closed 15m/5m signals
execute at the next observable 1m open.  R17 is an event/path study, not a
portfolio backtest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.core import aggregate_bars, normalize_1m_bars
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex

EPS = 1e-12


@dataclass(frozen=True)
class R17Config:
    research_start: pd.Timestamp = pd.Timestamp("2023-01-01 00:00:00")
    validation_start: pd.Timestamp = pd.Timestamp("2025-01-01 00:00:00")
    embargo_start: pd.Timestamp = pd.Timestamp("2025-07-01 00:00:00")
    holdout_start: pd.Timestamp = pd.Timestamp("2025-08-01 00:00:00")
    setup_expiry_minutes: int = 12 * 60
    atr_window: int = 20
    atr_min_periods: int = 5
    stop_buffer_atr: float = 0.25
    max_stop_distance_pct: float = 0.015
    path_horizon_minutes: int = 72 * 60
    fixed_r_targets: tuple[float, ...] = (1.0, 2.0, 3.0)
    market_roundtrip_cost: float = 0.0011
    cost_scales: tuple[float, ...] = (1.0, 2.0, 3.0)

    def validate(self) -> "R17Config":
        if not (self.research_start < self.validation_start < self.embargo_start < self.holdout_start):
            raise ValueError("R17 split boundaries must be strictly increasing")
        if self.setup_expiry_minutes <= 0 or self.path_horizon_minutes <= 0:
            raise ValueError("expiry and path horizon must be positive")
        if self.atr_window < 2 or not 1 <= self.atr_min_periods <= self.atr_window:
            raise ValueError("invalid ATR window")
        if self.stop_buffer_atr < 0:
            raise ValueError("stop buffer cannot be negative")
        if not 0 < self.max_stop_distance_pct < 1:
            raise ValueError("max stop distance must be in (0, 1)")
        if not self.fixed_r_targets or any(float(x) <= 0 for x in self.fixed_r_targets):
            raise ValueError("fixed-R targets must be positive")
        if self.market_roundtrip_cost < 0 or any(float(x) <= 0 for x in self.cost_scales):
            raise ValueError("cost assumptions are invalid")
        return self


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _datetime_ns(values: pd.Series | pd.Index | Iterable[object]) -> np.ndarray:
    parsed = pd.to_datetime(values, errors="coerce")
    return np.asarray(parsed, dtype="datetime64[ns]").astype(np.int64, copy=False)


def _pivot_mask(values: np.ndarray, side: str) -> np.ndarray:
    """Strict order-1 pivot with the same tie convention as MSS2 core."""
    x = np.asarray(values, dtype=float)
    out = np.zeros(len(x), dtype=bool)
    if len(x) < 3:
        return out
    if side == "high":
        out[1:-1] = (x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])
    elif side == "low":
        out[1:-1] = (x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])
    else:
        raise ValueError("side must be high or low")
    out &= np.isfinite(x)
    return out


def _true_range(frame: pd.DataFrame) -> pd.Series:
    high = _num(frame, "high")
    low = _num(frame, "low")
    previous_close = _num(frame, "close").shift(1)
    return pd.concat(
        [(high - low).abs(), (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)


def build_structural_state(
    bars_1m: pd.DataFrame,
    *,
    minutes: int,
    timeframe: str,
    atr_window: int = 20,
    atr_min_periods: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build causal order-1 pivot events and HH/HL or LH/LL state snapshots.

    A pivot at HTF position ``p`` is incorporated only in the snapshot emitted
    after position ``p + 1`` closes.  Snapshot timestamps are therefore direct
    information-availability timestamps, not pivot timestamps.
    """

    htf = aggregate_bars(bars_1m, int(minutes))
    if htf.empty:
        return pd.DataFrame(), pd.DataFrame()
    high = _num(htf, "high").to_numpy(float)
    low = _num(htf, "low").to_numpy(float)
    high_mask = _pivot_mask(high, "high")
    low_mask = _pivot_mask(low, "low")
    atr = _true_range(htf).rolling(int(atr_window), min_periods=int(atr_min_periods)).mean().to_numpy(float)
    delta = pd.Timedelta(minutes=int(minutes))

    latest_highs: list[tuple[float, pd.Timestamp, pd.Timestamp, int]] = []
    latest_lows: list[tuple[float, pd.Timestamp, pd.Timestamp, int]] = []
    pivot_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    prior_direction = 0
    direction_started = pd.NaT

    for right_pos in range(len(htf)):
        pivot_pos = right_pos - 1
        available = pd.Timestamp(htf.index[right_pos] + delta)
        if pivot_pos >= 1:
            for side, mask, price in (
                ("high", high_mask, high),
                ("low", low_mask, low),
            ):
                if not bool(mask[pivot_pos]):
                    continue
                item = (
                    float(price[pivot_pos]),
                    pd.Timestamp(htf.index[pivot_pos]),
                    available,
                    int(pivot_pos),
                )
                (latest_highs if side == "high" else latest_lows).append(item)
                pivot_rows.append(
                    {
                        "source_timeframe": str(timeframe),
                        "source_timeframe_min": int(minutes),
                        "pivot_side": side,
                        "pivot_pos_htf": int(pivot_pos),
                        "pivot_time": pd.Timestamp(htf.index[pivot_pos]),
                        "pivot_bar_end_time": pd.Timestamp(htf.index[pivot_pos] + delta),
                        "pivot_available_time": available,
                        "pivot_price": float(price[pivot_pos]),
                        "pivot_bar_open": float(htf.iloc[pivot_pos]["open"]),
                        "pivot_bar_high": float(htf.iloc[pivot_pos]["high"]),
                        "pivot_bar_low": float(htf.iloc[pivot_pos]["low"]),
                        "pivot_bar_close": float(htf.iloc[pivot_pos]["close"]),
                        "atr_at_available": float(atr[right_pos]) if np.isfinite(atr[right_pos]) else np.nan,
                    }
                )

        direction = 0
        if len(latest_highs) >= 2 and len(latest_lows) >= 2:
            higher_highs = latest_highs[-1][0] > latest_highs[-2][0] + EPS
            higher_lows = latest_lows[-1][0] > latest_lows[-2][0] + EPS
            lower_highs = latest_highs[-1][0] < latest_highs[-2][0] - EPS
            lower_lows = latest_lows[-1][0] < latest_lows[-2][0] - EPS
            if higher_highs and higher_lows:
                direction = 1
            elif lower_highs and lower_lows:
                direction = -1
        if direction != prior_direction:
            direction_started = available if direction else pd.NaT
        prior_direction = direction
        age_bars = (
            float((available - pd.Timestamp(direction_started)) / delta)
            if direction and pd.notna(direction_started)
            else np.nan
        )
        state_rows.append(
            {
                "source_timeframe": str(timeframe),
                "source_timeframe_min": int(minutes),
                "state_available_time": available,
                "structural_direction": int(direction),
                "state_started_time": direction_started,
                "state_age_bars": age_bars,
                "latest_high_price": latest_highs[-1][0] if latest_highs else np.nan,
                "latest_high_pivot_time": latest_highs[-1][1] if latest_highs else pd.NaT,
                "latest_high_available_time": latest_highs[-1][2] if latest_highs else pd.NaT,
                "latest_low_price": latest_lows[-1][0] if latest_lows else np.nan,
                "latest_low_pivot_time": latest_lows[-1][1] if latest_lows else pd.NaT,
                "latest_low_available_time": latest_lows[-1][2] if latest_lows else pd.NaT,
                "close_at_state": float(htf.iloc[right_pos]["close"]),
                "atr_at_state": float(atr[right_pos]) if np.isfinite(atr[right_pos]) else np.nan,
            }
        )

    pivots = pd.DataFrame(pivot_rows)
    if not pivots.empty:
        pivots = pivots.sort_values(
            ["pivot_available_time", "pivot_side", "pivot_time"], kind="stable"
        ).reset_index(drop=True)
        pivots["next_same_side_pivot_available_time"] = pivots.groupby(
            "pivot_side", sort=False
        )["pivot_available_time"].shift(-1)
    states = pd.DataFrame(state_rows).sort_values("state_available_time", kind="stable").reset_index(drop=True)
    return states, pivots


class _StateLookup:
    def __init__(self, states: pd.DataFrame):
        self.states = states.reset_index(drop=True)
        self.available_ns = _datetime_ns(self.states.get("state_available_time", pd.Series(dtype="datetime64[ns]")))

    def at(self, when: pd.Timestamp) -> pd.Series | None:
        if self.states.empty:
            return None
        pos = int(np.searchsorted(self.available_ns, np.datetime64(pd.Timestamp(when), "ns").astype(np.int64), side="right") - 1)
        return None if pos < 0 else self.states.iloc[pos]


def _first_condition_position(
    available_ns: np.ndarray,
    values: np.ndarray,
    *,
    start_time: pd.Timestamp,
    end_time_exclusive: pd.Timestamp,
    threshold: float,
    direction: int,
    start_strict: bool,
) -> int:
    start = int(
        np.searchsorted(
            available_ns,
            np.datetime64(pd.Timestamp(start_time), "ns").astype(np.int64),
            side="right" if start_strict else "left",
        )
    )
    end = int(
        np.searchsorted(
            available_ns,
            np.datetime64(pd.Timestamp(end_time_exclusive), "ns").astype(np.int64),
            side="left",
        )
    )
    if start >= end:
        return -1
    segment = values[start:end]
    hit = segment > float(threshold) + EPS if direction > 0 else segment < float(threshold) - EPS
    positions = np.flatnonzero(hit)
    return start + int(positions[0]) if len(positions) else -1


def _research_split(when: pd.Timestamp, cfg: R17Config) -> str:
    t = pd.Timestamp(when)
    if t < cfg.research_start:
        return "warmup"
    if t < cfg.validation_start:
        return "discovery"
    if t < cfg.embargo_start:
        return "validation"
    if t < cfg.holdout_start:
        return "embargo"
    return "holdout"


def build_pullback_setup_atlas(
    bars_1m: pd.DataFrame,
    *,
    config: R17Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the frozen R17 event sequence while keeping holdout details sealed."""

    cfg = (config or R17Config()).validate()
    bars = normalize_1m_bars(bars_1m)
    state_1d, _ = build_structural_state(
        bars, minutes=1440, timeframe="1D", atr_window=cfg.atr_window, atr_min_periods=cfg.atr_min_periods
    )
    state_4h, _ = build_structural_state(
        bars, minutes=240, timeframe="4H", atr_window=cfg.atr_window, atr_min_periods=cfg.atr_min_periods
    )
    _, pullbacks = build_structural_state(
        bars, minutes=30, timeframe="30m", atr_window=cfg.atr_window, atr_min_periods=cfg.atr_min_periods
    )
    bars_15m = aggregate_bars(bars, 15)
    bars_5m = aggregate_bars(bars, 5)
    lookup_1d = _StateLookup(state_1d)
    lookup_4h = _StateLookup(state_4h)

    available_15_ns = _datetime_ns(bars_15m["bar_end_time"])
    available_5_ns = _datetime_ns(bars_5m["bar_end_time"])
    close_15 = _num(bars_15m, "close").to_numpy(float)
    high_15 = _num(bars_15m, "high").to_numpy(float)
    low_15 = _num(bars_15m, "low").to_numpy(float)
    close_5 = _num(bars_5m, "close").to_numpy(float)
    bars_index_ns = _datetime_ns(bars.index)

    rows: list[dict[str, object]] = []
    raw_after_start = pullbacks.loc[
        pd.to_datetime(pullbacks.get("pivot_available_time"), errors="coerce").ge(cfg.research_start)
    ].copy()
    for p in raw_after_start.itertuples(index=False):
        direction = 1 if str(p.pivot_side) == "low" else -1
        direction_name = "LONG" if direction > 0 else "SHORT"
        pullback_available = pd.Timestamp(p.pivot_available_time)
        split_at_pullback = _research_split(pullback_available, cfg)
        s1 = lookup_1d.at(pullback_available)
        s4 = lookup_4h.at(pullback_available)
        aligned = bool(
            s1 is not None
            and s4 is not None
            and int(s1["structural_direction"]) == direction
            and int(s4["structural_direction"]) == direction
        )
        base: dict[str, object] = {
            "setup_id": f"R17_{direction_name}_{pullback_available:%Y%m%d%H%M}_{int(p.pivot_pos_htf):07d}",
            "direction": direction_name,
            "trade_direction": int(direction),
            "pivot_side": str(p.pivot_side),
            "pullback_pivot_time": pd.Timestamp(p.pivot_time),
            "pullback_bar_end_time": pd.Timestamp(p.pivot_bar_end_time),
            "pullback_available_time": pullback_available,
            "next_same_side_pivot_available_time": pd.Timestamp(p.next_same_side_pivot_available_time)
            if pd.notna(p.next_same_side_pivot_available_time)
            else pd.NaT,
            "pullback_price": float(p.pivot_price),
            "pullback_bar_high": float(p.pivot_bar_high),
            "pullback_bar_low": float(p.pivot_bar_low),
            "pullback_atr_30m": float(p.atr_at_available),
            "pullback_bar_range_atr": (
                (float(p.pivot_bar_high) - float(p.pivot_bar_low)) / float(p.atr_at_available)
                if np.isfinite(float(p.atr_at_available)) and float(p.atr_at_available) > EPS
                else np.nan
            ),
            "trend_1d_state_at_pullback": int(s1["structural_direction"]) if s1 is not None else 0,
            "trend_4h_state_at_pullback": int(s4["structural_direction"]) if s4 is not None else 0,
            "trend_1d_available_time_at_pullback": pd.Timestamp(s1["state_available_time"]) if s1 is not None else pd.NaT,
            "trend_4h_available_time_at_pullback": pd.Timestamp(s4["state_available_time"]) if s4 is not None else pd.NaT,
            "trend_1d_age_bars_at_pullback": float(s1["state_age_bars"]) if s1 is not None else np.nan,
            "trend_4h_age_bars_at_pullback": float(s4["state_age_bars"]) if s4 is not None else np.nan,
            "aligned_trend_at_pullback": int(aligned),
            "research_split": split_at_pullback,
            "setup_status": "trend_not_aligned",
        }
        if not aligned:
            rows.append(base)
            continue
        if split_at_pullback == "holdout":
            base["setup_status"] = "sealed_holdout_pullback"
            rows.append(base)
            continue
        if split_at_pullback == "embargo":
            base["setup_status"] = "embargo_pullback"
            rows.append(base)
            continue

        expiry = pullback_available + pd.Timedelta(minutes=cfg.setup_expiry_minutes)
        if pd.notna(p.next_same_side_pivot_available_time):
            expiry = min(expiry, pd.Timestamp(p.next_same_side_pivot_available_time))
        expiry = min(expiry, cfg.embargo_start)
        base["setup_expiry_time"] = expiry
        if not np.isfinite(float(p.atr_at_available)) or float(p.atr_at_available) <= EPS:
            base["setup_status"] = "atr_unavailable"
            rows.append(base)
            continue

        reclaim_threshold = float(p.pivot_bar_high) if direction > 0 else float(p.pivot_bar_low)
        reclaim_pos = _first_condition_position(
            available_15_ns,
            close_15,
            start_time=pullback_available,
            end_time_exclusive=expiry,
            threshold=reclaim_threshold,
            direction=direction,
            start_strict=False,
        )
        if reclaim_pos < 0:
            base["setup_status"] = "no_15m_reclaim"
            rows.append(base)
            continue
        reclaim_available = pd.Timestamp(bars_15m.iloc[reclaim_pos]["bar_end_time"])
        reclaim_break = float(high_15[reclaim_pos]) if direction > 0 else float(low_15[reclaim_pos])
        base.update(
            {
                "reclaim_15m_available_time": reclaim_available,
                "reclaim_15m_threshold": reclaim_threshold,
                "reclaim_15m_break_price": reclaim_break,
                "reclaim_delay_minutes": float((reclaim_available - pullback_available) / pd.Timedelta(minutes=1)),
            }
        )
        signal_pos = _first_condition_position(
            available_5_ns,
            close_5,
            start_time=reclaim_available,
            end_time_exclusive=expiry,
            threshold=reclaim_break,
            direction=direction,
            start_strict=True,
        )
        if signal_pos < 0:
            base["setup_status"] = "no_5m_reacceleration"
            rows.append(base)
            continue
        signal_available = pd.Timestamp(bars_5m.iloc[signal_pos]["bar_end_time"])
        base.update(
            {
                "signal_available_time": signal_available,
                "signal_5m_close": float(close_5[signal_pos]),
                "signal_delay_minutes": float((signal_available - pullback_available) / pd.Timedelta(minutes=1)),
            }
        )
        split_at_signal = _research_split(signal_available, cfg)
        base["research_split"] = split_at_signal
        if split_at_signal not in {"discovery", "validation"}:
            base["setup_status"] = f"{split_at_signal}_signal"
            rows.append(base)
            continue

        final_1d = lookup_1d.at(signal_available)
        final_4h = lookup_4h.at(signal_available)
        aligned_final = bool(
            final_1d is not None
            and final_4h is not None
            and int(final_1d["structural_direction"]) == direction
            and int(final_4h["structural_direction"]) == direction
        )
        base.update(
            {
                "trend_1d_state_at_signal": int(final_1d["structural_direction"]) if final_1d is not None else 0,
                "trend_4h_state_at_signal": int(final_4h["structural_direction"]) if final_4h is not None else 0,
                "trend_1d_available_time_at_signal": pd.Timestamp(final_1d["state_available_time"])
                if final_1d is not None
                else pd.NaT,
                "trend_4h_available_time_at_signal": pd.Timestamp(final_4h["state_available_time"])
                if final_4h is not None
                else pd.NaT,
                "trend_1d_age_bars_at_signal": float(final_1d["state_age_bars"]) if final_1d is not None else np.nan,
                "trend_4h_age_bars_at_signal": float(final_4h["state_age_bars"]) if final_4h is not None else np.nan,
                "aligned_trend_at_signal": int(aligned_final),
            }
        )
        if not aligned_final:
            base["setup_status"] = "trend_not_aligned_at_signal"
            rows.append(base)
            continue

        entry_pos = int(
            np.searchsorted(
                bars_index_ns,
                np.datetime64(signal_available, "ns").astype(np.int64),
                side="left",
            )
        )
        if entry_pos >= len(bars) or pd.Timestamp(bars.index[entry_pos]) != signal_available:
            base["setup_status"] = "next_1m_entry_unavailable"
            rows.append(base)
            continue
        entry_price = float(bars.iloc[entry_pos]["open"])
        stop = (
            float(p.pivot_bar_low) - cfg.stop_buffer_atr * float(p.atr_at_available)
            if direction > 0
            else float(p.pivot_bar_high) + cfg.stop_buffer_atr * float(p.atr_at_available)
        )
        risk = direction * (entry_price - stop) / entry_price if entry_price > EPS else np.nan
        target_col = "latest_high_price" if direction > 0 else "latest_low_price"
        target_time_col = "latest_high_available_time" if direction > 0 else "latest_low_available_time"
        target_price = float(final_4h[target_col]) if final_4h is not None else np.nan
        target_available = pd.Timestamp(final_4h[target_time_col]) if final_4h is not None else pd.NaT
        runway = direction * (target_price / entry_price - 1.0) if entry_price > EPS else np.nan
        base.update(
            {
                "entry_time": signal_available,
                "entry_price": entry_price,
                "stop_price": stop,
                "risk_distance_pct": risk,
                "structural_target_price": target_price,
                "structural_target_available_time": target_available,
                "structural_runway_pct": runway,
                "structural_reward_risk": runway / risk if np.isfinite(risk) and risk > EPS else np.nan,
            }
        )
        if not np.isfinite(risk) or risk <= EPS:
            base["setup_status"] = "invalid_stop_geometry"
        elif risk > cfg.max_stop_distance_pct + EPS:
            base["setup_status"] = "stop_too_wide"
        elif not np.isfinite(runway) or runway <= EPS:
            base["setup_status"] = "no_remaining_4h_runway"
        elif pd.isna(target_available) or target_available > signal_available:
            base["setup_status"] = "target_not_causally_available"
        else:
            base["setup_status"] = "executable"
        rows.append(base)

    all_rows = pd.DataFrame(rows)
    if all_rows.empty:
        return all_rows, pd.DataFrame(), pd.DataFrame()
    for column in [c for c in all_rows.columns if c.endswith("_time")]:
        all_rows[column] = pd.to_datetime(all_rows[column], errors="coerce")

    # Predeclared duplicate rule: the latest pullback owns a shared direction/signal.
    signaled = all_rows.loc[all_rows["signal_available_time"].notna()].sort_values(
        ["direction", "signal_available_time", "pullback_available_time"],
        ascending=[True, True, False],
        kind="stable",
    )
    duplicate_index = signaled.loc[
        signaled.duplicated(["direction", "signal_available_time"], keep="first")
    ].index
    all_rows.loc[duplicate_index, "setup_status"] = "duplicate_signal_superseded"

    holdout_mask = all_rows["research_split"].eq("holdout") | all_rows["setup_status"].astype(str).str.startswith(
        "sealed_holdout"
    )
    seal = pd.DataFrame(
        [
            {"check": "holdout_start", "value": str(cfg.holdout_start)},
            {"check": "aligned_holdout_pullbacks", "value": int((holdout_mask & all_rows["aligned_trend_at_pullback"].eq(1)).sum())},
            {"check": "holdout_outcome_rows_computed", "value": 0},
            {"check": "holdout_unsealed", "value": 0},
        ]
    )
    visible = all_rows.loc[all_rows["research_split"].isin(["discovery", "validation"])].copy()
    visible = visible.sort_values(["pullback_available_time", "direction", "setup_id"], kind="stable").reset_index(drop=True)
    engineering = pd.DataFrame(
        [
            {"check": "raw_30m_pivots_after_research_start", "value": int(len(raw_after_start))},
            {"check": "aligned_visible_pullbacks", "value": int(visible["aligned_trend_at_pullback"].eq(1).sum())},
            {"check": "visible_15m_reclaims", "value": int(visible["reclaim_15m_available_time"].notna().sum())},
            {"check": "visible_5m_reaccelerations", "value": int(visible["signal_available_time"].notna().sum())},
            {"check": "visible_executable_setups", "value": int(visible["setup_status"].eq("executable").sum())},
        ]
    )
    return visible, seal, engineering


def _first_barrier(
    high_tree: SegmentThresholdIndex,
    low_tree: SegmentThresholdIndex,
    *,
    direction: int,
    target: float,
    stop: float,
    start: int,
    end: int,
) -> tuple[str, int]:
    if direction > 0:
        tp = int(high_tree.first_geq(start, end, target))
        sl = int(low_tree.first_leq(start, end, stop))
    else:
        tp = int(low_tree.first_leq(start, end, target))
        sl = int(high_tree.first_geq(start, end, stop))
    if sl >= 0 and (tp < 0 or sl <= tp):
        return "sl_first", sl
    if tp >= 0:
        return "tp_first", tp
    return "horizon_exit", end


def build_first_passage_paths(
    bars_1m: pd.DataFrame,
    setups: pd.DataFrame,
    *,
    config: R17Config | None = None,
) -> pd.DataFrame:
    """Label structural and fixed-R paths with pessimistic same-bar ordering."""

    cfg = (config or R17Config()).validate()
    if setups.empty:
        return pd.DataFrame()
    events = setups.loc[setups["setup_status"].eq("executable")].copy()
    if events.empty:
        return pd.DataFrame()
    bars = normalize_1m_bars(bars_1m)
    index_ns = _datetime_ns(bars.index)
    high = _num(bars, "high").to_numpy(float)
    low = _num(bars, "low").to_numpy(float)
    close = _num(bars, "close").to_numpy(float)
    high_tree = SegmentThresholdIndex(high)
    low_tree = SegmentThresholdIndex(low)
    rows: list[dict[str, object]] = []
    target_specs: tuple[tuple[str, float | None], ...] = (
        ("H0_4H_STRUCTURAL", None),
        *tuple((f"R{float(r):g}", float(r)) for r in cfg.fixed_r_targets),
    )
    for event in events.itertuples(index=False):
        entry_time = pd.Timestamp(event.entry_time)
        entry_pos = int(
            np.searchsorted(index_ns, np.datetime64(entry_time, "ns").astype(np.int64), side="left")
        )
        if entry_pos >= len(bars) or pd.Timestamp(bars.index[entry_pos]) != entry_time:
            continue
        end = min(len(bars) - 1, entry_pos + int(cfg.path_horizon_minutes) - 1)
        direction = int(event.trade_direction)
        entry = float(event.entry_price)
        stop = float(event.stop_price)
        risk_price = abs(entry - stop)
        base = {
            "setup_id": str(event.setup_id),
            "direction": str(event.direction),
            "trade_direction": direction,
            "research_split": str(event.research_split),
            "year": int(pd.Timestamp(event.entry_time).year),
            "pullback_available_time": pd.Timestamp(event.pullback_available_time),
            "reclaim_15m_available_time": pd.Timestamp(event.reclaim_15m_available_time),
            "signal_available_time": pd.Timestamp(event.signal_available_time),
            "entry_time": entry_time,
            "entry_price": entry,
            "stop_price": stop,
            "risk_distance_pct": float(event.risk_distance_pct),
            "structural_runway_pct": float(event.structural_runway_pct),
            "structural_reward_risk": float(event.structural_reward_risk),
            "trend_1d_age_bars_at_signal": float(event.trend_1d_age_bars_at_signal),
            "trend_4h_age_bars_at_signal": float(event.trend_4h_age_bars_at_signal),
            "pullback_bar_range_atr": float(event.pullback_bar_range_atr),
            "signal_delay_minutes": float(event.signal_delay_minutes),
        }
        for model, multiple in target_specs:
            target = (
                float(event.structural_target_price)
                if multiple is None
                else entry + direction * float(multiple) * risk_price
            )
            geometry = direction * (target - entry) > EPS and direction * (entry - stop) > EPS
            rec = dict(base)
            rec.update({"target_model": model, "target_price": target})
            if not geometry:
                rec["outcome"] = "invalid_geometry"
                rows.append(rec)
                continue
            outcome, exit_pos = _first_barrier(
                high_tree,
                low_tree,
                direction=direction,
                target=target,
                stop=stop,
                start=entry_pos,
                end=end,
            )
            exit_price = target if outcome == "tp_first" else (stop if outcome == "sl_first" else float(close[exit_pos]))
            gross_return = direction * (exit_price / entry - 1.0)
            segment_high = high[entry_pos : exit_pos + 1]
            segment_low = low[entry_pos : exit_pos + 1]
            mfe = (
                float(np.nanmax(segment_high) / entry - 1.0)
                if direction > 0
                else float(1.0 - np.nanmin(segment_low) / entry)
            )
            mae = (
                float(1.0 - np.nanmin(segment_low) / entry)
                if direction > 0
                else float(np.nanmax(segment_high) / entry - 1.0)
            )
            rec.update(
                {
                    "outcome": outcome,
                    "exit_time": pd.Timestamp(bars.index[exit_pos]),
                    "exit_price": exit_price,
                    "holding_minutes": float(exit_pos - entry_pos + 1),
                    "gross_return": gross_return,
                    "gross_r": gross_return / float(event.risk_distance_pct),
                    "mfe_pct": mfe,
                    "mae_pct": mae,
                }
            )
            for scale in cfg.cost_scales:
                net = gross_return - cfg.market_roundtrip_cost * float(scale)
                rec[f"net_return_cost{float(scale):g}x"] = net
                rec[f"net_r_cost{float(scale):g}x"] = net / float(event.risk_distance_pct)
            rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce")
    return out


def _profit_factor(values: Iterable[float]) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    gain = float(x[x > 0].sum())
    loss = float(-x[x < 0].sum())
    return gain / loss if loss > EPS else (np.inf if gain > EPS else np.nan)


def _top_removed_pf(values: pd.Series, count: int) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False)
    return _profit_factor(x.iloc[int(count) :]) if len(x) > int(count) else np.nan


def summarize_setup_funnel(setups: pd.DataFrame) -> pd.DataFrame:
    if setups.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (split, direction), part in setups.groupby(["research_split", "direction"], sort=True):
        aligned = part.loc[part["aligned_trend_at_pullback"].eq(1)].copy()
        rows.append(
            {
                "research_split": split,
                "direction": direction,
                "raw_30m_pivots": int(len(part)),
                "aligned_pullbacks": int(len(aligned)),
                "reclaim_15m_rows": int(aligned["reclaim_15m_available_time"].notna().sum()),
                "reacceleration_5m_rows": int(aligned["signal_available_time"].notna().sum()),
                "executable_rows": int(aligned["setup_status"].eq("executable").sum()),
                "stop_too_wide_rows": int(aligned["setup_status"].eq("stop_too_wide").sum()),
                "no_runway_rows": int(aligned["setup_status"].eq("no_remaining_4h_runway").sum()),
                "duplicate_rows": int(aligned["setup_status"].eq("duplicate_signal_superseded").sum()),
                "median_signal_delay_minutes": _num(aligned.loc[aligned["signal_available_time"].notna()], "signal_delay_minutes").median(),
                "median_risk_distance_pct_executable": _num(aligned.loc[aligned["setup_status"].eq("executable")], "risk_distance_pct").median(),
                "median_structural_rr_executable": _num(aligned.loc[aligned["setup_status"].eq("executable")], "structural_reward_risk").median(),
            }
        )
    return pd.DataFrame(rows)


def _calendar_for_split(split: str) -> pd.PeriodIndex:
    if split == "discovery":
        return pd.period_range("2023-01", "2024-12", freq="M")
    if split == "validation":
        return pd.period_range("2025-01", "2025-06", freq="M")
    return pd.PeriodIndex([], freq="M")


def summarize_path_models(paths: pd.DataFrame, *, config: R17Config | None = None) -> pd.DataFrame:
    cfg = (config or R17Config()).validate()
    if paths.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (split, direction, model), part in paths.groupby(
        ["research_split", "direction", "target_model"], sort=True
    ):
        valid = part.loc[part["gross_return"].notna()].copy()
        times = pd.to_datetime(valid["entry_time"], errors="coerce").dropna().drop_duplicates().sort_values()
        calendar = _calendar_for_split(str(split))
        monthly = pd.Series(0.0, index=calendar)
        observed = (
            valid.assign(month=pd.to_datetime(valid["entry_time"]).dt.to_period("M"))
            .groupby("month")["net_return_cost2x"]
            .sum()
        )
        if len(calendar):
            common = calendar.intersection(observed.index)
            monthly.loc[common] = observed.reindex(common)
        rec: dict[str, object] = {
            "research_split": split,
            "direction": direction,
            "target_model": model,
            "trades": int(len(valid)),
            "trades_per_month": float(len(valid) / max(1, len(calendar))),
            "tp_rate": float(valid["outcome"].eq("tp_first").mean()),
            "sl_rate": float(valid["outcome"].eq("sl_first").mean()),
            "horizon_exit_rate": float(valid["outcome"].eq("horizon_exit").mean()),
            "gross_pf": _profit_factor(valid["gross_return"]),
            "mean_gross_r": _num(valid, "gross_r").mean(),
            "median_risk_distance_pct": _num(valid, "risk_distance_pct").median(),
            "median_structural_reward_risk": _num(valid, "structural_reward_risk").median(),
            "median_holding_minutes": _num(valid, "holding_minutes").median(),
            "positive_month_rate_cost2x": float((monthly > 0).mean()) if len(monthly) else np.nan,
            "longest_entry_gap_days": float(times.diff().max() / pd.Timedelta(days=1)) if len(times) >= 2 else np.nan,
            "net_pf_cost2x_top5_removed": _top_removed_pf(valid["net_return_cost2x"], 5),
            "net_pf_cost2x_top10_removed": _top_removed_pf(valid["net_return_cost2x"], 10),
        }
        for scale in cfg.cost_scales:
            net = _num(valid, f"net_return_cost{float(scale):g}x")
            net_r = _num(valid, f"net_r_cost{float(scale):g}x")
            rec[f"mean_net_return_cost{float(scale):g}x"] = net.mean()
            rec[f"net_pf_cost{float(scale):g}x"] = _profit_factor(net)
            rec[f"mean_net_r_cost{float(scale):g}x"] = net_r.mean()
            rec[f"r_pf_cost{float(scale):g}x"] = _profit_factor(net_r)
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_path_years(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (year, direction, model), part in paths.groupby(["year", "direction", "target_model"], sort=True):
        valid = part.loc[part["gross_return"].notna()].copy()
        rows.append(
            {
                "year": int(year),
                "direction": direction,
                "target_model": model,
                "trades": int(len(valid)),
                "tp_rate": float(valid["outcome"].eq("tp_first").mean()),
                "mean_net_return_cost2x": _num(valid, "net_return_cost2x").mean(),
                "net_pf_cost2x": _profit_factor(_num(valid, "net_return_cost2x")),
                "mean_net_r_cost2x": _num(valid, "net_r_cost2x").mean(),
                "net_pf_cost2x_top5_removed": _top_removed_pf(_num(valid, "net_return_cost2x"), 5),
            }
        )
    return pd.DataFrame(rows)


def r17_causal_audit(
    setups: pd.DataFrame,
    paths: pd.DataFrame,
    *,
    config: R17Config | None = None,
) -> pd.DataFrame:
    cfg = (config or R17Config()).validate()
    checks: list[dict[str, object]] = []

    def add(check: str, violations: int) -> None:
        checks.append({"check": check, "violations": int(violations), "status": "PASS" if int(violations) == 0 else "FAIL"})

    if setups.empty:
        add("nonempty_visible_setup_atlas", 1)
        return pd.DataFrame(checks)
    executable = setups.loc[setups["setup_status"].eq("executable")].copy()
    add("unique_setup_id", int(setups["setup_id"].duplicated().sum()))
    add(
        "trend_state_available_by_pullback",
        int(
            (
                pd.to_datetime(setups["trend_1d_available_time_at_pullback"], errors="coerce")
                > pd.to_datetime(setups["pullback_available_time"], errors="coerce")
            ).sum()
            + (
                pd.to_datetime(setups["trend_4h_available_time_at_pullback"], errors="coerce")
                > pd.to_datetime(setups["pullback_available_time"], errors="coerce")
            ).sum()
        ),
    )
    if executable.empty:
        add("nonempty_executable_setup_atlas", 1)
        return pd.DataFrame(checks)
    pullback = pd.to_datetime(executable["pullback_available_time"], errors="coerce")
    reclaim = pd.to_datetime(executable["reclaim_15m_available_time"], errors="coerce")
    signal = pd.to_datetime(executable["signal_available_time"], errors="coerce")
    entry = pd.to_datetime(executable["entry_time"], errors="coerce")
    add("pullback_before_or_at_reclaim", int((pullback > reclaim).sum()))
    add("reclaim_strictly_before_reacceleration", int((reclaim >= signal).sum()))
    add("closed_signal_executes_next_1m_boundary", int((entry != signal).sum()))
    add(
        "trend_state_available_by_signal",
        int(
            (
                pd.to_datetime(executable["trend_1d_available_time_at_signal"], errors="coerce") > signal
            ).sum()
            + (
                pd.to_datetime(executable["trend_4h_available_time_at_signal"], errors="coerce") > signal
            ).sum()
        ),
    )
    add(
        "structural_target_available_by_signal",
        int((pd.to_datetime(executable["structural_target_available_time"], errors="coerce") > signal).sum()),
    )
    add("maximum_stop_distance_respected", int((_num(executable, "risk_distance_pct") > cfg.max_stop_distance_pct + EPS).sum()))
    add("holdout_absent_from_visible_setups", int((pullback >= cfg.holdout_start).sum() + (signal >= cfg.holdout_start).sum()))
    add(
        "unique_direction_signal_after_dedup",
        int(executable.duplicated(["direction", "signal_available_time"]).sum()),
    )
    if paths.empty:
        add("nonempty_first_passage_paths", 1)
    else:
        add("paths_reference_executable_setups", int((~paths["setup_id"].isin(executable["setup_id"])).sum()))
        add("path_entry_not_before_signal", int((pd.to_datetime(paths["entry_time"]) < pd.to_datetime(paths["signal_available_time"])).sum()))
        add("holdout_absent_from_paths", int((pd.to_datetime(paths["entry_time"]) >= cfg.holdout_start).sum()))
    return pd.DataFrame(checks)
