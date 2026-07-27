#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal accumulated active-flow + price-action research primitives.

This module turns rolling taker-flow pressure into a *process* rather than a
single bar event.  It then waits for price-action confirmation:

* continuation: structure break -> retest holds -> directional resume;
* exhaustion: structure sweep -> reclaim after marginal impact decays.

All pivots become usable only after their right-hand confirmation bars plus one
extra bar.  Signals are generated on closed bars and entries occur at the next
bar open.  Stops and targets come from already-known price structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.research_common.flow_impact import FlowImpactConfig, build_flow_impact_features
from src.research_common.progress import ProgressReporter

_EPS = 1e-12


@dataclass(frozen=True)
class AccumulatedPAConfig:
    accumulation_windows: tuple[int, ...] = (5, 10, 20)
    baseline_bars: int = 1440
    baseline_min_periods: int = 720
    min_accumulation_z: float = 1.50
    pivot_left: int = 2
    pivot_right: int = 2
    structure_lookback_bars: int = 240
    confirmation_bars: int = 5
    retest_tolerance_bps: float = 5.0
    stop_buffer_bps: float = 3.0
    min_risk_bps: float = 10.0
    max_risk_bps: float = 150.0
    min_reward_risk: float = 1.10
    exhaustion_decay_ratio: float = 0.75
    continuation_min_persistence: float = 0.50
    continuation_min_effectiveness: float = 0.00
    event_cooldown_bars: int = 5
    max_holding_bars: int = 240

    def validate(self) -> None:
        if not self.accumulation_windows or any(int(v) < 4 for v in self.accumulation_windows):
            raise ValueError("accumulation_windows must contain integers >= 4")
        if int(self.baseline_min_periods) > int(self.baseline_bars):
            raise ValueError("baseline_min_periods must be <= baseline_bars")
        for name in (
            "pivot_left",
            "pivot_right",
            "structure_lookback_bars",
            "confirmation_bars",
            "event_cooldown_bars",
            "max_holding_bars",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if float(self.min_accumulation_z) <= 0.0:
            raise ValueError("min_accumulation_z must be positive")
        if float(self.max_risk_bps) <= float(self.min_risk_bps):
            raise ValueError("max_risk_bps must exceed min_risk_bps")
        if float(self.min_reward_risk) <= 0.0:
            raise ValueError("min_reward_risk must be positive")


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _rolling_sum(values: pd.Series, start: int, end: int) -> float:
    if end < start:
        return float("nan")
    part = pd.to_numeric(values.iloc[start : end + 1], errors="coerce")
    return float(part.sum(min_count=1))


def _directional_efficiency(
    *,
    direction: int,
    start_price: float,
    end_price: float,
    directional_flow: float,
) -> float:
    if not np.isfinite(start_price) or not np.isfinite(end_price) or start_price <= 0.0:
        return float("nan")
    flow_million = float(directional_flow) / 1_000_000.0
    if flow_million <= _EPS:
        return float("nan")
    move_bps = float(direction) * (float(end_price) / float(start_price) - 1.0) * 10_000.0
    return float(move_bps / flow_million)


def build_causal_pivots(
    bars: pd.DataFrame,
    *,
    left: int,
    right: int,
) -> pd.DataFrame:
    """Return swing highs/lows with their first tradably available position.

    The masks are vectorized because the formal data set contains more than two
    million 1m rows.  A pivot at ``center`` is usable only at
    ``center + right + 1``.
    """
    if left <= 0 or right <= 0:
        raise ValueError("left/right must be positive")
    high = _numeric(bars, "high")
    low = _numeric(bars, "low")
    n = len(bars)
    left_high = high.shift(1).rolling(left, min_periods=left).max()
    right_high = high.shift(-right).rolling(right, min_periods=right).max()
    left_low = low.shift(1).rolling(left, min_periods=left).min()
    right_low = low.shift(-right).rolling(right, min_periods=right).min()
    high_pos = np.flatnonzero(((high > left_high) & (high >= right_high)).fillna(False).to_numpy(dtype=bool))
    low_pos = np.flatnonzero(((low < left_low) & (low <= right_low)).fillna(False).to_numpy(dtype=bool))

    rows: list[pd.DataFrame] = []
    for pivot_type, positions, values in (("high", high_pos, high), ("low", low_pos, low)):
        available = positions + int(right) + 1
        valid = available < n
        positions = positions[valid]
        available = available[valid]
        if not len(positions):
            continue
        rows.append(
            pd.DataFrame(
                {
                    "pivot_type": pivot_type,
                    "pivot_pos": positions.astype(np.int64),
                    "available_pos": available.astype(np.int64),
                    "pivot_time": bars.index[positions],
                    "available_time": bars.index[available],
                    "level": values.iloc[positions].to_numpy(dtype=float),
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["pivot_type", "pivot_pos", "available_pos", "pivot_time", "available_time", "level"])
    return pd.concat(rows, ignore_index=True).sort_values(
        ["available_pos", "pivot_pos", "pivot_type"], kind="stable"
    ).reset_index(drop=True)


def build_accumulated_features(
    bars: pd.DataFrame,
    config: AccumulatedPAConfig | None = None,
) -> pd.DataFrame:
    """Build fixed-window accumulated flow plus early/late marginal impact."""
    cfg = config or AccumulatedPAConfig()
    cfg.validate()
    features = build_flow_impact_features(
        bars,
        FlowImpactConfig(
            pressure_windows=cfg.accumulation_windows,
            baseline_bars=cfg.baseline_bars,
            baseline_min_periods=cfg.baseline_min_periods,
            min_pressure_z=cfg.min_accumulation_z,
            event_cooldown_multiplier=1.0,
        ),
    )
    close = _numeric(features, "close")
    delta = _numeric(features, "delta_notional", 0.0)
    observed = (
        features["source_bar_observed_flag"].fillna(False).astype(bool)
        if "source_bar_observed_flag" in features.columns
        else pd.Series(True, index=features.index, dtype=bool)
    )
    for window in cfg.accumulation_windows:
        suffix = f"w{int(window)}"
        half = int(window) // 2
        late = int(window) - half
        direction = _numeric(features, f"pressure_direction_{suffix}", 0.0)
        early_flow = delta.rolling(half, min_periods=half).sum().shift(late)
        late_flow = delta.rolling(late, min_periods=late).sum()
        early_dir_flow = direction * early_flow
        late_dir_flow = direction * late_flow
        early_start = close.shift(window)
        midpoint = close.shift(late)
        late_end = close
        early_move = direction * (midpoint / early_start - 1.0) * 10_000.0
        late_move = direction * (late_end / midpoint - 1.0) * 10_000.0
        early_eff = early_move / (early_dir_flow / 1_000_000.0).replace(0.0, np.nan)
        late_eff = late_move / (late_dir_flow / 1_000_000.0).replace(0.0, np.nan)
        denom = early_eff.abs().clip(lower=0.25)
        decay_ratio = late_eff / denom
        late_flow_share = late_dir_flow / (early_dir_flow.clip(lower=0.0) + late_dir_flow.clip(lower=0.0)).replace(0.0, np.nan)
        observed_count = observed.astype(np.int16).rolling(window, min_periods=window).sum()
        valid = observed_count.eq(window) & features[f"feature_ready_{suffix}"].fillna(False).astype(bool)
        features[f"early_impact_bps_per_million_{suffix}"] = early_eff.where(valid).clip(-500.0, 500.0)
        features[f"late_impact_bps_per_million_{suffix}"] = late_eff.where(valid).clip(-500.0, 500.0)
        features[f"impact_decay_ratio_{suffix}"] = decay_ratio.where(valid).clip(-20.0, 20.0)
        features[f"late_directional_flow_share_{suffix}"] = late_flow_share.where(valid).clip(-2.0, 2.0)
    return features.replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class _PivotSideIndex:
    """NumPy-only causal pivot index for low-overhead event queries."""

    pivot_positions: np.ndarray
    available_positions: np.ndarray
    levels: np.ndarray

    def _bounds(self, *, available_pos: int, min_pivot_pos: int) -> tuple[int, int]:
        hi = int(np.searchsorted(self.available_positions, int(available_pos), side="right"))
        if hi <= 0:
            return 0, 0
        lo = int(np.searchsorted(self.pivot_positions[:hi], int(min_pivot_pos), side="left"))
        return lo, hi

    def last_level(self, *, available_pos: int, min_pivot_pos: int) -> float:
        lo, hi = self._bounds(available_pos=available_pos, min_pivot_pos=min_pivot_pos)
        if hi <= lo:
            return float("nan")
        return float(self.levels[hi - 1])

    def nearest_target(
        self,
        *,
        available_pos: int,
        min_pivot_pos: int,
        side: int,
        entry_price: float,
    ) -> float:
        lo, hi = self._bounds(available_pos=available_pos, min_pivot_pos=min_pivot_pos)
        if hi <= lo:
            return float("nan")
        levels = self.levels[lo:hi]
        if side > 0:
            candidates = levels[levels > float(entry_price)]
            return float(np.min(candidates)) if candidates.size else float("nan")
        candidates = levels[levels < float(entry_price)]
        return float(np.max(candidates)) if candidates.size else float("nan")


@dataclass(frozen=True)
class _CausalPivotIndex:
    highs: _PivotSideIndex
    lows: _PivotSideIndex

    @classmethod
    def from_frame(cls, pivots: pd.DataFrame) -> "_CausalPivotIndex":
        def build_side(pivot_type: str) -> _PivotSideIndex:
            part = pivots.loc[pivots["pivot_type"].eq(pivot_type)].sort_values(
                ["available_pos", "pivot_pos"], kind="stable"
            )
            return _PivotSideIndex(
                pivot_positions=pd.to_numeric(part["pivot_pos"], errors="coerce").to_numpy(dtype=np.int64),
                available_positions=pd.to_numeric(part["available_pos"], errors="coerce").to_numpy(dtype=np.int64),
                levels=pd.to_numeric(part["level"], errors="coerce").to_numpy(dtype=float),
            )

        return cls(highs=build_side("high"), lows=build_side("low"))

    def side(self, pivot_type: str) -> _PivotSideIndex:
        if pivot_type == "high":
            return self.highs
        if pivot_type == "low":
            return self.lows
        raise ValueError(f"unsupported pivot_type: {pivot_type}")


def _cooldown_positions(candidates: list[dict[str, Any]], cooldown_bars: int) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    last_by_key: dict[tuple[str, int], int] = {}
    for row in sorted(candidates, key=lambda x: (int(x["signal_pos"]), str(x["branch"]), int(x["trade_side"]))):
        key = (f"{row['branch']}::{row['profile']}", int(row["trade_side"]))
        last = last_by_key.get(key, -10**12)
        if int(row["signal_pos"]) - int(last) <= int(cooldown_bars):
            continue
        kept.append(row)
        last_by_key[key] = int(row["signal_pos"])
    return kept


def detect_accumulated_pa_setups(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    pivots: pd.DataFrame,
    config: AccumulatedPAConfig | None = None,
    *,
    progress_enabled: bool = True,
) -> pd.DataFrame:
    """Detect accumulated-flow PA setups in O(events × local-window) time.

    Full-column conversions are performed once per bar field / pressure window.
    This avoids the former event-by-event conversion of multi-million-row
    feature columns.
    """
    cfg = config or AccumulatedPAConfig()
    cfg.validate()
    if bars.empty or pivots.empty:
        return pd.DataFrame()

    open_values = _numeric(bars, "open").to_numpy(dtype=float)
    high_values = _numeric(bars, "high").to_numpy(dtype=float)
    low_values = _numeric(bars, "low").to_numpy(dtype=float)
    close_values = _numeric(bars, "close").to_numpy(dtype=float)
    n = len(bars)
    pivot_index = _CausalPivotIndex.from_frame(pivots)
    tol = float(cfg.retest_tolerance_bps) / 10_000.0
    buffer = float(cfg.stop_buffer_bps) / 10_000.0
    rows: list[dict[str, Any]] = []

    window_inputs: list[dict[str, Any]] = []
    total_events = 0
    for window in cfg.accumulation_windows:
        suffix = f"w{int(window)}"
        ready = features[f"feature_ready_{suffix}"].fillna(False).to_numpy(dtype=bool)
        pressure_z = _numeric(features, f"pressure_z_{suffix}").to_numpy(dtype=float)
        direction = (
            _numeric(features, f"pressure_direction_{suffix}", 0.0)
            .fillna(0.0)
            .to_numpy(dtype=np.int8)
        )
        active = ready & (pressure_z >= float(cfg.min_accumulation_z)) & (direction != 0)
        prior_active = np.empty_like(active)
        prior_active[0] = False
        prior_active[1:] = active[:-1]
        prior_direction = np.empty_like(direction)
        prior_direction[0] = 0
        prior_direction[1:] = direction[:-1]
        onset = active & ((~prior_active) | (direction != prior_direction))
        event_positions = np.flatnonzero(onset).astype(np.int64)
        total_events += int(event_positions.size)
        window_inputs.append(
            {
                "window": int(window),
                "pressure_z": pressure_z,
                "direction": direction,
                "event_positions": event_positions,
                "price_response": _numeric(features, f"price_response_{suffix}").to_numpy(dtype=float),
                "effectiveness": _numeric(features, f"pressure_effectiveness_{suffix}").to_numpy(dtype=float),
                "persistence": _numeric(features, f"flow_persistence_{suffix}").to_numpy(dtype=float),
                "decay_ratio": _numeric(features, f"impact_decay_ratio_{suffix}").to_numpy(dtype=float),
                "late_flow_share": _numeric(
                    features, f"late_directional_flow_share_{suffix}"
                ).to_numpy(dtype=float),
                "accumulated_notional": _numeric(
                    features, f"pressure_notional_{suffix}"
                ).to_numpy(dtype=float),
                "flow_ratio": _numeric(features, f"flow_ratio_{suffix}").to_numpy(dtype=float),
                "activity_z": _numeric(features, f"activity_z_{suffix}").to_numpy(dtype=float),
            }
        )

    reporter = ProgressReporter(
        "[pa-detect] pressure events",
        total_events,
        every=max(1, total_events // 100),
        enabled=progress_enabled,
    )
    processed = 0
    try:
        for data in window_inputs:
            window = int(data["window"])
            pressure_z_values = data["pressure_z"]
            direction_values = data["direction"]
            for raw_event_pos in data["event_positions"]:
                event_pos = int(raw_event_pos)
                processed += 1
                reporter.update(processed)

                start_pos = event_pos - window + 1
                if start_pos <= 0 or event_pos + int(cfg.confirmation_bars) + 1 >= n:
                    continue

                pressure_side = int(direction_values[event_pos])
                break_type = "high" if pressure_side > 0 else "low"
                opposite_type = "low" if pressure_side > 0 else "high"
                min_structure_pos = max(0, start_pos - int(cfg.structure_lookback_bars))
                break_level = pivot_index.side(break_type).last_level(
                    available_pos=start_pos,
                    min_pivot_pos=min_structure_pos,
                )
                opposite_level = pivot_index.side(opposite_type).last_level(
                    available_pos=start_pos,
                    min_pivot_pos=min_structure_pos,
                )
                if not np.isfinite(break_level):
                    continue

                attack_high = float(np.nanmax(high_values[start_pos : event_pos + 1]))
                attack_low = float(np.nanmin(low_values[start_pos : event_pos + 1]))
                price_response = float(data["price_response"][event_pos])
                effectiveness = float(data["effectiveness"][event_pos])
                persistence = float(data["persistence"][event_pos])
                decay_ratio = float(data["decay_ratio"][event_pos])
                late_flow_share = float(data["late_flow_share"][event_pos])
                accumulated_notional = float(data["accumulated_notional"][event_pos])
                flow_ratio = float(data["flow_ratio"][event_pos])
                activity_z = float(data["activity_z"][event_pos])

                if pressure_side > 0:
                    break_hits = np.flatnonzero(
                        high_values[start_pos : event_pos + 1] > float(break_level)
                    )
                else:
                    break_hits = np.flatnonzero(
                        low_values[start_pos : event_pos + 1] < float(break_level)
                    )
                broke = bool(break_hits.size)
                break_pos = int(start_pos + break_hits[0]) if broke else -1
                swept = broke

                confirm_end = min(n - 1, event_pos + int(cfg.confirmation_bars) + 1)
                for signal_pos in range(event_pos, confirm_end):
                    prior_close = float(close_values[signal_pos - 1])
                    signal_open = float(open_values[signal_pos])
                    signal_close = float(close_values[signal_pos])
                    signal_high = float(high_values[signal_pos])
                    signal_low = float(low_values[signal_pos])

                    reclaim = signal_close > break_level if pressure_side < 0 else signal_close < break_level
                    reversal_body = signal_close > signal_open if pressure_side < 0 else signal_close < signal_open
                    reversal_breakbar = (
                        signal_close > prior_close and signal_close > float(high_values[signal_pos - 1])
                        if pressure_side < 0
                        else signal_close < prior_close and signal_close < float(low_values[signal_pos - 1])
                    )
                    if swept and reclaim and reversal_body:
                        trade_side = -pressure_side
                        entry_pos = signal_pos + 1
                        entry_price = float(open_values[entry_pos])
                        if trade_side > 0:
                            stop = min(attack_low, signal_low) * (1.0 - buffer)
                            target_side = pivot_index.highs
                        else:
                            stop = max(attack_high, signal_high) * (1.0 + buffer)
                            target_side = pivot_index.lows
                        target = target_side.nearest_target(
                            available_pos=signal_pos,
                            min_pivot_pos=max(0, signal_pos - int(cfg.structure_lookback_bars)),
                            side=trade_side,
                            entry_price=entry_price,
                        )
                        if not np.isfinite(target):
                            target = opposite_level
                        row = _setup_row(
                            branch="exhaustion_reversal",
                            profile="sweep_reclaim_body",
                            bars=bars,
                            event_pos=event_pos,
                            signal_pos=signal_pos,
                            entry_pos=entry_pos,
                            pressure_window=window,
                            pressure_side=pressure_side,
                            trade_side=trade_side,
                            break_level=break_level,
                            opposite_level=opposite_level,
                            attack_high=attack_high,
                            attack_low=attack_low,
                            stop_price=stop,
                            target_price=target,
                            entry_price=entry_price,
                            pressure_z=float(pressure_z_values[event_pos]),
                            accumulated_notional=accumulated_notional,
                            flow_ratio=flow_ratio,
                            activity_z=activity_z,
                            persistence=persistence,
                            effectiveness=effectiveness,
                            decay_ratio=decay_ratio,
                            late_flow_share=late_flow_share,
                            price_response=price_response,
                            cfg=cfg,
                        )
                        if row is not None:
                            rows.append(row)
                            if reversal_breakbar:
                                strict = dict(row)
                                strict["profile"] = "sweep_reclaim_breakbar"
                                strict["spec_id"] = _spec_id(strict)
                                rows.append(strict)
                            if np.isfinite(decay_ratio) and decay_ratio <= float(cfg.exhaustion_decay_ratio):
                                decay = dict(row)
                                decay["profile"] = "sweep_reclaim_decay"
                                decay["spec_id"] = _spec_id(decay)
                                rows.append(decay)
                        break

                    if broke:
                        if pressure_side > 0:
                            held = signal_close > break_level
                            directional_body = signal_close > signal_open
                            touched = signal_low <= break_level * (1.0 + tol)
                            resumed = signal_close > prior_close and signal_close > float(high_values[signal_pos - 1])
                        else:
                            held = signal_close < break_level
                            directional_body = signal_close < signal_open
                            touched = signal_high >= break_level * (1.0 - tol)
                            resumed = signal_close < prior_close and signal_close < float(low_values[signal_pos - 1])
                        if held and directional_body:
                            trade_side = pressure_side
                            entry_pos = signal_pos + 1
                            entry_price = float(open_values[entry_pos])
                            if trade_side > 0:
                                invalidation = float(np.nanmin(low_values[break_pos : signal_pos + 1]))
                                stop = invalidation * (1.0 - buffer)
                                target_side = pivot_index.highs
                            else:
                                invalidation = float(np.nanmax(high_values[break_pos : signal_pos + 1]))
                                stop = invalidation * (1.0 + buffer)
                                target_side = pivot_index.lows
                            target = target_side.nearest_target(
                                available_pos=signal_pos,
                                min_pivot_pos=max(0, signal_pos - int(cfg.structure_lookback_bars)),
                                side=trade_side,
                                entry_price=entry_price,
                            )
                            if not np.isfinite(target) and np.isfinite(opposite_level):
                                measured = abs(float(break_level) - float(opposite_level))
                                target = (
                                    float(break_level) + measured
                                    if trade_side > 0
                                    else float(break_level) - measured
                                )
                            row = _setup_row(
                                branch="continuation",
                                profile="break_accept_body",
                                bars=bars,
                                event_pos=event_pos,
                                signal_pos=signal_pos,
                                entry_pos=entry_pos,
                                pressure_window=window,
                                pressure_side=pressure_side,
                                trade_side=trade_side,
                                break_level=break_level,
                                opposite_level=opposite_level,
                                attack_high=attack_high,
                                attack_low=attack_low,
                                stop_price=stop,
                                target_price=target,
                                entry_price=entry_price,
                                pressure_z=float(pressure_z_values[event_pos]),
                                accumulated_notional=accumulated_notional,
                                flow_ratio=flow_ratio,
                                activity_z=activity_z,
                                persistence=persistence,
                                effectiveness=effectiveness,
                                decay_ratio=decay_ratio,
                                late_flow_share=late_flow_share,
                                price_response=price_response,
                                cfg=cfg,
                            )
                            if row is not None:
                                rows.append(row)
                                retest_resume = signal_pos > break_pos and touched and resumed
                                if retest_resume:
                                    strict = dict(row)
                                    strict["profile"] = "break_retest_resume"
                                    strict["spec_id"] = _spec_id(strict)
                                    rows.append(strict)
                                    if (
                                        persistence >= float(cfg.continuation_min_persistence)
                                        and effectiveness >= float(cfg.continuation_min_effectiveness)
                                    ):
                                        persistent = dict(row)
                                        persistent["profile"] = "break_retest_persistent"
                                        persistent["spec_id"] = _spec_id(persistent)
                                        rows.append(persistent)
                            break
    finally:
        reporter.close()

    if not rows:
        return pd.DataFrame()
    kept = _cooldown_positions(rows, int(cfg.event_cooldown_bars))
    out = pd.DataFrame(kept).sort_values(
        ["signal_pos", "branch", "profile", "trade_side"], kind="stable"
    )
    out.insert(0, "setup_id", np.arange(1, len(out) + 1, dtype=np.int64))
    return out.reset_index(drop=True)


def _spec_id(row: dict[str, Any]) -> str:
    return f"{row['branch']}__{row['profile']}__w{int(row['pressure_window_bars'])}"


def _setup_row(
    *,
    branch: str,
    profile: str,
    bars: pd.DataFrame,
    event_pos: int,
    signal_pos: int,
    entry_pos: int,
    pressure_window: int,
    pressure_side: int,
    trade_side: int,
    break_level: float,
    opposite_level: float,
    attack_high: float,
    attack_low: float,
    stop_price: float,
    target_price: float,
    entry_price: float,
    pressure_z: float,
    accumulated_notional: float,
    flow_ratio: float,
    activity_z: float,
    persistence: float,
    effectiveness: float,
    decay_ratio: float,
    late_flow_share: float,
    price_response: float,
    cfg: AccumulatedPAConfig,
) -> dict[str, Any] | None:
    if not all(np.isfinite(v) for v in (entry_price, stop_price, target_price)) or entry_price <= 0.0:
        return None
    risk = float(trade_side) * (entry_price - stop_price) / entry_price
    reward = float(trade_side) * (target_price - entry_price) / entry_price
    risk_bps = risk * 10_000.0
    reward_bps = reward * 10_000.0
    if risk <= 0.0 or reward <= 0.0:
        return None
    if risk_bps < float(cfg.min_risk_bps) or risk_bps > float(cfg.max_risk_bps):
        return None
    rr = reward / risk
    if rr < float(cfg.min_reward_risk):
        return None
    row = {
        "branch": branch,
        "profile": profile,
        "pressure_window_bars": int(pressure_window),
        "pressure_side": int(pressure_side),
        "trade_side": int(trade_side),
        "side_name": "LONG" if trade_side > 0 else "SHORT",
        "event_pos": int(event_pos),
        "signal_pos": int(signal_pos),
        "entry_pos": int(entry_pos),
        "event_time": bars.index[event_pos],
        "signal_time": bars.index[signal_pos],
        "entry_time": bars.index[entry_pos],
        "entry_price": float(entry_price),
        "break_level": float(break_level),
        "opposite_level": float(opposite_level) if np.isfinite(opposite_level) else np.nan,
        "attack_high": float(attack_high),
        "attack_low": float(attack_low),
        "stop_price": float(stop_price),
        "target_price": float(target_price),
        "risk_bps": float(risk_bps),
        "reward_bps": float(reward_bps),
        "reward_risk": float(rr),
        "pressure_z": float(pressure_z),
        "accumulated_notional": float(accumulated_notional),
        "flow_ratio": float(flow_ratio),
        "activity_z": float(activity_z),
        "flow_persistence": float(persistence),
        "pressure_effectiveness": float(effectiveness),
        "impact_decay_ratio": float(decay_ratio),
        "late_directional_flow_share": float(late_flow_share),
        "price_response": float(price_response),
    }
    row["spec_id"] = _spec_id(row)
    return row


def simulate_structural_exits(
    bars: pd.DataFrame,
    setups: pd.DataFrame,
    *,
    normal_cost: float,
    fee_only_cost: float,
    max_holding_bars: int,
    progress_enabled: bool = True,
) -> pd.DataFrame:
    """Simulate structural target/stop with vectorized local first-touch scans."""
    if setups.empty:
        return setups.copy()
    high = _numeric(bars, "high").to_numpy(dtype=float)
    low = _numeric(bars, "low").to_numpy(dtype=float)
    close = _numeric(bars, "close").to_numpy(dtype=float)
    index = pd.DatetimeIndex(bars.index)
    rows: list[dict[str, Any]] = []
    total = int(len(setups))
    reporter = ProgressReporter(
        "[pa-exit] structural first-touch",
        total,
        every=max(1, total // 100),
        enabled=progress_enabled,
    )
    try:
        for done, setup in enumerate(setups.itertuples(index=False), start=1):
            entry_pos = int(setup.entry_pos)
            side = int(setup.trade_side)
            stop = float(setup.stop_price)
            target = float(setup.target_price)
            end_pos = min(len(bars) - 1, entry_pos + int(max_holding_bars))
            high_path = high[entry_pos : end_pos + 1]
            low_path = low[entry_pos : end_pos + 1]

            if side > 0:
                stop_offsets = np.flatnonzero(low_path <= stop)
                target_offsets = np.flatnonzero(high_path >= target)
            else:
                stop_offsets = np.flatnonzero(high_path >= stop)
                target_offsets = np.flatnonzero(low_path <= target)

            first_stop = int(stop_offsets[0]) if stop_offsets.size else None
            first_target = int(target_offsets[0]) if target_offsets.size else None
            exit_pos = end_pos
            exit_price = float(close[end_pos])
            reason = "safety_timeout"
            if first_stop is not None and (first_target is None or first_stop <= first_target):
                exit_pos = entry_pos + first_stop
                exit_price = stop
                reason = (
                    "stop_same_bar_conservative"
                    if first_target is not None and first_stop == first_target
                    else "structure_stop"
                )
            elif first_target is not None:
                exit_pos = entry_pos + first_target
                exit_price = target
                reason = "structure_target"

            gross = side * (exit_price / float(setup.entry_price) - 1.0)
            row = setup._asdict()
            row.update(
                {
                    "exit_pos": int(exit_pos),
                    "exit_time": index[exit_pos],
                    "exit_price": float(exit_price),
                    "exit_reason": reason,
                    "holding_bars": int(exit_pos - entry_pos + 1),
                    "gross_return": float(gross),
                    "fee_only_return": float(gross - fee_only_cost),
                    "net_return": float(gross - normal_cost),
                    "resolved_by_structure_flag": bool(reason != "safety_timeout"),
                }
            )
            rows.append(row)
            reporter.update(done)
    finally:
        reporter.close()
    return pd.DataFrame(rows)


def resolve_position_conflicts(trades: pd.DataFrame) -> pd.DataFrame:
    """Keep at most one open trade globally; rank same-time setups by PA RR."""
    if trades.empty:
        return trades.copy()
    ordered = trades.sort_values(
        ["entry_pos", "reward_risk", "pressure_z"],
        ascending=[True, False, False],
        kind="stable",
    )
    rows: list[pd.Series] = []
    last_exit = -1
    for _, row in ordered.iterrows():
        if int(row["entry_pos"]) <= int(last_exit):
            continue
        rows.append(row)
        last_exit = int(row["exit_pos"])
    if not rows:
        return ordered.iloc[0:0].copy()
    return pd.DataFrame(rows).sort_values("entry_pos", kind="stable").reset_index(drop=True)
