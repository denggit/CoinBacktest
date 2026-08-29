#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R13 reversal-quality and causal entry-discovery helpers.

R13 deliberately compares *direct* opposite-liquidity delivery with paths
whose deeper same-side completed-trend liquidity is reached first.  A later
opposite delivery after a cascade is therefore a failure for the direct
reversal thesis, even though it may remain interesting for another sleeve.

All early-response features are computed from bars that have closed by their
declared ``available_time``.  Entry signals execute on the next eligible bar;
same-bar TP/SL ambiguity is resolved pessimistically as a stop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.core import aggregate_bars, normalize_1m_bars
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex

EPS = 1e-12


@dataclass(frozen=True)
class R13Config:
    discovery_end: pd.Timestamp = pd.Timestamp("2024-12-31 23:59:59")
    # July 2025 is a 30-day embargo so validation labels cannot consume any
    # price path from the holdout beginning on 2025-08-01.
    validation_end: pd.Timestamp = pd.Timestamp("2025-06-30 23:59:59")
    holdout_start: pd.Timestamp = pd.Timestamp("2025-08-01 00:00:00")
    early_windows_minutes: tuple[int, ...] = (5, 15, 30, 60)
    landmark_max_minutes: int = 360
    limit_wait_minutes: int = 180
    atr_minutes: int = 60
    market_roundtrip_cost: float = 0.0011
    limit_roundtrip_cost: float = 0.0008
    cost_scales: tuple[float, ...] = (1.0, 2.0, 3.0)

    def validate(self) -> "R13Config":
        if self.discovery_end >= self.holdout_start:
            raise ValueError("discovery_end must precede holdout_start")
        if self.validation_end >= self.holdout_start:
            raise ValueError("validation_end must precede holdout_start")
        if self.discovery_end >= self.validation_end:
            raise ValueError("discovery_end must precede validation_end")
        if not self.early_windows_minutes or any(int(x) <= 0 for x in self.early_windows_minutes):
            raise ValueError("early windows must be positive")
        if self.landmark_max_minutes <= 0 or self.limit_wait_minutes <= 0 or self.atr_minutes <= 1:
            raise ValueError("landmark, limit-wait and ATR windows must be positive")
        if any(float(x) <= 0 for x in self.cost_scales):
            raise ValueError("cost scales must be positive")
        return self


TIME_COLUMNS = (
    "root_sweep_time", "root_sweep_available_time", "path_start_time",
    "opposite_1_touch_time", "deeper_same_side_touch_time", "next_open_time",
    "reclaim_available_time", "post_sweep_st_mss_1m_available_time",
    "post_sweep_st_mss_2m_available_time", "post_sweep_st_mss_5m_available_time",
    "first_directional_fvg_available_time",
)


def data_coverage_audit(
    bars_1m: pd.DataFrame,
    *,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
) -> pd.DataFrame:
    """Describe actual 1m coverage; never infer coverage from a manifest."""
    b = normalize_1m_bars(bars_1m)
    if b.empty:
        return pd.DataFrame([{"check": "bare_1m_rows", "value": 0}])
    actual_start = pd.Timestamp(b.index.min())
    actual_end = pd.Timestamp(b.index.max())
    expected = max(0, int((requested_end - requested_start) / pd.Timedelta(minutes=1)) + 1)
    in_range = b.loc[(b.index >= requested_start) & (b.index <= requested_end)]
    gaps = in_range.index.to_series().diff().gt(pd.Timedelta(minutes=1))
    return pd.DataFrame([
        {"check": "requested_start", "value": requested_start},
        {"check": "requested_end", "value": requested_end},
        {"check": "actual_start", "value": actual_start},
        {"check": "actual_end", "value": actual_end},
        {"check": "bare_1m_rows_in_requested_range", "value": len(in_range)},
        {"check": "expected_1m_rows", "value": expected},
        {"check": "coverage_ratio", "value": len(in_range) / expected if expected else np.nan},
        {"check": "internal_gap_count", "value": int(gaps.sum())},
        {"check": "requested_end_covered", "value": int(actual_end >= requested_end.floor("min"))},
    ])


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame.get(col, pd.Series(np.nan, index=frame.index)), errors="coerce")


def prepare_reversal_comparison_universe(
    r12_paths: pd.DataFrame,
    *,
    config: R13Config | None = None,
    include_holdout: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the direct-vs-same-side-first universe and a holdout seal audit."""
    cfg = (config or R13Config()).validate()
    q = r12_paths.copy()
    for col in TIME_COLUMNS:
        if col in q:
            q[col] = pd.to_datetime(q[col], errors="coerce")
    q = q.dropna(subset=["root_event_id", "root_sweep_time", "root_side", "next_open_price"])
    q = q.loc[q["root_side"].isin(["SSL", "BSL"])]
    q = q.loc[_num(q, "same_bar_two_sided_root_flag").fillna(0).eq(0)]
    q = q.loc[_num(q, "opposite_1_available_flag").fillna(0).eq(1)]
    q = q.loc[_num(q, "deeper_same_side_available_flag").fillna(0).eq(1)]
    direct = q["path_outcome"].eq("direct_opposite_delivery")
    same_first = q["path_outcome"].isin([
        "cascade_then_opposite_delivery", "same_side_continuation_no_opposite_hit"
    ])
    q = q.loc[direct | same_first].copy()
    q["direct_reversal_label"] = q["path_outcome"].eq("direct_opposite_delivery").astype(int)
    q["comparison_class"] = np.where(q["direct_reversal_label"].eq(1), "direct_reversal", "same_side_first_failure")
    q["year"] = q["root_sweep_time"].dt.year.astype(int)
    q["research_split"] = np.select(
        [
            q["root_sweep_time"].le(cfg.discovery_end),
            q["root_sweep_time"].le(cfg.validation_end),
            q["root_sweep_time"].lt(cfg.holdout_start),
        ],
        ["discovery", "validation", "embargo"], default="sealed_holdout",
    )
    entry = _num(q, "next_open_price")
    target = _num(q, "opposite_1_touch_price")
    stop = _num(q, "deeper_same_side_touch_price")
    is_long = q["root_side"].eq("SSL")
    q["root_target_distance_pct"] = np.where(is_long, target / entry - 1.0, entry / target - 1.0)
    q["root_structural_risk_pct"] = np.where(is_long, 1.0 - stop / entry, stop / entry - 1.0)
    q["root_structural_rr"] = q["root_target_distance_pct"] / q["root_structural_risk_pct"].where(q["root_structural_risk_pct"].gt(EPS))
    q["root_gross_first_passage_return"] = np.where(
        q["direct_reversal_label"].eq(1), q["root_target_distance_pct"], -q["root_structural_risk_pct"]
    )
    for scale in cfg.cost_scales:
        q[f"root_net_return_cost{scale:g}x"] = q["root_gross_first_passage_return"] - cfg.market_roundtrip_cost * float(scale)
    seal = pd.DataFrame([{
        "holdout_start": cfg.holdout_start,
        "available_holdout_rows_in_r12": int(q["research_split"].eq("sealed_holdout").sum()),
        "included_in_r13_outputs": int(include_holdout),
        "status": "UNSEALED_FOR_EXPLICIT_EVALUATION" if include_holdout else "SEALED_FROM_R13_RULE_DISCOVERY",
        "qualification": "not pristine relative to R12 aggregate path atlas; untouched for R13 feature/rule selection",
    }])
    allowed_splits = ["discovery", "validation"] + (["sealed_holdout"] if include_holdout else [])
    q = q.loc[q["research_split"].isin(allowed_splits)].copy()
    valid_geometry = q["root_target_distance_pct"].gt(EPS) & q["root_structural_risk_pct"].gt(EPS)
    q = q.loc[valid_geometry].sort_values(["root_sweep_time", "root_side"], kind="stable").reset_index(drop=True)
    return q, seal


def _atr_before_root(b: pd.DataFrame, minutes: int) -> np.ndarray:
    prev = pd.to_numeric(b["close"], errors="coerce").shift(1)
    tr = pd.concat([
        pd.to_numeric(b["high"], errors="coerce") - pd.to_numeric(b["low"], errors="coerce"),
        (pd.to_numeric(b["high"], errors="coerce") - prev).abs(),
        (pd.to_numeric(b["low"], errors="coerce") - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.shift(1).rolling(int(minutes), min_periods=max(10, int(minutes) // 2)).mean().to_numpy(float)


def _directional_excursions(high: np.ndarray, low: np.ndarray, entry: float, direction: int) -> tuple[float, float]:
    if direction == 1:
        return float(np.nanmax(high) / entry - 1.0), float(1.0 - np.nanmin(low) / entry)
    return float(entry / np.nanmin(low) - 1.0), float(np.nanmax(high) / entry - 1.0)


def _fvg_stats(high: np.ndarray, low: np.ndarray, start: int, end: int, direction: int, atr: float) -> tuple[int, float, int, float, float]:
    count = 0
    max_width = np.nan
    first = -1
    first_low = np.nan
    first_high = np.nan
    for i in range(max(2, start), end + 1):
        if direction == 1 and low[i] > high[i - 2] + EPS:
            lo, hi = float(high[i - 2]), float(low[i])
        elif direction == -1 and high[i] < low[i - 2] - EPS:
            lo, hi = float(high[i]), float(low[i - 2])
        else:
            continue
        count += 1
        width = (hi - lo) / atr if atr > EPS else np.nan
        max_width = width if pd.isna(max_width) else max(float(max_width), float(width))
        if first < 0:
            first, first_low, first_high = i, lo, hi
    return count, max_width, first, first_low, first_high


def _first_mss_quality(
    htf: pd.DataFrame,
    *,
    sweep_time: pd.Timestamp,
    side: str,
    minutes: int,
    max_minutes: int,
    root_extreme: float,
    atr: float,
) -> dict[str, object]:
    empty = {
        "available_time": pd.NaT, "level": np.nan, "delay_min": np.nan,
        "swing_excursion_atr": np.nan, "break_distance_atr": np.nan,
        "break_body_atr": np.nan, "break_body_ratio": np.nan,
        "displacement_atr": np.nan, "path_efficiency": np.nan,
        "fvg_count_to_break": 0, "max_fvg_width_atr_to_break": np.nan,
    }
    if htf.empty:
        return empty
    delta = pd.Timedelta(minutes=int(minutes))
    end_time = sweep_time + pd.Timedelta(minutes=int(max_minutes))
    q = htf.loc[(htf.index >= sweep_time.floor(f"{minutes}min")) & (htf.index <= end_time)]
    if len(q) < 5:
        return empty
    highs = _num(q, "high").to_numpy(float)
    lows = _num(q, "low").to_numpy(float)
    opens = _num(q, "open").to_numpy(float)
    closes = _num(q, "close").to_numpy(float)
    if side == "SSL":
        vals = highs
        pivots = np.flatnonzero((vals[1:-1] > vals[:-2]) & (vals[1:-1] > vals[2:])) + 1
        breaks = lambda x, level: x > level + EPS
        direction = 1
    else:
        vals = lows
        pivots = np.flatnonzero((vals[1:-1] < vals[:-2]) & (vals[1:-1] < vals[2:])) + 1
        breaks = lambda x, level: x < level - EPS
        direction = -1
    for pivot in pivots:
        if q.index[pivot] < sweep_time:
            continue
        pivot_available = q.index[pivot + 1] + delta
        j0 = int(q.index.searchsorted(pivot_available, side="left"))
        for j in range(j0, len(q)):
            level = float(vals[pivot])
            if not breaks(float(closes[j]), level):
                continue
            available = pd.Timestamp(q.index[j] + delta)
            break_distance = direction * (float(closes[j]) - level) / atr if atr > EPS else np.nan
            swing_excursion = direction * (level - root_extreme) / atr if atr > EPS else np.nan
            body = abs(float(closes[j]) - float(opens[j]))
            rng = float(highs[j] - lows[j])
            displacement = direction * (float(closes[j]) - root_extreme) / atr if atr > EPS else np.nan
            path = closes[: j + 1]
            total = float(np.nansum(np.abs(np.diff(path))))
            efficiency = direction * (float(path[-1]) - float(path[0])) / total if total > EPS else np.nan
            fvg_count, fvg_width, _, _, _ = _fvg_stats(highs, lows, 2, j, direction, atr)
            return {
                "available_time": available, "level": level,
                "delay_min": (available - sweep_time).total_seconds() / 60.0,
                "swing_excursion_atr": swing_excursion,
                "break_distance_atr": break_distance,
                "break_body_atr": body / atr if atr > EPS else np.nan,
                "break_body_ratio": body / rng if rng > EPS else np.nan,
                "displacement_atr": displacement,
                "path_efficiency": efficiency,
                "fvg_count_to_break": int(fvg_count),
                "max_fvg_width_atr_to_break": fvg_width,
            }
    return empty


def attach_reversal_quality_features(
    bars_1m: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    config: R13Config | None = None,
) -> pd.DataFrame:
    """Attach sweep morphology, expected response, reclaim, MSS and FVG quality."""
    cfg = (config or R13Config()).validate()
    if universe.empty:
        return universe.copy()
    b = normalize_1m_bars(bars_1m)
    idx = b.index
    opens = _num(b, "open").to_numpy(float)
    highs = _num(b, "high").to_numpy(float)
    lows = _num(b, "low").to_numpy(float)
    closes = _num(b, "close").to_numpy(float)
    atrs = _atr_before_root(b, cfg.atr_minutes)
    htf = {m: aggregate_bars(b, m) for m in (1, 2, 5)}
    records: list[dict[str, object]] = []
    for r in universe.itertuples(index=False):
        row = r._asdict()
        root_time = pd.Timestamp(r.root_sweep_time)
        pos = int(idx.searchsorted(root_time, side="left"))
        if pos >= len(b) or idx[pos] != root_time:
            continue
        direction = 1 if str(r.root_side) == "SSL" else -1
        entry_pos = pos + 1
        if entry_pos >= len(b):
            continue
        entry = float(opens[entry_pos])
        atr = float(atrs[pos]) if np.isfinite(atrs[pos]) else float(highs[pos] - lows[pos])
        row["pre_root_atr_60m"] = atr
        row["root_range_atr"] = (float(highs[pos]) - float(lows[pos])) / atr if atr > EPS else np.nan
        row["root_penetration_atr"] = abs(float(r.root_sweep_depth_bps)) * entry / 10000.0 / atr if atr > EPS else np.nan
        root_extreme = float(lows[pos]) if direction == 1 else float(highs[pos])
        target_distance = float(r.root_target_distance_pct)
        risk_distance = float(r.root_structural_risk_pct)
        for window in cfg.early_windows_minutes:
            end = entry_pos + int(window) - 1
            prefix = f"early_{int(window)}m"
            row[f"{prefix}_available_time"] = idx[end] + pd.Timedelta(minutes=1) if end < len(b) else pd.NaT
            if end >= len(b):
                for suffix in ("close_progress", "mfe_pct", "mae_pct", "target_progress", "risk_used", "mfe_mae_ratio", "path_efficiency", "directional_close_share", "outside_close_share", "max_body_atr", "max_range_atr", "fvg_count", "max_fvg_width_atr"):
                    row[f"{prefix}_{suffix}"] = np.nan
                continue
            seg_open = opens[entry_pos : end + 1]
            seg_high = highs[entry_pos : end + 1]
            seg_low = lows[entry_pos : end + 1]
            seg_close = closes[entry_pos : end + 1]
            mfe, mae = _directional_excursions(seg_high, seg_low, entry, direction)
            close_progress = (float(seg_close[-1]) / entry - 1.0) if direction == 1 else (entry / float(seg_close[-1]) - 1.0)
            steps = np.diff(np.r_[entry, seg_close])
            total = float(np.nansum(np.abs(steps)))
            signed_net = direction * (float(seg_close[-1]) - entry)
            if direction == 1:
                directional = seg_close > seg_open
                outside = seg_close <= float(r.root_zone_high)
            else:
                directional = seg_close < seg_open
                outside = seg_close >= float(r.root_zone_low)
            bodies = np.abs(seg_close - seg_open)
            ranges = seg_high - seg_low
            count, max_width, _, _, _ = _fvg_stats(highs, lows, entry_pos, end, direction, atr)
            row[f"{prefix}_close_progress"] = close_progress
            row[f"{prefix}_mfe_pct"] = mfe
            row[f"{prefix}_mae_pct"] = mae
            row[f"{prefix}_target_progress"] = mfe / target_distance if target_distance > EPS else np.nan
            row[f"{prefix}_risk_used"] = mae / risk_distance if risk_distance > EPS else np.nan
            row[f"{prefix}_mfe_mae_ratio"] = mfe / mae if mae > EPS else np.nan
            row[f"{prefix}_path_efficiency"] = signed_net / total if total > EPS else np.nan
            row[f"{prefix}_directional_close_share"] = float(np.mean(directional))
            row[f"{prefix}_outside_close_share"] = float(np.mean(outside))
            row[f"{prefix}_max_body_atr"] = float(np.nanmax(bodies)) / atr if atr > EPS else np.nan
            row[f"{prefix}_max_range_atr"] = float(np.nanmax(ranges)) / atr if atr > EPS else np.nan
            row[f"{prefix}_fvg_count"] = int(count)
            row[f"{prefix}_max_fvg_width_atr"] = max_width
        reclaim_time = pd.Timestamp(r.reclaim_available_time) if pd.notna(r.reclaim_available_time) else pd.NaT
        for window in (15, 30, 60):
            col = f"reclaim_{window}m_retained_share"
            if pd.isna(reclaim_time):
                row[col] = np.nan
                continue
            rp = int(idx.searchsorted(reclaim_time, side="left"))
            rend = min(len(b), rp + window)
            if rp >= len(b) or rend <= rp:
                row[col] = np.nan
            elif direction == 1:
                row[col] = float(np.mean(closes[rp:rend] > float(r.root_zone_high)))
            else:
                row[col] = float(np.mean(closes[rp:rend] < float(r.root_zone_low)))
        for minutes in (1, 2, 5):
            quality = _first_mss_quality(
                htf[minutes], sweep_time=root_time, side=str(r.root_side), minutes=minutes,
                max_minutes=cfg.landmark_max_minutes, root_extreme=root_extreme, atr=atr,
            )
            for key, value in quality.items():
                row[f"mss_{minutes}m_{key}"] = value
        fvg_end = min(len(b) - 1, pos + cfg.landmark_max_minutes)
        count, width, first, flo, fhi = _fvg_stats(highs, lows, entry_pos, fvg_end, direction, atr)
        row["directional_fvg_count_360m"] = int(count)
        row["directional_fvg_max_width_atr_360m"] = width
        row["first_fvg_available_time"] = idx[first] + pd.Timedelta(minutes=1) if first >= 0 else pd.NaT
        row["first_fvg_delay_min"] = (idx[first] - root_time).total_seconds() / 60.0 + 1 if first >= 0 else np.nan
        row["first_fvg_low"] = flo
        row["first_fvg_high"] = fhi
        records.append(row)
    return pd.DataFrame(records).sort_values(["root_sweep_time", "root_side"], kind="stable").reset_index(drop=True)


def _first_barrier(
    high_tree: SegmentThresholdIndex,
    low_tree: SegmentThresholdIndex,
    *,
    direction: int,
    target: float,
    stop: float,
    start: int,
    end: int,
    target_allowed_on_start: bool = True,
) -> tuple[str, int]:
    if start > end:
        return "censored", -1
    if direction == 1:
        tp = int(high_tree.first_geq(start, end, target))
        sl = int(low_tree.first_leq(start, end, stop))
    else:
        tp = int(low_tree.first_leq(start, end, target))
        sl = int(high_tree.first_geq(start, end, stop))
    if not target_allowed_on_start and tp == start:
        if start + 1 <= end:
            tp = int(high_tree.first_geq(start + 1, end, target)) if direction == 1 else int(low_tree.first_leq(start + 1, end, target))
        else:
            tp = -1
    if sl >= 0 and (tp < 0 or sl <= tp):
        return "sl_first", sl
    if tp >= 0:
        return "tp_first", tp
    return "censored", -1


def _signal_entry_pos(index: pd.DatetimeIndex, signal_available: object) -> int:
    if signal_available is None or pd.isna(signal_available):
        return -1
    return int(index.searchsorted(pd.Timestamp(signal_available), side="left"))


def _first_fvg_after(
    high: np.ndarray,
    low: np.ndarray,
    index: pd.DatetimeIndex,
    *,
    direction: int,
    start_available: pd.Timestamp,
    end: int,
) -> tuple[int, float, float]:
    start = max(2, int(index.searchsorted(start_available, side="left")))
    for i in range(start, end + 1):
        if direction == 1 and low[i] > high[i - 2] + EPS:
            return i, float(high[i - 2]), float(low[i])
        if direction == -1 and high[i] < low[i - 2] - EPS:
            return i, float(high[i]), float(low[i - 2])
    return -1, np.nan, np.nan


def build_entry_candidate_outcomes(
    bars_1m: pd.DataFrame,
    features: pd.DataFrame,
    *,
    config: R13Config | None = None,
) -> pd.DataFrame:
    """Compare a small predeclared set of causal market and FVG-limit entries."""
    cfg = (config or R13Config()).validate()
    if features.empty:
        return pd.DataFrame()
    b = normalize_1m_bars(bars_1m)
    idx = b.index
    high = _num(b, "high").to_numpy(float)
    low = _num(b, "low").to_numpy(float)
    open_ = _num(b, "open").to_numpy(float)
    high_tree = SegmentThresholdIndex(high)
    low_tree = SegmentThresholdIndex(low)
    rows: list[dict[str, object]] = []
    market_models = (
        ("root_next_open", "root_sweep_available_time", None),
        ("same_bar_reclaim", "root_sweep_available_time", "root_same_bar_full_reclaim_flag"),
        ("response_15m_market", "early_15m_available_time", None),
        ("reclaim_market", "reclaim_available_time", None),
        ("fvg_market", "first_fvg_available_time", None),
        ("mss_1m_market", "mss_1m_available_time", None),
        ("mss_2m_market", "mss_2m_available_time", None),
        ("mss_5m_market", "mss_5m_available_time", None),
    )
    for r in features.itertuples(index=False):
        base = r._asdict()
        root_pos = int(idx.searchsorted(pd.Timestamp(r.root_sweep_time), side="left"))
        direction = 1 if str(r.root_side) == "SSL" else -1
        target = float(r.opposite_1_touch_price)
        stop = float(r.deeper_same_side_touch_price)
        horizon_end = min(len(b) - 1, root_pos + int(r.path_horizon_minutes))
        first_path = root_pos + 1
        for model, time_col, required_flag in market_models:
            if required_flag and int(float(base.get(required_flag, 0) or 0)) != 1:
                continue
            signal_time = base.get(time_col)
            entry_pos = _signal_entry_pos(idx, signal_time)
            rec = {k: base.get(k) for k in (
                "root_event_id", "root_sweep_time", "root_side", "research_split", "year",
                "comparison_class", "direct_reversal_label", "root_structural_rr",
            )}
            rec.update({"entry_model": model, "entry_kind": "market", "signal_available_time": signal_time})
            if entry_pos < first_path or entry_pos > horizon_end or entry_pos >= len(b):
                rec.update({"entry_status": "no_causal_signal", "outcome": "no_entry"})
                rows.append(rec)
                continue
            stale, _ = _first_barrier(high_tree, low_tree, direction=direction, target=target, stop=stop, start=first_path, end=entry_pos - 1)
            if stale != "censored":
                rec.update({"entry_status": "barrier_before_entry", "outcome": "stale"})
                rows.append(rec)
                continue
            entry = float(open_[entry_pos])
            result, exit_pos = _first_barrier(high_tree, low_tree, direction=direction, target=target, stop=stop, start=entry_pos, end=horizon_end)
            rec.update(_entry_result_record(
                b, direction=direction, entry_pos=entry_pos, entry=entry, target=target, stop=stop,
                outcome=result, exit_pos=exit_pos, cost=cfg.market_roundtrip_cost, cost_scales=cfg.cost_scales,
            ))
            rows.append(rec)
        for model, anchor_col in (("reclaim_fvg_proximal_limit", "reclaim_available_time"), ("mss_2m_fvg_proximal_limit", "mss_2m_available_time")):
            anchor = base.get(anchor_col)
            rec = {k: base.get(k) for k in (
                "root_event_id", "root_sweep_time", "root_side", "research_split", "year",
                "comparison_class", "direct_reversal_label", "root_structural_rr",
            )}
            rec.update({"entry_model": model, "entry_kind": "fvg_limit", "signal_available_time": pd.NaT})
            if anchor is None or pd.isna(anchor):
                rec.update({"entry_status": "no_causal_signal", "outcome": "no_entry"}); rows.append(rec); continue
            fvg_end = min(horizon_end, int(idx.searchsorted(pd.Timestamp(anchor), side="left")) + cfg.landmark_max_minutes)
            fvg_pos, fvg_low, fvg_high = _first_fvg_after(high, low, idx, direction=direction, start_available=pd.Timestamp(anchor), end=fvg_end)
            if fvg_pos < 0:
                rec.update({"entry_status": "no_post_confirmation_fvg", "outcome": "no_entry"}); rows.append(rec); continue
            active_pos = fvg_pos + 1
            signal_available = idx[fvg_pos] + pd.Timedelta(minutes=1)
            rec["signal_available_time"] = signal_available
            rec["fvg_low"] = fvg_low; rec["fvg_high"] = fvg_high
            limit = fvg_high if direction == 1 else fvg_low
            fill_end = min(horizon_end, active_pos + cfg.limit_wait_minutes - 1)
            stale, _ = _first_barrier(high_tree, low_tree, direction=direction, target=target, stop=stop, start=first_path, end=active_pos - 1)
            if stale != "censored":
                rec.update({"entry_status": "barrier_before_order_active", "outcome": "stale"}); rows.append(rec); continue
            fill = int(low_tree.first_leq(active_pos, fill_end, limit)) if direction == 1 else int(high_tree.first_geq(active_pos, fill_end, limit))
            if fill < 0:
                rec.update({"entry_status": "limit_unfilled", "outcome": "unfilled", "entry_price": limit}); rows.append(rec); continue
            before_fill, _ = _first_barrier(high_tree, low_tree, direction=direction, target=target, stop=stop, start=active_pos, end=fill - 1)
            if before_fill != "censored":
                rec.update({"entry_status": "barrier_before_fill", "outcome": "stale", "entry_price": limit}); rows.append(rec); continue
            result, exit_pos = _first_barrier(
                high_tree, low_tree, direction=direction, target=target, stop=stop,
                start=fill, end=horizon_end, target_allowed_on_start=False,
            )
            rec.update(_entry_result_record(
                b, direction=direction, entry_pos=fill, entry=limit, target=target, stop=stop,
                outcome=result, exit_pos=exit_pos, cost=cfg.limit_roundtrip_cost, cost_scales=cfg.cost_scales,
            ))
            rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["entry_time"] = pd.to_datetime(out.get("entry_time"), errors="coerce")
        out["signal_available_time"] = pd.to_datetime(out.get("signal_available_time"), errors="coerce")
    return out


def _entry_result_record(
    bars: pd.DataFrame,
    *,
    direction: int,
    entry_pos: int,
    entry: float,
    target: float,
    stop: float,
    outcome: str,
    exit_pos: int,
    cost: float,
    cost_scales: Sequence[float],
) -> dict[str, object]:
    target_ret = (target / entry - 1.0) if direction == 1 else (entry / target - 1.0)
    risk = (1.0 - stop / entry) if direction == 1 else (stop / entry - 1.0)
    valid = target_ret > EPS and risk > EPS
    if not valid:
        return {"entry_status": "invalid_entry_geometry", "outcome": "no_entry", "entry_price": entry}
    gross = target_ret if outcome == "tp_first" else (-risk if outcome == "sl_first" else np.nan)
    end = exit_pos if exit_pos >= 0 else entry_pos
    seg = bars.iloc[entry_pos : end + 1]
    mfe, mae = _directional_excursions(
        _num(seg, "high").to_numpy(float), _num(seg, "low").to_numpy(float), entry, direction
    )
    rec: dict[str, object] = {
        "entry_status": "filled", "outcome": outcome,
        "entry_time": bars.index[entry_pos], "entry_price": entry,
        "target_price": target, "stop_price": stop,
        "risk_distance_pct": risk, "target_distance_pct": target_ret,
        "structural_rr": target_ret / risk,
        "exit_time": bars.index[exit_pos] if exit_pos >= 0 else pd.NaT,
        "holding_minutes": float(exit_pos - entry_pos) if exit_pos >= 0 else np.nan,
        "gross_return": gross, "gross_r": gross / risk if pd.notna(gross) else np.nan,
        "mfe_pct": mfe, "mae_pct": mae,
    }
    for scale in cost_scales:
        net = gross - float(cost) * float(scale) if pd.notna(gross) else np.nan
        rec[f"net_return_cost{float(scale):g}x"] = net
        rec[f"net_r_cost{float(scale):g}x"] = net / risk if pd.notna(net) else np.nan
    return rec


def _profit_factor(values: Iterable[float]) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    gp = float(x[x > 0].sum()); gl = float(-x[x < 0].sum())
    return gp / gl if gl > EPS else (np.inf if gp > EPS else np.nan)


def summarize_entry_models(entries: pd.DataFrame, *, cost_scale: float = 2.0) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame()
    net_col = f"net_return_cost{float(cost_scale):g}x"
    r_col = f"net_r_cost{float(cost_scale):g}x"
    rows: list[dict[str, object]] = []
    for key, p in entries.groupby(["research_split", "root_side", "entry_model"], dropna=False, sort=True):
        filled = p.loc[p["entry_status"].eq("filled")]
        resolved = filled.loc[filled["outcome"].isin(["tp_first", "sl_first"])]
        net = pd.to_numeric(resolved.get(net_col), errors="coerce").dropna()
        nr = pd.to_numeric(resolved.get(r_col), errors="coerce").dropna()
        rows.append({
            "research_split": key[0], "root_side": key[1], "entry_model": key[2],
            "opportunities": len(p), "filled": len(filled), "fill_rate": len(filled) / len(p) if len(p) else np.nan,
            "stale_or_barrier_before_entry": int(p["outcome"].eq("stale").sum()),
            "unfilled": int(p["outcome"].eq("unfilled").sum()), "resolved": len(resolved),
            "tp_first": int(resolved["outcome"].eq("tp_first").sum()),
            "sl_first": int(resolved["outcome"].eq("sl_first").sum()),
            "tp_before_sl_rate": float(resolved["outcome"].eq("tp_first").mean()) if len(resolved) else np.nan,
            "mean_net_return_cost2x": float(net.mean()) if len(net) else np.nan,
            "net_pf_cost2x": _profit_factor(net), "mean_net_r_cost2x": float(nr.mean()) if len(nr) else np.nan,
            "r_pf_cost2x": _profit_factor(nr),
            "median_risk_distance_pct": _num(resolved, "risk_distance_pct").median(),
            "median_structural_rr": _num(resolved, "structural_rr").median(),
            "median_mae_pct": _num(resolved, "mae_pct").median(),
            "median_mfe_pct": _num(resolved, "mfe_pct").median(),
            "median_holding_minutes": _num(resolved, "holding_minutes").median(),
        })
    return pd.DataFrame(rows)


def summarize_entry_years(entries: pd.DataFrame, *, cost_scale: float = 2.0) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame()
    net_col = f"net_return_cost{float(cost_scale):g}x"
    rows = []
    filled = entries.loc[entries["entry_status"].eq("filled") & entries["outcome"].isin(["tp_first", "sl_first"])]
    for key, p in filled.groupby(["root_side", "entry_model", "year"], dropna=False, sort=True):
        v = pd.to_numeric(p[net_col], errors="coerce").dropna()
        rows.append({"root_side": key[0], "entry_model": key[1], "year": key[2], "trades": len(p),
                     "tp_before_sl_rate": p["outcome"].eq("tp_first").mean(), "mean_net_return_cost2x": v.mean(),
                     "net_pf_cost2x": _profit_factor(v)})
    return pd.DataFrame(rows)


DEFAULT_BIN_FEATURES = (
    "root_oldest_age_days", "root_sweep_depth_bps", "root_range_atr", "root_rejection_wick_share",
    "root_reversal_close_location", "root_target_distance_pct", "root_structural_risk_pct", "root_structural_rr",
    "pre_sweep_ret_15m", "early_15m_target_progress", "early_15m_risk_used", "early_15m_mfe_mae_ratio",
    "early_15m_path_efficiency", "early_15m_outside_close_share", "early_15m_max_body_atr",
    "reclaim_delay_min", "reclaim_30m_retained_share", "mss_1m_delay_min", "mss_1m_break_distance_atr",
    "mss_1m_break_body_atr", "mss_1m_displacement_atr", "mss_1m_path_efficiency",
    "first_fvg_delay_min", "directional_fvg_max_width_atr_360m",
)


def build_feature_bin_atlas(
    features: pd.DataFrame,
    *,
    entries: pd.DataFrame | None = None,
    feature_names: Sequence[str] = DEFAULT_BIN_FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Freeze discovery quartiles and evaluate from each feature's causal entry.

    Root/pre-sweep features may use root-next-open.  A 15-minute response
    feature uses the response-15m next-open; reclaim, MSS and FVG features use
    their own confirmation entry.  This prevents an early-response bin from
    being credited with an entry that occurred before the feature existed.
    """
    if features.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    definitions: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    mono: list[dict[str, object]] = []
    entry_map: dict[str, pd.DataFrame] = {}
    if entries is not None and not entries.empty:
        for model, part in entries.groupby("entry_model", sort=False):
            cols = ["root_event_id", "entry_status", "outcome", "net_return_cost2x"]
            entry_map[str(model)] = part.loc[:, [c for c in cols if c in part]].drop_duplicates("root_event_id")

    def causal_model(feature: str) -> str:
        if feature.startswith("early_15m_"):
            return "response_15m_market"
        if feature.startswith("reclaim_") or feature == "reclaim_delay_min":
            return "reclaim_market"
        if feature.startswith("mss_1m_"):
            return "mss_1m_market"
        if feature.startswith("first_fvg_") or feature.startswith("directional_fvg_"):
            return "fvg_market"
        return "root_next_open"

    for side, side_frame in features.groupby("root_side", sort=True):
        discovery = side_frame.loc[side_frame["research_split"].eq("discovery")]
        for feature in feature_names:
            if feature not in side_frame:
                continue
            d = pd.to_numeric(discovery[feature], errors="coerce").dropna()
            if len(d) < 40 or d.nunique() < 4:
                continue
            edges = np.unique(np.quantile(d, [0.0, 0.25, 0.50, 0.75, 1.0]))
            if len(edges) < 3:
                continue
            edges[0] = -np.inf; edges[-1] = np.inf
            labels = [f"Q{i+1}" for i in range(len(edges) - 1)]
            model = causal_model(feature)
            definitions.append({"root_side": side, "feature": feature, "bin_edges": "|".join(map(str, edges)), "fit_split": "discovery", "causal_entry_model": model})
            values = pd.to_numeric(side_frame[feature], errors="coerce")
            bins = pd.cut(values, bins=edges, labels=labels, include_lowest=True, duplicates="drop")
            tmp = side_frame.assign(_bin=bins.astype(str), _feature_value=values)
            model_rows = entry_map.get(model)
            if model_rows is not None:
                model_rows = model_rows.rename(columns={
                    "entry_status": "_entry_status", "outcome": "_entry_outcome",
                    "net_return_cost2x": "_causal_net_return_cost2x",
                })
                tmp = tmp.merge(model_rows, on="root_event_id", how="left", validate="one_to_one")
            else:
                tmp["_entry_status"] = ""
                tmp["_entry_outcome"] = ""
                tmp["_causal_net_return_cost2x"] = np.nan
            tmp = tmp.loc[tmp["_bin"].isin(labels)]
            for key, p in tmp.groupby(["research_split", "_bin"], sort=True):
                net = pd.to_numeric(p["_causal_net_return_cost2x"], errors="coerce").dropna()
                rows.append({"root_side": side, "feature": feature, "research_split": key[0], "bin": key[1],
                             "causal_entry_model": model,
                             "events": len(p), "feature_median": p["_feature_value"].median(),
                             "direct_reversal_rate": p["direct_reversal_label"].mean(),
                             "causal_entry_filled": int(p["_entry_status"].eq("filled").sum()),
                             "causal_entry_resolved": len(net),
                             "mean_causal_net_return_cost2x": net.mean(), "causal_net_pf_cost2x": _profit_factor(net),
                             "median_structural_rr": _num(p, "root_structural_rr").median()})
            for split, p in tmp.groupby("research_split", sort=True):
                agg = p.groupby("_bin", sort=True).agg(feature_median=("_feature_value", "median"), direct_rate=("direct_reversal_label", "mean"), net_mean=("_causal_net_return_cost2x", "mean"), events=("root_event_id", "size")).reset_index()
                mono.append({"root_side": side, "feature": feature, "research_split": split, "bins": len(agg),
                             "causal_entry_model": model,
                             "events": int(agg["events"].sum()),
                             "direct_rate_spearman": agg[["feature_median", "direct_rate"]].corr(method="spearman").iloc[0, 1] if len(agg) >= 3 else np.nan,
                             "net_expectancy_spearman": agg[["feature_median", "net_mean"]].corr(method="spearman").iloc[0, 1] if len(agg) >= 3 else np.nan})
    return pd.DataFrame(definitions), pd.DataFrame(rows), pd.DataFrame(mono)


def summarize_direct_failure_divergence(
    features: pd.DataFrame,
    *,
    feature_names: Sequence[str] = DEFAULT_BIN_FEATURES,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    rows = []
    for key, p in features.groupby(["research_split", "root_side", "year"], sort=True):
        for feature in feature_names:
            if feature not in p:
                continue
            a = pd.to_numeric(p.loc[p["direct_reversal_label"].eq(1), feature], errors="coerce").dropna()
            z = pd.to_numeric(p.loc[p["direct_reversal_label"].eq(0), feature], errors="coerce").dropna()
            pooled = pd.concat([a, z]); sd = pooled.std(ddof=0) if len(pooled) else np.nan
            diff = a.mean() - z.mean() if len(a) and len(z) else np.nan
            rows.append({"research_split": key[0], "root_side": key[1], "year": key[2], "feature": feature,
                         "direct_n": len(a), "failure_n": len(z), "direct_mean": a.mean(), "failure_mean": z.mean(),
                         "direct_median": a.median(), "failure_median": z.median(), "mean_diff": diff,
                         "standardized_mean_diff": diff / sd if pd.notna(sd) and sd > EPS else np.nan})
    return pd.DataFrame(rows)


def r13_causal_audit(features: pd.DataFrame, entries: pd.DataFrame, *, holdout_start: pd.Timestamp) -> pd.DataFrame:
    rows = []
    if not features.empty:
        root_av = pd.to_datetime(features["root_sweep_available_time"], errors="coerce")
        for col in [c for c in features.columns if c.endswith("available_time") and c != "root_sweep_available_time"]:
            t = pd.to_datetime(features[col], errors="coerce")
            rows.append({"check": f"{col}_not_before_root_available", "violations": int((t.notna() & t.lt(root_av)).sum())})
        rows.append({"check": "sealed_holdout_absent_from_features", "violations": int(pd.to_datetime(features["root_sweep_time"]).ge(holdout_start).sum())})
    if not entries.empty:
        signal = pd.to_datetime(entries["signal_available_time"], errors="coerce")
        entry = pd.to_datetime(entries["entry_time"], errors="coerce")
        filled = entries["entry_status"].eq("filled")
        rows.append({"check": "filled_entry_not_before_signal_available", "violations": int((filled & signal.notna() & entry.lt(signal)).sum())})
        rows.append({"check": "single_entry_row_per_model_root", "violations": int(entries.duplicated(["root_event_id", "entry_model"]).sum())})
        risk = pd.to_numeric(entries.get("risk_distance_pct"), errors="coerce")
        rows.append({"check": "filled_positive_structural_risk", "violations": int((filled & ~risk.gt(0)).sum())})
    return pd.DataFrame(rows)
