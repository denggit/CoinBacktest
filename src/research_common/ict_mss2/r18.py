#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R18 causal Binance-positioning unwind path atlas.

The admission table contains only information known by the signal timestamp.
Binance base OI is a cross-exchange positioning proxy; price, execution, and
first-passage paths remain OKX ETH-USDT-SWAP.
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
class R18Config:
    research_start: pd.Timestamp = pd.Timestamp("2023-01-01 00:00:00")
    validation_start: pd.Timestamp = pd.Timestamp("2025-01-01 00:00:00")
    embargo_start: pd.Timestamp = pd.Timestamp("2025-07-01 00:00:00")
    holdout_start: pd.Timestamp = pd.Timestamp("2025-08-01 00:00:00")
    publication_lag: pd.Timedelta = pd.Timedelta("1min")
    metric_min_gap: pd.Timedelta = pd.Timedelta("4min")
    metric_max_gap: pd.Timedelta = pd.Timedelta("6min")
    baseline_tolerance_seconds: float = 60.0
    atr_window_5m: int = 12
    atr_min_periods_5m: int = 12
    stop_buffer_atr: float = 0.25
    max_stop_distance_pct: float = 0.015
    path_horizon_minutes: int = 24 * 60
    fixed_r_targets: tuple[float, ...] = (1.0, 2.0, 3.0)
    market_roundtrip_cost: float = 0.0011
    cost_scales: tuple[float, ...] = (1.0, 2.0, 3.0)

    def validate(self) -> "R18Config":
        if not (self.research_start < self.validation_start < self.embargo_start < self.holdout_start):
            raise ValueError("R18 split boundaries must be strictly increasing")
        if self.publication_lag < pd.Timedelta(0):
            raise ValueError("publication lag cannot be negative")
        if not (pd.Timedelta(0) < self.metric_min_gap <= self.metric_max_gap):
            raise ValueError("invalid metric gap bounds")
        if self.baseline_tolerance_seconds < 0:
            raise ValueError("baseline tolerance cannot be negative")
        if self.atr_window_5m < 2 or not 1 <= self.atr_min_periods_5m <= self.atr_window_5m:
            raise ValueError("invalid 5m ATR window")
        if self.stop_buffer_atr < 0 or not 0 < self.max_stop_distance_pct < 1:
            raise ValueError("invalid stop configuration")
        if self.path_horizon_minutes <= 0:
            raise ValueError("path horizon must be positive")
        if not self.fixed_r_targets or any(float(x) <= 0 for x in self.fixed_r_targets):
            raise ValueError("fixed-R targets must be positive")
        if self.market_roundtrip_cost < 0 or any(float(x) <= 0 for x in self.cost_scales):
            raise ValueError("invalid cost configuration")
        return self


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _datetime_ns(values: Iterable[object]) -> np.ndarray:
    return np.asarray(pd.to_datetime(values, errors="coerce"), dtype="datetime64[ns]").astype(np.int64)


def _true_range(frame: pd.DataFrame) -> pd.Series:
    high = _num(frame, "high")
    low = _num(frame, "low")
    previous_close = _num(frame, "close").shift(1)
    return pd.concat(
        [(high - low).abs(), (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)


def _split_at(when: pd.Timestamp, cfg: R18Config) -> str:
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


def _price_state(bars_1m: pd.DataFrame, cfg: R18Config) -> pd.DataFrame:
    five = aggregate_bars(bars_1m, 5).copy()
    close = _num(five, "close")
    high = _num(five, "high")
    low = _num(five, "low")
    atr = _true_range(five).rolling(
        cfg.atr_window_5m, min_periods=cfg.atr_min_periods_5m
    ).mean()
    return pd.DataFrame(
        {
            "price_bar_time": five.index,
            "price_available_time": pd.to_datetime(five["bar_end_time"]),
            "price_open": _num(five, "open").to_numpy(float),
            "price_high": high.to_numpy(float),
            "price_low": low.to_numpy(float),
            "price_close": close.to_numpy(float),
            "price_prior_high": high.shift(1).to_numpy(float),
            "price_prior_low": low.shift(1).to_numpy(float),
            "price_return_1h": (close / close.shift(12) - 1.0).to_numpy(float),
            "build_range_high_1h": high.rolling(12, min_periods=12).max().to_numpy(float),
            "build_range_low_1h": low.rolling(12, min_periods=12).min().to_numpy(float),
            "atr_5m_1h": atr.to_numpy(float),
        }
    ).sort_values("price_available_time", kind="stable").reset_index(drop=True)


def _age_valid(values: pd.Series, nominal_seconds: float, cfg: R18Config) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    return x.between(
        float(nominal_seconds),
        float(nominal_seconds + cfg.baseline_tolerance_seconds),
        inclusive="both",
    )


def _positive_current_and_baseline(current: pd.Series, change: pd.Series) -> pd.Series:
    current_num = pd.to_numeric(current, errors="coerce")
    change_num = pd.to_numeric(change, errors="coerce")
    return (
        current_num.gt(0)
        & np.isfinite(change_num)
        & (1.0 + change_num).gt(EPS)
    )


def prepare_positioning_alignment(
    bars_1m: pd.DataFrame,
    oi_features: pd.DataFrame,
    *,
    config: R18Config | None = None,
) -> pd.DataFrame:
    """Causally align gap-audited Binance base OI with completed OKX 5m bars.

    This is the shared positioning-transition foundation for R18 and later
    independent state machines. It deliberately carries no outcome fields.
    """

    cfg = (config or R18Config()).validate()
    required = {
        "timestamp",
        "available_time",
        "sum_open_interest",
        "oi_base_change_5m",
        "oi_base_change_1h",
        "oi_baseline_age_seconds_5m",
        "oi_baseline_age_seconds_1h",
    }
    missing = sorted(required - set(oi_features.columns))
    if missing:
        raise ValueError(f"OI features missing required columns: {missing}")
    forbidden = [name for name in oi_features.columns if name.startswith("future_") or "oracle" in name.lower()]
    if forbidden:
        raise RuntimeError(f"future leakage in positioning input feature table: {forbidden}")

    oi = oi_features.reset_index(drop=True).copy()
    oi["timestamp"] = pd.to_datetime(oi["timestamp"], errors="coerce")
    oi["available_time"] = pd.to_datetime(oi["available_time"], errors="coerce")
    oi = oi.dropna(subset=["timestamp", "available_time"]).sort_values("available_time", kind="stable")
    oi = oi.drop_duplicates("available_time", keep="last").reset_index(drop=True)
    price = _price_state(bars_1m, cfg)
    aligned = pd.merge_asof(
        oi,
        price,
        left_on="available_time",
        right_on="price_available_time",
        direction="backward",
        tolerance=pd.Timedelta("5min"),
        allow_exact_matches=True,
    )
    aligned["metric_gap"] = aligned["timestamp"].diff()
    shift_columns = [
        "timestamp",
        "available_time",
        "sum_open_interest",
        "oi_base_change_5m",
        "oi_base_change_1h",
        "oi_baseline_age_seconds_5m",
        "oi_baseline_age_seconds_1h",
        "price_bar_time",
        "price_available_time",
        "price_return_1h",
        "build_range_high_1h",
        "build_range_low_1h",
    ]
    for name in shift_columns:
        aligned[f"build_{name}"] = aligned[name].shift(1)

    current_oi = _num(aligned, "sum_open_interest")
    prior_oi = _num(aligned, "build_sum_open_interest")
    aligned["current_oi_valid"] = (
        _positive_current_and_baseline(current_oi, _num(aligned, "oi_base_change_5m"))
        & _age_valid(aligned["oi_baseline_age_seconds_5m"], 300.0, cfg)
    )
    aligned["build_oi_valid"] = (
        _positive_current_and_baseline(prior_oi, _num(aligned, "build_oi_base_change_5m"))
        & _positive_current_and_baseline(prior_oi, _num(aligned, "build_oi_base_change_1h"))
        & _age_valid(aligned["build_oi_baseline_age_seconds_5m"], 300.0, cfg)
        & _age_valid(aligned["build_oi_baseline_age_seconds_1h"], 3600.0, cfg)
    )
    aligned["metric_gap_valid"] = aligned["metric_gap"].between(
        cfg.metric_min_gap, cfg.metric_max_gap, inclusive="both"
    )
    price_gap = pd.to_datetime(aligned["price_bar_time"]) - pd.to_datetime(aligned["build_price_bar_time"])
    aligned["price_step_valid"] = (
        aligned["price_available_time"].notna()
        & aligned["build_price_available_time"].notna()
        & price_gap.eq(pd.Timedelta("5min"))
    )
    return aligned


def build_positioning_unwind_events(
    bars_1m: pd.DataFrame,
    oi_features: pd.DataFrame,
    *,
    config: R18Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the frozen R18 causal event table while sealing holdout details."""

    cfg = (config or R18Config()).validate()
    bars = normalize_1m_bars(bars_1m)
    aligned = prepare_positioning_alignment(bars, oi_features, config=cfg)

    current_oi = _num(aligned, "sum_open_interest")
    current_oi5 = _num(aligned, "oi_base_change_5m")
    prior_oi = _num(aligned, "build_sum_open_interest")
    prior_oi5 = _num(aligned, "build_oi_base_change_5m")
    prior_oi1h = _num(aligned, "build_oi_base_change_1h")
    current_valid = aligned["current_oi_valid"].astype(bool)
    prior_valid = aligned["build_oi_valid"].astype(bool)
    gap_valid = aligned["metric_gap_valid"].astype(bool)
    price_valid = aligned["price_step_valid"].astype(bool)
    release = current_oi5.lt(0) & prior_oi5.ge(0)
    build_price = _num(aligned, "build_price_return_1h")
    long_mask = (
        current_valid
        & prior_valid
        & gap_valid
        & price_valid
        & release
        & prior_oi1h.gt(0)
        & build_price.lt(0)
        & _num(aligned, "price_close").gt(_num(aligned, "price_prior_high"))
    )
    short_mask = (
        current_valid
        & prior_valid
        & gap_valid
        & price_valid
        & release
        & prior_oi1h.gt(0)
        & build_price.gt(0)
        & _num(aligned, "price_close").lt(_num(aligned, "price_prior_low"))
    )
    candidate = aligned.loc[long_mask | short_mask].copy()
    candidate["trade_direction"] = np.where(long_mask.loc[candidate.index], 1, -1)
    candidate["direction"] = np.where(candidate["trade_direction"].eq(1), "Long", "Short")
    candidate["signal_available_time"] = candidate[["available_time", "price_available_time"]].max(axis=1)
    candidate["research_split"] = candidate["signal_available_time"].map(lambda value: _split_at(value, cfg))

    holdout = candidate["research_split"].eq("holdout")
    seal = pd.DataFrame(
        [
            {"check": "holdout_start", "value": str(cfg.holdout_start)},
            {"check": "sealed_holdout_candidate_count", "value": int(holdout.sum())},
            {"check": "holdout_outcome_rows_computed", "value": 0},
            {"check": "holdout_unsealed", "value": 0},
        ]
    )
    visible = candidate.loc[candidate["research_split"].isin(["discovery", "validation"])].copy()

    index_ns = _datetime_ns(bars.index)
    rows: list[dict[str, object]] = []
    for ordinal, row in enumerate(visible.itertuples(index=False), start=1):
        signal = pd.Timestamp(row.signal_available_time)
        direction = int(row.trade_direction)
        entry_pos = int(np.searchsorted(index_ns, np.datetime64(signal, "ns").astype(np.int64), side="left"))
        base: dict[str, object] = {
            "setup_id": f"R18_{row.direction.upper()}_{signal.strftime('%Y%m%dT%H%M%S%f')}_{ordinal:06d}",
            "direction": str(row.direction),
            "trade_direction": direction,
            "research_split": str(row.research_split),
            "build_oi_metric_time": pd.Timestamp(row.build_timestamp),
            "build_oi_available_time": pd.Timestamp(row.build_available_time),
            "release_oi_metric_time": pd.Timestamp(row.timestamp),
            "release_oi_available_time": pd.Timestamp(row.available_time),
            "build_price_bar_time": pd.Timestamp(row.build_price_bar_time),
            "build_price_available_time": pd.Timestamp(row.build_price_available_time),
            "stabilization_price_bar_time": pd.Timestamp(row.price_bar_time),
            "stabilization_price_available_time": pd.Timestamp(row.price_available_time),
            "signal_available_time": signal,
            "metric_gap_seconds": float(pd.Timedelta(row.metric_gap).total_seconds()),
            "build_price_return_1h": float(row.build_price_return_1h),
            "build_oi_base_change_1h": float(row.build_oi_base_change_1h),
            "prior_oi_base_change_5m": float(row.build_oi_base_change_5m),
            "release_oi_base_change_5m": float(row.oi_base_change_5m),
            "release_oi_base": float(row.sum_open_interest),
            "release_oi_baseline_age_seconds_5m": float(row.oi_baseline_age_seconds_5m),
            "build_oi_baseline_age_seconds_1h": float(row.build_oi_baseline_age_seconds_1h),
            "stabilization_close": float(row.price_close),
            "stabilization_prior_high": float(row.price_prior_high),
            "stabilization_prior_low": float(row.price_prior_low),
            "stabilization_bar_high": float(row.price_high),
            "stabilization_bar_low": float(row.price_low),
            "atr_5m_1h_at_signal": float(row.atr_5m_1h),
            "build_range_high_1h": float(row.build_build_range_high_1h),
            "build_range_low_1h": float(row.build_build_range_low_1h),
            "setup_status": "pending_geometry",
        }
        if entry_pos >= len(bars):
            base["setup_status"] = "next_1m_entry_unavailable"
            rows.append(base)
            continue
        entry_time = pd.Timestamp(bars.index[entry_pos])
        entry = float(bars.iloc[entry_pos]["open"])
        atr = float(row.atr_5m_1h)
        stop = (
            min(float(row.price_low), float(row.price_prior_low)) - cfg.stop_buffer_atr * atr
            if direction > 0
            else max(float(row.price_high), float(row.price_prior_high)) + cfg.stop_buffer_atr * atr
        )
        target = float(row.build_build_range_high_1h if direction > 0 else row.build_build_range_low_1h)
        risk = direction * (entry - stop) / entry if entry > EPS else np.nan
        runway = direction * (target / entry - 1.0) if entry > EPS else np.nan
        base.update(
            {
                "entry_time": entry_time,
                "entry_price": entry,
                "stop_price": stop,
                "risk_distance_pct": risk,
                "structural_target_price": target,
                "structural_target_available_time": pd.Timestamp(row.build_available_time),
                "structural_runway_pct": runway,
                "structural_reward_risk": runway / risk if np.isfinite(risk) and risk > EPS else np.nan,
            }
        )
        if not np.isfinite(atr) or atr <= EPS:
            base["setup_status"] = "atr_unavailable"
        elif not np.isfinite(risk) or risk <= EPS:
            base["setup_status"] = "invalid_stop_geometry"
        elif risk > cfg.max_stop_distance_pct + EPS:
            base["setup_status"] = "stop_too_wide"
        elif not np.isfinite(runway) or runway <= EPS:
            base["setup_status"] = "no_remaining_structural_target"
        else:
            base["setup_status"] = "executable"
        rows.append(base)

    events = pd.DataFrame(rows)
    if not events.empty:
        for name in [c for c in events.columns if c.endswith("_time")]:
            events[name] = pd.to_datetime(events[name], errors="coerce")
        events = events.sort_values(["signal_available_time", "direction", "setup_id"], kind="stable").reset_index(drop=True)
    engineering = pd.DataFrame(
        [
            {"check": "oi_feature_rows", "value": int(len(aligned))},
            {"check": "gap_safe_current_oi_rows", "value": int((current_valid & gap_valid).sum())},
            {"check": "visible_release_transition_candidates", "value": int(len(visible))},
            {"check": "visible_long_candidates", "value": int(visible["direction"].eq("Long").sum())},
            {"check": "visible_short_candidates", "value": int(visible["direction"].eq("Short").sum())},
            {"check": "visible_executable_setups", "value": int(events.get("setup_status", pd.Series(dtype=str)).eq("executable").sum())},
        ]
    )
    return events, seal, engineering


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


def build_positioning_unwind_paths(
    bars_1m: pd.DataFrame,
    events: pd.DataFrame,
    *,
    config: R18Config | None = None,
) -> pd.DataFrame:
    """Build exact 1m first-passage paths with pessimistic same-bar ordering."""

    cfg = (config or R18Config()).validate()
    if events.empty:
        return pd.DataFrame()
    executable = events.loc[events["setup_status"].eq("executable")].copy()
    if executable.empty:
        return pd.DataFrame()
    bars = normalize_1m_bars(bars_1m)
    index_ns = _datetime_ns(bars.index)
    high = _num(bars, "high").to_numpy(float)
    low = _num(bars, "low").to_numpy(float)
    close = _num(bars, "close").to_numpy(float)
    high_tree = SegmentThresholdIndex(high)
    low_tree = SegmentThresholdIndex(low)
    target_specs: tuple[tuple[str, float | None], ...] = (
        ("H0_1H_BUILD_RANGE", None),
        *tuple((f"R{float(value):g}", float(value)) for value in cfg.fixed_r_targets),
    )
    rows: list[dict[str, object]] = []
    for event in executable.itertuples(index=False):
        entry_time = pd.Timestamp(event.entry_time)
        entry_pos = int(np.searchsorted(index_ns, np.datetime64(entry_time, "ns").astype(np.int64), side="left"))
        if entry_pos >= len(bars) or pd.Timestamp(bars.index[entry_pos]) != entry_time:
            continue
        end = entry_pos + cfg.path_horizon_minutes - 1
        # July is fully embargoed: do not calculate a validation path that spills into it.
        if end >= len(bars) or (
            str(event.research_split) == "validation"
            and pd.Timestamp(bars.index[end]) >= cfg.embargo_start
        ):
            continue
        direction = int(event.trade_direction)
        entry = float(event.entry_price)
        stop = float(event.stop_price)
        risk_price = abs(entry - stop)
        base = {
            "setup_id": str(event.setup_id),
            "direction": str(event.direction),
            "trade_direction": direction,
            "research_split": str(event.research_split),
            "year": int(entry_time.year),
            "build_oi_available_time": pd.Timestamp(event.build_oi_available_time),
            "release_oi_available_time": pd.Timestamp(event.release_oi_available_time),
            "signal_available_time": pd.Timestamp(event.signal_available_time),
            "entry_time": entry_time,
            "entry_price": entry,
            "stop_price": stop,
            "risk_distance_pct": float(event.risk_distance_pct),
            "structural_runway_pct": float(event.structural_runway_pct),
            "structural_reward_risk": float(event.structural_reward_risk),
            "build_price_return_1h": float(event.build_price_return_1h),
            "build_oi_base_change_1h": float(event.build_oi_base_change_1h),
            "release_oi_base_change_5m": float(event.release_oi_base_change_5m),
        }
        for model, multiple in target_specs:
            target = (
                float(event.structural_target_price)
                if multiple is None
                else entry + direction * float(multiple) * risk_price
            )
            rec = dict(base)
            rec.update({"target_model": model, "target_price": target})
            if direction * (target - entry) <= EPS or direction * (entry - stop) <= EPS:
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
            exit_price = target if outcome == "tp_first" else stop if outcome == "sl_first" else float(close[exit_pos])
            gross = direction * (exit_price / entry - 1.0)
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
                    "gross_return": gross,
                    "gross_r": gross / float(event.risk_distance_pct),
                    "mfe_pct": mfe,
                    "mae_pct": mae,
                }
            )
            for scale in cfg.cost_scales:
                net = gross - cfg.market_roundtrip_cost * float(scale)
                rec[f"net_return_cost{float(scale):g}x"] = net
                rec[f"net_r_cost{float(scale):g}x"] = net / float(event.risk_distance_pct)
            rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce")
        out = out.sort_values(["entry_time", "direction", "target_model"], kind="stable").reset_index(drop=True)
    return out


def _profit_factor(values: Iterable[float]) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    gain = float(x[x > 0].sum())
    loss = float(-x[x < 0].sum())
    return gain / loss if loss > EPS else (np.inf if gain > EPS else np.nan)


def _top_removed_pf(values: pd.Series, count: int) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False)
    return _profit_factor(x.iloc[int(count) :]) if len(x) > int(count) else np.nan


def _calendar_for_split(split: str) -> pd.PeriodIndex:
    if split == "discovery":
        return pd.period_range("2023-01", "2024-12", freq="M")
    if split == "validation":
        return pd.period_range("2025-01", "2025-06", freq="M")
    return pd.PeriodIndex([], freq="M")


def summarize_r18_funnel(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (split, direction), part in events.groupby(["research_split", "direction"], sort=True):
        executable = part.loc[part["setup_status"].eq("executable")]
        rows.append(
            {
                "research_split": split,
                "direction": direction,
                "transition_candidates": int(len(part)),
                "executable_rows": int(len(executable)),
                "stop_too_wide_rows": int(part["setup_status"].eq("stop_too_wide").sum()),
                "no_structural_target_rows": int(part["setup_status"].eq("no_remaining_structural_target").sum()),
                "median_risk_distance_pct": _num(executable, "risk_distance_pct").median(),
                "median_structural_reward_risk": _num(executable, "structural_reward_risk").median(),
                "median_metric_gap_seconds": _num(part, "metric_gap_seconds").median(),
            }
        )
    return pd.DataFrame(rows)


def summarize_r18_paths(paths: pd.DataFrame, *, config: R18Config | None = None) -> pd.DataFrame:
    cfg = (config or R18Config()).validate()
    if paths.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (split, direction, model), part in paths.groupby(
        ["research_split", "direction", "target_model"], sort=True
    ):
        valid = part.loc[part["gross_return"].notna()].copy()
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
        times = pd.to_datetime(valid["entry_time"], errors="coerce").dropna().drop_duplicates().sort_values()
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


def summarize_r18_years(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (year, direction, model), part in paths.groupby(["year", "direction", "target_model"], sort=True):
        valid = part.loc[part["gross_return"].notna()]
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


def r18_data_quality_audit(
    bars_1m: pd.DataFrame,
    oi_features: pd.DataFrame,
    coverage_by_day: pd.DataFrame,
    *,
    config: R18Config | None = None,
) -> pd.DataFrame:
    """Profile only pre-embargo feature values; later coverage is metadata only."""

    cfg = (config or R18Config()).validate()
    bars = normalize_1m_bars(bars_1m)
    oi = oi_features.reset_index(drop=True).copy()
    oi["timestamp"] = pd.to_datetime(oi["timestamp"], errors="coerce")
    oi["available_time"] = pd.to_datetime(oi["available_time"], errors="coerce")
    oi = oi.loc[(oi["available_time"] >= cfg.research_start) & (oi["available_time"] < cfg.embargo_start)].copy()
    delta = oi.sort_values("timestamp")["timestamp"].diff().dropna()
    bad_base_oi = _num(oi, "sum_open_interest").le(0)
    bad_usd_oi = (
        _num(oi, "sum_open_interest_value").le(0)
        if "sum_open_interest_value" in oi
        else pd.Series(False, index=oi.index)
    )
    index_delta = bars.index.to_series().diff().dropna()
    invalid_ohlc = (
        _num(bars, "high").lt(_num(bars, "low"))
        | _num(bars, "high").lt(_num(bars, "open"))
        | _num(bars, "high").lt(_num(bars, "close"))
        | _num(bars, "low").gt(_num(bars, "open"))
        | _num(bars, "low").gt(_num(bars, "close"))
    )
    coverage = coverage_by_day.copy()
    if not coverage.empty and "day_utc" in coverage:
        coverage["day_utc"] = pd.to_datetime(coverage["day_utc"], errors="coerce")
        coverage = coverage.loc[
            (coverage["day_utc"] >= cfg.research_start.normalize())
            & (coverage["day_utc"] < cfg.embargo_start)
        ]
    rows = [
        {"check": "oi_pre_embargo_rows", "value": int(len(oi)), "status": "INFO"},
        {"check": "oi_duplicate_timestamps", "value": int(oi["timestamp"].duplicated().sum()), "status": "PASS" if not oi["timestamp"].duplicated().any() else "FAIL"},
        {"check": "oi_nonexact_5m_intervals", "value": int(delta.ne(pd.Timedelta("5min")).sum()), "status": "EXCLUDE"},
        {"check": "oi_max_interval_seconds", "value": float(delta.max().total_seconds()) if len(delta) else np.nan, "status": "INFO"},
        {"check": "oi_nonpositive_base_rows", "value": int(bad_base_oi.sum()), "status": "EXCLUDE"},
        {"check": "oi_nonpositive_usd_rows", "value": int(bad_usd_oi.sum()), "status": "INFO_UNUSED"},
        {"check": "oi_base_null_rows", "value": int(_num(oi, "sum_open_interest").isna().sum()), "status": "PASS"},
        {"check": "oi_5m_feature_present_rows", "value": int(_num(oi, "oi_base_change_5m").notna().sum()), "status": "INFO"},
        {"check": "oi_1h_feature_present_rows", "value": int(_num(oi, "oi_base_change_1h").notna().sum()), "status": "INFO"},
        {"check": "oi_partial_archive_days_pre_embargo", "value": int(coverage.get("status", pd.Series(dtype=str)).eq("partial").sum()), "status": "EXCLUDE_GAPS"},
        {"check": "okx_1m_rows_loaded", "value": int(len(bars)), "status": "INFO"},
        {"check": "okx_1m_duplicate_timestamps", "value": int(bars.index.duplicated().sum()), "status": "PASS"},
        {"check": "okx_1m_nonexact_intervals", "value": int(index_delta.ne(pd.Timedelta("1min")).sum()), "status": "PASS" if index_delta.eq(pd.Timedelta("1min")).all() else "FAIL"},
        {"check": "okx_1m_invalid_ohlc_rows", "value": int(invalid_ohlc.sum()), "status": "PASS" if not invalid_ohlc.any() else "FAIL"},
    ]
    return pd.DataFrame(rows)


def r18_causal_audit(
    events: pd.DataFrame,
    paths: pd.DataFrame,
    *,
    config: R18Config | None = None,
) -> pd.DataFrame:
    cfg = (config or R18Config()).validate()
    checks: list[dict[str, object]] = []

    def add(name: str, violations: int) -> None:
        checks.append({"check": name, "violations": int(violations), "status": "PASS" if int(violations) == 0 else "FAIL"})

    if events.empty:
        add("nonempty_visible_event_table", 1)
        return pd.DataFrame(checks)
    signal = pd.to_datetime(events["signal_available_time"], errors="coerce")
    entry = pd.to_datetime(events["entry_time"], errors="coerce")
    executable = events.loc[events["setup_status"].eq("executable")].copy()
    add("unique_setup_id", int(events["setup_id"].duplicated().sum()))
    add("feature_schema_excludes_future_or_oracle", len([c for c in events.columns if c.startswith("future_") or "oracle" in c.lower()]))
    add("release_publication_lag_exact", int((pd.to_datetime(events["release_oi_metric_time"]) + cfg.publication_lag != pd.to_datetime(events["release_oi_available_time"])).sum()))
    add("build_information_available_before_signal", int((pd.to_datetime(events["build_oi_available_time"]) >= signal).sum()))
    add("release_information_available_by_signal", int((pd.to_datetime(events["release_oi_available_time"]) > signal).sum()))
    add("price_information_available_by_signal", int((pd.to_datetime(events["stabilization_price_available_time"]) > signal).sum()))
    add("metric_gap_within_frozen_bounds", int((~_num(events, "metric_gap_seconds").between(cfg.metric_min_gap.total_seconds(), cfg.metric_max_gap.total_seconds())).sum()))
    add("rising_oi_build", int((_num(events, "build_oi_base_change_1h") <= 0).sum()))
    add("first_oi_release_transition", int(((_num(events, "prior_oi_base_change_5m") < 0) | (_num(events, "release_oi_base_change_5m") >= 0)).sum()))
    direction = _num(events, "trade_direction")
    add("directional_build_sign", int((((direction > 0) & (_num(events, "build_price_return_1h") >= 0)) | ((direction < 0) & (_num(events, "build_price_return_1h") <= 0))).sum()))
    add("price_reacquisition_sign", int((((direction > 0) & (_num(events, "stabilization_close") <= _num(events, "stabilization_prior_high"))) | ((direction < 0) & (_num(events, "stabilization_close") >= _num(events, "stabilization_prior_low")))).sum()))
    expected_entry = signal.dt.ceil("min")
    add("next_eligible_1m_open", int((entry != expected_entry).sum()))
    add("holdout_absent_from_visible_events", int((signal >= cfg.holdout_start).sum()))
    if executable.empty:
        add("nonempty_executable_events", 1)
    else:
        add("maximum_stop_distance_respected", int((_num(executable, "risk_distance_pct") > cfg.max_stop_distance_pct + EPS).sum()))
        add("structural_target_available_by_signal", int((pd.to_datetime(executable["structural_target_available_time"]) > pd.to_datetime(executable["signal_available_time"])).sum()))
    if paths.empty:
        add("nonempty_first_passage_paths", 1)
    else:
        add("paths_reference_executable_events", int((~paths["setup_id"].isin(executable["setup_id"])).sum()))
        add("path_entry_not_before_signal", int((pd.to_datetime(paths["entry_time"]) < pd.to_datetime(paths["signal_available_time"])).sum()))
        add("holdout_absent_from_paths", int((pd.to_datetime(paths["entry_time"]) >= cfg.holdout_start).sum()))
        add("embargo_absent_from_paths", int((pd.to_datetime(paths["entry_time"]) >= cfg.embargo_start).sum()))
        add("unique_setup_target_path", int(paths.duplicated(["setup_id", "target_model"]).sum()))
    return pd.DataFrame(checks)
