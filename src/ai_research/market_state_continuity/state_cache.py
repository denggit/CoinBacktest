#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal universal OHLCV state cache for R03.3.3."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.swing_baseline.features import (
    LONG_CONTEXT_PROFILE,
    build_causal_minute_grid,
    build_multitimeframe_feature_bundle,
)
from src.research_common.progress import ProgressReporter

from .config import MarketStateContinuityConfig, StateTargetSpec
from .data import UnifiedOHLCVLoader


CACHE_SCHEMA_VERSION = 3
_EPS = 1e-9

FLOW_TOKENS = (
    "flow",
    "delta",
    "taker",
    "large_trade",
    "large_buy",
    "large_sell",
    "notional",
    "buy_",
    "sell_",
    "trades_count",
    "max_trade",
)


def datetime_index_to_ns(index: pd.DatetimeIndex | pd.Index | np.ndarray) -> np.ndarray:
    """Return true Unix nanoseconds regardless of the input datetime resolution."""
    values = pd.DatetimeIndex(pd.to_datetime(index, errors="raise"))
    return values.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)


def ns_to_datetime(values: np.ndarray) -> pd.DatetimeIndex:
    """Decode cache timestamps that are contractually stored in Unix nanoseconds."""
    return pd.to_datetime(np.asarray(values, dtype=np.int64), unit="ns")


def _cache_signature(config: MarketStateContinuityConfig) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "config": config.to_dict(),
        "feature_profile": LONG_CONTEXT_PROFILE,
        "universal_feature_rule": "exclude_trade_only_columns_v1",
        "state_formula": "r0333_hierarchical_continuous_state_v1",
        "timestamp_contract": "unix_nanoseconds_v2",
        "causal_availability": "completed_bar_available_time",
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _universal_columns(columns: tuple[str, ...]) -> tuple[str, ...]:
    selected: list[str] = []
    for column in columns:
        lower = column.lower()
        if any(token in lower for token in FLOW_TOKENS):
            continue
        selected.append(column)
    return tuple(selected)


def _safe_component(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").astype(float)


def _trend_component(frame: pd.DataFrame, prefix: str, window: int) -> pd.Series:
    ret = _safe_component(frame, f"{prefix}ret_{window}")
    rv = _safe_component(frame, f"{prefix}rv_{window}")
    efficiency = _safe_component(frame, f"{prefix}trend_eff_{window}").clip(lower=0.0, upper=1.0)
    scale = rv.abs() * np.sqrt(float(window))
    signal = np.tanh(ret / scale.where(scale > _EPS))
    return (signal * efficiency).replace([np.inf, -np.inf], np.nan)


def _activity_component(frame: pd.DataFrame, prefix: str, short: int, long: int) -> pd.Series:
    rv_short = _safe_component(frame, f"{prefix}rv_{short}").abs()
    rv_long = _safe_component(frame, f"{prefix}rv_{long}").abs()
    ratio = rv_short / rv_long.where(rv_long > _EPS)
    return np.tanh(np.log(ratio.where(ratio > _EPS))).replace([np.inf, -np.inf], np.nan)


def _weighted_mean(components: list[tuple[pd.Series, float]]) -> pd.Series:
    if not components:
        raise ValueError("state score requires components")
    numerator = pd.Series(0.0, index=components[0][0].index)
    denominator = pd.Series(0.0, index=components[0][0].index)
    for values, weight in components:
        valid = values.notna()
        numerator = numerator.add(values.fillna(0.0) * float(weight), fill_value=0.0)
        denominator = denominator.add(valid.astype(float) * float(weight), fill_value=0.0)
    return (numerator / denominator.where(denominator > 0)).clip(-1.0, 1.0)


def _causal_daily_strategic_thresholds(
    values: pd.Series,
    config: MarketStateContinuityConfig,
) -> pd.DataFrame:
    """Build prior-day-only adaptive strategic thresholds.

    Strategic scores mix 1D and 4H components, so one fixed threshold can become
    unreachable across regimes. Thresholds are estimated from the previous
    calendar days only and are therefore available live at the start of each day.
    """
    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError("strategic threshold calibration requires DatetimeIndex")
    daily = values.resample("1D").last()
    history = daily.shift(1)
    rolling = history.rolling(
        config.strategic_threshold_lookback_days,
        min_periods=config.strategic_threshold_min_days,
    )
    daily_thresholds = pd.DataFrame(
        {
            "long_enter": rolling.quantile(config.strategic_long_enter_quantile),
            "short_enter": rolling.quantile(config.strategic_short_enter_quantile),
            "long_exit": rolling.quantile(config.strategic_long_exit_quantile),
            "short_exit": rolling.quantile(config.strategic_short_exit_quantile),
        },
        index=daily.index,
    )
    fallbacks = {
        "long_enter": config.strategic_fallback_long_enter,
        "short_enter": config.strategic_fallback_short_enter,
        "long_exit": config.strategic_fallback_long_exit,
        "short_exit": config.strategic_fallback_short_exit,
    }
    daily_thresholds = daily_thresholds.fillna(fallbacks)
    # Enforce economically coherent ordering even in degenerate historical windows.
    daily_thresholds["long_enter"] = np.maximum(
        daily_thresholds["long_enter"], daily_thresholds["long_exit"] + 0.01
    )
    daily_thresholds["short_enter"] = np.minimum(
        daily_thresholds["short_enter"], daily_thresholds["short_exit"] - 0.01
    )
    expanded = daily_thresholds.reindex(values.index, method="ffill")
    return expanded.astype(float)


def causal_asymmetric_hysteresis_state(
    values: np.ndarray,
    *,
    long_enter: np.ndarray | float,
    short_enter: np.ndarray | float,
    long_exit: np.ndarray | float,
    short_exit: np.ndarray | float,
) -> np.ndarray:
    """Causal three-state hysteresis with possibly time-varying asymmetric thresholds."""
    array = np.asarray(values, dtype=float)
    long_enter_values = np.broadcast_to(np.asarray(long_enter, dtype=float), array.shape)
    short_enter_values = np.broadcast_to(np.asarray(short_enter, dtype=float), array.shape)
    long_exit_values = np.broadcast_to(np.asarray(long_exit, dtype=float), array.shape)
    short_exit_values = np.broadcast_to(np.asarray(short_exit, dtype=float), array.shape)
    output = np.zeros(len(array), dtype=np.int8)
    state = 0
    for index, value in enumerate(array):
        thresholds = (
            long_enter_values[index],
            short_enter_values[index],
            long_exit_values[index],
            short_exit_values[index],
        )
        if not np.isfinite(value) or not np.all(np.isfinite(thresholds)):
            output[index] = state
            continue
        le, se, lx, sx = thresholds
        if state == 0:
            if value >= le:
                state = 1
            elif value <= se:
                state = -1
        elif state == 1:
            if value <= se:
                state = -1
            elif value < lx:
                state = 0
        else:
            if value >= le:
                state = 1
            elif value > sx:
                state = 0
        output[index] = state
    return output


def _state_boundary_margin(
    values: np.ndarray,
    states: np.ndarray,
    *,
    long_enter: np.ndarray | float,
    short_enter: np.ndarray | float,
    long_exit: np.ndarray | float,
    short_exit: np.ndarray | float,
) -> np.ndarray:
    """Distance from the active state's next hysteresis boundary.

    Positive values mean the current state remains inside its hysteresis region;
    values near zero identify mechanically fragile states.
    """
    values = np.asarray(values, dtype=float)
    states = np.asarray(states, dtype=np.int8)
    le = np.broadcast_to(np.asarray(long_enter, dtype=float), values.shape)
    se = np.broadcast_to(np.asarray(short_enter, dtype=float), values.shape)
    lx = np.broadcast_to(np.asarray(long_exit, dtype=float), values.shape)
    sx = np.broadcast_to(np.asarray(short_exit, dtype=float), values.shape)
    margin = np.full(len(values), np.nan, dtype=float)
    long_mask = states == 1
    short_mask = states == -1
    neutral_mask = states == 0
    margin[long_mask] = values[long_mask] - lx[long_mask]
    margin[short_mask] = sx[short_mask] - values[short_mask]
    margin[neutral_mask] = np.minimum(
        le[neutral_mask] - values[neutral_mask],
        values[neutral_mask] - se[neutral_mask],
    )
    return margin


def causal_hysteresis_state(
    values: np.ndarray,
    *,
    enter_threshold: float,
    exit_threshold: float,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    output = np.zeros(len(array), dtype=np.int8)
    state = 0
    for index, value in enumerate(array):
        if not np.isfinite(value):
            output[index] = state
            continue
        if state == 0:
            if value >= enter_threshold:
                state = 1
            elif value <= -enter_threshold:
                state = -1
        elif state == 1:
            if value <= -enter_threshold:
                state = -1
            elif value < exit_threshold:
                state = 0
        else:
            if value >= enter_threshold:
                state = 1
            elif value > -exit_threshold:
                state = 0
        output[index] = state
    return output


def _causal_age(codes: np.ndarray) -> np.ndarray:
    values = np.asarray(codes)
    output = np.ones(len(values), dtype=np.int32)
    for index in range(1, len(values)):
        output[index] = output[index - 1] + 1 if values[index] == values[index - 1] else 1
    return output


def _rolling_flip_rate(codes: np.ndarray, window: int) -> np.ndarray:
    values = pd.Series(np.asarray(codes))
    flips = values.ne(values.shift(1)).astype(float)
    if len(flips):
        flips.iloc[0] = 0.0
    return flips.rolling(window, min_periods=max(2, window // 4)).mean().to_numpy(dtype=float)


def _time_to_next_change(codes: np.ndarray, max_steps: int) -> np.ndarray:
    values = np.asarray(codes)
    output = np.full(len(values), np.nan, dtype=float)
    if not len(values):
        return output
    run_remaining = 0
    output[-1] = np.nan
    for index in range(len(values) - 2, -1, -1):
        if values[index + 1] != values[index]:
            run_remaining = 1
        else:
            run_remaining += 1
        output[index] = float(min(run_remaining, max_steps + 1))
    return output


def build_state_frame(feature_frame: pd.DataFrame, config: MarketStateContinuityConfig) -> pd.DataFrame:
    """Build continuous hierarchical state scores using only data available now."""
    strategic = _weighted_mean(
        [
            (_trend_component(feature_frame, "tf1d_", 90), 1.0),
            (_trend_component(feature_frame, "tf1d_", 180), 1.3),
            (_trend_component(feature_frame, "tf1d_", 365), 1.5),
            (_trend_component(feature_frame, "tf4h_", 180), 0.8),
            (_trend_component(feature_frame, "tf4h_", 360), 1.0),
            (_trend_component(feature_frame, "tf4h_", 720), 1.1),
        ]
    )
    tactical = _weighted_mean(
        [
            (_trend_component(feature_frame, "tf4h_", 30), 1.0),
            (_trend_component(feature_frame, "tf4h_", 90), 1.2),
            (_trend_component(feature_frame, "tf1h_", 24), 0.8),
            (_trend_component(feature_frame, "tf1h_", 72), 1.1),
            (_trend_component(feature_frame, "tf1h_", 168), 1.0),
        ]
    )
    entry = _weighted_mean(
        [
            (_trend_component(feature_frame, "tf15m_", 8), 1.0),
            (_trend_component(feature_frame, "tf15m_", 32), 1.1),
            (_trend_component(feature_frame, "tf5m_", 12), 0.8),
            (_trend_component(feature_frame, "tf5m_", 48), 1.0),
            (_trend_component(feature_frame, "tf1m_", 60), 0.7),
            (_trend_component(feature_frame, "tf1m_", 240), 0.9),
        ]
    )
    strategic_activity = _weighted_mean(
        [
            (_activity_component(feature_frame, "tf1d_", 20, 180), 1.0),
            (_activity_component(feature_frame, "tf4h_", 30, 360), 1.0),
        ]
    )
    tactical_activity = _weighted_mean(
        [
            (_activity_component(feature_frame, "tf4h_", 12, 90), 1.0),
            (_activity_component(feature_frame, "tf1h_", 24, 168), 1.0),
        ]
    )
    entry_activity = _weighted_mean(
        [
            (_activity_component(feature_frame, "tf15m_", 8, 32), 1.0),
            (_activity_component(feature_frame, "tf5m_", 12, 48), 1.0),
            (_activity_component(feature_frame, "tf1m_", 60, 240), 1.0),
        ]
    )
    activity = _weighted_mean([(strategic_activity, 0.5), (tactical_activity, 1.0), (entry_activity, 1.2)])

    state = pd.DataFrame(index=feature_frame.index)
    state["strategic_score"] = strategic
    state["tactical_score"] = tactical
    state["entry_score"] = entry
    state["strategic_activity_score"] = strategic_activity
    state["tactical_activity_score"] = tactical_activity
    state["entry_activity_score"] = entry_activity
    state["activity_score"] = activity

    strategic_thresholds = _causal_daily_strategic_thresholds(state["strategic_score"], config)
    strategic_values = state["strategic_score"].to_numpy(dtype=float)
    strategic_raw = np.where(
        strategic_values >= strategic_thresholds["long_enter"].to_numpy(dtype=float),
        1,
        np.where(
            strategic_values <= strategic_thresholds["short_enter"].to_numpy(dtype=float),
            -1,
            0,
        ),
    ).astype(np.int8)
    strategic_stable = causal_asymmetric_hysteresis_state(
        strategic_values,
        long_enter=strategic_thresholds["long_enter"].to_numpy(dtype=float),
        short_enter=strategic_thresholds["short_enter"].to_numpy(dtype=float),
        long_exit=strategic_thresholds["long_exit"].to_numpy(dtype=float),
        short_exit=strategic_thresholds["short_exit"].to_numpy(dtype=float),
    )
    for column in strategic_thresholds.columns:
        state[f"strategic_{column}_threshold"] = strategic_thresholds[column]
    state["strategic_raw_state"] = strategic_raw
    state["strategic_state"] = strategic_stable
    state["strategic_boundary_margin"] = _state_boundary_margin(
        strategic_values,
        strategic_stable,
        long_enter=strategic_thresholds["long_enter"].to_numpy(dtype=float),
        short_enter=strategic_thresholds["short_enter"].to_numpy(dtype=float),
        long_exit=strategic_thresholds["long_exit"].to_numpy(dtype=float),
        short_exit=strategic_thresholds["short_exit"].to_numpy(dtype=float),
    )
    state["strategic_age_bars"] = _causal_age(strategic_stable)
    state["strategic_flip_rate_6h"] = _rolling_flip_rate(strategic_stable, 24)
    state["strategic_flip_rate_24h"] = _rolling_flip_rate(strategic_stable, 96)

    raw_threshold = config.direction_enter_threshold
    for layer in ("tactical", "entry"):
        values = state[f"{layer}_score"].to_numpy(dtype=float)
        raw = np.where(values >= raw_threshold, 1, np.where(values <= -raw_threshold, -1, 0)).astype(np.int8)
        stable = causal_hysteresis_state(
            values,
            enter_threshold=config.direction_enter_threshold,
            exit_threshold=config.direction_exit_threshold,
        )
        state[f"{layer}_raw_state"] = raw
        state[f"{layer}_state"] = stable
        state[f"{layer}_boundary_margin"] = _state_boundary_margin(
            values,
            stable,
            long_enter=config.direction_enter_threshold,
            short_enter=-config.direction_enter_threshold,
            long_exit=config.direction_exit_threshold,
            short_exit=-config.direction_exit_threshold,
        )
        state[f"{layer}_age_bars"] = _causal_age(stable)
        state[f"{layer}_flip_rate_6h"] = _rolling_flip_rate(stable, 24)
        state[f"{layer}_flip_rate_24h"] = _rolling_flip_rate(stable, 96)

    activity_values = state["activity_score"].to_numpy(dtype=float)
    activity_stable = causal_hysteresis_state(
        activity_values,
        enter_threshold=config.activity_enter_threshold,
        exit_threshold=config.activity_exit_threshold,
    )
    activity_raw = np.where(
        activity_values >= config.activity_enter_threshold,
        1,
        np.where(activity_values <= -config.activity_enter_threshold, -1, 0),
    ).astype(np.int8)
    state["activity_raw_state"] = activity_raw
    state["activity_state"] = activity_stable
    state["activity_boundary_margin"] = _state_boundary_margin(
        activity_values,
        activity_stable,
        long_enter=config.activity_enter_threshold,
        short_enter=-config.activity_enter_threshold,
        long_exit=config.activity_exit_threshold,
        short_exit=-config.activity_exit_threshold,
    )
    state["activity_age_bars"] = _causal_age(activity_stable)
    state["activity_flip_rate_6h"] = _rolling_flip_rate(activity_stable, 24)
    state["activity_flip_rate_24h"] = _rolling_flip_rate(activity_stable, 96)

    state["strategic_tactical_alignment"] = state["strategic_score"] * state["tactical_score"]
    state["tactical_entry_alignment"] = state["tactical_score"] * state["entry_score"]
    state["all_direction_alignment"] = (
        np.sign(state["strategic_score"]) * np.sign(state["tactical_score"])
        + np.sign(state["tactical_score"]) * np.sign(state["entry_score"])
        + np.sign(state["strategic_score"]) * np.sign(state["entry_score"])
    ) / 3.0
    state["long_pullback_setup"] = (
        state["strategic_score"].clip(lower=0)
        * (-state["tactical_score"]).clip(lower=0)
        * (state["entry_score"] + 1.0) / 2.0
    )
    state["short_pullback_setup"] = (
        (-state["strategic_score"]).clip(lower=0)
        * state["tactical_score"].clip(lower=0)
        * (1.0 - state["entry_score"]) / 2.0
    )
    state["trend_momentum_long"] = (
        state["strategic_score"].clip(lower=0)
        * state["tactical_score"].clip(lower=0)
        * state["entry_score"].clip(lower=0)
    )
    state["trend_momentum_short"] = (
        (-state["strategic_score"]).clip(lower=0)
        * (-state["tactical_score"]).clip(lower=0)
        * (-state["entry_score"]).clip(lower=0)
    )
    return state.replace([np.inf, -np.inf], np.nan)


def build_state_targets(
    state_frame: pd.DataFrame,
    config: MarketStateContinuityConfig,
) -> pd.DataFrame:
    targets = pd.DataFrame(index=state_frame.index)
    bars_per_hour = int(60 / config.decision_interval_minutes)
    for spec in config.targets:
        steps = int(spec.horizon_hours * bars_per_hour)
        current = pd.to_numeric(state_frame[spec.state_column], errors="coerce")
        future = current.shift(-steps)
        valid = current.notna() & future.notna()
        max_steps = max(steps, 1)
        transition_steps = _time_to_next_change(current.fillna(0).to_numpy(dtype=np.int8), max_steps)
        # Persistence means no intervening state change anywhere inside the horizon.
        # Endpoint equality alone would incorrectly count flip-away-and-return paths.
        uninterrupted = transition_steps > float(steps)
        targets[spec.target_id] = np.where(valid, uninterrupted.astype(float), np.nan)
        score_column = spec.state_column.replace("_state", "_score")
        if score_column in state_frame.columns:
            targets[f"{spec.target_id}_future_score"] = state_frame[score_column].shift(-steps)
            targets[f"{spec.target_id}_score_delta"] = state_frame[score_column].shift(-steps) - state_frame[score_column]
        transition_hours = transition_steps / float(bars_per_hour)
        targets[f"{spec.target_id}_time_to_change_hours"] = np.where(valid, transition_hours, np.nan)
    return targets


def state_cache_path(config: MarketStateContinuityConfig, year: int) -> Path:
    return config.cache_path / f"state_{int(year)}"


def _year_ranges(config: MarketStateContinuityConfig) -> list[int]:
    return list(range(pd.Timestamp(config.research_start).year, pd.Timestamp(config.research_end).year + 1))


def _build_year(
    loader: UnifiedOHLCVLoader,
    config: MarketStateContinuityConfig,
    year: int,
    *,
    force_rebuild: bool,
) -> Path:
    target = state_cache_path(config, year)
    manifest_path = target / "manifest.json"
    signature = _cache_signature(config)
    if manifest_path.exists() and not force_rebuild:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if int(manifest.get("schema_version", -1)) == CACHE_SCHEMA_VERSION and manifest.get("cache_signature") == signature:
                return target
        except (OSError, json.JSONDecodeError):
            pass

    year_start = pd.Timestamp(f"{year}-01-01 00:00:00")
    year_end = pd.Timestamp(f"{year}-12-31 23:59:59")
    read_start = max(pd.Timestamp(config.warmup_start), year_start - pd.Timedelta(days=config.feature_lookback_days))
    # Preserve the sealed 2026 boundary. Earlier years may read a short future
    # extension only to build labels at the end of the year.
    if year < pd.Timestamp(config.research_end).year:
        read_end = year_end + pd.Timedelta(hours=config.maximum_target_horizon_hours)
    else:
        read_end = year_end
    bars = loader.fetch_data_by_date_range(read_start, read_end)
    if bars.empty:
        raise RuntimeError(f"R03.3.3 no unified OHLCV data for {read_start} -> {read_end}")
    minute_grid, gap_stats = build_causal_minute_grid(bars.drop(columns=["source"], errors="ignore"), read_start, read_end)
    if minute_grid.empty:
        raise RuntimeError(f"R03.3.3 empty minute grid for year {year}")
    if float(gap_stats.get("gap_ratio", 1.0)) > 0.02:
        raise RuntimeError(
            f"R03.3.3 minute coverage gap too large for year {year}: "
            f"{float(gap_stats.get('gap_ratio', 1.0)):.4%}"
        )

    decision_index = pd.date_range(
        read_start.ceil(f"{config.decision_interval_minutes}min"),
        read_end.floor(f"{config.decision_interval_minutes}min"),
        freq=f"{config.decision_interval_minutes}min",
    )
    bundle = build_multitimeframe_feature_bundle(
        minute_grid,
        decision_index,
        structural_swing_bars_4h=config.structural_swing_bars_4h,
        feature_profile=LONG_CONTEXT_PROFILE,
    )
    universal_columns = _universal_columns(bundle.full_feature_columns)
    universal = bundle.frame.loc[:, list(universal_columns)].copy()
    state = build_state_frame(bundle.frame, config)
    targets = build_state_targets(state, config)

    keep = (decision_index >= year_start) & (decision_index <= year_end)
    selected_index = decision_index[keep]
    universal = universal.loc[selected_index]
    state = state.loc[selected_index]
    targets = targets.loc[selected_index]

    model_frame = pd.concat([universal, state], axis=1)
    model_frame = model_frame.loc[:, ~model_frame.columns.duplicated()]
    temp = target.with_name(target.name + ".part")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=True)
    decision_times_ns = datetime_index_to_ns(selected_index)
    decoded_years = ns_to_datetime(decision_times_ns).year
    if len(decision_times_ns) and not np.all(decoded_years == year):
        raise RuntimeError(
            f"R03.3.3 timestamp encoding mismatch for year {year}: "
            f"{sorted(set(int(value) for value in decoded_years))}"
        )
    np.save(temp / "decision_times_ns.npy", decision_times_ns, allow_pickle=False)
    np.save(temp / "features.npy", model_frame.to_numpy(dtype=np.float32, copy=True), allow_pickle=False)
    np.save(temp / "states.npy", state.to_numpy(dtype=np.float32, copy=True), allow_pickle=False)
    np.save(temp / "targets.npy", targets.to_numpy(dtype=np.float32, copy=True), allow_pickle=False)
    source_counts = {str(key): int(value) for key, value in bars["source"].value_counts().items()}
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_signature": signature,
        "year": year,
        "timestamp_unit": "ns",
        "timestamp_min": str(selected_index.min()) if len(selected_index) else None,
        "timestamp_max": str(selected_index.max()) if len(selected_index) else None,
        "rows": int(len(selected_index)),
        "feature_columns": list(model_frame.columns),
        "state_columns": list(state.columns),
        "target_columns": list(targets.columns),
        "universal_ohlcv_feature_columns": list(universal_columns),
        "source_counts": source_counts,
        "gap_stats": gap_stats,
        "read_start": str(read_start),
        "read_end": str(read_end),
        "config": config.to_dict(),
    }
    (temp / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if target.exists():
        shutil.rmtree(target)
    temp.replace(target)
    return target


def build_state_caches(
    loader: UnifiedOHLCVLoader,
    config: MarketStateContinuityConfig,
    *,
    force_rebuild: bool = False,
    progress: bool = True,
) -> list[Path]:
    config.cache_path.mkdir(parents=True, exist_ok=True)
    years = _year_ranges(config)
    reporter = ProgressReporter("[R03.3.3 state cache] years", len(years), every=1, enabled=progress)
    outputs: list[Path] = []
    for index, year in enumerate(years, start=1):
        outputs.append(_build_year(loader, config, year, force_rebuild=force_rebuild))
        reporter.update(index)
    reporter.close()
    return outputs


@dataclass(frozen=True)
class StateYearShard:
    path: Path
    year: int
    decision_times_ns: np.ndarray
    features: np.ndarray
    states: np.ndarray
    targets: np.ndarray
    feature_columns: tuple[str, ...]
    state_columns: tuple[str, ...]
    target_columns: tuple[str, ...]

    @property
    def state_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.state_columns)}

    @property
    def target_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.target_columns)}


def load_state_year_shard(path: str | Path) -> StateYearShard:
    target = Path(path)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported R03.3.3 state cache: {target}")
    if manifest.get("timestamp_unit") != "ns":
        raise RuntimeError(f"R03.3.3 cache timestamp unit is not ns: {target}")
    year = int(manifest["year"])
    decision_times_ns = np.load(target / "decision_times_ns.npy", mmap_mode="r")
    decoded = ns_to_datetime(decision_times_ns)
    if len(decoded):
        decoded_years = np.asarray(decoded.year, dtype=np.int16)
        if not np.all(decoded_years == year):
            raise RuntimeError(
                f"R03.3.3 cache timestamp/year mismatch path={target} "
                f"manifest_year={year} decoded_years={sorted(set(int(value) for value in decoded_years))}"
            )
        if not decoded.is_monotonic_increasing or not decoded.is_unique:
            raise RuntimeError(f"R03.3.3 cache timestamps must be increasing and unique: {target}")
    return StateYearShard(
        path=target,
        year=year,
        decision_times_ns=decision_times_ns,
        features=np.load(target / "features.npy", mmap_mode="r"),
        states=np.load(target / "states.npy", mmap_mode="r"),
        targets=np.load(target / "targets.npy", mmap_mode="r"),
        feature_columns=tuple(manifest["feature_columns"]),
        state_columns=tuple(manifest["state_columns"]),
        target_columns=tuple(manifest["target_columns"]),
    )


def list_state_caches(config: MarketStateContinuityConfig) -> list[Path]:
    return sorted(path for path in config.cache_path.glob("state_????") if (path / "manifest.json").exists())
