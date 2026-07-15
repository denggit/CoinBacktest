#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal multi-scale liquidity-map and sweep/reclaim features.

The feature builder intentionally keeps liquidity as an optional explanatory
layer.  It does not require a candidate to sweep a level and it never treats a
plain local low as liquidity by itself.  Levels are defined before the current
closed bar from several independent horizons:

* micro rolling floors (5/15/30/60 bars by default);
* prior completed day, week and session lows;
* current-session prior low;
* causally confirmed 1H/4H pivot lows whose right-hand confirmation bars have
  already closed.

The current closed bar may then approach, sweep, reclaim, or accept below those
pre-existing levels.  Trade-bar order flow is used only after the structural
state is known.  Future bars are never used as features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from src.research_common.progress import ProgressReporter
except Exception:  # pragma: no cover
    ProgressReporter = None  # type: ignore[assignment]

EPS = 1e-12
MICRO_GROUP = "L1_micro_liquidity"
MACRO_GROUP = "L2_macro_liquidity"
SWEEP_GROUP = "L3_sweep_reclaim_orderflow"


@dataclass(frozen=True)
class LiquidityFeatureBuildResult:
    frame: pd.DataFrame
    dictionary: pd.DataFrame
    diagnostics: pd.DataFrame
    group_membership: pd.DataFrame


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _safe_divide(numerator: np.ndarray | pd.Series, denominator: np.ndarray | pd.Series) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.zeros(np.broadcast_shapes(num.shape, den.shape), dtype=float)
    return np.divide(num, den, out=out, where=np.isfinite(den) & (np.abs(den) > EPS))


def _infer_bar_delta(index: pd.DatetimeIndex) -> pd.Timedelta:
    diffs = index.to_series().diff().dropna()
    positive = diffs[diffs > pd.Timedelta(0)]
    return positive.median() if not positive.empty else pd.Timedelta(minutes=1)


def _session_ids(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return monotonically increasing session ids, ordinal and hour bucket.

    The project data is normally UTC+8.  The builder deliberately uses the
    timestamps as stored instead of applying a hidden timezone conversion.
    Sessions are fixed 8-hour blocks: 00-08, 08-16 and 16-24.
    """

    normalized = index.normalize()
    day_ord = (normalized.view("i8") // pd.Timedelta(days=1).value).astype(np.int64)
    bucket = (index.hour.to_numpy(dtype=np.int16) // 8).astype(np.int16)
    session_id = day_ord * 3 + bucket
    return session_id, day_ord, bucket


def _segment_prior_min(low: pd.Series, segment_id: np.ndarray) -> pd.Series:
    groups = pd.Series(segment_id, index=low.index)
    shifted = low.groupby(groups, sort=False).shift(1)
    return shifted.groupby(groups, sort=False).cummin()


def _pivot_level_series(
    bars: pd.DataFrame,
    *,
    minutes: int,
    left_bars: int,
    right_bars: int,
) -> tuple[pd.Series, np.ndarray, pd.Series, pd.Series, pd.DataFrame]:
    """Build a full 1m-axis series of causally confirmed HTF pivot lows."""

    if minutes < 2 or left_bars < 1 or right_bars < 1:
        raise ValueError("HTF pivot configuration must be positive")
    index = pd.DatetimeIndex(bars.index)
    bar_delta = _infer_bar_delta(index)
    rule = f"{int(minutes)}min"
    htf_low = _numeric(bars, "low").resample(rule, label="left", closed="left").min().dropna()
    if len(htf_low) < left_bars + right_bars + 1:
        empty = pd.Series(np.nan, index=index, dtype=float)
        return empty, np.zeros(len(index), dtype=np.int64), empty.copy(), empty.copy(), pd.DataFrame()

    values = htf_low.to_numpy(dtype=float)
    pivot = np.ones(len(values), dtype=bool)
    pivot[:left_bars] = False
    pivot[len(values) - right_bars :] = False
    for lag in range(1, left_bars + 1):
        pivot &= values < np.roll(values, lag)
    for lead in range(1, right_bars + 1):
        pivot &= values <= np.roll(values, -lead)
    pivot_positions = np.flatnonzero(pivot)
    if not len(pivot_positions):
        empty = pd.Series(np.nan, index=index, dtype=float)
        return empty, np.zeros(len(index), dtype=np.int64), empty.copy(), empty.copy(), pd.DataFrame()

    htf_delta = pd.Timedelta(minutes=int(minutes))
    formed_time = htf_low.index[pivot_positions]
    available_time = formed_time + (right_bars + 1) * htf_delta
    levels = values[pivot_positions]
    event_table = pd.DataFrame(
        {
            "pivot_id": np.arange(1, len(pivot_positions) + 1, dtype=np.int64),
            "pivot_level": levels,
            "pivot_formed_time": formed_time,
            "pivot_available_time": available_time,
        }
    )

    bar_available = (index + bar_delta).view("i8")
    event_available = pd.DatetimeIndex(available_time).view("i8")
    selected = np.searchsorted(event_available, bar_available, side="right") - 1
    valid = selected >= 0
    level_array = np.full(len(index), np.nan, dtype=float)
    id_array = np.zeros(len(index), dtype=np.int64)
    formed_array = np.full(len(index), np.datetime64("NaT"), dtype="datetime64[ns]")
    available_array = np.full(len(index), np.datetime64("NaT"), dtype="datetime64[ns]")
    if valid.any():
        take = selected[valid]
        level_array[valid] = levels[take]
        id_array[valid] = event_table["pivot_id"].to_numpy(dtype=np.int64)[take]
        formed_array[valid] = pd.DatetimeIndex(formed_time).to_numpy()[take]
        available_array[valid] = pd.DatetimeIndex(available_time).to_numpy()[take]
    return (
        pd.Series(level_array, index=index, dtype=float),
        id_array,
        pd.Series(formed_array, index=index),
        pd.Series(available_array, index=index),
        event_table,
    )


def _feature_row(name: str, group: str, source: str, description: str, causal_rule: str) -> dict[str, object]:
    return {
        "feature": name,
        "feature_group": group,
        "source": source,
        "description": description,
        "causal_rule": causal_rule,
    }


def build_multiscale_liquidity_features(
    bars: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    micro_windows: Sequence[int] = (5, 15, 30, 60),
    equal_low_tolerance_bp: float = 8.0,
    approach_tolerance_bp: float = 15.0,
    htf_pivot_minutes: Sequence[int] = (60, 240),
    htf_pivot_left_bars: int = 2,
    htf_pivot_right_bars: int = 2,
    show_progress: bool = False,
) -> LiquidityFeatureBuildResult:
    """Create causal liquidity-map and sweep/reclaim features for candidates."""

    required_bars = {"open", "high", "low", "close", "notional", "trades_count", "delta_notional"}
    missing = sorted(required_bars.difference(bars.columns))
    if missing:
        raise RuntimeError(f"liquidity feature builder missing trade-bar fields: {missing}")
    required_candidates = {"event_id", "extreme_pos", "feature_available_time"}
    missing_candidates = sorted(required_candidates.difference(candidates.columns))
    if missing_candidates:
        raise RuntimeError(f"liquidity feature builder missing candidate fields: {missing_candidates}")

    index = pd.DatetimeIndex(bars.index)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise RuntimeError("bars index must be unique and increasing")
    positions = pd.to_numeric(candidates["extreme_pos"], errors="raise").to_numpy(dtype=np.int64)
    if positions.min(initial=0) < 0 or positions.max(initial=0) >= len(bars):
        raise RuntimeError("candidate positions are outside loaded bars")

    low = _numeric(bars, "low")
    high = _numeric(bars, "high")
    close = _numeric(bars, "close")
    open_ = _numeric(bars, "open")
    notional = _numeric(bars, "notional")
    trades = _numeric(bars, "trades_count")
    volume = _numeric(bars, "volume")
    delta = _numeric(bars, "delta_notional")
    buy = _numeric(bars, "buy_notional")
    sell = _numeric(bars, "sell_notional")
    large_buy = _numeric(bars, "large_buy_notional")
    large_sell = _numeric(bars, "large_sell_notional")
    large_delta = _numeric(bars, "large_delta_notional")

    n = len(bars)
    output: dict[str, np.ndarray] = {}
    dictionary: list[dict[str, object]] = []
    membership: list[dict[str, str]] = []
    source_diagnostics: list[dict[str, object]] = []

    # Full-axis aggregates used by L3.  Float32/int16 are sufficient and keep
    # peak memory bounded on the multi-year 1m dataset.
    any_sweep = np.zeros(n, dtype=bool)
    any_reclaim = np.zeros(n, dtype=bool)
    any_accept = np.zeros(n, dtype=bool)
    micro_sweep = np.zeros(n, dtype=bool)
    macro_sweep = np.zeros(n, dtype=bool)
    micro_reclaim = np.zeros(n, dtype=bool)
    macro_reclaim = np.zeros(n, dtype=bool)
    sweep_count = np.zeros(n, dtype=np.int16)
    reclaim_count = np.zeros(n, dtype=np.int16)
    near_count_10 = np.zeros(n, dtype=np.int16)
    near_count_25 = np.zeros(n, dtype=np.int16)
    max_sweep_depth = np.zeros(n, dtype=np.float32)
    max_reclaim_strength = np.zeros(n, dtype=np.float32)
    swept_strength_sum = np.zeros(n, dtype=np.float32)
    selected_sweep_strength = np.full(n, -np.inf, dtype=np.float32)
    selected_sweep_level = np.full(n, np.nan, dtype=np.float64)

    approach_tol = float(approach_tolerance_bp)
    equal_tol = float(equal_low_tolerance_bp) / 10_000.0
    progress_total = len(tuple(micro_windows)) + 4 + len(tuple(htf_pivot_minutes)) + 1
    reporter = (
        ProgressReporter("[liquidity] causal level/process features", total=progress_total, every=1)
        if ProgressReporter is not None and show_progress
        else None
    )
    progress_done = 0

    def progress_step() -> None:
        nonlocal progress_done
        progress_done += 1
        if reporter is not None and progress_done < progress_total:
            reporter.update(progress_done)

    def add_output(name: str, values: np.ndarray | pd.Series, group: str, source: str, description: str, causal_rule: str) -> None:
        array = np.asarray(values, dtype=float)
        output[name] = array[positions].astype(np.float32, copy=False)
        dictionary.append(_feature_row(name, group, source, description, causal_rule))
        membership.append({"feature": name, "feature_group": group})

    def consume_level(
        *,
        prefix: str,
        group: str,
        source: str,
        level: pd.Series,
        strength: np.ndarray | pd.Series | float,
        untouched_prior: np.ndarray | pd.Series,
        age_bars: np.ndarray | pd.Series,
        causal_rule: str,
        is_micro: bool,
    ) -> None:
        nonlocal any_sweep, any_reclaim, any_accept
        level_values = pd.to_numeric(level, errors="coerce").to_numpy(dtype=float, copy=False)
        strength_values = np.broadcast_to(np.asarray(strength, dtype=float), (n,))
        untouched_values = np.broadcast_to(np.asarray(untouched_prior, dtype=float), (n,))
        age_values = np.broadcast_to(np.asarray(age_bars, dtype=float), (n,))
        close_values = close.to_numpy(dtype=float, copy=False)
        low_values = low.to_numpy(dtype=float, copy=False)
        finite = np.isfinite(level_values) & (level_values > 0)
        distance_close_bp = np.where(finite, (close_values / level_values - 1.0) * 10_000.0, np.nan)
        penetration_bp = np.where(finite, np.maximum((level_values - low_values) / level_values * 10_000.0, 0.0), np.nan)
        reclaim_bp = np.where(finite, (close_values - level_values) / level_values * 10_000.0, np.nan)
        approach = finite & (low_values <= level_values * (1.0 + approach_tol / 10_000.0))
        sweep = finite & (low_values < level_values)
        reclaim = sweep & (close_values >= level_values)
        accept_below = sweep & (close_values < level_values)

        add_output(f"{prefix}_distance_close_bp", distance_close_bp, group, source, "current close distance from pre-existing liquidity level", causal_rule)
        add_output(f"{prefix}_penetration_bp", penetration_bp, group, source, "current low penetration below pre-existing level", causal_rule)
        add_output(f"{prefix}_reclaim_bp", reclaim_bp, group, source, "current close reclaim distance above level", causal_rule)
        add_output(f"{prefix}_approach", approach.astype(float), group, source, "current bar approaches the level", causal_rule)
        add_output(f"{prefix}_sweep", sweep.astype(float), group, source, "current low trades below the pre-existing level", causal_rule)
        add_output(f"{prefix}_reclaim", reclaim.astype(float), group, source, "current closed bar sweeps and closes back above level", causal_rule)
        add_output(f"{prefix}_accept_below", accept_below.astype(float), group, source, "current closed bar remains below swept level", causal_rule)
        add_output(f"{prefix}_untouched_prior", untouched_values, group, source, "level had not been traded below before current bar", causal_rule)
        add_output(f"{prefix}_age_bars", age_values, group, source, "bars since level became available", causal_rule)
        add_output(f"{prefix}_strength", strength_values, group, source, "predeclared structural importance score", causal_rule)

        near_count_10[:] += (finite & (np.abs(distance_close_bp) <= 10.0)).astype(np.int16)
        near_count_25[:] += (finite & (np.abs(distance_close_bp) <= 25.0)).astype(np.int16)
        any_sweep |= sweep
        any_reclaim |= reclaim
        any_accept |= accept_below
        sweep_count[:] += sweep.astype(np.int16)
        reclaim_count[:] += reclaim.astype(np.int16)
        max_sweep_depth[:] = np.maximum(max_sweep_depth, np.nan_to_num(penetration_bp, nan=0.0).astype(np.float32))
        max_reclaim_strength[:] = np.maximum(max_reclaim_strength, np.where(reclaim, np.maximum(reclaim_bp, 0.0), 0.0).astype(np.float32))
        swept_strength_sum[:] += np.where(sweep, strength_values, 0.0).astype(np.float32)
        choose = sweep & (strength_values > selected_sweep_strength)
        selected_sweep_strength[choose] = strength_values[choose].astype(np.float32)
        selected_sweep_level[choose] = level_values[choose]
        if is_micro:
            micro_sweep[:] |= sweep
            micro_reclaim[:] |= reclaim
        else:
            macro_sweep[:] |= sweep
            macro_reclaim[:] |= reclaim
        source_diagnostics.append(
            {
                "source": source,
                "feature_group": group,
                "level_non_null_rows": int(finite.sum()),
                "candidate_non_null_coverage": float(np.isfinite(level_values[positions]).mean()) if len(positions) else np.nan,
                "sweep_rows": int(sweep.sum()),
                "reclaim_rows": int(reclaim.sum()),
                "accept_below_rows": int(accept_below.sum()),
            }
        )

    # Micro rolling floors.  shift(1) guarantees the level exists before the
    # current closed bar.  Equal-low density is a causal proxy, not a future
    # cluster label.
    prior_low = low.shift(1)
    for window_raw in micro_windows:
        window = int(window_raw)
        if window < 3:
            raise ValueError("micro liquidity windows must be >= 3")
        level = prior_low.rolling(window, min_periods=window).min()
        near_historical_floor = (prior_low <= level * (1.0 + equal_tol)).astype(float)
        equal_density = near_historical_floor.rolling(window, min_periods=max(2, window // 3)).mean()
        equal_count_proxy = equal_density * float(window)
        strength = np.log1p(float(window)) + np.clip(equal_count_proxy.to_numpy(dtype=float), 0.0, 10.0) * 0.25
        add_output(
            f"micro_w{window}_equal_low_count_proxy",
            equal_count_proxy,
            MICRO_GROUP,
            f"prior_{window}m_floor",
            "causal density of repeated lows near rolling floor",
            "uses lows through t-1 only",
        )
        consume_level(
            prefix=f"micro_w{window}",
            group=MICRO_GROUP,
            source=f"prior_{window}m_floor",
            level=level,
            strength=strength,
            untouched_prior=np.ones(n, dtype=float),
            age_bars=np.full(n, float(window)),
            causal_rule="rolling level uses bars through t-1; current closed bar only tests/sweeps it",
            is_micro=True,
        )
        progress_step()

    # Macro calendar/session levels.
    session_id, day_id, _ = _session_ids(index)
    day_series = pd.Series(day_id, index=index)
    week_period = index.to_period("W-SUN")
    week_codes, week_uniques = pd.factorize(week_period, sort=False)
    session_series = pd.Series(session_id, index=index)

    day_lows = low.groupby(day_series, sort=False).min()
    prev_day_lookup = day_lows.shift(1)
    prev_day_level = day_series.map(prev_day_lookup)
    day_start_pos = pd.Series(np.arange(n), index=index).groupby(day_series, sort=False).transform("min")
    day_age = np.arange(n) - day_start_pos.to_numpy(dtype=np.int64)
    prev_day_prior_min = _segment_prior_min(low, day_id)
    prev_day_untouched = prev_day_prior_min.isna() | (prev_day_prior_min >= prev_day_level)
    consume_level(
        prefix="macro_prev_day",
        group=MACRO_GROUP,
        source="previous_completed_day_low",
        level=prev_day_level,
        strength=3.5,
        untouched_prior=prev_day_untouched,
        age_bars=day_age,
        causal_rule="previous day low becomes available at current day start",
        is_micro=False,
    )
    progress_step()

    week_series = pd.Series(week_codes, index=index)
    week_lows = low.groupby(week_series, sort=False).min()
    prev_week_lookup = week_lows.shift(1)
    prev_week_level = week_series.map(prev_week_lookup)
    week_start_pos = pd.Series(np.arange(n), index=index).groupby(week_series, sort=False).transform("min")
    week_age = np.arange(n) - week_start_pos.to_numpy(dtype=np.int64)
    prev_week_prior_min = _segment_prior_min(low, week_codes)
    prev_week_untouched = prev_week_prior_min.isna() | (prev_week_prior_min >= prev_week_level)
    consume_level(
        prefix="macro_prev_week",
        group=MACRO_GROUP,
        source="previous_completed_week_low",
        level=prev_week_level,
        strength=5.0,
        untouched_prior=prev_week_untouched,
        age_bars=week_age,
        causal_rule="previous week low becomes available at current week start",
        is_micro=False,
    )
    progress_step()

    session_lows = low.groupby(session_series, sort=False).min()
    prev_session_lookup = session_lows.shift(1)
    prev_session_level = session_series.map(prev_session_lookup)
    session_start_pos = pd.Series(np.arange(n), index=index).groupby(session_series, sort=False).transform("min")
    session_age = np.arange(n) - session_start_pos.to_numpy(dtype=np.int64)
    prev_session_prior_min = _segment_prior_min(low, session_id)
    prev_session_untouched = prev_session_prior_min.isna() | (prev_session_prior_min >= prev_session_level)
    consume_level(
        prefix="macro_prev_session",
        group=MACRO_GROUP,
        source="previous_completed_8h_session_low",
        level=prev_session_level,
        strength=2.5,
        untouched_prior=prev_session_untouched,
        age_bars=session_age,
        causal_rule="previous 8h session low is fixed before current session",
        is_micro=False,
    )
    progress_step()

    current_session_prior = low.groupby(session_series, sort=False).shift(1).groupby(session_series, sort=False).cummin()
    consume_level(
        prefix="macro_current_session",
        group=MACRO_GROUP,
        source="current_session_prior_low",
        level=current_session_prior,
        strength=2.0,
        untouched_prior=np.ones(n, dtype=float),
        age_bars=session_age,
        causal_rule="current session running low uses bars through t-1 only",
        is_micro=False,
    )
    progress_step()

    # Confirmed HTF pivot lows.  The pivot itself uses right-side HTF bars, but
    # the level is not exposed to 1m candidates until all confirmation bars are
    # closed.  This is the same available-time principle used elsewhere in the
    # project.
    feature_available = pd.to_datetime(candidates["feature_available_time"])
    htf_available_violations = 0
    for minutes_raw in htf_pivot_minutes:
        minutes = int(minutes_raw)
        level, pivot_id, _, available, events = _pivot_level_series(
            bars,
            minutes=minutes,
            left_bars=int(htf_pivot_left_bars),
            right_bars=int(htf_pivot_right_bars),
        )
        pivot_segment = pivot_id
        prior_min = _segment_prior_min(low, pivot_segment)
        untouched = prior_min.isna() | (prior_min >= level)
        available_ns = pd.to_datetime(available).astype("int64", copy=False).to_numpy()
        index_available_ns = (index + _infer_bar_delta(index)).view("i8")
        event_available_pos = np.searchsorted(index_available_ns, available_ns, side="left")
        age = np.arange(n) - np.where(pivot_id > 0, event_available_pos, np.arange(n))
        candidate_available = pd.to_datetime(available.iloc[positions]).to_numpy()
        feature_available_values = pd.to_datetime(feature_available).to_numpy()
        htf_available_violations += int(np.sum(candidate_available > feature_available_values))
        consume_level(
            prefix=f"macro_pivot_{minutes}m",
            group=MACRO_GROUP,
            source=f"confirmed_{minutes}m_pivot_low",
            level=level,
            strength=3.0 if minutes <= 60 else 4.5,
            untouched_prior=untouched,
            age_bars=np.maximum(age, 0),
            causal_rule=f"pivot available only after {htf_pivot_right_bars} right-side {minutes}m bars close",
            is_micro=False,
        )
        progress_step()

    # Sweep/order-flow interaction layer.  Volume and aggressive sell bursts do
    # not prove stop orders; they are observable proxies only.
    prior_notional = notional.shift(1).rolling(60, min_periods=20).mean()
    prior_trades = trades.shift(1).rolling(60, min_periods=20).mean()
    prior_volume = volume.shift(1).rolling(60, min_periods=20).mean()
    notional_intensity = _safe_divide(notional, prior_notional)
    trades_intensity = _safe_divide(trades, prior_trades)
    volume_intensity = _safe_divide(volume, prior_volume)
    delta_ratio = _safe_divide(delta, notional)
    negative_delta_ratio = np.maximum(-delta_ratio, 0.0)
    aggressive_sell_ratio = _safe_divide(sell, buy + sell)
    large_sell_ratio = _safe_divide(large_sell, large_buy + large_sell)
    large_negative_delta_ratio = np.maximum(-_safe_divide(large_delta, notional), 0.0)
    bar_range = (high - low).clip(lower=0.0)
    close_in_bar = _safe_divide(close - low, bar_range)

    pos_index = np.arange(n, dtype=float)
    last_sweep_pos = pd.Series(np.where(any_sweep, pos_index, np.nan), index=index).ffill().to_numpy(dtype=float)
    last_reclaim_pos = pd.Series(np.where(any_reclaim, pos_index, np.nan), index=index).ffill().to_numpy(dtype=float)
    bars_since_sweep = pos_index - last_sweep_pos
    bars_since_reclaim = pos_index - last_reclaim_pos
    last_sweep_neg_delta = pd.Series(np.where(any_sweep, negative_delta_ratio, np.nan), index=index).ffill().to_numpy(dtype=float)
    last_sweep_level = pd.Series(np.where(any_sweep, selected_sweep_level, np.nan), index=index).ffill().to_numpy(dtype=float)
    last_sweep_strength = pd.Series(np.where(any_sweep, np.maximum(selected_sweep_strength, 0.0), np.nan), index=index).ffill().to_numpy(dtype=float)
    post_sweep_sell_decay = last_sweep_neg_delta - negative_delta_ratio
    post_sweep_price_recovery_bp = np.where(
        np.isfinite(last_sweep_level) & (last_sweep_level > 0),
        (close.to_numpy(dtype=float) / last_sweep_level - 1.0) * 10_000.0,
        np.nan,
    )
    delta_3 = delta.rolling(3, min_periods=1).sum()
    notional_3 = notional.rolling(3, min_periods=1).sum()
    delta_5 = delta.rolling(5, min_periods=1).sum()
    notional_5 = notional.rolling(5, min_periods=1).sum()
    buy_recovery_3 = _safe_divide(delta_3, notional_3)
    buy_recovery_5 = _safe_divide(delta_5, notional_5)

    interaction_features: dict[str, tuple[np.ndarray, str]] = {
        "liq_any_sweep": (any_sweep.astype(float), "any pre-existing liquidity level swept on current closed bar"),
        "liq_any_reclaim": (any_reclaim.astype(float), "any swept level reclaimed by current close"),
        "liq_any_accept_below": (any_accept.astype(float), "any swept level accepted below by current close"),
        "liq_micro_sweep": (micro_sweep.astype(float), "micro liquidity swept"),
        "liq_macro_sweep": (macro_sweep.astype(float), "macro liquidity swept"),
        "liq_micro_reclaim": (micro_reclaim.astype(float), "micro liquidity sweep reclaimed"),
        "liq_macro_reclaim": (macro_reclaim.astype(float), "macro liquidity sweep reclaimed"),
        "liq_swept_level_count": (sweep_count.astype(float), "number of distinct tracked levels swept"),
        "liq_reclaimed_level_count": (reclaim_count.astype(float), "number of distinct swept levels reclaimed"),
        "liq_near_level_count_10bp": (near_count_10.astype(float), "tracked levels within 10bp of close"),
        "liq_near_level_count_25bp": (near_count_25.astype(float), "tracked levels within 25bp of close"),
        "liq_max_sweep_depth_bp": (max_sweep_depth, "deepest current liquidity penetration"),
        "liq_max_reclaim_strength_bp": (max_reclaim_strength, "largest close reclaim above a swept level"),
        "liq_swept_strength_sum": (swept_strength_sum, "sum of structural strengths of swept levels"),
        "liq_notional_intensity_60": (notional_intensity, "current notional versus prior 60-bar mean"),
        "liq_trades_intensity_60": (trades_intensity, "current trade count versus prior 60-bar mean"),
        "liq_volume_intensity_60": (volume_intensity, "current volume versus prior 60-bar mean"),
        "liq_negative_delta_ratio": (negative_delta_ratio, "observable aggressive sell imbalance"),
        "liq_aggressive_sell_ratio": (aggressive_sell_ratio, "aggressive sell notional share"),
        "liq_large_sell_ratio": (large_sell_ratio, "large aggressive sell share"),
        "liq_large_negative_delta_ratio": (large_negative_delta_ratio, "large-trade negative delta ratio"),
        "liq_sweep_sell_burst": (any_sweep.astype(float) * negative_delta_ratio * np.clip(notional_intensity, 0.0, 20.0), "sweep times aggressive-sell and activity burst"),
        "liq_sweep_large_sell_burst": (any_sweep.astype(float) * large_negative_delta_ratio * np.clip(notional_intensity, 0.0, 20.0), "sweep times large-sell burst"),
        "liq_reclaim_absorption_proxy": (any_reclaim.astype(float) * negative_delta_ratio * np.clip(close_in_bar, 0.0, 1.0), "negative aggressive flow with close reclaim"),
        "liq_accept_below_sell_efficiency": (any_accept.astype(float) * negative_delta_ratio * np.maximum(max_sweep_depth, 0.0), "continued sell pressure with accepted price displacement"),
        "liq_bars_since_sweep": (bars_since_sweep, "bars since the latest observed sweep"),
        "liq_bars_since_reclaim": (bars_since_reclaim, "bars since the latest observed sweep reclaim"),
        "liq_last_sweep_strength": (last_sweep_strength, "strength of latest swept level"),
        "liq_post_sweep_sell_decay": (post_sweep_sell_decay, "latest sweep sell imbalance minus current imbalance"),
        "liq_post_sweep_price_recovery_bp": (post_sweep_price_recovery_bp, "current close recovery from latest swept level"),
        "liq_post_sweep_delta_recovery_3": (buy_recovery_3, "3-bar cumulative delta ratio after/around sweep"),
        "liq_post_sweep_delta_recovery_5": (buy_recovery_5, "5-bar cumulative delta ratio after/around sweep"),
    }
    for name, (values, description) in interaction_features.items():
        add_output(
            name,
            values,
            SWEEP_GROUP,
            "multiscale_liquidity_and_trade_bar",
            description,
            "current and prior closed bars only; no inferred stop-order identity",
        )
    progress_step()
    if reporter is not None:
        reporter.close()

    feature_frame = candidates.reset_index(drop=True).copy()
    feature_frame = pd.concat([feature_frame, pd.DataFrame(output)], axis=1)
    dictionary_frame = pd.DataFrame(dictionary)
    membership_frame = pd.DataFrame(membership)
    diagnostics = pd.DataFrame(source_diagnostics)
    aggregate = pd.DataFrame(
        [
            {
                "source": "aggregate",
                "feature_group": "ALL",
                "level_non_null_rows": int(len(bars)),
                "candidate_non_null_coverage": float(feature_frame[dictionary_frame["feature"].tolist()].notna().any(axis=1).mean()),
                "sweep_rows": int(any_sweep.sum()),
                "reclaim_rows": int(any_reclaim.sum()),
                "accept_below_rows": int(any_accept.sum()),
                "candidate_rows": int(len(candidates)),
                "micro_feature_count": int((membership_frame["feature_group"] == MICRO_GROUP).sum()),
                "macro_feature_count": int((membership_frame["feature_group"] == MACRO_GROUP).sum()),
                "sweep_feature_count": int((membership_frame["feature_group"] == SWEEP_GROUP).sum()),
                "htf_available_time_violations": int(htf_available_violations),
            }
        ]
    )
    diagnostics = pd.concat([diagnostics, aggregate], ignore_index=True, sort=False)
    return LiquidityFeatureBuildResult(feature_frame, dictionary_frame, diagnostics, membership_frame)
