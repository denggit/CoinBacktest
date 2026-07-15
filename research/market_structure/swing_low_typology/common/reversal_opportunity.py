#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal event-level reversal-opportunity research helpers.

The module deliberately avoids the expensive event-by-event 315-column feature
builder used by research 02-04.  Features are calculated once on the full time
axis with causal rolling operations and then gathered at candidate positions.
High-timeframe context is joined by bar *available time*, never by bar start.

Forward labels are research metadata only.  They start from the next-bar open
and inspect future closed-bar closes; future high/low values are never used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from src.research_common.progress import ProgressReporter
except Exception:  # pragma: no cover
    ProgressReporter = None  # type: ignore[assignment]

EPS = 1e-12
CORE_WINDOWS: tuple[int, ...] = (5, 10, 15, 30, 60, 120, 240)
HTF_MINUTES: tuple[int, ...] = (5, 15, 60)
ADVERSE_LEVELS_PCT: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
FEATURE_GROUP_ORDER: tuple[str, ...] = ("M0_core", "M1_session", "M2_causal_htf")


@dataclass(frozen=True)
class FeatureBuildResult:
    frame: pd.DataFrame
    dictionary: pd.DataFrame
    group_membership: pd.DataFrame
    alignment_audit: pd.DataFrame


def _numeric(frame: pd.DataFrame, column: str, *, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.full(len(frame), default, dtype=float), index=frame.index, name=column)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.astype(float).where(denominator.abs() > EPS)
    return numerator.astype(float) / den


def _take_float(series: pd.Series, positions: np.ndarray) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=False)
    return values[positions].astype(np.float32, copy=False)


def _feature_row(
    feature: str,
    group: str,
    description: str,
    source: str,
    available_rule: str = "current closed 1m bar or older",
) -> dict[str, object]:
    return {
        "feature": feature,
        "feature_group": group,
        "description": description,
        "source": source,
        "available_rule": available_rule,
    }


def _session_codes(index: pd.DatetimeIndex) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Return fixed UTC+8/local-clock activity buckets and causal session date.

    The project trade-bar timestamps are used as stored.  No timezone conversion
    is performed.  Buckets are context features, not claims about exact exchange
    opening hours or daylight-saving-time schedules.
    """

    hour = index.hour.to_numpy()
    code = np.select(
        [((hour >= 21) | (hour < 4)), ((hour >= 8) & (hour < 15)), ((hour >= 15) & (hour < 21))],
        [3, 1, 2],
        default=0,
    ).astype(np.int8)
    anchor = index.normalize()
    # The 00:00-03:59 part of the US-active bucket belongs to the prior session date.
    prior_mask = (code == 3) & (hour < 4)
    if prior_mask.any():
        values = anchor.to_numpy(copy=True)
        values[prior_mask] = values[prior_mask] - np.timedelta64(1, "D")
        anchor = pd.DatetimeIndex(values)
    return code, anchor


def _build_session_features(
    bars: pd.DataFrame,
    positions: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    index = pd.DatetimeIndex(bars.index)
    open_ = _numeric(bars, "open")
    high = _numeric(bars, "high")
    low = _numeric(bars, "low")
    close = _numeric(bars, "close")
    notional = _numeric(bars, "notional")
    delta = _numeric(bars, "delta_notional")

    hour_float = index.hour.to_numpy(dtype=float) + index.minute.to_numpy(dtype=float) / 60.0
    dow = index.dayofweek.to_numpy(dtype=float)
    code, session_date = _session_codes(index)
    session_key = pd.MultiIndex.from_arrays([session_date, code])

    bar_number = pd.Series(np.arange(len(bars), dtype=np.int64), index=index).groupby(session_key).cumcount() + 1
    session_open = open_.groupby(session_key).transform("first")
    session_high = high.groupby(session_key).cummax()
    session_low = low.groupby(session_key).cummin()
    session_notional = notional.groupby(session_key).cumsum()
    session_delta = delta.groupby(session_key).cumsum()

    range_position = _safe_ratio(close - session_low, session_high - session_low).clip(-2.0, 3.0)
    features: dict[str, np.ndarray] = {
        "hour_sin": np.sin(2.0 * np.pi * hour_float / 24.0).astype(np.float32)[positions],
        "hour_cos": np.cos(2.0 * np.pi * hour_float / 24.0).astype(np.float32)[positions],
        "weekday_sin": np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32)[positions],
        "weekday_cos": np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32)[positions],
        "is_weekend": (dow >= 5).astype(np.float32)[positions],
        "session_off_hours": (code == 0).astype(np.float32)[positions],
        "session_asia_clock": (code == 1).astype(np.float32)[positions],
        "session_europe_clock": (code == 2).astype(np.float32)[positions],
        "session_us_clock": (code == 3).astype(np.float32)[positions],
        "session_bar_number": _take_float(bar_number, positions),
        "session_return_from_open": _take_float(_safe_ratio(close, session_open) - 1.0, positions),
        "session_range_position": _take_float(range_position, positions),
        "session_cum_delta_ratio": _take_float(_safe_ratio(session_delta, session_notional), positions),
        "session_cum_notional_log": _take_float(np.log1p(session_notional.clip(lower=0.0)), positions),
    }
    dictionary = [
        _feature_row(name, "M1_session", name.replace("_", " "), "local clock/session causal expanding state")
        for name in features
    ]
    return features, dictionary


def _aggregate_htf(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    rule = f"{int(minutes)}min"
    fields: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "notional": "sum",
        "delta_notional": "sum",
        "large_delta_notional": "sum",
        "trades_count": "sum",
    }
    available = {column: agg for column, agg in fields.items() if column in bars.columns}
    out = bars[list(available)].resample(rule, label="left", closed="left").agg(available)
    out["bar_count"] = bars["close"].resample(rule, label="left", closed="left").count()
    out = out.dropna(subset=["open", "high", "low", "close"])
    out.index.name = "bar_start_time"
    out["available_time"] = out.index + pd.Timedelta(minutes=int(minutes))
    return out


def _build_one_htf_features(htf: pd.DataFrame, minutes: int) -> pd.DataFrame:
    prefix = f"tf{int(minutes)}m"
    close = _numeric(htf, "close")
    high = _numeric(htf, "high")
    low = _numeric(htf, "low")
    open_ = _numeric(htf, "open")
    notional = _numeric(htf, "notional")
    delta = _numeric(htf, "delta_notional")
    large_delta = _numeric(htf, "large_delta_notional")
    ret1 = close.pct_change(fill_method=None)

    output = pd.DataFrame(index=htf.index)
    output[f"{prefix}_current_return"] = _safe_ratio(close, open_) - 1.0
    output[f"{prefix}_current_range_pct"] = _safe_ratio(high - low, close)
    output[f"{prefix}_current_delta_ratio"] = _safe_ratio(delta, notional)
    output[f"{prefix}_current_large_delta_ratio"] = _safe_ratio(large_delta, notional)
    for window in (3, 6, 12):
        rolling_low = low.rolling(window, min_periods=max(2, window // 2)).min()
        rolling_high = high.rolling(window, min_periods=max(2, window // 2)).max()
        output[f"{prefix}_return_{window}"] = _safe_ratio(close, close.shift(window)) - 1.0
        output[f"{prefix}_range_position_{window}"] = _safe_ratio(close - rolling_low, rolling_high - rolling_low)
        output[f"{prefix}_realized_vol_{window}"] = ret1.rolling(window, min_periods=max(2, window // 2)).std()
        output[f"{prefix}_delta_ratio_{window}"] = _safe_ratio(
            delta.rolling(window, min_periods=max(2, window // 2)).sum(),
            notional.rolling(window, min_periods=max(2, window // 2)).sum(),
        )
        output[f"{prefix}_large_delta_ratio_{window}"] = _safe_ratio(
            large_delta.rolling(window, min_periods=max(2, window // 2)).sum(),
            notional.rolling(window, min_periods=max(2, window // 2)).sum(),
        )
    prior_notional = notional.shift(1).rolling(12, min_periods=4).mean()
    output[f"{prefix}_notional_intensity"] = _safe_ratio(notional, prior_notional)
    output[f"{prefix}_return_acceleration"] = output[f"{prefix}_return_3"] - 0.5 * output[f"{prefix}_return_6"]
    output[f"{prefix}_available_time"] = htf["available_time"].to_numpy()
    return output.reset_index(drop=True)


def _merge_htf_to_candidates(
    bars: pd.DataFrame,
    candidates: pd.DataFrame,
    minutes: int,
) -> tuple[pd.DataFrame, dict[str, str]]:
    htf = _aggregate_htf(bars, minutes)
    features = _build_one_htf_features(htf, minutes)
    available_column = f"tf{int(minutes)}m_available_time"
    left = candidates[["event_id", "feature_available_time"]].copy()
    left["_row_order"] = np.arange(len(left), dtype=np.int64)
    left = left.sort_values("feature_available_time")
    right = features.sort_values(available_column)
    merged = pd.merge_asof(
        left,
        right,
        left_on="feature_available_time",
        right_on=available_column,
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_row_order")
    feature_columns = [
        column
        for column in merged.columns
        if column.startswith(f"tf{int(minutes)}m_") and column != available_column
    ]
    source_map = {
        column: f"closed {int(minutes)}m bars joined by available_time"
        for column in feature_columns
    }
    return merged[["event_id", available_column, *feature_columns]].reset_index(drop=True), source_map


def build_reversal_candidate_features(
    bars: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    windows: Sequence[int] = CORE_WINDOWS,
    htf_minutes: Sequence[int] = HTF_MINUTES,
    support_tolerance_bp: float = 25.0,
    include_session: bool = True,
    include_htf: bool = True,
    show_progress: bool = True,
) -> FeatureBuildResult:
    """Build compact causal features with vectorized rolling calculations."""

    if candidates.empty:
        return FeatureBuildResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    required = {"open", "high", "low", "close", "notional", "trades_count", "delta_notional"}
    missing = sorted(required - set(bars.columns))
    if missing:
        raise RuntimeError(f"reversal feature builder missing fields: {missing}")

    out = candidates.copy().reset_index(drop=True)
    positions = pd.to_numeric(out["extreme_pos"], errors="raise").to_numpy(dtype=np.int64)
    if positions.min(initial=0) < 0 or positions.max(initial=0) >= len(bars):
        raise RuntimeError("candidate extreme_pos is outside loaded bars")

    open_ = _numeric(bars, "open")
    high = _numeric(bars, "high")
    low = _numeric(bars, "low")
    close = _numeric(bars, "close")
    notional = _numeric(bars, "notional")
    trades = _numeric(bars, "trades_count")
    volume = _numeric(bars, "volume")
    delta = _numeric(bars, "delta_notional")
    large_delta = _numeric(bars, "large_delta_notional")
    buy = _numeric(bars, "buy_notional")
    sell = _numeric(bars, "sell_notional")
    large_buy = _numeric(bars, "large_buy_notional")
    large_sell = _numeric(bars, "large_sell_notional")
    large_count = _numeric(bars, "large_trades_count")
    max_trade = _numeric(bars, "max_trade_notional")
    avg_trade = _numeric(bars, "avg_trade_size")
    ret1 = close.pct_change(fill_method=None)
    abs_ret = ret1.abs()

    dictionary: list[dict[str, object]] = []
    group_features: dict[str, list[str]] = {group: [] for group in FEATURE_GROUP_ORDER}
    feature_arrays: dict[str, np.ndarray | pd.Series] = {}
    context_time_arrays: dict[str, pd.Series] = {}

    def add(name: str, series: pd.Series, group: str, description: str, source: str) -> None:
        feature_arrays[name] = _take_float(series, positions)
        dictionary.append(_feature_row(name, group, description, source))
        group_features[group].append(name)

    bar_range = (high - low).clip(lower=0.0)
    body = close - open_
    lower_wick = np.minimum(open_, close) - low
    upper_wick = high - np.maximum(open_, close)
    add("current_return_1", ret1, "M0_core", "current closed-bar close return", "1m OHLC")
    add("current_body_pct", _safe_ratio(body, open_), "M0_core", "current candle body / open", "1m OHLC")
    add("current_range_pct", _safe_ratio(bar_range, close), "M0_core", "current high-low range / close", "1m OHLC")
    add("current_lower_wick_share", _safe_ratio(lower_wick, bar_range), "M0_core", "lower wick share", "1m OHLC")
    add("current_upper_wick_share", _safe_ratio(upper_wick, bar_range), "M0_core", "upper wick share", "1m OHLC")
    add("current_close_in_bar", _safe_ratio(close - low, bar_range), "M0_core", "close position inside current bar", "1m OHLC")
    add("current_delta_ratio", _safe_ratio(delta, notional), "M0_core", "current delta / notional", "trade bar")
    add("current_large_delta_ratio", _safe_ratio(large_delta, notional), "M0_core", "current large delta / notional", "trade bar")
    add("current_buy_ratio", _safe_ratio(buy, buy + sell), "M0_core", "current aggressive buy share", "trade bar")
    add("current_large_buy_ratio", _safe_ratio(large_buy, large_buy + large_sell), "M0_core", "current large aggressive buy share", "trade bar")
    add("current_large_trade_share", _safe_ratio(large_count, trades), "M0_core", "large trade count share", "trade bar")
    add("current_max_trade_share", _safe_ratio(max_trade, notional), "M0_core", "largest trade / current notional", "trade bar")
    add("current_avg_trade_size_log", np.log1p(avg_trade.clip(lower=0.0)), "M0_core", "log average trade size", "trade bar")
    add("current_notional_log", np.log1p(notional.clip(lower=0.0)), "M0_core", "log current notional", "trade bar")
    add("current_trades_log", np.log1p(trades.clip(lower=0.0)), "M0_core", "log current trade count", "trade bar")

    reporter = ProgressReporter("[features] vectorized core windows", total=len(windows), every=1) if ProgressReporter and show_progress else None
    cached: dict[str, pd.Series] = {}
    tolerance = float(support_tolerance_bp) / 10_000.0
    for i, window_raw in enumerate(windows, start=1):
        window = int(window_raw)
        min_periods = max(3, window // 3)
        rolling_low = low.rolling(window, min_periods=min_periods).min()
        rolling_high = high.rolling(window, min_periods=min_periods).max()
        notional_sum = notional.rolling(window, min_periods=min_periods).sum()
        delta_sum = delta.rolling(window, min_periods=min_periods).sum()
        large_delta_sum = large_delta.rolling(window, min_periods=min_periods).sum()
        prior_notional_mean = notional.shift(1).rolling(window, min_periods=min_periods).mean()
        prior_trades_mean = trades.shift(1).rolling(window, min_periods=min_periods).mean()
        prior_volume_mean = volume.shift(1).rolling(window, min_periods=min_periods).mean()
        path_length = abs_ret.rolling(window, min_periods=min_periods).sum()
        return_w = _safe_ratio(close, close.shift(window)) - 1.0
        near_floor = (low <= rolling_low * (1.0 + tolerance)).astype(float)

        add(f"price_return_{window}", return_w, "M0_core", f"close return over {window} bars", "1m OHLC rolling")
        add(f"drawdown_from_high_{window}", _safe_ratio(close, rolling_high) - 1.0, "M0_core", f"distance from {window}-bar high", "1m OHLC rolling")
        add(f"rebound_from_low_{window}", _safe_ratio(close, rolling_low) - 1.0, "M0_core", f"distance from {window}-bar low", "1m OHLC rolling")
        add(f"range_position_{window}", _safe_ratio(close - rolling_low, rolling_high - rolling_low), "M0_core", f"position in {window}-bar range", "1m OHLC rolling")
        add(f"realized_vol_{window}", ret1.rolling(window, min_periods=min_periods).std(), "M0_core", f"close-return volatility over {window} bars", "1m close rolling")
        add(f"down_bar_share_{window}", (ret1 < 0.0).astype(float).rolling(window, min_periods=min_periods).mean(), "M0_core", f"down-bar share over {window} bars", "1m close rolling")
        add(f"path_efficiency_{window}", _safe_ratio(return_w.abs(), path_length), "M0_core", f"directional efficiency over {window} bars", "1m close rolling")
        add(f"delta_ratio_{window}", _safe_ratio(delta_sum, notional_sum), "M0_core", f"cumulative delta ratio over {window} bars", "trade bar rolling")
        add(f"large_delta_ratio_{window}", _safe_ratio(large_delta_sum, notional_sum), "M0_core", f"cumulative large delta ratio over {window} bars", "trade bar rolling")
        add(f"notional_intensity_{window}", _safe_ratio(notional, prior_notional_mean), "M0_core", f"current notional / prior {window}-bar mean", "trade bar rolling")
        add(f"trades_intensity_{window}", _safe_ratio(trades, prior_trades_mean), "M0_core", f"current trades / prior {window}-bar mean", "trade bar rolling")
        add(f"volume_intensity_{window}", _safe_ratio(volume, prior_volume_mean), "M0_core", f"current volume / prior {window}-bar mean", "trade bar rolling")
        add(f"support_test_density_{window}", near_floor.rolling(window, min_periods=min_periods).mean(), "M0_core", f"near-rolling-floor frequency over {window} bars", "1m low rolling")
        cached[f"ret_{window}"] = return_w
        cached[f"vol_{window}"] = ret1.rolling(window, min_periods=min_periods).std()
        cached[f"delta_{window}"] = _safe_ratio(delta_sum, notional_sum)
        cached[f"range_{window}"] = _safe_ratio(rolling_high - rolling_low, close)
        cached[f"notional_mean_{window}"] = notional.rolling(window, min_periods=min_periods).mean()
        if reporter is not None and i < len(windows):
            reporter.update(i)
    if reporter is not None:
        reporter.close()

    derived: Mapping[str, tuple[pd.Series, str]] = {
        "return_acceleration_5_30": (cached["ret_5"] - cached["ret_30"] / 6.0, "recent decline/rebound acceleration"),
        "return_acceleration_10_60": (cached["ret_10"] - cached["ret_60"] / 6.0, "10m versus 60m return acceleration"),
        "return_acceleration_15_120": (cached["ret_15"] - cached["ret_120"] / 8.0, "15m versus 120m return acceleration"),
        "vol_compression_10_60": (_safe_ratio(cached["vol_10"], cached["vol_60"]), "short/medium volatility compression"),
        "vol_compression_15_120": (_safe_ratio(cached["vol_15"], cached["vol_120"]), "15m/120m volatility compression"),
        "range_compression_10_60": (_safe_ratio(cached["range_10"], cached["range_60"]), "short/medium range compression"),
        "notional_compression_10_60": (_safe_ratio(cached["notional_mean_10"], cached["notional_mean_60"]), "short/medium activity compression"),
        "price_delta_divergence_30": (cached["ret_30"] - cached["delta_30"], "price return minus cumulative delta ratio"),
        "price_delta_divergence_60": (cached["ret_60"] - cached["delta_60"], "60m price/CVD divergence proxy"),
        "price_delta_divergence_120": (cached["ret_120"] - cached["delta_120"], "120m price/CVD divergence proxy"),
        "sell_pressure_absorption_30": ((-cached["delta_30"]).clip(lower=0.0) - (-cached["ret_30"]).clip(lower=0.0), "negative flow with reduced price impact"),
        "sell_pressure_absorption_60": ((-cached["delta_60"]).clip(lower=0.0) - (-cached["ret_60"]).clip(lower=0.0), "60m negative-flow absorption proxy"),
        "target_to_vol_30": (0.01 / cached["vol_30"].replace(0.0, np.nan), "1pct target relative to 30m volatility"),
        "target_to_vol_60": (0.01 / cached["vol_60"].replace(0.0, np.nan), "1pct target relative to 60m volatility"),
    }
    for name, (series, description) in derived.items():
        add(name, series.clip(-100.0, 100.0), "M0_core", description, "derived causal 1m structure/order flow")

    if include_session:
        if show_progress:
            print("[features] causal session context", flush=True)
        session_features, session_dictionary = _build_session_features(bars, positions)
        for name, values in session_features.items():
            feature_arrays[name] = values
            group_features["M1_session"].append(name)
        dictionary.extend(session_dictionary)

    htf_audits: list[dict[str, object]] = []
    effective_htf_minutes = tuple(htf_minutes) if include_htf else ()
    htf_reporter = ProgressReporter("[features] causal HTF", total=len(effective_htf_minutes), every=1) if ProgressReporter and show_progress and effective_htf_minutes else None
    for i, minutes_raw in enumerate(effective_htf_minutes, start=1):
        minutes = int(minutes_raw)
        merged, source_map = _merge_htf_to_candidates(bars, out, minutes)
        available_column = f"tf{minutes}m_available_time"
        if not merged["event_id"].equals(out["event_id"]):
            raise RuntimeError(f"{minutes}m context merge changed candidate order")
        context_time_arrays[available_column] = pd.to_datetime(merged[available_column])
        feature_available = pd.to_datetime(out["feature_available_time"])
        used_available = pd.to_datetime(context_time_arrays[available_column])
        violations = int((used_available > feature_available).fillna(False).sum())
        htf_audits.append(
            {
                "timeframe": f"{minutes}m",
                "candidate_count": int(len(out)),
                "context_non_null_count": int(used_available.notna().sum()),
                "available_time_violations": violations,
                "maximum_available_lag_seconds": float((feature_available - used_available).dt.total_seconds().max()),
                "passed": bool(violations == 0),
            }
        )
        for name, source in source_map.items():
            feature_arrays[name] = pd.to_numeric(merged[name], errors="coerce").to_numpy(dtype=np.float32, copy=False)
            group_features["M2_causal_htf"].append(name)
            dictionary.append(
                _feature_row(
                    name,
                    "M2_causal_htf",
                    name.replace("_", " "),
                    source,
                    f"{available_column} <= current 1m feature_available_time",
                )
            )
        if htf_reporter is not None and i < len(effective_htf_minutes):
            htf_reporter.update(i)
    if htf_reporter is not None:
        htf_reporter.close()

    feature_frame = pd.DataFrame(feature_arrays)
    if context_time_arrays:
        context_frame = pd.DataFrame(context_time_arrays)
        out = pd.concat([out.reset_index(drop=True), context_frame, feature_frame], axis=1)
    else:
        out = pd.concat([out.reset_index(drop=True), feature_frame], axis=1)

    # Nested groups: M1 includes M0, M2 includes M0+M1.
    nested: dict[str, list[str]] = {
        "M0_core": list(group_features["M0_core"]),
        "M1_session": [*group_features["M0_core"], *group_features["M1_session"]],
        "M2_causal_htf": [*group_features["M0_core"], *group_features["M1_session"], *group_features["M2_causal_htf"]],
    }
    membership_rows = [
        {"feature_group": group, "feature": feature, "feature_count": len(features)}
        for group, features in nested.items()
        for feature in features
    ]
    return FeatureBuildResult(
        frame=out,
        dictionary=pd.DataFrame(dictionary).drop_duplicates("feature").reset_index(drop=True),
        group_membership=pd.DataFrame(membership_rows),
        alignment_audit=pd.DataFrame(htf_audits),
    )


def build_reversal_forward_labels(
    bars: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    horizon: int = 60,
    target_move_pct: float = 1.0,
    adverse_levels_pct: Sequence[float] = ADVERSE_LEVELS_PCT,
    vectorized_chunk_size: int = 50_000,
    progress_every: int = 50_000,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Build next-open/future-close labels with several adverse first-touch levels."""

    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if vectorized_chunk_size < 1:
        raise ValueError("vectorized_chunk_size must be >= 1")
    target = float(target_move_pct) / 100.0
    adverse_levels = tuple(float(level) for level in adverse_levels_pct)
    if target <= 0 or any(level <= 0 for level in adverse_levels):
        raise ValueError("target and adverse levels must be positive")

    index = pd.DatetimeIndex(bars.index)
    open_values = _numeric(bars, "open").to_numpy(dtype=float, copy=False)
    close_values = _numeric(bars, "close").to_numpy(dtype=float, copy=False)
    windows = np.lib.stride_tricks.sliding_window_view(close_values, int(horizon))
    reporter = ProgressReporter("[labels] reversal close path", total=len(candidates), every=max(1, int(progress_every))) if ProgressReporter and show_progress else None
    parts: list[pd.DataFrame] = []
    processed = 0

    for start in range(0, len(candidates), int(vectorized_chunk_size)):
        source = candidates.iloc[start : start + int(vectorized_chunk_size)]
        positions = pd.to_numeric(source["extreme_pos"], errors="coerce").to_numpy(dtype=np.int64)
        entry_positions = positions + 1
        valid = (entry_positions >= 0) & (entry_positions < len(windows))
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size:
            ep = entry_positions[valid_indices]
            valid[valid_indices] &= np.isfinite(open_values[ep]) & (open_values[ep] > EPS)
        chosen = np.flatnonzero(valid)
        if not chosen.size:
            processed += len(source)
            if reporter is not None:
                reporter.update(processed)
            continue

        chunk = source.iloc[chosen].reset_index(drop=True)
        entry_positions = entry_positions[chosen]
        entry = open_values[entry_positions]
        path = windows[entry_positions]
        finite_any = np.isfinite(path).any(axis=1)
        if not finite_any.all():
            chunk = chunk.iloc[np.flatnonzero(finite_any)].reset_index(drop=True)
            entry_positions = entry_positions[finite_any]
            entry = entry[finite_any]
            path = path[finite_any]
        if not len(chunk):
            processed += len(source)
            continue

        max_close = np.nanmax(path, axis=1)
        min_close = np.nanmin(path, axis=1)
        mfe = np.maximum(0.0, max_close / entry - 1.0)
        mae = np.maximum(0.0, 1.0 - min_close / entry)
        terminal = path[:, -1] / entry - 1.0
        tp_mask = path >= entry[:, None] * (1.0 + target)
        tp_hit = tp_mask.any(axis=1)
        tp_index = np.argmax(tp_mask, axis=1)
        tp_bar = np.where(tp_hit, tp_index + 1, -1).astype(np.int16)
        axis = np.arange(int(horizon))[None, :]
        before_tp = axis <= tp_index[:, None]
        mae_before_tp = np.where(
            tp_hit,
            np.maximum(0.0, 1.0 - np.nanmin(np.where(before_tp, path, np.nan), axis=1) / entry),
            np.nan,
        )
        output: dict[str, object] = {
            "event_id": chunk["event_id"].to_numpy(),
            "entry_time": index[entry_positions],
            "entry_price": entry,
            "label_end_time": index[entry_positions + int(horizon) - 1],
            "forward_horizon_bars": np.full(len(chunk), int(horizon), dtype=np.int16),
            "tp_hit_1pct": tp_hit,
            "tp_first_touch_bar": tp_bar,
            "mfe_pct": (mfe * 100.0).astype(np.float32),
            "mae_horizon_pct": (mae * 100.0).astype(np.float32),
            "mae_before_tp_pct": (mae_before_tp * 100.0).astype(np.float32),
            "terminal_return_pct": (terminal * 100.0).astype(np.float32),
            "tp_within_15": tp_hit & (tp_index < min(15, int(horizon))),
            "tp_within_30": tp_hit & (tp_index < min(30, int(horizon))),
            "tp_within_45": tp_hit & (tp_index < min(45, int(horizon))),
        }
        for level in adverse_levels:
            suffix = str(level).replace(".", "p")
            adverse = level / 100.0
            adverse_mask = path <= entry[:, None] * (1.0 - adverse)
            adverse_hit = adverse_mask.any(axis=1)
            adverse_index = np.argmax(adverse_mask, axis=1)
            adverse_bar = np.where(adverse_hit, adverse_index + 1, -1).astype(np.int16)
            tp_before = tp_hit & (~adverse_hit | (tp_index < adverse_index))
            output[f"adverse_hit_{suffix}pct"] = adverse_hit
            output[f"adverse_first_touch_bar_{suffix}pct"] = adverse_bar
            output[f"tp_before_adverse_{suffix}pct"] = tp_before
        parts.append(pd.DataFrame(output))
        processed += len(source)
        if reporter is not None and processed < len(candidates):
            reporter.update(processed)
    if reporter is not None:
        reporter.close()
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def select_usable_features(
    fit: pd.DataFrame,
    requested: Sequence[str],
    *,
    max_missing_ratio: float = 0.30,
) -> tuple[str, ...]:
    """Fit-period-only unsupervised feature sanitation."""

    selected: list[str] = []
    for column in requested:
        if column not in fit.columns:
            continue
        values = pd.to_numeric(fit[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if float(values.isna().mean()) > float(max_missing_ratio):
            continue
        if int(values.nunique(dropna=True)) <= 1:
            continue
        selected.append(column)
    return tuple(selected)


def empirical_percentile(reference_scores: Sequence[float], scores: Sequence[float]) -> np.ndarray:
    reference = np.sort(np.asarray(reference_scores, dtype=float))
    values = np.asarray(scores, dtype=float)
    if not len(reference):
        return np.full(len(values), np.nan, dtype=float)
    return np.searchsorted(reference, values, side="right") / float(len(reference)) * 100.0


def select_first_crossing_events(
    frame: pd.DataFrame,
    *,
    score_column: str,
    threshold: float,
    cooldown_bars: int,
) -> pd.DataFrame:
    """Select the first causal threshold crossing in each contiguous candidate run."""

    if frame.empty:
        return frame.copy()
    data = frame.sort_values("extreme_pos").copy()
    score = pd.to_numeric(data[score_column], errors="coerce").to_numpy(dtype=float)
    positions = pd.to_numeric(data["extreme_pos"], errors="raise").to_numpy(dtype=np.int64)
    above = np.isfinite(score) & (score >= float(threshold))
    previous_above = np.r_[False, above[:-1]]
    gaps = np.r_[np.iinfo(np.int64).max, np.diff(positions)]
    run_start = above & (~previous_above | (gaps > 1))
    starts = np.flatnonzero(run_start)
    chosen: list[int] = []
    last_position = -10**18
    for row in starts:
        position = int(positions[row])
        if position - last_position < int(cooldown_bars):
            continue
        chosen.append(int(row))
        last_position = position
    result = data.iloc[chosen].copy().reset_index(drop=True)
    result["signal_threshold"] = float(threshold)
    result["cooldown_bars"] = int(cooldown_bars)
    return result


def _wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return np.nan
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    adjustment = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return float((center - adjustment) / denominator)


def opportunity_event_metrics(events: pd.DataFrame) -> dict[str, float]:
    if events.empty:
        return {
            "event_count": 0.0,
            "tp_rate": np.nan,
            "clean_0p25_rate": np.nan,
            "clean_0p50_rate": np.nan,
            "clean_0p75_rate": np.nan,
            "clean_1p0_rate": np.nan,
            "tp_wilson_lower": np.nan,
            "clean_0p25_wilson_lower": np.nan,
            "median_mfe_pct": np.nan,
            "median_mae_horizon_pct": np.nan,
            "median_mae_before_tp_pct": np.nan,
            "median_tp_bars": np.nan,
        }
    tp = events["tp_hit_1pct"].astype(bool)
    clean25 = events["tp_before_adverse_0p25pct"].astype(bool)
    clean50 = events["tp_before_adverse_0p5pct"].astype(bool)
    clean75 = events["tp_before_adverse_0p75pct"].astype(bool)
    clean100 = events["tp_before_adverse_1p0pct"].astype(bool)
    n = len(events)
    return {
        "event_count": float(n),
        "tp_rate": float(tp.mean()),
        "clean_0p25_rate": float(clean25.mean()),
        "clean_0p50_rate": float(clean50.mean()),
        "clean_0p75_rate": float(clean75.mean()),
        "clean_1p0_rate": float(clean100.mean()),
        "tp_wilson_lower": _wilson_lower(int(tp.sum()), n),
        "clean_0p25_wilson_lower": _wilson_lower(int(clean25.sum()), n),
        "median_mfe_pct": float(pd.to_numeric(events["mfe_pct"], errors="coerce").median()),
        "median_mae_horizon_pct": float(pd.to_numeric(events["mae_horizon_pct"], errors="coerce").median()),
        "median_mae_before_tp_pct": float(pd.to_numeric(events.loc[tp, "mae_before_tp_pct"], errors="coerce").median()) if tp.any() else np.nan,
        "median_tp_bars": float(pd.to_numeric(events.loc[tp, "tp_first_touch_bar"], errors="coerce").median()) if tp.any() else np.nan,
    }


def threshold_cooldown_grid(
    reference: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    score_column: str,
    fractions: Sequence[float],
    cooldowns: Sequence[int],
    threshold_source: str,
) -> tuple[pd.DataFrame, dict[tuple[float, int], pd.DataFrame]]:
    """Apply thresholds frozen from ``reference`` score quantiles."""

    scores = pd.to_numeric(reference[score_column], errors="coerce").dropna()
    rows: list[dict[str, object]] = []
    selected: dict[tuple[float, int], pd.DataFrame] = {}
    for fraction_raw in fractions:
        fraction = float(fraction_raw)
        if not 0.0 < fraction < 1.0:
            raise ValueError("fractions must be between 0 and 1")
        threshold = float(scores.quantile(1.0 - fraction))
        for cooldown_raw in cooldowns:
            cooldown = int(cooldown_raw)
            events = select_first_crossing_events(
                evaluation,
                score_column=score_column,
                threshold=threshold,
                cooldown_bars=cooldown,
            )
            metrics = opportunity_event_metrics(events)
            rows.append(
                {
                    "threshold_source": threshold_source,
                    "top_fraction": fraction,
                    "score_threshold": threshold,
                    "cooldown_bars": cooldown,
                    **metrics,
                }
            )
            selected[(fraction, cooldown)] = events
    return pd.DataFrame(rows), selected


def choose_validation_event_spec(
    grid: pd.DataFrame,
    *,
    minimum_events: int = 30,
) -> pd.Series:
    if grid.empty:
        raise RuntimeError("validation threshold grid is empty")
    data = grid.copy()
    eligible = data[pd.to_numeric(data["event_count"], errors="coerce") >= int(minimum_events)].copy()
    if eligible.empty:
        eligible = data.copy()
    eligible["selection_objective"] = (
        0.65 * pd.to_numeric(eligible["clean_0p25_wilson_lower"], errors="coerce").fillna(-1.0)
        + 0.35 * pd.to_numeric(eligible["tp_wilson_lower"], errors="coerce").fillna(-1.0)
    )
    eligible = eligible.sort_values(
        ["selection_objective", "clean_0p25_rate", "tp_rate", "event_count", "top_fraction", "cooldown_bars"],
        ascending=[False, False, False, False, True, False],
    )
    return eligible.iloc[0]


def attach_nearest_swing_distance(
    frame: pd.DataFrame,
    swing_events: pd.DataFrame,
) -> pd.DataFrame:
    """Post-label diagnostic only; never use these columns as model features."""

    out = frame.copy()
    reference = np.sort(pd.to_numeric(swing_events["extreme_pos"], errors="coerce").dropna().astype(np.int64).unique())
    positions = pd.to_numeric(out["extreme_pos"], errors="raise").to_numpy(dtype=np.int64)
    if not len(reference):
        out["nearest_swing_signed_distance_bars"] = np.nan
        out["nearest_swing_abs_distance_bars"] = np.nan
        return out
    insertion = np.searchsorted(reference, positions)
    left_idx = np.clip(insertion - 1, 0, len(reference) - 1)
    right_idx = np.clip(insertion, 0, len(reference) - 1)
    left_distance = reference[left_idx] - positions
    right_distance = reference[right_idx] - positions
    choose_right = np.abs(right_distance) < np.abs(left_distance)
    signed = np.where(choose_right, right_distance, left_distance)
    out["nearest_swing_signed_distance_bars"] = signed.astype(np.int32)
    out["nearest_swing_abs_distance_bars"] = np.abs(signed).astype(np.int32)
    return out
