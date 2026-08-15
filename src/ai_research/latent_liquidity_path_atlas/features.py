#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Strictly pre-event path features for broad liquidity-release candidates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LatentLiquidityPathAtlasConfig
from .time_axis import as_datetime_ns


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0.0, np.nan)


def _path_feature_series(
    frame: pd.DataFrame,
    config: LatentLiquidityPathAtlasConfig,
) -> dict[str, pd.Series]:
    """Build causal path Series ending at ``event_time - 1 second``.

    The release bar is excluded.  These features describe pre-existing path,
    turnover, directional pressure and impact efficiency, not the burst that
    reveals liquidity was just released.
    """
    history = frame.shift(1)
    close = history["close"].astype(float)
    ret1 = close.pct_change().fillna(0.0)
    abs_ret1 = ret1.abs()
    sign_change = np.sign(ret1).ne(np.sign(ret1.shift(1))).astype(float)
    additions: dict[str, pd.Series] = {}

    for window in config.path_windows_seconds:
        w = int(window)
        start_price = close.shift(w)
        path_ret = _safe_div(close - start_price, start_price)
        rolling_high = history["high"].rolling(w, min_periods=w).max()
        rolling_low = history["low"].rolling(w, min_periods=w).min()
        travelled = abs_ret1.rolling(w, min_periods=w).sum()
        buy = history["buy_notional"].rolling(w, min_periods=w).sum()
        sell = history["sell_notional"].rolling(w, min_periods=w).sum()
        notional_sum = history["notional"].rolling(w, min_periods=w).sum()
        trades_sum = history["trades_count"].rolling(w, min_periods=w).sum()
        delta_sum = history["delta_notional"].rolling(w, min_periods=w).sum()
        max_trade = history["max_trade_notional"].rolling(w, min_periods=w).max()
        prior_notional = history["notional"].shift(w).rolling(w, min_periods=w).sum()
        prior_trades = history["trades_count"].shift(w).rolling(w, min_periods=w).sum()
        prior_max_trade = history["max_trade_notional"].shift(w).rolling(w, min_periods=w).max()
        range_bp = _safe_div(rolling_high - rolling_low, close) * 1e4
        efficiency = _safe_div(path_ret.abs(), travelled).clip(0.0, 1.0)
        delta_share = _safe_div(delta_sum, notional_sum)
        turnover_per_range = _safe_div(notional_sum, range_bp.clip(lower=0.1))
        prior_turnover_per_range = turnover_per_range.shift(w)
        additions.update(
            {
                f"path_ret_{w}s": path_ret,
                f"path_high_excursion_{w}s": _safe_div(rolling_high - start_price, start_price),
                f"path_low_excursion_{w}s": _safe_div(rolling_low - start_price, start_price),
                f"path_range_bp_{w}s": range_bp,
                f"path_efficiency_{w}s": efficiency,
                f"path_realized_vol_{w}s": ret1.rolling(w, min_periods=w).std(ddof=0),
                f"path_sign_changes_{w}s": sign_change.rolling(w, min_periods=w).sum(),
                f"path_positive_share_{w}s": ret1.gt(0).astype(float).rolling(w, min_periods=w).mean(),
                f"path_notional_{w}s": notional_sum,
                f"path_notional_intensity_{w}s": _safe_div(notional_sum, prior_notional),
                f"path_delta_{w}s": delta_sum,
                f"path_delta_share_{w}s": delta_share,
                f"path_buy_share_{w}s": _safe_div(buy, buy + sell),
                f"path_trades_{w}s": trades_sum,
                f"path_trades_intensity_{w}s": _safe_div(trades_sum, prior_trades),
                f"path_max_trade_{w}s": max_trade,
                f"path_max_trade_ratio_{w}s": _safe_div(max_trade, prior_max_trade),
                f"path_large_delta_{w}s": history["large_delta_notional"].rolling(w, min_periods=w).sum(),
                # Liquidity accumulation/absorption proxies; no Swing is needed.
                f"path_turnover_per_range_{w}s": turnover_per_range,
                f"path_turnover_per_range_intensity_{w}s": _safe_div(turnover_per_range, prior_turnover_per_range),
                f"path_pressure_without_progress_{w}s": delta_share.abs() * (1.0 - efficiency),
                f"path_travel_bp_{w}s": travelled * 1e4,
            }
        )

    prior_low = history["low"].rolling(3600, min_periods=900).min()
    prior_high = history["high"].rolling(3600, min_periods=900).max()
    additions["distance_prior_1h_low_bp"] = _safe_div(close - prior_low, close) * 1e4
    additions["distance_prior_1h_high_bp"] = _safe_div(prior_high - close, close) * 1e4
    additions["position_in_prior_1h_range"] = _safe_div(close - prior_low, prior_high - prior_low)
    additions["pre_event_close"] = close
    return additions


def add_rolling_path_features(
    frame: pd.DataFrame,
    config: LatentLiquidityPathAtlasConfig,
) -> pd.DataFrame:
    additions = _path_feature_series(frame, config)
    return pd.concat([frame.copy(), pd.DataFrame(additions, index=frame.index)], axis=1)


def _sample_path_feature_values(
    frame: pd.DataFrame,
    event_index: pd.DatetimeIndex,
    config: LatentLiquidityPathAtlasConfig,
) -> pd.DataFrame:
    additions = _path_feature_series(frame, config)
    sampled = {name: values.reindex(event_index) for name, values in additions.items()}
    return pd.DataFrame(sampled, index=event_index)


def event_feature_table(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    config: LatentLiquidityPathAtlasConfig,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    event_index = as_datetime_ns(events.index)
    selected = frame.reindex(event_index).copy()
    selected = pd.concat([selected, _sample_path_feature_values(frame, event_index, config)], axis=1)
    selected["event_id"] = events["event_id"].to_numpy()
    selected["event_time"] = event_index.to_numpy(dtype="datetime64[ns]")
    selected["event_side"] = events["event_side"].to_numpy()
    for name in (
        "release_episode_id",
        "release_episode_number",
        "release_episode_ordinal",
        "release_episode_size",
        "release_episode_weight",
    ):
        if name in events:
            selected[name] = events[name].to_numpy()
    selected["period"] = np.select(
        [
            selected.index < pd.Timestamp("2025-01-01"),
            selected.index < pd.Timestamp("2025-10-01"),
        ],
        ["TRAIN_2023_2024", "VALIDATION_2025Q1_Q3"],
        default="HOLDOUT_2025Q4_2026H1",
    )

    source_cols = [name for name in selected.columns if name.startswith("source_")]
    selected["candidate_source_count"] = selected[source_cols].astype(bool).sum(axis=1)
    selected["pre_path_available_time"] = selected.index - pd.Timedelta(seconds=1)
    selected["causal_feature_time"] = selected.index
    return selected.reset_index(drop=True)


def model_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Normalized pre-event features used by discovery clustering.

    Event-time release scores, absolute price/flow scale, IDs and labels are
    excluded.  15m+ unswept Swing inventory is supplementary; liquidity-path
    families remain the dominant feature space.
    """
    allowed_prefixes = (
        "path_ret_",
        "path_high_excursion_",
        "path_low_excursion_",
        "path_range_bp_",
        "path_efficiency_",
        "path_realized_vol_",
        "path_sign_changes_",
        "path_positive_share_",
        "path_buy_share_",
        "path_notional_intensity_",
        "path_delta_share_",
        "path_trades_intensity_",
        "path_max_trade_ratio_",
        "path_turnover_per_range_intensity_",
        "path_pressure_without_progress_",
        "path_travel_bp_",
        "distance_prior_",
        "position_in_prior_",
        "macro_",
        "unswept_",
    )
    blocked = {
        "macro_bar_start_time",
        "macro_available_time",
        "macro_pre_event_close",
        "unswept_max_level_available_time",
    }
    columns: list[str] = []
    for name in frame.columns:
        if name in blocked or not name.startswith(allowed_prefixes):
            continue
        if name.startswith(("macro_notional_", "macro_delta_", "macro_trades_", "macro_turnover_per_range_")) and not name.startswith(
            (
                "macro_notional_intensity_",
                "macro_delta_share_",
                "macro_trades_intensity_",
                "macro_turnover_per_range_intensity_",
            )
        ):
            continue
        if name.startswith("unswept_") and name.endswith("available_time"):
            continue
        if pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name]):
            columns.append(name)
    return tuple(columns)
