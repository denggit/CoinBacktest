#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal order-flow, price-impact and absorption features.

The module consumes rich OKX trade-bar fields when they are present.  It never
silently reconstructs taker direction from candle colour.  Plain OHLCV and
range-bar inputs therefore remain usable, but order-flow outputs are explicitly
marked unavailable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.market_state.models import MarketStateConfig


CORE_ORDERFLOW_COLUMNS: tuple[str, ...] = (
    "notional",
    "buy_notional",
    "sell_notional",
    "delta_notional",
)
OPTIONAL_LARGE_COLUMNS: tuple[str, ...] = (
    "large_buy_notional",
    "large_sell_notional",
    "large_delta_notional",
)
_EPS = 1e-12


def _numeric(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").astype(float)


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _rolling_flow_ratio(delta: pd.Series, notional: pd.Series, window: int) -> pd.Series:
    return _safe_divide(
        delta.rolling(window, min_periods=window).sum(),
        notional.rolling(window, min_periods=window).sum(),
    ).clip(-1.0, 1.0)


def _historical_zscore(series: pd.Series, baseline_window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    history = values.shift(1)
    min_periods = max(30, baseline_window // 3)
    mean = history.rolling(baseline_window, min_periods=min_periods).mean()
    std = history.rolling(baseline_window, min_periods=min_periods).std(ddof=0)
    return ((values - mean) / std.replace(0.0, np.nan)).clip(-8.0, 8.0)


def orderflow_field_coverage(df: pd.DataFrame) -> dict[str, Any]:
    fields: dict[str, dict[str, Any]] = {}
    for column in CORE_ORDERFLOW_COLUMNS + OPTIONAL_LARGE_COLUMNS:
        present = column in df.columns
        values = _numeric(df, column) if present else pd.Series(dtype=float)
        fields[column] = {
            "present": bool(present),
            "non_null_ratio": float(values.notna().mean()) if present and len(values) else 0.0,
            "non_zero_ratio": float((values.fillna(0.0).abs() > _EPS).mean()) if present and len(values) else 0.0,
            "unique_values": int(values.nunique(dropna=True)) if present else 0,
        }
    core_usable = all(
        fields[column]["present"]
        and fields[column]["non_null_ratio"] >= 0.80
        and fields[column]["unique_values"] > 1
        for column in CORE_ORDERFLOW_COLUMNS
    )
    large_usable = all(
        fields[column]["present"]
        and fields[column]["non_null_ratio"] >= 0.60
        and fields[column]["unique_values"] > 1
        for column in OPTIONAL_LARGE_COLUMNS
    )
    return {
        "core_usable": bool(core_usable),
        "large_usable": bool(large_usable),
        "fields": fields,
    }


def compute_orderflow_features(
    df: pd.DataFrame,
    price_features: pd.DataFrame,
    config: MarketStateConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build closed-bar order-flow and impact features.

    Every rolling aggregate ends at the current closed bar.  Historical
    normalizers end one bar earlier.  The returned states are therefore usable
    only at the bar's ``available_time`` supplied by ``MarketStateDataBundle``.
    """

    coverage = orderflow_field_coverage(df)
    index = df.index
    empty = pd.DataFrame(index=index)
    if not coverage["core_usable"]:
        empty["orderflow_available"] = False
        for column in (
            "delta_ratio",
            "flow_fast_score",
            "flow_score",
            "flow_slow_score",
            "flow_persistence",
            "flow_acceleration",
            "large_flow_score",
            "flow_strength",
            "flow_intensity_z",
            "price_move_score",
            "flow_price_effectiveness",
            "sell_absorption_score",
            "buy_absorption_score",
            "signed_absorption_score",
        ):
            empty[column] = np.nan
        return empty, coverage

    notional = _numeric(df, "notional").clip(lower=0.0)
    buy = _numeric(df, "buy_notional").clip(lower=0.0)
    sell = _numeric(df, "sell_notional").clip(lower=0.0)
    delta = _numeric(df, "delta_notional")
    reconciled = buy - sell
    delta = delta.where(delta.notna(), reconciled)
    notional = notional.where(notional > 0.0, buy + sell)

    fast = _rolling_flow_ratio(delta, notional, config.flow_fast_window)
    medium = _rolling_flow_ratio(delta, notional, config.flow_window)
    slow = _rolling_flow_ratio(delta, notional, config.flow_slow_window)
    sign = np.sign(delta.where(notional > 0.0))
    persistence = sign.rolling(config.flow_window, min_periods=config.flow_window).mean().clip(-1.0, 1.0)
    acceleration = (fast - slow).clip(-1.0, 1.0)

    large_flow = pd.Series(np.nan, index=index, dtype=float)
    if coverage["large_usable"]:
        large_buy = _numeric(df, "large_buy_notional").clip(lower=0.0)
        large_sell = _numeric(df, "large_sell_notional").clip(lower=0.0)
        large_delta = _numeric(df, "large_delta_notional")
        large_delta = large_delta.where(large_delta.notna(), large_buy - large_sell)
        large_total = large_buy + large_sell
        large_flow = _rolling_flow_ratio(large_delta, large_total, config.flow_window)

    components = pd.concat(
        [
            0.25 * fast,
            0.45 * medium,
            0.15 * slow,
            0.10 * persistence,
            0.05 * large_flow,
        ],
        axis=1,
    )
    weights = pd.DataFrame(
        {
            "fast": fast.notna().astype(float) * 0.25,
            "medium": medium.notna().astype(float) * 0.45,
            "slow": slow.notna().astype(float) * 0.15,
            "persistence": persistence.notna().astype(float) * 0.10,
            "large": large_flow.notna().astype(float) * 0.05,
        },
        index=index,
    )
    flow_score = (components.sum(axis=1, min_count=1) / weights.sum(axis=1).replace(0.0, np.nan)).clip(-1.0, 1.0)
    flow_strength = np.tanh(flow_score.abs() / max(config.flow_scale, _EPS)).clip(0.0, 1.0)
    intensity_z = _historical_zscore(flow_score.abs(), config.baseline_window)

    close = _numeric(df, "close")
    open_ = _numeric(df, "open")
    high = _numeric(df, "high")
    low = _numeric(df, "low")
    window_return = close / close.shift(config.flow_window) - 1.0
    atr_pct = pd.to_numeric(price_features["atr_pct"], errors="coerce")
    expected_move = atr_pct * np.sqrt(float(config.flow_window))
    price_move_score = np.tanh(window_return / expected_move.replace(0.0, np.nan)).clip(-1.0, 1.0)

    flow_direction = np.sign(flow_score)
    directional_response = (flow_direction * price_move_score).clip(-1.0, 1.0)
    flow_effectiveness = (flow_strength * directional_response).clip(-1.0, 1.0)

    bar_range = (high - low).clip(lower=close.abs() * 1e-9)
    close_pos = ((close - low) / bar_range).clip(0.0, 1.0)
    lower_wick = ((pd.concat([open_, close], axis=1).min(axis=1) - low).clip(lower=0.0) / bar_range).clip(0.0, 1.0)
    upper_wick = ((high - pd.concat([open_, close], axis=1).max(axis=1)).clip(lower=0.0) / bar_range).clip(0.0, 1.0)

    # Flat or opposite price response under strong aggressive flow is treated
    # as an absorption proxy.  Wick and close-location evidence are secondary,
    # not standalone candle-pattern signals.
    ineffectiveness = ((config.impact_expected_response - directional_response) / 0.75).clip(0.0, 1.0)
    absorption_core = flow_strength * ineffectiveness
    sell_absorption = pd.Series(0.0, index=index, dtype=float)
    buy_absorption = pd.Series(0.0, index=index, dtype=float)
    sell_mask = flow_score < 0.0
    buy_mask = flow_score > 0.0
    sell_absorption.loc[sell_mask] = (
        0.65 * absorption_core.loc[sell_mask]
        + 0.20 * (flow_strength * lower_wick).loc[sell_mask]
        + 0.15 * (flow_strength * close_pos).loc[sell_mask]
    )
    buy_absorption.loc[buy_mask] = (
        0.65 * absorption_core.loc[buy_mask]
        + 0.20 * (flow_strength * upper_wick).loc[buy_mask]
        + 0.15 * (flow_strength * (1.0 - close_pos)).loc[buy_mask]
    )
    sell_absorption = sell_absorption.clip(0.0, 1.0)
    buy_absorption = buy_absorption.clip(0.0, 1.0)

    row_available = (
        notional.gt(0.0)
        & delta.notna()
        & fast.notna()
        & medium.notna()
        & slow.notna()
        & persistence.notna()
        & price_move_score.notna()
    )
    output = pd.DataFrame(
        {
            "orderflow_available": row_available,
            "delta_ratio": _safe_divide(delta, notional).clip(-1.0, 1.0),
            "flow_fast_score": fast,
            "flow_score": flow_score,
            "flow_slow_score": slow,
            "flow_persistence": persistence,
            "flow_acceleration": acceleration,
            "large_flow_score": large_flow,
            "flow_strength": flow_strength,
            "flow_intensity_z": intensity_z,
            "price_move_score": price_move_score,
            "flow_price_effectiveness": flow_effectiveness,
            "sell_absorption_score": sell_absorption.where(row_available),
            "buy_absorption_score": buy_absorption.where(row_available),
            "signed_absorption_score": (sell_absorption - buy_absorption).where(row_available),
        },
        index=index,
    )
    return output, coverage
