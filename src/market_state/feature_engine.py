#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fast, vectorized and causal features for Market State Map V0.2."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.market_state.models import MarketStateConfig


_EPS = 1e-12


def _rolling_linear_r2(values: pd.Series, window: int) -> pd.Series:
    """Rolling R² of values against bar number in O(n) vectorized form."""

    y = pd.to_numeric(values, errors="coerce").astype(float)
    t = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    sum_y = y.rolling(window, min_periods=window).sum()
    sum_y2 = y.pow(2).rolling(window, min_periods=window).sum()
    sum_t = t.rolling(window, min_periods=window).sum()
    sum_t2 = t.pow(2).rolling(window, min_periods=window).sum()
    sum_ty = (t * y).rolling(window, min_periods=window).sum()
    numerator = window * sum_ty - sum_t * sum_y
    denominator = np.sqrt(
        (window * sum_t2 - sum_t.pow(2)).clip(lower=0.0)
        * (window * sum_y2 - sum_y.pow(2)).clip(lower=0.0)
    )
    corr = numerator / denominator.replace(0.0, np.nan)
    return corr.pow(2).clip(0.0, 1.0)


def _historical_zscore(series: pd.Series, baseline_window: int) -> pd.Series:
    """Normalize current values against a baseline ending one bar earlier."""

    values = pd.to_numeric(series, errors="coerce").astype(float)
    history = values.shift(1)
    min_periods = max(20, baseline_window // 3)
    mean = history.rolling(baseline_window, min_periods=min_periods).mean()
    std = history.rolling(baseline_window, min_periods=min_periods).std(ddof=0)
    return (values - mean) / std.replace(0.0, np.nan)


def _historical_percentile_proxy(
    series: pd.Series,
    baseline_window: int,
    low_quantile: float,
    high_quantile: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Causal percentile proxy using rolling historical quantile anchors.

    Exact rolling ranks are costly for hundreds of thousands of bars.  Two
    shifted rolling quantiles provide a stable, causal and vectorized proxy:
    0.2 maps near the historical low anchor, 0.8 near the high anchor.
    """

    values = pd.to_numeric(series, errors="coerce").astype(float)
    history = values.shift(1)
    min_periods = max(30, baseline_window // 3)
    q_low = history.rolling(baseline_window, min_periods=min_periods).quantile(low_quantile)
    q_high = history.rolling(baseline_window, min_periods=min_periods).quantile(high_quantile)
    width = (q_high - q_low).replace(0.0, np.nan)
    middle = low_quantile + (values - q_low) / width * (high_quantile - low_quantile)
    below = low_quantile * values / q_low.replace(0.0, np.nan)
    above = high_quantile + (1.0 - high_quantile) * (
        (values - q_high) / (1.0 - q_high).replace(0.0, np.nan)
    )
    percentile = middle.where(values >= q_low, below).where(values <= q_high, above)
    return percentile.clip(0.0, 1.0), q_low, q_high


def _sigmoid(values: pd.Series) -> pd.Series:
    clipped = pd.to_numeric(values, errors="coerce").clip(-12.0, 12.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _trend_components(log_close: pd.Series, log_return: pd.Series, window: int) -> dict[str, pd.Series]:
    move = log_close - log_close.shift(window - 1)
    path_length = log_return.abs().rolling(window, min_periods=window).sum()
    efficiency = (move.abs() / path_length.replace(0.0, np.nan)).clip(0.0, 1.0)
    regression_r2 = _rolling_linear_r2(log_close, window)
    orderliness = (0.55 * efficiency + 0.45 * regression_r2).clip(0.0, 1.0)
    cumulative_noise = log_return.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(float(window))
    directional_signal = np.tanh(move / (2.0 * cumulative_noise.replace(0.0, np.nan)))
    score = (directional_signal * (0.45 + 0.55 * orderliness)).clip(-1.0, 1.0)
    return {
        "move": move,
        "path_efficiency": efficiency,
        "regression_r2": regression_r2,
        "orderliness": orderliness,
        "score": score,
    }


def _alignment_score(scores: list[pd.Series]) -> pd.Series:
    normalized = pd.concat([np.tanh(score / 0.20) for score in scores], axis=1)
    numerator = normalized.mean(axis=1).abs()
    denominator = normalized.abs().mean(axis=1).replace(0.0, np.nan)
    return (numerator / denominator).clip(0.0, 1.0)


def compute_market_state_features(df: pd.DataFrame, config: MarketStateConfig) -> pd.DataFrame:
    """Compute multi-horizon trend, volatility and activity features.

    Output at row *t* only uses rows ``<= t``.  Historical z-scores and
    quantiles use a baseline ending at ``t-1``.
    """

    config.validate()
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    high = pd.to_numeric(df["high"], errors="coerce").astype(float)
    low = pd.to_numeric(df["low"], errors="coerce").astype(float)
    volume = pd.to_numeric(df["volume"], errors="coerce").astype(float).clip(lower=0.0)

    log_close = np.log(close.clip(lower=_EPS))
    log_return = log_close.diff()
    abs_return = log_return.abs()

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    fast = _trend_components(log_close, log_return, config.fast_trend_window)
    medium = _trend_components(log_close, log_return, config.trend_window)
    slow = _trend_components(log_close, log_return, config.slow_trend_window)

    structural_trend_score = (0.65 * medium["score"] + 0.35 * slow["score"]).clip(-1.0, 1.0)
    combined_orderliness = (
        0.15 * fast["orderliness"] + 0.55 * medium["orderliness"] + 0.30 * slow["orderliness"]
    ).clip(0.0, 1.0)
    trend_alignment = _alignment_score([fast["score"], medium["score"], slow["score"]])
    orderliness_percentile, orderliness_q_low, orderliness_q_high = _historical_percentile_proxy(
        combined_orderliness,
        config.baseline_window,
        config.orderliness_low_quantile,
        config.orderliness_high_quantile,
    )

    realized_volatility = np.sqrt(
        log_return.pow(2).rolling(config.volatility_window, min_periods=config.volatility_window).mean()
    )
    atr_pct = (
        true_range.rolling(config.volatility_window, min_periods=config.volatility_window).mean()
        / close.replace(0.0, np.nan)
    )
    rv_z = _historical_zscore(np.log(realized_volatility.clip(lower=_EPS)), config.baseline_window)
    atr_z = _historical_zscore(np.log(atr_pct.clip(lower=_EPS)), config.baseline_window)
    volatility_z = (0.60 * rv_z + 0.40 * atr_z).clip(-8.0, 8.0)
    volatility_score = _sigmoid(volatility_z)

    volume_activity = np.log1p(
        volume.rolling(config.activity_window, min_periods=config.activity_window).mean().clip(lower=0.0)
    )
    activity_components = [_historical_zscore(volume_activity, config.baseline_window)]
    if "trades_count" in df.columns:
        trades = pd.to_numeric(df["trades_count"], errors="coerce").astype(float).clip(lower=0.0)
        trade_activity = np.log1p(
            trades.rolling(config.activity_window, min_periods=config.activity_window).mean().clip(lower=0.0)
        )
        activity_components.append(_historical_zscore(trade_activity, config.baseline_window))
    activity_z = pd.concat(activity_components, axis=1).mean(axis=1).clip(-8.0, 8.0)
    activity_score = _sigmoid(activity_z)

    bar_return_z = _historical_zscore(abs_return, config.baseline_window).clip(-8.0, 8.0)
    data_ready = pd.concat(
        [
            structural_trend_score,
            fast["score"],
            medium["score"],
            slow["score"],
            trend_alignment,
            combined_orderliness,
            orderliness_percentile,
            volatility_z,
            activity_z,
        ],
        axis=1,
    ).notna().all(axis=1)

    return pd.DataFrame(
        {
            "log_return": log_return,
            "true_range": true_range,
            "atr_pct": atr_pct,
            "realized_volatility": realized_volatility,
            "fast_trend_move": fast["move"],
            "medium_trend_move": medium["move"],
            "slow_trend_move": slow["move"],
            "fast_path_efficiency": fast["path_efficiency"],
            "medium_path_efficiency": medium["path_efficiency"],
            "slow_path_efficiency": slow["path_efficiency"],
            "fast_regression_r2": fast["regression_r2"],
            "medium_regression_r2": medium["regression_r2"],
            "slow_regression_r2": slow["regression_r2"],
            "fast_trend_score": fast["score"],
            "medium_trend_score": medium["score"],
            "slow_trend_score": slow["score"],
            "trend_score": structural_trend_score,
            "trend_alignment_score": trend_alignment,
            "orderliness_score": combined_orderliness,
            "orderliness_percentile": orderliness_percentile,
            "orderliness_q_low": orderliness_q_low,
            "orderliness_q_high": orderliness_q_high,
            "volatility_z": volatility_z,
            "volatility_score": volatility_score,
            "activity_z": activity_z,
            "activity_score": activity_score,
            "bar_return_z": bar_return_z,
            "data_ready": data_ready,
        },
        index=df.index,
    )
