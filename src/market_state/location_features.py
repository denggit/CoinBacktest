#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal rolling structure-location and sweep features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.market_state.models import MarketStateConfig


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").astype(float)


def compute_location_features(
    df: pd.DataFrame,
    price_features: pd.DataFrame,
    config: MarketStateConfig,
) -> pd.DataFrame:
    """Describe current price relative to levels known before the current bar.

    Rolling highs/lows are shifted by one bar.  Current-bar high/low/close may
    confirm a sweep or acceptance only after that bar closes.
    """

    high = _numeric(df, "high")
    low = _numeric(df, "low")
    close = _numeric(df, "close")
    atr_abs = pd.to_numeric(price_features["atr_pct"], errors="coerce") * close

    local_high = high.shift(1).rolling(config.location_window, min_periods=config.location_window).max()
    local_low = low.shift(1).rolling(config.location_window, min_periods=config.location_window).min()
    structural_high = high.shift(1).rolling(config.structure_window, min_periods=config.structure_window).max()
    structural_low = low.shift(1).rolling(config.structure_window, min_periods=config.structure_window).min()

    local_width = (local_high - local_low).replace(0.0, np.nan)
    structural_width = (structural_high - structural_low).replace(0.0, np.nan)
    local_position = ((close - local_low) / local_width).clip(-0.25, 1.25)
    structural_position = ((close - structural_low) / structural_width).clip(-0.25, 1.25)
    structural_location_score = (2.0 * structural_position - 1.0).clip(-1.25, 1.25)

    support_distance_atr = (close - local_low) / atr_abs.replace(0.0, np.nan)
    resistance_distance_atr = (local_high - close) / atr_abs.replace(0.0, np.nan)
    structural_support_distance_atr = (close - structural_low) / atr_abs.replace(0.0, np.nan)
    structural_resistance_distance_atr = (structural_high - close) / atr_abs.replace(0.0, np.nan)

    sweep_buffer = config.sweep_min_atr * atr_abs
    accept_buffer = config.breakout_accept_atr * atr_abs
    downside_sweep = (low < local_low - sweep_buffer) & (close >= local_low)
    upside_sweep = (high > local_high + sweep_buffer) & (close <= local_high)
    breakout_accept = close > local_high + accept_buffer
    breakdown_accept = close < local_low - accept_buffer
    near_support = support_distance_atr.between(-config.near_level_atr, config.near_level_atr)
    near_resistance = resistance_distance_atr.between(-config.near_level_atr, config.near_level_atr)

    ready = pd.concat([local_high, local_low, structural_high, structural_low, atr_abs], axis=1).notna().all(axis=1)
    return pd.DataFrame(
        {
            "location_available": ready,
            "local_support": local_low,
            "local_resistance": local_high,
            "structural_support": structural_low,
            "structural_resistance": structural_high,
            "local_position": local_position,
            "structural_position": structural_position,
            "structural_location_score": structural_location_score,
            "support_distance_atr": support_distance_atr,
            "resistance_distance_atr": resistance_distance_atr,
            "structural_support_distance_atr": structural_support_distance_atr,
            "structural_resistance_distance_atr": structural_resistance_distance_atr,
            "downside_sweep_reclaim": downside_sweep & ready,
            "upside_sweep_reject": upside_sweep & ready,
            "breakout_accept": breakout_accept & ready,
            "breakdown_accept": breakdown_accept & ready,
            "near_support": near_support & ready,
            "near_resistance": near_resistance & ready,
        },
        index=df.index,
    )
