#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic feature-schema construction."""

from __future__ import annotations

import pandas as pd

from .config import RLMarketAgentConfig
from .contracts import FeatureSpec
from .features import (
    fixed_bar_feature_names,
    footprint_event_feature_names,
    range_event_feature_names,
    trade_bar_feature_names,
)


def range_code(value: float) -> str:
    return f"r{int(round(float(value) * 10000)):04d}"


def build_feature_specs(config: RLMarketAgentConfig) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for tf in config.kline_timeframes:
        prefix = f"kline_{tf.lower()}"
        for name in fixed_bar_feature_names(prefix):
            specs.append(FeatureSpec(name, prefix, f"bar_start + {tf}", "Closed higher-timeframe K-line state."))
    trade_windows = [pd.Timedelta(minutes=x) for x in config.trade_windows_minutes]
    for name in trade_bar_feature_names("trade_1m", trade_windows):
        specs.append(FeatureSpec(name, "trade_1m", "bar_start + 1m", "Tick-derived 1m order-flow state."))
    micro_windows = [pd.Timedelta(seconds=x) for x in config.micro_windows_seconds]
    for name in trade_bar_feature_names("trade_5s", micro_windows):
        specs.append(FeatureSpec(name, "trade_5s", "bar_start + 5s", "Tick-derived 5s microstructure state.", optional=True))
    range_windows = [pd.Timedelta(minutes=x) for x in config.range_windows_minutes]
    for rp in config.range_pcts:
        prefix = f"range_{range_code(rp)}"
        for name in range_event_feature_names(prefix, range_windows):
            specs.append(FeatureSpec(name, prefix, "closed range-bar end_ts", "Closed range-bar path state.", optional=True))
    fp_prefix = f"footprint_{range_code(config.footprint_range_pct)}"
    for name in footprint_event_feature_names(fp_prefix):
        specs.append(FeatureSpec(name, fp_prefix, "closed range-bar end_ts", "Closed range-footprint state.", optional=True))
    # Explicit source-availability flags make incomplete enrichment auditable.
    sources = [f"kline_{tf.lower()}" for tf in config.kline_timeframes] + ["trade_1m", "trade_5s"]
    sources += [f"range_{range_code(rp)}" for rp in config.range_pcts] + [fp_prefix]
    for source in sources:
        specs.append(FeatureSpec(f"availability__{source}", source, "decision_time", "1 when the source has an observable state at decision_time.", optional=source not in {"trade_1m", *[f"kline_{tf.lower()}" for tf in config.kline_timeframes]}))
    return specs
