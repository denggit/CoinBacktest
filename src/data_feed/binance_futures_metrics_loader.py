#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backward-compatible facade for Binance futures metrics.

Implementation is split by responsibility under
``src.data_feed.binance_futures_metrics``.
"""

from src.data_feed.binance_futures_metrics import (
    BINANCE_VISION_BASE_URL,
    DEFAULT_RELATIVE_WINDOWS,
    EXPECTED_ROWS_PER_DAY,
    METRIC_COLUMNS,
    METRICS_PERIOD,
    BinanceFuturesMetricsLoader,
    BinanceMetricsCoverage,
    BinanceMetricsDayResult,
    BinanceMetricsDownloadError,
    BinanceMetricsDownloadSummary,
)

__all__ = [
    "BINANCE_VISION_BASE_URL",
    "DEFAULT_RELATIVE_WINDOWS",
    "EXPECTED_ROWS_PER_DAY",
    "METRIC_COLUMNS",
    "METRICS_PERIOD",
    "BinanceFuturesMetricsLoader",
    "BinanceMetricsCoverage",
    "BinanceMetricsDayResult",
    "BinanceMetricsDownloadError",
    "BinanceMetricsDownloadSummary",
]
