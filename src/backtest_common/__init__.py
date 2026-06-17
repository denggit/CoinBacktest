"""Common utilities shared by CoinBacktest strategy backtests.

Keep this package small and dependency-light: strategy-specific signal logic stays
in backtest scripts; duplicated data loading, indicators, execution helpers, and
report formatting live here.
"""
