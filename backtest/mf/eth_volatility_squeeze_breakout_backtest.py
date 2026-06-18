#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Volatility Squeeze Breakout Backtest.

Hypothesis:
    After multi-hour volatility compression, a close outside the recent range
    with volume expansion has better continuation odds than ordinary breakouts.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest_common.data import load_ohlcv_data as load_data  # noqa: E402
from src.backtest_common.indicators import atr  # noqa: E402
from src.backtest_common.ohlcv_backtest import (  # noqa: E402
    SignalBacktestParams,
    emit_signal_report,
    print_signal_summary,
    run_signal_backtest,
    summarize_signal_backtest,
    write_signal_outputs,
)

STRATEGY_NAME = "eth_volatility_squeeze_breakout"


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "ETH-USDT-SWAP"
    timeframe: str = "15m"
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.003
    max_notional_mult: float = 3.0
    fee_rate: float = 0.0005
    slippage_pct: float = 0.0002

    bb_period: int = 20
    bb_width_window: int = 200
    bb_width_quantile: float = 0.20
    atr_period: int = 14
    atr_ma_period: int = 50
    atr_compress_mult: float = 0.85
    donchian_lookback: int = 32
    compression_recent_bars: int = 8
    max_range_pct: float = 0.03
    volume_ma_period: int = 20
    volume_mult: float = 1.40
    stop_atr_mult: float = 1.25

    target_r: float = 2.5
    min_stop_pct: float = 0.0045
    max_stop_pct: float = 0.022
    cooldown_bars: int = 12
    max_hold_bars: int = 96
    no_progress_bars: int = 16
    no_progress_min_mfe_r: float = 0.6


def build_features(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    ma = out["close"].rolling(cfg.bb_period, min_periods=cfg.bb_period).mean()
    std = out["close"].rolling(cfg.bb_period, min_periods=cfg.bb_period).std()
    out["bb_up"] = ma + 2 * std
    out["bb_dn"] = ma - 2 * std
    out["bb_width"] = (out["bb_up"] - out["bb_dn"]) / out["close"]
    out["bb_width_q"] = out["bb_width"].rolling(cfg.bb_width_window, min_periods=cfg.bb_width_window // 2).quantile(cfg.bb_width_quantile).shift(1)
    out["atr"] = atr(out, cfg.atr_period)
    out["atr_ma"] = out["atr"].rolling(cfg.atr_ma_period, min_periods=cfg.atr_ma_period).mean().shift(1)
    out["range_high"] = out["high"].rolling(cfg.donchian_lookback, min_periods=cfg.donchian_lookback).max().shift(1)
    out["range_low"] = out["low"].rolling(cfg.donchian_lookback, min_periods=cfg.donchian_lookback).min().shift(1)
    out["range_pct"] = (out["range_high"] - out["range_low"]) / out["close"]
    out["volume_ma"] = out["volume"].rolling(cfg.volume_ma_period, min_periods=cfg.volume_ma_period).mean().shift(1)
    out["compression"] = (
        (out["bb_width"] < out["bb_width_q"])
        & (out["atr"] < out["atr_ma"] * cfg.atr_compress_mult)
        & (out["range_pct"] < cfg.max_range_pct)
    )
    out["compression_recent"] = out["compression"].rolling(cfg.compression_recent_bars, min_periods=1).max().shift(1).fillna(False).astype(bool)
    out["volume_ok"] = out["volume"] > out["volume_ma"] * cfg.volume_mult
    out["long_signal"] = out["compression_recent"] & out["volume_ok"] & (out["close"] > out["range_high"])
    out["short_signal"] = out["compression_recent"] & out["volume_ok"] & (out["close"] < out["range_low"])
    out["signal"] = 0
    out.loc[out["long_signal"], "signal"] = 1
    out.loc[out["short_signal"], "signal"] = -1
    out["stop"] = pd.NA
    out.loc[out["signal"] == 1, "stop"] = out["close"] - cfg.stop_atr_mult * out["atr"]
    out.loc[out["signal"] == -1, "stop"] = out["close"] + cfg.stop_atr_mult * out["atr"]
    return out.dropna().copy()


def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="ETH 15M volatility squeeze breakout backtest.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=today)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--risk-per-trade", type=float, default=0.003)
    p.add_argument("--fee-rate", type=float, default=0.0005)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--volume-mult", type=float, default=1.40)
    p.add_argument("--target-r", type=float, default=2.5)
    p.add_argument("--out-dir", default="data/reports/mf/eth_volatility_squeeze_breakout")
    p.add_argument("--write-full-audit", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = StrategyConfig(
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
        volume_mult=args.volume_mult,
        target_r=args.target_r,
    )
    base = load_data(cfg.symbol, args.start_date, args.end_date, cfg.timeframe)
    features = build_features(base, cfg)
    params = SignalBacktestParams(
        initial_capital=cfg.initial_capital,
        risk_per_trade=cfg.risk_per_trade,
        max_notional_mult=cfg.max_notional_mult,
        fee_rate=cfg.fee_rate,
        slippage_pct=cfg.slippage_pct,
        target_r=cfg.target_r,
        min_stop_pct=cfg.min_stop_pct,
        max_stop_pct=cfg.max_stop_pct,
        cooldown_bars=cfg.cooldown_bars,
        max_hold_bars=cfg.max_hold_bars,
        no_progress_bars=cfg.no_progress_bars,
        no_progress_min_mfe_r=cfg.no_progress_min_mfe_r,
    )
    trades, equity = run_signal_backtest(features, params)
    summary = summarize_signal_backtest(trades, equity, cfg.initial_capital, int((features["signal"] != 0).sum()))
    out_dir = PROJECT_ROOT / args.out_dir
    emit_signal_report(trades, features, cfg, out_dir, strategy_name=STRATEGY_NAME)
    write_signal_outputs(features, trades, equity, summary, out_dir, strategy_name=STRATEGY_NAME, write_full_audit=args.write_full_audit)
    print_signal_summary(summary, out_dir, strategy_name=STRATEGY_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
