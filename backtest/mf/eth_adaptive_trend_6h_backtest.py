#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Adaptive 6H Trend Backtest.

Hypothesis:
    ETH trend signals are less noisy on 6H bars. A simple 24H momentum + EMA
    direction filter, combined with volatility-adjusted stop/trailing logic, can
    outperform short-horizon Donchian breakouts after costs.
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
from src.backtest_common.indicators import atr, ema, resample_ohlcv  # noqa: E402
from src.backtest_common.ohlcv_backtest import (  # noqa: E402
    SignalBacktestParams,
    emit_signal_report,
    print_signal_summary,
    run_signal_backtest,
    summarize_signal_backtest,
    write_signal_outputs,
)

STRATEGY_NAME = "eth_adaptive_trend_6h"


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "ETH-USDT-SWAP"
    load_timeframe: str = "1H"
    signal_rule: str = "6h"
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.004
    max_notional_mult: float = 3.0
    fee_rate: float = 0.0005
    slippage_pct: float = 0.0002

    ema_period: int = 50
    momentum_lookback_bars: int = 4      # 4 * 6h = 24h
    momentum_threshold: float = 0.018
    atr_period: int = 14
    min_atr_pct: float = 0.008
    max_atr_pct: float = 0.06
    stop_atr_low_vol: float = 2.4
    stop_atr_high_vol: float = 3.2
    vol_quantile_window: int = 120

    target_r: float = 3.0
    trailing_atr_mult: float = 3.6
    trail_after_r: float = 1.2
    min_stop_pct: float = 0.008
    max_stop_pct: float = 0.055
    cooldown_bars: int = 2
    max_hold_bars: int = 24              # 6 days
    no_progress_bars: int = 4            # 24h
    no_progress_min_mfe_r: float = 0.5


def build_features(base_1h: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = resample_ohlcv(base_1h, cfg.signal_rule)
    df["ema"] = ema(df["close"], cfg.ema_period)
    df["momentum"] = df["close"] / df["close"].shift(cfg.momentum_lookback_bars) - 1
    df["atr"] = atr(df, cfg.atr_period)
    df["atr_pct"] = df["atr"] / df["close"]
    df["atr_pct_q70"] = df["atr_pct"].rolling(cfg.vol_quantile_window, min_periods=cfg.vol_quantile_window // 2).quantile(0.70).shift(1)
    df["stop_mult"] = cfg.stop_atr_low_vol
    df.loc[df["atr_pct"] > df["atr_pct_q70"], "stop_mult"] = cfg.stop_atr_high_vol
    df["vol_ok"] = df["atr_pct"].between(cfg.min_atr_pct, cfg.max_atr_pct)

    df["long_signal"] = (df["momentum"] > cfg.momentum_threshold) & (df["close"] > df["ema"]) & df["vol_ok"]
    df["short_signal"] = (df["momentum"] < -cfg.momentum_threshold) & (df["close"] < df["ema"]) & df["vol_ok"]
    df["signal"] = 0
    df.loc[df["long_signal"], "signal"] = 1
    df.loc[df["short_signal"], "signal"] = -1
    df["stop"] = pd.NA
    df.loc[df["signal"] == 1, "stop"] = df["close"] - df["stop_mult"] * df["atr"]
    df.loc[df["signal"] == -1, "stop"] = df["close"] + df["stop_mult"] * df["atr"]
    return df.dropna().copy()


def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="ETH 6H adaptive trend-following backtest.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=today)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--risk-per-trade", type=float, default=0.004)
    p.add_argument("--fee-rate", type=float, default=0.0005)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--momentum-threshold", type=float, default=0.018)
    p.add_argument("--target-r", type=float, default=3.0)
    p.add_argument("--out-dir", default="data/reports/mf/eth_adaptive_trend_6h")
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
        momentum_threshold=args.momentum_threshold,
        target_r=args.target_r,
    )
    base = load_data(cfg.symbol, args.start_date, args.end_date, cfg.load_timeframe)
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
        trailing_atr_col="atr",
        trailing_atr_mult=cfg.trailing_atr_mult,
        trail_after_r=cfg.trail_after_r,
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
