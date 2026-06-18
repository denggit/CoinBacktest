#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Donchian Breakout with BTC Regime Filter Backtest.

Hypothesis:
    ETH medium-frequency breakouts have higher quality when BTC confirms the
    market regime. ETH long breakouts are only allowed when BTC is above its
    trend EMA and has positive 24H momentum; shorts require the opposite.
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
from src.backtest_common.indicators import adx, atr, ema  # noqa: E402
from src.backtest_common.ohlcv_backtest import (  # noqa: E402
    SignalBacktestParams,
    emit_signal_report,
    print_signal_summary,
    run_signal_backtest,
    summarize_signal_backtest,
    write_signal_outputs,
)

STRATEGY_NAME = "eth_with_btc_regime_filter"


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "ETH-USDT-SWAP"
    btc_symbol: str = "BTC-USDT-SWAP"
    timeframe: str = "1H"
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.003
    max_notional_mult: float = 3.0
    fee_rate: float = 0.0005
    slippage_pct: float = 0.0002

    donchian_lookback: int = 96
    breakout_buffer_pct: float = 0.002
    atr_period: int = 14
    adx_period: int = 14
    adx_min: float = 19.0
    min_atr_pct: float = 0.005
    max_atr_pct: float = 0.04
    min_channel_width_pct: float = 0.02
    volume_ma_period: int = 20
    volume_mult: float = 1.05
    stop_atr_mult: float = 2.3

    btc_ema_period: int = 50
    btc_mom_lookback: int = 24
    btc_mom_threshold: float = 0.0

    target_r: float = 2.3
    trailing_atr_mult: float = 3.8
    trail_after_r: float = 1.0
    min_stop_pct: float = 0.006
    max_stop_pct: float = 0.035
    cooldown_bars: int = 8
    max_hold_bars: int = 96
    no_progress_bars: int = 24
    no_progress_min_mfe_r: float = 0.6


def build_btc_features(btc: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = btc.copy()
    out["btc_close"] = out["close"]
    out["btc_ema"] = ema(out["close"], cfg.btc_ema_period)
    out["btc_mom"] = out["close"] / out["close"].shift(cfg.btc_mom_lookback) - 1.0
    return out[["btc_close", "btc_ema", "btc_mom"]].dropna().copy()


def build_features(eth: pd.DataFrame, btc: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = eth.copy()
    btc_features = build_btc_features(btc, cfg)
    out = out.join(btc_features, how="inner")

    out["atr"] = atr(out, cfg.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]
    out["adx"] = adx(out, cfg.adx_period)
    out["volume_ma"] = out["volume"].rolling(cfg.volume_ma_period, min_periods=cfg.volume_ma_period).mean().shift(1)

    out["donchian_high"] = out["high"].rolling(cfg.donchian_lookback, min_periods=cfg.donchian_lookback).max().shift(1)
    out["donchian_low"] = out["low"].rolling(cfg.donchian_lookback, min_periods=cfg.donchian_lookback).min().shift(1)
    out["channel_width_pct"] = (out["donchian_high"] - out["donchian_low"]) / out["close"]

    out["eth_regime_ok"] = (
        out["atr_pct"].between(cfg.min_atr_pct, cfg.max_atr_pct)
        & (out["adx"] >= cfg.adx_min)
        & (out["channel_width_pct"] >= cfg.min_channel_width_pct)
        & (out["volume"] >= out["volume_ma"] * cfg.volume_mult)
    )
    out["btc_long_ok"] = (out["btc_close"] > out["btc_ema"]) & (out["btc_mom"] > cfg.btc_mom_threshold)
    out["btc_short_ok"] = (out["btc_close"] < out["btc_ema"]) & (out["btc_mom"] < -cfg.btc_mom_threshold)

    out["long_break"] = out["close"] > out["donchian_high"] * (1 + cfg.breakout_buffer_pct)
    out["short_break"] = out["close"] < out["donchian_low"] * (1 - cfg.breakout_buffer_pct)

    out["signal"] = 0
    out.loc[out["eth_regime_ok"] & out["btc_long_ok"] & out["long_break"], "signal"] = 1
    out.loc[out["eth_regime_ok"] & out["btc_short_ok"] & out["short_break"], "signal"] = -1

    out["stop"] = pd.NA
    out.loc[out["signal"] == 1, "stop"] = out["close"] - cfg.stop_atr_mult * out["atr"]
    out.loc[out["signal"] == -1, "stop"] = out["close"] + cfg.stop_atr_mult * out["atr"]
    return out.dropna().copy()


def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="ETH 1H Donchian breakout filtered by BTC trend/momentum regime.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--btc-symbol", default="BTC-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=today)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--risk-per-trade", type=float, default=0.003)
    p.add_argument("--fee-rate", type=float, default=0.0005)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--donchian-lookback", type=int, default=96)
    p.add_argument("--breakout-buffer-pct", type=float, default=0.002)
    p.add_argument("--adx-min", type=float, default=19.0)
    p.add_argument("--volume-mult", type=float, default=1.05)
    p.add_argument("--btc-ema-period", type=int, default=50)
    p.add_argument("--btc-mom-lookback", type=int, default=24)
    p.add_argument("--btc-mom-threshold", type=float, default=0.0)
    p.add_argument("--target-r", type=float, default=2.3)
    p.add_argument("--out-dir", default="data/reports/mf/eth_with_btc_regime_filter")
    p.add_argument("--write-full-audit", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = StrategyConfig(
        symbol=args.symbol,
        btc_symbol=args.btc_symbol,
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
        donchian_lookback=args.donchian_lookback,
        breakout_buffer_pct=args.breakout_buffer_pct,
        adx_min=args.adx_min,
        volume_mult=args.volume_mult,
        btc_ema_period=args.btc_ema_period,
        btc_mom_lookback=args.btc_mom_lookback,
        btc_mom_threshold=args.btc_mom_threshold,
        target_r=args.target_r,
    )
    eth = load_data(cfg.symbol, args.start_date, args.end_date, cfg.timeframe)
    btc = load_data(cfg.btc_symbol, args.start_date, args.end_date, cfg.timeframe)
    features = build_features(eth, btc, cfg)
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
