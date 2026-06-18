#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Low-ADX VWAP Mean Reversion Backtest.

Hypothesis:
    When the 1H market is non-trending, large 15M deviations from rolling VWAP
    are more likely to mean-revert than continue.
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
from src.backtest_common.indicators import adx, atr, resample_ohlcv  # noqa: E402
from src.backtest_common.ohlcv_backtest import (  # noqa: E402
    SignalBacktestParams,
    emit_signal_report,
    print_signal_summary,
    run_signal_backtest,
    summarize_signal_backtest,
    write_signal_outputs,
)

STRATEGY_NAME = "eth_low_adx_vwap_mean_reversion"


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "ETH-USDT-SWAP"
    timeframe: str = "15m"
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.0025
    max_notional_mult: float = 2.5
    fee_rate: float = 0.0005
    slippage_pct: float = 0.0002

    h1_adx_period: int = 14
    h1_adx_max: float = 17.0
    vwap_window: int = 96
    z_window: int = 96
    z_entry: float = 2.0
    atr_period: int = 14
    stop_atr_mult: float = 1.2
    min_abs_deviation_pct: float = 0.006

    target_r: float = 1.2
    min_stop_pct: float = 0.004
    max_stop_pct: float = 0.018
    cooldown_bars: int = 8
    max_hold_bars: int = 32
    no_progress_bars: int = 12
    no_progress_min_mfe_r: float = 0.35


def build_h1_adx(base_15m: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    h1 = resample_ohlcv(base_15m, "1h")
    h1["h1_adx"] = adx(h1, cfg.h1_adx_period)
    return h1[["h1_adx"]].shift(1)


def build_features(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    h1 = build_h1_adx(out, cfg)
    out = out.join(h1.reindex(out.index, method="ffill"))
    out["typical"] = (out["high"] + out["low"] + out["close"]) / 3
    pv = out["typical"] * out["volume"]
    out["vwap"] = pv.rolling(cfg.vwap_window, min_periods=cfg.vwap_window).sum() / out["volume"].rolling(cfg.vwap_window, min_periods=cfg.vwap_window).sum()
    out["dev"] = out["close"] - out["vwap"]
    out["dev_std"] = out["dev"].rolling(cfg.z_window, min_periods=cfg.z_window).std().shift(1)
    out["zscore"] = out["dev"] / out["dev_std"]
    out["abs_dev_pct"] = (out["close"] - out["vwap"]).abs() / out["close"]
    out["atr"] = atr(out, cfg.atr_period)
    out["regime_ok"] = (out["h1_adx"] <= cfg.h1_adx_max) & (out["abs_dev_pct"] >= cfg.min_abs_deviation_pct)
    out["long_signal"] = out["regime_ok"] & (out["zscore"] <= -cfg.z_entry)
    out["short_signal"] = out["regime_ok"] & (out["zscore"] >= cfg.z_entry)
    out["signal"] = 0
    out.loc[out["long_signal"], "signal"] = 1
    out.loc[out["short_signal"], "signal"] = -1
    out["stop"] = pd.NA
    out.loc[out["signal"] == 1, "stop"] = out["close"] - cfg.stop_atr_mult * out["atr"]
    out.loc[out["signal"] == -1, "stop"] = out["close"] + cfg.stop_atr_mult * out["atr"]
    out["target"] = out["vwap"]
    return out.dropna().copy()


def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="ETH low-ADX VWAP mean-reversion backtest.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=today)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--risk-per-trade", type=float, default=0.0025)
    p.add_argument("--fee-rate", type=float, default=0.0005)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--h1-adx-max", type=float, default=17.0)
    p.add_argument("--z-entry", type=float, default=2.0)
    p.add_argument("--out-dir", default="data/reports/mf/eth_low_adx_vwap_mean_reversion")
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
        h1_adx_max=args.h1_adx_max,
        z_entry=args.z_entry,
    )
    base = load_data(cfg.symbol, args.start_date, args.end_date, cfg.timeframe)
    features = build_features(base, cfg)
    params = SignalBacktestParams(
        initial_capital=cfg.initial_capital,
        risk_per_trade=cfg.risk_per_trade,
        max_notional_mult=cfg.max_notional_mult,
        fee_rate=cfg.fee_rate,
        slippage_pct=cfg.slippage_pct,
        target_col="target",
        target_r=cfg.target_r,
        min_stop_pct=cfg.min_stop_pct,
        max_stop_pct=cfg.max_stop_pct,
        cooldown_bars=cfg.cooldown_bars,
        max_hold_bars=cfg.max_hold_bars,
        no_progress_bars=cfg.no_progress_bars,
        no_progress_min_mfe_r=cfg.no_progress_min_mfe_r,
        exit_on_opposite_signal=False,
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
