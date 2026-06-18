#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Intraday Momentum/Reversal Backtest.

Hypothesis:
    ETH has hour-of-day dependent intraday momentum and reversal. The same 1H
    impulse can be traded as continuation during active windows and as fading
    during weaker windows.
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
from src.backtest_common.indicators import atr, ema  # noqa: E402
from src.backtest_common.ohlcv_backtest import (  # noqa: E402
    SignalBacktestParams,
    emit_signal_report,
    print_signal_summary,
    run_signal_backtest,
    summarize_signal_backtest,
    write_signal_outputs,
)

STRATEGY_NAME = "eth_intraday_momentum_reversal"


def parse_hours(raw: str) -> set[int]:
    return {int(x.strip()) for x in raw.split(",") if x.strip() != ""}


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "ETH-USDT-SWAP"
    timeframe: str = "1H"
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.0025
    max_notional_mult: float = 3.0
    fee_rate: float = 0.0005
    slippage_pct: float = 0.0002

    ema_period: int = 50
    atr_period: int = 14
    momentum_ret_threshold: float = 0.008
    reversal_ret_threshold: float = 0.018
    volume_ma_period: int = 20
    volume_mult: float = 0.9
    stop_atr_mult: float = 1.4

    momentum_hours: tuple[int, ...] = (0, 1, 2, 3, 12, 13, 14, 15, 16)
    reversal_hours: tuple[int, ...] = (4, 5, 6, 20, 21, 22, 23)

    target_r: float = 1.6
    min_stop_pct: float = 0.004
    max_stop_pct: float = 0.025
    cooldown_bars: int = 2
    max_hold_bars: int = 8
    no_progress_bars: int = 3
    no_progress_min_mfe_r: float = 0.4


def build_features(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    out["ema"] = ema(out["close"], cfg.ema_period)
    out["atr"] = atr(out, cfg.atr_period)
    out["ret_1h"] = out["close"].pct_change()
    out["prev_ret_1h"] = out["ret_1h"].shift(1)
    out["hour"] = out.index.hour
    out["volume_ma"] = out["volume"].rolling(cfg.volume_ma_period, min_periods=cfg.volume_ma_period).mean().shift(1)
    out["volume_ok"] = out["volume"] > out["volume_ma"] * cfg.volume_mult

    momo_hours = set(cfg.momentum_hours)
    rev_hours = set(cfg.reversal_hours)
    out["is_momentum_hour"] = out["hour"].isin(momo_hours)
    out["is_reversal_hour"] = out["hour"].isin(rev_hours)

    out["momo_long"] = out["is_momentum_hour"] & out["volume_ok"] & (out["prev_ret_1h"] > cfg.momentum_ret_threshold) & (out["close"] > out["ema"])
    out["momo_short"] = out["is_momentum_hour"] & out["volume_ok"] & (out["prev_ret_1h"] < -cfg.momentum_ret_threshold) & (out["close"] < out["ema"])
    out["rev_short"] = out["is_reversal_hour"] & (out["prev_ret_1h"] > cfg.reversal_ret_threshold) & (out["close"] < out["high"].shift(1))
    out["rev_long"] = out["is_reversal_hour"] & (out["prev_ret_1h"] < -cfg.reversal_ret_threshold) & (out["close"] > out["low"].shift(1))

    out["signal"] = 0
    out.loc[out["momo_long"] | out["rev_long"], "signal"] = 1
    out.loc[out["momo_short"] | out["rev_short"], "signal"] = -1
    out["signal_reason"] = ""
    out.loc[out["momo_long"], "signal_reason"] = "MOMO_LONG"
    out.loc[out["momo_short"], "signal_reason"] = "MOMO_SHORT"
    out.loc[out["rev_long"], "signal_reason"] = "REV_LONG"
    out.loc[out["rev_short"], "signal_reason"] = "REV_SHORT"
    out["stop"] = pd.NA
    out.loc[out["signal"] == 1, "stop"] = out["close"] - cfg.stop_atr_mult * out["atr"]
    out.loc[out["signal"] == -1, "stop"] = out["close"] + cfg.stop_atr_mult * out["atr"]
    return out.dropna().copy()


def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="ETH 1H intraday momentum/reversal backtest.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=today)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--risk-per-trade", type=float, default=0.0025)
    p.add_argument("--fee-rate", type=float, default=0.0005)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--momentum-ret-threshold", type=float, default=0.008)
    p.add_argument("--reversal-ret-threshold", type=float, default=0.018)
    p.add_argument("--momentum-hours", default="0,1,2,3,12,13,14,15,16")
    p.add_argument("--reversal-hours", default="4,5,6,20,21,22,23")
    p.add_argument("--target-r", type=float, default=1.6)
    p.add_argument("--out-dir", default="data/reports/mf/eth_intraday_momentum_reversal")
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
        momentum_ret_threshold=args.momentum_ret_threshold,
        reversal_ret_threshold=args.reversal_ret_threshold,
        momentum_hours=tuple(sorted(parse_hours(args.momentum_hours))),
        reversal_hours=tuple(sorted(parse_hours(args.reversal_hours))),
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
