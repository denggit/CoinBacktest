#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH 1D+4H Momentum Breakout V1
==============================

定位：
    新的独立引擎。不同于 V4B 的趋势骑乘/延续，也不同于 Pullback 的回踩，
    本策略只做 1D 趋势环境下的 4H 强动量突破。

反偷看：
    - 1D regime 全部 shift(1) 后映射到 4H；
    - 4H entry_high / entry_low / exit channel 全部 rolling(...).shift(1)；
    - 当前 4H 收盘确认突破，下一根 4H open 执行；
    - 执行复用 V8 SAFE 引擎：当前 close 更新的新 stop 下一根 bar 才生效。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backtest_common.data import load_ohlcv_data as load_data  # noqa: E402
from src.backtest_common.indicators import adx, atr, ema, resample_ohlcv  # noqa: E402
from src.backtest_common.reporting import build_report_trades  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402

from backtest.lf.eth_1d_4h_trend_rider_v8_position_lock_backtest import (  # noqa: E402
    StrategyConfig as ExecConfig,
    run_backtest as run_exec_backtest,
    summarize,
)

STRATEGY_NAME = "eth_1d_4h_momentum_breakout_v1"
REPORT_STRATEGY_NAME = "ETH_1D_4H_MomentumBreakout_V1"

PRESETS: dict[str, dict[str, float | int]] = {
    "stable": {"unit_risk_per_trade": 0.012, "max_total_notional_mult": 8.0, "max_units": 4, "max_risk_mult": 2.0},
    "high": {"unit_risk_per_trade": 0.020, "max_total_notional_mult": 10.0, "max_units": 4, "max_risk_mult": 2.0},
    "turbo": {"unit_risk_per_trade": 0.026, "max_total_notional_mult": 11.0, "max_units": 4, "max_risk_mult": 2.0},
    "ultra": {"unit_risk_per_trade": 0.032, "max_total_notional_mult": 12.0, "max_units": 4, "max_risk_mult": 2.0},
}


@dataclass(frozen=True)
class MomentumConfig:
    symbol: str = "ETH-USDT-SWAP"
    initial_capital: float = 1000.0
    unit_risk_per_trade: float = 0.020
    max_total_notional_mult: float = 10.0
    max_units: int = 4
    min_risk_mult: float = 0.35
    max_risk_mult: float = 2.0
    fee_rate: float = 0.00055
    slippage_pct: float = 0.0002
    enable_short: bool = True

    # 4H breakout / filters.
    entry_lookback: int = 12
    exit_lookback: int = 12
    atr_period: int = 20
    adx_period: int = 14
    min_adx_long: float = 10.0
    min_adx_short: float = 16.0
    max_adx_long: float = 38.0
    max_adx_short: float = 42.0
    min_atr_pct: float = 0.0030
    max_atr_pct: float = 0.0700
    volume_window: int = 60
    volume_mult: float = 1.05
    max_d1_distance_long: float = 0.120
    max_d1_distance_short: float = 0.140

    # Execution, delegated to V8 SAFE engine.
    initial_atr_mult: float = 2.2
    trailing_atr_mult: float = 4.0
    add_every_r: float = 1.0
    max_hold_bars: int = 180
    cooldown_bars: int = 4
    breakeven_after_r: float = 1.0
    breakeven_lock_r: float = 0.10
    lock_after_2r: float = 1.7
    lock_2r: float = 0.70
    lock_after_3r: float = 2.8
    lock_3r: float = 1.50

    # 1D regime.
    d1_ema_fast: int = 8
    d1_ema_slow: int = 30
    d1_slope_lookback: int = 10
    bull_slope_min: float = -0.0300
    bear_slope_max: float = -0.0030


def to_exec_config(cfg: MomentumConfig) -> ExecConfig:
    return ExecConfig(
        symbol=cfg.symbol,
        initial_capital=cfg.initial_capital,
        unit_risk_per_trade=cfg.unit_risk_per_trade,
        max_total_notional_mult=cfg.max_total_notional_mult,
        max_units=cfg.max_units,
        min_risk_mult=cfg.min_risk_mult,
        max_risk_mult=cfg.max_risk_mult,
        fee_rate=cfg.fee_rate,
        slippage_pct=cfg.slippage_pct,
        enable_short=cfg.enable_short,
        initial_atr_mult=cfg.initial_atr_mult,
        trailing_atr_mult=cfg.trailing_atr_mult,
        add_every_r=cfg.add_every_r,
        max_hold_bars=cfg.max_hold_bars,
        cooldown_bars=cfg.cooldown_bars,
        breakeven_after_r=cfg.breakeven_after_r,
        breakeven_lock_r=cfg.breakeven_lock_r,
        lock_after_2r=cfg.lock_after_2r,
        lock_2r=cfg.lock_2r,
        lock_after_3r=cfg.lock_after_3r,
        lock_3r=cfg.lock_3r,
        no_progress_bars=10000,
    )


def build_daily_regime(base_4h: pd.DataFrame, cfg: MomentumConfig) -> pd.DataFrame:
    d1 = resample_ohlcv(base_4h, "1D")
    d1["d1_ema_fast"] = ema(d1["close"], cfg.d1_ema_fast)
    d1["d1_ema_slow"] = ema(d1["close"], cfg.d1_ema_slow)
    d1["d1_slow_slope"] = d1["d1_ema_slow"] / d1["d1_ema_slow"].shift(cfg.d1_slope_lookback) - 1.0
    d1["d1_bull"] = (
        (d1["close"] > d1["d1_ema_slow"])
        & (d1["d1_ema_fast"] > d1["d1_ema_slow"] * 0.995)
        & (d1["d1_slow_slope"] > cfg.bull_slope_min)
    )
    d1["d1_bear"] = (
        (d1["close"] < d1["d1_ema_slow"])
        & (d1["d1_ema_fast"] < d1["d1_ema_slow"])
        & (d1["d1_slow_slope"] < cfg.bear_slope_max)
    )
    cols = ["d1_ema_fast", "d1_ema_slow", "d1_slow_slope", "d1_bull", "d1_bear"]
    return d1[cols].shift(1)


def build_features(base_4h: pd.DataFrame, cfg: MomentumConfig) -> pd.DataFrame:
    out = base_4h.copy()
    out["atr"] = atr(out, cfg.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]
    out["adx"] = adx(out, cfg.adx_period)
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["ema100"] = ema(out["close"], 100)
    out["ema200"] = ema(out["close"], 200)

    d1 = build_daily_regime(base_4h, cfg)
    out = out.join(d1.reindex(out.index, method="ffill"))

    out["entry_high"] = out["high"].rolling(cfg.entry_lookback, min_periods=cfg.entry_lookback).max().shift(1)
    out["entry_low"] = out["low"].rolling(cfg.entry_lookback, min_periods=cfg.entry_lookback).min().shift(1)
    out["exit_low"] = out["low"].rolling(cfg.exit_lookback, min_periods=cfg.exit_lookback).min().shift(1)
    out["exit_high"] = out["high"].rolling(cfg.exit_lookback, min_periods=cfg.exit_lookback).max().shift(1)
    out["volume_median"] = out["volume"].rolling(cfg.volume_window, min_periods=30).median().shift(1)

    d1_bull = out["d1_bull"].astype("boolean").fillna(False).astype(bool)
    d1_bear = out["d1_bear"].astype("boolean").fillna(False).astype(bool)
    d1_distance = out["close"] / out["d1_ema_slow"] - 1.0
    out["d1_distance"] = d1_distance

    vol_ok = out["volume"] > out["volume_median"] * cfg.volume_mult
    atr_ok = out["atr_pct"].between(cfg.min_atr_pct, cfg.max_atr_pct)

    long_filter = (
        d1_bull
        & atr_ok
        & out["adx"].between(cfg.min_adx_long, cfg.max_adx_long)
        & (d1_distance.abs() < cfg.max_d1_distance_long)
    )
    short_filter = (
        d1_bear
        & atr_ok
        & out["adx"].between(cfg.min_adx_short, cfg.max_adx_short)
        & (d1_distance.abs() < cfg.max_d1_distance_short)
        & cfg.enable_short
    )

    out["long_breakout_setup"] = (
        (out["close"] > out["entry_high"])
        & (out["close"] > out["open"])
        & (out["close"] > out["ema50"])
        & (out["ema20"] > out["ema50"])
        & vol_ok
    )
    out["short_breakout_setup"] = (
        (out["close"] < out["entry_low"])
        & (out["close"] < out["open"])
        & (out["close"] < out["ema50"])
        & (out["ema20"] < out["ema50"])
        & vol_ok
    )

    out["long_signal"] = long_filter & out["long_breakout_setup"]
    out["short_signal"] = short_filter & out["short_breakout_setup"]
    out["signal"] = 0
    out.loc[out["long_signal"], "signal"] = 1
    out.loc[out["short_signal"], "signal"] = -1

    out["long_exit_channel"] = out["close"] < out["exit_low"]
    out["short_exit_channel"] = out["close"] > out["exit_high"]

    # Coarse risk model. Uses only current closed 4H bar and shifted 1D regime.
    out["risk_mult"] = 1.0
    out.loc[out["adx"].between(14.0, 30.0), "risk_mult"] += 0.20
    out.loc[out["atr_pct"] > 0.040, "risk_mult"] -= 0.25
    out.loc[d1_distance.abs() > 0.100, "risk_mult"] -= 0.20
    out["risk_mult"] = out["risk_mult"].clip(cfg.min_risk_mult, cfg.max_risk_mult)

    out["quality_mult"] = 1.0
    out.loc[out["volume"] > out["volume_median"] * 1.50, "quality_mult"] *= 1.10

    return out.dropna().copy()


def run_strategy(features: pd.DataFrame, cfg: MomentumConfig) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    return run_exec_backtest(features, to_exec_config(cfg))


def write_outputs(trades: list[dict[str, Any]], equity: pd.DataFrame, features: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / f"{STRATEGY_NAME}_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / f"{STRATEGY_NAME}_equity.csv")
    audit_cols = [
        "open", "high", "low", "close", "volume", "atr", "atr_pct", "adx",
        "ema20", "ema50", "ema100", "ema200",
        "d1_ema_fast", "d1_ema_slow", "d1_slow_slope", "d1_bull", "d1_bear", "d1_distance",
        "entry_high", "entry_low", "exit_low", "exit_high",
        "volume_median", "risk_mult", "quality_mult",
        "long_breakout_setup", "short_breakout_setup", "long_signal", "short_signal", "signal",
    ]
    features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{STRATEGY_NAME}_signal_audit.csv")
    with (out_dir / f"{STRATEGY_NAME}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 92)
    print("ETH 1D+4H Momentum Breakout V1 Backtest Summary")
    print("=" * 92)
    for k, v in summary.items():
        print(f"{k:>34}: {v}")
    print("-" * 92)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 92 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETH 1D+4H Momentum Breakout V1")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-15")
    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default="high")
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--unit-risk-per-trade", type=float, default=None)
    p.add_argument("--max-total-notional-mult", type=float, default=None)
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--max-risk-mult", type=float, default=None)
    p.add_argument("--disable-short", action="store_true")
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preset = dict(PRESETS[args.preset])
    if args.unit_risk_per_trade is not None:
        preset["unit_risk_per_trade"] = args.unit_risk_per_trade
    if args.max_total_notional_mult is not None:
        preset["max_total_notional_mult"] = args.max_total_notional_mult
    if args.max_units is not None:
        preset["max_units"] = args.max_units
    if args.max_risk_mult is not None:
        preset["max_risk_mult"] = args.max_risk_mult

    cfg = MomentumConfig(
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        unit_risk_per_trade=float(preset["unit_risk_per_trade"]),
        max_total_notional_mult=float(preset["max_total_notional_mult"]),
        max_units=int(preset["max_units"]),
        max_risk_mult=float(preset["max_risk_mult"]),
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
        enable_short=not args.disable_short,
    )
    out_dir = Path(args.out_dir) if args.out_dir else Path("data/reports/lf") / STRATEGY_NAME / args.preset

    print(f"Loading {cfg.symbol} 4H: {args.start_date} -> {args.end_date}")
    base = load_data(cfg.symbol, args.start_date, args.end_date, "4H")
    print(f"Loaded {len(base)} rows: {base.index.min()} -> {base.index.max()}")
    features = build_features(base, cfg)
    print(f"Feature rows: {len(features)}")
    print("Signal counts:", {"long": int(features["long_signal"].sum()), "short": int(features["short_signal"].sum())})

    trades, equity = run_strategy(features, cfg)
    summary = summarize(trades, equity, cfg.initial_capital)
    summary.update({
        "preset": args.preset,
        "engine_role": "directional_momentum_breakout_candidate",
        "fee_rate_per_side": cfg.fee_rate,
    })
    write_outputs(trades, equity, features, summary, out_dir)
    print_summary(summary, out_dir)

    if trades:
        total_days = (pd.to_datetime(features.index.max()) - pd.to_datetime(features.index.min())).total_seconds() / 86400.0
        print_full_report(
            trade_history=build_report_trades(trades),
            df=features,
            initial_capital=cfg.initial_capital,
            capital=float(pd.DataFrame(trades).iloc[-1]["capital"]),
            strategy_name=REPORT_STRATEGY_NAME,
            total_days=total_days,
            ai_enabled=False,
            symbol=cfg.symbol,
            report_dir=out_dir,
        )


if __name__ == "__main__":
    main()
