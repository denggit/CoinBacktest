#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH 15m Timeframe Probe V1
==========================

独立测试版本，不并入 LF portfolio。

目的：测试极端低级别 15m timeframe 是否能在 ETH 上抓更多小趋势。

原则：
    - 不修改 V6/V7 portfolio 文件。
    - 不用年份/月度过滤。
    - 1D / 4H regime 全部 shift(1) 后映射到 15m，避免未来函数。
    - 15m Donchian entry/exit 全部 rolling(...).shift(1)。
    - 当前 15m close 确认信号，下一根 15m open 执行。
    - 当前 bar close 更新的新 stop 下一根 bar 才生效，由 V8 SAFE execution engine 负责。
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

STRATEGY_NAME = "eth_15m_timeframe_probe_v1"
REPORT_STRATEGY_NAME = "ETH_15M_TimeframeProbe_V1"

PRESETS: dict[str, dict[str, float | int]] = {
    # 15m 噪音更大，默认风险显著低于 LF portfolio。
    "stable": {"unit_risk_per_trade": 0.0030, "max_total_notional_mult": 5.0, "max_units": 3, "max_risk_mult": 1.5},
    "high": {"unit_risk_per_trade": 0.0040, "max_total_notional_mult": 6.0, "max_units": 3, "max_risk_mult": 1.8},
    "turbo": {"unit_risk_per_trade": 0.0060, "max_total_notional_mult": 7.0, "max_units": 4, "max_risk_mult": 2.0},
}


@dataclass(frozen=True)
class ProbeConfig:
    symbol: str = "ETH-USDT-SWAP"
    initial_capital: float = 1000.0
    unit_risk_per_trade: float = 0.0030
    max_total_notional_mult: float = 5.0
    max_units: int = 3
    min_risk_mult: float = 0.35
    max_risk_mult: float = 1.5
    fee_rate: float = 0.00055
    slippage_pct: float = 0.0002
    enable_short: bool = True

    # 15m structure windows. These are intentionally not optimized; they are broad probe settings.
    entry_lookback: int = 96       # 24h on 15m
    exit_lookback: int = 64        # 16h on 15m
    atr_period: int = 96           # 24h ATR smoothing
    adx_period: int = 96
    volume_window: int = 192       # 48h
    volume_mult: float = 1.05
    min_adx_long: float = 10.0
    min_adx_short: float = 12.0
    max_adx: float = 55.0
    min_atr_pct: float = 0.0008
    max_atr_pct: float = 0.025

    # 15m trend EMAs.
    ema_fast: int = 96             # 24h
    ema_mid: int = 320             # 80h, roughly 4H EMA20 horizon
    ema_slow: int = 800            # 200h, roughly 4H EMA50 horizon

    # 4H and 1D regimes, shifted before mapping down.
    h4_ema_fast: int = 20
    h4_ema_mid: int = 50
    h4_slope_lookback: int = 12
    d1_ema_fast: int = 8
    d1_ema_slow: int = 30
    d1_slope_lookback: int = 10
    bull_d1_slope_min: float = -0.030
    bear_d1_slope_max: float = -0.003

    # Execution on 15m bars.
    initial_atr_mult: float = 4.0
    trailing_atr_mult: float = 8.0
    add_every_r: float = 1.5
    max_hold_bars: int = 672       # 7 days on 15m
    cooldown_bars: int = 16        # 4 hours
    breakeven_after_r: float = 1.2
    breakeven_lock_r: float = 0.05
    lock_after_2r: float = 2.0
    lock_2r: float = 0.50
    lock_after_3r: float = 3.0
    lock_3r: float = 1.20


def to_exec_config(cfg: ProbeConfig) -> ExecConfig:
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


def shifted_d1_regime(base_15m: pd.DataFrame, cfg: ProbeConfig) -> pd.DataFrame:
    d1 = resample_ohlcv(base_15m, "1d")
    d1["d1_close"] = d1["close"]
    d1["d1_ema_fast"] = ema(d1["close"], cfg.d1_ema_fast)
    d1["d1_ema_slow"] = ema(d1["close"], cfg.d1_ema_slow)
    d1["d1_slow_slope"] = d1["d1_ema_slow"] / d1["d1_ema_slow"].shift(cfg.d1_slope_lookback) - 1.0
    d1["d1_bull"] = (d1["close"] > d1["d1_ema_slow"]) & (d1["d1_ema_fast"] > d1["d1_ema_slow"] * 0.995) & (d1["d1_slow_slope"] > cfg.bull_d1_slope_min)
    d1["d1_bear"] = (d1["close"] < d1["d1_ema_slow"]) & (d1["d1_ema_fast"] < d1["d1_ema_slow"]) & (d1["d1_slow_slope"] < cfg.bear_d1_slope_max)
    return d1[["d1_close", "d1_ema_fast", "d1_ema_slow", "d1_slow_slope", "d1_bull", "d1_bear"]].shift(1)


def shifted_h4_regime(base_15m: pd.DataFrame, cfg: ProbeConfig) -> pd.DataFrame:
    h4 = resample_ohlcv(base_15m, "4h")
    h4["h4_close"] = h4["close"]
    h4["h4_ema_fast"] = ema(h4["close"], cfg.h4_ema_fast)
    h4["h4_ema_mid"] = ema(h4["close"], cfg.h4_ema_mid)
    h4["h4_mid_slope"] = h4["h4_ema_mid"] / h4["h4_ema_mid"].shift(cfg.h4_slope_lookback) - 1.0
    h4["h4_bull"] = (h4["close"] > h4["h4_ema_mid"]) & (h4["h4_ema_fast"] > h4["h4_ema_mid"]) & (h4["h4_mid_slope"] > -0.010)
    h4["h4_bear"] = (h4["close"] < h4["h4_ema_mid"]) & (h4["h4_ema_fast"] < h4["h4_ema_mid"]) & (h4["h4_mid_slope"] < 0.000)
    return h4[["h4_close", "h4_ema_fast", "h4_ema_mid", "h4_mid_slope", "h4_bull", "h4_bear"]].shift(1)


def build_features(base: pd.DataFrame, cfg: ProbeConfig) -> pd.DataFrame:
    out = base.copy()
    out["atr"] = atr(out, cfg.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]
    out["adx"] = adx(out, cfg.adx_period)
    out["ema_fast"] = ema(out["close"], cfg.ema_fast)
    out["ema_mid"] = ema(out["close"], cfg.ema_mid)
    out["ema_slow"] = ema(out["close"], cfg.ema_slow)

    d1 = shifted_d1_regime(base, cfg)
    out = out.join(d1.reindex(out.index, method="ffill"))
    h4 = shifted_h4_regime(base, cfg)
    out = out.join(h4.reindex(out.index, method="ffill"))

    out["entry_high"] = out["high"].rolling(cfg.entry_lookback, min_periods=cfg.entry_lookback).max().shift(1)
    out["entry_low"] = out["low"].rolling(cfg.entry_lookback, min_periods=cfg.entry_lookback).min().shift(1)
    out["exit_low"] = out["low"].rolling(cfg.exit_lookback, min_periods=cfg.exit_lookback).min().shift(1)
    out["exit_high"] = out["high"].rolling(cfg.exit_lookback, min_periods=cfg.exit_lookback).max().shift(1)
    out["volume_median"] = out["volume"].rolling(cfg.volume_window, min_periods=max(30, cfg.volume_window // 4)).median().shift(1)

    d1_bull = out["d1_bull"].astype("boolean").fillna(False).astype(bool)
    d1_bear = out["d1_bear"].astype("boolean").fillna(False).astype(bool)
    h4_bull = out["h4_bull"].astype("boolean").fillna(False).astype(bool)
    h4_bear = out["h4_bear"].astype("boolean").fillna(False).astype(bool)

    atr_ok = out["atr_pct"].between(cfg.min_atr_pct, cfg.max_atr_pct)
    vol_ok = out["volume"] > out["volume_median"] * cfg.volume_mult
    long_filter = d1_bull & h4_bull & atr_ok & out["adx"].between(cfg.min_adx_long, cfg.max_adx)
    short_filter = d1_bear & h4_bear & atr_ok & out["adx"].between(cfg.min_adx_short, cfg.max_adx) & cfg.enable_short

    out["long_breakout_setup"] = (
        (out["close"] > out["entry_high"])
        & (out["close"] > out["open"])
        & (out["close"] > out["ema_mid"])
        & (out["ema_fast"] > out["ema_mid"])
        & vol_ok
    )
    out["short_breakout_setup"] = (
        (out["close"] < out["entry_low"])
        & (out["close"] < out["open"])
        & (out["close"] < out["ema_mid"])
        & (out["ema_fast"] < out["ema_mid"])
        & vol_ok
    )

    out["long_signal"] = out["long_breakout_setup"] & long_filter
    out["short_signal"] = out["short_breakout_setup"] & short_filter
    out["signal"] = 0
    out.loc[out["long_signal"], "signal"] = 1
    out.loc[out["short_signal"], "signal"] = -1
    out["long_exit_channel"] = out["close"] < out["exit_low"]
    out["short_exit_channel"] = out["close"] > out["exit_high"]

    out["risk_mult"] = 1.0
    out["quality_mult"] = 1.0
    # Mild quality downweight for late/chasing breakouts.
    out.loc[out["long_signal"] & (out["adx"] > 35), "quality_mult"] = 0.70
    out.loc[out["short_signal"] & (out["adx"] > 38), "quality_mult"] = 0.80
    return out


def write_outputs(trades: list[dict[str, Any]], equity: pd.DataFrame, features: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / f"{STRATEGY_NAME}_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / f"{STRATEGY_NAME}_equity.csv")
    audit_cols = [
        "open", "high", "low", "close", "volume", "atr", "atr_pct", "adx",
        "ema_fast", "ema_mid", "ema_slow", "d1_bull", "d1_bear", "h4_bull", "h4_bear",
        "entry_high", "entry_low", "exit_low", "exit_high",
        "risk_mult", "quality_mult", "long_signal", "short_signal", "signal",
        "long_exit_channel", "short_exit_channel",
    ]
    features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{STRATEGY_NAME}_signal_audit.csv")
    with (out_dir / f"{STRATEGY_NAME}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 92)
    print("ETH 15m Timeframe Probe V1 Backtest Summary")
    print("=" * 92)
    for k, v in summary.items():
        print(f"{k:>34}: {v}")
    print("-" * 92)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 92 + "\n")


def print_deep_report(trades: list[dict[str, Any]], features: pd.DataFrame, cfg: ProbeConfig, out_dir: Path) -> None:
    if not trades or features.empty:
        return
    total_days = max((features.index[-1] - features.index[0]).total_seconds() / 86400.0, 1e-9)
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


def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="ETH 15m standalone timeframe probe. Not part of LF portfolio.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=today)
    p.add_argument("--warmup-start-date", default=None)
    p.add_argument("--warmup-days", type=int, default=365)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--preset", choices=sorted(PRESETS), default="stable")
    p.add_argument("--unit-risk-per-trade", type=float, default=None)
    p.add_argument("--max-total-notional-mult", type=float, default=None)
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--min-risk-mult", type=float, default=0.35)
    p.add_argument("--max-risk-mult", type=float, default=None)
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--disable-short", action="store_true")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def make_config(args: argparse.Namespace) -> ProbeConfig:
    preset = PRESETS[args.preset]
    return ProbeConfig(
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        unit_risk_per_trade=float(args.unit_risk_per_trade if args.unit_risk_per_trade is not None else preset["unit_risk_per_trade"]),
        max_total_notional_mult=float(args.max_total_notional_mult if args.max_total_notional_mult is not None else preset["max_total_notional_mult"]),
        max_units=int(args.max_units if args.max_units is not None else preset["max_units"]),
        min_risk_mult=args.min_risk_mult,
        max_risk_mult=float(args.max_risk_mult if args.max_risk_mult is not None else preset["max_risk_mult"]),
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
        enable_short=not args.disable_short,
    )


def main() -> int:
    args = parse_args()
    cfg = make_config(args)
    exec_cfg = to_exec_config(cfg)
    out_dir = Path(args.out_dir) if args.out_dir else Path(PROJECT_ROOT) / "data/reports/mf" / STRATEGY_NAME / args.preset

    trade_start = pd.Timestamp(args.start_date)
    if args.warmup_start_date:
        load_start = pd.Timestamp(args.warmup_start_date)
    elif args.warmup_days and args.warmup_days > 0:
        load_start = trade_start - pd.Timedelta(days=int(args.warmup_days))
    else:
        load_start = trade_start
    load_start_str = load_start.strftime("%Y-%m-%d")

    print(f"Loading {args.symbol} 15m for warmup: {load_start_str} -> {args.end_date}; trade_start={args.start_date}")
    base = load_data(args.symbol, load_start_str, args.end_date, "15m")
    if base.empty:
        raise RuntimeError("No 15m data loaded. Please make sure local DB has 15m OHLCV.")
    print(f"Loaded {len(base)} rows: {base.index[0]} -> {base.index[-1]}")
    features = build_features(base, cfg)
    before_slice_rows = len(features)
    features = features.loc[trade_start: pd.Timestamp(args.end_date)].copy()
    print(f"Feature rows after warmup slice: {len(features)} / {before_slice_rows}; first tradeable bar={features.index[0] if not features.empty else 'NA'}")
    print("Signal counts:", {
        "long_signal": int(features.long_signal.sum()),
        "short_signal": int(features.short_signal.sum()),
        "signal_long": int((features.signal == 1).sum()),
        "signal_short": int((features.signal == -1).sum()),
    })

    trades, equity = run_exec_backtest(features, exec_cfg)
    summary = summarize(trades, equity, exec_cfg.initial_capital)
    summary["preset"] = args.preset
    summary["timeframe"] = "15m"
    summary["standalone_probe"] = True
    summary["not_portfolio_component"] = True
    summary["fee_rate_per_side"] = args.fee_rate
    summary["warmup_start_date"] = load_start_str
    summary["trade_start_date"] = args.start_date
    summary["warmup_days"] = int(args.warmup_days or 0)

    write_outputs(trades, equity, features, summary, out_dir)
    print_summary(summary, out_dir)
    print_deep_report(trades, features, cfg, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
