#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH 1D+4H Pullback Continuation V1
==================================

定位：
    第三个独立引擎的第一版。它不是为了替代 V4B 这种低胜率高盈亏比趋势系统，
    而是尝试提供更高胜率、更短持仓的顺趋势回踩延续交易。

反偷看：
    - 1D regime 全部 shift(1) 后映射到 4H；
    - 4H 信号只使用当前已经收盘的 bar；
    - 当前 4H 收盘确认，下一根 4H open 执行；
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
from src.backtest_common.indicators import adx, atr, ema, resample_ohlcv, rsi  # noqa: E402
from src.backtest_common.reporting import build_report_trades  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402

from backtest.lf.eth_1d_4h_trend_rider_v8_position_lock_backtest import (  # noqa: E402
    StrategyConfig as ExecConfig,
    run_backtest as run_exec_backtest,
    summarize,
)

STRATEGY_NAME = "eth_1d_4h_pullback_continuation_v1"
REPORT_STRATEGY_NAME = "ETH_1D_4H_PullbackContinuation_V1"

PRESETS: dict[str, dict[str, float | int]] = {
    "stable": {"unit_risk_per_trade": 0.006, "max_total_notional_mult": 4.0, "max_units": 1, "max_risk_mult": 1.6},
    "high": {"unit_risk_per_trade": 0.010, "max_total_notional_mult": 6.0, "max_units": 1, "max_risk_mult": 1.8},
    "turbo": {"unit_risk_per_trade": 0.014, "max_total_notional_mult": 7.0, "max_units": 1, "max_risk_mult": 2.0},
}


@dataclass(frozen=True)
class PullbackConfig:
    symbol: str = "ETH-USDT-SWAP"
    initial_capital: float = 1000.0
    unit_risk_per_trade: float = 0.010
    max_total_notional_mult: float = 6.0
    max_units: int = 1
    min_risk_mult: float = 0.30
    max_risk_mult: float = 1.8
    fee_rate: float = 0.00055
    slippage_pct: float = 0.0002
    enable_short: bool = True

    atr_period: int = 20
    adx_period: int = 14
    rsi_period: int = 14
    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 89
    pullback_lookback: int = 8
    pullback_band: float = 0.006
    min_adx_long: float = 8.0
    min_adx_short: float = 14.0
    max_adx: float = 36.0
    min_atr_pct: float = 0.0035
    max_atr_pct: float = 0.045
    max_d1_distance_long: float = 0.090
    max_d1_distance_short: float = 0.080
    max_ema50_distance: float = 0.055

    # V8 SAFE execution settings, tuned shorter than trend rider.
    initial_atr_mult: float = 2.0
    trailing_atr_mult: float = 3.0
    add_every_r: float = 1.0
    max_hold_bars: int = 90      # 15 days
    cooldown_bars: int = 4
    breakeven_after_r: float = 1.0
    breakeven_lock_r: float = 0.05
    lock_after_2r: float = 1.6
    lock_2r: float = 0.55
    lock_after_3r: float = 2.6
    lock_3r: float = 1.25

    # 1D regime.
    d1_ema_fast: int = 8
    d1_ema_slow: int = 30
    d1_slope_lookback: int = 10
    bull_slope_min: float = -0.020
    bear_slope_max: float = -0.004


def to_exec_config(cfg: PullbackConfig) -> ExecConfig:
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


def build_daily_regime(base_4h: pd.DataFrame, cfg: PullbackConfig) -> pd.DataFrame:
    d1 = resample_ohlcv(base_4h, "1D")
    d1["d1_close"] = d1["close"]
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
        & (d1["d1_slow_slope"] <= cfg.bear_slope_max)
    )
    # Shift by one day before mapping to 4H. Current incomplete day is never used.
    cols = ["d1_close", "d1_ema_fast", "d1_ema_slow", "d1_slow_slope", "d1_bull", "d1_bear"]
    shifted = pd.DataFrame({c: d1[c].shift(1) for c in cols}, index=d1.index)
    return shifted


def build_features(base_4h: pd.DataFrame, cfg: PullbackConfig) -> pd.DataFrame:
    out = base_4h.copy()
    out["ema20"] = ema(out["close"], cfg.ema_fast)
    out["ema50"] = ema(out["close"], cfg.ema_mid)
    out["ema89"] = ema(out["close"], cfg.ema_slow)
    out["atr"] = atr(out, cfg.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]
    out["adx"] = adx(out, cfg.adx_period)
    out["rsi"] = rsi(out["close"], cfg.rsi_period)
    out["rsi_prev"] = out["rsi"].shift(1)

    d1 = build_daily_regime(base_4h, cfg)
    out = out.join(d1.reindex(out.index, method="ffill"))

    out["trend_long"] = (
        out["d1_bull"].astype("boolean").fillna(False).astype(bool)
        & (out["ema20"] > out["ema50"])
        & (out["close"] > out["ema50"])
        & out["adx"].between(cfg.min_adx_long, cfg.max_adx)
        & out["atr_pct"].between(cfg.min_atr_pct, cfg.max_atr_pct)
    )
    out["trend_short"] = (
        out["d1_bear"].astype("boolean").fillna(False).astype(bool)
        & (out["ema20"] < out["ema50"])
        & (out["close"] < out["ema50"])
        & out["adx"].between(cfg.min_adx_short, cfg.max_adx)
        & out["atr_pct"].between(cfg.min_atr_pct, cfg.max_atr_pct)
        & cfg.enable_short
    )

    long_touch = (out["low"] <= out["ema20"] * (1.0 + cfg.pullback_band)) | (out["low"] <= out["ema50"] * (1.0 + cfg.pullback_band))
    short_touch = (out["high"] >= out["ema20"] * (1.0 - cfg.pullback_band)) | (out["high"] >= out["ema50"] * (1.0 - cfg.pullback_band))
    out["recent_long_pullback"] = long_touch.rolling(cfg.pullback_lookback, min_periods=1).max().astype(bool)
    out["recent_short_pullback"] = short_touch.rolling(cfg.pullback_lookback, min_periods=1).max().astype(bool)

    out["long_reclaim"] = (
        (out["close"] > out["ema20"])
        & (out["close"] > out["open"])
        & (out["rsi"] > out["rsi_prev"])
        & out["rsi_prev"].between(35.0, 56.0)
        & out["rsi"].between(43.0, 66.0)
    )
    out["short_reject"] = (
        (out["close"] < out["ema20"])
        & (out["close"] < out["open"])
        & (out["rsi"] < out["rsi_prev"])
        & out["rsi_prev"].between(44.0, 66.0)
        & out["rsi"].between(34.0, 57.0)
    )

    out["d1_distance"] = out["close"] / out["d1_ema_slow"] - 1.0
    out["ema50_distance"] = out["close"] / out["ema50"] - 1.0
    out["long_not_extended"] = (out["d1_distance"] < cfg.max_d1_distance_long) & (out["ema50_distance"] < cfg.max_ema50_distance)
    out["short_not_extended"] = (out["d1_distance"] > -cfg.max_d1_distance_short) & (out["ema50_distance"] > -cfg.max_ema50_distance)

    out["long_signal"] = out["trend_long"] & out["recent_long_pullback"] & out["long_reclaim"] & out["long_not_extended"]
    out["short_signal"] = out["trend_short"] & out["recent_short_pullback"] & out["short_reject"] & out["short_not_extended"]

    # Exit on failed reclaim / regime break. Actual execution happens next open in V8 SAFE engine.
    out["long_exit_channel"] = (out["close"] < out["ema50"]) | (~out["d1_bull"].astype("boolean").fillna(False).astype(bool))
    out["short_exit_channel"] = (out["close"] > out["ema50"]) | (~out["d1_bear"].astype("boolean").fillna(False).astype(bool))

    out["signal"] = 0
    out.loc[out["long_signal"], "signal"] = 1
    out.loc[out["short_signal"], "signal"] = -1

    # Risk model: pullback engine should be steadier, not as explosive as trend rider.
    out["risk_mult"] = 0.80
    out.loc[out["adx"].between(12.0, 26.0), "risk_mult"] += 0.25
    out.loc[out["atr_pct"].between(0.0045, 0.025), "risk_mult"] += 0.20
    out.loc[out["adx"] > 32.0, "risk_mult"] -= 0.25
    out.loc[out["atr_pct"] > 0.030, "risk_mult"] -= 0.25
    out["risk_mult"] = out["risk_mult"].clip(cfg.min_risk_mult, cfg.max_risk_mult)

    out["quality_mult"] = 1.0
    out.loc[out["long_signal"] & (out["low"] <= out["ema50"] * (1.0 + cfg.pullback_band)), "quality_mult"] *= 1.20
    out.loc[out["short_signal"] & (out["high"] >= out["ema50"] * (1.0 - cfg.pullback_band)), "quality_mult"] *= 1.10
    out.loc[out["long_signal"] & (out["rsi"] > 62), "quality_mult"] *= 0.80
    out.loc[out["short_signal"] & (out["rsi"] < 38), "quality_mult"] *= 0.80
    out["quality_mult"] = out["quality_mult"].clip(0.30, 1.50)

    return out.dropna().copy()


def run_backtest(features: pd.DataFrame, cfg: PullbackConfig) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    return run_exec_backtest(features, to_exec_config(cfg))


def write_outputs(trades: list[dict[str, Any]], equity: pd.DataFrame, features: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / f"{STRATEGY_NAME}_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / f"{STRATEGY_NAME}_equity.csv")
    audit_cols = [
        "open", "high", "low", "close", "volume", "atr", "atr_pct", "adx", "rsi",
        "d1_ema_fast", "d1_ema_slow", "d1_slow_slope", "d1_bull", "d1_bear",
        "ema20", "ema50", "ema89", "recent_long_pullback", "recent_short_pullback",
        "long_reclaim", "short_reject", "long_not_extended", "short_not_extended",
        "risk_mult", "quality_mult", "long_signal", "short_signal", "signal",
        "long_exit_channel", "short_exit_channel",
    ]
    features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{STRATEGY_NAME}_signal_audit.csv")
    with (out_dir / f"{STRATEGY_NAME}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 90)
    print("ETH 1D+4H Pullback Continuation V1 Backtest Summary")
    print("=" * 90)
    for k, v in summary.items():
        print(f"{k:>34}: {v}")
    print("-" * 90)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 90 + "\n")


def print_deep_report(trades: list[dict[str, Any]], features: pd.DataFrame, cfg: PullbackConfig, out_dir: Path) -> None:
    final_capital = float(trades[-1]["capital"]) if trades else float(cfg.initial_capital)
    if features.empty:
        return
    total_days = max((features.index[-1] - features.index[0]).total_seconds() / 86400.0, 1e-9)
    print_full_report(
        trade_history=build_report_trades(trades),
        df=features,
        initial_capital=cfg.initial_capital,
        capital=final_capital,
        strategy_name=REPORT_STRATEGY_NAME,
        total_days=total_days,
        ai_enabled=False,
        symbol=cfg.symbol,
        report_dir=out_dir,
    )


def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="ETH 1D+4H Pullback Continuation V1: higher win-rate trend pullback engine.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=today)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--preset", choices=sorted(PRESETS), default="high")
    p.add_argument("--unit-risk-per-trade", type=float, default=None)
    p.add_argument("--max-total-notional-mult", type=float, default=None)
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--min-risk-mult", type=float, default=0.30)
    p.add_argument("--max-risk-mult", type=float, default=None)
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--disable-short", action="store_true")
    p.add_argument("--out-dir", default="data/reports/lf/eth_1d_4h_pullback_continuation_v1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    preset = PRESETS[args.preset]
    cfg = PullbackConfig(
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

    print(f"Loading {args.symbol} 4H: {args.start_date} -> {args.end_date}")
    base = load_data(args.symbol, args.start_date, args.end_date, "4H")
    print(f"Loaded {len(base)} rows: {base.index[0]} -> {base.index[-1]}")
    features = build_features(base, cfg)
    print(f"Feature rows: {len(features)}")
    print("Signal counts:", {
        "long": int((features.signal == 1).sum()),
        "short": int((features.signal == -1).sum()),
    })

    trades, equity = run_backtest(features, cfg)
    summary = summarize(trades, equity, cfg.initial_capital)
    summary["preset"] = args.preset
    summary["engine_role"] = "higher_win_rate_pullback_supplement"
    summary["fee_rate_per_side"] = cfg.fee_rate

    out_dir = Path(PROJECT_ROOT) / args.out_dir
    write_outputs(trades, equity, features, summary, out_dir)
    print_summary(summary, out_dir)
    print_deep_report(trades, features, cfg, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
