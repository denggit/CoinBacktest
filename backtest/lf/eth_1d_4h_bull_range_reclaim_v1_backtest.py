#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH 1D+4H Bull Range Reclaim V1
================================

定位：
    独立补充引擎实验。目标不是替代 Momentum V3 / Portfolio V5，而是尝试捕捉
    ETH 在日线不熊、4H 区间回撤后重新站回短均线的慢趋势恢复行情。

核心假设：
    Momentum Breakout 擅长强突破，但 2023 这种“上涨 + 横盘 + 假突破 + 回踩恢复”的行情
    不一定舒服。本策略只做 long，尝试补充 Bull Range Reclaim alpha。

反偷看：
    - 1D regime 全部 shift(1) 后映射到 4H；
    - 4H pullback/reclaim 只使用当前已经收盘的 bar；
    - 当前 4H close 确认，下一根 4H open 执行；
    - 执行复用 V8 SAFE 引擎：当前 close 更新的新 stop 下一根 bar 才生效；
    - 支持 warmup 数据，warmup 只用于指标，不允许开仓、不计入报告。
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

STRATEGY_NAME = "eth_1d_4h_bull_range_reclaim_v1"
REPORT_STRATEGY_NAME = "ETH_1D_4H_BullRangeReclaim_V1"

PRESETS: dict[str, dict[str, float | int]] = {
    "stable": {"unit_risk_per_trade": 0.012, "max_total_notional_mult": 5.0, "max_units": 2, "max_risk_mult": 1.6},
    "high": {"unit_risk_per_trade": 0.020, "max_total_notional_mult": 8.0, "max_units": 3, "max_risk_mult": 1.8},
    "turbo": {"unit_risk_per_trade": 0.026, "max_total_notional_mult": 9.0, "max_units": 3, "max_risk_mult": 1.9},
    "ultra": {"unit_risk_per_trade": 0.032, "max_total_notional_mult": 10.0, "max_units": 3, "max_risk_mult": 2.0},
}


@dataclass(frozen=True)
class BullRangeConfig:
    symbol: str = "ETH-USDT-SWAP"
    initial_capital: float = 1000.0
    unit_risk_per_trade: float = 0.020
    max_total_notional_mult: float = 8.0
    max_units: int = 3
    min_risk_mult: float = 0.35
    max_risk_mult: float = 1.8
    fee_rate: float = 0.00055
    slippage_pct: float = 0.0002
    enable_short: bool = False

    atr_period: int = 20
    adx_period: int = 14
    rsi_period: int = 14
    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 100

    # Bull range reclaim setup.
    pullback_lookback: int = 8
    pb_dist50: float = 0.015
    pb_dist100: float = 0.030
    d1_close_mult: float = 0.980
    d1_fast_mult: float = 0.970
    d1_slope_min: float = 0.000
    d1_max_dist: float = 0.200
    d1_min_dist: float = -0.080
    reclaim_mult: float = 1.000
    rsi_min: float = 48.0
    adx_min: float = 6.0
    adx_max: float = 16.0
    atr_min: float = 0.003
    atr_max: float = 0.050
    vol_mult: float = 0.80
    h4_max_dist50: float = 0.080
    exit_ema50_mult: float = 0.970

    # SAFE execution settings.
    initial_atr_mult: float = 2.2
    trailing_atr_mult: float = 3.5
    add_every_r: float = 1.2
    max_hold_bars: int = 90
    cooldown_bars: int = 4
    breakeven_after_r: float = 0.80
    breakeven_lock_r: float = 0.05
    lock_after_2r: float = 1.60
    lock_2r: float = 0.60
    lock_after_3r: float = 2.60
    lock_3r: float = 1.20
    no_progress_bars: int = 60
    no_progress_min_r: float = 0.50

    # 1D regime.
    d1_ema_fast: int = 20
    d1_ema_mid: int = 50
    d1_ema_slow: int = 100
    d1_slope_lookback: int = 10


def to_exec_config(cfg: BullRangeConfig) -> ExecConfig:
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
        enable_short=False,
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
        no_progress_bars=cfg.no_progress_bars,
        no_progress_min_r=cfg.no_progress_min_r,
    )


def build_daily_regime(base_4h: pd.DataFrame, cfg: BullRangeConfig) -> pd.DataFrame:
    d1 = resample_ohlcv(base_4h, "1D")
    d1["d1_close"] = d1["close"]
    d1["d1_ema_fast"] = ema(d1["close"], cfg.d1_ema_fast)
    d1["d1_ema_mid"] = ema(d1["close"], cfg.d1_ema_mid)
    d1["d1_ema_slow"] = ema(d1["close"], cfg.d1_ema_slow)
    d1["d1_mid_slope"] = d1["d1_ema_mid"] / d1["d1_ema_mid"].shift(cfg.d1_slope_lookback) - 1.0
    d1["d1_slow_slope"] = d1["d1_ema_slow"] / d1["d1_ema_slow"].shift(cfg.d1_slope_lookback) - 1.0
    d1["d1_dist_mid"] = d1["close"] / d1["d1_ema_mid"] - 1.0
    d1["d1_not_bear"] = (
        (d1["close"] > d1["d1_ema_mid"] * cfg.d1_close_mult)
        & (d1["d1_ema_fast"] > d1["d1_ema_mid"] * cfg.d1_fast_mult)
        & (d1["d1_mid_slope"] > cfg.d1_slope_min)
        & (d1["d1_dist_mid"].between(cfg.d1_min_dist, cfg.d1_max_dist))
    )

    # Shift by one completed day before mapping to 4H. Current day is never used.
    cols = [
        "d1_close", "d1_ema_fast", "d1_ema_mid", "d1_ema_slow",
        "d1_mid_slope", "d1_slow_slope", "d1_dist_mid", "d1_not_bear",
    ]
    return pd.DataFrame({c: d1[c].shift(1) for c in cols}, index=d1.index)


def build_features(base_4h: pd.DataFrame, cfg: BullRangeConfig) -> pd.DataFrame:
    out = base_4h.copy()
    out["ema20"] = ema(out["close"], cfg.ema_fast)
    out["ema50"] = ema(out["close"], cfg.ema_mid)
    out["ema100"] = ema(out["close"], cfg.ema_slow)
    out["atr"] = atr(out, cfg.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]
    out["adx"] = adx(out, cfg.adx_period)
    out["rsi"] = rsi(out["close"], cfg.rsi_period)
    out["volume_med"] = out["volume"].rolling(30, min_periods=10).median().shift(1)

    d1 = build_daily_regime(base_4h, cfg)
    out = out.join(d1.reindex(out.index, method="ffill"))

    out["prev_close_below_ema20"] = out["close"].shift(1) < out["ema20"].shift(1)
    out["pb_min_dist50"] = (out["low"] / out["ema50"] - 1.0).rolling(cfg.pullback_lookback, min_periods=1).min().shift(1)
    out["pb_min_dist100"] = (out["low"] / out["ema100"] - 1.0).rolling(cfg.pullback_lookback, min_periods=1).min().shift(1)
    out["recent_pullback"] = (
        (out["pb_min_dist50"] < cfg.pb_dist50)
        | (out["pb_min_dist100"] < cfg.pb_dist100)
        | out["prev_close_below_ema20"]
    )

    out["reclaim"] = (
        (out["close"] > out["ema20"] * cfg.reclaim_mult)
        & (out["close"] > out["open"])
        & (out["close"] > out["close"].shift(1))
        & (out["rsi"] > cfg.rsi_min)
    )
    out["range_ok"] = out["adx"].between(cfg.adx_min, cfg.adx_max) & out["atr_pct"].between(cfg.atr_min, cfg.atr_max)
    out["volume_ok"] = out["volume"] > out["volume_med"] * cfg.vol_mult
    out["h4_dist50"] = out["close"] / out["ema50"] - 1.0
    out["not_extended"] = out["h4_dist50"] < cfg.h4_max_dist50
    out["daily_ok"] = out["d1_not_bear"].astype("boolean").fillna(False).astype(bool)

    out["long_signal"] = out["daily_ok"] & out["recent_pullback"] & out["reclaim"] & out["range_ok"] & out["volume_ok"] & out["not_extended"]
    out["short_signal"] = False
    out["signal"] = 0
    out.loc[out["long_signal"], "signal"] = 1

    # Exit when range reclaim fails. Actual execution uses next open in V8 SAFE engine.
    out["exit_low"] = out["low"].rolling(16, min_periods=4).min().shift(1)
    out["long_exit_channel"] = (
        (out["close"] < out["ema50"] * cfg.exit_ema50_mult)
        | (out["close"] < out["exit_low"])
    )
    out["short_exit_channel"] = False

    out["risk_mult"] = 1.0
    out.loc[out["adx"].between(10.0, 18.0), "risk_mult"] += 0.15
    out.loc[out["atr_pct"].between(0.004, 0.030), "risk_mult"] += 0.15
    out.loc[out["atr_pct"] > 0.040, "risk_mult"] -= 0.25
    out["risk_mult"] = out["risk_mult"].clip(cfg.min_risk_mult, cfg.max_risk_mult)

    out["quality_mult"] = 0.0
    out.loc[out["long_signal"], "quality_mult"] = 1.0
    out.loc[out["long_signal"] & (out["pb_min_dist50"] < 0.005), "quality_mult"] *= 0.80
    out.loc[out["long_signal"] & (out["h4_dist50"] > cfg.h4_max_dist50 * 0.75), "quality_mult"] *= 0.75
    out["quality_mult"] = out["quality_mult"].clip(0.25, 1.20)

    return out.dropna().copy()


def run_backtest(features: pd.DataFrame, cfg: BullRangeConfig) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    return run_exec_backtest(features, to_exec_config(cfg))


def yearly_returns_from_trades(trades: list[dict[str, Any]], initial_capital: float) -> dict[str, float]:
    if not trades:
        return {}
    tdf = pd.DataFrame(trades).copy()
    tdf["exit_time"] = pd.to_datetime(tdf["exit_time"])
    out: dict[str, float] = {}
    last_capital = initial_capital
    for year, grp in tdf.groupby(tdf["exit_time"].dt.year):
        start_cap = last_capital
        end_cap = float(grp.iloc[-1]["capital"])
        out[str(int(year))] = round((end_cap / start_cap - 1.0) * 100.0, 4) if start_cap > 0 else 0.0
        last_capital = end_cap
    return out


def write_outputs(trades: list[dict[str, Any]], equity: pd.DataFrame, features: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / f"{STRATEGY_NAME}_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / f"{STRATEGY_NAME}_equity.csv")
    audit_cols = [
        "open", "high", "low", "close", "volume", "atr", "atr_pct", "adx", "rsi",
        "ema20", "ema50", "ema100", "d1_ema_fast", "d1_ema_mid", "d1_ema_slow", "d1_mid_slope", "d1_dist_mid", "d1_not_bear",
        "pb_min_dist50", "pb_min_dist100", "recent_pullback", "reclaim", "range_ok", "volume_ok", "not_extended",
        "risk_mult", "quality_mult", "exit_low", "long_signal", "short_signal", "signal", "long_exit_channel",
    ]
    features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{STRATEGY_NAME}_signal_audit.csv")
    with (out_dir / f"{STRATEGY_NAME}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 94)
    print("ETH 1D+4H Bull Range Reclaim V1 Backtest Summary")
    print("=" * 94)
    for k, v in summary.items():
        print(f"{k:>36}: {v}")
    print("-" * 94)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 94 + "\n")


def print_deep_report(trades: list[dict[str, Any]], features: pd.DataFrame, cfg: BullRangeConfig, out_dir: Path) -> None:
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
    p = argparse.ArgumentParser(description="ETH 1D+4H Bull Range Reclaim V1: long-only bull range reclaim supplement engine.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01", help="交易开始日期。warmup 数据只用于计算指标，不允许开仓。")
    p.add_argument("--end-date", default=today)
    p.add_argument("--warmup-start-date", default=None, help="指标预热数据开始日期，例如 2022-01-01；交易仍从 --start-date 开始。")
    p.add_argument("--warmup-days", type=int, default=365, help="未传 --warmup-start-date 时默认向前加载多少天用于指标预热。设为 0 可关闭。")
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--preset", choices=sorted(PRESETS), default="high")
    p.add_argument("--unit-risk-per-trade", type=float, default=None)
    p.add_argument("--max-total-notional-mult", type=float, default=None)
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--min-risk-mult", type=float, default=0.35)
    p.add_argument("--max-risk-mult", type=float, default=None)
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    preset = PRESETS[args.preset]
    cfg = BullRangeConfig(
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        unit_risk_per_trade=float(args.unit_risk_per_trade if args.unit_risk_per_trade is not None else preset["unit_risk_per_trade"]),
        max_total_notional_mult=float(args.max_total_notional_mult if args.max_total_notional_mult is not None else preset["max_total_notional_mult"]),
        max_units=int(args.max_units if args.max_units is not None else preset["max_units"]),
        min_risk_mult=args.min_risk_mult,
        max_risk_mult=float(args.max_risk_mult if args.max_risk_mult is not None else preset["max_risk_mult"]),
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
    )
    out_dir = Path(args.out_dir) if args.out_dir else Path(PROJECT_ROOT) / "data/reports/lf" / STRATEGY_NAME / args.preset

    trade_start = pd.Timestamp(args.start_date)
    if args.warmup_start_date:
        load_start = pd.Timestamp(args.warmup_start_date)
    elif args.warmup_days and args.warmup_days > 0:
        load_start = trade_start - pd.Timedelta(days=int(args.warmup_days))
    else:
        load_start = trade_start
    load_start_str = load_start.strftime("%Y-%m-%d")

    print(f"Loading {args.symbol} 4H for warmup: {load_start_str} -> {args.end_date}; trade_start={args.start_date}")
    base = load_data(args.symbol, load_start_str, args.end_date, "4H")
    print(f"Loaded {len(base)} rows: {base.index[0]} -> {base.index[-1]}")
    features = build_features(base, cfg)
    before_slice_rows = len(features)
    features = features.loc[trade_start: pd.Timestamp(args.end_date)].copy()
    print(f"Feature rows after warmup slice: {len(features)} / {before_slice_rows}; first tradeable bar={features.index[0] if not features.empty else 'NA'}")
    print("Signal counts:", {"long": int((features.signal == 1).sum()), "short": int((features.signal == -1).sum())})

    trades, equity = run_backtest(features, cfg)
    summary = summarize(trades, equity, cfg.initial_capital)
    summary["preset"] = args.preset
    summary["engine_role"] = "bull_range_reclaim_supplement_candidate"
    summary["warmup_start_date"] = load_start_str
    summary["trade_start_date"] = args.start_date
    summary["warmup_days"] = int(args.warmup_days or 0)
    summary["fee_rate_per_side"] = cfg.fee_rate
    summary["yearly_return_pct"] = yearly_returns_from_trades(trades, cfg.initial_capital)

    write_outputs(trades, equity, features, summary, out_dir)
    print_summary(summary, out_dir)
    print_deep_report(trades, features, cfg, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
