#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH Timeframe Probe Grid V1
===========================

独立测试版本，不并入 LF portfolio。

目的：一次性测试不同低/中级别 timeframe（15m/30m/1h/2h/4h）是否能比 4H LF portfolio 更分散地抓小趋势。

原则：
    - 不修改 V6/V7 portfolio 文件。
    - 不把结果并入 portfolio，除非后续证明有稳定正期望。
    - 基础数据只加载本地 15m OHLCV，一次加载后重采样成目标 timeframe。
    - 1D / 4H regime 全部 shift(1) 后映射到目标 timeframe，避免未来函数。
    - 目标 timeframe 的 Donchian entry/exit 全部 rolling(...).shift(1)。
    - 当前 bar close 确认信号，下一根 bar open 执行。
    - 当前 bar close 更新的新 stop 下一根 bar 才生效，由 V8 SAFE execution engine 负责。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
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

STRATEGY_NAME = "eth_timeframe_probe_grid_v1"
REPORT_STRATEGY_NAME = "ETH_TimeframeProbeGrid_V1"

PRESETS: dict[str, dict[str, float | int]] = {
    "stable": {"unit_risk_per_trade": 0.0030, "max_total_notional_mult": 5.0, "max_units": 3, "max_risk_mult": 1.5},
    "high": {"unit_risk_per_trade": 0.0040, "max_total_notional_mult": 6.0, "max_units": 3, "max_risk_mult": 1.8},
    "turbo": {"unit_risk_per_trade": 0.0060, "max_total_notional_mult": 7.0, "max_units": 4, "max_risk_mult": 2.0},
}

TF_TO_RULE = {
    "15m": None,
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
}


def normalize_timeframe(tf: str) -> str:
    t = str(tf).strip().lower().replace(" ", "")
    aliases = {
        "15min": "15m",
        "30min": "30m",
        "1hour": "1h",
        "1hr": "1h",
        "2hour": "2h",
        "2hr": "2h",
        "4hour": "4h",
        "4hr": "4h",
        "1H".lower(): "1h",
        "2H".lower(): "2h",
        "4H".lower(): "4h",
    }
    t = aliases.get(t, t)
    if t not in TF_TO_RULE:
        raise ValueError(f"Unsupported timeframe: {tf}. Supported: {','.join(TF_TO_RULE)}")
    return t


def timeframe_hours(tf: str) -> float:
    tf = normalize_timeframe(tf)
    if tf.endswith("m"):
        return float(tf[:-1]) / 60.0
    if tf.endswith("h"):
        return float(tf[:-1])
    raise ValueError(f"Unsupported timeframe: {tf}")


def bars_for(hours: float, tf_hours: float, min_bars: int = 2) -> int:
    return max(min_bars, int(math.ceil(float(hours) / max(tf_hours, 1e-9))))


@dataclass(frozen=True)
class ProbeConfig:
    symbol: str = "ETH-USDT-SWAP"
    timeframe: str = "1h"
    timeframe_hours: float = 1.0
    initial_capital: float = 1000.0
    unit_risk_per_trade: float = 0.0030
    max_total_notional_mult: float = 5.0
    max_units: int = 3
    min_risk_mult: float = 0.35
    max_risk_mult: float = 1.5
    fee_rate: float = 0.00055
    slippage_pct: float = 0.0002
    enable_short: bool = True

    entry_lookback: int = 24
    exit_lookback: int = 16
    atr_period: int = 24
    adx_period: int = 24
    volume_window: int = 48
    volume_mult: float = 1.05
    min_adx_long: float = 10.0
    min_adx_short: float = 12.0
    max_adx: float = 55.0
    min_atr_pct: float = 0.0008
    max_atr_pct: float = 0.025

    ema_fast: int = 24
    ema_mid: int = 80
    ema_slow: int = 200

    h4_ema_fast: int = 20
    h4_ema_mid: int = 50
    h4_slope_lookback: int = 12
    d1_ema_fast: int = 8
    d1_ema_slow: int = 30
    d1_slope_lookback: int = 10
    bull_d1_slope_min: float = -0.030
    bear_d1_slope_max: float = -0.003

    initial_atr_mult: float = 4.0
    trailing_atr_mult: float = 8.0
    add_every_r: float = 1.5
    max_hold_bars: int = 168
    cooldown_bars: int = 4
    breakeven_after_r: float = 1.2
    breakeven_lock_r: float = 0.05
    lock_after_2r: float = 2.0
    lock_2r: float = 0.50
    lock_after_3r: float = 3.0
    lock_3r: float = 1.20


def make_config(args: argparse.Namespace, timeframe: str, preset_name: str) -> ProbeConfig:
    tf = normalize_timeframe(timeframe)
    tf_h = timeframe_hours(tf)
    preset = PRESETS[preset_name]
    return ProbeConfig(
        symbol=args.symbol,
        timeframe=tf,
        timeframe_hours=tf_h,
        initial_capital=args.initial_capital,
        unit_risk_per_trade=float(args.unit_risk_per_trade if args.unit_risk_per_trade is not None else preset["unit_risk_per_trade"]),
        max_total_notional_mult=float(args.max_total_notional_mult if args.max_total_notional_mult is not None else preset["max_total_notional_mult"]),
        max_units=int(args.max_units if args.max_units is not None else preset["max_units"]),
        min_risk_mult=args.min_risk_mult,
        max_risk_mult=float(args.max_risk_mult if args.max_risk_mult is not None else preset["max_risk_mult"]),
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
        enable_short=not args.disable_short,
        # Time-equivalent windows, with minimum bars to avoid extremely unstable 2H/4H estimates.
        entry_lookback=bars_for(args.entry_lookback_hours, tf_h, min_bars=12),
        exit_lookback=bars_for(args.exit_lookback_hours, tf_h, min_bars=8),
        atr_period=bars_for(args.atr_hours, tf_h, min_bars=20),
        adx_period=bars_for(args.adx_hours, tf_h, min_bars=20),
        volume_window=bars_for(args.volume_hours, tf_h, min_bars=30),
        ema_fast=bars_for(args.ema_fast_hours, tf_h, min_bars=20),
        ema_mid=bars_for(args.ema_mid_hours, tf_h, min_bars=50),
        ema_slow=bars_for(args.ema_slow_hours, tf_h, min_bars=100),
        max_hold_bars=bars_for(args.max_hold_hours, tf_h, min_bars=4),
        cooldown_bars=bars_for(args.cooldown_hours, tf_h, min_bars=1),
    )


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


def resample_base_15m(base_15m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    tf = normalize_timeframe(timeframe)
    rule = TF_TO_RULE[tf]
    if rule is None:
        return base_15m.copy()
    out = resample_ohlcv(base_15m, rule)
    return out.dropna(subset=["open", "high", "low", "close", "volume"])


def shifted_d1_regime(base_tf: pd.DataFrame, cfg: ProbeConfig) -> pd.DataFrame:
    d1 = resample_ohlcv(base_tf, "1d")
    d1["d1_close"] = d1["close"]
    d1["d1_ema_fast"] = ema(d1["close"], cfg.d1_ema_fast)
    d1["d1_ema_slow"] = ema(d1["close"], cfg.d1_ema_slow)
    d1["d1_slow_slope"] = d1["d1_ema_slow"] / d1["d1_ema_slow"].shift(cfg.d1_slope_lookback) - 1.0
    d1["d1_bull"] = (d1["close"] > d1["d1_ema_slow"]) & (d1["d1_ema_fast"] > d1["d1_ema_slow"] * 0.995) & (d1["d1_slow_slope"] > cfg.bull_d1_slope_min)
    d1["d1_bear"] = (d1["close"] < d1["d1_ema_slow"]) & (d1["d1_ema_fast"] < d1["d1_ema_slow"]) & (d1["d1_slow_slope"] < cfg.bear_d1_slope_max)
    return d1[["d1_close", "d1_ema_fast", "d1_ema_slow", "d1_slow_slope", "d1_bull", "d1_bear"]].shift(1)


def shifted_h4_regime(base_tf: pd.DataFrame, cfg: ProbeConfig) -> pd.DataFrame:
    h4 = resample_ohlcv(base_tf, "4h")
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
    out.loc[out["long_signal"] & (out["adx"] > 35), "quality_mult"] = 0.70
    out.loc[out["short_signal"] & (out["adx"] > 38), "quality_mult"] = 0.80
    return out


def write_outputs(trades: list[dict[str, Any]], equity: pd.DataFrame, features: pd.DataFrame, summary: dict[str, Any], out_dir: Path, name_prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / f"{name_prefix}_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / f"{name_prefix}_equity.csv")
    audit_cols = [
        "open", "high", "low", "close", "volume", "atr", "atr_pct", "adx",
        "ema_fast", "ema_mid", "ema_slow", "d1_bull", "d1_bear", "h4_bull", "h4_bear",
        "entry_high", "entry_low", "exit_low", "exit_high",
        "risk_mult", "quality_mult", "long_signal", "short_signal", "signal",
        "long_exit_channel", "short_exit_channel",
    ]
    features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{name_prefix}_signal_audit.csv")
    with (out_dir / f"{name_prefix}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 96)
    print(f"{REPORT_STRATEGY_NAME} Summary | timeframe={summary.get('timeframe')} preset={summary.get('preset')}")
    print("=" * 96)
    for k, v in summary.items():
        print(f"{k:>34}: {v}")
    print("-" * 96)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 96 + "\n")


def print_deep_report(trades: list[dict[str, Any]], features: pd.DataFrame, cfg: ProbeConfig, out_dir: Path) -> None:
    if not trades or features.empty:
        return
    total_days = max((features.index[-1] - features.index[0]).total_seconds() / 86400.0, 1e-9)
    print_full_report(
        trade_history=build_report_trades(trades),
        df=features,
        initial_capital=cfg.initial_capital,
        capital=float(pd.DataFrame(trades).iloc[-1]["capital"]),
        strategy_name=f"{REPORT_STRATEGY_NAME}_{cfg.timeframe}",
        total_days=total_days,
        ai_enabled=False,
        symbol=cfg.symbol,
        report_dir=out_dir,
    )


def run_one(base_15m: pd.DataFrame, args: argparse.Namespace, timeframe: str, preset: str, root_out_dir: Path, trade_start: pd.Timestamp) -> dict[str, Any]:
    tf = normalize_timeframe(timeframe)
    cfg = make_config(args, tf, preset)
    exec_cfg = to_exec_config(cfg)
    out_dir = root_out_dir / tf / preset
    name_prefix = f"{STRATEGY_NAME}_{tf}_{preset}"

    print(f"\n[probe] timeframe={tf} preset={preset} | resampling from 15m...")
    base_tf = resample_base_15m(base_15m, tf)
    print(f"[probe] {tf} rows={len(base_tf)} | {base_tf.index[0] if len(base_tf) else 'NA'} -> {base_tf.index[-1] if len(base_tf) else 'NA'}")
    features = build_features(base_tf, cfg)
    before_slice_rows = len(features)
    features = features.loc[trade_start: pd.Timestamp(args.end_date)].copy()
    if features.empty:
        raise RuntimeError(f"No features after warmup slice for timeframe={tf}, preset={preset}")
    sig_counts = {
        "long_signal": int(features.long_signal.sum()),
        "short_signal": int(features.short_signal.sum()),
        "signal_long": int((features.signal == 1).sum()),
        "signal_short": int((features.signal == -1).sum()),
    }
    print(f"[probe] {tf}/{preset} feature rows after warmup slice: {len(features)} / {before_slice_rows}; signals={sig_counts}")

    trades, equity = run_exec_backtest(features, exec_cfg)
    summary = summarize(trades, equity, exec_cfg.initial_capital)
    summary.update({
        "preset": preset,
        "timeframe": tf,
        "timeframe_hours": cfg.timeframe_hours,
        "standalone_probe": True,
        "not_portfolio_component": True,
        "fee_rate_per_side": args.fee_rate,
        "warmup_start_date": args._load_start_str,
        "trade_start_date": args.start_date,
        "base_data_timeframe": "15m",
        "feature_rows": int(len(features)),
        "pre_slice_feature_rows": int(before_slice_rows),
        "long_signal_count": sig_counts["signal_long"],
        "short_signal_count": sig_counts["signal_short"],
        "entry_lookback": cfg.entry_lookback,
        "exit_lookback": cfg.exit_lookback,
        "atr_period": cfg.atr_period,
        "ema_fast": cfg.ema_fast,
        "ema_mid": cfg.ema_mid,
        "ema_slow": cfg.ema_slow,
        "max_hold_bars": cfg.max_hold_bars,
        "config": asdict(cfg),
    })
    write_outputs(trades, equity, features, summary, out_dir, name_prefix)
    print_summary(summary, out_dir)
    if not args.skip_deep_report:
        print_deep_report(trades, features, cfg, out_dir)
    return {k: v for k, v in summary.items() if k != "config"}


def parse_csv_arg(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="ETH standalone multi-timeframe probe grid. Not part of LF portfolio.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=today)
    p.add_argument("--warmup-start-date", default=None)
    p.add_argument("--warmup-days", type=int, default=365)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--timeframes", default="1h,2h", help="Comma-separated: 15m,30m,1h,2h,4h")
    p.add_argument("--presets", default="stable,high", help="Comma-separated presets: stable,high,turbo")
    p.add_argument("--unit-risk-per-trade", type=float, default=None)
    p.add_argument("--max-total-notional-mult", type=float, default=None)
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--min-risk-mult", type=float, default=0.35)
    p.add_argument("--max-risk-mult", type=float, default=None)
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--disable-short", action="store_true")
    p.add_argument("--entry-lookback-hours", type=float, default=24.0)
    p.add_argument("--exit-lookback-hours", type=float, default=16.0)
    p.add_argument("--atr-hours", type=float, default=24.0)
    p.add_argument("--adx-hours", type=float, default=24.0)
    p.add_argument("--volume-hours", type=float, default=48.0)
    p.add_argument("--ema-fast-hours", type=float, default=24.0)
    p.add_argument("--ema-mid-hours", type=float, default=80.0)
    p.add_argument("--ema-slow-hours", type=float, default=200.0)
    p.add_argument("--max-hold-hours", type=float, default=168.0)
    p.add_argument("--cooldown-hours", type=float, default=4.0)
    p.add_argument("--skip-deep-report", action="store_true", help="Still writes CSV/JSON summaries, but skips print_full_report for faster grid runs.")
    p.add_argument("--continue-on-error", action="store_true", help="Continue remaining timeframe/preset runs if one run fails.")
    p.add_argument("--out-dir", default="data/reports/mf/eth_timeframe_probe_grid_v1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    timeframes = [normalize_timeframe(x) for x in parse_csv_arg(args.timeframes)]
    presets = parse_csv_arg(args.presets)
    for preset in presets:
        if preset not in PRESETS:
            raise ValueError(f"Unsupported preset: {preset}. Supported: {','.join(PRESETS)}")

    trade_start = pd.Timestamp(args.start_date)
    if args.warmup_start_date:
        load_start = pd.Timestamp(args.warmup_start_date)
    elif args.warmup_days and args.warmup_days > 0:
        load_start = trade_start - pd.Timedelta(days=int(args.warmup_days))
    else:
        load_start = trade_start
    args._load_start_str = load_start.strftime("%Y-%m-%d")

    root_out_dir = Path(args.out_dir)
    root_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.symbol} local 15m base for warmup: {args._load_start_str} -> {args.end_date}; trade_start={args.start_date}")
    base_15m = load_data(args.symbol, args._load_start_str, args.end_date, "15m")
    if base_15m.empty:
        raise RuntimeError("No 15m data loaded. Please make sure local DB has 15m OHLCV.")
    print(f"Loaded 15m base rows={len(base_15m)}: {base_15m.index[0]} -> {base_15m.index[-1]}")

    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for tf in timeframes:
        for preset in presets:
            try:
                summaries.append(run_one(base_15m, args, tf, preset, root_out_dir, trade_start))
            except Exception as exc:
                print(f"[ERROR] timeframe={tf} preset={preset}: {exc}")
                errors.append({"timeframe": tf, "preset": preset, "error": str(exc)})
                if not args.continue_on_error:
                    raise

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        preferred = [
            "timeframe", "preset", "total_trades", "long_trades", "short_trades", "final_capital",
            "total_return_pct", "profit_factor", "win_rate", "max_drawdown_pct", "avg_holding_hours",
            "total_fees", "long_signal_count", "short_signal_count", "feature_rows",
        ]
        cols = [c for c in preferred if c in summary_df.columns] + [c for c in summary_df.columns if c not in preferred]
        summary_df = summary_df[cols]
        summary_df.to_csv(root_out_dir / f"{STRATEGY_NAME}_aggregate_summary.csv", index=False)
        print("\n" + "=" * 96)
        print("Aggregate summary")
        print("=" * 96)
        print(summary_df[[c for c in preferred if c in summary_df.columns]].to_string(index=False))
        print(f"\nSaved aggregate summary: {(root_out_dir / f'{STRATEGY_NAME}_aggregate_summary.csv').resolve()}")
    if errors:
        pd.DataFrame(errors).to_csv(root_out_dir / f"{STRATEGY_NAME}_errors.csv", index=False)
    with (root_out_dir / f"{STRATEGY_NAME}_config.json").open("w", encoding="utf-8") as f:
        json.dump({
            "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
            "load_start": args._load_start_str,
            "timeframes": timeframes,
            "presets": presets,
        }, f, ensure_ascii=False, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
