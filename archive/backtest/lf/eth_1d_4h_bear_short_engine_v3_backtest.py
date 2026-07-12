#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH 1D+4H Bear Short Engine V3
==============================

放置位置建议：backtest/lf/eth_1d_4h_bear_short_engine_v3_backtest.py
运行示例：
    python backtest/lf/eth_1d_4h_bear_short_engine_v3_backtest.py --start-date 2023-01-01 --end-date 2026-06-15 --preset high

定位：
    第二个独立引擎的第三版。V3 在 V2 基础上进一步收紧反弹追空和深跌末端追空，目标是把非熊市年份亏损压到接近 0，同时保留 2026 熊市爆发力。

反偷看：
    - 日线、周线 regime 全部 shift(1) 后才映射到 4H；
    - 4H Donchian entry/exit 来自 V8 build_features，内部使用 rolling(...).shift(1)；
    - 当前 4H 收盘确认信号，下一根 4H open 执行；
    - 当前 bar close 更新的新 stop 下一根 bar 才生效；
    - 不允许同一根 K 线内用收盘后才知道的新 stop 触发。
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
from src.backtest_common.indicators import ema, resample_ohlcv  # noqa: E402
from src.backtest_common.reporting import build_report_trades  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402

from backtest.lf.eth_1d_4h_trend_rider_v8_position_lock_backtest import (  # noqa: E402
    StrategyConfig as ExecConfig,
    build_features as build_v8_features,
    run_backtest as run_exec_backtest,
    summarize,
)

STRATEGY_NAME = "eth_1d_4h_bear_short_engine_v3"
REPORT_STRATEGY_NAME = "ETH_1D_4H_BearShortEngine_V3"

PRESETS: dict[str, dict[str, float | int | str]] = {
    # 保守：超级熊市 breakdown 才做，低交易数、低拖累。
    "scout": {"unit_risk_per_trade": 0.022, "max_total_notional_mult": 8.0, "max_units": 4, "max_risk_mult": 2.0, "style": "breakdown"},
    # 默认：V2 bear permission，降低深跌末端追空和震荡空头。
    "stable": {"unit_risk_per_trade": 0.018, "max_total_notional_mult": 11.0, "max_units": 5, "max_risk_mult": 2.3, "style": "bear_permission_v3"},
    "high": {"unit_risk_per_trade": 0.022, "max_total_notional_mult": 11.0, "max_units": 5, "max_risk_mult": 2.3, "style": "bear_permission_v3"},
    "turbo": {"unit_risk_per_trade": 0.030, "max_total_notional_mult": 11.0, "max_units": 5, "max_risk_mult": 2.3, "style": "bear_permission_v3"},
    "ultra": {"unit_risk_per_trade": 0.040, "max_total_notional_mult": 11.0, "max_units": 5, "max_risk_mult": 2.3, "style": "bear_permission_v3"},
    "max": {"unit_risk_per_trade": 0.055, "max_total_notional_mult": 11.0, "max_units": 5, "max_risk_mult": 2.3, "style": "bear_permission_v3"},
}


@dataclass(frozen=True)
class BearConfig:
    symbol: str = "ETH-USDT-SWAP"
    initial_capital: float = 1000.0
    unit_risk_per_trade: float = 0.022
    max_total_notional_mult: float = 11.0
    max_units: int = 5
    min_risk_mult: float = 0.25
    max_risk_mult: float = 2.3
    fee_rate: float = 0.00055
    slippage_pct: float = 0.0002
    style: str = "crash_continuation"

    # Short-only bear engine filter.
    d1_ema_fast: int = 20
    d1_ema_mid: int = 50
    d1_ema_slow: int = 100
    d1_ema_major: int = 200
    d1_slope_lookback: int = 10
    w_ema_fast: int = 10
    w_ema_mid: int = 20
    w_ema_slow: int = 40
    w_slope_lookback: int = 4

    # Execution inherited from V8 safe engine.
    initial_atr_mult: float = 2.5
    trailing_atr_mult: float = 4.5
    add_every_r: float = 1.0
    max_hold_bars: int = 360
    cooldown_bars: int = 8


def to_exec_config(cfg: BearConfig) -> ExecConfig:
    """Use V8's safe execution engine with short-only settings."""
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
        enable_short=True,
        initial_atr_mult=cfg.initial_atr_mult,
        trailing_atr_mult=cfg.trailing_atr_mult,
        add_every_r=cfg.add_every_r,
        max_hold_bars=cfg.max_hold_bars,
        cooldown_bars=cfg.cooldown_bars,
    )


def add_shifted_higher_tf_features(base_4h: pd.DataFrame, features: pd.DataFrame, cfg: BearConfig) -> pd.DataFrame:
    out = features.copy()

    d1 = resample_ohlcv(base_4h, "1D")
    d1["d1_close"] = d1["close"]
    d1["d1_ema20"] = ema(d1["close"], cfg.d1_ema_fast)
    d1["d1_ema50"] = ema(d1["close"], cfg.d1_ema_mid)
    d1["d1_ema100"] = ema(d1["close"], cfg.d1_ema_slow)
    d1["d1_ema200"] = ema(d1["close"], cfg.d1_ema_major)
    d1["d1_ema50_slope"] = d1["d1_ema50"] / d1["d1_ema50"].shift(cfg.d1_slope_lookback) - 1.0
    d1["d1_ema100_slope"] = d1["d1_ema100"] / d1["d1_ema100"].shift(cfg.d1_slope_lookback) - 1.0
    d1_cols = ["d1_close", "d1_ema20", "d1_ema50", "d1_ema100", "d1_ema200", "d1_ema50_slope", "d1_ema100_slope"]
    d1_available = pd.DataFrame({col: d1[col].shift(1) for col in d1_cols}, index=d1.index)
    out = out.join(d1_available.reindex(out.index, method="ffill"))

    # Weekly features are only used after shift(1), so the current incomplete week is never used.
    wk = resample_ohlcv(base_4h, "1W")
    wk["w_close"] = wk["close"]
    wk["w_ema10"] = ema(wk["close"], cfg.w_ema_fast)
    wk["w_ema20"] = ema(wk["close"], cfg.w_ema_mid)
    wk["w_ema40"] = ema(wk["close"], cfg.w_ema_slow)
    wk["w_ema20_slope"] = wk["w_ema20"] / wk["w_ema20"].shift(cfg.w_slope_lookback) - 1.0
    w_cols = ["w_close", "w_ema10", "w_ema20", "w_ema40", "w_ema20_slope"]
    wk_available = pd.DataFrame({col: wk[col].shift(1) for col in w_cols}, index=wk.index)
    out = out.join(wk_available.reindex(out.index, method="ffill"))

    # Recent 4H returns are based only on historical closed bars. Used to avoid chasing after a rebound/noise move.
    out["ret_6"] = out["close"] / out["close"].shift(6) - 1.0
    out["ret_12"] = out["close"] / out["close"].shift(12) - 1.0
    out["ret_30"] = out["close"] / out["close"].shift(30) - 1.0
    return out


def build_bear_features(base_4h: pd.DataFrame, cfg: BearConfig) -> pd.DataFrame:
    exec_cfg = to_exec_config(cfg)
    out = build_v8_features(base_4h, exec_cfg)
    out = add_shifted_higher_tf_features(base_4h, out, cfg)

    out["long_signal"] = False

    weekly_bear = (
        (out["w_close"] < out["w_ema20"])
        & (out["w_ema20_slope"] < 0)
    )
    # V3 bear permission:
    # 1) major daily bear is required;
    # 2) d1_ema100_slope cannot be too negative, because extremely steep bear slopes often mean late-stage downside / rebound risk;
    # 3) ret_12 < 0.5% avoids shorting right after a 4H rebound squeeze;
    # 4) price cannot be more than 11% below daily EMA100, reducing deep-late short entries.
    d1_major_bear = (
        (out["d1_close"] < out["d1_ema100"])
        & (out["d1_ema50_slope"] < -0.008)
    )
    bear_permission_v2 = (
        d1_major_bear
        & (out["d1_ema100_slope"] > -0.025)
        & (out["ret_12"] < 0.005)
        & ((out["close"] / out["d1_ema100"] - 1.0) > -0.110)
    )
    four_h_bear = (
        (out["ema20"] < out["ema50"])
        & (out["close"] < out["ema20"])
        & (out["close"] < out["open"])
        & out["adx"].between(12.0, 32.0)
        & out["atr_pct"].between(0.006, 0.030)
        & ((out["close"] / out["d1_ema100"] - 1.0).between(-0.18, 0.02))
    )

    breakdown = (
        weekly_bear
        & (out["d1_close"] < out["d1_ema100"])
        & (out["ema20"] < out["ema50"])
        & (out["close"] < out["entry_low"])
        & (out["close"] < out["open"])
        & out["adx"].between(10.0, 30.0)
        & out["atr_pct"].between(0.004, 0.032)
    )
    crash_continuation = d1_major_bear & four_h_bear
    permission_continuation = bear_permission_v2 & four_h_bear

    if cfg.style == "breakdown":
        short_signal = breakdown
    elif cfg.style == "crash_continuation":
        short_signal = crash_continuation
    elif cfg.style in {"bear_permission_v2", "bear_permission_v3"}:
        short_signal = permission_continuation
    elif cfg.style == "combo":
        short_signal = breakdown | permission_continuation
    else:
        raise ValueError(f"Unsupported style: {cfg.style}")

    out["weekly_bear"] = weekly_bear.fillna(False).astype(bool)
    out["bear_permission_v3"] = bear_permission_v2.fillna(False).astype(bool)
    out["bear_permission_v2"] = out["bear_permission_v3"]

    out["short_signal"] = short_signal.fillna(False).astype(bool)
    out["signal"] = 0
    out.loc[out["short_signal"], "signal"] = -1

    # Short-specific fast exit. This cuts some chop, but still uses next-open exit in the execution engine.
    out["short_exit_channel"] = (
        (out["close"] > out["ema50"])
        | ((out["close"] > out["ema89"]) & (out["ema20"] > out["ema50"]))
        | (out["close"] > out["exit_high"])
    )
    out["long_exit_channel"] = False

    # Risk model: short engine should not fight choppy rebounds. It sizes up only in controlled bear continuation.
    out["risk_mult"] = 0.60
    out.loc[out["adx"].between(14.0, 28.0), "risk_mult"] += 0.30
    out.loc[out["d1_ema100_slope"] < -0.006, "risk_mult"] += 0.30
    out.loc[out["atr_pct"].between(0.006, 0.026), "risk_mult"] += 0.20
    out.loc[out["atr_pct"] > 0.030, "risk_mult"] -= 0.35
    out["risk_mult"] = out["risk_mult"].clip(cfg.min_risk_mult, cfg.max_risk_mult)

    out["quality_mult"] = 1.0
    trend_cont = (out["close"] < out["ema20"]) & (out["ema20"] < out["ema50"])
    out.loc[trend_cont, "quality_mult"] *= 1.35
    out.loc[out["close"] < out["entry_low"], "quality_mult"] *= 0.75  # pure breakdown can be late; keep but resize down.
    out.loc[out["adx"] > 32.0, "quality_mult"] *= 0.60
    out.loc[out["atr_pct"] > 0.025, "quality_mult"] *= 0.70
    out["quality_mult"] = out["quality_mult"].clip(0.20, 1.70)
    return out.dropna().copy()


def run_backtest(features: pd.DataFrame, cfg: BearConfig) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    return run_exec_backtest(features, to_exec_config(cfg))


def write_outputs(trades: list[dict[str, Any]], equity: pd.DataFrame, features: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / f"{STRATEGY_NAME}_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / f"{STRATEGY_NAME}_equity.csv")
    audit_cols = [
        "open", "high", "low", "close", "volume", "atr", "atr_pct", "adx",
        "d1_close", "d1_ema50", "d1_ema100", "d1_ema200", "d1_ema50_slope", "d1_ema100_slope",
        "w_close", "w_ema20", "w_ema40", "w_ema20_slope",
        "ret_6", "ret_12", "ret_30", "weekly_bear", "bear_permission_v2",
        "ema20", "ema50", "ema89", "ema100", "entry_low", "exit_high",
        "risk_mult", "quality_mult", "short_signal", "signal", "short_exit_channel",
    ]
    features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{STRATEGY_NAME}_signal_audit.csv")
    with (out_dir / f"{STRATEGY_NAME}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 86)
    print("ETH 1D+4H Bear Short Engine V3 Backtest Summary")
    print("=" * 86)
    for k, v in summary.items():
        print(f"{k:>32}: {v}")
    print("-" * 86)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 86 + "\n")


def print_deep_report(trades: list[dict[str, Any]], features: pd.DataFrame, cfg: BearConfig, out_dir: Path) -> None:
    if features.empty:
        return
    final_capital = float(trades[-1]["capital"]) if trades else float(cfg.initial_capital)
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
    p = argparse.ArgumentParser(description="ETH short-only Bear/Crash engine V3 with stricter bear-permission filter and safe execution.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=today)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--preset", choices=sorted(PRESETS), default="high")
    p.add_argument("--style", choices=["breakdown", "crash_continuation", "bear_permission_v2", "bear_permission_v3", "combo"], default=None)
    p.add_argument("--unit-risk-per-trade", type=float, default=None)
    p.add_argument("--max-total-notional-mult", type=float, default=None)
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--min-risk-mult", type=float, default=0.25)
    p.add_argument("--max-risk-mult", type=float, default=None)
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--out-dir", default="data/reports/lf/eth_1d_4h_bear_short_engine_v3")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    preset = PRESETS[args.preset]
    cfg = BearConfig(
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        unit_risk_per_trade=float(args.unit_risk_per_trade if args.unit_risk_per_trade is not None else preset["unit_risk_per_trade"]),
        max_total_notional_mult=float(args.max_total_notional_mult if args.max_total_notional_mult is not None else preset["max_total_notional_mult"]),
        max_units=int(args.max_units if args.max_units is not None else preset["max_units"]),
        min_risk_mult=args.min_risk_mult,
        max_risk_mult=float(args.max_risk_mult if args.max_risk_mult is not None else preset["max_risk_mult"]),
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
        style=str(args.style if args.style is not None else preset["style"]),
    )

    print(f"Loading {cfg.symbol} 4H: {args.start_date} -> {args.end_date}")
    base = load_data(cfg.symbol, args.start_date, args.end_date, "4H")
    print(f"Loaded {len(base)} rows: {base.index[0]} -> {base.index[-1]}")
    features = build_bear_features(base, cfg)
    print(f"Feature rows: {len(features)}")
    print("Signal counts:", {"short": int((features.signal == -1).sum())})

    trades, equity = run_backtest(features, cfg)
    summary = summarize(trades, equity, cfg.initial_capital)
    summary["preset"] = args.preset
    summary["style"] = cfg.style
    summary["short_only"] = True
    summary["fee_rate"] = cfg.fee_rate
    summary["safe_execution"] = True

    out_dir = Path(PROJECT_ROOT) / args.out_dir
    write_outputs(trades, equity, features, summary, out_dir)
    print_summary(summary, out_dir)
    print_deep_report(trades, features, cfg, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
