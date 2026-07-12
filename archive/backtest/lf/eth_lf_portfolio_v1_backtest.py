#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH LF Portfolio V1 Backtest
============================

放置位置建议：backtest/lf/eth_lf_portfolio_v1_backtest.py
运行示例：
    python backtest/lf/eth_lf_portfolio_v1_backtest.py --start-date 2023-01-01 --end-date 2026-06-15 --preset turbo

组合定位：
    低频组合调度器 V1。不是简单收益相加，而是单账户、单持仓、带优先级的策略调度：
        1) 同一时刻只允许一个组合持仓；
        2) V8 Trend Rider 是主引擎；
        3) Turtle V2 是超级趋势补充引擎；
        4) 同方向共振时提高质量权重；
        5) 多空冲突时按优先级选一个，不双开、不对冲；
        6) 入场/加仓/退出沿用 V8 SAFE 执行，不用同 bar 新 stop 触发。

反偷看：
    - V8 和 Turtle 的 Donchian / 日线 regime 已在各自 build_features 内 shift；
    - 组合调度只使用当前已收盘 4H bar 的两个子策略信号；
    - 组合信号在当前 4H 收盘确认，下一根 4H open 执行；
    - 当前 bar close 更新的新 stop 下一根 bar 才生效。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backtest_common.data import load_ohlcv_data as load_data  # noqa: E402
from src.backtest_common.reporting import build_report_trades  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402

from backtest.lf.eth_1d_4h_trend_rider_v8_position_lock_backtest import (  # noqa: E402
    PRESETS as V8_PRESETS,
    StrategyConfig as V8Config,
    build_features as build_v8_features,
    run_backtest as run_v8_backtest,
    summarize as summarize_v8,
)
from backtest.lf.eth_4h_turtle_pyramid_v2_backtest import (  # noqa: E402
    StrategyConfig as TurtleConfig,
    build_features as build_turtle_features,
)

STRATEGY_NAME = "eth_lf_portfolio_v1"
REPORT_STRATEGY_NAME = "ETH_LF_Portfolio_V1"


# 单持仓调度优先级。数值越大越优先。
PRIORITY = {
    "V8_TREND_RIDER": 100,
    "TURTLE_V2": 85,
    "V8_TURTLE_CONFLUENCE": 120,
}


def make_v8_config(args: argparse.Namespace) -> V8Config:
    preset = V8_PRESETS[args.preset]
    return V8Config(
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


def make_turtle_config(args: argparse.Namespace, v8_cfg: V8Config) -> TurtleConfig:
    return TurtleConfig(
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        # Turtle 只是补充引擎，不允许它单独抢太多风险。
        unit_risk_per_trade=v8_cfg.unit_risk_per_trade * args.turtle_unit_risk_scale,
        max_total_notional_mult=v8_cfg.max_total_notional_mult,
        max_units=max(1, min(v8_cfg.max_units, args.turtle_max_units)),
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
        enable_short=args.enable_turtle_short,
    )


def select_portfolio_signals(v8: pd.DataFrame, turtle: pd.DataFrame, cfg: V8Config, args: argparse.Namespace) -> pd.DataFrame:
    """Select one portfolio signal per bar using priority and conflict rules.

    Rules:
    1. If V8 and Turtle agree on direction, choose confluence and boost risk/quality.
    2. If they conflict, V8 wins because it has faster regime/risk filters.
    3. If only one strategy signals, choose that strategy.
    4. Only one final signal column is passed into the execution engine.
    """
    out = v8.copy()
    turtle = turtle.reindex(out.index)

    out["v8_signal"] = out["signal"].fillna(0).astype(int)
    out["turtle_signal"] = turtle["signal"].fillna(0).astype(int)
    out["turtle_long_exit_channel"] = turtle["long_exit_channel"].fillna(False).astype(bool)
    out["turtle_short_exit_channel"] = turtle["short_exit_channel"].fillna(False).astype(bool)

    out["selected_engine"] = "NONE"
    out["selected_priority"] = 0
    out["portfolio_conflict"] = False
    out["portfolio_agreement"] = False

    final_signal = pd.Series(0, index=out.index, dtype="int64")

    v8_sig = out["v8_signal"]
    turtle_sig = out["turtle_signal"]
    agree = (v8_sig != 0) & (v8_sig == turtle_sig)
    conflict = (v8_sig != 0) & (turtle_sig != 0) & (v8_sig != turtle_sig)
    only_v8 = (v8_sig != 0) & (turtle_sig == 0)
    only_turtle = (v8_sig == 0) & (turtle_sig != 0)

    final_signal.loc[agree] = v8_sig.loc[agree]
    final_signal.loc[conflict] = v8_sig.loc[conflict]
    final_signal.loc[only_v8] = v8_sig.loc[only_v8]
    final_signal.loc[only_turtle] = turtle_sig.loc[only_turtle]

    out.loc[agree, "selected_engine"] = "V8_TURTLE_CONFLUENCE"
    out.loc[agree, "selected_priority"] = PRIORITY["V8_TURTLE_CONFLUENCE"]
    out.loc[agree, "portfolio_agreement"] = True

    out.loc[conflict | only_v8, "selected_engine"] = "V8_TREND_RIDER"
    out.loc[conflict | only_v8, "selected_priority"] = PRIORITY["V8_TREND_RIDER"]
    out.loc[conflict, "portfolio_conflict"] = True

    out.loc[only_turtle, "selected_engine"] = "TURTLE_V2"
    out.loc[only_turtle, "selected_priority"] = PRIORITY["TURTLE_V2"]

    # 风险调整：共振加风险，Turtle 单独信号降风险。
    out.loc[agree, "quality_mult"] = (out.loc[agree, "quality_mult"] * args.confluence_quality_boost).clip(0.20, 2.20)
    out.loc[only_turtle, "risk_mult"] = args.turtle_risk_mult
    out.loc[only_turtle, "quality_mult"] = args.turtle_quality_mult

    # 对 Turtle 单独信号，退出通道也切到 Turtle 自己的通道。
    out.loc[only_turtle, "long_exit_channel"] = out.loc[only_turtle, "turtle_long_exit_channel"]
    out.loc[only_turtle, "short_exit_channel"] = out.loc[only_turtle, "turtle_short_exit_channel"]

    out["signal"] = final_signal
    return out


def attach_engine_to_trades(trades: list[dict[str, Any]], features: pd.DataFrame) -> list[dict[str, Any]]:
    if not trades or features.empty:
        return trades
    out: list[dict[str, Any]] = []
    for trade in trades:
        item = dict(trade)
        entry_time = pd.Timestamp(item["entry_time"])
        # 策略信号在上一根 4H bar 收盘确认，下一根 open 入场。
        signal_time = entry_time - pd.Timedelta(hours=4)
        if signal_time in features.index:
            row = features.loc[signal_time]
            item["engine"] = str(row.get("selected_engine", "UNKNOWN"))
            item["engine_priority"] = int(row.get("selected_priority", 0))
            item["portfolio_conflict"] = bool(row.get("portfolio_conflict", False))
            item["portfolio_agreement"] = bool(row.get("portfolio_agreement", False))
        else:
            item["engine"] = "UNKNOWN"
            item["engine_priority"] = 0
            item["portfolio_conflict"] = False
            item["portfolio_agreement"] = False
        out.append(item)
    return out


def write_outputs(trades: list[dict[str, Any]], equity: pd.DataFrame, features: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / f"{STRATEGY_NAME}_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / f"{STRATEGY_NAME}_equity.csv")
    audit_cols = [
        "open", "high", "low", "close", "volume", "atr", "atr_pct", "adx",
        "risk_mult", "quality_mult", "v8_signal", "turtle_signal", "signal",
        "selected_engine", "selected_priority", "portfolio_conflict", "portfolio_agreement",
        "long_signal", "short_signal", "long_exit_channel", "short_exit_channel",
    ]
    features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{STRATEGY_NAME}_signal_audit.csv")
    with (out_dir / f"{STRATEGY_NAME}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 86)
    print("ETH LF Portfolio V1 Backtest Summary")
    print("=" * 86)
    for k, v in summary.items():
        print(f"{k:>32}: {v}")
    print("-" * 86)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 86 + "\n")


def print_deep_report(trades: list[dict[str, Any]], features: pd.DataFrame, cfg: V8Config, out_dir: Path) -> None:
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
    p = argparse.ArgumentParser(description="ETH LF Portfolio V1: one-position priority scheduler for V8 Trend Rider + Turtle V2.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=today)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--preset", choices=sorted(V8_PRESETS), default="turbo")
    p.add_argument("--unit-risk-per-trade", type=float, default=None)
    p.add_argument("--max-total-notional-mult", type=float, default=None)
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--min-risk-mult", type=float, default=0.35)
    p.add_argument("--max-risk-mult", type=float, default=None)
    p.add_argument("--fee-rate", type=float, default=0.0005)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--disable-short", action="store_true")
    p.add_argument("--enable-turtle-short", action="store_true", help="默认 Turtle 只作为 long super-trend 补充；打开后允许 Turtle short。")
    p.add_argument("--turtle-unit-risk-scale", type=float, default=0.45)
    p.add_argument("--turtle-max-units", type=int, default=3)
    p.add_argument("--turtle-risk-mult", type=float, default=0.75)
    p.add_argument("--turtle-quality-mult", type=float, default=0.75)
    p.add_argument("--confluence-quality-boost", type=float, default=1.15)
    p.add_argument("--out-dir", default="data/reports/lf/eth_lf_portfolio_v1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    v8_cfg = make_v8_config(args)
    turtle_cfg = make_turtle_config(args, v8_cfg)

    print(f"Loading {args.symbol} 4H: {args.start_date} -> {args.end_date}")
    base = load_data(args.symbol, args.start_date, args.end_date, "4H")
    print(f"Loaded {len(base)} rows: {base.index[0]} -> {base.index[-1]}")

    v8_features = build_v8_features(base, v8_cfg)
    turtle_features = build_turtle_features(base, turtle_cfg)
    features = select_portfolio_signals(v8_features, turtle_features, v8_cfg, args)

    print("Signal counts:", {
        "v8_long": int((features.v8_signal == 1).sum()),
        "v8_short": int((features.v8_signal == -1).sum()),
        "turtle_long": int((features.turtle_signal == 1).sum()),
        "turtle_short": int((features.turtle_signal == -1).sum()),
        "portfolio_long": int((features.signal == 1).sum()),
        "portfolio_short": int((features.signal == -1).sum()),
        "agreement": int(features.portfolio_agreement.sum()),
        "conflict": int(features.portfolio_conflict.sum()),
    })

    trades, equity = run_v8_backtest(features, v8_cfg)
    trades = attach_engine_to_trades(trades, features)
    summary = summarize_v8(trades, equity, v8_cfg.initial_capital)
    if trades:
        tdf = pd.DataFrame(trades)
        summary["engine_counts"] = tdf["engine"].value_counts().to_dict() if "engine" in tdf.columns else {}
        summary["conflict_trade_count"] = int(tdf.get("portfolio_conflict", pd.Series(dtype=bool)).sum())
        summary["agreement_trade_count"] = int(tdf.get("portfolio_agreement", pd.Series(dtype=bool)).sum())
    summary["preset"] = args.preset
    summary["single_active_position"] = True
    summary["conflict_rule"] = "V8 wins conflict; same-direction V8+Turtle gets confluence boost; Turtle only fills when V8 is silent."

    out_dir = Path(PROJECT_ROOT) / args.out_dir
    write_outputs(trades, equity, features, summary, out_dir)
    print_summary(summary, out_dir)
    print_deep_report(trades, features, v8_cfg, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
