#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH LF Portfolio V4B Short Quality Backtest
=========================================

放置位置建议：backtest/lf/eth_lf_portfolio_v4b_short_quality_backtest.py
运行示例：
    python backtest/lf/eth_lf_portfolio_v4b_short_quality_backtest.py --start-date 2023-01-01 --end-date 2026-06-15 --preset turbo

组合定位：
    低频组合调度器 V3。V2 证明 Turtle 更适合作为确认器；V4B 在 V3 基础上做 short quality bucket，不做全局净值降风险：
        1) 同一时刻只允许一个组合持仓；
        2) V8 Trend Rider 是主引擎；
        3) Turtle V2 只做 long super-trend 确认，不允许单独开仓；
        4) Bear Short Engine V3 只做 short 确认 / 少量独立熊市 short 候选；
        5) V8 short + Bear V3 同向时提高质量权重；
        6) V8 long 与 Bear short 冲突时，不对冲，默认 V8 long 优先；
        7) 默认不强行禁止 V8 short，因为严格 gate 会砍掉 2026 的主要收益；
        8) 入场/加仓/退出沿用 V8 SAFE 执行，不用同 bar 新 stop 触发。

反偷看：
    - V8、Turtle、Bear 的 Donchian / 高周期 regime 都在各自 build_features 内 shift；
    - 组合调度只使用当前已收盘 4H bar 的子引擎信号；
    - 组合信号在当前 4H 收盘确认，下一根 4H open 执行；
    - 当前 bar close 更新的新 stop 下一根 bar 才生效；
    - 不允许同一根 K 线内用新 stop 立刻触发。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from backtest.lf.eth_1d_4h_bear_short_engine_v3_backtest import (  # noqa: E402
    PRESETS as BEAR_PRESETS,
    BearConfig,
    build_bear_features,
)

STRATEGY_NAME = "eth_lf_portfolio_v4b_short_quality"
REPORT_STRATEGY_NAME = "ETH_LF_Portfolio_V4B_ShortQuality"

PRIORITY = {
    "V8_TREND_RIDER": 100,
    "V8_LONG": 100,
    "V8_SHORT": 100,
    "V8_TURTLE_CONFLUENCE": 120,
    "V8_BEAR_CONFLUENCE": 125,
    "BEAR_V3_ONLY": 90,
    "TURTLE_V2_IGNORED": 50,
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
        # Turtle 在 V3 中只作为确认器；这里保留配置是为了生成 Turtle 信号。
        unit_risk_per_trade=v8_cfg.unit_risk_per_trade * args.turtle_unit_risk_scale,
        max_total_notional_mult=v8_cfg.max_total_notional_mult,
        max_units=max(1, min(v8_cfg.max_units, args.turtle_max_units)),
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
        enable_short=args.enable_turtle_short,
    )


def make_bear_config(args: argparse.Namespace, v8_cfg: V8Config) -> BearConfig:
    preset = BEAR_PRESETS[args.bear_preset]
    return BearConfig(
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        # Bear 在组合里主要贡献 signal / risk bucket。独立 Bear-only 用组合的 V8 执行器统一控总风险。
        unit_risk_per_trade=float(preset["unit_risk_per_trade"]),
        max_total_notional_mult=float(preset["max_total_notional_mult"]),
        max_units=int(preset["max_units"]),
        min_risk_mult=args.bear_min_risk_mult,
        max_risk_mult=float(preset["max_risk_mult"]),
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
        style=str(preset["style"]),
    )


def _bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype("boolean").fillna(False).astype(bool)


def select_portfolio_signals(v8: pd.DataFrame, turtle: pd.DataFrame, bear: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Select one signal per bar with coarse short quality buckets.

    V4B rule set:
    1. Long side keeps V3 logic. Do not reduce long trend profits.
    2. Bear confirmed short is still boosted.
    3. Bear permission short is allowed at normal risk.
    4. Fast-bear-only V8 shorts are reduced, because these are more likely to be bull/neutral pullbacks.
    5. Recent rebound / deep extension add only coarse penalties. No date/year/month rule is used.
    """
    out = v8.copy()
    turtle = turtle.reindex(out.index)
    bear = bear.reindex(out.index)

    out["v8_signal"] = out["signal"].fillna(0).astype(int)
    out["turtle_signal"] = turtle["signal"].fillna(0).astype(int)
    out["bear_signal"] = bear["signal"].fillna(0).astype(int)
    out["bear_permission_v3"] = _bool_col(bear, "bear_permission_v3")

    out["bear_d1_close"] = bear.get("d1_close", pd.Series(index=out.index, dtype=float))
    out["bear_d1_ema100"] = bear.get("d1_ema100", pd.Series(index=out.index, dtype=float))
    out["bear_d1_ema50_slope"] = bear.get("d1_ema50_slope", pd.Series(index=out.index, dtype=float))
    out["bear_ret_12"] = bear.get("ret_12", pd.Series(index=out.index, dtype=float))
    out["bear_price_vs_ema100"] = out["bear_d1_close"] / out["bear_d1_ema100"] - 1.0

    out["turtle_long_exit_channel"] = _bool_col(turtle, "long_exit_channel")
    out["turtle_short_exit_channel"] = _bool_col(turtle, "short_exit_channel")
    out["bear_short_exit_channel"] = _bool_col(bear, "short_exit_channel")

    out["selected_engine"] = "NONE"
    out["selected_priority"] = 0
    out["portfolio_conflict"] = False
    out["turtle_agreement"] = False
    out["turtle_only_ignored"] = False
    out["bear_confirmed"] = False
    out["bear_only"] = False
    out["bear_conflict_ignored"] = False
    out["short_quality_bucket"] = "NONE"
    out["nonbear_short_reduced"] = False

    final_signal = pd.Series(0, index=out.index, dtype="int64")

    v8_sig = out["v8_signal"]
    turtle_sig = out["turtle_signal"]
    bear_sig = out["bear_signal"]

    v8_long = v8_sig == 1
    v8_short = v8_sig == -1
    turtle_agree = (v8_sig != 0) & (v8_sig == turtle_sig)
    turtle_only = (v8_sig == 0) & (turtle_sig != 0)
    bear_confirm = v8_short & (bear_sig == -1)
    bear_only = (v8_sig == 0) & (bear_sig == -1) & (not args.disable_bear_standalone)
    bear_conflict = v8_long & (bear_sig == -1)

    # Coarse market-state buckets for V8 shorts.
    major_bear_context = (
        (out["bear_d1_close"] < out["bear_d1_ema100"])
        & (out["bear_d1_ema50_slope"] < -0.004)
    ).fillna(False)
    bear_permission = out["bear_permission_v3"]
    permission_short = v8_short & (~bear_confirm) & bear_permission
    major_bear_short = v8_short & (~bear_confirm) & (~bear_permission) & major_bear_context
    fast_bear_only_short = v8_short & (~bear_confirm) & (~bear_permission) & (~major_bear_context)
    rebound_risk = v8_short & (out["bear_ret_12"] > 0.005).fillna(False)
    extension_risk = v8_short & (out["bear_price_vs_ema100"] < -0.110).fillna(False)

    # Main V8 decisions.
    final_signal.loc[v8_long | v8_short] = v8_sig.loc[v8_long | v8_short]
    out.loc[v8_long, "selected_engine"] = "V8_LONG"
    out.loc[v8_long, "selected_priority"] = PRIORITY["V8_LONG"]
    out.loc[v8_short, "selected_engine"] = "V8_SHORT"
    out.loc[v8_short, "selected_priority"] = PRIORITY["V8_SHORT"]

    # Turtle confirms long super-trend. Turtle-only still ignored.
    out.loc[turtle_agree, "selected_engine"] = "V8_TURTLE_CONFLUENCE"
    out.loc[turtle_agree, "selected_priority"] = PRIORITY["V8_TURTLE_CONFLUENCE"]
    out.loc[turtle_agree, "turtle_agreement"] = True
    out.loc[turtle_agree, "quality_mult"] = (out.loc[turtle_agree, "quality_mult"] * args.turtle_confluence_quality_boost).clip(0.20, args.quality_mult_cap)

    out.loc[turtle_only, "selected_engine"] = "TURTLE_V2_IGNORED"
    out.loc[turtle_only, "selected_priority"] = PRIORITY["TURTLE_V2_IGNORED"]
    out.loc[turtle_only, "turtle_only_ignored"] = True

    # Short buckets. Keep these coarse to avoid fitting a few historical trades.
    out.loc[bear_confirm, "selected_engine"] = "V8_BEAR_CONFLUENCE"
    out.loc[bear_confirm, "selected_priority"] = PRIORITY["V8_BEAR_CONFLUENCE"]
    out.loc[bear_confirm, "bear_confirmed"] = True
    out.loc[bear_confirm, "short_quality_bucket"] = "A_BEAR_SIGNAL"
    out.loc[bear_confirm, "quality_mult"] = (out.loc[bear_confirm, "quality_mult"] * args.bear_confluence_quality_boost).clip(0.20, args.quality_mult_cap)

    out.loc[permission_short, "short_quality_bucket"] = "B_BEAR_PERMISSION"
    out.loc[permission_short, "quality_mult"] = (out.loc[permission_short, "quality_mult"] * args.short_permission_quality_mult).clip(0.20, args.quality_mult_cap)

    out.loc[major_bear_short, "short_quality_bucket"] = "C_MAJOR_BEAR_CONTEXT"
    out.loc[major_bear_short, "quality_mult"] = (out.loc[major_bear_short, "quality_mult"] * args.short_major_bear_quality_mult).clip(0.20, args.quality_mult_cap)

    out.loc[fast_bear_only_short, "short_quality_bucket"] = "D_FAST_BEAR_ONLY"
    out.loc[fast_bear_only_short, "quality_mult"] = (out.loc[fast_bear_only_short, "quality_mult"] * args.short_fast_bear_only_quality_mult).clip(0.20, args.quality_mult_cap)

    out.loc[rebound_risk, "quality_mult"] = (out.loc[rebound_risk, "quality_mult"] * args.short_rebound_penalty_mult).clip(0.20, args.quality_mult_cap)
    out.loc[extension_risk, "quality_mult"] = (out.loc[extension_risk, "quality_mult"] * args.short_extension_penalty_mult).clip(0.20, args.quality_mult_cap)

    # Bear-only standalone short. It uses Bear V3 risk/quality buckets and Bear's short exit channel.
    if not args.disable_bear_standalone:
        final_signal.loc[bear_only] = -1
        out.loc[bear_only, "selected_engine"] = "BEAR_V3_ONLY"
        out.loc[bear_only, "selected_priority"] = PRIORITY["BEAR_V3_ONLY"]
        out.loc[bear_only, "bear_only"] = True
        out.loc[bear_only, "short_quality_bucket"] = "A_BEAR_ONLY"
        out.loc[bear_only, "risk_mult"] = (bear.loc[bear_only, "risk_mult"].fillna(out.loc[bear_only, "risk_mult"]) * args.bear_standalone_risk_scale).clip(args.min_risk_mult, args.max_risk_mult or 10.0)
        out.loc[bear_only, "quality_mult"] = (bear.loc[bear_only, "quality_mult"].fillna(1.0) * args.bear_standalone_quality_scale).clip(0.20, args.quality_mult_cap)
        out.loc[bear_only, "short_exit_channel"] = out.loc[bear_only, "bear_short_exit_channel"]

    out.loc[bear_conflict, "bear_conflict_ignored"] = True
    out.loc[bear_conflict, "portfolio_conflict"] = True
    v8_turtle_conflict = (v8_sig != 0) & (turtle_sig != 0) & (v8_sig != turtle_sig)
    out.loc[v8_turtle_conflict, "portfolio_conflict"] = True

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
            item["turtle_agreement"] = bool(row.get("turtle_agreement", False))
            item["bear_confirmed"] = bool(row.get("bear_confirmed", False))
            item["bear_only"] = bool(row.get("bear_only", False))
            item["bear_conflict_ignored"] = bool(row.get("bear_conflict_ignored", False))
            item["nonbear_short_reduced"] = bool(row.get("nonbear_short_reduced", False))
            item["short_quality_bucket"] = str(row.get("short_quality_bucket", "UNKNOWN"))
        else:
            item["engine"] = "UNKNOWN"
            item["engine_priority"] = 0
            item["portfolio_conflict"] = False
            item["turtle_agreement"] = False
            item["bear_confirmed"] = False
            item["bear_only"] = False
            item["bear_conflict_ignored"] = False
            item["nonbear_short_reduced"] = False
            item["short_quality_bucket"] = "UNKNOWN"
        out.append(item)
    return out


def write_outputs(trades: list[dict[str, Any]], equity: pd.DataFrame, features: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / f"{STRATEGY_NAME}_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / f"{STRATEGY_NAME}_equity.csv")
    audit_cols = [
        "open", "high", "low", "close", "volume", "atr", "atr_pct", "adx",
        "risk_mult", "quality_mult", "v8_signal", "turtle_signal", "bear_signal", "bear_permission_v3", "signal",
        "selected_engine", "selected_priority", "portfolio_conflict", "turtle_agreement", "turtle_only_ignored",
        "bear_confirmed", "bear_only", "bear_conflict_ignored", "nonbear_short_reduced", "short_quality_bucket",
        "long_signal", "short_signal", "long_exit_channel", "short_exit_channel",
        "turtle_long_exit_channel", "turtle_short_exit_channel", "bear_short_exit_channel",
    ]
    features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{STRATEGY_NAME}_signal_audit.csv")
    with (out_dir / f"{STRATEGY_NAME}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 88)
    print("ETH LF Portfolio V4B Short Quality Backtest Summary")
    print("=" * 88)
    for k, v in summary.items():
        print(f"{k:>34}: {v}")
    print("-" * 88)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 88 + "\n")


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
    p = argparse.ArgumentParser(description="ETH LF Portfolio V4B: V8 main + Turtle confirm + Bear V3 short confirm/standalone.")
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
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--disable-short", action="store_true")

    p.add_argument("--enable-turtle-short", action="store_true", help="默认 Turtle 只确认 long super-trend；打开后也允许 Turtle short 参与确认。")
    p.add_argument("--turtle-unit-risk-scale", type=float, default=0.45, help="仅用于生成 Turtle 信号；组合不允许 Turtle 单独开仓。")
    p.add_argument("--turtle-max-units", type=int, default=3, help="仅用于生成 Turtle 信号；组合不允许 Turtle 单独开仓。")
    p.add_argument("--turtle-confluence-quality-boost", type=float, default=1.30)

    p.add_argument("--bear-preset", choices=sorted(BEAR_PRESETS), default="high", help="Bear V3 信号/质量桶 preset。默认 high，避免过度拟合极端年份。")
    p.add_argument("--bear-min-risk-mult", type=float, default=0.25)
    p.add_argument("--bear-confluence-quality-boost", type=float, default=1.30)
    p.add_argument("--bear-standalone-risk-scale", type=float, default=1.0)
    p.add_argument("--bear-standalone-quality-scale", type=float, default=1.0)
    p.add_argument("--disable-bear-standalone", action="store_true", help="关闭 Bear V3 在 V8 无信号时的独立 short 候选。")
    p.add_argument("--short-permission-quality-mult", type=float, default=1.00, help="V8 short 且 Bear permission 成立但没有正式 Bear signal。")
    p.add_argument("--short-major-bear-quality-mult", type=float, default=0.88, help="V8 short 处于大级别偏空，但 Bear permission 不成立。")
    p.add_argument("--short-fast-bear-only-quality-mult", type=float, default=0.50, help="只有 V8 快速日线 bear，Bear 大级别过滤不认可。粗粒度降到半仓，不按年份调参。")
    p.add_argument("--short-rebound-penalty-mult", type=float, default=0.82, help="近期 4H 反弹后追空的额外惩罚。")
    p.add_argument("--short-extension-penalty-mult", type=float, default=0.82, help="价格相对 1D EMA100 跌太远后的追空惩罚。")
    p.add_argument("--quality-mult-cap", type=float, default=2.20)

    p.add_argument("--out-dir", default="data/reports/lf/eth_lf_portfolio_v4b_short_quality")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    v8_cfg = make_v8_config(args)
    turtle_cfg = make_turtle_config(args, v8_cfg)
    bear_cfg = make_bear_config(args, v8_cfg)

    print(f"Loading {args.symbol} 4H: {args.start_date} -> {args.end_date}")
    base = load_data(args.symbol, args.start_date, args.end_date, "4H")
    print(f"Loaded {len(base)} rows: {base.index[0]} -> {base.index[-1]}")

    v8_features = build_v8_features(base, v8_cfg)
    turtle_features = build_turtle_features(base, turtle_cfg)
    bear_features = build_bear_features(base, bear_cfg)
    features = select_portfolio_signals(v8_features, turtle_features, bear_features, args)

    print("Signal counts:", {
        "v8_long": int((features.v8_signal == 1).sum()),
        "v8_short": int((features.v8_signal == -1).sum()),
        "turtle_long": int((features.turtle_signal == 1).sum()),
        "turtle_short": int((features.turtle_signal == -1).sum()),
        "bear_short": int((features.bear_signal == -1).sum()),
        "portfolio_long": int((features.signal == 1).sum()),
        "portfolio_short": int((features.signal == -1).sum()),
        "turtle_agreement": int(features.turtle_agreement.sum()),
        "bear_confirmed": int(features.bear_confirmed.sum()),
        "bear_only": int(features.bear_only.sum()),
        "conflict": int(features.portfolio_conflict.sum()),
        "nonbear_short_reduced": int(features.nonbear_short_reduced.sum()),
    })

    trades, equity = run_v8_backtest(features, v8_cfg)
    trades = attach_engine_to_trades(trades, features)
    summary = summarize_v8(trades, equity, v8_cfg.initial_capital)
    if trades:
        tdf = pd.DataFrame(trades)
        summary["engine_counts"] = tdf["engine"].value_counts().to_dict() if "engine" in tdf.columns else {}
        summary["conflict_trade_count"] = int(tdf.get("portfolio_conflict", pd.Series(dtype=bool)).sum())
        summary["turtle_agreement_trade_count"] = int(tdf.get("turtle_agreement", pd.Series(dtype=bool)).sum())
        summary["bear_confirmed_trade_count"] = int(tdf.get("bear_confirmed", pd.Series(dtype=bool)).sum())
        summary["bear_only_trade_count"] = int(tdf.get("bear_only", pd.Series(dtype=bool)).sum())
        summary["nonbear_short_reduced_trade_count"] = int(tdf.get("nonbear_short_reduced", pd.Series(dtype=bool)).sum())
        if "short_quality_bucket" in tdf.columns:
            summary["short_quality_bucket_counts"] = tdf["short_quality_bucket"].value_counts().to_dict()
    summary["preset"] = args.preset
    summary["bear_preset"] = args.bear_preset
    summary["single_active_position"] = True
    summary["conflict_rule"] = "V8 main; Turtle confirms long; Bear confirms short or takes standalone short when V8 silent; no hedge; V8 long beats Bear short conflict."
    summary["fee_rate_per_side"] = args.fee_rate

    out_dir = Path(PROJECT_ROOT) / args.out_dir
    write_outputs(trades, equity, features, summary, out_dir)
    print_summary(summary, out_dir)
    print_deep_report(trades, features, v8_cfg, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
