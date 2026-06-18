#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH LF Portfolio V4 Risk Governor Backtest
=========================================

放置位置建议：backtest/lf/eth_lf_portfolio_v4_risk_governor_backtest.py
运行示例：
    python backtest/lf/eth_lf_portfolio_v4_risk_governor_backtest.py --start-date 2023-01-01 --end-date 2026-06-15 --preset turbo

组合定位：
    低频组合调度器 V3。V2 证明 Turtle 更适合作为确认器；V4 在 V3 基础上加入组合级风险 governor：
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
import math
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
    apply_entry_slippage,
    apply_exit_slippage,
    close_trade,
    protected_stop,
    unit_qty,
    weighted_avg_price,
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

STRATEGY_NAME = "eth_lf_portfolio_v4_risk_governor"
REPORT_STRATEGY_NAME = "ETH_LF_Portfolio_V4_RiskGovernor"

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
    """Select one portfolio signal per bar using V8 as main engine and Bear V3 as short confirmer.

    V3 rule set:
    1. V8 is the main standalone trading engine.
    2. Turtle confirms V8 long only by default. Turtle-only is ignored.
    3. Bear V3 confirms V8 short and boosts quality. It does not block V8 shorts by default.
    4. Bear V3 can open a standalone short only when V8 is silent and --disable-bear-standalone is not set.
    5. V8 long vs Bear short conflict: V8 long wins; no hedge.
    6. Only one final signal column is passed into the execution engine.
    """
    out = v8.copy()
    turtle = turtle.reindex(out.index)
    bear = bear.reindex(out.index)

    out["v8_signal"] = out["signal"].fillna(0).astype(int)
    out["turtle_signal"] = turtle["signal"].fillna(0).astype(int)
    out["bear_signal"] = bear["signal"].fillna(0).astype(int)
    out["bear_permission_v3"] = _bool_col(bear, "bear_permission_v3")

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
    only_v8 = (v8_sig != 0) & ~bear_confirm & ~turtle_agree

    # Main V8 decisions.
    final_signal.loc[v8_long | v8_short] = v8_sig.loc[v8_long | v8_short]
    out.loc[v8_long, "selected_engine"] = "V8_LONG"
    out.loc[v8_long, "selected_priority"] = PRIORITY["V8_LONG"]
    out.loc[v8_short, "selected_engine"] = "V8_SHORT"
    out.loc[v8_short, "selected_priority"] = PRIORITY["V8_SHORT"]

    # Turtle confluence: primarily long super-trend confirmation. It does not open standalone trades.
    out.loc[turtle_agree, "selected_engine"] = "V8_TURTLE_CONFLUENCE"
    out.loc[turtle_agree, "selected_priority"] = PRIORITY["V8_TURTLE_CONFLUENCE"]
    out.loc[turtle_agree, "turtle_agreement"] = True
    out.loc[turtle_agree, "quality_mult"] = (out.loc[turtle_agree, "quality_mult"] * args.turtle_confluence_quality_boost).clip(0.20, args.quality_mult_cap)

    out.loc[turtle_only, "selected_engine"] = "TURTLE_V2_IGNORED"
    out.loc[turtle_only, "selected_priority"] = PRIORITY["TURTLE_V2_IGNORED"]
    out.loc[turtle_only, "turtle_only_ignored"] = True

    # Bear V3 confluence should override Turtle label on V8 short, because it is the more relevant short confirmer.
    out.loc[bear_confirm, "selected_engine"] = "V8_BEAR_CONFLUENCE"
    out.loc[bear_confirm, "selected_priority"] = PRIORITY["V8_BEAR_CONFLUENCE"]
    out.loc[bear_confirm, "bear_confirmed"] = True
    out.loc[bear_confirm, "quality_mult"] = (out.loc[bear_confirm, "quality_mult"] * args.bear_confluence_quality_boost).clip(0.20, args.quality_mult_cap)

    # Optional defensive mode: reduce V8 shorts that are not supported by Bear's regime permission.
    nonbear_v8_short = v8_short & ~(out["bear_permission_v3"] | (bear_sig == -1))
    if args.reduce_nonbear_short_quality < 0.999:
        out.loc[nonbear_v8_short, "quality_mult"] = (out.loc[nonbear_v8_short, "quality_mult"] * args.reduce_nonbear_short_quality).clip(0.20, args.quality_mult_cap)
        out.loc[nonbear_v8_short, "nonbear_short_reduced"] = True

    # Bear-only standalone short. It uses Bear V3 risk/quality buckets and Bear's short exit channel.
    if not args.disable_bear_standalone:
        final_signal.loc[bear_only] = -1
        out.loc[bear_only, "selected_engine"] = "BEAR_V3_ONLY"
        out.loc[bear_only, "selected_priority"] = PRIORITY["BEAR_V3_ONLY"]
        out.loc[bear_only, "bear_only"] = True
        out.loc[bear_only, "risk_mult"] = (bear.loc[bear_only, "risk_mult"].fillna(out.loc[bear_only, "risk_mult"]) * args.bear_standalone_risk_scale).clip(args.min_risk_mult, args.max_risk_mult or 10.0)
        out.loc[bear_only, "quality_mult"] = (bear.loc[bear_only, "quality_mult"].fillna(1.0) * args.bear_standalone_quality_scale).clip(0.20, args.quality_mult_cap)
        out.loc[bear_only, "short_exit_channel"] = out.loc[bear_only, "bear_short_exit_channel"]

    out.loc[bear_conflict, "bear_conflict_ignored"] = True
    out.loc[bear_conflict, "portfolio_conflict"] = True

    # V8/Turtle conflict is diagnostic only. V8 already wins.
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
        else:
            item["engine"] = "UNKNOWN"
            item["engine_priority"] = 0
            item["portfolio_conflict"] = False
            item["turtle_agreement"] = False
            item["bear_confirmed"] = False
            item["bear_only"] = False
            item["bear_conflict_ignored"] = False
            item["nonbear_short_reduced"] = False
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
        "bear_confirmed", "bear_only", "bear_conflict_ignored", "nonbear_short_reduced",
        "long_signal", "short_signal", "long_exit_channel", "short_exit_channel",
        "turtle_long_exit_channel", "turtle_short_exit_channel", "bear_short_exit_channel",
    ]
    features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{STRATEGY_NAME}_signal_audit.csv")
    with (out_dir / f"{STRATEGY_NAME}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 88)
    print("ETH LF Portfolio V4 Risk Governor Backtest Summary")
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



def _recent_loss_governor(trades: list[dict[str, Any]], args: argparse.Namespace) -> float:
    """Use only already closed trades to scale down after loss clusters."""
    mult = 1.0
    if len(trades) >= 5:
        last5 = trades[-5:]
        if sum(1 for t in last5 if float(t.get("pnl", 0.0)) < 0.0) >= 3:
            mult *= args.loss_3of5_mult
    if len(trades) >= 8:
        last8 = trades[-8:]
        if sum(1 for t in last8 if float(t.get("pnl", 0.0)) < 0.0) >= 5:
            mult *= args.loss_5of8_mult
    return mult


def _equity_governor(capital: float, peak: float, equity_ema: float, trades: list[dict[str, Any]], args: argparse.Namespace) -> float:
    """Realized-equity governor. No mark-to-market and no future bars are used."""
    mult = 1.0
    if equity_ema > 0 and capital < equity_ema:
        mult *= args.equity_below_ema_mult
    dd = (peak - capital) / peak if peak > 0 else 0.0
    if dd >= 0.20:
        mult *= args.dd_20_mult
    elif dd >= 0.12:
        mult *= args.dd_12_mult
    mult *= _recent_loss_governor(trades, args)
    return max(float(args.min_governor_mult), float(mult))


def run_portfolio_backtest(df: pd.DataFrame, cfg: V8Config, args: argparse.Namespace) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """V8 SAFE execution plus portfolio-level realized-risk governor.

    It is intentionally conservative:
    - Governor is computed from already closed trades and realized capital only.
    - Current bar close can update the next stop only for the next bar.
    - Signals still execute at next 4H open.
    """
    capital = cfg.initial_capital
    peak = capital
    equity_ema = capital
    alpha = 2.0 / max(float(args.equity_ema_span) + 1.0, 2.0)
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    in_pos = False
    side = 0
    entry_i = -1
    entry_time = None
    first_entry = 0.0
    avg_entry = 0.0
    initial_sl = 0.0
    stop_price = 0.0
    risk_per_coin = 0.0
    qty = 0.0
    total_entry_fee = 0.0
    units = 0
    max_fav = 0.0
    max_adv = 0.0
    entry_risk_mult = 1.0
    entry_governor_mult = 1.0
    last_exit_i = -10**9

    rows = list(df.itertuples())
    idx = df.index

    for i in range(len(rows) - 1):
        row = rows[i]
        ts = idx[i]
        equity_ema = alpha * capital + (1.0 - alpha) * equity_ema

        if in_pos:
            high = float(row.high)
            low = float(row.low)
            close = float(row.close)
            atr_value = float(row.atr)
            hold_bars = i - entry_i

            active_stop = stop_price
            if side == 1:
                max_fav = max(max_fav, high)
                max_adv = min(max_adv, low)
                touched_stop = low <= active_stop
                channel_exit = bool(getattr(row, "long_exit_channel"))
                opposite = bool(getattr(row, "short_signal"))
                next_stop = max(stop_price, close - cfg.trailing_atr_mult * atr_value)
                locked = protected_stop(first_entry, avg_entry, side, risk_per_coin, max_fav, cfg)
                if locked is not None:
                    next_stop = max(next_stop, locked)
            else:
                max_fav = min(max_fav, low)
                max_adv = max(max_adv, high)
                touched_stop = high >= active_stop
                channel_exit = bool(getattr(row, "short_exit_channel"))
                opposite = bool(getattr(row, "long_signal"))
                next_stop = min(stop_price, close + cfg.trailing_atr_mult * atr_value)
                locked = protected_stop(first_entry, avg_entry, side, risk_per_coin, max_fav, cfg)
                if locked is not None:
                    next_stop = min(next_stop, locked)

            exit_now = False
            reason = ""
            exit_price = 0.0
            exit_time = ts
            if touched_stop:
                exit_now = True
                exit_price = apply_exit_slippage(active_stop, side, cfg.slippage_pct)
                reason = "PROTECTED_TRAILING_STOP"
            elif channel_exit:
                exit_now = True
                exit_price = apply_exit_slippage(float(rows[i + 1].open), side, cfg.slippage_pct)
                exit_time = idx[i + 1]
                reason = "DONCHIAN_EXIT_NEXT_OPEN"
            elif opposite:
                exit_now = True
                exit_price = apply_exit_slippage(float(rows[i + 1].open), side, cfg.slippage_pct)
                exit_time = idx[i + 1]
                reason = "OPPOSITE_BREAKOUT_NEXT_OPEN"
            elif hold_bars >= cfg.no_progress_bars:
                fav_r = (max_fav - first_entry) / risk_per_coin if side == 1 else (first_entry - max_fav) / risk_per_coin
                if fav_r < cfg.no_progress_min_r:
                    exit_now = True
                    exit_price = apply_exit_slippage(float(rows[i + 1].open), side, cfg.slippage_pct)
                    exit_time = idx[i + 1]
                    reason = "NO_PROGRESS_EXIT_NEXT_OPEN"
            if not exit_now and hold_bars >= cfg.max_hold_bars:
                exit_now = True
                exit_price = apply_exit_slippage(float(rows[i + 1].open), side, cfg.slippage_pct)
                exit_time = idx[i + 1]
                reason = "MAX_HOLD_EXIT_NEXT_OPEN"

            if exit_now:
                capital = close_trade(
                    trades=trades,
                    capital=capital,
                    side=side,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    first_entry=first_entry,
                    avg_entry=avg_entry,
                    exit_price=exit_price,
                    initial_sl=initial_sl,
                    stop_price=stop_price,
                    qty=qty,
                    units=units,
                    total_entry_fee=total_entry_fee,
                    fee_rate=cfg.fee_rate,
                    max_fav=max_fav,
                    max_adv=max_adv,
                    risk_per_coin=risk_per_coin,
                    holding_bars=hold_bars,
                    reason=reason,
                    risk_mult=entry_risk_mult,
                )
                trades[-1]["governor_mult"] = round(float(entry_governor_mult), 4)
                peak = max(peak, capital)
                in_pos = False
                side = 0
                last_exit_i = i
            else:
                stop_price = next_stop

            if in_pos and units < cfg.max_units:
                next_unit_number = units + 1
                trigger_r = (next_unit_number - 1) * cfg.add_every_r
                if side == 1:
                    add_triggered = high >= first_entry + trigger_r * risk_per_coin
                else:
                    add_triggered = low <= first_entry - trigger_r * risk_per_coin
                if add_triggered:
                    add_price = apply_entry_slippage(float(rows[i + 1].open), side, cfg.slippage_pct)
                    add_stop_dist = max(cfg.initial_atr_mult * atr_value, risk_per_coin)
                    add_mult = float(getattr(row, "risk_mult", entry_risk_mult)) * float(getattr(row, "quality_mult", 1.0)) * entry_governor_mult
                    add_q = unit_qty(capital, add_price, add_stop_dist, qty, cfg, add_mult)
                    if add_q > 0 and math.isfinite(add_q):
                        total_entry_fee += add_q * add_price * cfg.fee_rate
                        avg_entry = weighted_avg_price(avg_entry, qty, add_price, add_q)
                        qty += add_q
                        units += 1

        if not in_pos and i - last_exit_i >= cfg.cooldown_bars:
            signal = int(getattr(row, "signal"))
            if signal != 0:
                next_open = float(rows[i + 1].open)
                entry = apply_entry_slippage(next_open, signal, cfg.slippage_pct)
                atr_value = float(row.atr)
                sl = entry - cfg.initial_atr_mult * atr_value if signal == 1 else entry + cfg.initial_atr_mult * atr_value
                stop_dist = abs(entry - sl)
                governor_mult = _equity_governor(capital, peak, equity_ema, trades, args)
                entry_risk_mult = float(getattr(row, "risk_mult", 1.0)) * float(getattr(row, "quality_mult", 1.0)) * governor_mult
                q = unit_qty(capital, entry, stop_dist, 0.0, cfg, entry_risk_mult)
                if q > 0 and math.isfinite(q):
                    in_pos = True
                    side = signal
                    entry_i = i + 1
                    entry_time = idx[i + 1]
                    first_entry = entry
                    avg_entry = entry
                    initial_sl = sl
                    stop_price = sl
                    risk_per_coin = stop_dist
                    qty = q
                    total_entry_fee = qty * entry * cfg.fee_rate
                    units = 1
                    max_fav = entry
                    max_adv = entry
                    entry_governor_mult = governor_mult

        equity_rows.append({"time": ts, "capital": capital, "drawdown_pct": (peak - capital) / peak if peak > 0 else 0.0, "equity_ema": equity_ema})

    if in_pos:
        ts = idx[-1]
        close = float(df.iloc[-1]["close"])
        exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
        capital = close_trade(
            trades=trades,
            capital=capital,
            side=side,
            entry_time=entry_time,
            exit_time=ts,
            first_entry=first_entry,
            avg_entry=avg_entry,
            exit_price=exit_price,
            initial_sl=initial_sl,
            stop_price=stop_price,
            qty=qty,
            units=units,
            total_entry_fee=total_entry_fee,
            fee_rate=cfg.fee_rate,
            max_fav=max_fav,
            max_adv=max_adv,
            risk_per_coin=risk_per_coin,
            holding_bars=len(df) - 1 - entry_i,
            reason="FORCE_CLOSE_END",
            risk_mult=entry_risk_mult,
        )
        trades[-1]["governor_mult"] = round(float(entry_governor_mult), 4)

    equity = pd.DataFrame(equity_rows).set_index("time") if equity_rows else pd.DataFrame()
    return trades, equity

def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="ETH LF Portfolio V4: V8 main + Turtle confirm + Bear V3 short confirm/standalone.")
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
    p.add_argument("--reduce-nonbear-short-quality", type=float, default=0.95, help="将 Bear 未确认的 V8 short 质量乘以该系数。V4 默认 0.95，轻微降低假空头回撤。")
    p.add_argument("--quality-mult-cap", type=float, default=2.20)
    p.add_argument("--equity-ema-span", type=int, default=80)
    p.add_argument("--equity-below-ema-mult", type=float, default=0.99)
    p.add_argument("--dd-12-mult", type=float, default=0.96)
    p.add_argument("--dd-20-mult", type=float, default=0.88)
    p.add_argument("--loss-3of5-mult", type=float, default=0.96)
    p.add_argument("--loss-5of8-mult", type=float, default=0.88)
    p.add_argument("--min-governor-mult", type=float, default=0.80)

    p.add_argument("--out-dir", default="data/reports/lf/eth_lf_portfolio_v4_risk_governor")
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

    trades, equity = run_portfolio_backtest(features, v8_cfg, args)
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
