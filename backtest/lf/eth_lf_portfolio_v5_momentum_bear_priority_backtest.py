#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH LF Portfolio V5 Momentum + Bear Priority
===========================================

组合定位：
    Momentum Breakout V3 是最高优先级主引擎。
    Bear Short Engine V3 只在 Momentum 无信号时作为熊市 short 补充。

设计原则：
    - 不把 V4B/V8 fallback 硬塞进来；测试显示 V8 fallback 与 Momentum 高重叠，反而增加噪音。
    - 同一时间只允许一个 active position。
    - 不对冲，不双开。
    - 当前 4H close 确认信号，下一根 4H open 执行。
    - 当前 bar close 更新的新 stop 下一根 bar 才生效。
    - 不用年份、月份、日期过滤。
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
from src.backtest_common.execution import apply_entry_slippage, apply_exit_slippage  # noqa: E402
from src.backtest_common.reporting import build_report_trades  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402

from backtest.lf.eth_1d_4h_momentum_breakout_v3_backtest import (  # noqa: E402
    PRESETS as MOMENTUM_PRESETS,
    MomentumConfig,
    build_features as build_momentum_features,
)
from backtest.lf.eth_1d_4h_bear_short_engine_v3_backtest import (  # noqa: E402
    PRESETS as BEAR_PRESETS,
    BearConfig,
    build_bear_features,
)
from backtest.lf.eth_1d_4h_trend_rider_v8_position_lock_backtest import (  # noqa: E402
    StrategyConfig as ExecConfig,
    close_trade,
    protected_stop,
    summarize,
    unit_qty,
    weighted_avg_price,
)

STRATEGY_NAME = "eth_lf_portfolio_v5_momentum_bear_priority"
REPORT_STRATEGY_NAME = "ETH_LF_Portfolio_V5_MomentumBearPriority"

PRIORITY = {
    "MOMENTUM_V3": 150,
    "BEAR_V3_ONLY": 90,
}


def make_momentum_config(args: argparse.Namespace) -> MomentumConfig:
    preset = MOMENTUM_PRESETS[args.preset]
    return MomentumConfig(
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


def make_exec_config(cfg: MomentumConfig) -> ExecConfig:
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


def make_bear_config(args: argparse.Namespace) -> BearConfig:
    preset = BEAR_PRESETS[args.bear_preset]
    return BearConfig(
        symbol=args.symbol,
        initial_capital=args.initial_capital,
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


def select_portfolio_signals(momentum: pd.DataFrame, bear: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = momentum.copy()
    bear = bear.reindex(out.index)

    out["momentum_signal"] = out["signal"].fillna(0).astype(int)
    out["bear_signal"] = bear["signal"].fillna(0).astype(int)
    out["momentum_long_exit_channel"] = _bool_col(momentum, "long_exit_channel")
    out["momentum_short_exit_channel"] = _bool_col(momentum, "short_exit_channel")
    out["bear_short_exit_channel"] = _bool_col(bear, "short_exit_channel")

    out["selected_engine"] = "NONE"
    out["selected_priority"] = 0
    out["momentum_selected"] = False
    out["bear_only"] = False
    out["portfolio_conflict"] = False

    final_signal = pd.Series(0, index=out.index, dtype="int64")
    mom_active = out["momentum_signal"] != 0
    final_signal.loc[mom_active] = out.loc[mom_active, "momentum_signal"]
    out.loc[mom_active, "selected_engine"] = "MOMENTUM_V3"
    out.loc[mom_active, "selected_priority"] = PRIORITY["MOMENTUM_V3"]
    out.loc[mom_active, "momentum_selected"] = True

    bear_only = (~mom_active) & (out["bear_signal"] == -1) & (not args.disable_bear_standalone)
    final_signal.loc[bear_only] = -1
    out.loc[bear_only, "selected_engine"] = "BEAR_V3_ONLY"
    out.loc[bear_only, "selected_priority"] = PRIORITY["BEAR_V3_ONLY"]
    out.loc[bear_only, "bear_only"] = True
    out.loc[bear_only, "risk_mult"] = (bear.loc[bear_only, "risk_mult"].fillna(1.0) * args.bear_standalone_risk_scale).clip(args.min_risk_mult, args.max_risk_mult or 10.0)
    out.loc[bear_only, "quality_mult"] = (bear.loc[bear_only, "quality_mult"].fillna(1.0) * args.bear_standalone_quality_scale).clip(0.20, args.quality_mult_cap)

    # Final signal columns for the SAFE executor.
    out["signal"] = final_signal
    out["long_signal"] = out["signal"] == 1
    out["short_signal"] = out["signal"] == -1
    return out


def _entry_exit_channel(row: Any, entry_engine: str, side: int) -> bool:
    if entry_engine.startswith("BEAR") and side == -1:
        return bool(getattr(row, "bear_short_exit_channel", False))
    return bool(getattr(row, "momentum_long_exit_channel" if side == 1 else "momentum_short_exit_channel", False))


def run_priority_backtest(df: pd.DataFrame, cfg: ExecConfig) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    capital = cfg.initial_capital
    peak = capital
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
    entry_engine = "NONE"
    last_exit_i = -10**9

    rows = list(df.itertuples())
    idx = df.index

    for i in range(len(rows) - 1):
        row = rows[i]
        ts = idx[i]
        if in_pos:
            high = float(row.high)
            low = float(row.low)
            close = float(row.close)
            atr_value = float(row.atr)
            hold_bars = i - entry_i
            active_stop = stop_price
            current_signal = int(getattr(row, "signal", 0))

            if side == 1:
                max_fav = max(max_fav, high)
                max_adv = min(max_adv, low)
                touched_stop = low <= active_stop
                channel_exit = _entry_exit_channel(row, entry_engine, side)
                opposite = current_signal == -1
                next_stop = max(stop_price, close - cfg.trailing_atr_mult * atr_value)
                locked = protected_stop(first_entry, avg_entry, side, risk_per_coin, max_fav, cfg)
                if locked is not None:
                    next_stop = max(next_stop, locked)
            else:
                max_fav = min(max_fav, low)
                max_adv = max(max_adv, high)
                touched_stop = high >= active_stop
                channel_exit = _entry_exit_channel(row, entry_engine, side)
                opposite = current_signal == 1
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
            elif hold_bars >= cfg.max_hold_bars:
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
                peak = max(peak, capital)
                in_pos = False
                side = 0
                last_exit_i = i
            else:
                stop_price = next_stop

            if in_pos and units < cfg.max_units:
                next_unit_number = units + 1
                trigger_r = (next_unit_number - 1) * cfg.add_every_r
                add_triggered = high >= first_entry + trigger_r * risk_per_coin if side == 1 else low <= first_entry - trigger_r * risk_per_coin
                if add_triggered:
                    add_price = apply_entry_slippage(float(rows[i + 1].open), side, cfg.slippage_pct)
                    add_stop_dist = max(cfg.initial_atr_mult * atr_value, risk_per_coin)
                    add_q = unit_qty(capital, add_price, add_stop_dist, qty, cfg, float(getattr(row, "risk_mult", entry_risk_mult)) * float(getattr(row, "quality_mult", 1.0)))
                    if add_q > 0 and math.isfinite(add_q):
                        total_entry_fee += add_q * add_price * cfg.fee_rate
                        avg_entry = weighted_avg_price(avg_entry, qty, add_price, add_q)
                        qty += add_q
                        units += 1

        if not in_pos and i - last_exit_i >= cfg.cooldown_bars:
            signal = int(getattr(row, "signal", 0))
            if signal != 0:
                next_open = float(rows[i + 1].open)
                entry = apply_entry_slippage(next_open, signal, cfg.slippage_pct)
                atr_value = float(row.atr)
                sl = entry - cfg.initial_atr_mult * atr_value if signal == 1 else entry + cfg.initial_atr_mult * atr_value
                stop_dist = abs(entry - sl)
                entry_risk_mult = float(getattr(row, "risk_mult", 1.0)) * float(getattr(row, "quality_mult", 1.0))
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
                    entry_engine = str(getattr(row, "selected_engine", "UNKNOWN"))

        equity_rows.append({"time": ts, "capital": capital, "drawdown_pct": (peak - capital) / peak if peak > 0 else 0.0})

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
    equity = pd.DataFrame(equity_rows).set_index("time") if equity_rows else pd.DataFrame()
    return trades, equity


def attach_engine_to_trades(trades: list[dict[str, Any]], features: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for trade in trades:
        item = dict(trade)
        signal_time = pd.Timestamp(item["entry_time"]) - pd.Timedelta(hours=4)
        if signal_time in features.index:
            row = features.loc[signal_time]
            item["engine"] = str(row.get("selected_engine", "UNKNOWN"))
            item["engine_priority"] = int(row.get("selected_priority", 0))
            item["momentum_selected"] = bool(row.get("momentum_selected", False))
            item["bear_only"] = bool(row.get("bear_only", False))
        else:
            item["engine"] = "UNKNOWN"
            item["engine_priority"] = 0
            item["momentum_selected"] = False
            item["bear_only"] = False
        out.append(item)
    return out


def write_outputs(trades: list[dict[str, Any]], equity: pd.DataFrame, features: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / f"{STRATEGY_NAME}_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / f"{STRATEGY_NAME}_equity.csv")
    audit_cols = [
        "open", "high", "low", "close", "volume", "atr", "atr_pct", "adx",
        "risk_mult", "quality_mult", "momentum_signal", "bear_signal", "signal",
        "selected_engine", "selected_priority", "momentum_selected", "bear_only",
        "long_signal", "short_signal", "momentum_long_exit_channel", "momentum_short_exit_channel", "bear_short_exit_channel",
    ]
    features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{STRATEGY_NAME}_signal_audit.csv")
    with (out_dir / f"{STRATEGY_NAME}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 92)
    print("ETH LF Portfolio V5 Momentum + Bear Priority Backtest Summary")
    print("=" * 92)
    for k, v in summary.items():
        print(f"{k:>34}: {v}")
    print("-" * 92)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 92 + "\n")


def print_deep_report(trades: list[dict[str, Any]], features: pd.DataFrame, cfg: ExecConfig, out_dir: Path) -> None:
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
    p = argparse.ArgumentParser(description="ETH LF Portfolio V5: Momentum V3 main + Bear V3 standalone supplement.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=today)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--preset", choices=sorted(MOMENTUM_PRESETS), default="turbo")
    p.add_argument("--unit-risk-per-trade", type=float, default=None)
    p.add_argument("--max-total-notional-mult", type=float, default=None)
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--min-risk-mult", type=float, default=0.35)
    p.add_argument("--max-risk-mult", type=float, default=None)
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--disable-short", action="store_true")
    p.add_argument("--bear-preset", choices=sorted(BEAR_PRESETS), default="high")
    p.add_argument("--bear-min-risk-mult", type=float, default=0.25)
    p.add_argument("--bear-standalone-risk-scale", type=float, default=1.0)
    p.add_argument("--bear-standalone-quality-scale", type=float, default=1.0)
    p.add_argument("--disable-bear-standalone", action="store_true")
    p.add_argument("--quality-mult-cap", type=float, default=2.20)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    mom_cfg = make_momentum_config(args)
    bear_cfg = make_bear_config(args)
    exec_cfg = make_exec_config(mom_cfg)
    out_dir = Path(args.out_dir) if args.out_dir else Path(PROJECT_ROOT) / "data/reports/lf" / STRATEGY_NAME / args.preset

    print(f"Loading {args.symbol} 4H: {args.start_date} -> {args.end_date}")
    base = load_data(args.symbol, args.start_date, args.end_date, "4H")
    print(f"Loaded {len(base)} rows: {base.index[0]} -> {base.index[-1]}")
    momentum = build_momentum_features(base, mom_cfg)
    bear = build_bear_features(base, bear_cfg)
    features = select_portfolio_signals(momentum, bear, args)

    print("Signal counts:", {
        "momentum_long": int((features.momentum_signal == 1).sum()),
        "momentum_short": int((features.momentum_signal == -1).sum()),
        "bear_short": int((features.bear_signal == -1).sum()),
        "portfolio_long": int((features.signal == 1).sum()),
        "portfolio_short": int((features.signal == -1).sum()),
        "bear_only": int(features.bear_only.sum()),
    })

    trades, equity = run_priority_backtest(features, exec_cfg)
    trades = attach_engine_to_trades(trades, features)
    summary = summarize(trades, equity, exec_cfg.initial_capital)
    if trades:
        tdf = pd.DataFrame(trades)
        summary["engine_counts"] = tdf["engine"].value_counts().to_dict()
        summary["momentum_trade_count"] = int(tdf.get("momentum_selected", pd.Series(dtype=bool)).sum())
        summary["bear_only_trade_count"] = int(tdf.get("bear_only", pd.Series(dtype=bool)).sum())
    summary["preset"] = args.preset
    summary["bear_preset"] = args.bear_preset
    summary["single_active_position"] = True
    summary["conflict_rule"] = "Momentum V3 first; Bear V3 standalone only when Momentum silent; no V8 fallback; no hedge."
    summary["fee_rate_per_side"] = args.fee_rate

    write_outputs(trades, equity, features, summary, out_dir)
    print_summary(summary, out_dir)
    print_deep_report(trades, features, exec_cfg, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
