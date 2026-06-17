#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH HF Sweep Reclaim Backtest
=============================

放置位置：backtest/eth_hf_sweep_reclaim_backtest.py

运行示例：
    python backtest/eth_hf_sweep_reclaim_backtest.py --start-date 2026-06-01 --end-date 2026-06-07

重要说明：
    - 使用 OKXTickLoader 读取 ETH-USDT-SWAP trades/tick 数据。
    - OKX ETH-USDT-SWAP trades 的 size 是张数；默认 1 张 = 0.1 ETH。
    - 先把 tick 聚合成 1秒 bar，再做信号和撮合。
    - 信号在当前 1秒 bar 收盘后产生，下一秒 open 入场，避免偷看。
    - 同一秒同时触发止损/止盈，按保守原则先算止损。
    - 默认按 taker 入场/出场，手续费和滑点都可用 CLI 覆盖。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_feed.okx_tick_loader import OKXTickLoader  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402
from src.backtest_common.execution import apply_entry_slippage, apply_exit_slippage  # noqa: E402
from src.backtest_common.data import aggregate_trades_to_seconds, load_second_bars, merge_second_bars  # noqa: E402
from src.backtest_common.indicators import rolling_quantile_shifted, rolling_sum  # noqa: E402
from src.backtest_common.reporting import summarize_hf_trades as summarize  # noqa: E402
from src.backtest_common.reporting import emit_hf_platform_report, print_hf_summary, write_hf_outputs  # noqa: E402

DEFAULT_OKX_TRADES_URL_TEMPLATE = "https://www.okx.com/cdn/okex/traderecords/trades/daily/{yyyymmdd}/{symbol}-trades-{date}.zip"

STRATEGY_NAME = "eth_hf_sweep_reclaim"


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "ETH-USDT-SWAP"
    contract_value: float = 0.1
    bar_seconds: int = 1

    initial_capital: float = 1000.0
    risk_per_trade: float = 0.002
    max_notional_mult: float = 3.0
    taker_fee_rate: float = 0.0005
    slippage_pct: float = 0.00005
    cooldown_seconds: int = 30

    local_lookback_seconds: int = 30
    trade_window_seconds: int = 3
    reclaim_window_seconds: int = 20
    context_seconds: int = 300
    flow_quantile: float = 0.80
    min_sweep_depth_pct: float = 0.0008
    max_sweep_depth_pct: float = 0.0040
    reclaim_buffer_pct: float = 0.00005
    stop_loss_pct: float = 0.0018
    take_profit_pct: float = 0.0032
    no_progress_seconds: int = 90
    no_progress_min_mfe_pct: float = 0.0010
    max_hold_seconds: int = 480



def build_features(bars: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """流动性扫单失败 + 主动成交吸收 + reclaim。"""
    df = bars.copy()
    lookback = int(cfg.local_lookback_seconds)
    trade_win = int(cfg.trade_window_seconds)
    df["local_low"] = df["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    df["local_high"] = df["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    df["sell_notional_3s"] = rolling_sum(df["sell_notional"], trade_win)
    df["buy_notional_3s"] = rolling_sum(df["buy_notional"], trade_win)
    df["sell_q"] = rolling_quantile_shifted(df["sell_notional_3s"], cfg.context_seconds, cfg.flow_quantile)
    df["buy_q"] = rolling_quantile_shifted(df["buy_notional_3s"], cfg.context_seconds, cfg.flow_quantile)
    df["signal"] = 0
    df["signal_level"] = pd.NA
    df["signal_extreme"] = pd.NA
    df["signal_reason"] = ""

    active_long: dict[str, Any] | None = None
    active_short: dict[str, Any] | None = None

    for i, row in enumerate(df.itertuples()):
        close = float(row.close)
        high = float(row.high)
        low = float(row.low)
        sig = 0
        reason = ""
        level_for_signal: float | None = None
        extreme_for_signal: float | None = None

        if active_long is not None:
            active_long["extreme"] = min(active_long["extreme"], low)
            level = active_long["level"]
            if i > active_long["expiry_i"] or low < level * (1 - cfg.max_sweep_depth_pct):
                active_long = None
            elif close > level * (1 + cfg.reclaim_buffer_pct):
                sig = 1
                reason = "SWEEP_LOW_RECLAIM"
                level_for_signal = level
                extreme_for_signal = active_long["extreme"]
                active_long = None

        if active_short is not None and sig == 0:
            active_short["extreme"] = max(active_short["extreme"], high)
            level = active_short["level"]
            if i > active_short["expiry_i"] or high > level * (1 + cfg.max_sweep_depth_pct):
                active_short = None
            elif close < level * (1 - cfg.reclaim_buffer_pct):
                sig = -1
                reason = "SWEEP_HIGH_RECLAIM"
                level_for_signal = level
                extreme_for_signal = active_short["extreme"]
                active_short = None

        local_low = getattr(row, "local_low")
        local_high = getattr(row, "local_high")
        sell_q = getattr(row, "sell_q")
        buy_q = getattr(row, "buy_q")

        if pd.notna(local_low) and pd.notna(sell_q):
            level = float(local_low)
            depth = (level - low) / level if level > 0 else 0.0
            sell_burst = float(getattr(row, "sell_notional_3s")) >= float(sell_q)
            if active_long is None and depth >= cfg.min_sweep_depth_pct and depth <= cfg.max_sweep_depth_pct and sell_burst:
                active_long = {"level": level, "extreme": low, "expiry_i": i + int(cfg.reclaim_window_seconds)}

        if pd.notna(local_high) and pd.notna(buy_q):
            level = float(local_high)
            depth = (high - level) / level if level > 0 else 0.0
            buy_burst = float(getattr(row, "buy_notional_3s")) >= float(buy_q)
            if active_short is None and depth >= cfg.min_sweep_depth_pct and depth <= cfg.max_sweep_depth_pct and buy_burst:
                active_short = {"level": level, "extreme": high, "expiry_i": i + int(cfg.reclaim_window_seconds)}

        if sig != 0:
            df.iat[i, df.columns.get_loc("signal")] = sig
            df.iat[i, df.columns.get_loc("signal_level")] = level_for_signal
            df.iat[i, df.columns.get_loc("signal_extreme")] = extreme_for_signal
            df.iat[i, df.columns.get_loc("signal_reason")] = reason
    return df.dropna(subset=["open", "high", "low", "close"]).copy()




def run_backtest(features: pd.DataFrame, cfg: StrategyConfig) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    capital = cfg.initial_capital
    peak = capital
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    in_pos = False
    side = 0
    entry_i = -1
    entry_time = None
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    qty_eth = 0.0
    entry_fee = 0.0
    max_fav = 0.0
    max_adv = 0.0
    last_exit_i = -10**9

    rows = list(features.itertuples())
    idx = features.index

    for i in range(len(rows) - 1):
        row = rows[i]
        ts = idx[i]

        if in_pos:
            high = float(row.high)
            low = float(row.low)
            close = float(row.close)
            hold_seconds = i - entry_i
            exit_now = False
            exit_price = 0.0
            reason = ""

            if side == 1:
                max_fav = max(max_fav, high)
                max_adv = min(max_adv, low)
                if low <= stop_price:
                    exit_now = True
                    exit_price = apply_exit_slippage(stop_price, side, cfg.slippage_pct)
                    reason = "STOP_LOSS"
                elif high >= target_price:
                    exit_now = True
                    exit_price = apply_exit_slippage(target_price, side, cfg.slippage_pct)
                    reason = "TAKE_PROFIT"
                elif cfg.no_progress_seconds and hold_seconds >= cfg.no_progress_seconds:
                    mfe_pct = (max_fav - entry_price) / entry_price
                    if mfe_pct < cfg.no_progress_min_mfe_pct:
                        exit_now = True
                        exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                        reason = "NO_PROGRESS_EXIT"
            else:
                max_fav = min(max_fav, low)
                max_adv = max(max_adv, high)
                if high >= stop_price:
                    exit_now = True
                    exit_price = apply_exit_slippage(stop_price, side, cfg.slippage_pct)
                    reason = "STOP_LOSS"
                elif low <= target_price:
                    exit_now = True
                    exit_price = apply_exit_slippage(target_price, side, cfg.slippage_pct)
                    reason = "TAKE_PROFIT"
                elif cfg.no_progress_seconds and hold_seconds >= cfg.no_progress_seconds:
                    mfe_pct = (entry_price - max_fav) / entry_price
                    if mfe_pct < cfg.no_progress_min_mfe_pct:
                        exit_now = True
                        exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                        reason = "NO_PROGRESS_EXIT"

            if not exit_now and hold_seconds >= cfg.max_hold_seconds:
                exit_now = True
                exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                reason = "MAX_HOLD_EXIT"

            if exit_now:
                exit_fee = qty_eth * exit_price * cfg.taker_fee_rate
                if side == 1:
                    pnl = (exit_price - entry_price) * qty_eth - entry_fee - exit_fee
                    mfe_pct = (max_fav - entry_price) / entry_price
                    mae_pct = (entry_price - max_adv) / entry_price
                else:
                    pnl = (entry_price - exit_price) * qty_eth - entry_fee - exit_fee
                    mfe_pct = (entry_price - max_fav) / entry_price
                    mae_pct = (max_adv - entry_price) / entry_price
                cap_before = capital
                capital += pnl
                peak = max(peak, capital)
                trades.append({
                    "strategy": STRATEGY_NAME,
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "type": "LONG" if side == 1 else "SHORT",
                    "entry": round(entry_price, 6),
                    "exit": round(exit_price, 6),
                    "initial_sl": round(stop_price, 6),
                    "target": round(target_price, 6),
                    "qty_eth": qty_eth,
                    "notional_entry": qty_eth * entry_price,
                    "pnl": pnl,
                    "fee": entry_fee + exit_fee,
                    "capital": capital,
                    "return_pct": pnl / max(cap_before, 1e-12),
                    "mfe_pct": mfe_pct,
                    "mae_pct": mae_pct,
                    "holding_seconds": int(hold_seconds),
                    "note": reason,
                })
                in_pos = False
                side = 0
                last_exit_i = i

        if not in_pos and i - last_exit_i >= int(cfg.cooldown_seconds):
            sig = int(getattr(row, "signal", 0))
            if sig != 0:
                next_open = float(rows[i + 1].open)
                entry = apply_entry_slippage(next_open, sig, cfg.slippage_pct)
                if sig == 1:
                    stop = entry * (1 - cfg.stop_loss_pct)
                    target = entry * (1 + cfg.take_profit_pct)
                else:
                    stop = entry * (1 + cfg.stop_loss_pct)
                    target = entry * (1 - cfg.take_profit_pct)
                risk_per_eth = abs(entry - stop)
                if risk_per_eth > 0 and math.isfinite(risk_per_eth):
                    risk_usdt = capital * cfg.risk_per_trade
                    q = risk_usdt / risk_per_eth
                    q = min(q, (capital * cfg.max_notional_mult) / entry)
                    if q > 0 and math.isfinite(q):
                        in_pos = True
                        side = sig
                        entry_i = i + 1
                        entry_time = idx[i + 1]
                        entry_price = entry
                        stop_price = stop
                        target_price = target
                        qty_eth = q
                        entry_fee = qty_eth * entry_price * cfg.taker_fee_rate
                        max_fav = entry_price
                        max_adv = entry_price

        equity_rows.append({"time": ts, "capital": capital, "drawdown_pct": (peak - capital) / peak if peak > 0 else 0.0})

    if in_pos:
        ts = idx[-1]
        close = float(features.iloc[-1]["close"])
        exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
        exit_fee = qty_eth * exit_price * cfg.taker_fee_rate
        if side == 1:
            pnl = (exit_price - entry_price) * qty_eth - entry_fee - exit_fee
            mfe_pct = (max_fav - entry_price) / entry_price
            mae_pct = (entry_price - max_adv) / entry_price
        else:
            pnl = (entry_price - exit_price) * qty_eth - entry_fee - exit_fee
            mfe_pct = (entry_price - max_fav) / entry_price
            mae_pct = (max_adv - entry_price) / entry_price
        cap_before = capital
        capital += pnl
        trades.append({
            "strategy": STRATEGY_NAME,
            "entry_time": entry_time,
            "exit_time": ts,
            "type": "LONG" if side == 1 else "SHORT",
            "entry": round(entry_price, 6),
            "exit": round(exit_price, 6),
            "initial_sl": round(stop_price, 6),
            "target": round(target_price, 6),
            "qty_eth": qty_eth,
            "notional_entry": qty_eth * entry_price,
            "pnl": pnl,
            "fee": entry_fee + exit_fee,
            "capital": capital,
            "return_pct": pnl / max(cap_before, 1e-12),
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            "holding_seconds": int(len(features) - 1 - entry_i),
            "note": "FORCE_CLOSE_END",
        })

    equity = pd.DataFrame(equity_rows).set_index("time") if equity_rows else pd.DataFrame()
    return trades, equity




def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="ETH 高频流动性扫单回收策略回测")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2026-06-01")
    p.add_argument("--end-date", default=today)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--trades-url-template", default=DEFAULT_OKX_TRADES_URL_TEMPLATE, help="本地 tick db 缺失时，自动用这个 OKX trade zip URL 模板下载并缓存；传空字符串则只读本地")
    p.add_argument("--chunksize", type=int, default=100_000)
    p.add_argument("--contract-value", type=float, default=0.1)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--risk-per-trade", type=float, default=0.002)
    p.add_argument("--max-notional-mult", type=float, default=3.0)
    p.add_argument("--taker-fee-rate", type=float, default=0.0005)
    p.add_argument("--slippage-pct", type=float, default=0.00005)
    p.add_argument("--cooldown-seconds", type=int, default=30)
    p.add_argument("--out-dir", default="data/reports/eth_hf_sweep_reclaim")
    p.add_argument("--write-full-audit", action="store_true", help="写出 1秒全量特征审计，文件可能很大")
    p.add_argument("--min-sweep-depth-pct", type=float, default=0.0008)
    p.add_argument("--max-sweep-depth-pct", type=float, default=0.0040)
    p.add_argument("--stop-loss-pct", type=float, default=0.0018)
    p.add_argument("--take-profit-pct", type=float, default=0.0032)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = StrategyConfig(
        symbol=args.symbol,
        contract_value=args.contract_value,
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        max_notional_mult=args.max_notional_mult,
        taker_fee_rate=args.taker_fee_rate,
        slippage_pct=args.slippage_pct,
        cooldown_seconds=args.cooldown_seconds,
        min_sweep_depth_pct=args.min_sweep_depth_pct,
        max_sweep_depth_pct=args.max_sweep_depth_pct,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
    )
    print(f"Loading tick data: {cfg.symbol} {args.start_date} -> {args.end_date}")
    bars = load_second_bars(
        cfg.symbol,
        args.start_date,
        args.end_date,
        cfg,
        chunksize=args.chunksize,
        trades_url_template=args.trades_url_template,
        data_dir=args.data_dir,
    )
    print(f"Second bars: {len(bars)} rows | {bars.index[0]} -> {bars.index[-1]}")
    features = build_features(bars, cfg)
    signal_count = int((features["signal"] != 0).sum())
    print(f"Signals: {signal_count} | long={int((features.signal == 1).sum())} short={int((features.signal == -1).sum())}")
    trades, equity = run_backtest(features, cfg)
    summary = summarize(trades, equity, cfg.initial_capital, signal_count)
    out_dir = Path(PROJECT_ROOT) / args.out_dir
    emit_hf_platform_report(trades, features, cfg, out_dir, strategy_name=STRATEGY_NAME)
    write_hf_outputs(features, trades, equity, summary, out_dir, write_full_audit=args.write_full_audit, strategy_name=STRATEGY_NAME)
    print_hf_summary(summary, out_dir, strategy_name=STRATEGY_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
