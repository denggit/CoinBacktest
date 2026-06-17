#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH HF Compression Breakout Backtest
====================================

放置位置：backtest/eth_hf_compression_breakout_backtest.py

运行示例：
    python backtest/eth_hf_compression_breakout_backtest.py --start-date 2026-06-01 --end-date 2026-06-07

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

STRATEGY_NAME = "eth_hf_compression_breakout"


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

    range_seconds: int = 20 * 60
    context_seconds: int = 6 * 60 * 60
    confirm_seconds: int = 10
    max_range_pct: float = 0.0045
    range_quantile: float = 0.30
    breakout_buffer_pct: float = 0.0003
    max_chase_pct: float = 0.0012
    flow_quantile: float = 0.80
    stop_loss_pct: float = 0.0015
    take_profit_pct: float = 0.0035
    max_hold_seconds: int = 900



def build_features(bars: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """20分钟低波动压缩后，突破 + 10秒主动成交确认。"""
    df = bars.copy()
    range_win = int(cfg.range_seconds)
    confirm_win = int(cfg.confirm_seconds)
    df["range_high"] = df["high"].rolling(range_win, min_periods=range_win).max().shift(1)
    df["range_low"] = df["low"].rolling(range_win, min_periods=range_win).min().shift(1)
    df["range_pct"] = df["range_high"] / df["range_low"] - 1.0
    df["range_q"] = rolling_quantile_shifted(df["range_pct"], cfg.context_seconds, cfg.range_quantile)
    df["compression_ok"] = (df["range_pct"] <= cfg.max_range_pct) & (df["range_pct"] <= df["range_q"])

    df["buy_notional_10s"] = rolling_sum(df["buy_notional"], confirm_win)
    df["sell_notional_10s"] = rolling_sum(df["sell_notional"], confirm_win)
    df["cvd_10s"] = rolling_sum(df["cvd_notional"], confirm_win)
    df["buy_flow_q"] = rolling_quantile_shifted(df["buy_notional_10s"], cfg.context_seconds, cfg.flow_quantile)
    df["sell_flow_q"] = rolling_quantile_shifted(df["sell_notional_10s"], cfg.context_seconds, cfg.flow_quantile)

    long_break = (
        df["compression_ok"]
        & (df["close"] > df["range_high"] * (1 + cfg.breakout_buffer_pct))
        & (df["close"].shift(1) <= df["range_high"])
        & (df["close"] <= df["range_high"] * (1 + cfg.max_chase_pct))
        & (df["buy_notional_10s"] >= df["buy_flow_q"])
        & (df["cvd_10s"] > 0)
    )
    short_break = (
        df["compression_ok"]
        & (df["close"] < df["range_low"] * (1 - cfg.breakout_buffer_pct))
        & (df["close"].shift(1) >= df["range_low"])
        & (df["close"] >= df["range_low"] * (1 - cfg.max_chase_pct))
        & (df["sell_notional_10s"] >= df["sell_flow_q"])
        & (df["cvd_10s"] < 0)
    )
    df["signal"] = 0
    df.loc[long_break, "signal"] = 1
    df.loc[short_break, "signal"] = -1
    df["signal_reason"] = ""
    df.loc[long_break, "signal_reason"] = "COMPRESSION_LONG_BREAKOUT"
    df.loc[short_break, "signal_reason"] = "COMPRESSION_SHORT_BREAKOUT"
    df["signal_level"] = pd.NA
    df.loc[long_break, "signal_level"] = df.loc[long_break, "range_high"]
    df.loc[short_break, "signal_level"] = df.loc[short_break, "range_low"]
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
    p = argparse.ArgumentParser(description="ETH 高频/中高频压缩突破订单流确认策略回测")
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
    p.add_argument("--out-dir", default="data/reports/eth_hf_compression_breakout")
    p.add_argument("--write-full-audit", action="store_true", help="写出 1秒全量特征审计，文件可能很大")
    p.add_argument("--max-range-pct", type=float, default=0.0045)
    p.add_argument("--stop-loss-pct", type=float, default=0.0015)
    p.add_argument("--take-profit-pct", type=float, default=0.0035)
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
        max_range_pct=args.max_range_pct,
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
