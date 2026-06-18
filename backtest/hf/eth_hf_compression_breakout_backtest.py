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

from src.backtest_common.execution import apply_entry_slippage, apply_exit_slippage  # noqa: E402
from src.backtest_common.indicators import rolling_quantile_shifted, rolling_sum  # noqa: E402
from src.backtest_common.hf_streaming import HFBacktestState, iter_second_bar_windows_by_day  # noqa: E402
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





def _elapsed_seconds(start: Any, end: Any) -> int:
    if start is None or end is None:
        return 0
    return max(0, int((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()))


def _cooldown_ok(state: HFBacktestState, ts: pd.Timestamp, cfg: StrategyConfig) -> bool:
    if state.last_exit_time is None:
        return True
    return _elapsed_seconds(state.last_exit_time, ts) >= int(cfg.cooldown_seconds)


def _close_position(state: HFBacktestState, ts: pd.Timestamp, exit_price: float, reason: str, cfg: StrategyConfig) -> None:
    exit_fee = state.qty_eth * exit_price * cfg.taker_fee_rate
    if state.side == 1:
        pnl = (exit_price - state.entry_price) * state.qty_eth - state.entry_fee - exit_fee
        mfe_pct = (state.max_fav - state.entry_price) / state.entry_price
        mae_pct = (state.entry_price - state.max_adv) / state.entry_price
    else:
        pnl = (state.entry_price - exit_price) * state.qty_eth - state.entry_fee - exit_fee
        mfe_pct = (state.entry_price - state.max_fav) / state.entry_price
        mae_pct = (state.max_adv - state.entry_price) / state.entry_price

    cap_before = state.capital
    state.capital += pnl
    state.peak = max(state.peak, state.capital)
    state.trades.append({
        "strategy": STRATEGY_NAME,
        "entry_time": state.entry_time,
        "exit_time": ts,
        "type": "LONG" if state.side == 1 else "SHORT",
        "entry": round(state.entry_price, 6),
        "exit": round(exit_price, 6),
        "initial_sl": round(state.stop_price, 6),
        "target": round(state.target_price, 6),
        "qty_eth": state.qty_eth,
        "notional_entry": state.qty_eth * state.entry_price,
        "pnl": pnl,
        "fee": state.entry_fee + exit_fee,
        "capital": state.capital,
        "return_pct": pnl / max(cap_before, 1e-12),
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "holding_seconds": _elapsed_seconds(state.entry_time, ts),
        "note": reason,
    })
    state.in_pos = False
    state.side = 0
    state.entry_time = None
    state.entry_price = 0.0
    state.stop_price = 0.0
    state.target_price = 0.0
    state.qty_eth = 0.0
    state.entry_fee = 0.0
    state.max_fav = 0.0
    state.max_adv = 0.0
    state.last_exit_time = pd.Timestamp(ts)


def _try_open_position(state: HFBacktestState, sig: int, next_ts: pd.Timestamp, next_open: float, cfg: StrategyConfig) -> None:
    entry = apply_entry_slippage(next_open, sig, cfg.slippage_pct)
    if sig == 1:
        stop = entry * (1 - cfg.stop_loss_pct)
        target = entry * (1 + cfg.take_profit_pct)
    else:
        stop = entry * (1 + cfg.stop_loss_pct)
        target = entry * (1 - cfg.take_profit_pct)

    risk_per_eth = abs(entry - stop)
    if risk_per_eth <= 0 or not math.isfinite(risk_per_eth):
        return
    risk_usdt = state.capital * cfg.risk_per_trade
    q = risk_usdt / risk_per_eth
    q = min(q, (state.capital * cfg.max_notional_mult) / entry)
    if q <= 0 or not math.isfinite(q):
        return

    state.in_pos = True
    state.side = sig
    state.entry_time = pd.Timestamp(next_ts)
    state.entry_price = entry
    state.stop_price = stop
    state.target_price = target
    state.qty_eth = q
    state.entry_fee = state.qty_eth * state.entry_price * cfg.taker_fee_rate
    state.max_fav = entry
    state.max_adv = entry


def run_backtest_chunk(features: pd.DataFrame, cfg: StrategyConfig, state: HFBacktestState) -> HFBacktestState:
    """Run one streaming feature window and carry state across calls.

    The loop intentionally uses len(rows) - 1 because signals are filled at the
    next bar open.  The final row is left pending for the next streaming window.
    """
    if features.empty or len(features) < 2:
        return state

    features = features.sort_index()
    rows = list(features.itertuples())
    idx = features.index

    for i in range(len(rows) - 1):
        row = rows[i]
        ts = pd.Timestamp(idx[i])
        close = float(row.close)
        state.last_ts = ts
        state.last_close = close

        if state.in_pos:
            high = float(row.high)
            low = float(row.low)
            hold_seconds = _elapsed_seconds(state.entry_time, ts)
            exit_now = False
            exit_price = 0.0
            reason = ""

            if state.side == 1:
                state.max_fav = max(state.max_fav, high)
                state.max_adv = min(state.max_adv, low)
                if low <= state.stop_price:
                    exit_now = True
                    exit_price = apply_exit_slippage(state.stop_price, state.side, cfg.slippage_pct)
                    reason = "STOP_LOSS"
                elif high >= state.target_price:
                    exit_now = True
                    exit_price = apply_exit_slippage(state.target_price, state.side, cfg.slippage_pct)
                    reason = "TAKE_PROFIT"

            else:
                state.max_fav = min(state.max_fav, low)
                state.max_adv = max(state.max_adv, high)
                if high >= state.stop_price:
                    exit_now = True
                    exit_price = apply_exit_slippage(state.stop_price, state.side, cfg.slippage_pct)
                    reason = "STOP_LOSS"
                elif low <= state.target_price:
                    exit_now = True
                    exit_price = apply_exit_slippage(state.target_price, state.side, cfg.slippage_pct)
                    reason = "TAKE_PROFIT"


            if not exit_now and hold_seconds >= cfg.max_hold_seconds:
                exit_now = True
                exit_price = apply_exit_slippage(close, state.side, cfg.slippage_pct)
                reason = "MAX_HOLD_EXIT"

            if exit_now:
                _close_position(state, ts, exit_price, reason, cfg)

        if not state.in_pos and _cooldown_ok(state, ts, cfg):
            sig = int(getattr(row, "signal", 0))
            if sig != 0:
                next_open = float(rows[i + 1].open)
                _try_open_position(state, sig, pd.Timestamp(idx[i + 1]), next_open, cfg)

        state.record_equity(ts)

    return state


def finalize_backtest(state: HFBacktestState, cfg: StrategyConfig) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Force-close any remaining open position at the last processed close."""
    if state.in_pos and state.last_ts is not None and state.last_close > 0:
        exit_price = apply_exit_slippage(state.last_close, state.side, cfg.slippage_pct)
        _close_position(state, pd.Timestamp(state.last_ts), exit_price, "FORCE_CLOSE_END", cfg)
        state.record_equity(pd.Timestamp(state.last_ts))
    return state.to_result()


def run_backtest(features: pd.DataFrame, cfg: StrategyConfig) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Compatibility wrapper for short test ranges."""
    state = HFBacktestState.initial(cfg.initial_capital)
    state = run_backtest_chunk(features, cfg, state)
    return finalize_backtest(state, cfg)


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
    print(f"Streaming tick data by day: {cfg.symbol} {args.start_date} -> {args.end_date}")
    state = HFBacktestState.initial(cfg.initial_capital)
    signal_count = 0
    long_signal_count = 0
    short_signal_count = 0
    processed_rows = 0
    signal_feature_parts: list[pd.DataFrame] = []
    full_feature_parts: list[pd.DataFrame] = [] if args.write_full_audit else []
    report_rows: list[dict[str, Any]] = []

    for window in iter_second_bar_windows_by_day(
        cfg.symbol,
        args.start_date,
        args.end_date,
        cfg,
        chunksize=args.chunksize,
        trades_url_template=args.trades_url_template,
        data_dir=args.data_dir,
        progress=True,
    ):
        features = build_features(window.work, cfg)
        current_features = window.slice_current(features)
        exec_features = window.slice_execution(features)
        if current_features.empty:
            continue

        processed_rows += int(len(current_features))
        sig_mask = current_features["signal"] != 0
        signal_count += int(sig_mask.sum())
        long_signal_count += int((current_features["signal"] == 1).sum())
        short_signal_count += int((current_features["signal"] == -1).sum())
        if sig_mask.any():
            signal_feature_parts.append(current_features.loc[sig_mask].copy())
        if args.write_full_audit:
            full_feature_parts.append(current_features.copy())

        report_rows.append({"time": window.current_start, "close": float(current_features.iloc[0]["close"])})
        report_rows.append({"time": window.current_end, "close": float(current_features.iloc[-1]["close"])})

        if len(exec_features) >= 2:
            state = run_backtest_chunk(exec_features, cfg, state)

        print(
            f"streamed day={window.meta.get('day')} "
            f"rows={len(current_features)} warmup_rows={window.warmup_rows} "
            f"signals={int(sig_mask.sum())} capital={state.capital:.2f}"
        )

    trades, equity = finalize_backtest(state, cfg)
    if processed_rows <= 0:
        raise RuntimeError(f"No tick data loaded for {cfg.symbol} {args.start_date} -> {args.end_date}")

    print(
        f"Streamed second bars: {processed_rows} current rows | "
        f"signals={signal_count} long={long_signal_count} short={short_signal_count}"
    )

    if args.write_full_audit and full_feature_parts:
        output_features = pd.concat(full_feature_parts).sort_index()
    elif signal_feature_parts:
        output_features = pd.concat(signal_feature_parts).sort_index()
    else:
        output_features = pd.DataFrame(columns=["signal"])

    report_features = pd.DataFrame(report_rows).drop_duplicates(subset=["time"]).set_index("time").sort_index() if report_rows else output_features
    summary = summarize(trades, equity, cfg.initial_capital, signal_count)
    out_dir = Path(PROJECT_ROOT) / args.out_dir
    emit_hf_platform_report(trades, report_features, cfg, out_dir, strategy_name=STRATEGY_NAME)
    write_hf_outputs(output_features, trades, equity, summary, out_dir, write_full_audit=args.write_full_audit, strategy_name=STRATEGY_NAME)
    print_hf_summary(summary, out_dir, strategy_name=STRATEGY_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
