#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH Donchian Breakout Backtest
==============================

放置位置建议：backtest/mf/eth_donchian_breakout_backtest.py
运行示例：
    python backtest/mf/eth_donchian_breakout_backtest.py --start-date 2023-01-01 --end-date 2026-06-17

策略逻辑：
    - 使用 1H K 线
    - Donchian breakout 判断趋势突破
    - 不使用 EMA 回调
    - 不使用 VWAP
    - 不使用 TP1 / 分批止盈
    - 全仓使用 ATR 初始止损 + ATR trailing stop

反偷看：
    - 当前 1H close 突破过去 N 根已完成 1H high/low 才生成信号
    - 下一根 1H open 入场
    - Donchian high/low 使用 shift(1)，不包含当前 K 线
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402
from src.backtest_common.execution import apply_entry_slippage, apply_exit_slippage  # noqa: E402
from src.backtest_common.indicators import atr  # noqa: E402
from src.backtest_common.data import load_ohlcv_data as load_data  # noqa: E402
from src.backtest_common.reporting import summarize_r_trades as summarize  # noqa: E402


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "ETH-USDT-SWAP"
    timeframe: str = "1H"

    breakout_lookback: int = 24       # previous 24 hours high/low
    atr_period: int = 14
    initial_atr_mult: float = 2.5
    trailing_atr_mult: float = 3.0

    # Volatility filter: avoid dead chop and abnormal chaos.
    min_atr_pct: float = 0.0025       # 0.25%
    max_atr_pct: float = 0.0400       # 4.00%

    # Optional anti-overtrade cooldown after any exit.
    cooldown_bars: int = 2
    max_hold_bars: int = 24 * 14      # max 14 days on 1H bars

    # Finance
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.002     # 0.20% equity risk per trade
    max_notional_mult: float = 3.0
    fee_rate: float = 0.0005          # per side
    slippage_pct: float = 0.0002      # per entry/exit



def build_features(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = atr(out, cfg.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]

    # Donchian channel excluding current bar.
    out["donchian_high"] = out["high"].rolling(cfg.breakout_lookback, min_periods=cfg.breakout_lookback).max().shift(1)
    out["donchian_low"] = out["low"].rolling(cfg.breakout_lookback, min_periods=cfg.breakout_lookback).min().shift(1)

    out["vol_ok"] = out["atr_pct"].between(cfg.min_atr_pct, cfg.max_atr_pct)
    out["long_signal"] = (out["close"] > out["donchian_high"]) & out["vol_ok"]
    out["short_signal"] = (out["close"] < out["donchian_low"]) & out["vol_ok"]
    out["signal"] = 0
    out.loc[out["long_signal"], "signal"] = 1
    out.loc[out["short_signal"], "signal"] = -1
    return out.dropna().copy()



def build_stop(entry_price: float, side: int, atr_value: float, cfg: StrategyConfig) -> float:
    if side == 1:
        return entry_price - cfg.initial_atr_mult * atr_value
    return entry_price + cfg.initial_atr_mult * atr_value


def run_backtest(df: pd.DataFrame, cfg: StrategyConfig) -> tuple[list[dict[str, Any]], pd.DataFrame]:
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
    initial_stop = 0.0
    risk_per_coin = 0.0
    qty = 0.0
    entry_fee = 0.0
    max_fav = 0.0
    max_adv = 0.0
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

            if side == 1:
                max_fav = max(max_fav, high)
                max_adv = min(max_adv, low)
                trailing = close - cfg.trailing_atr_mult * atr_value
                stop_price = max(stop_price, trailing)
                touched_stop = low <= stop_price
                opposite_breakout = bool(getattr(row, "short_signal"))
            else:
                max_fav = min(max_fav, low)
                max_adv = max(max_adv, high)
                trailing = close + cfg.trailing_atr_mult * atr_value
                stop_price = min(stop_price, trailing)
                touched_stop = high >= stop_price
                opposite_breakout = bool(getattr(row, "long_signal"))

            exit_now = False
            exit_price = 0.0
            reason = ""

            if touched_stop:
                exit_now = True
                exit_price = apply_exit_slippage(stop_price, side, cfg.slippage_pct)
                reason = "ATR_TRAILING_STOP"
            elif opposite_breakout:
                # Exit on opposite Donchian breakout at current close.
                exit_now = True
                exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                reason = "OPPOSITE_BREAKOUT_EXIT"
            elif hold_bars >= cfg.max_hold_bars:
                exit_now = True
                exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                reason = "MAX_HOLD_EXIT"

            if exit_now:
                exit_fee = qty * exit_price * cfg.fee_rate
                if side == 1:
                    pnl = (exit_price - entry_price) * qty - entry_fee - exit_fee
                    mfe_r = (max_fav - entry_price) / risk_per_coin
                    mae_r = (entry_price - max_adv) / risk_per_coin
                else:
                    pnl = (entry_price - exit_price) * qty - entry_fee - exit_fee
                    mfe_r = (entry_price - max_fav) / risk_per_coin
                    mae_r = (max_adv - entry_price) / risk_per_coin
                cap_before = capital
                capital += pnl
                peak = max(peak, capital)
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "type": "LONG" if side == 1 else "SHORT",
                        "entry": entry_price,
                        "exit": exit_price,
                        "initial_sl": initial_stop,
                        "final_sl": stop_price,
                        "qty": qty,
                        "pnl": pnl,
                        "fee": entry_fee + exit_fee,
                        "capital": capital,
                        "return_pct": pnl / max(cap_before, 1e-12),
                        "mfe_r": round(float(mfe_r), 4),
                        "mae_r": round(float(mae_r), 4),
                        "sl_pct": round(abs(entry_price - initial_stop) / entry_price * 100, 4),
                        "holding_bars_1h": int(hold_bars),
                        "holding_hours": int(hold_bars),
                        "note": reason,
                    }
                )
                in_pos = False
                side = 0
                last_exit_i = i

        if not in_pos and i - last_exit_i >= cfg.cooldown_bars:
            signal = int(getattr(row, "signal"))
            if signal != 0:
                next_open = float(rows[i + 1].open)
                ep = apply_entry_slippage(next_open, signal, cfg.slippage_pct)
                atr_value = float(row.atr)
                sl = build_stop(ep, signal, atr_value, cfg)
                stop_dist = abs(ep - sl)
                if stop_dist > 0:
                    risk_usdt = capital * cfg.risk_per_trade
                    q = risk_usdt / stop_dist
                    q = min(q, (capital * cfg.max_notional_mult) / ep)
                    if q > 0 and math.isfinite(q):
                        in_pos = True
                        side = signal
                        entry_i = i + 1
                        entry_time = idx[i + 1]
                        entry_price = ep
                        stop_price = sl
                        initial_stop = sl
                        risk_per_coin = stop_dist
                        qty = q
                        entry_fee = qty * entry_price * cfg.fee_rate
                        max_fav = entry_price
                        max_adv = entry_price

        equity_rows.append({"time": ts, "capital": capital, "drawdown_pct": (peak - capital) / peak if peak > 0 else 0.0})

    if in_pos:
        ts = idx[-1]
        close = float(df.iloc[-1]["close"])
        exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
        exit_fee = qty * exit_price * cfg.fee_rate
        if side == 1:
            pnl = (exit_price - entry_price) * qty - entry_fee - exit_fee
            mfe_r = (max_fav - entry_price) / risk_per_coin
            mae_r = (entry_price - max_adv) / risk_per_coin
        else:
            pnl = (entry_price - exit_price) * qty - entry_fee - exit_fee
            mfe_r = (entry_price - max_fav) / risk_per_coin
            mae_r = (max_adv - entry_price) / risk_per_coin
        cap_before = capital
        capital += pnl
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": ts,
                "type": "LONG" if side == 1 else "SHORT",
                "entry": entry_price,
                "exit": exit_price,
                "initial_sl": initial_stop,
                "final_sl": stop_price,
                "qty": qty,
                "pnl": pnl,
                "fee": entry_fee + exit_fee,
                "capital": capital,
                "return_pct": pnl / max(cap_before, 1e-12),
                "mfe_r": round(float(mfe_r), 4),
                "mae_r": round(float(mae_r), 4),
                "sl_pct": round(abs(entry_price - initial_stop) / entry_price * 100, 4),
                "holding_bars_1h": int(len(df) - 1 - entry_i),
                "holding_hours": int(len(df) - 1 - entry_i),
                "note": "FORCE_CLOSE_END",
            }
        )

    equity = pd.DataFrame(equity_rows).set_index("time") if equity_rows else pd.DataFrame()
    return trades, equity



def write_outputs(trades: list[dict[str, Any]], equity: pd.DataFrame, features: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / "eth_donchian_breakout_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / "eth_donchian_breakout_equity.csv")
    cols = ["open", "high", "low", "close", "volume", "atr", "atr_pct", "donchian_high", "donchian_low", "vol_ok", "long_signal", "short_signal", "signal"]
    features[[c for c in cols if c in features.columns]].to_csv(out_dir / "eth_donchian_breakout_signal_audit.csv")
    pd.Series(summary).to_json(out_dir / "eth_donchian_breakout_summary.json", force_ascii=False, indent=2)


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 72)
    print("ETH Donchian Breakout Backtest Summary")
    print("=" * 72)
    for k, v in summary.items():
        print(f"{k:>24}: {v}")
    print("-" * 72)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 72 + "\n")


def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="ETH 1H Donchian breakout + ATR trailing backtest.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=today)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--risk-per-trade", type=float, default=0.002)
    p.add_argument("--fee-rate", type=float, default=0.0005)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--breakout-lookback", type=int, default=24)
    p.add_argument("--initial-atr-mult", type=float, default=2.5)
    p.add_argument("--trailing-atr-mult", type=float, default=3.0)
    p.add_argument("--out-dir", default="data/reports/mf/eth_donchian_breakout")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = StrategyConfig(
        symbol=args.symbol,
        breakout_lookback=args.breakout_lookback,
        initial_atr_mult=args.initial_atr_mult,
        trailing_atr_mult=args.trailing_atr_mult,
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
    )
    print(f"Loading {cfg.symbol} {cfg.timeframe}: {args.start_date} -> {args.end_date}")
    base = load_data(cfg.symbol, args.start_date, args.end_date, cfg.timeframe)
    print(f"Loaded {len(base)} rows: {base.index[0]} -> {base.index[-1]}")
    features = build_features(base, cfg)
    print(f"Feature rows: {len(features)}")
    print("Signal counts:", {"long": int((features.signal == 1).sum()), "short": int((features.signal == -1).sum())})
    trades, equity = run_backtest(features, cfg)
    summary = summarize(trades, equity, cfg.initial_capital)
    out_dir = PROJECT_ROOT / args.out_dir
    if trades:
        total_days = max((features.index[-1] - features.index[0]).total_seconds() / 86400, 1e-9)
        print_full_report(
            trade_history=trades,
            df=features,
            initial_capital=cfg.initial_capital,
            capital=float(trades[-1]["capital"]),
            strategy_name="ETH_MF_DonchianBreakout_1H",
            total_days=total_days,
            ai_enabled=False,
            symbol=cfg.symbol,
            report_dir=out_dir,
        )

    write_outputs(trades, equity, features, summary, out_dir)
    print_summary(summary, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
