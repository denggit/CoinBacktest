#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH 4H Turtle Pyramid Backtest
==============================

放置位置建议：backtest/lf/eth_4h_turtle_pyramid_backtest.py
运行示例：
    python backtest/lf/eth_4h_turtle_pyramid_backtest.py --start-date 2023-01-01 --end-date 2026-06-17

策略定位：
    低频趋势突破 + 顺势加仓。目标不是高胜率，而是抓 ETH 大趋势时把收益放大。

核心逻辑：
    - 4H close 突破过去 120 根高/低点，下一根 4H open 首仓入场。
    - ATR% 过滤死水和极端行情。
    - 盈利每推进 add_every_atr * ATR 加一层，最多 max_units 层。
    - 加仓只顺势加，不补亏损单。
    - 全仓共用 ATR trailing stop / opposite Donchian exit / max hold。

反偷看：
    - Donchian entry/exit 通道全部 shift(1)，不包含当前 K。
    - 信号当前 4H 收盘确认，下一根 4H open 成交。
    - 加仓触发用当前 K high/low 判断，成交价用下一根 4H open。
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

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "ETH-USDT-SWAP"
    timeframe: str = "4H"

    entry_lookback: int = 120          # 20 天趋势突破
    exit_lookback: int = 55            # 约 9 天反向通道退出
    atr_period: int = 20
    min_atr_pct: float = 0.0040
    max_atr_pct: float = 0.0800
    ema_filter: int = 200

    initial_atr_mult: float = 2.4
    trailing_atr_mult: float = 4.0
    add_every_atr: float = 1.0
    max_units: int = 3
    max_hold_bars: int = 180           # 30 天
    cooldown_bars: int = 4

    initial_capital: float = 1000.0
    unit_risk_per_trade: float = 0.0025  # 每层 0.25%，三层合计理论风险约 0.75%
    max_total_notional_mult: float = 3.0
    fee_rate: float = 0.0005
    slippage_pct: float = 0.0002


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def load_data(symbol: str, start_date: str, end_date: str, timeframe: str) -> pd.DataFrame:
    loader = OKXDataLoader(symbol=symbol, timeframe=timeframe)
    df = loader.fetch_data_by_date_range(start_date, end_date)
    if df.empty:
        raise RuntimeError(f"No data loaded for {symbol} {timeframe} {start_date} -> {end_date}")
    df = df.sort_index().copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close", "volume"])


def build_features(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = atr(out, cfg.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]
    out["atr_ok"] = out["atr_pct"].between(cfg.min_atr_pct, cfg.max_atr_pct)
    out["ema_filter"] = out["close"].ewm(span=cfg.ema_filter, adjust=False, min_periods=cfg.ema_filter).mean()

    out["entry_high"] = out["high"].rolling(cfg.entry_lookback, min_periods=cfg.entry_lookback).max().shift(1)
    out["entry_low"] = out["low"].rolling(cfg.entry_lookback, min_periods=cfg.entry_lookback).min().shift(1)
    out["exit_high"] = out["high"].rolling(cfg.exit_lookback, min_periods=cfg.exit_lookback).max().shift(1)
    out["exit_low"] = out["low"].rolling(cfg.exit_lookback, min_periods=cfg.exit_lookback).min().shift(1)

    out["long_signal"] = (out["close"] > out["entry_high"]) & (out["close"] > out["ema_filter"]) & out["atr_ok"]
    out["short_signal"] = (out["close"] < out["entry_low"]) & (out["close"] < out["ema_filter"]) & out["atr_ok"]
    out["long_exit_channel"] = out["close"] < out["exit_low"]
    out["short_exit_channel"] = out["close"] > out["exit_high"]
    out["signal"] = 0
    out.loc[out["long_signal"], "signal"] = 1
    out.loc[out["short_signal"], "signal"] = -1
    return out.dropna().copy()


def apply_entry_slippage(price: float, side: int, slippage_pct: float) -> float:
    return price * (1 + slippage_pct) if side == 1 else price * (1 - slippage_pct)


def apply_exit_slippage(price: float, side: int, slippage_pct: float) -> float:
    return price * (1 - slippage_pct) if side == 1 else price * (1 + slippage_pct)


def weighted_avg_price(old_price: float, old_qty: float, add_price: float, add_qty: float) -> float:
    total = old_qty + add_qty
    if total <= 0:
        return add_price
    return (old_price * old_qty + add_price * add_qty) / total


def unit_qty(capital: float, entry_price: float, stop_dist: float, current_qty: float, cfg: StrategyConfig) -> float:
    if stop_dist <= 0:
        return 0.0
    risk_qty = capital * cfg.unit_risk_per_trade / stop_dist
    max_total_qty = (capital * cfg.max_total_notional_mult) / entry_price
    remaining_qty = max(0.0, max_total_qty - current_qty)
    return max(0.0, min(risk_qty, remaining_qty))


def run_backtest(df: pd.DataFrame, cfg: StrategyConfig) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    capital = cfg.initial_capital
    peak = capital
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    in_pos = False
    side = 0
    entry_i = -1
    entry_time = None
    avg_entry = 0.0
    first_entry = 0.0
    stop_price = 0.0
    initial_sl = 0.0
    risk_per_coin = 0.0
    qty = 0.0
    total_entry_fee = 0.0
    units = 0
    last_add_price = 0.0
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

            exit_now = False
            reason = ""
            exit_price = 0.0
            if side == 1:
                max_fav = max(max_fav, high)
                max_adv = min(max_adv, low)
                trailing = close - cfg.trailing_atr_mult * atr_value
                stop_price = max(stop_price, trailing)
                if low <= stop_price:
                    exit_now = True
                    exit_price = apply_exit_slippage(stop_price, side, cfg.slippage_pct)
                    reason = "ATR_TRAILING_STOP"
                elif bool(getattr(row, "long_exit_channel")):
                    exit_now = True
                    exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                    reason = "DONCHIAN_EXIT"
                elif bool(getattr(row, "short_signal")):
                    exit_now = True
                    exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                    reason = "OPPOSITE_BREAKOUT"
            else:
                max_fav = min(max_fav, low)
                max_adv = max(max_adv, high)
                trailing = close + cfg.trailing_atr_mult * atr_value
                stop_price = min(stop_price, trailing)
                if high >= stop_price:
                    exit_now = True
                    exit_price = apply_exit_slippage(stop_price, side, cfg.slippage_pct)
                    reason = "ATR_TRAILING_STOP"
                elif bool(getattr(row, "short_exit_channel")):
                    exit_now = True
                    exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                    reason = "DONCHIAN_EXIT"
                elif bool(getattr(row, "long_signal")):
                    exit_now = True
                    exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                    reason = "OPPOSITE_BREAKOUT"

            if not exit_now and hold_bars >= cfg.max_hold_bars:
                exit_now = True
                exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                reason = "MAX_HOLD_EXIT"

            if exit_now:
                exit_fee = qty * exit_price * cfg.fee_rate
                if side == 1:
                    pnl = (exit_price - avg_entry) * qty - total_entry_fee - exit_fee
                    mfe_r = (max_fav - first_entry) / risk_per_coin
                    mae_r = (first_entry - max_adv) / risk_per_coin
                else:
                    pnl = (avg_entry - exit_price) * qty - total_entry_fee - exit_fee
                    mfe_r = (first_entry - max_fav) / risk_per_coin
                    mae_r = (max_adv - first_entry) / risk_per_coin
                cap_before = capital
                capital += pnl
                peak = max(peak, capital)
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "type": "LONG" if side == 1 else "SHORT",
                        "first_entry": first_entry,
                        "avg_entry": avg_entry,
                        "exit": exit_price,
                        "initial_sl": initial_sl,
                        "final_sl": stop_price,
                        "qty": qty,
                        "units": units,
                        "pnl": pnl,
                        "fee": total_entry_fee + exit_fee,
                        "capital": capital,
                        "return_pct": pnl / max(cap_before, 1e-12),
                        "mfe_r": round(float(mfe_r), 4),
                        "mae_r": round(float(mae_r), 4),
                        "sl_pct": round(abs(first_entry - initial_sl) / first_entry * 100, 4),
                        "holding_bars_4h": int(hold_bars),
                        "holding_hours": int(hold_bars * 4),
                        "note": reason,
                    }
                )
                in_pos = False
                side = 0
                last_exit_i = i
            elif units < cfg.max_units:
                # 顺势加仓：只有当前 K 没有触发退出，才允许下一根 open 加仓；同 K 先出后加一律保守忽略。
                add_triggered = False
                if side == 1 and high >= last_add_price + cfg.add_every_atr * atr_value:
                    add_triggered = True
                elif side == -1 and low <= last_add_price - cfg.add_every_atr * atr_value:
                    add_triggered = True

                if add_triggered:
                    add_price = apply_entry_slippage(float(rows[i + 1].open), side, cfg.slippage_pct)
                    add_q = unit_qty(capital, add_price, cfg.initial_atr_mult * atr_value, qty, cfg)
                    if add_q > 0 and math.isfinite(add_q):
                        total_entry_fee += add_q * add_price * cfg.fee_rate
                        avg_entry = weighted_avg_price(avg_entry, qty, add_price, add_q)
                        qty += add_q
                        units += 1
                        last_add_price = add_price

        if not in_pos and i - last_exit_i >= cfg.cooldown_bars:
            signal = int(getattr(row, "signal"))
            if signal != 0:
                next_open = float(rows[i + 1].open)
                entry = apply_entry_slippage(next_open, signal, cfg.slippage_pct)
                atr_value = float(row.atr)
                if signal == 1:
                    sl = entry - cfg.initial_atr_mult * atr_value
                else:
                    sl = entry + cfg.initial_atr_mult * atr_value
                stop_dist = abs(entry - sl)
                q = unit_qty(capital, entry, stop_dist, 0.0, cfg)
                if q > 0 and math.isfinite(q):
                    in_pos = True
                    side = signal
                    entry_i = i + 1
                    entry_time = idx[i + 1]
                    avg_entry = entry
                    first_entry = entry
                    initial_sl = sl
                    stop_price = sl
                    risk_per_coin = stop_dist
                    qty = q
                    total_entry_fee = qty * entry * cfg.fee_rate
                    units = 1
                    last_add_price = entry
                    max_fav = entry
                    max_adv = entry

        equity_rows.append({"time": ts, "capital": capital, "drawdown_pct": (peak - capital) / peak if peak > 0 else 0.0})

    if in_pos:
        ts = idx[-1]
        close = float(df.iloc[-1]["close"])
        exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
        exit_fee = qty * exit_price * cfg.fee_rate
        if side == 1:
            pnl = (exit_price - avg_entry) * qty - total_entry_fee - exit_fee
            mfe_r = (max_fav - first_entry) / risk_per_coin
            mae_r = (first_entry - max_adv) / risk_per_coin
        else:
            pnl = (avg_entry - exit_price) * qty - total_entry_fee - exit_fee
            mfe_r = (first_entry - max_fav) / risk_per_coin
            mae_r = (max_adv - first_entry) / risk_per_coin
        cap_before = capital
        capital += pnl
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": ts,
                "type": "LONG" if side == 1 else "SHORT",
                "first_entry": first_entry,
                "avg_entry": avg_entry,
                "exit": exit_price,
                "initial_sl": initial_sl,
                "final_sl": stop_price,
                "qty": qty,
                "units": units,
                "pnl": pnl,
                "fee": total_entry_fee + exit_fee,
                "capital": capital,
                "return_pct": pnl / max(cap_before, 1e-12),
                "mfe_r": round(float(mfe_r), 4),
                "mae_r": round(float(mae_r), 4),
                "sl_pct": round(abs(first_entry - initial_sl) / first_entry * 100, 4),
                "holding_bars_4h": int(len(df) - 1 - entry_i),
                "holding_hours": int((len(df) - 1 - entry_i) * 4),
                "note": "FORCE_CLOSE_END",
            }
        )

    equity = pd.DataFrame(equity_rows).set_index("time") if equity_rows else pd.DataFrame()
    return trades, equity


def summarize(trades: list[dict[str, Any]], equity: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    if not trades:
        return {"total_trades": 0, "final_capital": initial_capital, "total_return_pct": 0.0}
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["pnl"] > 0]
    losses = tdf[tdf["pnl"] <= 0]
    gross_profit = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(-losses["pnl"].sum()) if not losses.empty else 0.0
    final_capital = float(tdf.iloc[-1]["capital"])
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return {
        "total_trades": int(len(tdf)),
        "long_trades": int((tdf["type"] == "LONG").sum()),
        "short_trades": int((tdf["type"] == "SHORT").sum()),
        "final_capital": round(final_capital, 4),
        "total_return_pct": round((final_capital / initial_capital - 1) * 100, 4),
        "win_rate": round(float((tdf["pnl"] > 0).mean() * 100), 4),
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
        "profit_factor": round(pf, 4) if math.isfinite(pf) else "inf",
        "expectancy_pct": round(float(tdf["return_pct"].mean() * 100), 6),
        "max_drawdown_pct": round(float(equity["drawdown_pct"].max() * 100), 4) if not equity.empty else 0.0,
        "avg_mfe_r": round(float(tdf["mfe_r"].mean()), 4),
        "avg_mae_r": round(float(tdf["mae_r"].mean()), 4),
        "avg_units": round(float(tdf["units"].mean()), 4),
        "avg_holding_hours": round(float(tdf["holding_hours"].mean()), 2),
        "total_fees": round(float(tdf["fee"].sum()), 4),
    }


def write_outputs(trades: list[dict[str, Any]], equity: pd.DataFrame, features: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / "eth_4h_turtle_pyramid_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / "eth_4h_turtle_pyramid_equity.csv")
    audit_cols = [
        "open", "high", "low", "close", "volume", "atr", "atr_pct", "atr_ok", "ema_filter",
        "entry_high", "entry_low", "exit_high", "exit_low", "long_signal", "short_signal", "signal",
    ]
    features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / "eth_4h_turtle_pyramid_signal_audit.csv")
    pd.Series(summary).to_json(out_dir / "eth_4h_turtle_pyramid_summary.json", force_ascii=False, indent=2)


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 78)
    print("ETH 4H Turtle Pyramid Backtest Summary")
    print("=" * 78)
    for k, v in summary.items():
        print(f"{k:>24}: {v}")
    print("-" * 78)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 78 + "\n")




def build_report_trades(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal trade records into the format expected by src.utils.report."""
    report_trades: list[dict[str, Any]] = []
    for trade in trades:
        item = dict(trade)
        # print_full_report expects entry/exit keys. Pyramiding strategy uses avg_entry/first_entry internally.
        if "entry" not in item:
            item["entry"] = item.get("avg_entry", item.get("first_entry", 0.0))
        if "exit" not in item:
            item["exit"] = item.get("exit_price", 0.0)
        report_trades.append(item)
    return report_trades


def print_deep_report(trades: list[dict[str, Any]], features: pd.DataFrame, cfg: StrategyConfig, out_dir) -> None:
    """Print the project's standard full performance report."""
    final_capital = float(trades[-1]["capital"]) if trades else float(cfg.initial_capital)
    total_days = 0.0
    if not features.empty:
        total_days = max((features.index[-1] - features.index[0]).total_seconds() / 86400.0, 1e-9)
    print_full_report(
        trade_history=build_report_trades(trades),
        df=features,
        initial_capital=cfg.initial_capital,
        capital=final_capital,
        strategy_name="ETH_4H_TurtlePyramid_LF",
        total_days=total_days,
        ai_enabled=False,
        symbol=cfg.symbol,
        report_dir=out_dir,
    )

def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="ETH 4H turtle breakout with pyramiding backtest.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=today)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--unit-risk-per-trade", type=float, default=0.0025)
    p.add_argument("--max-total-notional-mult", type=float, default=3.0)
    p.add_argument("--fee-rate", type=float, default=0.0005)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--entry-lookback", type=int, default=120)
    p.add_argument("--exit-lookback", type=int, default=55)
    p.add_argument("--max-units", type=int, default=3)
    p.add_argument("--out-dir", default="data/reports/lf/eth_4h_turtle_pyramid")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = StrategyConfig(
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        unit_risk_per_trade=args.unit_risk_per_trade,
        max_total_notional_mult=args.max_total_notional_mult,
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
        entry_lookback=args.entry_lookback,
        exit_lookback=args.exit_lookback,
        max_units=args.max_units,
    )
    print(f"Loading {cfg.symbol} {cfg.timeframe}: {args.start_date} -> {args.end_date}")
    base = load_data(cfg.symbol, args.start_date, args.end_date, cfg.timeframe)
    print(f"Loaded {len(base)} rows: {base.index[0]} -> {base.index[-1]}")
    features = build_features(base, cfg)
    print(f"Feature rows: {len(features)}")
    print("Signal counts:", {"long": int((features.signal == 1).sum()), "short": int((features.signal == -1).sum())})
    trades, equity = run_backtest(features, cfg)
    summary = summarize(trades, equity, cfg.initial_capital)
    out_dir = Path(PROJECT_ROOT) / args.out_dir
    write_outputs(trades, equity, features, summary, out_dir)
    print_summary(summary, out_dir)
    print_deep_report(trades, features, cfg, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
