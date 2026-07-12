#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH 1D + 4H Trend Rider V3 Backtest
====================================

放置位置建议：backtest/lf/eth_1d_4h_trend_rider_v3_backtest.py
运行示例：
    python backtest/lf/eth_1d_4h_trend_rider_v3_backtest.py --start-date 2023-01-01 --end-date 2026-06-15

策略定位：
    低频趋势骑乘 V3。
    目标不再是等少数 Turtle 大突破，而是提高年度稳定性：
        1) 1D 快速趋势 regime 判断大方向；
        2) 4H 顺势突破/趋势延续入场；
        3) 多空都允许，但空头门槛更高；
        4) 持仓周期比中频更长，交易次数比 V2 更多；
        5) 保留 R 保护、顺势加仓、趋势止损。

反偷看：
    - 4H breakout / exit 通道全部 shift(1)。
    - 1D regime 使用日线收盘后 shift(1)，再映射到 4H。
    - 4H 信号收盘确认，下一根 4H open 入场/加仓。
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backtest_common.data import load_ohlcv_data as load_data  # noqa: E402
from src.backtest_common.execution import apply_entry_slippage, apply_exit_slippage  # noqa: E402
from src.backtest_common.indicators import adx, atr, ema, resample_ohlcv  # noqa: E402
from src.backtest_common.reporting import build_report_trades  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402

STRATEGY_NAME = "eth_1d_4h_trend_rider_v3"
REPORT_STRATEGY_NAME = "ETH_1D_4H_TrendRider_V3_LF"


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "ETH-USDT-SWAP"
    timeframe: str = "4H"

    # 4H trend riding / breakout / exit
    entry_lookback: int = 40             # ~6.7 days on 4H bars
    exit_lookback: int = 36              # 6 days on 4H bars
    atr_period: int = 20
    adx_period: int = 14
    min_adx_long: float = 8.0
    min_adx_short: float = 18.0
    min_atr_pct: float = 0.0030
    max_atr_pct: float = 0.0800

    # 1D fast regime filter. Short EMA intentionally faster than V2 to avoid waiting years for one trade.
    d1_ema_fast: int = 8
    d1_ema_slow: int = 30
    d1_slope_lookback: int = 10
    bull_slope_min: float = -0.0300
    short_slope_max: float = -0.0030

    # Risk / stop / pyramid
    initial_atr_mult: float = 2.5
    trailing_atr_mult: float = 4.5
    unit_risk_per_trade: float = 0.0060  # each unit risks 0.60% before protection
    max_units: int = 3
    add_every_r: float = 1.0
    max_total_notional_mult: float = 5.0

    # Profit protection, based on initial R
    breakeven_after_r: float = 1.0
    breakeven_lock_r: float = 0.05
    lock_after_2r: float = 2.0
    lock_2r: float = 0.50
    lock_after_3r: float = 3.0
    lock_3r: float = 1.00

    # Anti-chop
    no_progress_bars: int = 10000        # disabled by default; trend rider should not exit just because it moves slowly
    no_progress_min_r: float = 0.0
    max_hold_bars: int = 360             # 60 days
    cooldown_bars: int = 8
    enable_short: bool = True

    # Finance
    initial_capital: float = 1000.0
    fee_rate: float = 0.0005
    slippage_pct: float = 0.0002


def build_daily_regime(base_4h: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    d1 = resample_ohlcv(base_4h, "1D")
    d1["d1_ema_fast"] = ema(d1["close"], cfg.d1_ema_fast)
    d1["d1_ema_slow"] = ema(d1["close"], cfg.d1_ema_slow)
    d1["d1_slow_slope"] = d1["d1_ema_slow"] / d1["d1_ema_slow"].shift(cfg.d1_slope_lookback) - 1.0
    d1["d1_bull"] = (
        (d1["close"] > d1["d1_ema_slow"])
        & (d1["d1_ema_fast"] > d1["d1_ema_slow"] * 0.995)
        & (d1["d1_slow_slope"] > cfg.bull_slope_min)
    )
    d1["d1_bear"] = (
        (d1["close"] < d1["d1_ema_slow"])
        & (d1["d1_ema_fast"] < d1["d1_ema_slow"])
        & (d1["d1_slow_slope"] <= cfg.short_slope_max)
    )

    # Use yesterday's completed daily regime on 4H bars.
    for col in ["d1_ema_fast", "d1_ema_slow", "d1_slow_slope", "d1_bull", "d1_bear"]:
        d1[f"{col}_available"] = d1[col].shift(1)
    return d1


def build_features(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    d1 = build_daily_regime(out, cfg)
    out = out.join(
        d1[["d1_ema_fast_available", "d1_ema_slow_available", "d1_slow_slope_available", "d1_bull_available", "d1_bear_available"]]
        .reindex(out.index, method="ffill")
    )
    out = out.rename(
        columns={
            "d1_ema_fast_available": "d1_ema_fast",
            "d1_ema_slow_available": "d1_ema_slow",
            "d1_slow_slope_available": "d1_slow_slope",
            "d1_bull_available": "d1_bull",
            "d1_bear_available": "d1_bear",
        }
    )

    out["atr"] = atr(out, cfg.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]
    out["atr_ok"] = out["atr_pct"].between(cfg.min_atr_pct, cfg.max_atr_pct)
    out["adx"] = adx(out, cfg.adx_period)
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["ema89"] = ema(out["close"], 89)
    out["ema100"] = ema(out["close"], 100)
    out["ema200"] = ema(out["close"], 200)

    out["entry_high"] = out["high"].rolling(cfg.entry_lookback, min_periods=cfg.entry_lookback).max().shift(1)
    out["entry_low"] = out["low"].rolling(cfg.entry_lookback, min_periods=cfg.entry_lookback).min().shift(1)
    out["exit_high"] = out["high"].rolling(cfg.exit_lookback, min_periods=cfg.exit_lookback).max().shift(1)
    out["exit_low"] = out["low"].rolling(cfg.exit_lookback, min_periods=cfg.exit_lookback).min().shift(1)

    d1_bull = out["d1_bull"].astype("boolean").fillna(False).astype(bool)
    d1_bear = out["d1_bear"].astype("boolean").fillna(False).astype(bool)
    long_filter = d1_bull & out["atr_ok"] & (out["adx"] >= cfg.min_adx_long)
    short_filter = d1_bear & out["atr_ok"] & (out["adx"] >= cfg.min_adx_short) & cfg.enable_short

    # Two entry styles:
    # 1) breakout confirms expansion;
    # 2) trend candle keeps the strategy on the trend instead of waiting for one huge Donchian break.
    breakout_long = (out["close"] > out["entry_high"]) & (out["close"] > out["ema100"])
    trend_long = (out["close"] > out["ema50"]) & (out["ema20"] > out["ema50"]) & (out["close"] > out["open"])
    breakout_short = (out["close"] < out["entry_low"]) & (out["close"] < out["ema100"])
    trend_short = (out["close"] < out["ema50"]) & (out["ema20"] < out["ema50"]) & (out["close"] < out["open"])

    out["long_signal"] = long_filter & (breakout_long | trend_long)
    out["short_signal"] = short_filter & (breakout_short | trend_short)

    out["long_exit_channel"] = (
        (out["close"] < out["exit_low"])
        | ((out["close"] < out["ema89"]) & (out["ema20"] < out["ema50"]))
        | ((~d1_bull) & (out["close"] < out["ema50"]))
    )
    out["short_exit_channel"] = (
        (out["close"] > out["exit_high"])
        | ((out["close"] > out["ema89"]) & (out["ema20"] > out["ema50"]))
        | ((~d1_bear) & (out["close"] > out["ema50"]))
    )
    out["signal"] = 0
    out.loc[out["long_signal"], "signal"] = 1
    out.loc[out["short_signal"], "signal"] = -1
    return out.dropna().copy()

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


def protected_stop(first_entry: float, side: int, risk_per_coin: float, max_fav: float, cfg: StrategyConfig) -> float | None:
    if risk_per_coin <= 0:
        return None
    fav_r = (max_fav - first_entry) / risk_per_coin if side == 1 else (first_entry - max_fav) / risk_per_coin
    lock_r: float | None = None
    if fav_r >= cfg.lock_after_3r:
        lock_r = cfg.lock_3r
    elif fav_r >= cfg.lock_after_2r:
        lock_r = cfg.lock_2r
    elif fav_r >= cfg.breakeven_after_r:
        lock_r = cfg.breakeven_lock_r
    if lock_r is None:
        return None
    return first_entry + side * lock_r * risk_per_coin


def close_trade(
    *,
    trades: list[dict[str, Any]],
    capital: float,
    side: int,
    entry_time: Any,
    exit_time: Any,
    first_entry: float,
    avg_entry: float,
    exit_price: float,
    initial_sl: float,
    stop_price: float,
    qty: float,
    units: int,
    total_entry_fee: float,
    fee_rate: float,
    max_fav: float,
    max_adv: float,
    risk_per_coin: float,
    holding_bars: int,
    reason: str,
) -> float:
    exit_fee = qty * exit_price * fee_rate
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
            "exit_time": exit_time,
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
            "holding_bars_4h": int(holding_bars),
            "holding_hours": int(holding_bars * 4),
            "note": reason,
        }
    )
    return capital


def run_backtest(df: pd.DataFrame, cfg: StrategyConfig) -> tuple[list[dict[str, Any]], pd.DataFrame]:
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
                stop_price = max(stop_price, close - cfg.trailing_atr_mult * atr_value)
                locked = protected_stop(first_entry, side, risk_per_coin, max_fav, cfg)
                if locked is not None:
                    stop_price = max(stop_price, locked)
                touched_stop = low <= stop_price
                channel_exit = bool(getattr(row, "long_exit_channel"))
                opposite = bool(getattr(row, "short_signal"))
            else:
                max_fav = min(max_fav, low)
                max_adv = max(max_adv, high)
                stop_price = min(stop_price, close + cfg.trailing_atr_mult * atr_value)
                locked = protected_stop(first_entry, side, risk_per_coin, max_fav, cfg)
                if locked is not None:
                    stop_price = min(stop_price, locked)
                touched_stop = high >= stop_price
                channel_exit = bool(getattr(row, "short_exit_channel"))
                opposite = bool(getattr(row, "long_signal"))

            exit_now = False
            reason = ""
            exit_price = 0.0
            if touched_stop:
                exit_now = True
                exit_price = apply_exit_slippage(stop_price, side, cfg.slippage_pct)
                reason = "PROTECTED_TRAILING_STOP"
            elif channel_exit:
                exit_now = True
                exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                reason = "DONCHIAN_EXIT"
            elif opposite:
                exit_now = True
                exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                reason = "OPPOSITE_BREAKOUT"
            elif hold_bars >= cfg.no_progress_bars:
                fav_r = (max_fav - first_entry) / risk_per_coin if side == 1 else (first_entry - max_fav) / risk_per_coin
                if fav_r < cfg.no_progress_min_r:
                    exit_now = True
                    exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                    reason = "NO_PROGRESS_EXIT"
            if not exit_now and hold_bars >= cfg.max_hold_bars:
                exit_now = True
                exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                reason = "MAX_HOLD_EXIT"

            if exit_now:
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
                    holding_bars=hold_bars,
                    reason=reason,
                )
                peak = max(peak, capital)
                in_pos = False
                side = 0
                last_exit_i = i
            elif units < cfg.max_units:
                # Add on R-based progress. Same bar exit wins over add, so add is only checked after exit logic.
                next_unit_number = units + 1
                trigger_r = (next_unit_number - 1) * cfg.add_every_r
                if side == 1:
                    add_triggered = high >= first_entry + trigger_r * risk_per_coin
                else:
                    add_triggered = low <= first_entry - trigger_r * risk_per_coin
                if add_triggered:
                    add_price = apply_entry_slippage(float(rows[i + 1].open), side, cfg.slippage_pct)
                    add_stop_dist = max(cfg.initial_atr_mult * atr_value, risk_per_coin)
                    add_q = unit_qty(capital, add_price, add_stop_dist, qty, cfg)
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
                q = unit_qty(capital, entry, stop_dist, 0.0, cfg)
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
    pd.DataFrame(trades).to_csv(out_dir / f"{STRATEGY_NAME}_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / f"{STRATEGY_NAME}_equity.csv")
    audit_cols = [
        "open", "high", "low", "close", "volume", "atr", "atr_pct", "atr_ok", "adx",
        "ema20", "ema50", "ema89", "ema100", "ema200",
        "d1_ema_fast", "d1_ema_slow", "d1_slow_slope", "d1_bull", "d1_bear",
        "entry_high", "entry_low", "exit_high", "exit_low", "long_signal", "short_signal", "signal",
    ]
    features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{STRATEGY_NAME}_signal_audit.csv")
    with (out_dir / f"{STRATEGY_NAME}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 86)
    print("ETH 1D+4H Trend Rider V3 Backtest Summary")
    print("=" * 86)
    for k, v in summary.items():
        print(f"{k:>28}: {v}")
    print("-" * 86)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 86 + "\n")


def print_deep_report(trades: list[dict[str, Any]], features: pd.DataFrame, cfg: StrategyConfig, out_dir: Path) -> None:
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
    p = argparse.ArgumentParser(description="ETH 1D+4H Trend Rider V3 with daily regime, 4H trend entries, R-lock protection.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=today)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--unit-risk-per-trade", type=float, default=0.006)
    p.add_argument("--max-total-notional-mult", type=float, default=5.0)
    p.add_argument("--fee-rate", type=float, default=0.0005)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--entry-lookback", type=int, default=40)
    p.add_argument("--exit-lookback", type=int, default=36)
    p.add_argument("--max-units", type=int, default=3)
    p.add_argument("--disable-short", action="store_true")
    p.add_argument("--out-dir", default="data/reports/lf/eth_1d_4h_trend_rider_v3")
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
        enable_short=not args.disable_short,
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
