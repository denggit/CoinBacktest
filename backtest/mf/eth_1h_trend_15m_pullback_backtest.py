#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH 1H Trend + 15M Pullback Backtest
====================================

放置位置：backtest/mf/eth_1h_trend_15m_pullback_backtest.py
运行示例：
    python backtest/mf/eth_1h_trend_15m_pullback_backtest.py --start-date 2023-07-01 --end-date 2026-06-17

策略定位：
    中频顺势策略。
    1H 判断 ETH 大方向 regime，15M 等回踩结束后顺势入场。

核心逻辑：
    - 只用 15M K 线，并在本文件内重采样得到 1H 特征。
    - 1H close/EMA50/EMA200/ADX/ATR_pct 过滤趋势环境。
    - 15M 回踩 EMA20/EMA50 后重新收回 EMA20 才入场。
    - 下一根 15M open 入场，避免偷看。
    - ATR + 最近 6 根 15M 高低点做初始止损。
    - TP1/TP2 分批止盈，尾仓使用 ATR trailing。

反偷看：
    - 1H 特征在映射回 15M 前整体 shift(1)，避免使用未完成高周期状态。
    - 15M 信号由当前 K 线收盘确认，下一根 15M open 入场。
    - 同一根 K 同时触发止损和止盈，按保守原则先算止损。
"""

from __future__ import annotations

import argparse
import math
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
from src.backtest_common.indicators import adx, atr, ema, resample_ohlcv, rsi  # noqa: E402


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "ETH-USDT-SWAP"
    base_timeframe: str = "15m"

    # 1H trend regime
    h1_ema_fast: int = 50
    h1_ema_slow: int = 200
    h1_adx_period: int = 14
    h1_adx_min: float = 18.0
    h1_atr_period: int = 14
    h1_min_atr_pct: float = 0.0060      # 0.60%; avoid dead chop
    h1_max_atr_pct: float = 0.0400      # 4.00%; avoid abnormal chaos
    h1_slope_lookback: int = 5

    # 15M entry
    ema_pullback_fast: int = 20
    ema_pullback_slow: int = 50
    rsi_period: int = 14
    long_rsi_min: float = 42.0
    long_rsi_max: float = 65.0
    short_rsi_min: float = 35.0
    short_rsi_max: float = 58.0
    volume_ma_period: int = 20
    volume_mult_min: float = 0.80
    cooldown_bars: int = 4              # 1h cooldown after any exit

    # Stop / exit
    atr_15m_period: int = 14
    atr_stop_mult: float = 1.20
    stop_lookback_bars: int = 6
    min_stop_pct: float = 0.0045        # 0.45%
    max_stop_pct: float = 0.0220        # 2.20%
    tp1_r: float = 1.0
    tp1_qty_pct: float = 0.35
    tp2_r: float = 2.0
    tp2_qty_pct: float = 0.35
    trailing_atr_mult: float = 1.50
    no_progress_bars: int = 48          # 12h on 15M; exit if MFE < 0.7R before TP1
    no_progress_min_mfe_r: float = 0.70
    max_hold_bars: int = 192            # 48h on 15M

    # Risk / finance
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.005       # 0.50% equity risk per trade
    max_notional_mult: float = 3.0
    fee_rate: float = 0.0005            # taker fee per side by default
    slippage_pct: float = 0.0002        # per entry/exit



def load_base_data(symbol: str, start_date: str, end_date: str, timeframe: str = "15m") -> pd.DataFrame:
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


def build_h1_features(base_15m: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    h1 = resample_ohlcv(base_15m, "1h")
    h1["h1_ema_fast"] = ema(h1["close"], cfg.h1_ema_fast)
    h1["h1_ema_slow"] = ema(h1["close"], cfg.h1_ema_slow)
    h1["h1_ema_fast_slope"] = h1["h1_ema_fast"] - h1["h1_ema_fast"].shift(cfg.h1_slope_lookback)
    h1["h1_atr"] = atr(h1, cfg.h1_atr_period)
    h1["h1_atr_pct"] = h1["h1_atr"] / h1["close"]
    h1["h1_adx"] = adx(h1, cfg.h1_adx_period)

    h1["h1_vol_ok"] = h1["h1_atr_pct"].between(cfg.h1_min_atr_pct, cfg.h1_max_atr_pct)
    h1["h1_long_trend"] = (
        (h1["close"] > h1["h1_ema_slow"])
        & (h1["h1_ema_fast"] > h1["h1_ema_slow"])
        & (h1["h1_ema_fast_slope"] > 0)
        & (h1["h1_adx"] >= cfg.h1_adx_min)
        & h1["h1_vol_ok"]
    )
    h1["h1_short_trend"] = (
        (h1["close"] < h1["h1_ema_slow"])
        & (h1["h1_ema_fast"] < h1["h1_ema_slow"])
        & (h1["h1_ema_fast_slope"] < 0)
        & (h1["h1_adx"] >= cfg.h1_adx_min)
        & h1["h1_vol_ok"]
    )

    # Conservative availability: use only the previous completed 1H state on 15M bars.
    return h1.shift(1)


def build_features(base_15m: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = base_15m.copy()
    h1 = build_h1_features(df, cfg)
    h1_cols = [
        "h1_ema_fast", "h1_ema_slow", "h1_ema_fast_slope", "h1_atr", "h1_atr_pct", "h1_adx",
        "h1_vol_ok", "h1_long_trend", "h1_short_trend",
    ]
    df = df.join(h1[h1_cols].reindex(df.index, method="ffill"))

    df["ema20"] = ema(df["close"], cfg.ema_pullback_fast)
    df["ema50"] = ema(df["close"], cfg.ema_pullback_slow)
    df["atr_15m"] = atr(df, cfg.atr_15m_period)
    df["rsi"] = rsi(df["close"], cfg.rsi_period)
    df["volume_ma"] = df["volume"].rolling(cfg.volume_ma_period, min_periods=cfg.volume_ma_period).mean().shift(1)
    df["volume_ok"] = df["volume"] > df["volume_ma"] * cfg.volume_mult_min
    df["swing_low_6"] = df["low"].rolling(cfg.stop_lookback_bars, min_periods=cfg.stop_lookback_bars).min().shift(1)
    df["swing_high_6"] = df["high"].rolling(cfg.stop_lookback_bars, min_periods=cfg.stop_lookback_bars).max().shift(1)

    df["long_signal"] = (
        (df["h1_long_trend"] == True)  # noqa: E712
        & ((df["low"] <= df["ema20"]) | (df["low"] <= df["ema50"]))
        & (df["close"] > df["ema20"])
        & (df["close"] > df["open"])
        & df["volume_ok"]
        & df["rsi"].between(cfg.long_rsi_min, cfg.long_rsi_max)
    )
    df["short_signal"] = (
        (df["h1_short_trend"] == True)  # noqa: E712
        & ((df["high"] >= df["ema20"]) | (df["high"] >= df["ema50"]))
        & (df["close"] < df["ema20"])
        & (df["close"] < df["open"])
        & df["volume_ok"]
        & df["rsi"].between(cfg.short_rsi_min, cfg.short_rsi_max)
    )
    df["signal"] = 0
    df.loc[df["long_signal"], "signal"] = 1
    df.loc[df["short_signal"], "signal"] = -1
    return df.dropna().copy()



def build_initial_stop(df: pd.DataFrame, i: int, entry_price: float, side: int, cfg: StrategyConfig) -> tuple[float, float, str]:
    row = df.iloc[i]
    atr_value = float(row["atr_15m"])
    if side == 1:
        raw_sl = min(float(row["swing_low_6"]), entry_price - cfg.atr_stop_mult * atr_value)
        min_sl = entry_price * (1 - cfg.min_stop_pct)
        sl = min(raw_sl, min_sl)
        stop_pct = (entry_price - sl) / entry_price
    else:
        raw_sl = max(float(row["swing_high_6"]), entry_price + cfg.atr_stop_mult * atr_value)
        min_sl = entry_price * (1 + cfg.min_stop_pct)
        sl = max(raw_sl, min_sl)
        stop_pct = (sl - entry_price) / entry_price

    if stop_pct <= 0 or not math.isfinite(stop_pct):
        return sl, stop_pct, "bad_stop"
    if stop_pct > cfg.max_stop_pct:
        return sl, stop_pct, "stop_too_wide"
    return sl, stop_pct, "ok"


def calc_part_pnl(entry: float, exit_price: float, side: int, qty: float, fee_rate: float, entry_fee_part: float) -> tuple[float, float]:
    exit_fee = qty * exit_price * fee_rate
    if side == 1:
        pnl = (exit_price - entry) * qty - entry_fee_part - exit_fee
    else:
        pnl = (entry - exit_price) * qty - entry_fee_part - exit_fee
    return pnl, exit_fee


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
    initial_sl = 0.0
    stop_price = 0.0
    risk_per_coin = 0.0
    qty_initial = 0.0
    qty_remaining = 0.0
    entry_fee_total = 0.0
    realized_pnl = 0.0
    realized_fee = 0.0
    tp1_hit = False
    tp2_hit = False
    tp1_price = 0.0
    tp2_price = 0.0
    max_fav = 0.0
    max_adv = 0.0
    last_exit_i = -10**9
    exit_notes: list[str] = []

    rows = list(df.itertuples())
    idx = df.index

    for i in range(len(rows) - 1):
        row = rows[i]
        ts = idx[i]

        if in_pos:
            high = float(row.high)
            low = float(row.low)
            close = float(row.close)
            atr_value = float(row.atr_15m)
            hold_bars = i - entry_i

            if side == 1:
                max_fav = max(max_fav, high)
                max_adv = min(max_adv, low)
                mfe_r = (max_fav - entry_price) / risk_per_coin
                mae_r = (entry_price - max_adv) / risk_per_coin
                if tp1_hit:
                    stop_price = max(stop_price, entry_price)
                if tp2_hit:
                    stop_price = max(stop_price, close - cfg.trailing_atr_mult * atr_value)
                stop_hit = low <= stop_price
                tp1_reached = (not tp1_hit) and high >= tp1_price
                tp2_reached = (not tp2_hit) and high >= tp2_price
                trend_exit = bool(getattr(row, "h1_short_trend"))
            else:
                max_fav = min(max_fav, low)
                max_adv = max(max_adv, high)
                mfe_r = (entry_price - max_fav) / risk_per_coin
                mae_r = (max_adv - entry_price) / risk_per_coin
                if tp1_hit:
                    stop_price = min(stop_price, entry_price)
                if tp2_hit:
                    stop_price = min(stop_price, close + cfg.trailing_atr_mult * atr_value)
                stop_hit = high >= stop_price
                tp1_reached = (not tp1_hit) and low <= tp1_price
                tp2_reached = (not tp2_hit) and low <= tp2_price
                trend_exit = bool(getattr(row, "h1_long_trend"))

            full_exit = False
            final_exit_price = 0.0
            final_reason = ""

            # Conservative order: stop first if same bar can hit both TP and SL.
            if stop_hit:
                full_exit = True
                final_exit_price = apply_exit_slippage(stop_price, side, cfg.slippage_pct)
                final_reason = "STOP_OR_TRAILING_STOP"
            else:
                if tp1_reached and qty_remaining > 0:
                    q_exit = min(qty_initial * cfg.tp1_qty_pct, qty_remaining)
                    exit_price = apply_exit_slippage(tp1_price, side, cfg.slippage_pct)
                    entry_fee_part = entry_fee_total * (q_exit / qty_initial)
                    pnl_part, exit_fee = calc_part_pnl(entry_price, exit_price, side, q_exit, cfg.fee_rate, entry_fee_part)
                    capital += pnl_part
                    peak = max(peak, capital)
                    qty_remaining -= q_exit
                    realized_pnl += pnl_part
                    realized_fee += entry_fee_part + exit_fee
                    tp1_hit = True
                    exit_notes.append("TP1")
                    if side == 1:
                        stop_price = max(stop_price, entry_price)
                    else:
                        stop_price = min(stop_price, entry_price)

                if tp2_reached and qty_remaining > 0:
                    q_exit = min(qty_initial * cfg.tp2_qty_pct, qty_remaining)
                    exit_price = apply_exit_slippage(tp2_price, side, cfg.slippage_pct)
                    entry_fee_part = entry_fee_total * (q_exit / qty_initial)
                    pnl_part, exit_fee = calc_part_pnl(entry_price, exit_price, side, q_exit, cfg.fee_rate, entry_fee_part)
                    capital += pnl_part
                    peak = max(peak, capital)
                    qty_remaining -= q_exit
                    realized_pnl += pnl_part
                    realized_fee += entry_fee_part + exit_fee
                    tp2_hit = True
                    exit_notes.append("TP2")

                if qty_remaining <= qty_initial * 1e-9:
                    full_exit = True
                    final_exit_price = tp2_price
                    final_reason = "TP1_TP2_FULL_EXIT"
                elif hold_bars >= cfg.no_progress_bars and (not tp1_hit) and mfe_r < cfg.no_progress_min_mfe_r:
                    full_exit = True
                    final_exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                    final_reason = "NO_PROGRESS_EXIT"
                elif trend_exit:
                    full_exit = True
                    final_exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                    final_reason = "H1_TREND_REVERSE_EXIT"
                elif hold_bars >= cfg.max_hold_bars:
                    full_exit = True
                    final_exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                    final_reason = "MAX_HOLD_EXIT"

            if full_exit and qty_remaining > 0:
                q_exit = qty_remaining
                entry_fee_part = entry_fee_total * (q_exit / qty_initial)
                pnl_part, exit_fee = calc_part_pnl(entry_price, final_exit_price, side, q_exit, cfg.fee_rate, entry_fee_part)
                capital += pnl_part
                peak = max(peak, capital)
                qty_remaining = 0.0
                realized_pnl += pnl_part
                realized_fee += entry_fee_part + exit_fee
                exit_notes.append(final_reason)

            if full_exit:
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "type": "LONG" if side == 1 else "SHORT",
                        "entry": entry_price,
                        "exit": final_exit_price,
                        "initial_sl": initial_sl,
                        "final_sl": stop_price,
                        "tp1": tp1_price,
                        "tp2": tp2_price,
                        "tp1_hit": tp1_hit,
                        "tp2_hit": tp2_hit,
                        "qty": qty_initial,
                        "pnl": realized_pnl,
                        "fee": realized_fee,
                        "capital": capital,
                        "return_pct": realized_pnl / max(capital - realized_pnl, 1e-12),
                        "mfe_r": round(float(mfe_r), 4),
                        "mae_r": round(float(mae_r), 4),
                        "sl_pct": round(abs(entry_price - initial_sl) / entry_price * 100, 4),
                        "holding_bars_15m": int(hold_bars),
                        "holding_hours": round(float(hold_bars * 0.25), 2),
                        "note": "+".join(exit_notes) if exit_notes else final_reason,
                    }
                )
                in_pos = False
                side = 0
                last_exit_i = i

        if not in_pos and i - last_exit_i >= cfg.cooldown_bars:
            sig = int(getattr(row, "signal"))
            if sig != 0:
                next_open = float(rows[i + 1].open)
                entry = apply_entry_slippage(next_open, sig, cfg.slippage_pct)
                sl, stop_pct, stop_status = build_initial_stop(df, i, entry, sig, cfg)
                if stop_status == "ok":
                    rpc = abs(entry - sl)
                    risk_usdt = capital * cfg.risk_per_trade
                    q = risk_usdt / rpc
                    q = min(q, (capital * cfg.max_notional_mult) / entry)
                    if q > 0 and math.isfinite(q):
                        in_pos = True
                        side = sig
                        entry_i = i + 1
                        entry_time = idx[i + 1]
                        entry_price = entry
                        initial_sl = sl
                        stop_price = sl
                        risk_per_coin = rpc
                        qty_initial = q
                        qty_remaining = q
                        entry_fee_total = qty_initial * entry_price * cfg.fee_rate
                        realized_pnl = 0.0
                        realized_fee = 0.0
                        tp1_hit = False
                        tp2_hit = False
                        if side == 1:
                            tp1_price = entry_price + cfg.tp1_r * risk_per_coin
                            tp2_price = entry_price + cfg.tp2_r * risk_per_coin
                        else:
                            tp1_price = entry_price - cfg.tp1_r * risk_per_coin
                            tp2_price = entry_price - cfg.tp2_r * risk_per_coin
                        max_fav = entry_price
                        max_adv = entry_price
                        exit_notes = []

        equity_rows.append({"time": ts, "capital": capital, "drawdown_pct": (peak - capital) / peak if peak > 0 else 0.0})

    if in_pos:
        ts = idx[-1]
        close = float(df.iloc[-1]["close"])
        final_exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
        q_exit = qty_remaining
        entry_fee_part = entry_fee_total * (q_exit / qty_initial)
        pnl_part, exit_fee = calc_part_pnl(entry_price, final_exit_price, side, q_exit, cfg.fee_rate, entry_fee_part)
        capital += pnl_part
        realized_pnl += pnl_part
        realized_fee += entry_fee_part + exit_fee
        if side == 1:
            mfe_r = (max_fav - entry_price) / risk_per_coin
            mae_r = (entry_price - max_adv) / risk_per_coin
        else:
            mfe_r = (entry_price - max_fav) / risk_per_coin
            mae_r = (max_adv - entry_price) / risk_per_coin
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": ts,
                "type": "LONG" if side == 1 else "SHORT",
                "entry": entry_price,
                "exit": final_exit_price,
                "initial_sl": initial_sl,
                "final_sl": stop_price,
                "tp1": tp1_price,
                "tp2": tp2_price,
                "tp1_hit": tp1_hit,
                "tp2_hit": tp2_hit,
                "qty": qty_initial,
                "pnl": realized_pnl,
                "fee": realized_fee,
                "capital": capital,
                "return_pct": realized_pnl / max(capital - realized_pnl, 1e-12),
                "mfe_r": round(float(mfe_r), 4),
                "mae_r": round(float(mae_r), 4),
                "sl_pct": round(abs(entry_price - initial_sl) / entry_price * 100, 4),
                "holding_bars_15m": int(len(df) - 1 - entry_i),
                "holding_hours": round(float((len(df) - 1 - entry_i) * 0.25), 2),
                "note": "+".join(exit_notes + ["FORCE_CLOSE_END"]),
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
        "profit_factor": round(pf, 4) if math.isfinite(pf) else "inf",
        "expectancy_pct": round(float(tdf["return_pct"].mean() * 100), 6),
        "max_drawdown_pct": round(float(equity["drawdown_pct"].max() * 100), 4) if not equity.empty else 0.0,
        "avg_mfe_r": round(float(tdf["mfe_r"].mean()), 4),
        "avg_mae_r": round(float(tdf["mae_r"].mean()), 4),
        "avg_holding_hours": round(float(tdf["holding_hours"].mean()), 2),
        "tp1_hit_rate": round(float(tdf["tp1_hit"].mean() * 100), 4),
        "tp2_hit_rate": round(float(tdf["tp2_hit"].mean() * 100), 4),
        "total_fees": round(float(tdf["fee"].sum()), 4),
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
    }


def write_outputs(trades: list[dict[str, Any]], equity: pd.DataFrame, features: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / "eth_1h_trend_15m_pullback_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / "eth_1h_trend_15m_pullback_equity.csv")
    audit_cols = [
        "open", "high", "low", "close", "volume", "ema20", "ema50", "atr_15m", "rsi", "volume_ma", "volume_ok",
        "h1_ema_fast", "h1_ema_slow", "h1_ema_fast_slope", "h1_atr_pct", "h1_adx", "h1_long_trend", "h1_short_trend",
        "swing_low_6", "swing_high_6", "long_signal", "short_signal", "signal",
    ]
    features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / "eth_1h_trend_15m_pullback_signal_audit.csv")
    pd.Series(summary).to_json(out_dir / "eth_1h_trend_15m_pullback_summary.json", force_ascii=False, indent=2)


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 80)
    print("ETH 1H Trend + 15M Pullback Backtest Summary")
    print("=" * 80)
    for k, v in summary.items():
        print(f"{k:>24}: {v}")
    print("-" * 80)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="ETH 1H trend regime + 15M pullback continuation backtest.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-07-01")
    p.add_argument("--end-date", default=today)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--risk-per-trade", type=float, default=0.005)
    p.add_argument("--max-notional-mult", type=float, default=3.0)
    p.add_argument("--fee-rate", type=float, default=0.0005)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--h1-adx-min", type=float, default=18.0)
    p.add_argument("--h1-min-atr-pct", type=float, default=0.0060)
    p.add_argument("--h1-max-atr-pct", type=float, default=0.0400)
    p.add_argument("--tp1-r", type=float, default=1.0)
    p.add_argument("--tp2-r", type=float, default=2.0)
    p.add_argument("--out-dir", default="data/reports/mf/eth_1h_trend_15m_pullback")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = StrategyConfig(
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        max_notional_mult=args.max_notional_mult,
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
        h1_adx_min=args.h1_adx_min,
        h1_min_atr_pct=args.h1_min_atr_pct,
        h1_max_atr_pct=args.h1_max_atr_pct,
        tp1_r=args.tp1_r,
        tp2_r=args.tp2_r,
    )
    print(f"Loading {cfg.symbol} {cfg.base_timeframe}: {args.start_date} -> {args.end_date}")
    base = load_base_data(cfg.symbol, args.start_date, args.end_date, cfg.base_timeframe)
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
            strategy_name="ETH_MF_TrendPullback_1H_15M",
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
