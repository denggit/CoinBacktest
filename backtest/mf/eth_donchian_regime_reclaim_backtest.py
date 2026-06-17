#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH Donchian Regime + 5M Reclaim Backtest
=========================================

放置位置建议：backtest/mf/eth_donchian_regime_reclaim_backtest.py
运行示例：
    python backtest/mf/eth_donchian_regime_reclaim_backtest.py --start-date 2023-01-01 --end-date 2026-06-17

策略定位：
    Donchian 120H 只负责判断大方向 regime。
    5M 只在 Donchian regime 方向内，做局部扫低/扫高后的 reclaim 入场。

重要说明：
    这是 K 线可回测近似版，不是真逐笔 trades/books 盘口吸收版。
    当前使用 OKXDataLoader 读取 5m K 线，因此用：
        - sweep 局部高低点
        - reclaim 收回局部高低点
        - wick ratio
        - volume burst
    近似表达“流动性扫单失败 + 反向恢复”。

反偷看：
    - 1H Donchian 使用 shift(1)，不包含当前 1H K 线。
    - 1H regime 再 shift(1) 映射到 5M，避免用未完成高周期状态。
    - 5M 信号在当前 K 收盘后产生，下一根 5M open 入场。
    - 同一根 K 同时触发 TP/SL，按保守原则先算 SL。
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
from src.backtest_common.indicators import atr, resample_ohlcv  # noqa: E402


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "ETH-USDT-SWAP"
    base_timeframe: str = "5m"

    # 1H Donchian regime
    regime_lookback_1h: int = 120
    regime_breakout_buffer_pct: float = 0.0020

    # 5M reclaim entry
    local_level_lookback_bars: int = 24      # 24 * 5m = 2h local high/low
    sweep_min_pct: float = 0.0002            # at least 0.02% through local level
    sweep_max_pct: float = 0.0040            # max 0.40%; deeper = likely true break, skip
    reclaim_buffer_pct: float = 0.00005      # close back beyond level by 0.005%
    wick_ratio_min: float = 0.30             # rejection wick ratio
    volume_median_window: int = 48           # 4h volume median on 5m bars
    volume_mult: float = 1.20

    # 5M ATR / stop
    atr_period_5m: int = 14
    atr_stop_mult: float = 0.5
    min_stop_pct: float = 0.0025             # 0.25%
    max_stop_pct: float = 0.0120             # 1.20%

    # Exit: full position, no TP1, no partial
    target_r: float = 1.5
    no_progress_bars: int = 24               # 2h; if MFE < 0.5R, exit
    max_hold_bars: int = 48                  # 4h hard exit

    # Risk / fee
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.002
    max_notional_mult: float = 3.0
    fee_rate: float = 0.0005
    slippage_pct: float = 0.0002

    # Cooldown
    same_side_cooldown_bars: int = 6          # 30m same-side cooldown after SL



def load_base_data(symbol: str, start_date: str, end_date: str, timeframe: str = "5m") -> pd.DataFrame:
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


def build_regime_1h(base_5m: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    h1 = resample_ohlcv(base_5m, "1h")
    h1["donchian_high_120h"] = h1["high"].rolling(cfg.regime_lookback_1h, min_periods=cfg.regime_lookback_1h).max().shift(1)
    h1["donchian_low_120h"] = h1["low"].rolling(cfg.regime_lookback_1h, min_periods=cfg.regime_lookback_1h).min().shift(1)
    h1["long_regime_break"] = h1["close"] > h1["donchian_high_120h"] * (1 + cfg.regime_breakout_buffer_pct)
    h1["short_regime_break"] = h1["close"] < h1["donchian_low_120h"] * (1 - cfg.regime_breakout_buffer_pct)

    regime = []
    current = 0
    for row in h1.itertuples():
        if bool(row.long_regime_break):
            current = 1
        elif bool(row.short_regime_break):
            current = -1
        regime.append(current)
    h1["donchian_regime"] = regime

    # Shift one 1H bar before mapping to 5M for safety.
    h1["donchian_regime_available"] = h1["donchian_regime"].shift(1)
    return h1


def build_features(base_5m: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = base_5m.copy()
    h1 = build_regime_1h(df, cfg)
    df = df.join(h1[["donchian_regime_available", "donchian_high_120h", "donchian_low_120h"]].reindex(df.index, method="ffill"))
    df = df.rename(columns={"donchian_regime_available": "regime"})

    df["atr_5m"] = atr(df, cfg.atr_period_5m)
    df["volume_median"] = df["volume"].rolling(cfg.volume_median_window, min_periods=cfg.volume_median_window).median().shift(1)
    df["local_low"] = df["low"].rolling(cfg.local_level_lookback_bars, min_periods=cfg.local_level_lookback_bars).min().shift(1)
    df["local_high"] = df["high"].rolling(cfg.local_level_lookback_bars, min_periods=cfg.local_level_lookback_bars).max().shift(1)

    bar_range = (df["high"] - df["low"]).replace(0, pd.NA)
    lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]).clip(lower=0)
    upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)).clip(lower=0)
    df["lower_wick_ratio"] = (lower_wick / bar_range).fillna(0.0)
    df["upper_wick_ratio"] = (upper_wick / bar_range).fillna(0.0)
    df["volume_burst"] = df["volume"] > df["volume_median"] * cfg.volume_mult

    # Long: in long regime, sweep local low but close reclaims local low.
    long_sweep_depth = (df["local_low"] - df["low"]) / df["local_low"]
    df["long_sweep_depth_pct"] = long_sweep_depth
    df["long_sweep_ok"] = long_sweep_depth.between(cfg.sweep_min_pct, cfg.sweep_max_pct)
    df["long_reclaim_ok"] = df["close"] > df["local_low"] * (1 + cfg.reclaim_buffer_pct)
    df["long_reclaim_signal"] = (
        (df["regime"] == 1)
        & df["long_sweep_ok"]
        & df["long_reclaim_ok"]
        & (df["lower_wick_ratio"] >= cfg.wick_ratio_min)
        & df["volume_burst"]
    )

    # Short: in short regime, sweep local high but close reclaims below local high.
    short_sweep_depth = (df["high"] - df["local_high"]) / df["local_high"]
    df["short_sweep_depth_pct"] = short_sweep_depth
    df["short_sweep_ok"] = short_sweep_depth.between(cfg.sweep_min_pct, cfg.sweep_max_pct)
    df["short_reclaim_ok"] = df["close"] < df["local_high"] * (1 - cfg.reclaim_buffer_pct)
    df["short_reclaim_signal"] = (
        (df["regime"] == -1)
        & df["short_sweep_ok"]
        & df["short_reclaim_ok"]
        & (df["upper_wick_ratio"] >= cfg.wick_ratio_min)
        & df["volume_burst"]
    )

    df["signal"] = 0
    df.loc[df["long_reclaim_signal"], "signal"] = 1
    df.loc[df["short_reclaim_signal"], "signal"] = -1
    return df.dropna().copy()



def build_initial_stop(df: pd.DataFrame, i: int, entry_price: float, side: int, cfg: StrategyConfig) -> tuple[float, float, str]:
    row = df.iloc[i]
    atr_value = float(row["atr_5m"])
    if side == 1:
        raw_sl = float(row["low"]) - cfg.atr_stop_mult * atr_value
        min_sl = entry_price * (1 - cfg.min_stop_pct)
        sl = min(raw_sl, min_sl)
        stop_pct = (entry_price - sl) / entry_price
    else:
        raw_sl = float(row["high"]) + cfg.atr_stop_mult * atr_value
        min_sl = entry_price * (1 + cfg.min_stop_pct)
        sl = max(raw_sl, min_sl)
        stop_pct = (sl - entry_price) / entry_price
    if stop_pct <= 0:
        return sl, stop_pct, "bad_stop"
    if stop_pct > cfg.max_stop_pct:
        return sl, stop_pct, "stop_too_wide"
    return sl, stop_pct, "ok"


def run_backtest(df: pd.DataFrame, cfg: StrategyConfig) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    capital = cfg.initial_capital
    peak = capital
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    in_pos = False
    side = 0
    entry_time = None
    entry_i = -1
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    initial_sl = 0.0
    risk_per_coin = 0.0
    qty = 0.0
    entry_fee = 0.0
    max_fav = 0.0
    max_adv = 0.0
    last_long_sl_i = -10**9
    last_short_sl_i = -10**9

    rows = list(df.itertuples())
    idx = df.index

    for i in range(len(rows) - 1):
        row = rows[i]
        ts = idx[i]

        if in_pos:
            high = float(row.high)
            low = float(row.low)
            close = float(row.close)
            hold_bars = i - entry_i
            exit_now = False
            exit_price = 0.0
            reason = ""

            if side == 1:
                max_fav = max(max_fav, high)
                max_adv = min(max_adv, low)
                # Conservative: if same bar touches both SL and target, SL first.
                if low <= stop_price:
                    exit_now = True
                    exit_price = apply_exit_slippage(stop_price, side, cfg.slippage_pct)
                    reason = "SL"
                    last_long_sl_i = i
                elif high >= target_price:
                    exit_now = True
                    exit_price = apply_exit_slippage(target_price, side, cfg.slippage_pct)
                    reason = "TARGET_R"
                elif hold_bars >= cfg.no_progress_bars and (max_fav - entry_price) / risk_per_coin < 0.5:
                    exit_now = True
                    exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                    reason = "NO_PROGRESS_EXIT"
                elif hold_bars >= cfg.max_hold_bars:
                    exit_now = True
                    exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                    reason = "MAX_HOLD_EXIT"
            else:
                max_fav = min(max_fav, low)
                max_adv = max(max_adv, high)
                if high >= stop_price:
                    exit_now = True
                    exit_price = apply_exit_slippage(stop_price, side, cfg.slippage_pct)
                    reason = "SL"
                    last_short_sl_i = i
                elif low <= target_price:
                    exit_now = True
                    exit_price = apply_exit_slippage(target_price, side, cfg.slippage_pct)
                    reason = "TARGET_R"
                elif hold_bars >= cfg.no_progress_bars and (entry_price - max_fav) / risk_per_coin < 0.5:
                    exit_now = True
                    exit_price = apply_exit_slippage(close, side, cfg.slippage_pct)
                    reason = "NO_PROGRESS_EXIT"
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
                        "initial_sl": initial_sl,
                        "target": target_price,
                        "qty": qty,
                        "pnl": pnl,
                        "fee": entry_fee + exit_fee,
                        "capital": capital,
                        "return_pct": pnl / max(cap_before, 1e-12),
                        "mfe_r": round(float(mfe_r), 4),
                        "mae_r": round(float(mae_r), 4),
                        "sl_pct": round(abs(entry_price - initial_sl) / entry_price * 100, 4),
                        "holding_bars_5m": int(hold_bars),
                        "holding_minutes": int(hold_bars * 5),
                        "note": reason,
                    }
                )
                in_pos = False
                side = 0

        if not in_pos:
            sig = int(getattr(row, "signal"))
            if sig == 1 and i - last_long_sl_i < cfg.same_side_cooldown_bars:
                sig = 0
            elif sig == -1 and i - last_short_sl_i < cfg.same_side_cooldown_bars:
                sig = 0

            if sig != 0:
                next_open = float(rows[i + 1].open)
                entry = apply_entry_slippage(next_open, sig, cfg.slippage_pct)
                sl, stop_pct, reason = build_initial_stop(df, i, entry, sig, cfg)
                if reason == "ok":
                    risk_usdt = capital * cfg.risk_per_trade
                    rpc = abs(entry - sl)
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
                        qty = q
                        entry_fee = qty * entry_price * cfg.fee_rate
                        if side == 1:
                            target_price = entry_price + cfg.target_r * risk_per_coin
                            max_fav = entry_price
                            max_adv = entry_price
                        else:
                            target_price = entry_price - cfg.target_r * risk_per_coin
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
                "initial_sl": initial_sl,
                "target": target_price,
                "qty": qty,
                "pnl": pnl,
                "fee": entry_fee + exit_fee,
                "capital": capital,
                "return_pct": pnl / max(cap_before, 1e-12),
                "mfe_r": round(float(mfe_r), 4),
                "mae_r": round(float(mae_r), 4),
                "sl_pct": round(abs(entry_price - initial_sl) / entry_price * 100, 4),
                "holding_bars_5m": int(len(df) - 1 - entry_i),
                "holding_minutes": int((len(df) - 1 - entry_i) * 5),
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
        "avg_holding_minutes": round(float(tdf["holding_minutes"].mean()), 2),
        "target_hit_rate": round(float((tdf["note"] == "TARGET_R").mean() * 100), 4),
        "total_fees": round(float(tdf["fee"].sum()), 4),
    }


def write_outputs(trades: list[dict[str, Any]], equity: pd.DataFrame, features: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / "eth_donchian_regime_reclaim_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / "eth_donchian_regime_reclaim_equity.csv")
    audit_cols = [
        "open", "high", "low", "close", "volume", "regime", "donchian_high_120h", "donchian_low_120h",
        "local_low", "local_high", "long_sweep_depth_pct", "short_sweep_depth_pct", "lower_wick_ratio", "upper_wick_ratio",
        "volume_burst", "long_reclaim_signal", "short_reclaim_signal", "signal",
    ]
    features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / "eth_donchian_regime_reclaim_signal_audit.csv")
    pd.Series(summary).to_json(out_dir / "eth_donchian_regime_reclaim_summary.json", force_ascii=False, indent=2)


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 80)
    print("ETH Donchian Regime + 5M Reclaim Backtest Summary")
    print("=" * 80)
    for k, v in summary.items():
        print(f"{k:>24}: {v}")
    print("-" * 80)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="ETH Donchian 120H regime + 5M reclaim backtest, K-line approximation.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=today)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--risk-per-trade", type=float, default=0.002)
    p.add_argument("--fee-rate", type=float, default=0.0005)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--target-r", type=float, default=1.5)
    p.add_argument("--volume-mult", type=float, default=1.2)
    p.add_argument("--local-lookback-bars", type=int, default=24)
    p.add_argument("--out-dir", default="data/reports/mf/eth_donchian_regime_reclaim")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = StrategyConfig(
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
        target_r=args.target_r,
        volume_mult=args.volume_mult,
        local_level_lookback_bars=args.local_lookback_bars,
    )
    print(f"Loading {cfg.symbol} {cfg.base_timeframe}: {args.start_date} -> {args.end_date}")
    loader = OKXDataLoader(symbol=cfg.symbol, timeframe=cfg.base_timeframe)
    base = loader.fetch_data_by_date_range(args.start_date, args.end_date)
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
            strategy_name="ETH_MF_DonchianRegimeReclaim_5M",
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
