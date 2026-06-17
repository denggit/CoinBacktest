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


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    elif "ts_ms" in out.columns:
        ts = pd.to_datetime(pd.to_numeric(out["ts_ms"], errors="coerce"), unit="ms", utc=True, errors="coerce")
    else:
        raise RuntimeError("tick chunk missing timestamp/ts_ms")
    out = out.loc[ts.notna()].copy()
    out.index = ts.loc[ts.notna()]
    return out.sort_index()


def aggregate_trades_to_seconds(chunk: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    if chunk.empty:
        return pd.DataFrame()
    df = _ensure_utc_index(chunk)
    for col in ["price", "size"]:
        if col not in df.columns:
            raise RuntimeError(f"tick chunk missing column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "side" not in df.columns:
        raise RuntimeError("tick chunk missing column: side")

    df = df.dropna(subset=["price", "size"])
    if df.empty:
        return pd.DataFrame()

    side = df["side"].astype(str).str.lower()
    notional = df["price"] * df["size"] * cfg.contract_value
    df["buy_notional"] = notional.where(side == "buy", 0.0)
    df["sell_notional"] = notional.where(side == "sell", 0.0)
    df["buy_contracts"] = df["size"].where(side == "buy", 0.0)
    df["sell_contracts"] = df["size"].where(side == "sell", 0.0)
    df["notional"] = notional
    df["trades_count"] = 1

    rule = f"{int(cfg.bar_seconds)}s"
    bars = df.resample(rule).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        buy_notional=("buy_notional", "sum"),
        sell_notional=("sell_notional", "sum"),
        buy_contracts=("buy_contracts", "sum"),
        sell_contracts=("sell_contracts", "sum"),
        volume_notional=("notional", "sum"),
        trades_count=("trades_count", "sum"),
    )
    return bars.dropna(subset=["open", "high", "low", "close"])


def merge_second_bars(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts).sort_index()
    merged = raw.groupby(level=0).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        buy_notional=("buy_notional", "sum"),
        sell_notional=("sell_notional", "sum"),
        buy_contracts=("buy_contracts", "sum"),
        sell_contracts=("sell_contracts", "sum"),
        volume_notional=("volume_notional", "sum"),
        trades_count=("trades_count", "sum"),
    )
    if merged.empty:
        return merged
    full_index = pd.date_range(merged.index.min(), merged.index.max(), freq="1s", tz="UTC")
    merged = merged.reindex(full_index)
    merged["close"] = merged["close"].ffill()
    for col in ["open", "high", "low"]:
        merged[col] = merged[col].fillna(merged["close"])
    for col in ["buy_notional", "sell_notional", "buy_contracts", "sell_contracts", "volume_notional", "trades_count"]:
        merged[col] = merged[col].fillna(0.0)
    merged["cvd_notional"] = merged["buy_notional"] - merged["sell_notional"]
    merged["signed_contracts"] = merged["buy_contracts"] - merged["sell_contracts"]
    return merged.dropna(subset=["open", "high", "low", "close"])


def load_second_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    cfg: StrategyConfig,
    *,
    chunksize: int,
    trades_url_template: str,
    data_dir: str | None,
) -> pd.DataFrame:
    loader = OKXTickLoader(symbol=symbol, data_dir=data_dir, trades_url_template=trades_url_template)
    parts: list[pd.DataFrame] = []
    for day, chunk in loader.iter_trades(start_date, end_date, chunksize=chunksize):
        bars = aggregate_trades_to_seconds(chunk, cfg)
        if not bars.empty:
            parts.append(bars)
        print(f"loaded tick chunk day={day} rows={len(chunk)} bars={len(bars)}")
        del chunk, bars
    out = merge_second_bars(parts)
    if out.empty:
        raise RuntimeError(f"No tick data loaded for {symbol} {start_date} -> {end_date}")
    return out


def rolling_sum(s: pd.Series, seconds: int) -> pd.Series:
    return s.rolling(int(seconds), min_periods=max(1, int(seconds))).sum()


def rolling_quantile_shifted(s: pd.Series, seconds: int, q: float) -> pd.Series:
    return s.rolling(int(seconds), min_periods=max(10, int(seconds) // 3)).quantile(q).shift(1)


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



def apply_entry_slippage(price: float, side: int, slippage_pct: float) -> float:
    return price * (1 + slippage_pct) if side == 1 else price * (1 - slippage_pct)


def apply_exit_slippage(price: float, side: int, slippage_pct: float) -> float:
    return price * (1 - slippage_pct) if side == 1 else price * (1 + slippage_pct)


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


def summarize(trades: list[dict[str, Any]], equity: pd.DataFrame, initial_capital: float, signal_count: int) -> dict[str, Any]:
    if not trades:
        return {"signal_count": int(signal_count), "total_trades": 0, "final_capital": round(initial_capital, 4), "total_return_pct": 0.0}
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["pnl"] > 0]
    losses = tdf[tdf["pnl"] <= 0]
    gross_profit = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(-losses["pnl"].sum()) if not losses.empty else 0.0
    final_capital = float(tdf.iloc[-1]["capital"])
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_fee = float(tdf["fee"].sum())
    return {
        "signal_count": int(signal_count),
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
        "avg_mfe_pct": round(float(tdf["mfe_pct"].mean() * 100), 4),
        "avg_mae_pct": round(float(tdf["mae_pct"].mean() * 100), 4),
        "avg_holding_seconds": round(float(tdf["holding_seconds"].mean()), 2),
        "total_fees": round(total_fee, 4),
        "fee_to_gross_profit_pct": round(total_fee / gross_profit * 100, 4) if gross_profit > 0 else None,
    }


def write_outputs(features: pd.DataFrame, trades: list[dict[str, Any]], equity: pd.DataFrame, summary: dict[str, Any], out_dir: Path, *, write_full_audit: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / f"{STRATEGY_NAME}_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / f"{STRATEGY_NAME}_equity.csv")

    audit_cols = [
        "open", "high", "low", "close", "buy_notional", "sell_notional", "volume_notional", "cvd_notional",
        "signal", "signal_reason", "signal_level", "signal_extreme", "local_low", "local_high",
        "sell_notional_3s", "buy_notional_3s", "cvd_5s", "buy_ratio_5s", "sell_ratio_5s",
        "range_high", "range_low", "range_pct", "compression_ok", "cvd_10s",
    ]
    signal_rows = features[features["signal"] != 0].copy()
    signal_rows[[c for c in audit_cols if c in signal_rows.columns]].to_csv(out_dir / f"{STRATEGY_NAME}_signal_audit.csv")
    if write_full_audit:
        features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{STRATEGY_NAME}_full_audit.csv")

    with (out_dir / f"{STRATEGY_NAME}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def emit_platform_report(trades: list[dict[str, Any]], features: pd.DataFrame, cfg: StrategyConfig, out_dir) -> None:
    """使用项目统一报告模块输出深度量化报告。"""
    if features.empty:
        return
    final_capital = float(trades[-1]["capital"]) if trades else float(cfg.initial_capital)
    total_days = max((features.index[-1] - features.index[0]).total_seconds() / 86400.0, 1.0 / 86400.0)
    print_full_report(
        trade_history=trades,
        df=features,
        initial_capital=cfg.initial_capital,
        capital=final_capital,
        strategy_name=STRATEGY_NAME,
        total_days=total_days,
        ai_enabled=False,
        symbol=cfg.symbol,
        report_dir=out_dir,
    )


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 88)
    print(f"ETH HF Backtest Summary | {STRATEGY_NAME}")
    print("=" * 88)
    for k, v in summary.items():
        print(f"{k:>28}: {v}")
    print("-" * 88)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 88 + "\n")


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
    emit_platform_report(trades, features, cfg, out_dir)
    write_outputs(features, trades, equity, summary, out_dir, write_full_audit=args.write_full_audit)
    print_summary(summary, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
