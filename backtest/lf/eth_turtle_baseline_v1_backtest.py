#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH Turtle Baseline V1
======================

Goal:
    A deliberately simple, auditable ETH trend-following baseline for CoinBacktest.
    No parameter grid. No ML. No discretionary filters. No future-looking signals.

Variants (pre-specified, not optimized):
    fast : 10-day breakout / 5-day exit
    s1   : 20-day breakout / 10-day exit (classic Turtle System 1 structure)
    s2   : 55-day breakout / 20-day exit (classic Turtle System 2 structure)

Execution:
    - Base data: 1H ETH-USDT-SWAP bars from src.data_feed.okx_loader.OKXDataLoader
    - Daily channels and daily N are built only from completed UTC daily bars.
    - Breakout orders are assumed resting before the 1H bar; if touched intrabar,
      fill at the breakout level (or worse open on a gap) plus configured slippage.
    - Initial protective stop: 2N by default.
    - No pyramiding in V1. This keeps the first baseline easy to audit.
    - Mark-to-market equity is recorded every 1H bar for drawdown calculations.
    - Funding is NOT guessed. If actual funding is needed, add the project's real
      funding series later rather than inserting a fabricated constant.

Default cost assumptions:
    fee_rate_per_side = 0.00055
    slippage_pct_per_fill = 0.00020

Outputs per variant / side mode:
    summary.csv, trades.csv, equity.csv, yearly.csv, monthly.csv,
    cost_stress.csv, run_config.json, report.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

VariantName = Literal["fast", "s1", "s2"]
SideMode = Literal["both", "long", "short"]


@dataclass(frozen=True)
class VariantConfig:
    name: str
    entry_days: int
    exit_days: int


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str = "ETH-USDT-SWAP"
    timeframe: str = "1H"
    n_period_days: int = 20
    initial_stop_n: float = 2.0
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.01
    max_notional_mult: float = 3.0
    fee_rate_per_side: float = 0.00055
    slippage_pct_per_fill: float = 0.00020
    cooldown_hours: int = 0
    side_mode: SideMode = "both"


VARIANTS: dict[str, VariantConfig] = {
    "fast": VariantConfig("FAST_10D_5D", entry_days=10, exit_days=5),
    "s1": VariantConfig("TURTLE_S1_20D_10D", entry_days=20, exit_days=10),
    "s2": VariantConfig("TURTLE_S2_55D_20D", entry_days=55, exit_days=20),
}


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def turtle_n(daily: pd.DataFrame, period: int) -> pd.Series:
    # Wilder-like EMA of daily True Range; then shift one full day so every
    # intraday bar only sees N from already-completed daily data.
    tr = true_range(daily)
    n = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return n.shift(1)


def load_data(symbol: str, start_date: str, end_date: str, timeframe: str) -> pd.DataFrame:
    from src.data_feed.okx_loader import OKXDataLoader

    loader = OKXDataLoader(symbol=symbol, timeframe=timeframe)
    df = loader.fetch_data_by_date_range(start_date, end_date)
    if df.empty:
        raise RuntimeError(f"No data loaded for {symbol} {timeframe} {start_date} -> {end_date}")
    df = df.sort_index().copy()
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")
    for c in sorted(required):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=sorted(required))
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("Data index must be a DatetimeIndex")
    if df.index.tz is not None:
        df = df.tz_convert("UTC").tz_localize(None)
    return df


def build_features(hourly: pd.DataFrame, variant: VariantConfig, cfg: BacktestConfig) -> pd.DataFrame:
    # UTC daily bars. label='left', closed='left' means a row stamped YYYY-MM-DD
    # contains exactly that day's completed 1H bars once the day is over.
    daily = hourly.resample("1D", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()

    daily["entry_high"] = daily["high"].rolling(variant.entry_days, min_periods=variant.entry_days).max().shift(1)
    daily["entry_low"] = daily["low"].rolling(variant.entry_days, min_periods=variant.entry_days).min().shift(1)
    daily["exit_high"] = daily["high"].rolling(variant.exit_days, min_periods=variant.exit_days).max().shift(1)
    daily["exit_low"] = daily["low"].rolling(variant.exit_days, min_periods=variant.exit_days).min().shift(1)
    daily["n"] = turtle_n(daily, cfg.n_period_days)

    feat = hourly.copy()
    map_cols = daily[["entry_high", "entry_low", "exit_high", "exit_low", "n"]]
    feat = feat.join(map_cols, how="left")
    # Daily row values already refer only to prior completed daily data because
    # every channel/N itself is shifted one day. Forward-fill within the UTC day.
    feat[["entry_high", "entry_low", "exit_high", "exit_low", "n"]] = (
        feat[["entry_high", "entry_low", "exit_high", "exit_low", "n"]].ffill()
    )
    return feat.dropna(subset=["entry_high", "entry_low", "exit_high", "exit_low", "n"]).copy()


def entry_fill(raw_level: float, bar_open: float, side: int, slippage: float) -> float:
    if side == 1:
        raw = max(raw_level, bar_open)  # gap through breakout -> pay the open
        return raw * (1 + slippage)
    raw = min(raw_level, bar_open)
    return raw * (1 - slippage)


def exit_fill(raw_level: float, bar_open: float, side: int, slippage: float) -> float:
    if side == 1:
        raw = min(raw_level, bar_open) if bar_open < raw_level else raw_level
        return raw * (1 - slippage)
    raw = max(raw_level, bar_open) if bar_open > raw_level else raw_level
    return raw * (1 + slippage)


def max_consecutive_true(values: pd.Series) -> int:
    best = cur = 0
    for x in values.fillna(False).astype(bool):
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return int(best)


def max_entry_gap_days(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if trades.empty:
        return float((end - start).total_seconds() / 86400)
    times = [start] + list(pd.to_datetime(trades["entry_time"])) + [end]
    gaps = [(b - a).total_seconds() / 86400 for a, b in zip(times[:-1], times[1:])]
    return float(max(gaps)) if gaps else 0.0


def marked_equity(cash: float, side: int, qty: float, entry_price: float, mark: float) -> float:
    if side == 0:
        return cash
    unrealized = (mark - entry_price) * qty if side == 1 else (entry_price - mark) * qty
    return cash + unrealized


def run_backtest(features: pd.DataFrame, cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    cash = cfg.initial_capital
    side = 0
    qty = 0.0
    entry_price = 0.0
    entry_time: pd.Timestamp | None = None
    entry_cash_before = cash
    entry_fee = 0.0
    stop_price = 0.0
    initial_n = 0.0
    max_fav = 0.0
    max_adv = 0.0
    last_exit_time: pd.Timestamp | None = None

    trades: list[dict[str, Any]] = []
    eq_rows: list[dict[str, Any]] = []

    rows = list(features.itertuples())
    idx = features.index

    for i, row in enumerate(rows):
        ts = idx[i]
        o = float(row.open)
        h = float(row.high)
        l = float(row.low)
        c = float(row.close)
        n = float(row.n)

        # -------------------------------------------------------------
        # Manage existing position first. Orders existed before this bar.
        # -------------------------------------------------------------
        exited_this_bar = False
        if side != 0:
            if side == 1:
                max_fav = max(max_fav, h)
                max_adv = min(max_adv, l)
                stop_hit = l <= stop_price
                channel_hit = l <= float(row.exit_low)
                if stop_hit or channel_hit:
                    # On a descending path from an open above both levels, the
                    # higher protective order triggers first. Gap-through uses open.
                    levels = []
                    if stop_hit:
                        levels.append(stop_price)
                    if channel_hit:
                        levels.append(float(row.exit_low))
                    raw_level = max(levels)
                    px = exit_fill(raw_level, o, side, cfg.slippage_pct_per_fill)
                    reason = "INITIAL_2N_STOP" if stop_hit and stop_price >= float(row.exit_low) else "DONCHIAN_EXIT"
                    exited_this_bar = True
            else:
                max_fav = min(max_fav, l)
                max_adv = max(max_adv, h)
                stop_hit = h >= stop_price
                channel_hit = h >= float(row.exit_high)
                if stop_hit or channel_hit:
                    levels = []
                    if stop_hit:
                        levels.append(stop_price)
                    if channel_hit:
                        levels.append(float(row.exit_high))
                    raw_level = min(levels)
                    px = exit_fill(raw_level, o, side, cfg.slippage_pct_per_fill)
                    reason = "INITIAL_2N_STOP" if stop_hit and stop_price <= float(row.exit_high) else "DONCHIAN_EXIT"
                    exited_this_bar = True

            if exited_this_bar:
                exit_fee = qty * px * cfg.fee_rate_per_side
                gross = (px - entry_price) * qty if side == 1 else (entry_price - px) * qty
                cash += gross - exit_fee
                net_pnl = cash - entry_cash_before
                risk_per_coin = cfg.initial_stop_n * initial_n
                mfe_r = ((max_fav - entry_price) / risk_per_coin) if side == 1 else ((entry_price - max_fav) / risk_per_coin)
                mae_r = ((entry_price - max_adv) / risk_per_coin) if side == 1 else ((max_adv - entry_price) / risk_per_coin)
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "type": "LONG" if side == 1 else "SHORT",
                        "entry": entry_price,
                        "exit": px,
                        "qty": qty,
                        "n_at_entry": initial_n,
                        "initial_stop": stop_price,
                        "gross_pnl": gross,
                        "fee": entry_fee + exit_fee,
                        "pnl": net_pnl,
                        "return_on_pretrade_equity": net_pnl / max(entry_cash_before, 1e-12),
                        "mfe_r": mfe_r,
                        "mae_r": mae_r,
                        "holding_hours": int((ts - entry_time).total_seconds() / 3600) if entry_time is not None else 0,
                        "note": reason,
                        "capital": cash,
                    }
                )
                side = 0
                qty = 0.0
                entry_price = 0.0
                entry_time = None
                last_exit_time = ts

        # -------------------------------------------------------------
        # Entry. A breakout stop order is considered resting before bar.
        # Skip bars where both sides break because intrabar path is unknown.
        # -------------------------------------------------------------
        if side == 0 and not exited_this_bar:
            cooldown_ok = last_exit_time is None or (ts - last_exit_time).total_seconds() >= cfg.cooldown_hours * 3600
            if cooldown_ok:
                long_touch = h >= float(row.entry_high)
                short_touch = l <= float(row.entry_low)
                if cfg.side_mode == "long":
                    short_touch = False
                elif cfg.side_mode == "short":
                    long_touch = False

                if long_touch ^ short_touch:
                    new_side = 1 if long_touch else -1
                    level = float(row.entry_high) if new_side == 1 else float(row.entry_low)
                    px = entry_fill(level, o, new_side, cfg.slippage_pct_per_fill)
                    stop_dist = cfg.initial_stop_n * n
                    if stop_dist > 0 and math.isfinite(stop_dist):
                        pre_entry_equity = cash
                        risk_usdt = pre_entry_equity * cfg.risk_per_trade
                        q_risk = risk_usdt / stop_dist
                        q_notional = pre_entry_equity * cfg.max_notional_mult / px
                        q = min(q_risk, q_notional)
                        if q > 0 and math.isfinite(q):
                            fee = q * px * cfg.fee_rate_per_side
                            entry_cash_before = cash
                            cash -= fee
                            side = new_side
                            qty = q
                            entry_price = px
                            entry_time = ts
                            entry_fee = fee
                            initial_n = n
                            stop_price = px - stop_dist if side == 1 else px + stop_dist
                            max_fav = px
                            max_adv = px

                            # Conservative same-bar path rule: with OHLC only we
                            # cannot know whether the bar hit the protective stop
                            # before or after the breakout entry. If the stop is
                            # also inside this bar, assume entry first then stop.
                            same_bar_stop = (side == 1 and l <= stop_price) or (side == -1 and h >= stop_price)
                            if same_bar_stop:
                                stop_px = stop_price * (1 - cfg.slippage_pct_per_fill) if side == 1 else stop_price * (1 + cfg.slippage_pct_per_fill)
                                exit_fee = qty * stop_px * cfg.fee_rate_per_side
                                gross = (stop_px - entry_price) * qty if side == 1 else (entry_price - stop_px) * qty
                                cash += gross - exit_fee
                                net_pnl = cash - entry_cash_before
                                trades.append(
                                    {
                                        "entry_time": entry_time,
                                        "exit_time": ts,
                                        "type": "LONG" if side == 1 else "SHORT",
                                        "entry": entry_price,
                                        "exit": stop_px,
                                        "qty": qty,
                                        "n_at_entry": initial_n,
                                        "initial_stop": stop_price,
                                        "gross_pnl": gross,
                                        "fee": entry_fee + exit_fee,
                                        "pnl": net_pnl,
                                        "return_on_pretrade_equity": net_pnl / max(entry_cash_before, 1e-12),
                                        "mfe_r": 0.0,
                                        "mae_r": 1.0,
                                        "holding_hours": 0,
                                        "note": "SAME_BAR_2N_STOP_CONSERVATIVE",
                                        "capital": cash,
                                    }
                                )
                                side = 0
                                qty = 0.0
                                entry_price = 0.0
                                entry_time = None
                                last_exit_time = ts

        eq = marked_equity(cash, side, qty, entry_price, c)
        eq_rows.append({"time": ts, "equity": eq, "cash": cash, "position_side": side, "close": c})

        if i == 0 or (i + 1) % max(1, len(rows) // 10) == 0 or i == len(rows) - 1:
            pct = (i + 1) / len(rows) * 100
            print(f"  progress {pct:5.1f}% | {ts} | trades={len(trades)} | equity={eq:.2f}")

    # Force close at final close so terminal capital is comparable.
    if side != 0 and len(rows) > 0:
        ts = idx[-1]
        o = float(rows[-1].close)
        px = o * (1 - cfg.slippage_pct_per_fill) if side == 1 else o * (1 + cfg.slippage_pct_per_fill)
        exit_fee = qty * px * cfg.fee_rate_per_side
        gross = (px - entry_price) * qty if side == 1 else (entry_price - px) * qty
        cash += gross - exit_fee
        net_pnl = cash - entry_cash_before
        risk_per_coin = cfg.initial_stop_n * initial_n
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": ts,
                "type": "LONG" if side == 1 else "SHORT",
                "entry": entry_price,
                "exit": px,
                "qty": qty,
                "n_at_entry": initial_n,
                "initial_stop": stop_price,
                "gross_pnl": gross,
                "fee": entry_fee + exit_fee,
                "pnl": net_pnl,
                "return_on_pretrade_equity": net_pnl / max(entry_cash_before, 1e-12),
                "mfe_r": float("nan"),
                "mae_r": float("nan"),
                "holding_hours": int((ts - entry_time).total_seconds() / 3600) if entry_time is not None else 0,
                "note": "FORCE_CLOSE_END",
                "capital": cash,
            }
        )
        eq_rows[-1]["equity"] = cash
        eq_rows[-1]["cash"] = cash
        eq_rows[-1]["position_side"] = 0

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(eq_rows).set_index("time")
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["drawdown_pct"] = equity_df["equity"] / equity_df["peak"] - 1.0
    return trades_df, equity_df


def period_returns(equity: pd.DataFrame, freq: str) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame(columns=["period", "start_equity", "end_equity", "return_pct"])
    s = equity["equity"]
    starts = s.resample(freq).first()
    ends = s.resample(freq).last()
    out = pd.DataFrame({"start_equity": starts, "end_equity": ends}).dropna()
    out["return_pct"] = (out["end_equity"] / out["start_equity"] - 1) * 100
    out.index.name = "period"
    return out.reset_index()


def summarize(trades: pd.DataFrame, equity: pd.DataFrame, cfg: BacktestConfig, variant: VariantConfig) -> dict[str, Any]:
    start = equity.index[0]
    end = equity.index[-1]
    years = max((end - start).total_seconds() / (365.25 * 86400), 1e-9)
    final_equity = float(equity["equity"].iloc[-1])
    total_return = final_equity / cfg.initial_capital - 1
    cagr = (final_equity / cfg.initial_capital) ** (1 / years) - 1 if final_equity > 0 else -1.0
    max_dd = -float(equity["drawdown_pct"].min()) if not equity.empty else 0.0

    daily = equity["equity"].resample("1D").last().dropna()
    daily_ret = daily.pct_change().dropna()
    losing_days = max_consecutive_true(daily_ret < 0)

    monthly = period_returns(equity, "ME")
    positive_month_pct = float((monthly["return_pct"] > 0).mean() * 100) if not monthly.empty else 0.0

    if trades.empty:
        return {
            "variant": variant.name,
            "side_mode": cfg.side_mode,
            "total_trades": 0,
            "total_return_pct": total_return * 100,
            "cagr_pct": cagr * 100,
            "max_drawdown_pct": max_dd * 100,
            "profit_factor": 0.0,
            "win_rate_pct": 0.0,
            "expectancy_pct": 0.0,
            "max_consecutive_losing_trades": 0,
            "max_consecutive_losing_days": losing_days,
            "max_no_entry_days": max_entry_gap_days(trades, start, end),
            "positive_month_pct": positive_month_pct,
            "final_equity": final_equity,
            "total_fees": 0.0,
        }

    wins = trades.loc[trades["pnl"] > 0, "pnl"]
    losses = trades.loc[trades["pnl"] <= 0, "pnl"]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    loss_streak = max_consecutive_true(trades["pnl"] <= 0)

    return {
        "variant": variant.name,
        "side_mode": cfg.side_mode,
        "entry_days": variant.entry_days,
        "exit_days": variant.exit_days,
        "n_period_days": cfg.n_period_days,
        "initial_stop_n": cfg.initial_stop_n,
        "total_trades": int(len(trades)),
        "long_trades": int((trades["type"] == "LONG").sum()),
        "short_trades": int((trades["type"] == "SHORT").sum()),
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": max_dd * 100,
        "profit_factor": pf if math.isfinite(pf) else 999.0,
        "win_rate_pct": float((trades["pnl"] > 0).mean() * 100),
        "expectancy_pct": float(trades["return_on_pretrade_equity"].mean() * 100),
        "avg_holding_hours": float(trades["holding_hours"].mean()),
        "avg_mfe_r": float(trades["mfe_r"].mean(skipna=True)),
        "avg_mae_r": float(trades["mae_r"].mean(skipna=True)),
        "max_consecutive_losing_trades": loss_streak,
        "max_consecutive_losing_days": losing_days,
        "max_no_entry_days": max_entry_gap_days(trades, start, end),
        "positive_month_pct": positive_month_pct,
        "final_equity": final_equity,
        "total_fees": float(trades["fee"].sum()),
    }


def write_report(out_dir: Path, summary: dict[str, Any], yearly: pd.DataFrame, monthly: pd.DataFrame, stress: pd.DataFrame, cfg: BacktestConfig, variant: VariantConfig) -> None:
    def f(key: str, digits: int = 2) -> str:
        v = summary.get(key)
        return f"{float(v):.{digits}f}" if v is not None else "n/a"

    lines = [
        f"# ETH Turtle Baseline V1 — {variant.name}",
        "",
        "## Result",
        "",
        f"- Trades: **{summary['total_trades']}**",
        f"- Total return: **{f('total_return_pct')}%**",
        f"- CAGR: **{f('cagr_pct')}%**",
        f"- Max drawdown (marked equity): **{f('max_drawdown_pct')}%**",
        f"- Profit factor: **{f('profit_factor')}**",
        f"- Win rate: **{f('win_rate_pct')}%**",
        f"- Max consecutive losing trades: **{summary['max_consecutive_losing_trades']}**",
        f"- Max consecutive losing days: **{summary['max_consecutive_losing_days']}**",
        f"- Max no-entry gap: **{f('max_no_entry_days')} days**",
        f"- Positive months: **{f('positive_month_pct')}%**",
        "",
        "## Rules",
        "",
        f"- Entry: {variant.entry_days}-day Donchian breakout.",
        f"- Exit: {variant.exit_days}-day opposite Donchian channel.",
        f"- N: {cfg.n_period_days}-day Wilder-style EMA of daily True Range, lagged one completed day.",
        f"- Initial stop: {cfg.initial_stop_n}N.",
        f"- Risk per trade: {cfg.risk_per_trade * 100:.2f}% of equity, capped at {cfg.max_notional_mult:.2f}x notional.",
        f"- Fee: {cfg.fee_rate_per_side * 100:.4f}% per side; slippage: {cfg.slippage_pct_per_fill * 100:.4f}% per fill.",
        f"- Side mode: {cfg.side_mode}.",
        "- No pyramiding in V1.",
        "- Funding is intentionally not fabricated and is not included in this V1 report.",
        "",
        "## Yearly",
        "",
        yearly.to_markdown(index=False) if not yearly.empty else "No yearly rows.",
        "",
        "## Cost stress",
        "",
        stress.to_markdown(index=False) if not stress.empty else "No stress rows.",
        "",
        "## Interpretation gate",
        "",
        "Do not promote this strategy because one aggregate return is positive. Prefer variants that stay alive across years, remain profitable at 2x costs, keep drawdown acceptable, and have enough trades to be meaningful.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_one(hourly: pd.DataFrame, variant: VariantConfig, cfg: BacktestConfig, out_dir: Path, do_stress: bool = True) -> dict[str, Any]:
    print(f"\n=== {variant.name} | sides={cfg.side_mode} ===")
    feat = build_features(hourly, variant, cfg)
    if feat.empty:
        raise RuntimeError(f"No usable features for {variant.name}; expand the start-date warmup.")
    print(f"Feature rows: {len(feat)} | {feat.index[0]} -> {feat.index[-1]}")

    trades, equity = run_backtest(feat, cfg)
    summary = summarize(trades, equity, cfg, variant)
    yearly = period_returns(equity, "YE")
    monthly = period_returns(equity, "ME")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    trades.to_csv(out_dir / "trades.csv", index=False)
    equity.to_csv(out_dir / "equity.csv")
    yearly.to_csv(out_dir / "yearly.csv", index=False)
    monthly.to_csv(out_dir / "monthly.csv", index=False)

    stress_rows: list[dict[str, Any]] = []
    if do_stress:
        for mult in (1.0, 2.0, 3.0):
            stress_cfg = BacktestConfig(
                **{
                    **asdict(cfg),
                    "fee_rate_per_side": cfg.fee_rate_per_side * mult,
                    "slippage_pct_per_fill": cfg.slippage_pct_per_fill * mult,
                }
            )
            t, e = run_backtest(feat, stress_cfg)
            s = summarize(t, e, stress_cfg, variant)
            stress_rows.append(
                {
                    "cost_mult": mult,
                    "trades": s["total_trades"],
                    "total_return_pct": s["total_return_pct"],
                    "cagr_pct": s["cagr_pct"],
                    "max_drawdown_pct": s["max_drawdown_pct"],
                    "profit_factor": s["profit_factor"],
                }
            )
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(out_dir / "cost_stress.csv", index=False)

    config_blob = {"strategy": "ETH_TURTLE_BASELINE_V1", "variant": asdict(variant), "backtest": asdict(cfg)}
    (out_dir / "run_config.json").write_text(json.dumps(config_blob, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir, summary, yearly, monthly, stress, cfg, variant)

    print(
        f"DONE {variant.name}: trades={summary['total_trades']} | return={summary['total_return_pct']:.2f}% | "
        f"CAGR={summary['cagr_pct']:.2f}% | MDD={summary['max_drawdown_pct']:.2f}% | PF={summary['profit_factor']:.2f}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETH Turtle Baseline V1 backtest")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2022-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--variant", choices=["fast", "s1", "s2", "all"], default="all")
    p.add_argument("--sides", choices=["both", "long", "short"], default="both")
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--risk-per-trade", type=float, default=0.01)
    p.add_argument("--max-notional-mult", type=float, default=3.0)
    p.add_argument("--fee-rate-per-side", type=float, default=0.00055)
    p.add_argument("--slippage-pct-per-fill", type=float, default=0.00020)
    p.add_argument("--initial-stop-n", type=float, default=2.0)
    p.add_argument("--cooldown-hours", type=int, default=0)
    p.add_argument("--out-dir", default="data/reports/research/trend/eth_turtle_baseline_v1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = BacktestConfig(
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        max_notional_mult=args.max_notional_mult,
        fee_rate_per_side=args.fee_rate_per_side,
        slippage_pct_per_fill=args.slippage_pct_per_fill,
        initial_stop_n=args.initial_stop_n,
        cooldown_hours=args.cooldown_hours,
        side_mode=args.sides,
    )

    print(f"Loading {cfg.symbol} {cfg.timeframe}: {args.start_date} -> {args.end_date}")
    hourly = load_data(cfg.symbol, args.start_date, args.end_date, cfg.timeframe)
    print(f"Loaded {len(hourly)} rows: {hourly.index[0]} -> {hourly.index[-1]}")

    names = ["fast", "s1", "s2"] if args.variant == "all" else [args.variant]
    all_summaries: list[dict[str, Any]] = []
    root = Path(args.out_dir)
    for name in names:
        variant = VARIANTS[name]
        variant_dir = root / f"{name}_{args.sides}"
        all_summaries.append(run_one(hourly, variant, cfg, variant_dir, do_stress=True))

    root.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(all_summaries)
    comparison.to_csv(root / f"comparison_{args.sides}.csv", index=False)
    print("\n=== COMPARISON ===")
    cols = [
        "variant", "total_trades", "total_return_pct", "cagr_pct", "max_drawdown_pct",
        "profit_factor", "win_rate_pct", "max_consecutive_losing_trades",
        "max_consecutive_losing_days", "max_no_entry_days", "positive_month_pct",
    ]
    print(comparison[cols].to_string(index=False))
    print(f"\nReports: {root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
