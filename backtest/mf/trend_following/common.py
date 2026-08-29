#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared plumbing for the ETH trend-following baseline suite.

Alpha definitions stay in the six strategy modules.  This module only owns
common CLI parameters, local data loading, causal rolling helpers, execution,
and report/output wiring so all strategies are compared on identical costs and
risk assumptions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.backtest_common.execution import apply_entry_slippage, apply_exit_slippage
from src.backtest_common.ohlcv_backtest import (
    emit_signal_report,
    print_signal_summary,
    summarize_signal_backtest,
    write_signal_outputs,
)
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader

DEFAULT_SYMBOL = "ETH-USDT-SWAP"
DEFAULT_TIMEFRAME = "15m"
DEFAULT_WARMUP_START = "2022-01-01"
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-06-30"
DEFAULT_REPORT_ROOT = "data/reports/backtest/mf/trend_following"


@dataclass(frozen=True)
class StrategySpec:
    strategy_name: str
    build_features: Callable[[pd.DataFrame], pd.DataFrame]
    audit_cols: Sequence[str]
    trailing_atr_mult: float = 3.0
    trail_after_r: float = 1.0
    max_hold_bars: int = 672  # 7 days on 15m bars
    min_stop_pct: float = 0.003
    max_stop_pct: float = 0.03


def make_parser(description: str, strategy_name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    p.add_argument("--warmup-start-date", default=DEFAULT_WARMUP_START)
    p.add_argument("--start-date", default=DEFAULT_START)
    p.add_argument("--end-date", default=DEFAULT_END)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--no-build-missing", action="store_true")
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--risk-per-trade", type=float, default=0.01)
    p.add_argument("--max-notional-mult", type=float, default=3.0)
    p.add_argument("--fee-rate-per-side", type=float, default=0.00055)
    p.add_argument("--slippage-rate-per-side", type=float, default=0.00020)
    p.add_argument("--out-dir", default=f"{DEFAULT_REPORT_ROOT}/{strategy_name}")
    p.add_argument("--write-full-audit", action="store_true")
    return p


def inclusive_end(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if len(str(value).strip()) <= 10:
        return ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return ts


def load_bars(args: argparse.Namespace) -> pd.DataFrame:
    kwargs: dict[str, object] = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "db_name": args.db_name,
    }
    if args.data_dir:
        kwargs["data_dir"] = Path(args.data_dir)
    loader = OKXTradeBarLoader(**kwargs)
    end_ts = inclusive_end(args.end_date)
    print(
        f"[load] {args.symbol} trade bars {args.timeframe} "
        f"{args.warmup_start_date} -> {end_ts}",
        flush=True,
    )
    bars = loader.fetch_data_by_date_range(
        args.warmup_start_date,
        end_ts,
        chunksize=args.chunksize,
        force_rebuild=bool(args.force_rebuild),
        cvd_mode="range",
        build_missing=not bool(args.no_build_missing),
    )
    if bars.empty:
        raise RuntimeError("OKXTradeBarLoader returned no rows")
    bars = bars.sort_index().copy()
    bars.index = pd.to_datetime(bars.index)
    required = ("open", "high", "low", "close", "volume")
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise RuntimeError(f"trade bar data missing required columns: {missing}")
    for col in required:
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    bars = bars.dropna(subset=list(required))
    print(f"[load] rows={len(bars):,} range={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


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


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    # EWM is causal: every value uses current/past closed bars only.
    return true_range(df).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def crossed_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossed_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def state_entry_signal(long_state: pd.Series, short_state: pd.Series) -> pd.Series:
    long_fire = long_state.fillna(False) & ~long_state.shift(1, fill_value=False)
    short_fire = short_state.fillna(False) & ~short_state.shift(1, fill_value=False)
    return pd.Series(np.select([long_fire, short_fire], [1, -1], default=0), index=long_state.index, dtype="int8")


def atr_stop(close: pd.Series, atr_value: pd.Series, signal: pd.Series, mult: float = 2.0) -> pd.Series:
    return pd.Series(
        np.where(signal > 0, close - mult * atr_value, np.where(signal < 0, close + mult * atr_value, np.nan)),
        index=close.index,
        dtype=float,
    )


def _valid_stop(entry: float, stop: float, side: int, *, min_stop_pct: float, max_stop_pct: float) -> tuple[float, bool]:
    if not np.isfinite(entry) or not np.isfinite(stop) or entry <= 0:
        return stop, False
    if side > 0:
        if stop >= entry:
            return stop, False
        pct = (entry - stop) / entry
        if pct < min_stop_pct:
            stop = entry * (1.0 - min_stop_pct)
            pct = min_stop_pct
    else:
        if stop <= entry:
            return stop, False
        pct = (stop - entry) / entry
        if pct < min_stop_pct:
            stop = entry * (1.0 + min_stop_pct)
            pct = min_stop_pct
    return stop, bool(pct <= max_stop_pct)


def run_causal_trend_backtest(
    df: pd.DataFrame,
    *,
    initial_capital: float,
    risk_per_trade: float,
    max_notional_mult: float,
    fee_rate_per_side: float,
    slippage_rate_per_side: float,
    min_stop_pct: float,
    max_stop_pct: float,
    max_hold_bars: int,
    trailing_atr_mult: float,
    trail_after_r: float,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """One-position, next-open, causally trailed OHLC backtest.

    Critical timing rule: a trailing stop calculated from bar ``i`` close/ATR
    only becomes active for bar ``i+1``.  It can therefore never use the final
    close of a bar to decide an intrabar stop on that same bar.
    """
    if len(df) < 2:
        return [], pd.DataFrame()

    rows = list(df.itertuples())
    idx = df.index
    capital = float(initial_capital)
    peak = capital
    trades: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []

    in_pos = False
    side = 0
    entry_i = -1
    entry_time: pd.Timestamp | None = None
    entry_price = 0.0
    initial_stop = 0.0
    active_stop = 0.0
    risk_per_coin = 0.0
    qty = 0.0
    entry_fee = 0.0
    max_fav = 0.0
    max_adv = 0.0
    pending_exit = False
    pending_exit_note = ""
    block_entry_i = -1

    def close_trade(exit_i: int, raw_exit: float, note: str) -> None:
        nonlocal capital, peak, in_pos, side, block_entry_i
        exit_price = apply_exit_slippage(float(raw_exit), side, slippage_rate_per_side)
        exit_fee = qty * exit_price * fee_rate_per_side
        if side > 0:
            pnl = (exit_price - entry_price) * qty - entry_fee - exit_fee
            mfe_r = (max_fav - entry_price) / risk_per_coin
            mae_r = (entry_price - max_adv) / risk_per_coin
        else:
            pnl = (entry_price - exit_price) * qty - entry_fee - exit_fee
            mfe_r = (entry_price - max_fav) / risk_per_coin
            mae_r = (max_adv - entry_price) / risk_per_coin
        before = capital
        capital += pnl
        peak = max(peak, capital)
        hold_bars = max(0, exit_i - entry_i)
        if len(idx) >= 3:
            seconds = pd.Series(idx).diff().dt.total_seconds().dropna().median()
            holding_hours = hold_bars * float(seconds) / 3600.0 if np.isfinite(seconds) else float(hold_bars)
        else:
            holding_hours = float(hold_bars)
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": idx[exit_i],
                "type": "LONG" if side > 0 else "SHORT",
                "entry": entry_price,
                "exit": exit_price,
                "initial_sl": initial_stop,
                "final_sl": active_stop,
                "target": np.nan,
                "qty": qty,
                "pnl": pnl,
                "fee": entry_fee + exit_fee,
                "capital": capital,
                "return_pct": pnl / max(before, 1e-12),
                "mfe_r": round(float(mfe_r), 4),
                "mae_r": round(float(mae_r), 4),
                "sl_pct": round(abs(entry_price - initial_stop) / entry_price * 100.0, 4),
                "holding_bars": int(hold_bars),
                "holding_hours": round(float(holding_hours), 4),
                "note": note,
            }
        )
        in_pos = False
        block_entry_i = exit_i
        side = 0

    for i in range(len(rows)):
        row = rows[i]
        ts = idx[i]

        # A close-based exit decision from bar i-1 executes now at bar i open.
        if in_pos and pending_exit:
            close_trade(i, float(row.open), pending_exit_note)
            pending_exit = False
            pending_exit_note = ""

        if in_pos:
            open_ = float(row.open)
            high = float(row.high)
            low = float(row.low)
            close = float(row.close)

            if side > 0:
                max_fav = max(max_fav, high)
                max_adv = min(max_adv, low)
                stop_hit = low <= active_stop
                gap_exit = min(open_, active_stop)
            else:
                max_fav = min(max_fav, low)
                max_adv = max(max_adv, high)
                stop_hit = high >= active_stop
                gap_exit = max(open_, active_stop)

            if stop_hit:
                close_trade(i, gap_exit, "STOP")
            else:
                # Everything below is known only after this bar has closed and
                # therefore affects the next bar, never this bar's path.
                if side > 0:
                    mfe_r = (max_fav - entry_price) / risk_per_coin
                else:
                    mfe_r = (entry_price - max_fav) / risk_per_coin
                atr_value = float(getattr(row, "atr14", np.nan))
                if mfe_r >= trail_after_r and np.isfinite(atr_value) and atr_value > 0:
                    candidate = close - side * trailing_atr_mult * atr_value
                    if side > 0:
                        active_stop = max(active_stop, candidate)
                    else:
                        active_stop = min(active_stop, candidate)

                sig = int(getattr(row, "signal", 0))
                hold_bars = i - entry_i + 1
                if sig == -side:
                    pending_exit = True
                    pending_exit_note = "OPPOSITE_SIGNAL_NEXT_OPEN"
                elif hold_bars >= max_hold_bars:
                    pending_exit = True
                    pending_exit_note = "MAX_HOLD_NEXT_OPEN"

        # Signals from the current *closed* bar can only enter at next bar open.
        if not in_pos and i < len(rows) - 1 and i != block_entry_i:
            sig = int(getattr(row, "signal", 0))
            if sig != 0:
                next_open = float(rows[i + 1].open)
                entry = apply_entry_slippage(next_open, sig, slippage_rate_per_side)
                raw_stop = float(getattr(row, "stop", np.nan))
                stop, ok = _valid_stop(
                    entry, raw_stop, sig, min_stop_pct=min_stop_pct, max_stop_pct=max_stop_pct
                )
                if ok:
                    rpc = abs(entry - stop)
                    risk_usdt = capital * risk_per_trade
                    q = min(risk_usdt / rpc, (capital * max_notional_mult) / entry)
                    if np.isfinite(q) and q > 0:
                        in_pos = True
                        side = sig
                        entry_i = i + 1
                        entry_time = idx[i + 1]
                        entry_price = entry
                        initial_stop = stop
                        active_stop = stop
                        risk_per_coin = rpc
                        qty = q
                        entry_fee = qty * entry_price * fee_rate_per_side
                        max_fav = entry_price
                        max_adv = entry_price
                        pending_exit = False
                        pending_exit_note = ""

        equity_rows.append(
            {
                "time": ts,
                "capital": capital,
                "drawdown_pct": (peak - capital) / peak if peak > 0 else 0.0,
            }
        )

    if in_pos:
        # Dataset end is not a signal; use the final known close only to settle
        # the accounting record.
        close_trade(len(rows) - 1, float(rows[-1].close), "FORCE_CLOSE_END")
        if equity_rows:
            equity_rows[-1]["capital"] = capital
            equity_rows[-1]["drawdown_pct"] = (peak - capital) / peak if peak > 0 else 0.0

    equity = pd.DataFrame(equity_rows).set_index("time") if equity_rows else pd.DataFrame()
    return trades, equity


def run_spec(
    args: argparse.Namespace,
    spec: StrategySpec,
    *,
    bars: pd.DataFrame | None = None,
    emit_report: bool = True,
) -> dict[str, object]:
    base = bars if bars is not None else load_bars(args)
    print(f"[features] {spec.strategy_name}", flush=True)
    features = spec.build_features(base.copy())
    if "signal" not in features.columns or "stop" not in features.columns or "atr14" not in features.columns:
        raise RuntimeError("strategy feature builder must produce signal, stop and atr14")

    start_ts = pd.Timestamp(args.start_date)
    end_ts = inclusive_end(args.end_date)
    features = features.loc[(features.index >= start_ts) & (features.index <= end_ts)].copy()
    if features.empty:
        raise RuntimeError(f"no feature rows in backtest window {start_ts} -> {end_ts}")

    signal_count = int((pd.to_numeric(features["signal"], errors="coerce").fillna(0) != 0).sum())
    print(f"[backtest] signals={signal_count:,} closed-bar -> next-open", flush=True)
    trades, equity = run_causal_trend_backtest(
        features,
        initial_capital=float(args.initial_capital),
        risk_per_trade=float(args.risk_per_trade),
        max_notional_mult=float(args.max_notional_mult),
        fee_rate_per_side=float(args.fee_rate_per_side),
        slippage_rate_per_side=float(args.slippage_rate_per_side),
        min_stop_pct=float(spec.min_stop_pct),
        max_stop_pct=float(spec.max_stop_pct),
        max_hold_bars=int(spec.max_hold_bars),
        trailing_atr_mult=float(spec.trailing_atr_mult),
        trail_after_r=float(spec.trail_after_r),
    )
    summary = summarize_signal_backtest(trades, equity, float(args.initial_capital), signal_count=signal_count)
    summary = {
        "strategy": spec.strategy_name,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "warmup_start_date": str(args.warmup_start_date),
        "start_date": str(args.start_date),
        "end_date": str(args.end_date),
        "fee_rate_per_side": float(args.fee_rate_per_side),
        "slippage_rate_per_side": float(args.slippage_rate_per_side),
        "risk_per_trade": float(args.risk_per_trade),
        **summary,
    }
    out_dir = Path(args.out_dir)
    write_signal_outputs(
        features,
        trades,
        equity,
        summary,
        out_dir,
        strategy_name=spec.strategy_name,
        audit_cols=spec.audit_cols,
        write_full_audit=bool(args.write_full_audit),
    )
    if emit_report:
        # Existing project report writes yearly breakdown + full trade report.
        emit_signal_report(trades, features, args, out_dir, strategy_name=spec.strategy_name)
    print_signal_summary(summary, out_dir, strategy_name=spec.strategy_name)
    return summary
