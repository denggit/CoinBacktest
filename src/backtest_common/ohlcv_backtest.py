#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generic one-position OHLCV signal backtest helper.

Strategy files should build a feature DataFrame with at least:
    open/high/low/close/signal/stop
Then call run_signal_backtest(). Strategy-specific alpha logic stays in the
strategy file; sizing, slippage, fee, R-multiple summary, report emission and
CSV output stay here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.backtest_common.execution import apply_entry_slippage, apply_exit_slippage
from src.backtest_common.reporting import summarize_r_trades
from src.utils.report import print_full_report


@dataclass(frozen=True)
class SignalBacktestParams:
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.003
    max_notional_mult: float = 3.0
    fee_rate: float = 0.0005
    slippage_pct: float = 0.0002
    risk_mult_col: str | None = None
    min_risk_mult: float = 0.0
    max_risk_mult: float = 2.0

    signal_col: str = "signal"
    stop_col: str = "stop"
    target_col: str | None = None
    target_r: float = 2.0

    min_stop_pct: float = 0.003
    max_stop_pct: float = 0.03
    cooldown_bars: int = 0
    max_hold_bars: int = 96

    no_progress_bars: int = 0
    no_progress_min_mfe_r: float = 0.5

    exit_on_opposite_signal: bool = True
    trailing_atr_col: str | None = None
    trailing_atr_mult: float = 0.0
    trail_after_r: float = 1.0


def _valid_stop(entry: float, stop: float, side: int, params: SignalBacktestParams) -> tuple[float, float, bool, str]:
    if not math.isfinite(entry) or not math.isfinite(stop) or entry <= 0:
        return stop, 0.0, False, "BAD_STOP"
    if side == 1:
        if stop >= entry:
            return stop, 0.0, False, "STOP_ABOVE_LONG_ENTRY"
        stop_pct = (entry - stop) / entry
        if stop_pct < params.min_stop_pct:
            stop = entry * (1 - params.min_stop_pct)
            stop_pct = params.min_stop_pct
    else:
        if stop <= entry:
            return stop, 0.0, False, "STOP_BELOW_SHORT_ENTRY"
        stop_pct = (stop - entry) / entry
        if stop_pct < params.min_stop_pct:
            stop = entry * (1 + params.min_stop_pct)
            stop_pct = params.min_stop_pct
    if stop_pct > params.max_stop_pct:
        return stop, stop_pct, False, "STOP_TOO_WIDE"
    return stop, stop_pct, True, "OK"


def _target_from_row(row: Any, entry: float, stop: float, side: int, params: SignalBacktestParams) -> float:
    if params.target_col:
        value = getattr(row, params.target_col, None)
        if value is not None and math.isfinite(float(value)):
            target = float(value)
            if (side == 1 and target > entry) or (side == -1 and target < entry):
                return target
    risk = abs(entry - stop)
    return entry + side * params.target_r * risk


def run_signal_backtest(df: pd.DataFrame, params: SignalBacktestParams) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    capital = float(params.initial_capital)
    peak = capital
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    in_pos = False
    side = 0
    entry_i = -1
    entry_time = None
    entry_price = 0.0
    stop_price = 0.0
    initial_sl = 0.0
    target_price = 0.0
    risk_per_coin = 0.0
    qty = 0.0
    entry_fee = 0.0
    max_fav = 0.0
    max_adv = 0.0
    entry_risk_mult = 1.0
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
            hold_bars = i - entry_i

            if side == 1:
                max_fav = max(max_fav, high)
                max_adv = min(max_adv, low)
                mfe_r = (max_fav - entry_price) / risk_per_coin
                mae_r = (entry_price - max_adv) / risk_per_coin
            else:
                max_fav = min(max_fav, low)
                max_adv = max(max_adv, high)
                mfe_r = (entry_price - max_fav) / risk_per_coin
                mae_r = (max_adv - entry_price) / risk_per_coin

            if params.trailing_atr_col and params.trailing_atr_mult > 0 and mfe_r >= params.trail_after_r:
                atr_value = getattr(row, params.trailing_atr_col, None)
                if atr_value is not None and math.isfinite(float(atr_value)):
                    if side == 1:
                        stop_price = max(stop_price, close - params.trailing_atr_mult * float(atr_value))
                    else:
                        stop_price = min(stop_price, close + params.trailing_atr_mult * float(atr_value))

            if side == 1:
                stop_hit = low <= stop_price
                target_hit = high >= target_price
            else:
                stop_hit = high >= stop_price
                target_hit = low <= target_price

            exit_now = False
            exit_price = 0.0
            note = ""

            # Conservative ordering when one candle can touch both sides.
            if stop_hit:
                exit_now = True
                exit_price = apply_exit_slippage(stop_price, side, params.slippage_pct)
                note = "STOP"
            elif target_hit:
                exit_now = True
                exit_price = apply_exit_slippage(target_price, side, params.slippage_pct)
                note = "TARGET_R"
            elif params.no_progress_bars > 0 and hold_bars >= params.no_progress_bars and mfe_r < params.no_progress_min_mfe_r:
                exit_now = True
                exit_price = apply_exit_slippage(close, side, params.slippage_pct)
                note = "NO_PROGRESS_EXIT"
            elif params.exit_on_opposite_signal and int(getattr(row, params.signal_col, 0)) == -side:
                exit_now = True
                exit_price = apply_exit_slippage(close, side, params.slippage_pct)
                note = "OPPOSITE_SIGNAL_EXIT"
            elif hold_bars >= params.max_hold_bars:
                exit_now = True
                exit_price = apply_exit_slippage(close, side, params.slippage_pct)
                note = "MAX_HOLD_EXIT"

            if exit_now:
                exit_fee = qty * exit_price * params.fee_rate
                if side == 1:
                    pnl = (exit_price - entry_price) * qty - entry_fee - exit_fee
                else:
                    pnl = (entry_price - exit_price) * qty - entry_fee - exit_fee
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
                        "final_sl": stop_price,
                        "target": target_price,
                        "qty": qty,
                        "pnl": pnl,
                        "fee": entry_fee + exit_fee,
                        "capital": capital,
                        "return_pct": pnl / max(cap_before, 1e-12),
                        "mfe_r": round(float(mfe_r), 4),
                        "mae_r": round(float(mae_r), 4),
                        "sl_pct": round(abs(entry_price - initial_sl) / entry_price * 100, 4),
                        "holding_bars": int(hold_bars),
                        "holding_hours": round(float(_infer_holding_hours(df, hold_bars)), 4),
                        "risk_mult": float(entry_risk_mult),
                        "note": note,
                    }
                )
                in_pos = False
                side = 0
                last_exit_i = i

        if not in_pos and i - last_exit_i >= params.cooldown_bars:
            sig = int(getattr(row, params.signal_col, 0))
            if sig != 0:
                next_open = float(rows[i + 1].open)
                entry = apply_entry_slippage(next_open, sig, params.slippage_pct)
                raw_stop = getattr(row, params.stop_col, None)
                if raw_stop is None:
                    continue
                stop, stop_pct, ok, _ = _valid_stop(entry, float(raw_stop), sig, params)
                if not ok:
                    continue
                rpc = abs(entry - stop)
                risk_mult = 1.0
                if params.risk_mult_col:
                    raw_risk_mult = getattr(row, params.risk_mult_col, 1.0)
                    try:
                        risk_mult = float(raw_risk_mult)
                    except (TypeError, ValueError):
                        risk_mult = 1.0
                    if not math.isfinite(risk_mult):
                        risk_mult = 1.0
                    risk_mult = max(params.min_risk_mult, min(risk_mult, params.max_risk_mult))
                risk_usdt = capital * params.risk_per_trade * risk_mult
                q = risk_usdt / rpc
                q = min(q, (capital * params.max_notional_mult) / entry)
                if q > 0 and math.isfinite(q):
                    in_pos = True
                    side = sig
                    entry_i = i + 1
                    entry_time = idx[i + 1]
                    entry_price = entry
                    stop_price = stop
                    initial_sl = stop
                    target_price = _target_from_row(row, entry, stop, sig, params)
                    risk_per_coin = rpc
                    qty = q
                    entry_fee = qty * entry_price * params.fee_rate
                    max_fav = entry_price
                    max_adv = entry_price
                    entry_risk_mult = risk_mult

        equity_rows.append({"time": ts, "capital": capital, "drawdown_pct": (peak - capital) / peak if peak > 0 else 0.0})

    if in_pos:
        ts = idx[-1]
        close = float(df.iloc[-1]["close"])
        exit_price = apply_exit_slippage(close, side, params.slippage_pct)
        exit_fee = qty * exit_price * params.fee_rate
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
                "final_sl": stop_price,
                "target": target_price,
                "qty": qty,
                "pnl": pnl,
                "fee": entry_fee + exit_fee,
                "capital": capital,
                "return_pct": pnl / max(cap_before, 1e-12),
                "mfe_r": round(float(mfe_r), 4),
                "mae_r": round(float(mae_r), 4),
                "sl_pct": round(abs(entry_price - initial_sl) / entry_price * 100, 4),
                "holding_bars": int(len(df) - 1 - entry_i),
                "holding_hours": round(float(_infer_holding_hours(df, len(df) - 1 - entry_i)), 4),
                "risk_mult": float(entry_risk_mult),
                "note": "FORCE_CLOSE_END",
            }
        )

    equity = pd.DataFrame(equity_rows).set_index("time") if equity_rows else pd.DataFrame()
    return trades, equity


def _infer_holding_hours(df: pd.DataFrame, bars: int) -> float:
    if len(df.index) >= 3:
        seconds = pd.Series(df.index).diff().dt.total_seconds().dropna().median()
        if math.isfinite(float(seconds)) and seconds > 0:
            return bars * float(seconds) / 3600.0
    return float(bars)


def summarize_signal_backtest(trades: list[dict[str, Any]], equity: pd.DataFrame, initial_capital: float, signal_count: int | None = None) -> dict[str, Any]:
    summary = summarize_r_trades(trades, equity, initial_capital)
    if signal_count is not None:
        summary = {"signal_count": int(signal_count), **summary}
    return summary


def write_signal_outputs(
    features: pd.DataFrame,
    trades: list[dict[str, Any]],
    equity: pd.DataFrame,
    summary: dict[str, Any],
    out_dir: Path,
    *,
    strategy_name: str,
    audit_cols: Sequence[str] | None = None,
    write_full_audit: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / f"{strategy_name}_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / f"{strategy_name}_equity.csv")
    pd.Series(summary).to_json(out_dir / f"{strategy_name}_summary.json", force_ascii=False, indent=2)

    if audit_cols:
        cols = [c for c in audit_cols if c in features.columns]
    else:
        cols = [c for c in features.columns if c not in {"raw_json"}]
    signal_rows = features[features.get("signal", 0) != 0].copy() if "signal" in features.columns else features.iloc[0:0].copy()
    signal_rows[cols].to_csv(out_dir / f"{strategy_name}_signal_audit.csv")
    if write_full_audit:
        features[cols].to_csv(out_dir / f"{strategy_name}_full_audit.csv")


def emit_signal_report(trades: list[dict[str, Any]], features: pd.DataFrame, cfg: Any, out_dir: Path, *, strategy_name: str) -> None:
    if features.empty:
        return
    final_capital = float(trades[-1]["capital"]) if trades else float(cfg.initial_capital)
    total_days = max((features.index[-1] - features.index[0]).total_seconds() / 86400.0, 1.0 / 86400.0)
    print_full_report(
        trade_history=trades,
        df=features,
        initial_capital=cfg.initial_capital,
        capital=final_capital,
        strategy_name=strategy_name,
        total_days=total_days,
        ai_enabled=False,
        symbol=cfg.symbol,
        report_dir=out_dir,
    )


def print_signal_summary(summary: dict[str, Any], out_dir: Path, *, strategy_name: str) -> None:
    print("\n" + "=" * 88)
    print(f"ETH MF Backtest Summary | {strategy_name}")
    print("=" * 88)
    for k, v in summary.items():
        print(f"{k:>28}: {v}")
    print("-" * 88)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 88 + "\n")
