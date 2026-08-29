#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Execution, funding accounting, and performance metrics for Trend Pullback V1."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
import math

import numpy as np
import pandas as pd

from src.backtest_common.execution import apply_entry_slippage, apply_exit_slippage


@dataclass(frozen=True)
class ExecutionConfig:
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.01
    max_notional_mult: float = 3.0
    fee_rate_per_side: float = 0.00055
    slippage_rate_per_side: float = 0.00020
    min_stop_pct: float = 0.004
    max_stop_pct: float = 0.025
    cooldown_bars: int = 8  # 2h after an exit
    max_hold_bars: int = 288  # 72h on 15m bars
    no_progress_bars: int = 48  # 12h
    no_progress_min_mfe_r: float = 0.40
    breakeven_after_r: float = 0.80
    structure_trail_after_r: float = 1.25
    h1_trail_atr_buffer: float = 0.50
    funding_multiplier: float = 1.0
    side_mode: str = "both"  # both/long/short


def _valid_stop(entry: float, raw_stop: float, side: int, cfg: ExecutionConfig) -> tuple[float, bool]:
    if not np.isfinite(entry) or not np.isfinite(raw_stop) or entry <= 0:
        return raw_stop, False
    if side > 0:
        if raw_stop >= entry:
            return raw_stop, False
        pct = (entry - raw_stop) / entry
        stop = raw_stop
        if pct < cfg.min_stop_pct:
            stop = entry * (1.0 - cfg.min_stop_pct)
            pct = cfg.min_stop_pct
    else:
        if raw_stop <= entry:
            return raw_stop, False
        pct = (raw_stop - entry) / entry
        stop = raw_stop
        if pct < cfg.min_stop_pct:
            stop = entry * (1.0 + cfg.min_stop_pct)
            pct = cfg.min_stop_pct
    return float(stop), bool(pct <= cfg.max_stop_pct)


def _prepare_funding(funding: pd.DataFrame | None, features: pd.DataFrame) -> pd.DataFrame:
    if funding is None or funding.empty:
        return pd.DataFrame(columns=["funding_rate", "mark_price", "source"])
    out = funding.sort_index().copy()
    out.index = pd.to_datetime(out.index)
    if "funding_rate" not in out.columns:
        raise RuntimeError("funding frame missing funding_rate")
    out["funding_rate"] = pd.to_numeric(out["funding_rate"], errors="coerce")
    if "mark_price" not in out.columns:
        out["mark_price"] = np.nan
    out["mark_price"] = pd.to_numeric(out["mark_price"], errors="coerce")

    # Where the archive lacks markPrice, use only a completed 15m close.  A bar
    # starting at t is available at t+15m, so the asof key is bar_available_time.
    bars = features[["close"]].copy()
    bars["bar_available_time"] = bars.index + pd.Timedelta(minutes=15)
    right = bars.reset_index(names="bar_time").sort_values("bar_available_time")
    left = out.reset_index(names="funding_time").sort_values("funding_time")
    aligned = pd.merge_asof(
        left,
        right[["bar_available_time", "close"]],
        left_on="funding_time",
        right_on="bar_available_time",
        direction="backward",
        allow_exact_matches=True,
    )
    aligned["mark_price"] = aligned["mark_price"].where(aligned["mark_price"].notna(), aligned["close"])
    aligned = aligned.set_index("funding_time")
    return aligned[[c for c in ("funding_rate", "mark_price", "source") if c in aligned.columns]]


def run_backtest(
    features: pd.DataFrame,
    cfg: ExecutionConfig,
    *,
    funding: pd.DataFrame | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]:
    """Run one-position causal backtest with marked equity and funding.

    Close-derived exits and trailing-stop changes become actionable on the next
    15m bar.  Initial stop is known before entry and may be hit on the entry bar.
    Funding is booked at its published settlement time and never used as alpha.
    """
    if len(features) < 2:
        return [], pd.DataFrame(), []

    frame = features.sort_index().copy()
    rows = list(frame.itertuples())
    idx = frame.index
    funding_frame = _prepare_funding(funding, frame)
    f_times = funding_frame.index.to_numpy() if not funding_frame.empty else np.array([], dtype="datetime64[ns]")
    f_ptr = 0

    capital = float(cfg.initial_capital)
    peak_marked_equity = capital
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    funding_ledger: list[dict[str, Any]] = []

    in_pos = False
    side = 0
    entry_i = -1
    entry_time: pd.Timestamp | None = None
    signal_time: pd.Timestamp | None = None
    entry_price = 0.0
    initial_stop = 0.0
    active_stop = 0.0
    risk_per_coin = 0.0
    qty = 0.0
    entry_fee = 0.0
    trade_capital_before = capital
    trade_funding_pnl = 0.0
    trade_funding_events = 0
    max_fav = 0.0
    max_adv = 0.0
    pending_exit = False
    pending_reason = ""
    last_exit_i = -10**9
    audit_ctx: dict[str, Any] = {}

    def mark_equity(close: float, ts: pd.Timestamp) -> float:
        if not in_pos or entry_time is None or ts < pd.Timestamp(entry_time):
            return capital
        gross = side * (close - entry_price) * qty
        est_exit = qty * close * cfg.fee_rate_per_side
        return capital + gross - entry_fee - est_exit

    def close_trade(exit_i: int, raw_exit: float, reason: str) -> None:
        nonlocal capital, in_pos, side, last_exit_i, trade_funding_pnl, trade_funding_events
        exit_price = apply_exit_slippage(float(raw_exit), side, cfg.slippage_rate_per_side)
        exit_fee = qty * exit_price * cfg.fee_rate_per_side
        gross = side * (exit_price - entry_price) * qty
        trading_pnl = gross - entry_fee - exit_fee
        # Funding has already been booked into capital at each settlement.
        capital += trading_pnl
        net_trade_pnl = trading_pnl + trade_funding_pnl
        if side > 0:
            mfe_r = (max_fav - entry_price) / risk_per_coin
            mae_r = (entry_price - max_adv) / risk_per_coin
        else:
            mfe_r = (entry_price - max_fav) / risk_per_coin
            mae_r = (max_adv - entry_price) / risk_per_coin
        holding_hours = max(0, exit_i - entry_i) * 0.25
        trades.append(
            {
                "entry_time": entry_time,
                "signal_time": signal_time,
                "exit_time": idx[exit_i],
                "type": "LONG" if side > 0 else "SHORT",
                "entry": entry_price,
                "exit": exit_price,
                "initial_sl": initial_stop,
                "final_sl": active_stop,
                "qty": qty,
                "gross_price_pnl": gross,
                "trading_fee": entry_fee + exit_fee,
                "fee": entry_fee + exit_fee,
                "funding_pnl": trade_funding_pnl,
                "funding_events": int(trade_funding_events),
                "pnl": net_trade_pnl,
                "capital": capital,
                "return_pct": net_trade_pnl / max(trade_capital_before, 1e-12),
                "mfe_r": round(float(mfe_r), 4),
                "mae_r": round(float(mae_r), 4),
                "sl_pct": round(abs(entry_price - initial_stop) / entry_price * 100.0, 4),
                "holding_bars": int(max(0, exit_i - entry_i)),
                "holding_hours": round(float(holding_hours), 4),
                "note": reason,
                "same_bar_exit_flag": bool(entry_time is not None and pd.Timestamp(idx[exit_i]) == pd.Timestamp(entry_time)),
                "same_bar_stop_tp_both_hit_flag": False,
                **audit_ctx,
            }
        )
        in_pos = False
        last_exit_i = exit_i
        side = 0
        trade_funding_pnl = 0.0
        trade_funding_events = 0

    for i, row in enumerate(rows):
        ts = pd.Timestamp(idx[i])

        # Apply funding settlements through the current bar start before a
        # pending open-price exit.  At an exact exit boundary, adverse funding
        # is charged while favorable funding is not credited.  At an exact
        # entry boundary, the same conservative rule is used.
        while f_ptr < len(f_times) and pd.Timestamp(f_times[f_ptr]) <= ts:
            fts = pd.Timestamp(f_times[f_ptr])
            if in_pos and entry_time is not None and fts >= pd.Timestamp(entry_time):
                frow = funding_frame.iloc[f_ptr]
                rate = float(frow.get("funding_rate", np.nan))
                mark = float(frow.get("mark_price", np.nan))
                if np.isfinite(rate) and np.isfinite(mark) and mark > 0:
                    raw = -side * qty * mark * rate * cfg.funding_multiplier
                    exact_entry = fts == pd.Timestamp(entry_time)
                    exact_exit = bool(pending_exit and fts == ts)
                    include = raw < 0 or (not exact_entry and not exact_exit)
                    if include:
                        capital += raw
                        trade_funding_pnl += raw
                        trade_funding_events += 1
                        funding_ledger.append(
                            {
                                "time": fts,
                                "side": "LONG" if side > 0 else "SHORT",
                                "rate": rate,
                                "mark_price": mark,
                                "qty": qty,
                                "funding_pnl": raw,
                                "source": frow.get("source", "UNKNOWN"),
                            }
                        )
            f_ptr += 1

        # A close-based exit decided on the previous closed bar executes now.
        if in_pos and pending_exit:
            close_trade(i, float(row.open), pending_reason)
            pending_exit = False
            pending_reason = ""

        if in_pos:
            open_ = float(row.open)
            high = float(row.high)
            low = float(row.low)
            close = float(row.close)
            if side > 0:
                max_fav = max(max_fav, high)
                max_adv = min(max_adv, low)
                stop_hit = low <= active_stop
                stop_raw = min(open_, active_stop)
            else:
                max_fav = min(max_fav, low)
                max_adv = max(max_adv, high)
                stop_hit = high >= active_stop
                stop_raw = max(open_, active_stop)

            if stop_hit:
                close_trade(i, stop_raw, "STOP")
            else:
                mfe_r = (max_fav - entry_price) / risk_per_coin if side > 0 else (entry_price - max_fav) / risk_per_coin

                # Stop changes below are known only after this 15m bar closes,
                # hence they affect the next bar path.
                if mfe_r >= cfg.breakeven_after_r:
                    cost_buffer = 2.0 * (cfg.fee_rate_per_side + cfg.slippage_rate_per_side)
                    be = entry_price * (1.0 + side * cost_buffer)
                    active_stop = max(active_stop, be) if side > 0 else min(active_stop, be)

                h1_ema20 = float(getattr(row, "h1_ema20", np.nan))
                h1_atr = float(getattr(row, "h1_atr14", np.nan))
                if mfe_r >= cfg.structure_trail_after_r and np.isfinite(h1_ema20) and np.isfinite(h1_atr):
                    candidate = h1_ema20 - side * cfg.h1_trail_atr_buffer * h1_atr
                    active_stop = max(active_stop, candidate) if side > 0 else min(active_stop, candidate)

                hold_bars = i - entry_i + 1
                h4_ok = bool(getattr(row, "h4_regime_long", False)) if side > 0 else bool(getattr(row, "h4_regime_short", False))
                h1_close = float(getattr(row, "h1_close", np.nan))
                h1_ema20_now = float(getattr(row, "h1_ema20", np.nan))
                h1_failed = (
                    np.isfinite(h1_close)
                    and np.isfinite(h1_ema20_now)
                    and ((side > 0 and h1_close < h1_ema20_now) or (side < 0 and h1_close > h1_ema20_now))
                )
                if not h4_ok:
                    pending_exit = True
                    pending_reason = "H4_REGIME_LOST_NEXT_OPEN"
                elif h1_failed:
                    pending_exit = True
                    pending_reason = "H1_EMA20_LOST_NEXT_OPEN"
                elif hold_bars >= cfg.no_progress_bars and mfe_r < cfg.no_progress_min_mfe_r:
                    pending_exit = True
                    pending_reason = "NO_PROGRESS_12H_NEXT_OPEN"
                elif hold_bars >= cfg.max_hold_bars:
                    pending_exit = True
                    pending_reason = "MAX_HOLD_72H_NEXT_OPEN"

        # Signal from this closed bar enters at the next bar open.
        if not in_pos and i < len(rows) - 1 and i - last_exit_i >= cfg.cooldown_bars:
            sig = int(getattr(row, "signal", 0))
            if cfg.side_mode == "long" and sig < 0:
                sig = 0
            elif cfg.side_mode == "short" and sig > 0:
                sig = 0
            if sig != 0:
                # Reject any feature row whose HTF context was not available by
                # signal close time.  Fail closed rather than trusting a flag.
                if not bool(getattr(row, "context_available_time_flag", False)):
                    sig = 0
            if sig != 0:
                next_open = float(rows[i + 1].open)
                entry = apply_entry_slippage(next_open, sig, cfg.slippage_rate_per_side)
                raw_stop = float(getattr(row, "stop", np.nan))
                stop, ok = _valid_stop(entry, raw_stop, sig, cfg)
                if ok:
                    rpc = abs(entry - stop)
                    risk_usdt = capital * cfg.risk_per_trade
                    q = min(risk_usdt / rpc, (capital * cfg.max_notional_mult) / entry)
                    if np.isfinite(q) and q > 0:
                        in_pos = True
                        side = sig
                        entry_i = i + 1
                        entry_time = pd.Timestamp(idx[i + 1])
                        signal_time = pd.Timestamp(getattr(row, "signal_available_time"))
                        entry_price = entry
                        initial_stop = stop
                        active_stop = stop
                        risk_per_coin = rpc
                        qty = q
                        entry_fee = qty * entry_price * cfg.fee_rate_per_side
                        trade_capital_before = capital
                        trade_funding_pnl = 0.0
                        trade_funding_events = 0
                        max_fav = entry_price
                        max_adv = entry_price
                        pending_exit = False
                        pending_reason = ""
                        audit_ctx = {
                            "spec_id": "ETH_TREND_PULLBACK_V1",
                            "system": "trend_pullback",
                            "entry_model": "4H_regime_1H_reclaim_15m_reacceleration",
                            "signal_frame": "15m",
                            "used_tf15m_timestamp": pd.Timestamp(idx[i]),
                            "used_tf15m_available_time": pd.Timestamp(getattr(row, "signal_available_time")),
                            "expected_entry_time": pd.Timestamp(idx[i + 1]),
                            "expected_entry_price": float(next_open),
                            "used_h1_timestamp": getattr(row, "used_h1_timestamp", pd.NaT),
                            "used_h1_available_time": getattr(row, "used_h1_available_time", pd.NaT),
                            "used_h4_timestamp": getattr(row, "used_h4_timestamp", pd.NaT),
                            "used_h4_available_time": getattr(row, "used_h4_available_time", pd.NaT),
                            "context_available_time_flag": bool(getattr(row, "context_available_time_flag", False)),
                            "entry_not_next_open_flag": False,
                            "entry_price_mismatch_flag": False,
                        }

        close_mark = float(row.close)
        marked = mark_equity(close_mark, ts)
        peak_marked_equity = max(peak_marked_equity, marked)
        dd = (peak_marked_equity - marked) / peak_marked_equity if peak_marked_equity > 0 else 0.0
        equity_rows.append({"time": ts, "equity": marked, "capital": capital, "drawdown_pct": dd})

    if in_pos:
        close_trade(len(rows) - 1, float(rows[-1].close), "FORCE_CLOSE_END")
        if equity_rows:
            marked = capital
            peak_marked_equity = max(peak_marked_equity, marked)
            equity_rows[-1].update(
                {"equity": marked, "capital": capital, "drawdown_pct": (peak_marked_equity - marked) / peak_marked_equity if peak_marked_equity > 0 else 0.0}
            )

    equity = pd.DataFrame(equity_rows).set_index("time") if equity_rows else pd.DataFrame()
    return trades, equity, funding_ledger


def _max_streak(mask: pd.Series) -> int:
    best = cur = 0
    for value in mask.fillna(False).astype(bool).to_numpy():
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def summarize(
    trades: list[dict[str, Any]],
    equity: pd.DataFrame,
    cfg: ExecutionConfig,
    *,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    funding_source: str,
    funding_rows: int,
) -> dict[str, Any]:
    days = max((pd.Timestamp(window_end) - pd.Timestamp(window_start)).total_seconds() / 86400.0, 1.0)
    final_equity = float(equity.iloc[-1]["equity"]) if not equity.empty else float(cfg.initial_capital)
    total_return = final_equity / cfg.initial_capital - 1.0
    cagr = (final_equity / cfg.initial_capital) ** (365.25 / days) - 1.0 if final_equity > 0 else -1.0
    mdd = float(equity["drawdown_pct"].max()) if not equity.empty else 0.0
    calmar = cagr / mdd if mdd > 0 else float("inf")

    if trades:
        tdf = pd.DataFrame(trades)
        wins = tdf.loc[tdf["pnl"] > 0, "pnl"]
        losses = tdf.loc[tdf["pnl"] <= 0, "pnl"]
        gp = float(wins.sum()) if len(wins) else 0.0
        gl = float(-losses.sum()) if len(losses) else 0.0
        pf = gp / gl if gl > 0 else float("inf")
        entries = pd.to_datetime(tdf["entry_time"]).sort_values()
        gaps = [float((entries.iloc[0] - window_start).total_seconds() / 86400.0)]
        gaps += [float(x.total_seconds() / 86400.0) for x in entries.diff().dropna()]
        gaps += [float((window_end - entries.iloc[-1]).total_seconds() / 86400.0)]
        max_flat_days = max(gaps) if gaps else days
        max_consecutive_losses = _max_streak(tdf["pnl"] <= 0)
        avg_hold = float(tdf["holding_hours"].mean())
        median_hold = float(tdf["holding_hours"].median())
        p90_hold = float(tdf["holding_hours"].quantile(0.90))
        funding_pnl = float(tdf["funding_pnl"].sum())
        fee_total = float(tdf["trading_fee"].sum())
        win_rate = float((tdf["pnl"] > 0).mean())
        long_n = int((tdf["type"] == "LONG").sum())
        short_n = int((tdf["type"] == "SHORT").sum())
    else:
        gp = gl = funding_pnl = fee_total = 0.0
        pf = 0.0
        max_flat_days = days
        max_consecutive_losses = 0
        avg_hold = median_hold = p90_hold = 0.0
        win_rate = 0.0
        long_n = short_n = 0

    if not equity.empty:
        daily = equity["equity"].resample("1D").last().ffill()
        daily_ret = daily.pct_change().fillna(0.0)
        max_loss_days = _max_streak(daily_ret < 0)
        monthly = equity["equity"].resample("ME").last().ffill()
        monthly_ret = monthly.pct_change().dropna()
        positive_month_ratio = float((monthly_ret > 0).mean()) if len(monthly_ret) else 0.0
    else:
        max_loss_days = 0
        positive_month_ratio = 0.0

    return {
        "side_mode": cfg.side_mode,
        "total_trades": int(len(trades)),
        "long_trades": long_n,
        "short_trades": short_n,
        "trades_per_year": round(len(trades) / (days / 365.25), 3),
        "final_equity": round(final_equity, 4),
        "total_return_pct": round(total_return * 100.0, 4),
        "cagr_pct": round(cagr * 100.0, 4),
        "max_drawdown_pct": round(mdd * 100.0, 4),
        "calmar": round(calmar, 4) if math.isfinite(calmar) else "inf",
        "profit_factor": round(pf, 4) if math.isfinite(pf) else "inf",
        "win_rate_pct": round(win_rate * 100.0, 4),
        "avg_holding_hours": round(avg_hold, 3),
        "median_holding_hours": round(median_hold, 3),
        "p90_holding_hours": round(p90_hold, 3),
        "max_no_entry_days": round(max_flat_days, 3),
        "max_consecutive_loss_trades": int(max_consecutive_losses),
        "max_consecutive_loss_days": int(max_loss_days),
        "positive_month_ratio_pct": round(positive_month_ratio * 100.0, 3),
        "trading_fees": round(fee_total, 4),
        "funding_pnl": round(funding_pnl, 4),
        "funding_source": funding_source,
        "funding_rows_in_window": int(funding_rows),
        "fee_rate_per_side": cfg.fee_rate_per_side,
        "slippage_rate_per_side": cfg.slippage_rate_per_side,
        "risk_per_trade": cfg.risk_per_trade,
        "max_notional_mult": cfg.max_notional_mult,
    }


def cost_stress_configs(base: ExecutionConfig) -> list[tuple[str, ExecutionConfig]]:
    return [
        ("COST_1X", base),
        ("COST_2X", replace(base, fee_rate_per_side=base.fee_rate_per_side * 2.0, slippage_rate_per_side=base.slippage_rate_per_side * 2.0)),
        ("COST_3X", replace(base, fee_rate_per_side=base.fee_rate_per_side * 3.0, slippage_rate_per_side=base.slippage_rate_per_side * 3.0)),
        ("FUNDING_1_5X", replace(base, funding_multiplier=1.5)),
    ]
