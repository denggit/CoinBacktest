#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report and artifact writers for martingale backtest runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.report import print_full_report

from .engine import MartingaleEngine

SCRIPT_NAME = "eth_martingale_limit_long_backtest"


def summarize_engine(engine: MartingaleEngine) -> dict[str, Any]:
    trades = pd.DataFrame(engine.trades)
    tp_count = int((trades.get("note") == "TAKE_PROFIT").sum()) if not trades.empty else 0
    liquidation_count = int((trades.get("note") == "LIQUIDATION").sum()) if not trades.empty else 0
    force_close_count = int((trades.get("note") == "FORCE_CLOSE_END").sum()) if not trades.empty else 0
    closed_fees = float(trades["fee"].sum()) if not trades.empty else 0.0
    total_fees = closed_fees + float(engine.entry_fees)
    gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum()) if not trades.empty else 0.0
    gross_loss = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum()) if not trades.empty else 0.0
    profit_factor: float | str = gross_profit / gross_loss if gross_loss > 0 else "inf"
    avg_additions = float(trades["additions"].mean()) if not trades.empty else 0.0
    max_additions = int(trades["additions"].max()) if not trades.empty else 0
    final_mtm = (
        engine.mark_equity(engine.last_price)
        if engine.last_price is not None
        else engine.capital
    )
    return {
        "variant": engine.variant.key,
        "display_name": engine.variant.display_name,
        "total_cycles_closed": int(len(engine.trades)),
        "take_profit_cycles": tp_count,
        "liquidation_cycles": liquidation_count,
        "force_close_cycles": force_close_count,
        "win_rate_pct": float((trades["pnl"] > 0).mean() * 100.0) if not trades.empty else 0.0,
        "initial_capital": engine.initial_capital,
        "final_capital_realized": engine.capital,
        "final_equity_mtm": final_mtm,
        "total_return_realized_pct": (engine.capital / engine.initial_capital - 1.0) * 100.0,
        "total_return_mtm_pct": (final_mtm / engine.initial_capital - 1.0) * 100.0,
        "max_mtm_drawdown_pct": engine.max_mtm_drawdown * 100.0,
        "min_mtm_equity": engine.min_mtm_equity,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "total_fees": total_fees,
        "avg_additions_per_closed_cycle": avg_additions,
        "max_additions_filled": max_additions,
        "open_position": engine.in_position,
        "bankrupt": engine.bankrupt,
    }


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def write_engine_outputs(
    engine: MartingaleEngine,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir = Path(args.out_dir) / args.data_source / engine.variant.key
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_df = pd.DataFrame(engine.trades)
    fills_df = pd.DataFrame(engine.fills)
    equity = engine.equity_frame()
    summary = summarize_engine(engine)

    trades_df.to_csv(out_dir / "01_trades.csv", index=False)
    fills_df.to_csv(out_dir / "02_order_fills.csv", index=False)
    equity.to_csv(out_dir / "03_equity_daily.csv")
    _json_dump(out_dir / "04_summary.json", summary)
    _json_dump(
        out_dir / "05_config.json",
        {
            "script": SCRIPT_NAME,
            "data_source": args.data_source,
            "symbol": args.symbol,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "trade_bar_timeframe": args.trade_bar_timeframe,
            "range_pct": args.range_pct,
            "engine": asdict(engine.config),
            "variant": asdict(engine.variant),
            "execution_assumptions": {
                "initial_entry": "resting buy limit one entry_drop_pct below cycle anchor",
                "max_additions_excludes_initial": True,
                "notional_ratio": "initial order : first add = 1 : initial_add_ratio",
                "bar_intrabar_policy": "downward adverse path first; fill reachable buys; liquidation check; no TP on a bar with fills",
                "raw_trade_policy": "timestamp-ordered prints; fill and TP cannot use same print",
                "fee_policy": "fee_rate charged on every entry/add and exit fill",
                "liquidation_model": "approximate cross-margin equation with maintenance margin and exit fee; not exact OKX tiered mark-price engine",
            },
        },
    )
    _json_dump(out_dir / "06_open_position.json", engine.open_position_snapshot())

    if not args.skip_full_report:
        report_df = equity.copy()
        if report_df.empty and engine.last_time is not None:
            report_df = pd.DataFrame(index=pd.DatetimeIndex([engine.last_time]))
        total_days = max(
            (pd.Timestamp(args.end_date) - pd.Timestamp(args.start_date)).total_seconds() / 86400.0,
            1.0,
        )
        print_full_report(
            trade_history=[dict(item) for item in engine.trades],
            df=report_df,
            initial_capital=engine.initial_capital,
            capital=engine.capital,
            strategy_name=f"ETH_Martingale_{engine.variant.key}_{args.data_source}",
            total_days=total_days,
            ai_enabled=False,
            symbol=args.symbol,
            report_dir=str(out_dir),
        )
    return summary


def print_comparison(comparison: pd.DataFrame, out_dir: Path) -> None:
    print("\n" + "=" * 108)
    print("ETH Long Martingale Limit-Order Backtest")
    print("=" * 108)
    cols = [
        "variant",
        "total_cycles_closed",
        "take_profit_cycles",
        "liquidation_cycles",
        "final_capital_realized",
        "total_return_realized_pct",
        "max_mtm_drawdown_pct",
        "total_fees",
    ]
    if comparison.empty:
        print("No results")
    else:
        print(comparison[cols].to_string(index=False))
    print("-" * 108)
    print("Risk note: use max_mtm_drawdown_pct in 04_summary.json; project full report DD is closed-cycle only.")
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 108 + "\n")


