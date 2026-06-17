#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Small report-format helpers shared by backtest scripts."""

from __future__ import annotations

from typing import Any


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

import math
import pandas as pd


def summarize_hf_trades(
    trades: list[dict[str, Any]],
    equity: pd.DataFrame,
    initial_capital: float,
    signal_count: int,
) -> dict[str, Any]:
    """Shared summary for 1-second / tick-driven HF backtests."""
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


def summarize_r_trades(trades: list[dict[str, Any]], equity: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    """Shared summary for R-multiple OHLCV backtests without pyramiding-specific fields."""
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
        "avg_holding_hours": round(float(tdf["holding_hours"].mean()), 2),
        "total_fees": round(float(tdf["fee"].sum()), 4),
    }

import json
from pathlib import Path


def write_hf_outputs(
    features: pd.DataFrame,
    trades: list[dict[str, Any]],
    equity: pd.DataFrame,
    summary: dict[str, Any],
    out_dir: Path,
    *,
    write_full_audit: bool,
    strategy_name: str,
) -> None:
    """Write common HF trade/equity/audit/summary artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / f"{strategy_name}_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / f"{strategy_name}_equity.csv")

    audit_cols = [
        "open", "high", "low", "close", "buy_notional", "sell_notional", "volume_notional", "cvd_notional",
        "signal", "signal_reason", "signal_level", "signal_extreme", "local_low", "local_high",
        "sell_notional_3s", "buy_notional_3s", "cvd_5s", "buy_ratio_5s", "sell_ratio_5s",
        "range_high", "range_low", "range_pct", "compression_ok", "cvd_10s",
    ]
    signal_rows = features[features["signal"] != 0].copy()
    signal_rows[[c for c in audit_cols if c in signal_rows.columns]].to_csv(out_dir / f"{strategy_name}_signal_audit.csv")
    if write_full_audit:
        features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{strategy_name}_full_audit.csv")

    with (out_dir / f"{strategy_name}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def emit_hf_platform_report(trades: list[dict[str, Any]], features: pd.DataFrame, cfg: Any, out_dir: Path, *, strategy_name: str) -> None:
    """Use the project-wide report module for HF strategy reports."""
    if features.empty:
        return
    from src.utils.report import print_full_report

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


def print_hf_summary(summary: dict[str, Any], out_dir: Path, *, strategy_name: str) -> None:
    print("\n" + "=" * 88)
    print(f"ETH HF Backtest Summary | {strategy_name}")
    print("=" * 88)
    for k, v in summary.items():
        print(f"{k:>28}: {v}")
    print("-" * 88)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 88 + "\n")
