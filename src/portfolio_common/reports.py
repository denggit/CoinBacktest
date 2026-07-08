#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Portfolio V1 report shaping helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.portfolio_common.allocator import PORTFOLIO_ID


def build_standard_summary(summary: pd.DataFrame, combined_all: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "portfolio_id",
                "scenario",
                "trades",
                "total_return",
                "max_drawdown",
                "win_rate",
                "profit_factor",
                "avg_trade_return",
                "total_fee",
            ]
        )
    out = summary.copy()
    if "total_trades" in out.columns:
        out["trades"] = pd.to_numeric(out["total_trades"], errors="coerce").fillna(0).astype(int)
    elif "trades" not in out.columns:
        out["trades"] = 0
    if "return_total" in out.columns:
        out["total_return"] = pd.to_numeric(out["return_total"], errors="coerce")
    if "portfolio_id" not in out.columns:
        out.insert(0, "portfolio_id", PORTFOLIO_ID)

    fee_by_scenario: dict[str, float] = {}
    if not combined_all.empty and {"scenario"}.issubset(combined_all.columns):
        if "fee" in combined_all.columns:
            fee = pd.to_numeric(combined_all["fee"], errors="coerce").fillna(0.0)
        else:
            fee = pd.Series(0.0, index=combined_all.index, dtype="float64")
        tmp = combined_all[["scenario"]].copy()
        tmp["_fee"] = fee
        fee_by_scenario = tmp.groupby("scenario", dropna=False)["_fee"].sum().to_dict()
    out["total_fee"] = out["scenario"].map(fee_by_scenario).fillna(0.0)

    preferred = [
        "portfolio_id",
        "scenario",
        "trades",
        "total_return",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "avg_trade_return",
        "total_fee",
        "final_capital",
        "lf_weight",
        "mf_variant_name",
        "mf_exposure",
        "conflict_mode",
        "guard_mode",
        "margin_cap",
        "notional_cap",
        "lf_trades",
        "mf_trades",
        "mf_skipped_by_conflict",
        "mf_guard_scaled_count",
        "mf_guard_skipped_count",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out.loc[:, cols].replace([np.inf, -np.inf], np.nan)


def build_primary_summary(
    summary: pd.DataFrame,
    combined_all: pd.DataFrame,
    primary_scenario: str,
) -> pd.DataFrame:
    """Return a single-row summary for the primary scenario only.

    Used for 01_summary.csv.
    """
    full = build_standard_summary(summary, combined_all)
    if full.empty:
        return full
    mask = full["scenario"].astype(str).eq(str(primary_scenario))
    if not mask.any():
        return pd.DataFrame(columns=full.columns)
    out = full.loc[mask].head(1).reset_index(drop=True)
    # Ensure required columns are present.
    primary_cols = [
        "portfolio_id",
        "trades",
        "total_return",
        "final_capital",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "avg_trade_return",
        "total_fee",
    ]
    for col in primary_cols:
        if col not in out.columns:
            out[col] = np.nan
    # Add time bounds from combined trades if available.
    if not combined_all.empty and "scenario" in combined_all.columns:
        primary_trades = combined_all.loc[combined_all["scenario"].astype(str).eq(str(primary_scenario))].copy()
        if not primary_trades.empty:
            out["first_entry_time"] = primary_trades["entry_time"].min() if "entry_time" in primary_trades.columns else np.nan
            out["last_exit_time"] = primary_trades["exit_time"].max() if "exit_time" in primary_trades.columns else np.nan
    if "first_entry_time" not in out.columns:
        out["first_entry_time"] = np.nan
    if "last_exit_time" not in out.columns:
        out["last_exit_time"] = np.nan
    # Add primary_scenario label column.
    out["primary_scenario"] = primary_scenario
    # Reorder: portfolio_id, primary_scenario first, then metrics, then time bounds.
    ordered = [
        "portfolio_id",
        "primary_scenario",
        "trades",
        "total_return",
        "final_capital",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "avg_trade_return",
        "total_fee",
        "first_entry_time",
        "last_exit_time",
    ]
    # Only keep the required columns.
    cols = [c for c in ordered if c in out.columns]
    return out.loc[:, cols]


def select_primary_trades(combined_all: pd.DataFrame, primary_scenario: str) -> pd.DataFrame:
    if combined_all.empty:
        return combined_all.copy()
    out = combined_all.loc[combined_all["scenario"].astype(str).eq(str(primary_scenario))].copy()
    return out.sort_values(["exit_time", "strategy_leg", "entry_time"]).reset_index(drop=True)


def filter_yearly_monthly(df: pd.DataFrame, primary_scenario: str | None = None) -> pd.DataFrame:
    """Optionally filter yearly/monthly rows to a single scenario."""
    if df.empty:
        return df
    if primary_scenario is not None and "scenario" in df.columns:
        return df.loc[df["scenario"].astype(str).eq(str(primary_scenario))].copy()
    return df.copy()


def build_manifest(
    args: Any,
    *,
    source_of_truth: str,
    artifacts: list[str],
    parity_old_report_dir: str | None = None,
) -> dict[str, object]:
    return {
        "portfolio_id": PORTFOLIO_ID,
        "script": "backtest/portfolio/eth_portfolio_V1_backtest.py",
        "class": "portfolio_backtest_refactor",
        "source_of_truth": source_of_truth,
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "initial_capital": float(args.initial_capital),
        "fee_rate": float(args.fee_rate),
        "slippage": float(args.slippage),
        "primary_scenario": args.primary_scenario,
        "parity_old_report_dir": parity_old_report_dir,
        "architecture": {
            "edge_lib": [
                "src/edge_lib/lf_momentum_breakout",
                "src/edge_lib/lf_bear_short",
                "src/edge_lib/lf_bull_range_reclaim",
                "src/edge_lib/mf_low_sweep",
            ],
            "sleeve_lib": ["src/sleeve_lib/lf_v10b"],
            "portfolio_common": ["src/portfolio_common"],
        },
        "artifacts": artifacts,
    }
