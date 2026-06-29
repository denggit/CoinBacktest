#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V9H Add-on Micro Risk Scale Probe
=================================

Runs V9C-equivalent control and add-on micro/range risk-scale variants.
This is a research-only probe. It does not modify V9C/V9E and does not place orders.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

STRATEGY_NAME = "eth_lf_portfolio_v9h_addon_micro_risk_scale"
TRADES_FILE = f"{STRATEGY_NAME}_trades.csv"
SUMMARY_FILE = f"{STRATEGY_NAME}_summary.json"


@dataclass(frozen=True)
class RunScenario:
    name: str
    note: str
    addon_micro_risk_mode: str
    end_date: str | None = None
    fee_rate: float | None = None
    slippage_pct: float | None = None
    addon_micro_contra_risk_scale: float | None = None
    addon_micro_not_aligned_risk_scale: float | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run V9H add-on micro risk-scale probe.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-15")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--global-risk-scale", type=float, default=1.3)
    p.add_argument("--priority-mode", choices=["v8", "reclaim_first", "reclaim_bear_second"], default="reclaim_first")
    p.add_argument("--preset", default="turbo")
    p.add_argument("--bear-preset", default="high")
    p.add_argument("--bull-preset", default="high")
    p.add_argument("--bull-execution-mode", choices=["inherit", "own"], default="inherit")
    p.add_argument("--micro-filter-mode", choices=["off", "soft", "strict"], default="soft")
    p.add_argument("--micro-not-aligned-risk-scale", type=float, default=0.5)
    p.add_argument("--micro-contra-risk-scale", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--range-pct", type=float, default=0.002)
    p.add_argument("--price-step", type=float, default=1.0)
    p.add_argument("--script", default="backtest/lf/eth_lf_portfolio_v9h_addon_micro_risk_scale_backtest.py")
    p.add_argument("--out-dir", default="data/reports/research/v9h_addon_micro_risk_scale_probe")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--skip-backtests", action="store_true", help="Only aggregate existing outputs under --out-dir/runs.")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def scenarios(args: argparse.Namespace) -> list[RunScenario]:
    return [
        RunScenario(
            name="off_control",
            note="V9C-equivalent: micro/range risk scale affects initial entry only",
            addon_micro_risk_mode="off",
        ),
        RunScenario(
            name="entry_micro_on_addons",
            note="Reuse initial entry micro risk scale on every pyramid add-on",
            addon_micro_risk_mode="entry",
        ),
        RunScenario(
            name="current_micro_on_addons",
            note="Use current completed 4H range/footprint context to risk-scale each add-on",
            addon_micro_risk_mode="current",
        ),
        RunScenario(
            name="both_min_micro_on_addons",
            note="Use min(initial entry micro scale, current add-on micro scale)",
            addon_micro_risk_mode="both_min",
        ),
        RunScenario(
            name="current_more_conservative",
            note="Current add-on micro scale with contra=0.25 and not_aligned=0.50",
            addon_micro_risk_mode="current",
            addon_micro_contra_risk_scale=0.25,
            addon_micro_not_aligned_risk_scale=0.50,
        ),
        RunScenario(
            name="both_min_more_conservative",
            note="Both-min add-on micro scale with contra=0.25 and not_aligned=0.50",
            addon_micro_risk_mode="both_min",
            addon_micro_contra_risk_scale=0.25,
            addon_micro_not_aligned_risk_scale=0.50,
        ),
        RunScenario(
            name="current_fee_2x",
            note="Current add-on micro scale with double fee",
            addon_micro_risk_mode="current",
            fee_rate=args.fee_rate * 2.0,
        ),
        RunScenario(
            name="current_slippage_2x",
            note="Current add-on micro scale with double slippage",
            addon_micro_risk_mode="current",
            slippage_pct=args.slippage_pct * 2.0,
        ),
        RunScenario(
            name="current_no_2026",
            note="Current add-on micro scale ending before 2026",
            addon_micro_risk_mode="current",
            end_date="2025-12-31",
        ),
    ]


def run_cmd(cmd: list[str]) -> None:
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def strategy_cmd(args: argparse.Namespace, sc: RunScenario, run_dir: Path) -> list[str]:
    end_date = sc.end_date or args.end_date
    fee_rate = sc.fee_rate if sc.fee_rate is not None else args.fee_rate
    slippage_pct = sc.slippage_pct if sc.slippage_pct is not None else args.slippage_pct
    contra_scale = sc.addon_micro_contra_risk_scale if sc.addon_micro_contra_risk_scale is not None else args.micro_contra_risk_scale
    not_aligned_scale = sc.addon_micro_not_aligned_risk_scale if sc.addon_micro_not_aligned_risk_scale is not None else args.micro_not_aligned_risk_scale
    return [
        args.python,
        args.script,
        "--symbol", args.symbol,
        "--start-date", args.start_date,
        "--end-date", end_date,
        "--warmup-start-date", args.warmup_start_date,
        "--initial-capital", str(args.initial_capital),
        "--preset", args.preset,
        "--bear-preset", args.bear_preset,
        "--bull-preset", args.bull_preset,
        "--bull-execution-mode", args.bull_execution_mode,
        "--micro-filter-mode", args.micro_filter_mode,
        "--micro-not-aligned-risk-scale", str(args.micro_not_aligned_risk_scale),
        "--micro-contra-risk-scale", str(args.micro_contra_risk_scale),
        "--global-risk-scale", str(args.global_risk_scale),
        "--priority-mode", args.priority_mode,
        "--fee-rate", str(fee_rate),
        "--slippage-pct", str(slippage_pct),
        "--range-pct", str(args.range_pct),
        "--price-step", str(args.price_step),
        "--addon-micro-risk-mode", sc.addon_micro_risk_mode,
        "--addon-micro-contra-risk-scale", str(contra_scale),
        "--addon-micro-not-aligned-risk-scale", str(not_aligned_scale),
        "--out-dir", str(run_dir),
    ]


def load_summary(run_dir: Path, sc: RunScenario) -> dict[str, Any]:
    summary_path = run_dir / SUMMARY_FILE
    if not summary_path.exists():
        summary_path = run_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        s = json.load(f)
    row: dict[str, Any] = {
        "scenario": sc.name,
        "scenario_note": sc.note,
        "run_dir": str(run_dir),
    }
    keys = [
        "total_trades", "long_trades", "short_trades", "final_capital", "total_return_pct",
        "max_drawdown_pct", "profit_factor", "win_rate", "expectancy_pct",
        "gross_profit", "gross_loss", "total_fees", "engine_counts", "priority_mode",
        "global_risk_scale", "micro_filter_mode", "addon_micro_risk_mode",
        "addon_micro_contra_risk_scale", "addon_micro_not_aligned_risk_scale",
        "trades_with_addons", "total_addons", "micro_scaled_addons",
        "addon_micro_scale_avg_trade_weighted", "yearly_return_pct",
    ]
    for k in keys:
        row[k] = s.get(k)
    return row


def load_trades(run_dir: Path) -> pd.DataFrame:
    path = run_dir / TRADES_FILE
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def add_trade_detail_exports(out_dir: Path, scenario_rows: list[dict[str, Any]]) -> None:
    details_dir = out_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    for row in scenario_rows:
        run_dir = Path(str(row["run_dir"]))
        tdf = load_trades(run_dir)
        if tdf.empty:
            continue
        add_cols = [
            c for c in [
                "entry_time", "exit_time", "type", "engine", "note", "return_pct", "mfe_r", "mae_r",
                "units", "addon_count", "addon_micro_scaled_count", "addon_micro_scale_avg",
                "addon_micro_scale_min", "last_addon_micro_scale", "last_addon_micro_action",
                "micro_entry_risk_scale", "micro_filter_action", "rf_imbalance", "rf_close_pos",
            ] if c in tdf.columns
        ]
        if add_cols:
            tdf[add_cols].to_csv(details_dir / f"{row['scenario']}_addon_detail.csv", index=False)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for sc in scenarios(args):
        run_dir = runs_dir / sc.name
        run_dir.mkdir(parents=True, exist_ok=True)
        if not args.skip_backtests:
            try:
                run_cmd(strategy_cmd(args, sc, run_dir))
            except subprocess.CalledProcessError:
                if not args.continue_on_error:
                    raise
                print(f"[WARN] scenario failed: {sc.name}", flush=True)
                continue
        rows.append(load_summary(run_dir, sc))

    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / "v9h_addon_micro_risk_scale_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    add_trade_detail_exports(out_dir, rows)
    print("\nWrote:", summary_path)
    if not summary_df.empty:
        display_cols = [
            "scenario", "total_return_pct", "max_drawdown_pct", "profit_factor", "win_rate",
            "total_trades", "trades_with_addons", "total_addons", "micro_scaled_addons",
            "addon_micro_risk_mode", "run_dir",
        ]
        print(summary_df[[c for c in display_cols if c in summary_df.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
