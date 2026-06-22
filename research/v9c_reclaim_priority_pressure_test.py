#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V9C Reclaim Priority Pressure Test
==================================

Runs real V9C backtest stress scenarios and post-processes trade-removal stresses.

Real backtest scenarios:
    - base
    - fee_2x
    - slippage_2x
    - no_2026

Post-process stresses from base trades:
    - remove_top1_pnl
    - remove_top3_pnl
    - remove_trade_2025_07_07

This tool is research-only. It does not change the strategy file.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

STRATEGY_NAME = "eth_lf_portfolio_v9c_reclaim_priority"
TRADES_FILE = f"{STRATEGY_NAME}_trades.csv"
SUMMARY_FILE = f"{STRATEGY_NAME}_summary.json"


@dataclass(frozen=True)
class RunScenario:
    name: str
    end_date: str | None = None
    fee_rate: float | None = None
    slippage_pct: float | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run V9C Reclaim-first pressure tests.")
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
    p.add_argument("--script", default="backtest/lf/eth_lf_portfolio_v9c_reclaim_priority_backtest.py")
    p.add_argument("--out-dir", default="data/reports/research/v9c_reclaim_priority_pressure_test")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--skip-backtests", action="store_true", help="Only postprocess existing base trades in --out-dir/runs/base.")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("\n$ " + " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def strategy_cmd(args: argparse.Namespace, scenario: RunScenario, run_dir: Path) -> list[str]:
    end_date = scenario.end_date or args.end_date
    fee_rate = scenario.fee_rate if scenario.fee_rate is not None else args.fee_rate
    slippage_pct = scenario.slippage_pct if scenario.slippage_pct is not None else args.slippage_pct
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
        "--out-dir", str(run_dir),
    ]


def load_summary(run_dir: Path, scenario_name: str, scenario_type: str) -> dict[str, Any]:
    summary_path = run_dir / SUMMARY_FILE
    if not summary_path.exists():
        # Some older outputs may use summary.json alias.
        summary_path = run_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        s = json.load(f)
    row = {"scenario": scenario_name, "scenario_type": scenario_type, "run_dir": str(run_dir)}
    for k in [
        "total_trades", "long_trades", "short_trades", "final_capital", "total_return_pct",
        "max_drawdown_pct", "profit_factor", "win_rate", "gross_profit", "gross_loss",
        "total_fees", "engine_counts", "priority_mode", "global_risk_scale",
        "micro_not_aligned_risk_scale", "micro_contra_risk_scale",
    ]:
        row[k] = s.get(k)
    return row


def trade_returns_from_csv(path: Path, initial_capital: float) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    t = pd.read_csv(path)
    if t.empty:
        return t
    if "return_pct" in t.columns:
        t["account_return"] = pd.to_numeric(t["return_pct"], errors="coerce")
    elif {"pnl", "capital"}.issubset(t.columns):
        pnl = pd.to_numeric(t["pnl"], errors="coerce")
        cap_after = pd.to_numeric(t["capital"], errors="coerce")
        t["account_return"] = pnl / (cap_after - pnl)
    else:
        raise RuntimeError("Trades file must contain return_pct or pnl+capital.")
    t["pnl"] = pd.to_numeric(t.get("pnl", 0.0), errors="coerce").fillna(0.0)
    t["entry_time"] = pd.to_datetime(t["entry_time"], errors="coerce")
    t["exit_time"] = pd.to_datetime(t["exit_time"], errors="coerce")
    t["_orig_index"] = range(len(t))
    return t


def summarize_trade_return_path(trades: pd.DataFrame, *, initial_capital: float) -> dict[str, Any]:
    capital = float(initial_capital)
    peak = capital
    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    losses = 0
    equity_rows: list[dict[str, Any]] = []
    for row in trades.itertuples(index=False):
        r = float(getattr(row, "account_return"))
        before = capital
        capital = capital * (1.0 + r)
        pnl = capital - before
        if pnl >= 0:
            gross_profit += pnl
            if pnl > 0:
                wins += 1
        else:
            gross_loss += -pnl
            losses += 1
        peak = max(peak, capital)
        dd = (peak - capital) / peak if peak > 0 else 0.0
        equity_rows.append({"time": getattr(row, "exit_time"), "capital": capital, "drawdown_pct": dd})
    n = int(len(trades))
    max_dd_pct = max((x["drawdown_pct"] for x in equity_rows), default=0.0) * 100.0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return {
        "total_trades": n,
        "final_capital": round(capital, 4),
        "total_return_pct": round((capital / initial_capital - 1.0) * 100.0, 4),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "profit_factor": round(pf, 4) if pf != float("inf") else pf,
        "win_rate": round((wins / n * 100.0), 4) if n else 0.0,
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
    }


def postprocess_removal_scenarios(args: argparse.Namespace, root_out: Path) -> list[dict[str, Any]]:
    base_run = root_out / "runs" / "base"
    trades_path = base_run / TRADES_FILE
    if not trades_path.exists():
        trades_path = base_run / "trades.csv"
    trades = trade_returns_from_csv(trades_path, args.initial_capital)
    if trades.empty:
        return []

    scenarios: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    # Remove by realized pnl because this answers "largest contribution to final capital".
    top1_idx = trades.sort_values("pnl", ascending=False).head(1)["_orig_index"].tolist()
    top3_idx = trades.sort_values("pnl", ascending=False).head(3)["_orig_index"].tolist()
    special_dt = pd.Timestamp("2025-07-07 00:00:00")
    special_idx = trades.loc[trades["entry_time"] == special_dt, "_orig_index"].tolist()
    scenarios.append(("remove_top1_pnl", trades.loc[~trades["_orig_index"].isin(top1_idx)].copy(), trades.loc[trades["_orig_index"].isin(top1_idx)].copy()))
    scenarios.append(("remove_top3_pnl", trades.loc[~trades["_orig_index"].isin(top3_idx)].copy(), trades.loc[trades["_orig_index"].isin(top3_idx)].copy()))
    scenarios.append(("remove_2025_07_07_long", trades.loc[~trades["_orig_index"].isin(special_idx)].copy(), trades.loc[trades["_orig_index"].isin(special_idx)].copy()))

    rows: list[dict[str, Any]] = []
    post_dir = root_out / "postprocess"
    post_dir.mkdir(parents=True, exist_ok=True)
    for name, kept, removed in scenarios:
        row = summarize_trade_return_path(kept, initial_capital=args.initial_capital)
        row.update({
            "scenario": name,
            "scenario_type": "trade_removal_postprocess",
            "removed_trades": int(len(removed)),
            "priority_mode": args.priority_mode,
            "global_risk_scale": args.global_risk_scale,
        })
        rows.append(row)
        kept.to_csv(post_dir / f"{name}_kept_trades.csv", index=False)
        removed.to_csv(post_dir / f"{name}_removed_trades.csv", index=False)
    return rows


def main() -> int:
    args = parse_args()
    root_out = Path(args.out_dir)
    runs_dir = root_out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        RunScenario("base"),
        RunScenario("fee_2x", fee_rate=args.fee_rate * 2.0),
        RunScenario("slippage_2x", slippage_pct=args.slippage_pct * 2.0),
        RunScenario("no_2026", end_date="2025-12-31"),
    ]

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if not args.skip_backtests:
        for scenario in scenarios:
            run_dir = runs_dir / scenario.name
            cmd = strategy_cmd(args, scenario, run_dir)
            try:
                run_cmd(cmd)
                rows.append(load_summary(run_dir, scenario.name, "real_backtest"))
            except Exception as exc:
                print(f"[ERROR] scenario={scenario.name}: {exc}")
                errors.append({"scenario": scenario.name, "error": str(exc)})
                if not args.continue_on_error:
                    raise
    else:
        for scenario in scenarios:
            run_dir = runs_dir / scenario.name
            if (run_dir / SUMMARY_FILE).exists() or (run_dir / "summary.json").exists():
                rows.append(load_summary(run_dir, scenario.name, "real_backtest"))

    rows.extend(postprocess_removal_scenarios(args, root_out))
    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        preferred = [
            "scenario", "scenario_type", "total_trades", "final_capital", "total_return_pct",
            "max_drawdown_pct", "profit_factor", "win_rate", "gross_profit", "gross_loss",
            "removed_trades", "priority_mode", "global_risk_scale", "run_dir",
        ]
        cols = [c for c in preferred if c in summary_df.columns] + [c for c in summary_df.columns if c not in preferred]
        summary_df = summary_df[cols]
        summary_path = root_out / "v9c_reclaim_priority_pressure_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print("\n" + "=" * 120)
        print("V9C Reclaim Priority Pressure Summary")
        print("=" * 120)
        print(summary_df[[c for c in preferred if c in summary_df.columns]].to_string(index=False))
        print(f"\nSaved: {summary_path.resolve()}")

    if errors:
        pd.DataFrame(errors).to_csv(root_out / "v9c_reclaim_priority_pressure_errors.csv", index=False)
    with (root_out / "v9c_reclaim_priority_pressure_config.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "scenarios": [asdict(x) for x in scenarios]}, f, ensure_ascii=False, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
