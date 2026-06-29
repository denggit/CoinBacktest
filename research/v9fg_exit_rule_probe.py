#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V9F / V9G Direction-3 Exit Rule Probe
====================================

Runs independent V9F and V9G protective-exit probes against the same V9C-equivalent
baseline. This tool does not modify V9E/V9C. It only launches backtests and collects
summary/trade details.
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


@dataclass(frozen=True)
class Scenario:
    name: str
    script: str
    strategy_name: str
    note: str
    range_exit_mode: str = "soft"
    range_exit_min_mfe_r: float = 2.0
    range_exit_giveback_frac: float = 0.35
    range_exit_min_hold_bars: int = 2
    range_exit_delay_bars: int = 0
    fee_rate: float | None = None
    slippage_pct: float | None = None
    extra_args: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run V9F/V9G direction-3 exit rule probes.")
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
    p.add_argument("--v9f-script", default="backtest/lf/eth_lf_portfolio_v9f_failed_continuation_exit_backtest.py")
    p.add_argument("--v9g-script", default="backtest/lf/eth_lf_portfolio_v9g_absorption_exit_backtest.py")
    p.add_argument("--out-dir", default="data/reports/research/v9fg_exit_rule_probe")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--skip-backtests", action="store_true", help="Only collect existing outputs under --out-dir/runs.")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def scenarios(args: argparse.Namespace) -> list[Scenario]:
    return [
        Scenario(
            name="off_control_v9f_file",
            script=args.v9f_script,
            strategy_name="eth_lf_portfolio_v9f_failed_continuation_exit",
            note="V9C-equivalent control through V9F file; range_exit_mode=off",
            range_exit_mode="off",
        ),
        Scenario(
            name="v9f_base_failed_continuation",
            script=args.v9f_script,
            strategy_name="eth_lf_portfolio_v9f_failed_continuation_exit",
            note="V9F default failed-continuation exit",
            range_exit_giveback_frac=0.35,
            extra_args=(
                "--failed-cont-min-current-r", "0.5",
                "--failed-cont-min-flow-imbalance", "0.05",
                "--failed-cont-bad-close-pos", "0.40",
            ),
        ),
        Scenario(
            name="v9f_conservative",
            script=args.v9f_script,
            strategy_name="eth_lf_portfolio_v9f_failed_continuation_exit",
            note="V9F conservative: MFE>=3R, giveback>=45%",
            range_exit_min_mfe_r=3.0,
            range_exit_giveback_frac=0.45,
            extra_args=(
                "--failed-cont-min-current-r", "0.8",
                "--failed-cont-min-flow-imbalance", "0.08",
                "--failed-cont-bad-close-pos", "0.35",
            ),
        ),
        Scenario(
            name="v9f_delay_1bar",
            script=args.v9f_script,
            strategy_name="eth_lf_portfolio_v9f_failed_continuation_exit",
            note="V9F delay stress: one additional 4H bar after normal next-open exit",
            range_exit_giveback_frac=0.35,
            range_exit_delay_bars=1,
            extra_args=(
                "--failed-cont-min-current-r", "0.5",
                "--failed-cont-min-flow-imbalance", "0.05",
                "--failed-cont-bad-close-pos", "0.40",
            ),
        ),
        Scenario(
            name="v9f_fee_2x",
            script=args.v9f_script,
            strategy_name="eth_lf_portfolio_v9f_failed_continuation_exit",
            note="V9F with double fee",
            range_exit_giveback_frac=0.35,
            fee_rate=args.fee_rate * 2.0,
            extra_args=(
                "--failed-cont-min-current-r", "0.5",
                "--failed-cont-min-flow-imbalance", "0.05",
                "--failed-cont-bad-close-pos", "0.40",
            ),
        ),
        Scenario(
            name="v9g_base_absorption",
            script=args.v9g_script,
            strategy_name="eth_lf_portfolio_v9g_absorption_exit",
            note="V9G default absorption exit",
            range_exit_giveback_frac=0.30,
            extra_args=(
                "--absorption-min-current-r", "0.5",
                "--absorption-bucket-share", "0.20",
                "--absorption-min-flow-imbalance", "0.03",
                "--absorption-bad-close-pos", "0.45",
            ),
        ),
        Scenario(
            name="v9g_conservative",
            script=args.v9g_script,
            strategy_name="eth_lf_portfolio_v9g_absorption_exit",
            note="V9G conservative: MFE>=3R, bucket>=25%, giveback>=40%",
            range_exit_min_mfe_r=3.0,
            range_exit_giveback_frac=0.40,
            extra_args=(
                "--absorption-min-current-r", "0.8",
                "--absorption-bucket-share", "0.25",
                "--absorption-min-flow-imbalance", "0.05",
                "--absorption-bad-close-pos", "0.40",
            ),
        ),
        Scenario(
            name="v9g_delay_1bar",
            script=args.v9g_script,
            strategy_name="eth_lf_portfolio_v9g_absorption_exit",
            note="V9G delay stress: one additional 4H bar after normal next-open exit",
            range_exit_giveback_frac=0.30,
            range_exit_delay_bars=1,
            extra_args=(
                "--absorption-min-current-r", "0.5",
                "--absorption-bucket-share", "0.20",
                "--absorption-min-flow-imbalance", "0.03",
                "--absorption-bad-close-pos", "0.45",
            ),
        ),
        Scenario(
            name="v9g_fee_2x",
            script=args.v9g_script,
            strategy_name="eth_lf_portfolio_v9g_absorption_exit",
            note="V9G with double fee",
            range_exit_giveback_frac=0.30,
            fee_rate=args.fee_rate * 2.0,
            extra_args=(
                "--absorption-min-current-r", "0.5",
                "--absorption-bucket-share", "0.20",
                "--absorption-min-flow-imbalance", "0.03",
                "--absorption-bad-close-pos", "0.45",
            ),
        ),
    ]


def strategy_cmd(args: argparse.Namespace, sc: Scenario, run_dir: Path) -> list[str]:
    fee_rate = sc.fee_rate if sc.fee_rate is not None else args.fee_rate
    slippage_pct = sc.slippage_pct if sc.slippage_pct is not None else args.slippage_pct
    cmd = [
        args.python, sc.script,
        "--symbol", args.symbol,
        "--start-date", args.start_date,
        "--end-date", args.end_date,
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
        "--range-exit-mode", sc.range_exit_mode,
        "--range-exit-min-mfe-r", str(sc.range_exit_min_mfe_r),
        "--range-exit-giveback-frac", str(sc.range_exit_giveback_frac),
        "--range-exit-min-hold-bars", str(sc.range_exit_min_hold_bars),
        "--range-exit-delay-bars", str(sc.range_exit_delay_bars),
        "--out-dir", str(run_dir),
    ]
    cmd.extend(sc.extra_args)
    return cmd


def _load_summary(run_dir: Path, sc: Scenario) -> dict[str, Any]:
    path = run_dir / f"{sc.strategy_name}_summary.json"
    if not path.exists():
        path = run_dir / "summary.json"
    with path.open("r", encoding="utf-8") as f:
        s = json.load(f)
    row = {
        "scenario": sc.name,
        "strategy_name": sc.strategy_name,
        "scenario_note": sc.note,
        "run_dir": str(run_dir),
    }
    keys = [
        "total_trades", "long_trades", "short_trades", "final_capital", "total_return_pct",
        "max_drawdown_pct", "profit_factor", "win_rate", "expectancy_pct",
        "gross_profit", "gross_loss", "total_fees", "engine_counts", "priority_mode",
        "global_risk_scale", "range_exit_mode", "range_exit_trade_count",
        "range_exit_avg_peak_r", "range_exit_avg_giveback_frac", "range_exit_min_mfe_r",
        "range_exit_giveback_frac", "range_exit_min_hold_bars", "range_exit_delay_bars",
        "mfe_ge_1r_ended_loss", "mfe_ge_2r_ended_loss",
        "failed_cont_min_current_r", "failed_cont_min_flow_imbalance", "failed_cont_bad_close_pos",
        "absorption_min_current_r", "absorption_bucket_share", "absorption_min_flow_imbalance", "absorption_bad_close_pos",
    ]
    for k in keys:
        row[k] = s.get(k)
    return row


def _copy_range_exit_trades(run_dir: Path, sc: Scenario, details_dir: Path) -> None:
    path = run_dir / f"{sc.strategy_name}_trades.csv"
    if not path.exists():
        path = run_dir / "trades.csv"
    if not path.exists():
        return
    t = pd.read_csv(path)
    if t.empty or "note" not in t.columns:
        return
    rx = t[t["note"].astype(str).str.startswith("RANGE_EXIT")].copy()
    if not rx.empty:
        rx.to_csv(details_dir / f"{sc.name}_range_exit_trades.csv", index=False)


def main() -> int:
    args = parse_args()
    root = Path(args.out_dir)
    runs = root / "runs"
    details = root / "details"
    runs.mkdir(parents=True, exist_ok=True)
    details.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for sc in scenarios(args):
        run_dir = runs / sc.name
        run_dir.mkdir(parents=True, exist_ok=True)
        if not args.skip_backtests:
            cmd = strategy_cmd(args, sc, run_dir)
            print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                if not args.continue_on_error:
                    raise
                print(f"[WARN] scenario failed: {sc.name}", file=sys.stderr)
                continue
        rows.append(_load_summary(run_dir, sc))
        _copy_range_exit_trades(run_dir, sc, details)

    out = pd.DataFrame(rows)
    out_path = root / "v9fg_exit_rule_probe_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[OK] wrote {out_path}")
    if not out.empty:
        cols = ["scenario", "total_return_pct", "max_drawdown_pct", "profit_factor", "win_rate", "total_trades", "range_exit_trade_count"]
        print(out[[c for c in cols if c in out.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
