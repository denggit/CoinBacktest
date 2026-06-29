#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V9E Range Exit Overlay Pressure Test
====================================

Runs real V9E backtest stress scenarios and post-processes trade-removal stresses.

Real backtest scenarios:
    - off_control              : V9C-equivalent control, range_exit_mode=off
    - base                     : default V9E range exit overlay
    - fee_2x                   : double fee
    - slippage_2x              : double slippage
    - no_2026                  : out before 2026, checks whether improvement is not only 2026
    - mfe3_giveback70          : more conservative range-exit rule
    - no_footprint             : disables footprint max-bucket context, keeps range-bar aggregate context
    - delay_1bar               : exits one additional 4H bar later than normal V9E
    - delay_2bar               : exits two additional 4H bars later than normal V9E

Post-process stresses from base trades:
    - remove_top1_pnl
    - remove_top3_pnl
    - remove_2025_07_07_long
    - remove_range_exit_trades
    - remove_top1_range_exit_pnl

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

STRATEGY_NAME = "eth_lf_portfolio_v9e_range_exit_overlay"
TRADES_FILE = f"{STRATEGY_NAME}_trades.csv"
SUMMARY_FILE = f"{STRATEGY_NAME}_summary.json"


@dataclass(frozen=True)
class RunScenario:
    name: str
    scenario_note: str
    end_date: str | None = None
    fee_rate: float | None = None
    slippage_pct: float | None = None
    range_exit_mode: str | None = None
    range_exit_min_mfe_r: float | None = None
    range_exit_giveback_frac: float | None = None
    range_exit_min_hold_bars: int | None = None
    range_exit_delay_bars: int | None = None
    disable_footprint_context: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run V9E Range Exit Overlay pressure tests.")
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
    p.add_argument("--range-exit-min-mfe-r", type=float, default=2.0)
    p.add_argument("--range-exit-giveback-frac", type=float, default=0.65)
    p.add_argument("--range-exit-min-hold-bars", type=int, default=2)
    p.add_argument("--range-exit-delay-bars", type=int, default=0, help="Default delay for scenarios unless overridden. 0=normal next-open V9E.")
    p.add_argument("--range-exit-contra-imbalance", type=float, default=0.05)
    p.add_argument("--range-exit-bad-close-pos", type=float, default=0.35)
    p.add_argument("--script", default="backtest/lf/eth_lf_portfolio_v9e_range_exit_overlay_backtest.py")
    p.add_argument("--out-dir", default="data/reports/research/v9e_range_exit_overlay_pressure_test")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--skip-backtests", action="store_true", help="Only postprocess existing outputs under --out-dir/runs.")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def strategy_cmd(args: argparse.Namespace, scenario: RunScenario, run_dir: Path) -> list[str]:
    end_date = scenario.end_date or args.end_date
    fee_rate = scenario.fee_rate if scenario.fee_rate is not None else args.fee_rate
    slippage_pct = scenario.slippage_pct if scenario.slippage_pct is not None else args.slippage_pct
    range_exit_mode = scenario.range_exit_mode or "soft"
    range_exit_min_mfe_r = scenario.range_exit_min_mfe_r if scenario.range_exit_min_mfe_r is not None else args.range_exit_min_mfe_r
    range_exit_giveback_frac = scenario.range_exit_giveback_frac if scenario.range_exit_giveback_frac is not None else args.range_exit_giveback_frac
    range_exit_min_hold_bars = scenario.range_exit_min_hold_bars if scenario.range_exit_min_hold_bars is not None else args.range_exit_min_hold_bars
    range_exit_delay_bars = scenario.range_exit_delay_bars if scenario.range_exit_delay_bars is not None else args.range_exit_delay_bars

    cmd = [
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
        "--range-exit-mode", range_exit_mode,
        "--range-exit-min-mfe-r", str(range_exit_min_mfe_r),
        "--range-exit-giveback-frac", str(range_exit_giveback_frac),
        "--range-exit-min-hold-bars", str(range_exit_min_hold_bars),
        "--range-exit-delay-bars", str(range_exit_delay_bars),
        "--range-exit-contra-imbalance", str(args.range_exit_contra_imbalance),
        "--range-exit-bad-close-pos", str(args.range_exit_bad_close_pos),
        "--out-dir", str(run_dir),
    ]
    if scenario.disable_footprint_context:
        cmd.append("--disable-footprint-context")
    return cmd


def load_summary(run_dir: Path, scenario: RunScenario, scenario_type: str) -> dict[str, Any]:
    summary_path = run_dir / SUMMARY_FILE
    if not summary_path.exists():
        summary_path = run_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        s = json.load(f)
    row = {
        "scenario": scenario.name,
        "scenario_type": scenario_type,
        "scenario_note": scenario.scenario_note,
        "run_dir": str(run_dir),
    }
    for k in [
        "total_trades", "long_trades", "short_trades", "final_capital", "total_return_pct",
        "max_drawdown_pct", "profit_factor", "win_rate", "expectancy_pct",
        "gross_profit", "gross_loss", "total_fees", "engine_counts", "priority_mode",
        "global_risk_scale", "micro_not_aligned_risk_scale", "micro_contra_risk_scale",
        "range_exit_mode", "range_exit_trade_count", "range_exit_avg_peak_r",
        "range_exit_avg_giveback_frac", "range_exit_min_mfe_r", "range_exit_giveback_frac",
        "range_exit_min_hold_bars", "range_exit_delay_bars", "range_exit_contra_imbalance", "range_exit_bad_close_pos",
        "mfe_ge_1r_ended_loss", "mfe_ge_2r_ended_loss",
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
    t["note"] = t.get("note", "").astype(str)
    t["is_range_exit"] = t["note"].str.startswith("RANGE_EXIT")
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


def _append_removal_scenario(rows: list[dict[str, Any]], *, args: argparse.Namespace, name: str, kept: pd.DataFrame, removed: pd.DataFrame, post_dir: Path) -> None:
    row = summarize_trade_return_path(kept, initial_capital=args.initial_capital)
    row.update({
        "scenario": name,
        "scenario_type": "trade_removal_postprocess",
        "scenario_note": "postprocess from base trades",
        "removed_trades": int(len(removed)),
        "removed_pnl_sum": float(pd.to_numeric(removed.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if len(removed) else 0.0,
        "removed_range_exit_trades": int(pd.Series(removed.get("is_range_exit", pd.Series(dtype=bool))).fillna(False).astype(bool).sum()) if len(removed) else 0,
        "priority_mode": args.priority_mode,
        "global_risk_scale": args.global_risk_scale,
    })
    rows.append(row)
    kept.to_csv(post_dir / f"{name}_kept_trades.csv", index=False)
    removed.to_csv(post_dir / f"{name}_removed_trades.csv", index=False)


def postprocess_removal_scenarios(args: argparse.Namespace, root_out: Path) -> list[dict[str, Any]]:
    base_run = root_out / "runs" / "base"
    trades_path = base_run / TRADES_FILE
    if not trades_path.exists():
        trades_path = base_run / "trades.csv"
    trades = trade_returns_from_csv(trades_path, args.initial_capital)
    if trades.empty:
        return []

    rows: list[dict[str, Any]] = []
    post_dir = root_out / "postprocess"
    post_dir.mkdir(parents=True, exist_ok=True)

    top1_idx = trades.sort_values("pnl", ascending=False).head(1)["_orig_index"].tolist()
    top3_idx = trades.sort_values("pnl", ascending=False).head(3)["_orig_index"].tolist()
    special_dt = pd.Timestamp("2025-07-07 00:00:00")
    special_idx = trades.loc[trades["entry_time"] == special_dt, "_orig_index"].tolist()
    range_exit_idx = trades.loc[trades["is_range_exit"], "_orig_index"].tolist()
    top1_range_idx = trades.loc[trades["is_range_exit"]].sort_values("pnl", ascending=False).head(1)["_orig_index"].tolist()

    scenarios = [
        ("remove_top1_pnl", top1_idx),
        ("remove_top3_pnl", top3_idx),
        ("remove_2025_07_07_long", special_idx),
        ("remove_range_exit_trades", range_exit_idx),
        ("remove_top1_range_exit_pnl", top1_range_idx),
    ]
    for name, idxs in scenarios:
        removed = trades.loc[trades["_orig_index"].isin(idxs)].copy()
        kept = trades.loc[~trades["_orig_index"].isin(idxs)].copy()
        _append_removal_scenario(rows, args=args, name=name, kept=kept, removed=removed, post_dir=post_dir)
    return rows


def write_range_exit_details(root_out: Path) -> None:
    runs_dir = root_out / "runs"
    base_path = runs_dir / "base" / TRADES_FILE
    off_path = runs_dir / "off_control" / TRADES_FILE
    if not base_path.exists():
        return
    base = pd.read_csv(base_path)
    base["entry_time"] = pd.to_datetime(base["entry_time"], errors="coerce")
    base["exit_time"] = pd.to_datetime(base["exit_time"], errors="coerce")
    base["note"] = base.get("note", "").astype(str)
    range_exits = base.loc[base["note"].str.startswith("RANGE_EXIT")].copy()
    detail_dir = root_out / "details"
    detail_dir.mkdir(parents=True, exist_ok=True)
    range_exits.to_csv(detail_dir / "base_range_exit_trades.csv", index=False)

    if off_path.exists() and not range_exits.empty:
        off = pd.read_csv(off_path)
        off["entry_time"] = pd.to_datetime(off["entry_time"], errors="coerce")
        off["exit_time"] = pd.to_datetime(off["exit_time"], errors="coerce")
        merged = range_exits.merge(
            off,
            on=["entry_time", "type", "engine"],
            suffixes=("_base", "_off"),
            how="left",
        )
        keep_cols = [
            "entry_time", "type", "engine",
            "exit_time_off", "note_off", "return_pct_off", "pnl_off", "capital_off",
            "exit_time_base", "note_base", "return_pct_base", "pnl_base", "capital_base",
            "mfe_r_base", "mae_r_base", "range_exit_peak_r_base", "range_exit_current_r_base",
            "range_exit_giveback_frac_base", "range_exit_reversal_base", "rf_imbalance_base", "rf_close_pos_base",
        ]
        merged[[c for c in keep_cols if c in merged.columns]].to_csv(detail_dir / "base_range_exit_vs_off_control.csv", index=False)


def build_scenarios(args: argparse.Namespace) -> list[RunScenario]:
    return [
        RunScenario("off_control", "V9C-equivalent control: range_exit_mode=off", range_exit_mode="off"),
        RunScenario("base", "Default V9E range-exit overlay", range_exit_mode="soft"),
        RunScenario("fee_2x", "Default V9E with doubled fee", fee_rate=args.fee_rate * 2.0, range_exit_mode="soft"),
        RunScenario("slippage_2x", "Default V9E with doubled slippage", slippage_pct=args.slippage_pct * 2.0, range_exit_mode="soft"),
        RunScenario("no_2026", "Default V9E ending 2025-12-31", end_date="2025-12-31", range_exit_mode="soft"),
        RunScenario("mfe3_giveback70", "Conservative V9E: min MFE 3R, giveback 70%", range_exit_mode="soft", range_exit_min_mfe_r=3.0, range_exit_giveback_frac=0.70),
        RunScenario("no_footprint", "Default V9E without footprint max-bucket context", range_exit_mode="soft", disable_footprint_context=True),
        RunScenario("delay_1bar", "Default V9E but range exits delayed by one extra 4H bar", range_exit_mode="soft", range_exit_delay_bars=1),
        RunScenario("delay_2bar", "Default V9E but range exits delayed by two extra 4H bars", range_exit_mode="soft", range_exit_delay_bars=2),
    ]


def main() -> int:
    args = parse_args()
    root_out = Path(args.out_dir)
    runs_dir = root_out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_scenarios(args)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    if not args.skip_backtests:
        for scenario in scenarios:
            run_dir = runs_dir / scenario.name
            cmd = strategy_cmd(args, scenario, run_dir)
            try:
                run_cmd(cmd)
                rows.append(load_summary(run_dir, scenario, "real_backtest"))
            except Exception as exc:
                print(f"[ERROR] scenario={scenario.name}: {exc}", flush=True)
                errors.append({"scenario": scenario.name, "error": str(exc)})
                if not args.continue_on_error:
                    raise
    else:
        for scenario in scenarios:
            run_dir = runs_dir / scenario.name
            if (run_dir / SUMMARY_FILE).exists() or (run_dir / "summary.json").exists():
                rows.append(load_summary(run_dir, scenario, "real_backtest"))

    rows.extend(postprocess_removal_scenarios(args, root_out))
    write_range_exit_details(root_out)

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        preferred = [
            "scenario", "scenario_type", "scenario_note", "total_trades", "final_capital", "total_return_pct",
            "max_drawdown_pct", "profit_factor", "win_rate", "expectancy_pct", "gross_profit", "gross_loss",
            "total_fees", "range_exit_trade_count", "range_exit_avg_peak_r", "range_exit_avg_giveback_frac", "range_exit_delay_bars",
            "removed_trades", "removed_range_exit_trades", "removed_pnl_sum", "priority_mode", "global_risk_scale", "run_dir",
        ]
        cols = [c for c in preferred if c in summary_df.columns] + [c for c in summary_df.columns if c not in preferred]
        summary_df = summary_df[cols]
        summary_path = root_out / "v9e_range_exit_overlay_pressure_v2_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print("\n" + "=" * 140)
        print("V9E Range Exit Overlay Pressure V2 Summary")
        print("=" * 140)
        print(summary_df[[c for c in preferred if c in summary_df.columns]].to_string(index=False))
        print(f"\nSaved: {summary_path.resolve()}", flush=True)

    if errors:
        pd.DataFrame(errors).to_csv(root_out / "v9e_range_exit_overlay_pressure_errors.csv", index=False)
    with (root_out / "v9e_range_exit_overlay_pressure_v2_config.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "scenarios": [asdict(x) for x in scenarios]}, f, ensure_ascii=False, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
