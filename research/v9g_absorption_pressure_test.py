#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V9G Absorption Exit Pressure Test
=================================

Runs focused pressure tests for V9G absorption-exit overlay.

Real backtest scenarios:
    - off_control
    - base
    - fee_2x
    - slippage_2x
    - no_2026
    - conservative
    - conservative_no_2026
    - delay_1bar
    - delay_2bar

Post-process stresses from base trades:
    - remove_top1_pnl
    - remove_top3_pnl
    - remove_2026_01_29_trade
    - remove_2026_05_25_trade
    - remove_all_2026_range_exit_trades
    - remove_range_exit_trades
    - remove_top1_range_exit_pnl

This tool is research-only. It does not modify V9C/V9E/V9G strategy code.
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

STRATEGY_NAME = "eth_lf_portfolio_v9g_absorption_exit"
TRADES_FILE = f"{STRATEGY_NAME}_trades.csv"
SUMMARY_FILE = f"{STRATEGY_NAME}_summary.json"


@dataclass(frozen=True)
class RunScenario:
    name: str
    note: str
    end_date: str | None = None
    fee_rate: float | None = None
    slippage_pct: float | None = None
    range_exit_mode: str = "soft"
    range_exit_min_mfe_r: float = 2.0
    range_exit_giveback_frac: float = 0.30
    range_exit_min_hold_bars: int = 2
    range_exit_delay_bars: int = 0
    absorption_min_current_r: float = 0.5
    absorption_bucket_share: float = 0.20
    absorption_min_flow_imbalance: float = 0.03
    absorption_bad_close_pos: float = 0.45


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run V9G Absorption Exit pressure tests.")
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
    p.add_argument("--script", default="backtest/lf/eth_lf_portfolio_v9g_absorption_exit_backtest.py")
    p.add_argument("--out-dir", default="data/reports/research/v9g_absorption_pressure_test")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--skip-backtests", action="store_true", help="Only postprocess existing outputs under --out-dir/runs.")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def scenarios(args: argparse.Namespace) -> list[RunScenario]:
    return [
        RunScenario(
            name="off_control",
            note="V9C-equivalent control, V9G file with range_exit_mode=off",
            range_exit_mode="off",
        ),
        RunScenario(
            name="base",
            note="V9G default absorption exit",
        ),
        RunScenario(
            name="fee_2x",
            note="V9G base with double fee",
            fee_rate=args.fee_rate * 2.0,
        ),
        RunScenario(
            name="slippage_2x",
            note="V9G base with double slippage",
            slippage_pct=args.slippage_pct * 2.0,
        ),
        RunScenario(
            name="no_2026",
            note="V9G base ending before 2026",
            end_date="2025-12-31",
        ),
        RunScenario(
            name="conservative",
            note="Conservative: MFE>=3R, giveback>=40%, bucket>=25%, stronger flow, stricter close",
            range_exit_min_mfe_r=3.0,
            range_exit_giveback_frac=0.40,
            absorption_min_current_r=0.8,
            absorption_bucket_share=0.25,
            absorption_min_flow_imbalance=0.05,
            absorption_bad_close_pos=0.40,
        ),
        RunScenario(
            name="conservative_no_2026",
            note="Conservative ending before 2026",
            end_date="2025-12-31",
            range_exit_min_mfe_r=3.0,
            range_exit_giveback_frac=0.40,
            absorption_min_current_r=0.8,
            absorption_bucket_share=0.25,
            absorption_min_flow_imbalance=0.05,
            absorption_bad_close_pos=0.40,
        ),
        RunScenario(
            name="delay_1bar",
            note="V9G base, exit delayed one additional 4H bar",
            range_exit_delay_bars=1,
        ),
        RunScenario(
            name="delay_2bar",
            note="V9G base, exit delayed two additional 4H bars",
            range_exit_delay_bars=2,
        ),
    ]


def run_cmd(cmd: list[str]) -> None:
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def strategy_cmd(args: argparse.Namespace, sc: RunScenario, run_dir: Path) -> list[str]:
    end_date = sc.end_date or args.end_date
    fee_rate = sc.fee_rate if sc.fee_rate is not None else args.fee_rate
    slippage_pct = sc.slippage_pct if sc.slippage_pct is not None else args.slippage_pct
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
        "--range-exit-mode", sc.range_exit_mode,
        "--range-exit-min-mfe-r", str(sc.range_exit_min_mfe_r),
        "--range-exit-giveback-frac", str(sc.range_exit_giveback_frac),
        "--range-exit-min-hold-bars", str(sc.range_exit_min_hold_bars),
        "--range-exit-delay-bars", str(sc.range_exit_delay_bars),
        "--absorption-min-current-r", str(sc.absorption_min_current_r),
        "--absorption-bucket-share", str(sc.absorption_bucket_share),
        "--absorption-min-flow-imbalance", str(sc.absorption_min_flow_imbalance),
        "--absorption-bad-close-pos", str(sc.absorption_bad_close_pos),
        "--out-dir", str(run_dir),
    ]


def load_summary(run_dir: Path, sc: RunScenario, scenario_type: str = "real_backtest") -> dict[str, Any]:
    summary_path = run_dir / SUMMARY_FILE
    if not summary_path.exists():
        summary_path = run_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        s = json.load(f)
    row = {
        "scenario": sc.name,
        "scenario_type": scenario_type,
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
        "absorption_min_current_r", "absorption_bucket_share", "absorption_min_flow_imbalance",
        "absorption_bad_close_pos", "absorption_require_bucket", "absorption_require_flow",
        "mfe_ge_1r_ended_loss", "mfe_ge_2r_ended_loss",
    ]
    for k in keys:
        row[k] = s.get(k)
    return row


def load_trades(run_dir: Path) -> pd.DataFrame:
    path = run_dir / TRADES_FILE
    if not path.exists():
        path = run_dir / "trades.csv"
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
        raise RuntimeError("Trades file must contain return_pct or pnl+capital")
    t["entry_time"] = pd.to_datetime(t["entry_time"], errors="coerce")
    t["exit_time"] = pd.to_datetime(t["exit_time"], errors="coerce")
    t["note"] = t.get("note", "").astype(str)
    t["is_range_exit"] = t["note"].str.startswith("RANGE_EXIT")
    t["pnl"] = pd.to_numeric(t.get("pnl", 0.0), errors="coerce").fillna(0.0)
    t["_orig_index"] = range(len(t))
    return t


def summarize_path(trades: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    capital = float(initial_capital)
    peak = capital
    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    equity_dd = []
    for r in pd.to_numeric(trades.get("account_return", pd.Series(dtype=float)), errors="coerce").fillna(0.0):
        before = capital
        capital *= 1.0 + float(r)
        pnl = capital - before
        if pnl >= 0:
            gross_profit += pnl
            if pnl > 0:
                wins += 1
        else:
            gross_loss += -pnl
        peak = max(peak, capital)
        equity_dd.append((peak - capital) / peak if peak > 0 else 0.0)
    n = len(trades)
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return {
        "total_trades": int(n),
        "final_capital": round(capital, 4),
        "total_return_pct": round((capital / initial_capital - 1.0) * 100.0, 4),
        "max_drawdown_pct": round(max(equity_dd, default=0.0) * 100.0, 4),
        "profit_factor": round(pf, 4) if pf != float("inf") else pf,
        "win_rate": round(wins / n * 100.0, 4) if n else 0.0,
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
    }


def append_removal(rows: list[dict[str, Any]], *, args: argparse.Namespace, name: str, kept: pd.DataFrame, removed: pd.DataFrame, out_dir: Path, note: str) -> None:
    row = summarize_path(kept, args.initial_capital)
    row.update({
        "scenario": name,
        "scenario_type": "trade_removal_postprocess",
        "scenario_note": note,
        "removed_trades": int(len(removed)),
        "removed_pnl_sum": float(pd.to_numeric(removed.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if len(removed) else 0.0,
        "removed_range_exit_trades": int(removed.get("is_range_exit", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if len(removed) else 0,
        "priority_mode": args.priority_mode,
        "global_risk_scale": args.global_risk_scale,
    })
    rows.append(row)
    kept.to_csv(out_dir / f"{name}_kept_trades.csv", index=False)
    removed.to_csv(out_dir / f"{name}_removed_trades.csv", index=False)


def postprocess(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    base_dir = root / "runs" / "base"
    trades = load_trades(base_dir)
    out_dir = root / "postprocess"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if trades.empty:
        return rows

    by_pnl = trades.sort_values("pnl", ascending=False)
    for n in [1, 3]:
        removed_idx = set(by_pnl.head(n)["_orig_index"])
        removed = trades[trades["_orig_index"].isin(removed_idx)].copy()
        kept = trades[~trades["_orig_index"].isin(removed_idx)].copy()
        append_removal(rows, args=args, name=f"remove_top{n}_pnl", kept=kept, removed=removed, out_dir=out_dir, note=f"Remove top {n} pnl trades from V9G base")

    for label, date_prefix in [
        ("remove_2026_01_29_trade", "2026-01-29"),
        ("remove_2026_05_25_trade", "2026-05-25"),
    ]:
        mask = trades["entry_time"].dt.strftime("%Y-%m-%d").eq(date_prefix)
        append_removal(rows, args=args, name=label, kept=trades[~mask].copy(), removed=trades[mask].copy(), out_dir=out_dir, note=f"Remove trade with entry date {date_prefix}")

    mask_2026_rx = trades["is_range_exit"] & (trades["entry_time"] >= pd.Timestamp("2026-01-01"))
    append_removal(rows, args=args, name="remove_all_2026_range_exit_trades", kept=trades[~mask_2026_rx].copy(), removed=trades[mask_2026_rx].copy(), out_dir=out_dir, note="Remove all 2026 V9G range-exit trades")

    mask_rx = trades["is_range_exit"]
    append_removal(rows, args=args, name="remove_range_exit_trades", kept=trades[~mask_rx].copy(), removed=trades[mask_rx].copy(), out_dir=out_dir, note="Remove all V9G range-exit trades")

    rx = trades[mask_rx].sort_values("pnl", ascending=False)
    if not rx.empty:
        removed_idx = set(rx.head(1)["_orig_index"])
        append_removal(rows, args=args, name="remove_top1_range_exit_pnl", kept=trades[~trades["_orig_index"].isin(removed_idx)].copy(), removed=trades[trades["_orig_index"].isin(removed_idx)].copy(), out_dir=out_dir, note="Remove top pnl V9G range-exit trade")

    trades[mask_rx].to_csv(root / "details" / "base_range_exit_trades.csv", index=False)
    return rows


def compare_base_vs_off(root: Path) -> None:
    details = root / "details"
    details.mkdir(parents=True, exist_ok=True)
    try:
        off = load_trades(root / "runs" / "off_control")
        base = load_trades(root / "runs" / "base")
    except Exception:
        return
    if off.empty or base.empty:
        return
    off_key = off.set_index(off["entry_time"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    rows = []
    for _, r in base[base["is_range_exit"]].iterrows():
        key = r["entry_time"].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(r["entry_time"]) else ""
        if key not in off_key.index:
            continue
        o = off_key.loc[key]
        if isinstance(o, pd.DataFrame):
            o = o.iloc[0]
        rows.append({
            "entry_time": key,
            "base_exit_time": r.get("exit_time"),
            "off_exit_time": o.get("exit_time"),
            "base_note": r.get("note"),
            "off_note": o.get("note"),
            "base_return_pct": r.get("return_pct"),
            "off_return_pct": o.get("return_pct"),
            "delta_return_pct": r.get("return_pct") - o.get("return_pct"),
            "base_pnl": r.get("pnl"),
            "off_pnl": o.get("pnl"),
        })
    if rows:
        pd.DataFrame(rows).to_csv(details / "base_range_exit_vs_off_control.csv", index=False)


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
            try:
                run_cmd(strategy_cmd(args, sc, run_dir))
            except subprocess.CalledProcessError:
                if not args.continue_on_error:
                    raise
                print(f"[WARN] scenario failed: {sc.name}", file=sys.stderr)
                continue
        rows.append(load_summary(run_dir, sc))

    compare_base_vs_off(root)
    rows.extend(postprocess(args, root))

    out = pd.DataFrame(rows)
    out_path = root / "v9g_absorption_pressure_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[OK] wrote {out_path}")
    if not out.empty:
        cols = ["scenario", "scenario_type", "total_return_pct", "max_drawdown_pct", "profit_factor", "win_rate", "total_trades", "range_exit_trade_count"]
        print(out[[c for c in cols if c in out.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
