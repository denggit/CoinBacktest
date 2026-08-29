#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent KPI, cost, and artifact audit of the repository LF V10B candidate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sleeve_lib.lf_v10b import structural_stop  # noqa: E402
from src.sleeve_lib.lf_v10b.config import (  # noqa: E402
    build_lf_args,
    bull_to_exec_config,
    make_bull_config,
    make_exec_config,
    make_momentum_config,
)
from src.sleeve_lib.lf_v10b.selector import run_lf_v10b_leg  # noqa: E402

DEFAULT_ARTIFACT_DIR = Path("data/reports/lf/eth_lf_portfolio_v10b_all_swing_structural_stop/turbo")
DEFAULT_OUT_DIR = Path("data/reports/research/ict/mss2/v10b_candidate_audit")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-06-15")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--fee-rate", type=float, default=0.00055)
    parser.add_argument("--slippage-pct", type=float, default=0.0002)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args(argv)


def _runner_args(args: argparse.Namespace, scale: float) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        warmup_start_date=args.warmup_start_date,
        initial_capital=float(args.initial_capital),
        fee_rate=float(args.fee_rate) * scale,
        slippage_pct=float(args.slippage_pct) * scale,
        slippage=float(args.slippage_pct) * scale,
        lf_preset="turbo",
        lf_bear_preset="high",
        lf_bull_preset="high",
        lf_priority_mode="reclaim_first",
        lf_global_risk_scale=1.30,
        lf_micro_filter_mode="soft",
    )


def _pf(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    return gains / losses if losses > 0 else (np.inf if gains > 0 else np.nan)


def _longest_underwater_days(equity: pd.DataFrame) -> float:
    if equity.empty or "capital" not in equity:
        return np.nan
    capital = pd.to_numeric(equity["capital"], errors="coerce")
    below = capital.lt(capital.cummax())
    longest = pd.Timedelta(0)
    start: pd.Timestamp | None = None
    for ts, is_below in below.items():
        stamp = pd.Timestamp(ts)
        if bool(is_below) and start is None:
            start = stamp
        elif not bool(is_below) and start is not None:
            longest = max(longest, stamp - start)
            start = None
    if start is not None:
        longest = max(longest, pd.Timestamp(below.index[-1]) - start)
    return float(longest / pd.Timedelta(days=1))


def _period_rates(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, float]:
    exits = pd.to_datetime(trades["exit_time"], errors="coerce")
    returns = pd.to_numeric(trades["return_pct"], errors="coerce")
    month_index = pd.period_range(start=start, end=end, freq="M")
    quarter_index = pd.period_range(start=start, end=end, freq="Q")
    monthly = (1.0 + returns).groupby(exits.dt.to_period("M")).prod().sub(1.0).reindex(month_index, fill_value=0.0)
    quarterly = (1.0 + returns).groupby(exits.dt.to_period("Q")).prod().sub(1.0).reindex(quarter_index, fill_value=0.0)
    return float(monthly.gt(0).mean()), float(quarterly.gt(0).mean())


def _rolling_90_positive_rate(equity: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if equity.empty:
        return np.nan
    capital = pd.to_numeric(equity["capital"], errors="coerce")
    daily = capital.groupby(pd.to_datetime(equity.index).normalize()).last()
    daily = daily.reindex(pd.date_range(start.normalize(), end.normalize(), freq="1D")).ffill()
    rolling = daily.div(daily.shift(90)).sub(1.0).dropna()
    return float(rolling.gt(0).mean()) if len(rolling) else np.nan


def _flat_stats(trades: pd.DataFrame) -> tuple[float, float, float, float]:
    ordered = trades.sort_values("entry_time", kind="stable")
    entries = pd.to_datetime(ordered["entry_time"], errors="coerce")
    exits = pd.to_datetime(ordered["exit_time"], errors="coerce")
    entry_gap = entries.diff().dropna() / pd.Timedelta(days=1)
    flat = (entries.iloc[1:].reset_index(drop=True) - exits.iloc[:-1].reset_index(drop=True)) / pd.Timedelta(days=1)
    flat = flat.clip(lower=0)
    return (
        float(entry_gap.max()) if len(entry_gap) else np.nan,
        float(flat.max()) if len(flat) else np.nan,
        float(flat.median()) if len(flat) else np.nan,
        float(flat.quantile(0.90)) if len(flat) else np.nan,
    )


def _summary(scale: float, trades: pd.DataFrame, equity: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    returns = pd.to_numeric(trades["return_pct"], errors="coerce")
    pnl = pd.to_numeric(trades["pnl"], errors="coerce")
    start, end = pd.Timestamp(args.start_date), pd.Timestamp(args.end_date)
    months = max((end - start) / pd.Timedelta(days=30.4375), 1.0)
    positive_months, positive_quarters = _period_rates(trades, start, end)
    entry_gap, flat_max, flat_median, flat_p90 = _flat_stats(trades)
    final_capital = float(pd.to_numeric(trades["capital"], errors="coerce").iloc[-1]) if len(trades) else args.initial_capital
    top = returns.sort_values(ascending=False, kind="stable")
    without5 = returns.drop(index=top.head(5).index)
    without10 = returns.drop(index=top.head(10).index)
    return {
        "cost_scale": scale,
        "trades": int(len(trades)),
        "trades_per_month": float(len(trades) / months),
        "long_trades": int(trades["type"].astype(str).str.upper().eq("LONG").sum()),
        "short_trades": int(trades["type"].astype(str).str.upper().eq("SHORT").sum()),
        "win_rate": float(returns.gt(0).mean()),
        "final_capital": final_capital,
        "total_return_pct": (final_capital / float(args.initial_capital) - 1.0) * 100.0,
        "dollar_pnl_pf": _pf(pnl),
        "trade_return_pf": _pf(returns),
        "mean_trade_return": float(returns.mean()),
        "max_drawdown_pct": float(pd.to_numeric(equity.get("drawdown_pct"), errors="coerce").max() * 100.0),
        "positive_month_rate": positive_months,
        "positive_quarter_rate": positive_quarters,
        "rolling_90d_positive_rate_realized_equity": _rolling_90_positive_rate(equity, start, end),
        "longest_entry_gap_days": entry_gap,
        "longest_flat_days": flat_max,
        "median_flat_days": flat_median,
        "p90_flat_days": flat_p90,
        "longest_underwater_days_realized_equity": _longest_underwater_days(equity),
        "top5_removed_return_pf": _pf(without5),
        "top10_removed_return_pf": _pf(without10),
        "top5_removed_compound_return_pct": float(((1.0 + without5).prod() - 1.0) * 100.0),
        "top10_removed_compound_return_pct": float(((1.0 + without10).prod() - 1.0) * 100.0),
        "total_fees": float(pd.to_numeric(trades.get("fee"), errors="coerce").sum()),
    }


def _yearly(scale: float, trades: pd.DataFrame) -> pd.DataFrame:
    work = trades.copy()
    work["year"] = pd.to_datetime(work["exit_time"], errors="coerce").dt.year
    rows: list[dict[str, object]] = []
    for year, part in work.groupby("year", sort=True):
        returns = pd.to_numeric(part["return_pct"], errors="coerce")
        rows.append(
            {
                "cost_scale": scale,
                "year": int(year),
                "trades": int(len(part)),
                "win_rate": float(returns.gt(0).mean()),
                "compound_return_pct": float(((1.0 + returns).prod() - 1.0) * 100.0),
                "sum_return_pct": float(returns.sum() * 100.0),
                "trade_return_pf": _pf(returns),
            }
        )
    return pd.DataFrame(rows)


def _rerun_with_features(args: argparse.Namespace, features: pd.DataFrame, scale: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    runner = _runner_args(args, scale)
    lf_args = build_lf_args(runner)
    momentum = make_momentum_config(lf_args)
    bull = make_bull_config(lf_args)
    exec_cfg = make_exec_config(momentum)
    bull_exec = exec_cfg if lf_args.bull_execution_mode == "inherit" else bull_to_exec_config(bull)
    trades, equity = structural_stop.run_v10b_backtest(
        features,
        exec_cfg,
        engine_cfgs={"MOMENTUM_V3": exec_cfg, "BEAR_V3_ONLY": exec_cfg, "BULL_RECLAIM_V2": bull_exec},
        global_risk_scale=lf_args.global_risk_scale,
        args=lf_args,
    )
    return pd.DataFrame(trades), equity


def _path_compare(base: pd.DataFrame, stressed: pd.DataFrame, scale: float) -> dict[str, object]:
    keys = ["entry_time", "exit_time", "type", "note", "units"]
    left = base[keys].astype(str).reset_index(drop=True)
    right = stressed[keys].astype(str).reset_index(drop=True)
    unequal = 0 if left.equals(right) else int((left.reindex_like(right) != right).any(axis=1).sum())
    return {"cost_scale": scale, "base_rows": len(left), "stress_rows": len(right), "path_key_mismatches": unequal}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[v10b-audit] build frozen current V10B feature path once", flush=True)
    base_trades, base_equity, features = run_lf_v10b_leg(_runner_args(args, 1.0))
    runs: dict[float, tuple[pd.DataFrame, pd.DataFrame]] = {1.0: (base_trades, base_equity)}
    for scale in (2.0, 3.0):
        print(f"[v10b-audit] exact executor rerun at {scale:g}x fee and slippage", flush=True)
        runs[scale] = _rerun_with_features(args, features, scale)

    summary = pd.DataFrame([_summary(scale, *runs[scale], args) for scale in runs])
    yearly = pd.concat([_yearly(scale, runs[scale][0]) for scale in runs], ignore_index=True)
    paths = pd.DataFrame([_path_compare(runs[1.0][0], runs[scale][0], scale) for scale in (2.0, 3.0)])

    artifact_dir = Path(args.artifact_dir)
    official_summary = json.loads((artifact_dir / "eth_lf_portfolio_v10b_all_swing_structural_stop_summary.json").read_text(encoding="utf-8"))
    official_trades = pd.read_csv(artifact_dir / "eth_lf_portfolio_v10b_all_swing_structural_stop_trades.csv")
    official_equity = pd.read_csv(
        artifact_dir / "eth_lf_portfolio_v10b_all_swing_structural_stop_equity.csv",
        parse_dates=["time"],
    ).set_index("time")
    for column in ("entry_time", "exit_time"):
        official_trades[column] = pd.to_datetime(official_trades[column], errors="coerce")
    official_kpis = pd.DataFrame([_summary(1.0, official_trades, official_equity, args)])
    reproduced = summary.loc[summary["cost_scale"].eq(1.0)].iloc[0]
    reconcile = pd.DataFrame(
        [
            {"metric": "trades", "official": official_summary.get("total_trades"), "reproduced": reproduced["trades"]},
            {"metric": "total_return_pct", "official": official_summary.get("total_return_pct"), "reproduced": reproduced["total_return_pct"]},
            {"metric": "dollar_pnl_pf", "official": official_summary.get("profit_factor"), "reproduced": reproduced["dollar_pnl_pf"]},
            {"metric": "max_drawdown_pct", "official": official_summary.get("max_drawdown_pct"), "reproduced": reproduced["max_drawdown_pct"]},
            {"metric": "official_trade_rows", "official": len(official_trades), "reproduced": len(base_trades)},
        ]
    )
    reconcile["abs_difference"] = (pd.to_numeric(reconcile["official"], errors="coerce") - pd.to_numeric(reconcile["reproduced"], errors="coerce")).abs()

    summary.to_csv(out / "01_cost_and_master_kpi_audit.csv", index=False)
    yearly.to_csv(out / "02_yearly_cost_audit.csv", index=False)
    paths.to_csv(out / "03_cost_path_consistency.csv", index=False)
    reconcile.to_csv(out / "04_official_artifact_reconciliation.csv", index=False)
    official_kpis.to_csv(out / "05_official_artifact_master_kpis.csv", index=False)
    (out / "00_audit_manifest.json").write_text(
        json.dumps(
            {
                "candidate": "LF V10B all-engine swing structural stop",
                "source_module": "src/sleeve_lib/lf_v10b",
                "official_artifact_dir": str(artifact_dir),
                "window": [args.start_date, args.end_date],
                "cost_scales": [1, 2, 3],
                "selection_warning": "Candidate was selected from a multi-dimensional full-window structural-stop grid; no untouched split is claimed.",
                "holdout_status": "none; all 2023-2026 history in the headline artifact was part of research/verification",
                "equity_caveat": "MDD, rolling-90d, and underwater use realized-capital equity, not daily mark-to-market equity.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(f"[v10b-audit] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
