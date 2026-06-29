#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V9E Walk-Forward Filter Lab
===========================

Research-only walk-forward validator for V9E bad-entry filters.

This script uses ``research/v9e_signal_opportunity_lab.py`` to:
    1. Build V9E features once.
    2. Mine bad-entry groups only on each training window.
    3. Apply the mined rules only to the following test window.
    4. Compare out-of-sample baseline vs block_bad_groups.

Important:
    - It does not modify V9E strategy logic.
    - It does not place orders.
    - Rules are mined from training folds only; test metrics are out-of-sample
      for that fold.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
RESEARCH_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(RESEARCH_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if RESEARCH_DIR not in sys.path:
    sys.path.insert(0, RESEARCH_DIR)

import v9e_signal_opportunity_lab as lab  # noqa: E402


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def parse_walkforward_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace]:
    """Parse walk-forward-only flags first, then reuse signal lab parser for V9E flags."""
    wf_parser = argparse.ArgumentParser(add_help=False)
    wf_parser.add_argument(
        "--fold-mode",
        choices=["standard", "rolling_yearly", "all"],
        default="standard",
        help="standard: 2023-2024->2025-2026 and 2023-2025->2026; rolling_yearly: 2023->2024, 2023-2024->2025, 2024-2025->2026; all: both sets.",
    )
    wf_parser.add_argument(
        "--strict-positive-improvement",
        action="store_true",
        help="Mark a fold as passed only when filtered closed capital and PF both improve and max drawdown does not increase.",
    )
    wf_parser.add_argument(
        "--min-test-closed-trades",
        type=int,
        default=8,
        help="Minimum closed trades in a test fold before judging pass/fail.",
    )
    wf_parser.add_argument(
        "--custom-fold",
        action="append",
        default=[],
        metavar="ID,TRAIN_START,TRAIN_END,TEST_START,TEST_END",
        help="Add custom fold. Example: wf1,2023-01-01,2024-12-31,2025-01-01,2026-06-15",
    )
    wf_parser.add_argument(
        "--write-fold-runs",
        action="store_true",
        help="Write per-fold filtered/baseline trades, equity and compact signal audit CSVs. More disk usage.",
    )

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    wf_args, lab_argv = wf_parser.parse_known_args(raw_argv)

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]] + lab_argv
        base_args = lab.parse_args()
    finally:
        sys.argv = old_argv
    return wf_args, base_args


def _date(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize(None) if pd.Timestamp(value).tzinfo else pd.Timestamp(value)


def _inclusive_slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    start_ts = _date(start)
    end_ts = _date(end)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()


def _build_folds(args: argparse.Namespace, wf_args: argparse.Namespace) -> list[FoldSpec]:
    start = str(args.start_date)
    end = str(args.end_date)
    specs: list[FoldSpec] = []

    if wf_args.fold_mode in {"standard", "all"}:
        specs.extend([
            FoldSpec(
                fold_id="wf_2023_2024_to_2025_2026",
                train_start=max(start, "2023-01-01"),
                train_end="2024-12-31 23:59:59",
                test_start="2025-01-01",
                test_end=end,
                description="Mine bad groups on 2023-2024; validate on 2025-2026.",
            ),
            FoldSpec(
                fold_id="wf_2023_2025_to_2026",
                train_start=max(start, "2023-01-01"),
                train_end="2025-12-31 23:59:59",
                test_start="2026-01-01",
                test_end=end,
                description="Mine bad groups on 2023-2025; validate on 2026.",
            ),
        ])

    if wf_args.fold_mode in {"rolling_yearly", "all"}:
        specs.extend([
            FoldSpec(
                fold_id="wf_2023_to_2024",
                train_start=max(start, "2023-01-01"),
                train_end="2023-12-31 23:59:59",
                test_start="2024-01-01",
                test_end="2024-12-31 23:59:59",
                description="Mine on 2023; validate on 2024.",
            ),
            FoldSpec(
                fold_id="wf_2023_2024_to_2025",
                train_start=max(start, "2023-01-01"),
                train_end="2024-12-31 23:59:59",
                test_start="2025-01-01",
                test_end="2025-12-31 23:59:59",
                description="Mine on 2023-2024; validate on 2025.",
            ),
            FoldSpec(
                fold_id="wf_2024_2025_to_2026",
                train_start="2024-01-01",
                train_end="2025-12-31 23:59:59",
                test_start="2026-01-01",
                test_end=end,
                description="Mine on 2024-2025; validate on 2026.",
            ),
        ])

    for item in wf_args.custom_fold:
        parts = [p.strip() for p in str(item).split(",")]
        if len(parts) != 5:
            raise ValueError(f"Invalid --custom-fold format: {item!r}; expected ID,TRAIN_START,TRAIN_END,TEST_START,TEST_END")
        fid, tr_s, tr_e, te_s, te_e = parts
        specs.append(FoldSpec(fid, tr_s, tr_e, te_s, te_e, "Custom fold."))

    # Keep only folds overlapping requested backtest window.
    start_ts = _date(start)
    end_ts = _date(end)
    valid: list[FoldSpec] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.fold_id in seen:
            continue
        seen.add(spec.fold_id)
        if _date(spec.test_start) > end_ts or _date(spec.test_end) < start_ts:
            continue
        if _date(spec.train_start) > end_ts or _date(spec.train_end) < start_ts:
            continue
        valid.append(spec)
    return valid


def _with_fold_dates(args: argparse.Namespace, spec: FoldSpec, *, test: bool) -> argparse.Namespace:
    out = copy.copy(args)
    if test:
        out.start_date = spec.test_start
        out.end_date = spec.test_end
    else:
        out.start_date = spec.train_start
        out.end_date = spec.train_end
    return out


def _rules_to_frame(rules: list[lab.GroupRule], spec: FoldSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rule in rules:
        row = rule.to_dict()
        row.update({
            "fold_id": spec.fold_id,
            "train_start": spec.train_start,
            "train_end": spec.train_end,
            "test_start": spec.test_start,
            "test_end": spec.test_end,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_subset(summary: dict[str, Any], prefix: str) -> dict[str, Any]:
    keys = [
        "total_trades", "closed_total_trades", "win_rate", "closed_win_rate", "final_capital", "closed_final_capital",
        "profit_factor", "closed_profit_factor", "expectancy_pct", "closed_expectancy_pct", "max_drawdown_pct",
        "total_return_pct", "force_close_count", "force_close_pnl", "filtered_signal_count", "filter_reason",
    ]
    return {f"{prefix}_{k}": summary.get(k) for k in keys if k in summary}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _judge_fold(row: dict[str, Any], wf_args: argparse.Namespace) -> str:
    base_closed_trades = int(_float(row.get("baseline_closed_total_trades"), 0))
    filt_closed_trades = int(_float(row.get("filtered_closed_total_trades"), 0))
    if min(base_closed_trades, filt_closed_trades) < int(wf_args.min_test_closed_trades):
        return "TOO_FEW_TEST_TRADES"

    cap_delta = _float(row.get("delta_closed_final_capital"))
    pf_delta = _float(row.get("delta_closed_profit_factor"))
    dd_delta = _float(row.get("delta_max_drawdown_pct"))
    wr_delta = _float(row.get("delta_closed_win_rate"))

    if wf_args.strict_positive_improvement:
        return "PASS" if cap_delta > 0 and pf_delta > 0 and dd_delta <= 0 else "FAIL"

    # Default: capital improvement is primary; allow small DD increase only when PF/win-rate also improve.
    if cap_delta > 0 and pf_delta > 0 and (dd_delta <= 3.0 or wr_delta > 0):
        return "PASS"
    if cap_delta > 0 and pf_delta >= 0:
        return "MIXED_CAP_UP"
    return "FAIL"


def _save_run_outputs(base_dir: Path, name: str, features: pd.DataFrame, trades: list[dict[str, Any]], equity: pd.DataFrame, summary: dict[str, Any]) -> None:
    run_dir = base_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(run_dir / "trades.csv", index=False)
    equity.to_csv(run_dir / "equity.csv", index=False)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    cols = [c for c in [
        "signal", "selected_engine", "micro_filter_action", "micro_aligned", "micro_contra", "quality_mult",
        "risk_mult", "adx", "atr_pct", "rf_close_pos", "rf_imbalance", "signal_lab_blocked", "signal_lab_block_reason",
    ] if c in features.columns]
    if cols:
        features.loc[features["signal"].fillna(0).astype(int).ne(0) | features.get("signal_lab_blocked", pd.Series(False, index=features.index)).astype(bool), cols].to_csv(run_dir / "signal_audit_compact.csv")


def run_fold(features: pd.DataFrame, cfg: Any, engine_cfgs: dict[str, Any], args: argparse.Namespace, wf_args: argparse.Namespace, spec: FoldSpec, out_dir: Path) -> dict[str, Any]:
    print("\n" + "=" * 110)
    print(f"Fold {spec.fold_id}: train {spec.train_start} -> {spec.train_end}; test {spec.test_start} -> {spec.test_end}")
    print(spec.description)
    print("=" * 110, flush=True)

    train_features = _inclusive_slice(features, spec.train_start, spec.train_end)
    test_features = _inclusive_slice(features, spec.test_start, spec.test_end)
    if train_features.empty or test_features.empty:
        raise ValueError(f"Fold {spec.fold_id} has empty train/test features: train={len(train_features)} test={len(test_features)}")

    train_args = _with_fold_dates(args, spec, test=False)
    test_args = _with_fold_dates(args, spec, test=True)

    train_trades, train_equity, train_baseline_summary = lab.run_v9e_baseline(train_features, cfg, engine_cfgs, train_args)
    train_ops = lab.build_signal_opportunity_table(train_features, train_trades)
    train_stats = lab.group_stats(train_ops, min_samples=int(args.min_group_samples))
    rules = lab.mine_bad_entry_rules(train_stats, args)

    fold_dir = out_dir / "folds" / spec.fold_id
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_ops.to_csv(fold_dir / "train_signal_opportunity_table.csv", index=False)
    train_stats.to_csv(fold_dir / "train_signal_group_stats.csv", index=False)
    _rules_to_frame(rules, spec).to_csv(fold_dir / "train_bad_entry_rule_candidates.csv", index=False)

    test_baseline_trades, test_baseline_equity, test_baseline_summary = lab.run_v9e_baseline(test_features, cfg, engine_cfgs, test_args)
    filtered_summary, filtered_features, filtered_trades, filtered_equity = lab.run_filter_scenario(
        "block_bad_groups", test_features, cfg, engine_cfgs, test_args, rules
    )

    if wf_args.write_fold_runs:
        _save_run_outputs(fold_dir, "test_baseline_v9e", lab.add_research_bins(test_features), test_baseline_trades, test_baseline_equity, test_baseline_summary)
        _save_run_outputs(fold_dir, "test_block_bad_groups", filtered_features, filtered_trades, filtered_equity, filtered_summary)

    row: dict[str, Any] = {
        **spec.to_dict(),
        "train_feature_rows": int(len(train_features)),
        "test_feature_rows": int(len(test_features)),
        "train_portfolio_signal_count": int(len(train_ops)),
        "train_executed_entry_count": int(train_ops["action_taken"].eq("EXECUTED_ENTRY").sum()) if not train_ops.empty else 0,
        "mined_rule_count": int(len(rules)),
        "mined_rules_json": json.dumps([r.to_dict() for r in rules], ensure_ascii=False),
    }
    row.update(_summary_subset(train_baseline_summary, "train_baseline"))
    row.update(_summary_subset(test_baseline_summary, "baseline"))
    row.update(_summary_subset(filtered_summary, "filtered"))

    row["delta_closed_final_capital"] = _float(row.get("filtered_closed_final_capital")) - _float(row.get("baseline_closed_final_capital"))
    row["delta_closed_final_capital_pct"] = (row["delta_closed_final_capital"] / _float(row.get("baseline_closed_final_capital"), 1.0) * 100.0) if _float(row.get("baseline_closed_final_capital"), 0.0) else 0.0
    row["delta_closed_profit_factor"] = _float(row.get("filtered_closed_profit_factor")) - _float(row.get("baseline_closed_profit_factor"))
    row["delta_closed_win_rate"] = _float(row.get("filtered_closed_win_rate")) - _float(row.get("baseline_closed_win_rate"))
    row["delta_closed_expectancy_pct"] = _float(row.get("filtered_closed_expectancy_pct")) - _float(row.get("baseline_closed_expectancy_pct"))
    row["delta_max_drawdown_pct"] = _float(row.get("filtered_max_drawdown_pct")) - _float(row.get("baseline_max_drawdown_pct"))
    row["judgement"] = _judge_fold(row, wf_args)

    compact = {
        "fold_id": row["fold_id"],
        "rules": row["mined_rule_count"],
        "baseline_cap": row.get("baseline_closed_final_capital"),
        "filtered_cap": row.get("filtered_closed_final_capital"),
        "delta_cap_pct": row.get("delta_closed_final_capital_pct"),
        "baseline_pf": row.get("baseline_closed_profit_factor"),
        "filtered_pf": row.get("filtered_closed_profit_factor"),
        "baseline_wr": row.get("baseline_closed_win_rate"),
        "filtered_wr": row.get("filtered_closed_win_rate"),
        "baseline_dd": row.get("baseline_max_drawdown_pct"),
        "filtered_dd": row.get("filtered_max_drawdown_pct"),
        "judgement": row.get("judgement"),
    }
    print(pd.DataFrame([compact]).to_string(index=False), flush=True)
    return row


def main() -> int:
    wf_args, args = parse_walkforward_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 110)
    print("V9E Walk-Forward Filter Lab")
    print("=" * 110)
    print(f"Base output directory: {out_dir.resolve()}")
    print(f"Fold mode: {wf_args.fold_mode}")
    print("Rules are mined on train folds only, then applied to the following test folds.")
    print("=" * 110 + "\n", flush=True)

    features, cfg, engine_cfgs = lab.build_features(args)
    folds = _build_folds(args, wf_args)
    if not folds:
        raise ValueError("No valid walk-forward folds for the requested date range.")

    rows: list[dict[str, Any]] = []
    for spec in folds:
        rows.append(run_fold(features, cfg, engine_cfgs, args, wf_args, spec, out_dir))

    summary = pd.DataFrame(rows)
    preferred_cols = [
        "fold_id", "train_start", "train_end", "test_start", "test_end", "judgement", "mined_rule_count",
        "train_portfolio_signal_count", "train_executed_entry_count",
        "baseline_closed_total_trades", "filtered_closed_total_trades",
        "baseline_closed_final_capital", "filtered_closed_final_capital", "delta_closed_final_capital", "delta_closed_final_capital_pct",
        "baseline_closed_profit_factor", "filtered_closed_profit_factor", "delta_closed_profit_factor",
        "baseline_closed_win_rate", "filtered_closed_win_rate", "delta_closed_win_rate",
        "baseline_closed_expectancy_pct", "filtered_closed_expectancy_pct", "delta_closed_expectancy_pct",
        "baseline_max_drawdown_pct", "filtered_max_drawdown_pct", "delta_max_drawdown_pct",
        "baseline_force_close_count", "filtered_force_close_count", "baseline_force_close_pnl", "filtered_force_close_pnl",
        "filtered_filtered_signal_count", "mined_rules_json",
    ]
    cols = [c for c in preferred_cols if c in summary.columns] + [c for c in summary.columns if c not in preferred_cols]
    summary = summary[cols]
    summary_path = out_dir / "v9e_walkforward_filter_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Aggregate overview.
    pass_count = int(summary["judgement"].astype(str).eq("PASS").sum()) if "judgement" in summary.columns else 0
    fail_count = int(summary["judgement"].astype(str).eq("FAIL").sum()) if "judgement" in summary.columns else 0
    mixed_count = int(summary["judgement"].astype(str).str.contains("MIXED", na=False).sum()) if "judgement" in summary.columns else 0
    overview = {
        "strategy": "V9E walk-forward bad-entry filter lab",
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "fold_mode": wf_args.fold_mode,
        "fold_count": int(len(summary)),
        "pass_count": pass_count,
        "mixed_count": mixed_count,
        "fail_count": fail_count,
        "too_few_test_trades_count": int(summary["judgement"].astype(str).eq("TOO_FEW_TEST_TRADES").sum()) if "judgement" in summary.columns else 0,
        "avg_delta_closed_final_capital_pct": float(pd.to_numeric(summary.get("delta_closed_final_capital_pct"), errors="coerce").mean()) if not summary.empty else 0.0,
        "avg_delta_closed_profit_factor": float(pd.to_numeric(summary.get("delta_closed_profit_factor"), errors="coerce").mean()) if not summary.empty else 0.0,
        "avg_delta_closed_win_rate": float(pd.to_numeric(summary.get("delta_closed_win_rate"), errors="coerce").mean()) if not summary.empty else 0.0,
        "avg_delta_max_drawdown_pct": float(pd.to_numeric(summary.get("delta_max_drawdown_pct"), errors="coerce").mean()) if not summary.empty else 0.0,
        "summary_csv": str(summary_path),
        "warning": "A filter is a V10 candidate only if it improves out-of-sample folds and survives separate fee/slippage/top-trade pressure tests.",
    }
    with (out_dir / "v9e_walkforward_filter_summary.json").open("w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 110)
    print("V9E Walk-Forward Filter Lab completed")
    print("=" * 110)
    print(f"Summary CSV: {summary_path.resolve()}")
    show_cols = [c for c in [
        "fold_id", "judgement", "mined_rule_count", "baseline_closed_final_capital", "filtered_closed_final_capital",
        "delta_closed_final_capital_pct", "baseline_closed_profit_factor", "filtered_closed_profit_factor",
        "baseline_closed_win_rate", "filtered_closed_win_rate", "baseline_max_drawdown_pct", "filtered_max_drawdown_pct",
    ] if c in summary.columns]
    if show_cols:
        print(summary[show_cols].to_string(index=False))
    print("\nOverview:")
    print(json.dumps(overview, ensure_ascii=False, indent=2, default=str))
    print("=" * 110 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
