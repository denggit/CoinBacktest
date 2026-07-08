#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Portfolio V1 refactored backtest entry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.edge_lib.mf_low_sweep.signals import run_low_sweep_time48_leg  # noqa: E402
from src.portfolio_common.allocator import (  # noqa: E402
    DEFAULT_LEVERAGE,
    build_equity_curve,
    build_scenarios,
    daily_returns,
    edge_attribution,
    simulate_portfolio_scenario,
    standardize_trades,
    stress_report,
)
from src.portfolio_common.artifacts import finalize_review_pack, write_json, write_standard_artifacts  # noqa: E402
from src.portfolio_common.parity import run_parity  # noqa: E402
from src.portfolio_common.reports import build_manifest, build_standard_summary, select_primary_trades  # noqa: E402
from src.sleeve_lib.lf_v10b.selector import run_lf_v10b_leg  # noqa: E402

SCRIPT_NAME = "eth_portfolio_V1_backtest"
SOURCE_OF_TRUTH = PROJECT_ROOT / "backtest/portfolio/eth_portfolio_V1_lf_v10b_low_sweep_mf_backtest.py"
DEFAULT_OUT_DIR = "data/reports/backtest/portfolio/eth_portfolio_V1"
PRIMARY_SCENARIO = "portfolio_v1_lf100_mf150_time48_independent"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETH Portfolio V1 refactored LF+MF backtest")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--parity-old-report-dir", default=None)
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage", "--slippage-pct", dest="slippage", type=float, default=0.0002)

    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--leverage", type=float, default=DEFAULT_LEVERAGE)
    p.add_argument("--lf-weight", type=float, default=1.0)
    p.add_argument("--mf-exposures", default="0.5,1.0,1.5")
    p.add_argument("--conflict-modes", default="independent")
    p.add_argument("--guard-modes", default="none,margin80,margin85,notional12,notional13,margin85_notional13")
    p.add_argument("--guard-margin-cap", type=float, default=0.85)
    p.add_argument("--guard-notional-cap", type=float, default=13.0)
    p.add_argument("--min-mf-exposure", type=float, default=0.05)
    p.add_argument("--primary-scenario", default=PRIMARY_SCENARIO)
    p.add_argument("--skip-full-report", action="store_true")
    p.add_argument("--save-combined-trades", type=int, default=200000)

    p.add_argument("--lf-preset", default="turbo")
    p.add_argument("--lf-bear-preset", default="high")
    p.add_argument("--lf-bull-preset", default="high")
    p.add_argument("--lf-priority-mode", default="reclaim_first")
    p.add_argument("--lf-global-risk-scale", type=float, default=1.30)
    p.add_argument("--lf-micro-filter-mode", default="soft")
    args = p.parse_args(argv)
    args.slippage_pct = float(args.slippage)
    return args


def _current_artifact_names(out_dir: Path) -> list[str]:
    wanted_prefixes = ("00_", "01_", "02_", "03_", "04_", "05_", "06_", "08_", "09_")
    wanted_names = {"GPT_REVIEW_PROMPT.md", "REVIEW_PACK_MANIFEST.json", "gpt_review_pack.zip"}
    names = []
    for path in sorted(out_dir.iterdir()):
        if path.name.startswith(wanted_prefixes) or path.name in wanted_names:
            names.append(path.name)
    return names


def _print_parity_failure(first_diff: dict[str, object]) -> None:
    print("[parity] FAILED", flush=True)
    print(f"[parity] first_file={first_diff.get('file')}", flush=True)
    print(f"[parity] first_key={first_diff.get('key')}", flush=True)
    print(f"[parity] field={first_diff.get('field')}", flush=True)
    print(f"[parity] new_value={first_diff.get('new_value')}", flush=True)
    print(f"[parity] old_value={first_diff.get('old_value')}", flush=True)
    print(f"[parity] reason={first_diff.get('reason')}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    if not SOURCE_OF_TRUTH.exists():
        print(f"[error] legacy source of truth not found: {SOURCE_OF_TRUTH}", flush=True)
        return 2

    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {SCRIPT_NAME}", flush=True)
    print(f"[source_of_truth] {SOURCE_OF_TRUTH}", flush=True)
    print(f"[args] symbol={args.symbol} start={args.start_date} end={args.end_date} warmup={args.warmup_start_date}", flush=True)
    print(f"[args] out_dir={out_dir}", flush=True)
    print("[scope] refactored src modules only; legacy file is read-only source of truth/parity source", flush=True)

    lf_trades, _lf_equity, _lf_features = run_lf_v10b_leg(args)
    print(f"[lf] trades={len(lf_trades):,}", flush=True)
    mf_trades, _mf_events, _mf_summary = run_low_sweep_time48_leg(args)
    print(f"[mf] trades={len(mf_trades):,}", flush=True)

    scenarios = build_scenarios(args)
    all_trades: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        combined, summary = simulate_portfolio_scenario(
            lf_trades,
            mf_trades,
            scenario,
            initial_capital=float(args.initial_capital),
            leverage=float(args.leverage),
        )
        all_trades.append(combined)
        summary_rows.append(summary)

    combined_all = pd.concat(all_trades, ignore_index=True, sort=False) if all_trades else pd.DataFrame()
    raw_summary = pd.DataFrame(summary_rows)
    summary = build_standard_summary(raw_summary, combined_all)
    primary = select_primary_trades(combined_all, args.primary_scenario)
    if primary.empty:
        print(f"[warn] primary scenario has no trades: {args.primary_scenario}", flush=True)

    trades = standardize_trades(primary)
    equity = build_equity_curve(primary, float(args.initial_capital))
    edge_attr = edge_attribution(trades)
    daily = daily_returns(equity, float(args.initial_capital))
    stress = stress_report(lf_trades, mf_trades, args)

    manifest = build_manifest(
        args,
        source_of_truth=str(SOURCE_OF_TRUTH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        artifacts=[],
        parity_old_report_dir=args.parity_old_report_dir,
    )
    write_standard_artifacts(
        out_dir,
        manifest=manifest,
        summary=summary,
        trades=trades,
        equity=equity,
        edge_attribution=edge_attr,
        daily_returns=daily,
        stress=stress,
    )

    parity_failed = False
    if args.parity_old_report_dir:
        print(f"[parity] compare old_report_dir={args.parity_old_report_dir}", flush=True)
        result = run_parity(
            new_report_dir=out_dir,
            old_report_dir=args.parity_old_report_dir,
            primary_scenario=args.primary_scenario,
        )
        parity_failed = not result.passed
        if result.passed:
            print("[parity] PASS", flush=True)
        elif result.first_diff is not None:
            _print_parity_failure(result.first_diff)

    manifest = dict(manifest)
    manifest["artifacts"] = _current_artifact_names(out_dir)
    write_json(manifest, out_dir / "00_manifest.json", "manifest")
    finalize_review_pack(out_dir)
    manifest["artifacts"] = _current_artifact_names(out_dir)
    write_json(manifest, out_dir / "00_manifest.json", "manifest")

    print("[done] ETH Portfolio V1 refactored backtest complete", flush=True)
    return 1 if parity_failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

