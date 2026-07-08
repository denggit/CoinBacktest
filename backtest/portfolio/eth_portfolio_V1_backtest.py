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

from src.backtest_common.data import load_ohlcv_data  # noqa: E402
from src.edge_lib.lf_bear_short.config import PRESETS as BEAR_PRESETS  # noqa: E402
from src.edge_lib.lf_bull_range_reclaim.config import PRESETS as BULL_PRESETS  # noqa: E402
from src.edge_lib.lf_momentum_breakout.config import PRESETS as MOMENTUM_PRESETS  # noqa: E402
from src.edge_lib.mf_low_sweep.signals import run_low_sweep_time48_leg  # noqa: E402
from src.portfolio_common.allocator import (  # noqa: E402
    DEFAULT_LEVERAGE,
    LF_LEG,
    MF_TIME48_LEG,
    build_equity_curve,
    build_guard_summary,
    build_margin_overlap_stress,
    build_mf_by_lf_state_report,
    build_overlap_report,
    build_report_trades,
    build_scenarios,
    daily_returns,
    edge_attribution,
    simulate_portfolio_scenario,
    standardize_trades,
    stress_report,
    summarize_period,
)
from src.portfolio_common.artifacts import (  # noqa: E402
    finalize_review_pack,
    write_all_scenario_trades,
    write_csv,
    write_diagnostic_artifacts,
    write_json,
    write_standard_artifacts,
)
from src.portfolio_common.parity import run_parity  # noqa: E402
from src.portfolio_common.reports import (  # noqa: E402
    build_manifest,
    build_primary_summary,
    build_standard_summary,
    filter_yearly_monthly,
    select_primary_trades,
)
from src.sleeve_lib.lf_v10b.selector import run_lf_v10b_leg  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402

SCRIPT_NAME = "eth_portfolio_V1_backtest"
SOURCE_OF_TRUTH = PROJECT_ROOT / "backtest/portfolio/eth_portfolio_V1_lf_v10b_low_sweep_mf_backtest.py"
DEFAULT_OUT_DIR = "data/reports/backtest/portfolio/eth_portfolio_V1"
PRIMARY_SCENARIO = "portfolio_v1_lf100_mf150_time48_independent"
REPORT_STRATEGY_NAME = "ETH_Portfolio_V1_LF_V10B_LowSweepMF"


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
    p.add_argument("--write-all-scenario-trades", action="store_true")

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
    """Discover all output files in the report directory for manifest."""
    wanted_prefixes = (
        "00_", "01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_", "09_",
        "10_", "11_", "12_", "13_", "14_",
        "80_", "90_", "91_",
    )
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


def _build_sizing_report(lf: pd.DataFrame, mf: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Summarize LF risk-budget sizing and MF fixed exposure assumptions."""
    leverage = float(getattr(args, "leverage", DEFAULT_LEVERAGE) or DEFAULT_LEVERAGE)
    exposures = [float(x) for x in str(args.mf_exposures).split(",") if x.strip()]
    rows: list[dict[str, object]] = []

    # --- LF config-level sizing ---
    try:
        mom = MOMENTUM_PRESETS[str(args.lf_preset)]
    except KeyError:
        mom = {"unit_risk_per_trade": 0.006, "max_total_notional_mult": 5.0, "max_units": 3, "max_risk_mult": 1.5}
    try:
        bear = BEAR_PRESETS[str(args.lf_bear_preset)]
    except KeyError:
        bear = {"unit_risk_per_trade": 0.006, "max_total_notional_mult": 5.0, "max_units": 3, "max_risk_mult": 1.5}
    try:
        bull = BULL_PRESETS[str(args.lf_bull_preset)]
    except KeyError:
        bull = {"unit_risk_per_trade": 0.006, "max_total_notional_mult": 5.0, "max_units": 3, "max_risk_mult": 1.5}

    for engine, preset, min_mult in [
        ("MOMENTUM_V3", mom, 0.35),
        ("BEAR_V3_ONLY", bear, 0.25),
        ("BULL_RECLAIM_V2", bull, 0.35),
    ]:
        rows.append(
            {
                "section": "lf_config_cap",
                "leg": LF_LEG,
                "engine": engine,
                "unit_risk_per_trade": float(preset["unit_risk_per_trade"]),
                "max_total_notional_mult": float(preset["max_total_notional_mult"]),
                "max_units": int(preset["max_units"]),
                "min_risk_mult": float(min_mult),
                "max_risk_mult": float(preset["max_risk_mult"]),
                "global_risk_scale": float(args.lf_global_risk_scale),
                "leverage_for_margin_report": leverage,
            }
        )

    # --- MF assumed fixed exposure ---
    for exp in exposures:
        rows.append(
            {
                "section": "mf_assumed_fixed_exposure",
                "leg": "MF_LOW_SWEEP",
                "engine": "A0_FOOTPRINT",
                "notional_mult_fixed": float(exp),
                "margin_fraction_at_leverage": float(exp) / leverage,
                "leverage_for_margin_report": leverage,
                "close_scope": "mf_low_sweep_only",
            }
        )
    return pd.DataFrame(rows)


def _print_full_report_for_primary(
    primary: pd.DataFrame,
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    """Print human-readable full report for the primary scenario.

    Exports 90_full_report_trades.csv and 90_full_report.txt.
    """
    if bool(args.skip_full_report) or primary.empty:
        return

    # Load 4H OHLCV for report calendar.
    df = load_ohlcv_data(args.symbol, args.start_date, args.end_date, "4H")
    if len(df):
        primary_end = pd.to_datetime(primary["exit_time"], errors="coerce").max()
        primary_start = pd.to_datetime(primary["entry_time"], errors="coerce").min()
        if pd.notna(primary_end) and primary_end > df.index[-1]:
            last = df.iloc[-1:].copy()
            last.index = pd.DatetimeIndex([primary_end])
            df = pd.concat([df, last]).sort_index()
        if pd.notna(primary_start) and primary_start < df.index[0]:
            first = df.iloc[:1].copy()
            first.index = pd.DatetimeIndex([primary_start])
            df = pd.concat([first, df]).sort_index()

    total_days = max((df.index[-1] - df.index[0]).total_seconds() / 86400.0, 1e-9) if len(df) else 1e-9
    report_trades, final_capital = build_report_trades(primary, float(args.initial_capital))

    # Export full report trades as 90_full_report_trades.csv (with display_exit_reason).
    if report_trades:
        trades_df = pd.DataFrame(report_trades)
        write_csv(trades_df, out_dir / "90_full_report_trades.csv", "full_report_trades")

    # Record files present before print_full_report so we can identify its txt output.
    before_txt = set(out_dir.glob("*.txt"))

    print(f"[report] print_full_report primary={args.primary_scenario}", flush=True)
    print_full_report(
        trade_history=report_trades,
        df=df,
        initial_capital=float(args.initial_capital),
        capital=float(final_capital),
        strategy_name=REPORT_STRATEGY_NAME,
        total_days=total_days,
        ai_enabled=False,
        symbol=args.symbol,
        report_dir=out_dir,
    )

    # Rename the txt file that print_full_report just created → 90_full_report.txt.
    after_txt = set(out_dir.glob("*.txt")) - before_txt
    for txt_path in after_txt:
        target = out_dir / "90_full_report.txt"
        if txt_path != target:
            txt_path.replace(target)
            print(f"[write] full_report_txt -> {target}", flush=True)
            break


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

    # --- Run child strategies ---
    lf_trades, _lf_equity, _lf_features = run_lf_v10b_leg(args)
    print(f"[lf] trades={len(lf_trades):,}", flush=True)
    mf_trades, _mf_events, _mf_summary = run_low_sweep_time48_leg(args)
    print(f"[mf] trades={len(mf_trades):,}", flush=True)

    # --- Build and simulate portfolio scenarios ---
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

    # --- Standard report shaping ---
    full_summary = build_standard_summary(raw_summary, combined_all)
    primary_summary = build_primary_summary(raw_summary, combined_all, args.primary_scenario)
    primary = select_primary_trades(combined_all, args.primary_scenario)
    if primary.empty:
        print(f"[warn] primary scenario has no trades: {args.primary_scenario}", flush=True)

    trades = standardize_trades(primary)
    equity = build_equity_curve(primary, float(args.initial_capital))
    edge_attr = edge_attribution(trades)
    daily = daily_returns(equity, float(args.initial_capital))
    stress = stress_report(lf_trades, mf_trades, args)

    # --- Diagnostic reports ---
    yearly_all = summarize_period(combined_all, "year")
    monthly_all = summarize_period(combined_all, "month")
    yearly_primary = filter_yearly_monthly(yearly_all, args.primary_scenario)
    monthly_primary = filter_yearly_monthly(monthly_all, args.primary_scenario)
    overlap = build_overlap_report(lf_trades, mf_trades)
    margin_stress = build_margin_overlap_stress(lf_trades, mf_trades, args)
    mf_by_lf_state = build_mf_by_lf_state_report(lf_trades, mf_trades, args)
    guard_summary = build_guard_summary(raw_summary)
    sizing = _build_sizing_report(lf_trades, mf_trades, args)

    # --- Manifest ---
    manifest = build_manifest(
        args,
        source_of_truth=str(SOURCE_OF_TRUTH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        artifacts=[],
        parity_old_report_dir=args.parity_old_report_dir,
    )

    # --- Write standard artifacts (01 = primary only) ---
    write_standard_artifacts(
        out_dir,
        manifest=manifest,
        summary=primary_summary,
        trades=trades,
        equity=equity,
        edge_attribution=edge_attr,
        daily_returns=daily,
        stress=stress,
    )

    # --- Write diagnostic artifacts ---
    write_diagnostic_artifacts(
        out_dir,
        scenario_summary=full_summary,
        yearly=yearly_primary,
        monthly=monthly_primary,
        overlap=overlap,
        sizing=sizing,
        margin_stress=margin_stress,
        mf_by_lf_state=mf_by_lf_state,
        guard_summary=guard_summary,
    )

    # --- Optionally write all-scenario yearly/monthly ---
    if not yearly_all.empty:
        write_csv(yearly_all, out_dir / "08_yearly_all_scenarios.csv", "yearly_all_scenarios")
    if not monthly_all.empty:
        write_csv(monthly_all, out_dir / "09_monthly_all_scenarios.csv", "monthly_all_scenarios")

    # --- Optionally write all-scenario trades ---
    if args.write_all_scenario_trades:
        write_all_scenario_trades(out_dir, combined_all)

    # --- Human-readable full report ---
    _print_full_report_for_primary(primary, args, out_dir)

    # --- Parity ---
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

    # --- Finalize manifest with all artifacts ---
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
