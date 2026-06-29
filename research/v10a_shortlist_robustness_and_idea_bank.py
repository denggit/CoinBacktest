#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V10A Shortlist Robustness + Idea Bank
=====================================

Research-only helper after v10a_structural_stop_grid_research.py.

Goals:
  1) Do not promote any structural stop directly to production.
  2) First validate shortlisted ideas at signal/trade level, then validate with
     backtest-level robustness.
  3) Avoid future-function / overfit traps by checking neighbourhoods,
     yearly stability, FORCE_CLOSE_END, top-winner dependency, fee/slippage stress.
  4) Maintain a broad idea bank beyond range bars / footprint / structural stops.

This script does not modify the official V10A strategy. It can either:
  - Read existing structural grid outputs and build shortlist + idea-bank reports.
  - Optionally rerun only the shortlisted candidates and neighbours for deeper
    trade-level diagnostics and stress tests.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research import v10a_integrated_signal_research_suite as suite  # noqa: E402
from research import v10a_structural_stop_grid_research as structural  # noqa: E402
from backtest.lf import eth_lf_portfolio_v10a_momentum_micro_short_speed_filter_backtest as v10a  # noqa: E402

BASELINE = structural.BASELINE
OUT_NAME = "v10a_shortlist_robustness_and_idea_bank"

SHORTLIST = [
    "struct_stop_bear_current_bar_n8_buf0p1_trig0p5_h1",
    "struct_stop_bear_current_bar_n8_buf0p0_trig0p5_h1",
    "struct_stop_bear_current_bar_n8_buf0p25_trig0p5_h1",
    "struct_giveback_all_swing_n5_trig1p5_gb0p5",
    "struct_stop_all_swing_n21_buf0p0_trig0p0_h0",
    "struct_fail_bull_hybrid_tighter_n8_buf0p1_h3",
    "initial_struct_bear_swing_n13_buf0p5",
]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Build V10A shortlist robustness reports and broad idea bank. Unknown args are forwarded to structural research loader."
    )
    p.add_argument("--structural-out-dir", default="data/reports/research/v10a_structural_stop_grid_research")
    p.add_argument("--out-dir", default=f"data/reports/research/{OUT_NAME}")
    p.add_argument("--from-existing", action="store_true", help="Only read existing structural grid outputs. This is fast and default if --rerun-shortlist is not set.")
    p.add_argument("--rerun-shortlist", action="store_true", help="Rerun shortlist candidates and neighbourhood specs for trade-level diagnostics.")
    p.add_argument("--stress", action="store_true", help="With --rerun-shortlist, run fee/slippage stress on core shortlist.")
    p.add_argument("--include-neighbourhood", action="store_true", help="With --rerun-shortlist, include parameter-neighbour specs around shortlist candidates.")
    p.add_argument("--write-trades", action="store_true", help="Write trades CSVs for rerun shortlist scenarios.")
    p.add_argument("--max-rerun-scenarios", type=int, default=None)
    p.add_argument("--min-return-ratio", type=float, default=0.85)
    p.add_argument("--min-win-rate-delta", type=float, default=5.0)
    p.add_argument("--max-dd-delta", type=float, default=3.0)
    p.add_argument("--max-pf-drop", type=float, default=1.0)
    args, unknown = p.parse_known_args()
    if not args.rerun_shortlist:
        args.from_existing = True
    return args, unknown


def _project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(PROJECT_ROOT) / p


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "UNKNOWN"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def load_existing_grid(structural_out_dir: Path) -> dict[str, pd.DataFrame]:
    summary = _read_csv(structural_out_dir / "02_structural_stop_grid_summary.csv")
    compare = _read_csv(structural_out_dir / "03_compare_to_v10a.csv")
    audit = _read_csv(structural_out_dir / "05_structural_stop_audit.csv")
    scoreboard = _read_csv(structural_out_dir / "06_candidate_scoreboard.csv")
    yearly = _read_csv(structural_out_dir / "07_variant_yearly.csv")
    top = _read_csv(structural_out_dir / "08_top_trade_dependency.csv")
    if compare.empty and not summary.empty:
        compare = structural.compare_to_baseline(summary)
    if scoreboard.empty and not compare.empty:
        scoreboard = structural.build_structural_scoreboard(compare, structural._normalize_yearly_columns(yearly), top)
    return {
        "summary": summary,
        "compare": compare,
        "audit": audit,
        "scoreboard": scoreboard,
        "yearly": structural._normalize_yearly_columns(yearly),
        "top": top,
    }


def classify_scenario(name: str) -> str:
    s = str(name)
    if s == BASELINE:
        return "baseline"
    if s.startswith("struct_stop_bear_current_bar"):
        return "bear_current_bar_struct_stop"
    if s.startswith("struct_stop_all_swing"):
        return "all_engine_swing_struct_stop"
    if s.startswith("struct_giveback_all_swing"):
        return "all_engine_swing_giveback"
    if s.startswith("struct_fail_bull_hybrid_tighter"):
        return "bull_hybrid_failure_exit"
    if s.startswith("initial_struct_bear_swing"):
        return "bear_initial_struct_stop"
    if s.startswith("initial_struct"):
        return "initial_struct_stop_other"
    if s.startswith("struct_fail"):
        return "close_confirm_failure_exit_other"
    if s.startswith("struct_giveback"):
        return "giveback_struct_stop_other"
    if s.startswith("struct_stop"):
        return "struct_stop_other"
    return "other"


def build_existing_shortlist_reports(dfs: dict[str, pd.DataFrame], out_dir: Path, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    compare = dfs["compare"].copy()
    audit = dfs["audit"].copy()
    yearly = dfs["yearly"].copy()
    top = dfs["top"].copy()
    scoreboard = dfs["scoreboard"].copy()

    if compare.empty:
        raise RuntimeError("No compare/summary CSV found. Please run v10a_structural_stop_grid_research.py first.")

    compare["scenario_family"] = compare["scenario"].map(classify_scenario)
    compare["is_core_shortlist"] = compare["scenario"].isin(SHORTLIST)

    # Broad family-level view: which research direction actually moves win-rate without killing return.
    fam_rows: list[dict[str, Any]] = []
    for fam, g in compare.loc[~compare["scenario"].eq(BASELINE)].groupby("scenario_family", dropna=False):
        rr = _num(g, "return_ratio_vs_baseline")
        wd = _num(g, "win_rate_delta")
        dd = _num(g, "max_drawdown_pct_delta")
        pf = _num(g, "profit_factor_delta")
        passed = g.loc[
            rr.ge(float(args.min_return_ratio))
            & wd.ge(float(args.min_win_rate_delta))
            & dd.le(float(args.max_dd_delta))
            & pf.ge(-float(args.max_pf_drop))
        ]
        best_win_keep = g.loc[rr.ge(float(args.min_return_ratio))].sort_values(["win_rate_delta", "return_ratio_vs_baseline"], ascending=[False, False]).head(1)
        best_return = g.sort_values(["total_return_pct", "win_rate_delta"], ascending=[False, False]).head(1)
        fam_rows.append({
            "scenario_family": fam,
            "scenario_count": int(len(g)),
            "pass_count": int(len(passed)),
            "best_win_keep_scenario": "" if best_win_keep.empty else str(best_win_keep.iloc[0]["scenario"]),
            "best_win_keep_win_rate_delta": np.nan if best_win_keep.empty else float(best_win_keep.iloc[0].get("win_rate_delta", np.nan)),
            "best_win_keep_return_ratio": np.nan if best_win_keep.empty else float(best_win_keep.iloc[0].get("return_ratio_vs_baseline", np.nan)),
            "best_win_keep_dd_delta": np.nan if best_win_keep.empty else float(best_win_keep.iloc[0].get("max_drawdown_pct_delta", np.nan)),
            "best_return_scenario": "" if best_return.empty else str(best_return.iloc[0]["scenario"]),
            "best_return_pct": np.nan if best_return.empty else float(best_return.iloc[0].get("total_return_pct", np.nan)),
            "best_return_win_rate_delta": np.nan if best_return.empty else float(best_return.iloc[0].get("win_rate_delta", np.nan)),
        })
    family_df = pd.DataFrame(fam_rows).sort_values(["pass_count", "best_win_keep_win_rate_delta"], ascending=[False, False])

    shortlist = compare.loc[compare["is_core_shortlist"]].copy()
    if not audit.empty:
        shortlist = shortlist.merge(audit, on="scenario", how="left", suffixes=("", "_audit"))
    if not top.empty:
        shortlist = shortlist.merge(top, on="scenario", how="left", suffixes=("", "_top"))
    shortlist["shortlist_decision_precheck"] = np.select(
        [
            _num(shortlist, "return_ratio_vs_baseline").ge(0.90) & _num(shortlist, "win_rate_delta").ge(5.0) & _num(shortlist, "max_drawdown_pct_delta").le(3.0),
            _num(shortlist, "return_ratio_vs_baseline").ge(1.10) & _num(shortlist, "max_drawdown_pct_delta").le(3.0),
        ],
        ["RETEST_WIN_RATE_CANDIDATE", "RETEST_RETURN_CANDIDATE"],
        default="WATCH_ONLY",
    )

    # Yearly stability for shortlist: not a model fit, only a robustness warning.
    yearly_short = yearly.loc[yearly.get("scenario", pd.Series(index=yearly.index)).isin([BASELINE] + SHORTLIST)].copy() if not yearly.empty else pd.DataFrame()
    if not yearly_short.empty:
        yearly_short["yearly_return_pct"] = _num(yearly_short, "yearly_return_pct")
        yearly_short["yearly_win_rate"] = _num(yearly_short, "yearly_win_rate")
        yagg = yearly_short.groupby("scenario").agg(
            yearly_return_min=("yearly_return_pct", "min"),
            yearly_return_median=("yearly_return_pct", "median"),
            yearly_win_rate_median=("yearly_win_rate", "median"),
            positive_years=("yearly_return_pct", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
            year_count=("year", "nunique") if "year" in yearly_short.columns else ("yearly_return_pct", "count"),
        ).reset_index()
    else:
        yagg = pd.DataFrame()

    # Signal-first validation from grid outputs: structural audit + engine exit breakdown if available.
    signal_precheck = shortlist.copy()
    if not yagg.empty:
        signal_precheck = signal_precheck.merge(yagg, on="scenario", how="left")
    signal_precheck["signal_validation_question"] = signal_precheck["scenario"].map(_signal_question_for_scenario)
    signal_precheck["overfit_warning"] = signal_precheck.apply(_overfit_warning, axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(out_dir / "01_shortlist_existing_grid_precheck.csv", index=False)
    family_df.to_csv(out_dir / "02_direction_family_screen.csv", index=False)
    signal_precheck.to_csv(out_dir / "03_signal_first_precheck.csv", index=False)
    if not yearly_short.empty:
        yearly_short.to_csv(out_dir / "04_shortlist_yearly_existing.csv", index=False)
    if not scoreboard.empty:
        scoreboard.to_csv(out_dir / "05_full_grid_scoreboard_copy.csv", index=False)
    return {
        "shortlist": shortlist,
        "family": family_df,
        "signal_precheck": signal_precheck,
        "yearly_short": yearly_short,
    }


def _signal_question_for_scenario(scenario: str) -> str:
    if scenario.startswith("struct_stop_bear_current_bar"):
        return "For BEAR shorts, after >=0.5R MFE, does current-bar structure stop reduce false-winner-to-loser trades without clipping top tail?"
    if scenario.startswith("struct_stop_all_swing"):
        return "Does a broad swing structural stop improve tail retention or is it mostly optimizing a few large winners?"
    if scenario.startswith("struct_giveback_all_swing"):
        return "After large MFE, does swing giveback control reduce winner giveback while preserving convexity?"
    if scenario.startswith("struct_fail_bull_hybrid_tighter"):
        return "For BULL reclaim, does close-confirm hybrid failure identify failed reclaims before full stop?"
    if scenario.startswith("initial_struct_bear_swing"):
        return "Does initial BEAR structural stop improve R definition, or does it just change sizing/path leverage?"
    return "General structural-stop signal validation."


def _overfit_warning(row: pd.Series) -> str:
    warnings: list[str] = []
    rr = float(row.get("return_ratio_vs_baseline", np.nan)) if pd.notna(row.get("return_ratio_vs_baseline", np.nan)) else np.nan
    win_delta = float(row.get("win_rate_delta", np.nan)) if pd.notna(row.get("win_rate_delta", np.nan)) else np.nan
    top3 = float(row.get("top_3_trade_dependency_pct", np.nan)) if pd.notna(row.get("top_3_trade_dependency_pct", np.nan)) else np.nan
    pf_delta = float(row.get("profit_factor_delta", np.nan)) if pd.notna(row.get("profit_factor_delta", np.nan)) else np.nan
    if np.isfinite(rr) and rr > 1.5:
        warnings.append("return_jump_large_check_top_winner_dependency")
    if np.isfinite(win_delta) and win_delta < 3:
        warnings.append("win_rate_not_material")
    if np.isfinite(top3) and top3 > 60:
        warnings.append("top3_dependency_high")
    if np.isfinite(pf_delta) and pf_delta < -1.0:
        warnings.append("pf_drop_large")
    return ";".join(warnings) if warnings else "OK_FOR_RETEST"


def build_idea_bank(out_dir: Path) -> pd.DataFrame:
    rows = [
        # Exit / structure beyond current grid
        ("adaptive_profit_lock", "exit", "When MFE>=1R/1.5R, move stop to structural level or breakeven only if structure confirms failure; not fixed partial TP.", "existing 4H OHLCV + trades/range optional", "medium", "medium", "Potentially lifts win rate without cutting size; must audit tail clipping."),
        ("engine_specific_failure_exit", "exit", "Separate failure-exit logic for BULL reclaim, BEAR breakdown, MOM long/short instead of one global rule.", "existing 4H OHLCV + signal audit", "medium", "low", "Likely more robust than global stop grid; validate per-engine signal labels first."),
        ("delayed_confirmation_entry", "entry_timing", "Do not enter immediately on weak reclaim/breakdown; wait 1 bar confirm or retest for selected sub-signals.", "existing 4H OHLCV", "medium", "low", "Can improve win rate but may miss best trend entries; use signal-event validation before backtest."),
        ("entry_quality_risk_scaling", "risk", "Keep all signals but scale risk by past-only quality score: trend strength, close location, wick, volume, footprint.", "existing OHLCV + footprint", "high", "low", "Avoid hard filters; requires neighbourhood and walk-forward validation."),
        ("volatility_state_router", "router", "Different rules in compression / expansion / panic volatility regimes.", "OHLCV; optional range speed", "medium", "low", "Use past rolling ATR/range percentiles only; no full-sample quantiles."),
        ("funding_oi_crowding_filter", "external_data", "Use funding rate + open interest to identify crowded longs/shorts and avoid bad continuation signals.", "funding + open interest", "medium", "medium", "Strong candidate for LF 4H strategy; needs new local data loader."),
        ("basis_mark_index_filter", "external_data", "Use mark-index basis / premium to detect crowded perp pressure or liquidation risk.", "mark price + index price", "medium", "medium", "Useful for perp-specific entries/exits; easier than orderbook."),
        ("liquidation_cluster_proxy", "external_data", "Detect forced-move conditions using trades bursts, range speed, wick absorption, maybe external liquidation feed later.", "trades/range footprint; optional liquidation data", "high", "medium", "Could improve crash/reclaim decisions; watch data availability bias."),
        ("btc_eth_context_filter", "cross_asset", "ETH signal quality conditioned on BTC trend/regime and ETH/BTC relative strength.", "BTC OHLCV + ETH/BTC", "medium", "low", "Likely useful for avoiding ETH longs when BTC macro trend weak."),
        ("session_time_filter", "market_microstructure", "Check if certain UTC sessions / weekend periods degrade signals.", "timestamp only", "high", "low", "High overfit risk; only use broad robust bins if stable by year."),
        ("range_bar_acceptance_followthrough", "microstructure", "After signal, require early range-bar follow-through/acceptance before keeping full risk.", "trades-derived range bars", "medium", "medium", "More live complexity but very aligned with current V10 direction."),
        ("drawdown_state_risk_throttle", "portfolio", "Reduce risk after strategy-level drawdown or after consecutive failed signals; restore after recovery.", "portfolio equity only", "medium", "low", "May improve realized drawdown/win experience but can cut recovery trades."),
        ("engine_correlation_allocator", "portfolio", "Keep single net position but allocate risk by recent engine hit-rate / correlation / market regime.", "signal audit + trades", "high", "medium", "Better than independent books; must avoid adapting to too little sample."),
        ("orderflow_absorption_reclaim", "microstructure", "For BULL reclaim, distinguish real absorption from low-volume bounce using taker ratio + close position + wick.", "footprint/range footprint", "medium", "medium", "Targets BULL specifically; test signal labels first."),
        ("bear_breakdown_exhaustion_guard", "microstructure", "For BEAR shorts, avoid shorting when recent downside move already extended and footprint shows absorption.", "OHLCV + footprint", "medium", "low", "A more general version of weak-footprint but must beat overfit test."),
        ("regime_ensemble_voting", "router", "Require agreement between base engine and independent regime classifier trained only on past labels or hand-built robust bins.", "OHLCV + signal labels", "high", "medium", "Could overfit; if ML is used, strict walk-forward only."),
    ]
    df = pd.DataFrame(rows, columns=["idea", "category", "description", "data_required", "overfit_risk", "live_complexity", "why_it_might_help"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "09_broad_idea_bank.csv", index=False)
    return df


def _make_structural_args(forwarded: list[str], out_dir: Path) -> argparse.Namespace:
    old = sys.argv[:]
    try:
        # Structural parser has all data/backtest options. Unknown args from this script are forwarded.
        sys.argv = ["v10a_structural_stop_grid_research.py", "--out-dir", str(out_dir)] + list(forwarded)
        return structural.parse_args()
    finally:
        sys.argv = old


def _select_specs(include_neighbourhood: bool, max_scenarios: int | None = None) -> list[Any]:
    all_specs = {s.name: s for s in structural.build_structural_specs(fast=False)}
    selected: dict[str, Any] = {BASELINE: all_specs[BASELINE]}
    for name in SHORTLIST:
        if name in all_specs:
            selected[name] = all_specs[name]

    if include_neighbourhood:
        for name, spec in all_specs.items():
            if name == BASELINE:
                continue
            keep = False
            if name.startswith("struct_stop_bear_current_bar_"):
                keep = any(k in name for k in ["_n5_", "_n8_", "_n13_"]) and any(k in name for k in ["buf0p0", "buf0p1", "buf0p25"]) and any(k in name for k in ["trig0p5", "trig1p0"]) and any(name.endswith(k) for k in ["h1", "h2"])
            elif name.startswith("struct_stop_all_swing_"):
                keep = any(k in name for k in ["_n13_", "_n21_", "_n34_"]) and any(k in name for k in ["buf0p0", "buf0p25", "buf0p5"]) and any(k in name for k in ["trig0p0", "trig0p5"])
            elif name.startswith("struct_giveback_all_swing_"):
                keep = any(k in name for k in ["_n5_", "_n8_", "_n13_"])
            elif name.startswith("struct_fail_bull_hybrid_tighter_"):
                keep = any(k in name for k in ["_n5_", "_n8_", "_n13_"]) and any(k in name for k in ["buf0p0", "buf0p1", "buf0p25"]) and any(name.endswith(k) for k in ["h2", "h3", "h4"])
            elif name.startswith("initial_struct_bear_swing_"):
                keep = any(k in name for k in ["_n8_", "_n13_", "_n21_"]) and any(k in name for k in ["buf0p1", "buf0p25", "buf0p5"])
            if keep:
                selected[name] = spec
    specs = list(selected.values())
    if max_scenarios is not None:
        # Always preserve baseline + core shortlist first.
        specs = specs[: max(1, int(max_scenarios))]
    return specs


def _prepare_base_data(sargs: argparse.Namespace) -> dict[str, Any]:
    data = suite.load_inputs(sargs)
    flags = suite.build_flags(data["raw"], data["micro_ctx"], sargs)
    features = suite.make_features(
        data["raw"],
        data["micro_ctx"],
        sargs,
        flags,
        scenario=BASELINE,
        mom_long_block_mask=flags["v10_mom_long_not_aligned"],
        mom_short_block_mask=flags["v10a_mom_short_fast_speed"],
    )
    features = suite.slice_trade_window(features, sargs)
    features = structural.add_structural_columns(features)
    data["features"] = features
    return data


def _run_spec(data: dict[str, Any], sargs: argparse.Namespace, spec: Any) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    if spec.name == BASELINE:
        return v10a.run_priority_backtest(
            data["features"],
            data["exec_cfg"],
            engine_cfgs=data["engine_cfgs"],
            global_risk_scale=sargs.global_risk_scale,
            args=sargs,
        )
    return structural.run_structural_backtest(
        data["features"],
        data["exec_cfg"],
        data["engine_cfgs"],
        global_risk_scale=sargs.global_risk_scale,
        args=sargs,
        spec=spec,
    )


def _engine_signal_diagnostics(scenario: str, trades: list[dict[str, Any]]) -> pd.DataFrame:
    tdf = pd.DataFrame(trades)
    rows: list[dict[str, Any]] = []
    if tdf.empty:
        return pd.DataFrame()
    tdf = tdf.copy()
    tdf["engine"] = tdf.get("engine", pd.Series("UNKNOWN", index=tdf.index)).astype(str)
    tdf["side"] = tdf.get("type", pd.Series("UNKNOWN", index=tdf.index)).astype(str)
    tdf["note"] = tdf.get("note", pd.Series("UNKNOWN", index=tdf.index)).astype(str)
    ret = pd.to_numeric(tdf.get("return_pct", pd.Series(0, index=tdf.index)), errors="coerce").fillna(0.0) * 100.0
    mfe = pd.to_numeric(tdf.get("mfe_r", pd.Series(0, index=tdf.index)), errors="coerce").fillna(0.0)
    mae = pd.to_numeric(tdf.get("mae_r", pd.Series(0, index=tdf.index)), errors="coerce").fillna(0.0)
    tdf["_ret_pct"] = ret
    tdf["_mfe_r"] = mfe
    tdf["_mae_r"] = mae
    for (engine, side), g in tdf.groupby(["engine", "side"], dropna=False):
        r = pd.to_numeric(g["_ret_pct"], errors="coerce").fillna(0.0)
        wins = r > 0
        pos = r[r > 0].sum()
        neg = -r[r < 0].sum()
        rows.append({
            "scenario": scenario,
            "engine": engine,
            "side": side,
            "trades": int(len(g)),
            "win_rate": float(wins.mean() * 100.0) if len(g) else 0.0,
            "sum_return_pct": float(r.sum()),
            "avg_return_pct": float(r.mean()) if len(g) else 0.0,
            "median_return_pct": float(r.median()) if len(g) else 0.0,
            "profit_factor_pct": float(pos / neg) if neg > 1e-12 else float("inf") if pos > 0 else 0.0,
            "avg_mfe_r": float(pd.to_numeric(g["_mfe_r"], errors="coerce").fillna(0.0).mean()),
            "avg_mae_r": float(pd.to_numeric(g["_mae_r"], errors="coerce").fillna(0.0).mean()),
            "mfe_ge_1r_ended_loss": int(((g["_mfe_r"] >= 1.0) & (~wins)).sum()),
            "structural_stop_exits": int(g["note"].eq("STRUCTURE_STOP").sum()),
            "structural_close_confirm_exits": int(g["note"].eq("STRUCTURE_CLOSE_CONFIRM_NEXT_OPEN").sum()),
        })
    return pd.DataFrame(rows)


def rerun_shortlist(args: argparse.Namespace, forwarded: list[str], out_dir: Path) -> dict[str, pd.DataFrame]:
    rerun_dir = out_dir / "rerun_shortlist"
    rerun_dir.mkdir(parents=True, exist_ok=True)
    sargs = _make_structural_args(forwarded, rerun_dir)
    specs = _select_specs(args.include_neighbourhood, args.max_rerun_scenarios)
    print(f"[rerun] scenarios={len(specs)} include_neighbourhood={args.include_neighbourhood}", flush=True)
    data = _prepare_base_data(sargs)

    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[pd.DataFrame] = []
    top_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    engine_diag_rows: list[pd.DataFrame] = []

    for i, spec in enumerate(specs, start=1):
        trades, equity = _run_spec(data, sargs, spec)
        trades = v10a.attach_engine_to_trades(trades, data["features"])
        extra = {
            "scenario_family": classify_scenario(spec.name),
            "rule_note": json.dumps(asdict(spec), ensure_ascii=False),
        }
        summary_rows.append(suite.summary_metrics(spec.name, trades, equity, data["exec_cfg"].initial_capital, extra=extra))
        yearly_rows.append(suite.yearly_metrics(spec.name, trades, equity))
        top_rows.append(suite.top_trade_dependency(spec.name, trades))
        tdf = pd.DataFrame(trades)
        if not tdf.empty:
            note = tdf.get("note", pd.Series("", index=tdf.index)).astype(str)
            audit_rows.append({
                "scenario": spec.name,
                "structural_stop_exits": int(note.eq("STRUCTURE_STOP").sum()),
                "structural_close_confirm_exits": int(note.eq("STRUCTURE_CLOSE_CONFIRM_NEXT_OPEN").sum()),
                "structure_update_trades": int(pd.to_numeric(tdf.get("structure_updates", pd.Series(0, index=tdf.index)), errors="coerce").fillna(0).gt(0).sum()),
            })
        else:
            audit_rows.append({"scenario": spec.name, "structural_stop_exits": 0, "structural_close_confirm_exits": 0, "structure_update_trades": 0})
        engine_diag_rows.append(_engine_signal_diagnostics(spec.name, trades))
        if args.write_trades:
            tdf.to_csv(rerun_dir / f"{spec.name}__trades.csv", index=False)
            if not equity.empty:
                equity.to_csv(rerun_dir / f"{spec.name}__equity.csv")
        print(f"  rerun {i}/{len(specs)} {spec.name}", flush=True)

    summary = pd.DataFrame(summary_rows)
    yearly = structural._normalize_yearly_columns(pd.concat([x for x in yearly_rows if x is not None and not x.empty], ignore_index=True) if yearly_rows else pd.DataFrame())
    top = pd.DataFrame(top_rows)
    audit = pd.DataFrame(audit_rows)
    compare = structural.compare_to_baseline(summary)
    scoreboard = structural.build_structural_scoreboard(compare, yearly, top)
    engine_diag = pd.concat([x for x in engine_diag_rows if x is not None and not x.empty], ignore_index=True) if engine_diag_rows else pd.DataFrame()

    summary.to_csv(out_dir / "10_rerun_shortlist_summary.csv", index=False)
    compare.to_csv(out_dir / "11_rerun_shortlist_compare.csv", index=False)
    scoreboard.to_csv(out_dir / "12_rerun_shortlist_scoreboard.csv", index=False)
    engine_diag.to_csv(out_dir / "13_rerun_signal_trade_diagnostics.csv", index=False)
    yearly.to_csv(out_dir / "14_rerun_yearly.csv", index=False)
    audit.to_csv(out_dir / "15_rerun_audit.csv", index=False)
    top.to_csv(out_dir / "16_rerun_top_dependency.csv", index=False)

    stress_df = pd.DataFrame()
    if args.stress:
        stress_df = run_stress_tests(args, forwarded, out_dir)

    return {
        "summary": summary,
        "compare": compare,
        "scoreboard": scoreboard,
        "engine_diag": engine_diag,
        "yearly": yearly,
        "audit": audit,
        "top": top,
        "stress": stress_df,
    }


def run_stress_tests(args: argparse.Namespace, forwarded: list[str], out_dir: Path) -> pd.DataFrame:
    stress_dir = out_dir / "stress_cache"
    stress_dir.mkdir(parents=True, exist_ok=True)
    specs_by_name = {s.name: s for s in structural.build_structural_specs(fast=False)}
    names = [BASELINE] + [x for x in SHORTLIST if x in specs_by_name]
    fees = [0.00055, 0.00075, 0.00100]
    slips = [0.0002, 0.0005, 0.0010]
    rows: list[dict[str, Any]] = []
    for fee in fees:
        for slip in slips:
            extra = [x for x in forwarded if x not in ["--fee-rate", "--slippage-pct"]]
            sargs = _make_structural_args(extra + ["--fee-rate", str(fee), "--slippage-pct", str(slip)], stress_dir)
            print(f"[stress] fee={fee} slip={slip}", flush=True)
            data = _prepare_base_data(sargs)
            for name in names:
                spec = specs_by_name[name]
                trades, equity = _run_spec(data, sargs, spec)
                trades = v10a.attach_engine_to_trades(trades, data["features"])
                sm = suite.summary_metrics(name, trades, equity, data["exec_cfg"].initial_capital)
                sm.update({"fee_rate": fee, "slippage_pct": slip})
                rows.append(sm)
    df = pd.DataFrame(rows)
    if not df.empty:
        base = df.loc[df["scenario"].eq(BASELINE), ["fee_rate", "slippage_pct", "total_return_pct", "win_rate", "max_drawdown_pct", "profit_factor"]].rename(columns={
            "total_return_pct": "baseline_total_return_pct",
            "win_rate": "baseline_win_rate",
            "max_drawdown_pct": "baseline_max_drawdown_pct",
            "profit_factor": "baseline_profit_factor",
        })
        df = df.merge(base, on=["fee_rate", "slippage_pct"], how="left")
        df["return_ratio_vs_baseline_stress"] = df["total_return_pct"] / df["baseline_total_return_pct"].replace(0, np.nan)
        df["win_rate_delta_stress"] = df["win_rate"] - df["baseline_win_rate"]
        df["dd_delta_stress"] = df["max_drawdown_pct"] - df["baseline_max_drawdown_pct"]
        df["pf_delta_stress"] = df["profit_factor"] - df["baseline_profit_factor"]
    df.to_csv(out_dir / "17_fee_slippage_stress.csv", index=False)
    return df


def write_meta(out_dir: Path, args: argparse.Namespace, forwarded: list[str]) -> None:
    meta = {
        "script": "research/v10a_shortlist_robustness_and_idea_bank.py",
        "generated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "args": vars(args),
        "forwarded_structural_args": forwarded,
        "shortlist": SHORTLIST,
        "no_lookahead_notes": [
            "This script does not create rules from future labels.",
            "Rerun mode imports the structural research executor, where entries remain next-open after completed 4H signals.",
            "Structural updates are based on completed bars and are treated as research-only until separate validation passes.",
            "Idea bank is hypothesis generation only; no idea is promoted without signal diagnostics, backtest, robustness, and live feasibility review.",
        ],
    }
    with (out_dir / "99_research_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)


def main() -> int:
    args, forwarded = parse_args()
    out_dir = _project_path(args.out_dir)
    structural_out_dir = _project_path(args.structural_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96, flush=True)
    print("V10A Shortlist Robustness + Idea Bank", flush=True)
    print("No official strategy modification. No promotion without robustness.", flush=True)
    print("=" * 96, flush=True)

    dfs = load_existing_grid(structural_out_dir)
    existing_reports = build_existing_shortlist_reports(dfs, out_dir, args)
    idea_df = build_idea_bank(out_dir)
    print(f"Existing-grid shortlist rows: {len(existing_reports['shortlist'])}", flush=True)
    print(f"Idea bank rows: {len(idea_df)}", flush=True)

    if args.rerun_shortlist:
        rerun_shortlist(args, forwarded, out_dir)

    write_meta(out_dir, args, forwarded)
    print(f"Outputs written to: {out_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
