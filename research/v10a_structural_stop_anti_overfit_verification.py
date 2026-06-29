#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V10A Structural Stop Anti-Overfit Verification
==============================================

Research-only second-pass verification for V10A structural-stop candidates.

This script does not promote any rule to the official strategy. It is designed
for the step after the broad 2,225-scenario grid and shortlist precheck:
  1) Validate candidate behaviour at trade/signal level before trusting summary.
  2) Rerun only a small candidate + neighbourhood set.
  3) Stress test fee/slippage and top-winner dependency.
  4) Flag high path-change candidates such as initial structural stops.
  5) Produce an explicit decision matrix: promote-to-next-research, watch, reject.

No-lookahead assumptions inherited from v10a_structural_stop_grid_research.py:
  - V10A signals are based on completed 4H bars.
  - Entries still execute on the next 4H open.
  - Structural stop updates use completed bars and are applied as a tightened stop
    after the current closed bar; intrabar touches only use the already-active stop.
  - Close-confirm exits execute on next 4H open.

Important: this script can sanity-check timestamps and changed-trade paths, but it
cannot replace code review. Candidates with large path changes remain research-only
until the structural-level implementation is manually reviewed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest.lf import eth_lf_portfolio_v10a_momentum_micro_short_speed_filter_backtest as v10a  # noqa: E402
from research import v10a_integrated_signal_research_suite as suite  # noqa: E402
from research import v10a_structural_stop_grid_research as structural  # noqa: E402

BASELINE = structural.BASELINE
OUT_NAME = "v10a_structural_stop_anti_overfit_verification"

CORE_CANDIDATES = [
    "struct_stop_all_swing_n21_buf0p0_trig0p0_h0",
    "struct_stop_bear_current_bar_n8_buf0p1_trig0p5_h1",
    "initial_struct_bear_swing_n13_buf0p5",
]

WATCH_CANDIDATES = [
    "struct_stop_all_swing_n13_buf0p5_trig0p0_h0",
    "struct_giveback_all_swing_n5_trig1p5_gb0p5",
    "struct_fail_bull_hybrid_tighter_n8_buf0p1_h3",
]

# Manually-created neighbours include values not in the original 2,225 grid, such as n34.
# These are not for fitting a best parameter; they test whether the region is stable.
def build_candidate_specs(include_watch: bool = True, include_neighbourhood: bool = True) -> list[structural.StructuralStopSpec]:
    specs: list[structural.StructuralStopSpec] = [structural.StructuralStopSpec(name=BASELINE, enabled=False)]
    by_name = {s.name: s for s in structural.build_structural_specs(fast=False)}

    for name in CORE_CANDIDATES + (WATCH_CANDIDATES if include_watch else []):
        spec = by_name.get(name)
        if spec is not None:
            specs.append(spec)

    if include_neighbourhood:
        # 1) all-engine swing structural stop: stress n/buffer/trigger/hold neighbourhood.
        for n in [13, 21, 34]:
            for buf in [0.0, 0.10, 0.25, 0.50]:
                for trig in [0.0, 0.5, 1.0]:
                    hold = 1 if trig > 0 else 0
                    specs.append(structural.StructuralStopSpec(
                        name=f"verify_struct_stop_all_swing_n{n}_buf{_fmt(buf)}_trig{_fmt(trig)}_h{hold}",
                        source="swing",
                        action="stop",
                        engine_scope="ALL",
                        trigger_mfe_r=trig,
                        min_hold_bars=hold,
                        lookback=n,
                        buffer_atr=buf,
                    ))

        # 2) Bear current-bar win-rate candidate: check nearby buffers/triggers/hold.
        for n in [5, 8, 13]:
            for buf in [0.0, 0.10, 0.25, 0.50]:
                for trig in [0.5, 1.0, 1.5]:
                    for hold in [1, 2]:
                        specs.append(structural.StructuralStopSpec(
                            name=f"verify_struct_stop_bear_current_bar_n{n}_buf{_fmt(buf)}_trig{_fmt(trig)}_h{hold}",
                            source="current_bar",
                            action="stop",
                            engine_scope="BEAR",
                            trigger_mfe_r=trig,
                            min_hold_bars=hold,
                            lookback=n,
                            buffer_atr=buf,
                        ))

        # 3) Initial Bear swing: path-change candidate; test smaller neighbourhood only.
        for n in [8, 13, 21]:
            for buf in [0.10, 0.25, 0.50, 0.75]:
                specs.append(structural.StructuralStopSpec(
                    name=f"verify_initial_struct_bear_swing_n{n}_buf{_fmt(buf)}",
                    source="swing",
                    action="stop",
                    engine_scope="BEAR",
                    trigger_mfe_r=0.0,
                    min_hold_bars=0,
                    lookback=n,
                    buffer_atr=buf,
                    initial_struct_stop=True,
                ))

        # 4) Bull failure exit and giveback watchers, not first-line live candidates.
        for n in [5, 8, 13]:
            for buf in [0.0, 0.10, 0.25]:
                for hold in [2, 3, 4]:
                    specs.append(structural.StructuralStopSpec(
                        name=f"verify_struct_fail_bull_hybrid_tighter_n{n}_buf{_fmt(buf)}_h{hold}",
                        source="hybrid_tighter",
                        action="close_confirm",
                        engine_scope="BULL",
                        trigger_mfe_r=0.0,
                        min_hold_bars=hold,
                        lookback=n,
                        buffer_atr=buf,
                    ))
        for n in [5, 8, 13]:
            for trig in [1.0, 1.5, 2.0]:
                for gb in [0.35, 0.50, 0.65]:
                    specs.append(structural.StructuralStopSpec(
                        name=f"verify_struct_giveback_all_swing_n{n}_trig{_fmt(trig)}_gb{_fmt(gb)}",
                        source="swing",
                        action="stop",
                        engine_scope="ALL",
                        trigger_mfe_r=trig,
                        min_hold_bars=2,
                        lookback=n,
                        buffer_atr=0.25,
                        require_giveback_frac=gb,
                    ))

    return _dedupe_by_name(specs)


def _fmt(x: float) -> str:
    return str(x).replace(".", "p").replace("-", "m")


def _dedupe_by_name(specs: Iterable[structural.StructuralStopSpec]) -> list[structural.StructuralStopSpec]:
    seen: set[str] = set()
    out: list[structural.StructuralStopSpec] = []
    for s in specs:
        if s.name in seen:
            continue
        seen.add(s.name)
        out.append(s)
    return out


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Second-pass anti-overfit verification for V10A structural-stop candidates. Unknown args are forwarded to the V10A data loader."
    )
    p.add_argument("--out-dir", default=f"data/reports/research/{OUT_NAME}")
    p.add_argument("--from-existing", action="store_true", help="Only interpret existing shortlist/grid outputs; does not rerun backtests.")
    p.add_argument("--existing-shortlist-dir", default="data/reports/research/v10a_shortlist_robustness_and_idea_bank")
    p.add_argument("--existing-grid-dir", default="data/reports/research/v10a_structural_stop_grid_research")
    p.add_argument("--rerun-core", action="store_true", help="Rerun baseline + core candidates + optional watch/neighbour scenarios.")
    p.add_argument("--include-watch", action="store_true", help="Include watchlist scenarios besides the three core candidates.")
    p.add_argument("--include-neighbourhood", action="store_true", help="Rerun parameter neighbourhood around core/watch candidates.")
    p.add_argument("--stress", action="store_true", help="Run fee/slippage stress on baseline + core candidates.")
    p.add_argument("--write-trades", action="store_true", help="Write per-scenario trade/equity files for manual inspection.")
    p.add_argument("--max-scenarios", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--strict-return-ratio", type=float, default=1.00)
    p.add_argument("--min-live-return-ratio", type=float, default=0.90)
    p.add_argument("--min-win-rate-delta", type=float, default=5.0)
    p.add_argument("--max-dd-delta", type=float, default=3.0)
    p.add_argument("--max-pf-drop", type=float, default=1.0)
    args, forwarded = p.parse_known_args()
    if not args.from_existing and not args.rerun_core:
        args.from_existing = True
    return args, forwarded


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


def _make_structural_args(forwarded: list[str], out_dir: Path) -> argparse.Namespace:
    old = sys.argv[:]
    try:
        sys.argv = ["v10a_structural_stop_grid_research.py"] + list(forwarded) + ["--out-dir", str(out_dir)]
        return structural.parse_args()
    finally:
        sys.argv = old


def _prepare_base_data(args: argparse.Namespace) -> dict[str, Any]:
    data = suite.load_inputs(args)
    flags = suite.build_flags(data["raw"], data["micro_ctx"], args)
    features = suite.make_features(
        data["raw"],
        data["micro_ctx"],
        args,
        flags,
        scenario=BASELINE,
        mom_long_block_mask=flags["v10_mom_long_not_aligned"],
        mom_short_block_mask=flags["v10a_mom_short_fast_speed"],
    )
    features = suite.slice_trade_window(features, args)
    features = structural.add_structural_columns(features, max_lookback=34)
    data["features"] = features
    return data


def _run_spec(data: dict[str, Any], args: argparse.Namespace, spec: structural.StructuralStopSpec) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    if not spec.enabled or spec.name == BASELINE:
        return v10a.run_priority_backtest(
            data["features"],
            data["exec_cfg"],
            engine_cfgs=data["engine_cfgs"],
            global_risk_scale=args.global_risk_scale,
            args=args,
        )
    return structural.run_structural_backtest(
        data["features"],
        data["exec_cfg"],
        data["engine_cfgs"],
        global_risk_scale=args.global_risk_scale,
        args=args,
        spec=spec,
    )


def _trade_frame(trades: list[dict[str, Any]]) -> pd.DataFrame:
    tdf = pd.DataFrame(trades)
    if tdf.empty:
        return tdf
    tdf = tdf.copy()
    for c in ["entry_time", "exit_time"]:
        if c in tdf.columns:
            tdf[c] = pd.to_datetime(tdf[c], errors="coerce")
    tdf["engine"] = tdf.get("engine", pd.Series("UNKNOWN", index=tdf.index)).astype(str)
    tdf["side"] = tdf.get("type", pd.Series("UNKNOWN", index=tdf.index)).astype(str)
    tdf["entry_price_round"] = pd.to_numeric(tdf.get("entry_price", pd.Series(np.nan, index=tdf.index)), errors="coerce").round(2)
    tdf["_dup"] = tdf.groupby(["entry_time", "engine", "side", "entry_price_round"], dropna=False).cumcount()
    tdf["trade_key"] = (
        tdf["entry_time"].astype(str) + "|" + tdf["engine"] + "|" + tdf["side"] + "|" + tdf["entry_price_round"].astype(str) + "|" + tdf["_dup"].astype(str)
    )
    tdf["return_pct_trade"] = pd.to_numeric(tdf.get("return_pct", pd.Series(0, index=tdf.index)), errors="coerce").fillna(0.0) * 100.0
    tdf["mfe_r_trade"] = pd.to_numeric(tdf.get("mfe_r", pd.Series(np.nan, index=tdf.index)), errors="coerce")
    tdf["mae_r_trade"] = pd.to_numeric(tdf.get("mae_r", pd.Series(np.nan, index=tdf.index)), errors="coerce")
    tdf["note"] = tdf.get("note", pd.Series("", index=tdf.index)).astype(str)
    return tdf


def build_trade_diff(scenario: str, baseline_trades: list[dict[str, Any]], cand_trades: list[dict[str, Any]]) -> pd.DataFrame:
    b = _trade_frame(baseline_trades)
    c = _trade_frame(cand_trades)
    if b.empty and c.empty:
        return pd.DataFrame()
    cols_keep = [
        "trade_key", "entry_time", "exit_time", "engine", "side", "entry_price", "exit_price", "note",
        "return_pct_trade", "mfe_r_trade", "mae_r_trade", "units", "structure_updates",
    ]
    b2 = b[[x for x in cols_keep if x in b.columns]].add_prefix("baseline_") if not b.empty else pd.DataFrame()
    c2 = c[[x for x in cols_keep if x in c.columns]].add_prefix("candidate_") if not c.empty else pd.DataFrame()
    if not b2.empty:
        b2 = b2.rename(columns={"baseline_trade_key": "trade_key"})
    if not c2.empty:
        c2 = c2.rename(columns={"candidate_trade_key": "trade_key"})
    merged = b2.merge(c2, on="trade_key", how="outer", indicator=True)
    merged.insert(0, "scenario", scenario)
    merged["trade_match_status"] = merged["_merge"].map({"both": "MATCHED", "left_only": "BASELINE_ONLY", "right_only": "CANDIDATE_ONLY"})
    merged = merged.drop(columns=["_merge"])
    merged["return_delta_pct"] = pd.to_numeric(merged.get("candidate_return_pct_trade"), errors="coerce") - pd.to_numeric(merged.get("baseline_return_pct_trade"), errors="coerce")
    merged["exit_changed"] = (
        merged.get("baseline_exit_time", pd.Series(index=merged.index)).astype(str) != merged.get("candidate_exit_time", pd.Series(index=merged.index)).astype(str)
    ) | (
        merged.get("baseline_note", pd.Series(index=merged.index)).astype(str) != merged.get("candidate_note", pd.Series(index=merged.index)).astype(str)
    )
    merged["loss_to_win"] = (pd.to_numeric(merged.get("baseline_return_pct_trade"), errors="coerce") <= 0) & (pd.to_numeric(merged.get("candidate_return_pct_trade"), errors="coerce") > 0)
    merged["win_to_loss"] = (pd.to_numeric(merged.get("baseline_return_pct_trade"), errors="coerce") > 0) & (pd.to_numeric(merged.get("candidate_return_pct_trade"), errors="coerce") <= 0)
    merged["tail_winner_cut"] = (pd.to_numeric(merged.get("baseline_return_pct_trade"), errors="coerce") >= 20.0) & (pd.to_numeric(merged.get("return_delta_pct"), errors="coerce") <= -5.0)
    return merged


def summarize_trade_diff(diff_df: pd.DataFrame) -> pd.DataFrame:
    if diff_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for scenario, g in diff_df.groupby("scenario", dropna=False):
        ret_delta = pd.to_numeric(g.get("return_delta_pct"), errors="coerce")
        rows.append({
            "scenario": scenario,
            "matched_trades": int(g["trade_match_status"].eq("MATCHED").sum()),
            "baseline_only_trades": int(g["trade_match_status"].eq("BASELINE_ONLY").sum()),
            "candidate_only_trades": int(g["trade_match_status"].eq("CANDIDATE_ONLY").sum()),
            "exit_changed_count": int(g.get("exit_changed", pd.Series(False, index=g.index)).fillna(False).sum()),
            "loss_to_win_count": int(g.get("loss_to_win", pd.Series(False, index=g.index)).fillna(False).sum()),
            "win_to_loss_count": int(g.get("win_to_loss", pd.Series(False, index=g.index)).fillna(False).sum()),
            "tail_winner_cut_count": int(g.get("tail_winner_cut", pd.Series(False, index=g.index)).fillna(False).sum()),
            "avg_return_delta_pct": float(ret_delta.mean()) if ret_delta.notna().any() else np.nan,
            "median_return_delta_pct": float(ret_delta.median()) if ret_delta.notna().any() else np.nan,
            "sum_return_delta_pct": float(ret_delta.sum()) if ret_delta.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def engine_diagnostics(scenario: str, trades: list[dict[str, Any]]) -> pd.DataFrame:
    tdf = _trade_frame(trades)
    if tdf.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (engine, side), g in tdf.groupby(["engine", "side"], dropna=False):
        r = pd.to_numeric(g["return_pct_trade"], errors="coerce").fillna(0.0)
        pos = r[r > 0].sum()
        neg = -r[r < 0].sum()
        rows.append({
            "scenario": scenario,
            "engine": engine,
            "side": side,
            "trades": int(len(g)),
            "win_rate": float((r > 0).mean() * 100.0) if len(g) else 0.0,
            "sum_return_pct": float(r.sum()),
            "avg_return_pct": float(r.mean()) if len(g) else 0.0,
            "median_return_pct": float(r.median()) if len(g) else 0.0,
            "profit_factor_pct": float(pos / neg) if neg > 1e-12 else float("inf") if pos > 0 else 0.0,
            "avg_mfe_r": float(pd.to_numeric(g["mfe_r_trade"], errors="coerce").mean()),
            "avg_mae_r": float(pd.to_numeric(g["mae_r_trade"], errors="coerce").mean()),
            "structural_stop_exits": int(g["note"].eq("STRUCTURE_STOP").sum()),
            "structural_close_confirm_exits": int(g["note"].eq("STRUCTURE_CLOSE_CONFIRM_NEXT_OPEN").sum()),
        })
    return pd.DataFrame(rows)


def top_dependency_from_trades(scenario: str, trades: list[dict[str, Any]]) -> dict[str, Any]:
    return suite.top_trade_dependency(scenario, trades)


def scenario_family(name: str) -> str:
    if name == BASELINE:
        return "baseline"
    if "initial_struct_bear_swing" in name:
        return "initial_bear_swing_path_change"
    if "struct_stop_all_swing" in name:
        return "all_engine_swing_struct_stop"
    if "struct_stop_bear_current_bar" in name:
        return "bear_current_bar_winrate_stop"
    if "struct_giveback_all_swing" in name:
        return "all_engine_swing_giveback_watch"
    if "struct_fail_bull_hybrid_tighter" in name:
        return "bull_failure_exit_watch"
    return "other"


def run_verification(args: argparse.Namespace, forwarded: list[str], out_dir: Path) -> dict[str, pd.DataFrame]:
    rerun_dir = out_dir / "rerun_cache"
    rerun_dir.mkdir(parents=True, exist_ok=True)
    sargs = _make_structural_args(forwarded, rerun_dir)
    specs = build_candidate_specs(include_watch=args.include_watch, include_neighbourhood=args.include_neighbourhood)
    if args.max_scenarios is not None:
        # Preserve baseline + core first.
        specs = specs[: max(1, int(args.max_scenarios))]
    spec_by_name = {s.name: s for s in specs}
    print(f"[verify] scenarios={len(specs)} include_watch={args.include_watch} include_neighbourhood={args.include_neighbourhood}", flush=True)
    data = _prepare_base_data(sargs)

    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[pd.DataFrame] = []
    top_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    engine_rows: list[pd.DataFrame] = []
    diff_rows: list[pd.DataFrame] = []
    trades_by_name: dict[str, list[dict[str, Any]]] = {}

    baseline_trades: list[dict[str, Any]] = []

    for i, spec in enumerate(specs, start=1):
        trades, equity = _run_spec(data, sargs, spec)
        trades = v10a.attach_engine_to_trades(trades, data["features"])
        trades_by_name[spec.name] = trades
        if spec.name == BASELINE:
            baseline_trades = trades
        extra = {
            "scenario_family": scenario_family(spec.name),
            "candidate_tier": "CORE" if spec.name in CORE_CANDIDATES else "WATCH" if spec.name in WATCH_CANDIDATES else "NEIGHBOUR" if spec.name != BASELINE else "BASELINE",
            "struct_source": spec.source,
            "struct_action": spec.action,
            "struct_engine_scope": spec.engine_scope,
            "struct_trigger_mfe_r": spec.trigger_mfe_r,
            "struct_min_hold_bars": spec.min_hold_bars,
            "struct_lookback": spec.lookback,
            "struct_buffer_atr": spec.buffer_atr,
            "struct_initial_stop": spec.initial_struct_stop,
            "rule_note": json.dumps(asdict(spec), ensure_ascii=False),
        }
        summary_rows.append(suite.summary_metrics(spec.name, trades, equity, data["exec_cfg"].initial_capital, extra=extra))
        yearly_rows.append(suite.yearly_metrics(spec.name, trades, equity))
        top_rows.append(top_dependency_from_trades(spec.name, trades))
        tdf = _trade_frame(trades)
        note = tdf.get("note", pd.Series("", index=tdf.index)).astype(str) if not tdf.empty else pd.Series(dtype=str)
        audit_rows.append({
            "scenario": spec.name,
            "scenario_family": scenario_family(spec.name),
            "structural_stop_exits": int(note.eq("STRUCTURE_STOP").sum()) if len(note) else 0,
            "structural_close_confirm_exits": int(note.eq("STRUCTURE_CLOSE_CONFIRM_NEXT_OPEN").sum()) if len(note) else 0,
            "structure_update_trades": int(pd.to_numeric(tdf.get("structure_updates", pd.Series(0, index=tdf.index)), errors="coerce").fillna(0).gt(0).sum()) if not tdf.empty else 0,
            "initial_struct_stop": bool(spec.initial_struct_stop),
            "path_change_risk": "HIGH" if spec.initial_struct_stop else "MEDIUM" if spec.engine_scope == "ALL" else "LOW_TO_MEDIUM",
        })
        engine_rows.append(engine_diagnostics(spec.name, trades))
        if spec.name != BASELINE:
            diff_rows.append(build_trade_diff(spec.name, baseline_trades, trades))
        if args.write_trades:
            tdf.to_csv(rerun_dir / f"{spec.name}__trades.csv", index=False)
            if not equity.empty:
                equity.to_csv(rerun_dir / f"{spec.name}__equity.csv")
        if args.checkpoint_every and i % max(1, int(args.checkpoint_every)) == 0:
            pd.DataFrame(summary_rows).to_csv(out_dir / "_checkpoint_01_summary.csv", index=False)
            pd.DataFrame(audit_rows).to_csv(out_dir / "_checkpoint_06_path_audit.csv", index=False)
        print(f"  verified {i}/{len(specs)} {spec.name}", flush=True)

    summary = pd.DataFrame(summary_rows)
    compare = structural.compare_to_baseline(summary)
    yearly = structural._normalize_yearly_columns(pd.concat([x for x in yearly_rows if x is not None and not x.empty], ignore_index=True) if yearly_rows else pd.DataFrame())
    top = pd.DataFrame(top_rows)
    audit = pd.DataFrame(audit_rows)
    engine = pd.concat([x for x in engine_rows if x is not None and not x.empty], ignore_index=True) if engine_rows else pd.DataFrame()
    trade_diff = pd.concat([x for x in diff_rows if x is not None and not x.empty], ignore_index=True) if diff_rows else pd.DataFrame()
    trade_diff_summary = summarize_trade_diff(trade_diff)
    decision = build_decision_matrix(compare, yearly, top, audit, trade_diff_summary, pd.DataFrame(), args, spec_by_name)

    summary.to_csv(out_dir / "01_verification_summary.csv", index=False)
    compare.to_csv(out_dir / "02_compare_to_v10a.csv", index=False)
    yearly.to_csv(out_dir / "03_yearly_robustness.csv", index=False)
    top.to_csv(out_dir / "04_top_trade_dependency.csv", index=False)
    engine.to_csv(out_dir / "05_engine_signal_diagnostics.csv", index=False)
    audit.to_csv(out_dir / "06_path_change_audit.csv", index=False)
    trade_diff_summary.to_csv(out_dir / "07_trade_diff_summary_vs_baseline.csv", index=False)
    trade_diff.to_csv(out_dir / "08_trade_diff_detail_vs_baseline.csv", index=False)

    stress_df = pd.DataFrame()
    if args.stress:
        stress_df = run_stress(args, forwarded, out_dir)
        decision = build_decision_matrix(compare, yearly, top, audit, trade_diff_summary, stress_df, args, spec_by_name)

    decision.to_csv(out_dir / "09_candidate_decision_matrix.csv", index=False)
    lookahead = build_no_lookahead_audit(specs, trade_diff, audit)
    lookahead.to_csv(out_dir / "10_no_lookahead_and_overfit_audit.csv", index=False)
    neighbourhood = build_neighbourhood_summary(compare, decision)
    neighbourhood.to_csv(out_dir / "11_parameter_neighbourhood_summary.csv", index=False)

    return {
        "summary": summary,
        "compare": compare,
        "yearly": yearly,
        "top": top,
        "engine": engine,
        "audit": audit,
        "trade_diff_summary": trade_diff_summary,
        "stress": stress_df,
        "decision": decision,
        "lookahead": lookahead,
        "neighbourhood": neighbourhood,
    }


def run_stress(args: argparse.Namespace, forwarded: list[str], out_dir: Path) -> pd.DataFrame:
    stress_dir = out_dir / "stress_cache"
    stress_dir.mkdir(parents=True, exist_ok=True)
    # Stress only baseline + core + watch, not full neighbourhood.
    specs = build_candidate_specs(include_watch=args.include_watch, include_neighbourhood=False)
    fees = [0.00055, 0.00075, 0.00100]
    slips = [0.0002, 0.0005, 0.0010]
    rows: list[dict[str, Any]] = []
    for fee in fees:
        for slip in slips:
            extra = [x for x in forwarded if x not in ["--fee-rate", "--slippage-pct"]]
            sargs = _make_structural_args(extra + ["--fee-rate", str(fee), "--slippage-pct", str(slip)], stress_dir)
            data = _prepare_base_data(sargs)
            print(f"[stress] fee={fee} slippage={slip} scenarios={len(specs)}", flush=True)
            for spec in specs:
                trades, equity = _run_spec(data, sargs, spec)
                trades = v10a.attach_engine_to_trades(trades, data["features"])
                sm = suite.summary_metrics(spec.name, trades, equity, data["exec_cfg"].initial_capital)
                sm.update({"fee_rate": fee, "slippage_pct": slip, "scenario_family": scenario_family(spec.name)})
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
        df["stress_return_ratio_vs_baseline"] = _num(df, "total_return_pct") / _num(df, "baseline_total_return_pct").replace(0, np.nan)
        df["stress_win_rate_delta"] = _num(df, "win_rate") - _num(df, "baseline_win_rate")
        df["stress_dd_delta"] = _num(df, "max_drawdown_pct") - _num(df, "baseline_max_drawdown_pct")
        df["stress_pf_delta"] = _num(df, "profit_factor") - _num(df, "baseline_profit_factor")
    df.to_csv(out_dir / "12_fee_slippage_stress.csv", index=False)
    return df


def build_decision_matrix(
    compare: pd.DataFrame,
    yearly: pd.DataFrame,
    top: pd.DataFrame,
    audit: pd.DataFrame,
    trade_diff_summary: pd.DataFrame,
    stress: pd.DataFrame,
    args: argparse.Namespace,
    spec_by_name: dict[str, structural.StructuralStopSpec],
) -> pd.DataFrame:
    if compare.empty:
        return pd.DataFrame()
    df = compare.loc[~compare["scenario"].eq(BASELINE)].copy()
    df["scenario_family"] = df["scenario"].map(scenario_family)
    df["candidate_tier"] = np.select(
        [df["scenario"].isin(CORE_CANDIDATES), df["scenario"].isin(WATCH_CANDIDATES)],
        ["CORE", "WATCH"],
        default="NEIGHBOUR",
    )
    if not top.empty:
        df = df.merge(top, on="scenario", how="left", suffixes=("", "_top"))
    if not audit.empty:
        df = df.merge(audit, on="scenario", how="left", suffixes=("", "_audit"))
    if not trade_diff_summary.empty:
        df = df.merge(trade_diff_summary, on="scenario", how="left", suffixes=("", "_diff"))
    if not yearly.empty:
        y = structural._normalize_yearly_columns(yearly)
        if not y.empty:
            yagg = y.groupby("scenario").agg(
                yearly_return_min=("yearly_return_pct", "min"),
                yearly_return_median=("yearly_return_pct", "median"),
                positive_years=("yearly_return_pct", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
                year_count=("yearly_return_pct", "count"),
                yearly_win_rate_median=("yearly_win_rate", "median"),
            ).reset_index()
            df = df.merge(yagg, on="scenario", how="left")
    if not stress.empty:
        sagg = stress.loc[~stress["scenario"].eq(BASELINE)].groupby("scenario").agg(
            stress_return_ratio_min=("stress_return_ratio_vs_baseline", "min"),
            stress_return_ratio_median=("stress_return_ratio_vs_baseline", "median"),
            stress_win_rate_delta_min=("stress_win_rate_delta", "min"),
            stress_dd_delta_max=("stress_dd_delta", "max"),
            stress_pf_delta_min=("stress_pf_delta", "min"),
        ).reset_index()
        df = df.merge(sagg, on="scenario", how="left")

    df["is_initial_struct_stop"] = df["scenario"].map(lambda n: bool(spec_by_name.get(str(n), structural.StructuralStopSpec(name=str(n))).initial_struct_stop))
    df["return_ratio_ok"] = _num(df, "return_ratio_vs_baseline").ge(float(args.min_live_return_ratio))
    df["winrate_goal_ok"] = _num(df, "win_rate_delta").ge(float(args.min_win_rate_delta))
    df["dd_ok"] = _num(df, "max_drawdown_pct_delta").le(float(args.max_dd_delta))
    df["pf_ok"] = _num(df, "profit_factor_delta").ge(-float(args.max_pf_drop))
    df["stress_ok"] = _num(df, "stress_return_ratio_min", 1.0).ge(0.85) & _num(df, "stress_pf_delta_min", 0.0).ge(-1.0)
    df["tail_cut_warning"] = _num(df, "tail_winner_cut_count").ge(2)
    df["path_change_warning"] = df["is_initial_struct_stop"] | _num(df, "baseline_only_trades").gt(5) | _num(df, "candidate_only_trades").gt(5)

    conditions = [
        df["scenario_family"].eq("all_engine_swing_struct_stop")
        & _num(df, "return_ratio_vs_baseline").ge(1.15)
        & df["dd_ok"]
        & _num(df, "profit_factor_delta").ge(0.0)
        & df["stress_ok"]
        & (~df["tail_cut_warning"]),
        df["scenario_family"].eq("bear_current_bar_winrate_stop")
        & df["return_ratio_ok"]
        & df["winrate_goal_ok"]
        & df["dd_ok"]
        & df["pf_ok"]
        & df["stress_ok"],
        df["scenario_family"].eq("initial_bear_swing_path_change")
        & _num(df, "return_ratio_vs_baseline").ge(1.10)
        & df["dd_ok"]
        & df["pf_ok"],
    ]
    choices = [
        "NEXT_RESEARCH_CANDIDATE__RETURN_ENHANCER",
        "NEXT_RESEARCH_CANDIDATE__WINRATE_ENHANCER",
        "RESEARCH_ONLY__HIGH_PATH_CHANGE_INITIAL_STOP",
    ]
    df["anti_overfit_decision"] = np.select(conditions, choices, default="WATCH_OR_REJECT")
    df["live_readiness_note"] = np.where(
        df["is_initial_struct_stop"],
        "Initial stop changes initial R/position size/add-on path; do not migrate before separate implementation audit.",
        np.where(
            df["scenario_family"].eq("bear_current_bar_winrate_stop"),
            "Win-rate improvement must survive stress and trade-diff review; PF/tail truncation are key risks.",
            np.where(
                df["scenario_family"].eq("all_engine_swing_struct_stop"),
                "Best live candidate only if structural level timing is manually reviewed and neighbourhood is stable.",
                "Research-only watchlist; not a live candidate yet.",
            ),
        ),
    )
    sort_cols = ["anti_overfit_decision", "return_ratio_vs_baseline", "win_rate_delta", "profit_factor_delta"]
    return df.sort_values(sort_cols, ascending=[True, False, False, False])


def build_no_lookahead_audit(specs: list[structural.StructuralStopSpec], trade_diff: pd.DataFrame, path_audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    path_by_name = path_audit.set_index("scenario").to_dict("index") if not path_audit.empty and "scenario" in path_audit.columns else {}
    for spec in specs:
        rows.append({
            "scenario": spec.name,
            "scenario_family": scenario_family(spec.name),
            "source": spec.source,
            "action": spec.action,
            "engine_scope": spec.engine_scope,
            "initial_struct_stop": spec.initial_struct_stop,
            "lookback": spec.lookback,
            "trigger_mfe_r": spec.trigger_mfe_r,
            "min_hold_bars": spec.min_hold_bars,
            "uses_future_label_for_rule": False,
            "uses_completed_bar_structure": True,
            "entry_timing_expected": "signal_bar_close -> next_4h_open",
            "exit_timing_expected": "active_stop_intrabar or close_confirm -> next_4h_open",
            "manual_code_review_required": True,
            "path_change_risk": path_by_name.get(spec.name, {}).get("path_change_risk", "BASELINE" if spec.name == BASELINE else "UNKNOWN"),
            "lookahead_audit_note": (
                "Baseline official V10A." if spec.name == BASELINE else
                "High-risk path change: initial stop alters initial R/size. Needs separate code review." if spec.initial_struct_stop else
                "Research executor is designed to use completed bars only; still requires manual review before production."
            ),
        })
    df = pd.DataFrame(rows)
    return df


def build_neighbourhood_summary(compare: pd.DataFrame, decision: pd.DataFrame) -> pd.DataFrame:
    if compare.empty:
        return pd.DataFrame()
    df = compare.loc[~compare["scenario"].eq(BASELINE)].copy()
    df["scenario_family"] = df["scenario"].map(scenario_family)
    rows: list[dict[str, Any]] = []
    for fam, g in df.groupby("scenario_family", dropna=False):
        rr = _num(g, "return_ratio_vs_baseline")
        wr = _num(g, "win_rate_delta")
        dd = _num(g, "max_drawdown_pct_delta")
        pf = _num(g, "profit_factor_delta")
        good_return = g.loc[rr.ge(1.0) & dd.le(3.0) & pf.ge(-0.5)]
        good_win = g.loc[rr.ge(0.9) & wr.ge(5.0) & dd.le(3.0) & pf.ge(-1.0)]
        rows.append({
            "scenario_family": fam,
            "scenario_count": int(len(g)),
            "good_return_count": int(len(good_return)),
            "good_winrate_count": int(len(good_win)),
            "best_return_scenario": "" if g.empty else str(g.sort_values("total_return_pct", ascending=False).iloc[0]["scenario"]),
            "best_return_ratio": float(rr.max()) if len(rr) else np.nan,
            "best_winrate_scenario": "" if g.empty else str(g.sort_values("win_rate_delta", ascending=False).iloc[0]["scenario"]),
            "best_winrate_delta": float(wr.max()) if len(wr) else np.nan,
            "median_return_ratio": float(rr.median()) if len(rr) else np.nan,
            "median_winrate_delta": float(wr.median()) if len(wr) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["good_return_count", "good_winrate_count", "best_return_ratio"], ascending=[False, False, False])


def build_existing_interpretation(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    shortlist_dir = _project_path(args.existing_shortlist_dir)
    grid_dir = _project_path(args.existing_grid_dir)
    compare = _read_csv(shortlist_dir / "11_rerun_shortlist_compare.csv")
    stress = _read_csv(shortlist_dir / "17_fee_slippage_stress.csv")
    decision = _read_csv(shortlist_dir / "12_rerun_shortlist_scoreboard.csv")
    grid_compare = _read_csv(grid_dir / "03_compare_to_v10a.csv")
    if compare.empty and not grid_compare.empty:
        compare = grid_compare.loc[grid_compare["scenario"].isin([BASELINE] + CORE_CANDIDATES + WATCH_CANDIDATES)].copy()
    if compare.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for name in CORE_CANDIDATES:
        g = compare.loc[compare["scenario"].eq(name)]
        if g.empty:
            continue
        r = g.iloc[0].to_dict()
        stress_g = stress.loc[stress["scenario"].eq(name)] if not stress.empty and "scenario" in stress.columns else pd.DataFrame()
        rows.append({
            "scenario": name,
            "current_interpretation": (
                "Front-runner return enhancer; not live-ready until timing/neighbourhood/trade-diff pass." if "all_swing" in name else
                "Win-rate enhancer but stress/PF/tail truncation risk; research-only until stress improves." if "bear_current_bar" in name else
                "Good metrics but high path-change initial stop; needs separate implementation audit."
            ),
            "total_return_pct": r.get("total_return_pct"),
            "return_ratio_vs_baseline": r.get("return_ratio_vs_baseline"),
            "win_rate_delta": r.get("win_rate_delta"),
            "max_drawdown_pct_delta": r.get("max_drawdown_pct_delta"),
            "profit_factor_delta": r.get("profit_factor_delta"),
            "stress_return_ratio_min": np.nan if stress_g.empty else pd.to_numeric(stress_g.get("return_ratio_vs_baseline_stress", stress_g.get("stress_return_ratio_vs_baseline")), errors="coerce").min(),
            "stress_pf_delta_min": np.nan if stress_g.empty else pd.to_numeric(stress_g.get("pf_delta_stress", stress_g.get("stress_pf_delta")), errors="coerce").min(),
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "00_existing_result_interpretation.csv", index=False)
    return out


def write_meta(out_dir: Path, args: argparse.Namespace, forwarded: list[str]) -> None:
    meta = {
        "script": "research/v10a_structural_stop_anti_overfit_verification.py",
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "args": vars(args),
        "forwarded_args": forwarded,
        "core_candidates": CORE_CANDIDATES,
        "watch_candidates": WATCH_CANDIDATES,
        "strict_notes": [
            "This is research-only and does not modify official V10A.",
            "No candidate is promoted directly to AetherEdge from this script.",
            "Initial structural stops are marked high path-change risk because they alter initial R, sizing, and add-on path.",
            "Manual code review is still required to rule out implementation-level lookahead.",
        ],
    }
    with (out_dir / "99_research_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)


def main() -> int:
    args, forwarded = parse_args()
    out_dir = _project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96, flush=True)
    print("V10A Structural Stop Anti-Overfit Verification", flush=True)
    print("Research only. No official strategy change. No live promotion.", flush=True)
    print("=" * 96, flush=True)

    if args.from_existing:
        existing = build_existing_interpretation(args, out_dir)
        print(f"Existing interpretation rows: {len(existing)}", flush=True)
    if args.rerun_core:
        run_verification(args, forwarded, out_dir)
    write_meta(out_dir, args, forwarded)
    print(f"Outputs written to: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
