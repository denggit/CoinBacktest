#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R03: accumulated active-flow pressure + causal Price Action.

The study does not optimize fixed TP/SL.  It tests two predeclared processes:

1. accumulated pressure -> old structure sweep -> reclaim -> reversal;
2. accumulated pressure -> old structure break -> retest holds -> continuation.

Entry is next-bar open.  Stop and target are derived from causal swing structure.
A broad timeout exists only as an explicit operational safety fallback and must
remain a small minority of exits before any candidate can qualify.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_common.conditional_edge import profit_factor  # noqa: E402
from src.research_common.flow_impact import flow_field_coverage  # noqa: E402
from src.research_common.flow_impact_io import inclusive_end, load_rich_trade_bars, timeframe_delta  # noqa: E402
from src.research_common.flow_pa_accumulation import (  # noqa: E402
    AccumulatedPAConfig,
    build_accumulated_features,
    build_causal_pivots,
    detect_accumulated_pa_setups,
    resolve_position_conflicts,
    simulate_structural_exits,
)
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402

SCRIPT_NAME = "03_accumulated_pressure_pa"
SCRIPT_VERSION = "1.0.1"
EXPERIMENT_ID = "ETH_MHF_FLOW_IMPACT_STATE_R03"
EDGE_ID = "ETH_MHF_FLOW_IMPACT_STATE"
TITLE = "OKX Accumulated Active-Flow + Causal Price Action"
DEFAULT_OUT_DIR = "data/reports/research/mhf/flow_impact_state/03_accumulated_pressure_pa"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Accumulated active-flow pressure with causal Price Action entry and structural exits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-06-30")
    parser.add_argument("--discovery-end", default="2024-12-31 23:59:59")
    parser.add_argument("--validation-end", default="2025-09-30 23:59:59")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--trade-bar-db-name", default="okx_trade_bars.db")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--accumulation-windows", default="5,10,20")
    parser.add_argument("--baseline-bars", type=int, default=1440)
    parser.add_argument("--baseline-min-periods", type=int, default=720)
    parser.add_argument("--min-accumulation-z", type=float, default=1.50)
    parser.add_argument("--pivot-left", type=int, default=2)
    parser.add_argument("--pivot-right", type=int, default=2)
    parser.add_argument("--structure-lookback-bars", type=int, default=240)
    parser.add_argument("--confirmation-bars", type=int, default=5)
    parser.add_argument("--retest-tolerance-bps", type=float, default=5.0)
    parser.add_argument("--stop-buffer-bps", type=float, default=3.0)
    parser.add_argument("--min-risk-bps", type=float, default=10.0)
    parser.add_argument("--max-risk-bps", type=float, default=150.0)
    parser.add_argument("--min-reward-risk", type=float, default=1.10)
    parser.add_argument("--exhaustion-decay-ratio", type=float, default=0.75)
    parser.add_argument("--continuation-min-persistence", type=float, default=0.50)
    parser.add_argument("--continuation-min-effectiveness", type=float, default=0.0)
    parser.add_argument("--event-cooldown-bars", type=int, default=5)
    parser.add_argument("--max-holding-bars", type=int, default=240)
    parser.add_argument("--entry-fee-rate", type=float, default=0.00055)
    parser.add_argument("--exit-fee-rate", type=float, default=0.00055)
    parser.add_argument("--entry-slippage", type=float, default=0.00020)
    parser.add_argument("--exit-slippage", type=float, default=0.00020)
    parser.add_argument("--minimum-total-trades", type=int, default=1000)
    parser.add_argument("--minimum-discovery-trades", type=int, default=500)
    parser.add_argument("--minimum-validation-trades", type=int, default=200)
    parser.add_argument("--minimum-holdout-trades", type=int, default=200)
    parser.add_argument("--minimum-net-pf", type=float, default=1.20)
    parser.add_argument("--minimum-positive-month-ratio", type=float, default=0.65)
    parser.add_argument("--maximum-timeout-ratio", type=float, default=0.10)
    parser.add_argument("--maximum-top5-winner-share", type=float, default=0.20)
    parser.add_argument("--sample-rows", type=int, default=20_000)
    parser.add_argument("--write-full-trades", action="store_true")
    parser.add_argument("--skip-review-pack", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def _parse_int_csv(value: str) -> tuple[int, ...]:
    out = tuple(sorted(set(int(part.strip()) for part in str(value).split(",") if part.strip())))
    if not out or any(v <= 0 for v in out):
        raise ValueError(f"invalid positive integer CSV: {value!r}")
    return out


def _config(args: argparse.Namespace) -> AccumulatedPAConfig:
    cfg = AccumulatedPAConfig(
        accumulation_windows=_parse_int_csv(args.accumulation_windows),
        baseline_bars=int(args.baseline_bars),
        baseline_min_periods=int(args.baseline_min_periods),
        min_accumulation_z=float(args.min_accumulation_z),
        pivot_left=int(args.pivot_left),
        pivot_right=int(args.pivot_right),
        structure_lookback_bars=int(args.structure_lookback_bars),
        confirmation_bars=int(args.confirmation_bars),
        retest_tolerance_bps=float(args.retest_tolerance_bps),
        stop_buffer_bps=float(args.stop_buffer_bps),
        min_risk_bps=float(args.min_risk_bps),
        max_risk_bps=float(args.max_risk_bps),
        min_reward_risk=float(args.min_reward_risk),
        exhaustion_decay_ratio=float(args.exhaustion_decay_ratio),
        continuation_min_persistence=float(args.continuation_min_persistence),
        continuation_min_effectiveness=float(args.continuation_min_effectiveness),
        event_cooldown_bars=int(args.event_cooldown_bars),
        max_holding_bars=int(args.max_holding_bars),
    )
    cfg.validate()
    return cfg


def _assign_time_fields(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = frame.copy()
    t = pd.to_datetime(out["signal_time"])
    discovery_end = pd.Timestamp(args.discovery_end)
    validation_end = pd.Timestamp(args.validation_end)
    out["research_split"] = np.select(
        [t <= discovery_end, t <= validation_end],
        ["discovery", "validation"],
        default="holdout",
    )
    out["year"] = t.dt.year.astype(int)
    out["month"] = t.dt.to_period("M").astype(str)
    out["date"] = t.dt.date.astype(str)
    return out


def _summary(part: pd.DataFrame, *, prefix: str = "") -> dict[str, Any]:
    if part.empty:
        return {
            f"{prefix}trades": 0,
            f"{prefix}net_mean": np.nan,
            f"{prefix}net_median": np.nan,
            f"{prefix}net_pf": np.nan,
            f"{prefix}win_rate": np.nan,
            f"{prefix}gross_mean": np.nan,
            f"{prefix}timeout_ratio": np.nan,
            f"{prefix}positive_month_ratio": np.nan,
            f"{prefix}positive_years": 0,
            f"{prefix}top5_winner_share": np.nan,
            f"{prefix}events_per_month": np.nan,
            f"{prefix}active_date_ratio": np.nan,
            f"{prefix}longest_gap_days": np.nan,
            f"{prefix}avg_holding_bars": np.nan,
        }
    net = pd.to_numeric(part["net_return"], errors="coerce")
    gross = pd.to_numeric(part["gross_return"], errors="coerce")
    monthly = part.assign(_net=net).groupby("month", observed=False)["_net"].mean()
    yearly = part.assign(_net=net).groupby("year", observed=False)["_net"].mean()
    winners = net[net > 0.0].sort_values(ascending=False)
    top5 = float(winners.head(5).sum() / winners.sum()) if float(winners.sum()) > 0.0 else np.nan
    dates = pd.to_datetime(part["signal_time"]).dt.normalize().drop_duplicates().sort_values()
    start = pd.to_datetime(part["signal_time"]).min().normalize()
    end = pd.to_datetime(part["signal_time"]).max().normalize()
    months = max(1.0, (end - start + pd.Timedelta(days=1)).total_seconds() / (365.2425 / 12.0 * 86400.0))
    calendar_days = max(1, int((end - start).days + 1))
    gaps = dates.diff().dt.total_seconds().div(86400.0).dropna()
    return {
        f"{prefix}trades": int(len(part)),
        f"{prefix}net_mean": float(net.mean()),
        f"{prefix}net_median": float(net.median()),
        f"{prefix}net_pf": float(profit_factor(net)),
        f"{prefix}win_rate": float((net > 0.0).mean()),
        f"{prefix}gross_mean": float(gross.mean()),
        f"{prefix}timeout_ratio": float(part["exit_reason"].eq("safety_timeout").mean()),
        f"{prefix}positive_month_ratio": float((monthly > 0.0).mean()),
        f"{prefix}positive_years": int((yearly > 0.0).sum()),
        f"{prefix}top5_winner_share": top5,
        f"{prefix}events_per_month": float(len(part) / months),
        f"{prefix}active_date_ratio": float(len(dates) / calendar_days),
        f"{prefix}longest_gap_days": float(gaps.max()) if len(gaps) else np.nan,
        f"{prefix}avg_holding_bars": float(pd.to_numeric(part["holding_bars"], errors="coerce").mean()),
    }


def _ranking(trades: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec_id, part in trades.groupby("spec_id", observed=False):
        row: dict[str, Any] = {
            "spec_id": spec_id,
            "branch": str(part.iloc[0]["branch"]),
            "profile": str(part.iloc[0]["profile"]),
            "pressure_window_bars": int(part.iloc[0]["pressure_window_bars"]),
        }
        row.update(_summary(part, prefix="full_"))
        for split in ("discovery", "validation", "holdout"):
            row.update(_summary(part.loc[part["research_split"].eq(split)], prefix=f"{split}_"))
        rows.append(row)
    rank = pd.DataFrame(rows)
    if rank.empty:
        return rank
    rank["sample_gate"] = (
        (rank["full_trades"] >= int(args.minimum_total_trades))
        & (rank["discovery_trades"] >= int(args.minimum_discovery_trades))
        & (rank["validation_trades"] >= int(args.minimum_validation_trades))
        & (rank["holdout_trades"] >= int(args.minimum_holdout_trades))
    )
    rank["split_expectancy_gate"] = (
        (rank["discovery_net_mean"] > 0.0)
        & (rank["validation_net_mean"] > 0.0)
        & (rank["holdout_net_mean"] > 0.0)
    )
    rank["quality_gate"] = (
        (rank["full_net_pf"] >= float(args.minimum_net_pf))
        & (rank["full_positive_month_ratio"] >= float(args.minimum_positive_month_ratio))
        & (rank["full_timeout_ratio"] <= float(args.maximum_timeout_ratio))
        & (rank["full_top5_winner_share"] <= float(args.maximum_top5_winner_share))
        & (rank["full_positive_years"] >= 3)
    )
    rank["qualified_edge_flag"] = rank["sample_gate"] & rank["split_expectancy_gate"] & rank["quality_gate"]
    return rank.sort_values(
        ["qualified_edge_flag", "full_net_mean", "full_net_pf", "full_trades"],
        ascending=[False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def _yearly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby(["spec_id", "year"], observed=False)
        .agg(
            trades=("net_return", "size"),
            net_mean=("net_return", "mean"),
            gross_mean=("gross_return", "mean"),
            win_rate=("net_return", lambda x: float((pd.to_numeric(x) > 0.0).mean())),
            timeout_ratio=("exit_reason", lambda x: float(pd.Series(x).eq("safety_timeout").mean())),
        )
        .reset_index()
    )


def _monthly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby(["spec_id", "month"], observed=False)
        .agg(
            trades=("net_return", "size"),
            net_mean=("net_return", "mean"),
            gross_mean=("gross_return", "mean"),
            win_rate=("net_return", lambda x: float((pd.to_numeric(x) > 0.0).mean())),
        )
        .reset_index()
    )


def _geometry(setups: pd.DataFrame) -> pd.DataFrame:
    if setups.empty:
        return pd.DataFrame()
    return (
        setups.groupby(["branch", "profile", "pressure_window_bars"], observed=False)
        .agg(
            setups=("setup_id", "size"),
            median_risk_bps=("risk_bps", "median"),
            median_reward_bps=("reward_bps", "median"),
            median_reward_risk=("reward_risk", "median"),
            median_pressure_z=("pressure_z", "median"),
            median_decay_ratio=("impact_decay_ratio", "median"),
            median_persistence=("flow_persistence", "median"),
        )
        .reset_index()
    )


def _exit_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby(["spec_id", "exit_reason"], observed=False)
        .agg(trades=("net_return", "size"), net_mean=("net_return", "mean"), holding_bars=("holding_bars", "mean"))
        .reset_index()
    )


def _causal_audit(bars: pd.DataFrame, setups: pd.DataFrame, pivots: pd.DataFrame) -> pd.DataFrame:
    if setups.empty:
        return pd.DataFrame(
            [{"check": "no_setups", "failed_rows": 0, "pass_flag": True}]
        )
    checks = [
        {
            "check": "entry_is_next_bar_open",
            "failed_rows": int((setups["entry_pos"] != setups["signal_pos"] + 1).sum()),
        },
        {
            "check": "entry_after_signal",
            "failed_rows": int((pd.to_datetime(setups["entry_time"]) <= pd.to_datetime(setups["signal_time"])).sum()),
        },
        {
            "check": "positive_structure_risk",
            "failed_rows": int((pd.to_numeric(setups["risk_bps"], errors="coerce") <= 0.0).sum()),
        },
        {
            "check": "positive_structure_reward",
            "failed_rows": int((pd.to_numeric(setups["reward_bps"], errors="coerce") <= 0.0).sum()),
        },
        {
            "check": "pivot_available_after_confirmation_plus_one",
            "failed_rows": int((pivots["available_pos"] <= pivots["pivot_pos"] + 2).sum()) if not pivots.empty else 0,
        },
    ]
    out = pd.DataFrame(checks)
    out["pass_flag"] = out["failed_rows"].eq(0)
    return out


def _feature_dictionary() -> pd.DataFrame:
    rows = [
        ("accumulated_notional", "Absolute net taker notional accumulated over 5/10/20 closed bars."),
        ("pressure_z", "Log accumulated net taker notional relative to a prior-only rolling baseline."),
        ("flow_persistence", "Share of bars whose taker-flow sign agrees with the accumulated direction."),
        ("impact_decay_ratio", "Late-half directional impact per flow divided by early-half absolute impact per flow."),
        ("break_level", "Last causally confirmed swing high/low available before the accumulated attack began."),
        ("exhaustion_reversal", "Old structure sweep followed by closed-bar reclaim and opposite PA resume."),
        ("continuation", "Old structure break followed by retest hold and same-direction PA resume."),
        ("stop_price", "Attack/retest structural invalidation level plus a small fixed buffer."),
        ("target_price", "Nearest already-confirmed structure target; measured move only when no known target exists."),
        ("safety_timeout", "Operational fallback only; candidate fails if timeout share exceeds the hard limit."),
    ]
    return pd.DataFrame(rows, columns=["field", "definition"])


def _diagnostic_full_report(bars: pd.DataFrame, trades: pd.DataFrame, out_dir: Path, symbol: str) -> None:
    primary = trades.loc[trades["profile"].isin(["sweep_reclaim_body", "break_accept_body"])].copy()
    primary = resolve_position_conflicts(primary)
    history: list[dict[str, Any]] = []
    capital = 10_000.0
    for row in primary.itertuples(index=False):
        cap_before = capital
        pnl = cap_before * float(row.net_return)
        capital += pnl
        history.append(
            {
                "entry_time": pd.Timestamp(row.entry_time),
                "exit_time": pd.Timestamp(row.exit_time),
                "type": f"{row.branch}:{row.side_name}",
                "entry": float(row.entry_price),
                "exit": float(row.exit_price),
                "pnl": float(pnl),
                "fee": float(cap_before * max(0.0, float(row.gross_return) - float(row.fee_only_return))),
                "capital": float(capital),
            }
        )
    total_days = max(1.0, (bars.index[-1] - bars.index[0]).total_seconds() / 86400.0)
    print_full_report(
        history,
        bars,
        10_000.0,
        capital,
        "FlowImpact_R03_PA_Diagnostic",
        total_days,
        "False",
        symbol=symbol,
        report_dir=str(out_dir),
    )


def run_research(bars: pd.DataFrame, args: argparse.Namespace) -> dict[str, Path]:
    cfg = _config(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bar_delta = timeframe_delta(args.timeframe)
    start = pd.Timestamp(args.start_date)
    end = inclusive_end(args.end_date, bar_delta)
    fee_only_cost = float(args.entry_fee_rate + args.exit_fee_rate)
    normal_cost = float(fee_only_cost + args.entry_slippage + args.exit_slippage)

    print(f"[features] accumulated pressure windows={cfg.accumulation_windows}", flush=True)
    features = build_accumulated_features(bars, cfg)
    print(f"[pa] causal pivots left={cfg.pivot_left} right={cfg.pivot_right}", flush=True)
    pivots = build_causal_pivots(bars, left=cfg.pivot_left, right=cfg.pivot_right)
    print("[pa] accumulated pressure -> sweep/reclaim or break/retest", flush=True)
    setups = detect_accumulated_pa_setups(
        bars,
        features,
        pivots,
        cfg,
        progress_enabled=not args.no_progress,
    )
    if setups.empty:
        raise RuntimeError("No accumulated-flow Price Action setups were produced")
    setups = setups.loc[pd.to_datetime(setups["signal_time"]).between(start, end)].copy()
    if setups.empty:
        raise RuntimeError("No setups inside the formal research window")
    print(f"[pa] setups={len(setups):,}", flush=True)
    print("[backtest] PA structural TP/SL first-touch", flush=True)
    trades = simulate_structural_exits(
        bars,
        setups,
        normal_cost=normal_cost,
        fee_only_cost=fee_only_cost,
        max_holding_bars=cfg.max_holding_bars,
        progress_enabled=not args.no_progress,
    )
    trades = _assign_time_fields(trades, args)
    conflict_parts = [resolve_position_conflicts(part) for _, part in trades.groupby("spec_id", observed=False)]
    conflict = pd.concat(conflict_parts, ignore_index=True) if conflict_parts else trades.iloc[0:0].copy()
    conflict = _assign_time_fields(conflict, args) if not conflict.empty else conflict

    coverage = flow_field_coverage(bars)
    feature_coverage = pd.DataFrame(
        [
            {
                "feature": column,
                "non_null_ratio": float(pd.to_numeric(features[column], errors="coerce").notna().mean()),
                "unique_values": int(pd.to_numeric(features[column], errors="coerce").nunique(dropna=True)),
            }
            for column in features.columns
            if column.startswith(("impact_decay_ratio_", "early_impact_", "late_impact_", "late_directional_flow_share_"))
        ]
    )
    setup_counts = (
        setups.groupby(["branch", "profile", "pressure_window_bars", "side_name"], observed=False)
        .size()
        .rename("setups")
        .reset_index()
    )
    geometry = _geometry(setups)
    independent_rank = _ranking(trades, args)
    conflict_rank = _ranking(conflict, args)
    qualified = conflict_rank.loc[conflict_rank.get("qualified_edge_flag", False).fillna(False)].copy() if not conflict_rank.empty else pd.DataFrame()
    yearly = _yearly(conflict)
    monthly = _monthly(conflict)
    exits = _exit_summary(conflict)
    audit = _causal_audit(bars, setups, pivots)
    sample = conflict.sort_values("entry_time", kind="stable").head(int(args.sample_rows))
    feature_dict = _feature_dictionary()

    decision = "promote_to_backtest" if not qualified.empty else "research_continue_or_reject"
    if qualified.empty:
        decision_reason = (
            "No predeclared accumulated-flow + PA specification passed the >=1000-trade, "
            "three-split net-positive, PF, monthly stability and timeout gates."
        )
    else:
        decision_reason = "At least one predeclared PA specification passed every hard gate."
    design = f"""# R03 Research Design

- Object: accumulated OKX taker-flow pressure plus causal Price Action.
- Windows: {list(cfg.accumulation_windows)} closed 1m bars.
- Branch A: old swing sweep -> reclaim -> exhaustion reversal.
- Branch B: old swing break -> retest hold -> continuation.
- Entry: confirmation close -> next bar open.
- Stop: attack/retest structural invalidation plus {cfg.stop_buffer_bps:.1f} bps buffer.
- Target: nearest causally known structure; measured move fallback only.
- No fixed TP/SL grid and no strategy time-exit optimisation.
- Safety timeout: {cfg.max_holding_bars} bars, explicit and qualification-capped at {float(args.maximum_timeout_ratio):.1%}.
- Cost: fee-only={fee_only_cost:.4%}; normal={normal_cost:.4%}.
- Discovery/validation/holdout: <= {args.discovery_end} / <= {args.validation_end} / later.
- Books and Liquidity: not used.
"""
    brief = f"""# R03 Research Brief

Primary decision: `{decision}`

{decision_reason}

## Counts
- Causal pivots: {len(pivots):,}
- PA setups: {len(setups):,}
- Independent trades: {len(trades):,}
- Conflict-resolved trades: {len(conflict):,}
- Qualified specs: {len(qualified):,}

## Interpretation rule
R03 tests whether accumulated pressure becomes tradable only after a genuine PA
sequence.  A profitable small cell is not acceptable.  Any candidate must retain
at least {int(args.minimum_total_trades):,} conflict-resolved trades and remain net
positive in discovery, validation and holdout.
"""

    artifacts = [
        (coverage, "01_input_field_coverage.csv"),
        (feature_coverage, "02_accumulation_feature_coverage.csv"),
        (setup_counts, "03_pa_setup_counts.csv"),
        (geometry, "04_structural_geometry.csv"),
        (independent_rank, "05_independent_candidate_ranking.csv"),
        (conflict_rank, "06_conflict_resolved_candidate_ranking.csv"),
        (qualified, "07_qualified_candidates.csv"),
        (yearly, "08_yearly_stability.csv"),
        (monthly, "09_monthly_stability.csv"),
        (exits, "10_exit_reason_summary.csv"),
        (audit, "11_causal_audit.csv"),
        (sample, "12_trade_sample.csv"),
        (feature_dict, "13_feature_dictionary.csv"),
    ]
    reporter = ProgressReporter("[artifacts] R03 tables", len(artifacts), every=1, enabled=not args.no_progress)
    for done, (frame, name) in enumerate(artifacts, start=1):
        frame.to_csv(out_dir / name, index=False, float_format="%.10g", lineterminator="\n")
        reporter.update(done)
    reporter.close()
    if args.write_full_trades:
        conflict.to_csv(out_dir / "12b_full_conflict_resolved_trades.csv.gz", index=False, compression="gzip")
    (out_dir / "00_research_design.md").write_text(design, encoding="utf-8")
    (out_dir / "14_research_brief.md").write_text(brief, encoding="utf-8")
    manifest = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "status": "research_only",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "warmup_start_date": args.warmup_start_date,
        "research_start": args.start_date,
        "research_end": args.end_date,
        "accumulation_windows": list(cfg.accumulation_windows),
        "min_accumulation_z": cfg.min_accumulation_z,
        "pa_pivot_left": cfg.pivot_left,
        "pa_pivot_right": cfg.pivot_right,
        "entry_model": "closed PA confirmation -> next open",
        "stop_model": "price-action structure invalidation",
        "target_model": "causal structure target / measured move fallback",
        "fixed_tp_sl_optimized": False,
        "time_exit_optimized": False,
        "safety_timeout_bars": cfg.max_holding_bars,
        "fee_only_cost": fee_only_cost,
        "normal_cost": normal_cost,
        "setups": int(len(setups)),
        "independent_trades": int(len(trades)),
        "conflict_resolved_trades": int(len(conflict)),
        "qualified_specs": int(len(qualified)),
        "books_used": False,
        "liquidity_used": False,
        "created_at": pd.Timestamp.now("UTC").isoformat(),
    }
    (out_dir / "15_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    _diagnostic_full_report(bars.loc[start:end], conflict, out_dir, args.symbol)
    if not args.skip_review_pack:
        finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(
        f"[done] report_dir={out_dir} setups={len(setups):,} conflict_trades={len(conflict):,} qualified={len(qualified):,}",
        flush=True,
    )
    return {"report_dir": out_dir, "review_pack": out_dir / "gpt_review_pack.zip"}


def _synthetic_bars(n: int = 2400) -> pd.DataFrame:
    rng = np.random.default_rng(20260725)
    index = pd.date_range("2022-01-01", periods=n, freq="1min")
    ret = rng.normal(0.0, 0.00018, n)
    delta = rng.normal(0.0, 40_000.0, n)
    # Repeated sustained attacks and retraces so the pipeline has PA material.
    for start in range(900, n - 40, 120):
        sign = 1 if (start // 120) % 2 == 0 else -1
        delta[start : start + 10] += sign * 550_000.0
        ret[start : start + 5] += sign * 0.0007
        ret[start + 5 : start + 10] += sign * 0.00005
        ret[start + 10 : start + 14] -= sign * 0.0005
    notional = 1_300_000.0 + rng.lognormal(12.0, 0.3, n)
    buy = np.maximum((notional + delta) / 2.0, 1.0)
    sell = np.maximum((notional - delta) / 2.0, 1.0)
    close = 1800.0 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.00015
    low = np.minimum(open_, close) * 0.99985
    trades = np.maximum(50, np.round(notional / 5000.0).astype(int))
    buy_trades = np.clip(np.round(trades * (0.5 + 0.35 * delta / notional)), 1, trades - 1).astype(int)
    sell_trades = trades - buy_trades
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": notional / close, "notional": notional,
            "buy_notional": buy, "sell_notional": sell, "delta_notional": buy - sell,
            "trades_count": trades, "buy_trades_count": buy_trades, "sell_trades_count": sell_trades,
            "large_buy_notional": np.maximum(delta, 0.0),
            "large_sell_notional": np.maximum(-delta, 0.0),
            "large_delta_notional": delta,
            "large_trades_count": np.full(n, 5),
            "max_trade_notional": np.maximum(np.abs(delta) * 0.25, 1000.0),
        },
        index=index,
    )


def run_self_test(args: argparse.Namespace) -> int:
    print("[self-test] R03 accumulated pressure + PA", flush=True)
    bars = _synthetic_bars()
    with tempfile.TemporaryDirectory(prefix="flow_pa_r03_") as tmp:
        args.out_dir = tmp
        args.warmup_start_date = "2022-01-01"
        args.start_date = "2022-01-01"
        args.end_date = str(bars.index[-1])
        args.discovery_end = str(bars.index[int(len(bars) * 0.50)])
        args.validation_end = str(bars.index[int(len(bars) * 0.75)])
        args.baseline_bars = 240
        args.baseline_min_periods = 120
        args.min_accumulation_z = 0.8
        args.min_risk_bps = 1.0
        args.max_risk_bps = 300.0
        args.min_reward_risk = 0.2
        args.minimum_total_trades = 1
        args.minimum_discovery_trades = 1
        args.minimum_validation_trades = 1
        args.minimum_holdout_trades = 1
        args.skip_review_pack = True
        args.no_progress = True
        try:
            result = run_research(bars, args)
        except RuntimeError as exc:
            # Synthetic path may not create every PA branch, but feature/pivot
            # and detection code still must execute cleanly.
            if "No accumulated-flow Price Action setups" not in str(exc):
                raise
            cfg = _config(args)
            features = build_accumulated_features(bars, cfg)
            pivots = build_causal_pivots(bars, left=cfg.pivot_left, right=cfg.pivot_right)
            if features.empty or pivots.empty:
                raise AssertionError("R03 self-test failed to build causal features/pivots") from exc
            print("[self-test] PASS (no synthetic setup, primitives valid)", flush=True)
            return 0
        required = ["00_research_design.md", "06_conflict_resolved_candidate_ranking.csv", "11_causal_audit.csv", "15_manifest.json"]
        missing = [name for name in required if not (result["report_dir"] / name).exists()]
        if missing:
            raise AssertionError(f"missing R03 self-test artifacts: {missing}")
    print("[self-test] PASS", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test(args)
    bars = load_rich_trade_bars(
        project_root=PROJECT_ROOT,
        symbol=args.symbol,
        timeframe=args.timeframe,
        warmup_start_date=args.warmup_start_date,
        end_date=args.end_date,
        data_dir=args.data_dir,
        db_name=args.trade_bar_db_name,
    )
    run_research(bars, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
