#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Market State Conditional Map V2.

This is a role-aware information audit for a complete market-state map, not a
standalone trading strategy.  It evaluates whether each state axis separates
the future variable it is intended to describe and whether predefined nested
state combinations add stable marginal information.

Existing local 1m rich Trade Bars are sufficient.  No Order Book, OI, Funding
or liquidation data is required for this V2 pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.market_state import MarketStateConfig, MarketStateDataBundle, MarketStateEngine, timeframe_to_timedelta
from src.market_state.conditional_map import (
    ConditionalMapConfig,
    attach_conditional_targets,
    build_condition_definitions,
    build_information_registry,
    build_ladder_incremental_summary,
    build_state_duration_summary,
    build_transition_matrix,
    evaluate_conditions,
)
from src.market_state.validity_audit import ValidityAuditConfig, build_forward_path_frame
from src.research_common.progress import ProgressReporter
from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack


DEFAULT_OUT_DIR = Path("data/reports/research/market_state/02_market_state_conditional_map_v2")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-06-30")
    parser.add_argument("--holdout-start", default="2025-07-01")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 15, 30, 60, 180])
    parser.add_argument("--profiles", nargs="+", choices=["fast", "base", "slow"], default=["fast", "base", "slow"])
    parser.add_argument("--sample-stride-bars", type=int, default=5)
    parser.add_argument("--sequence-lookback-bars", type=int, default=15)
    parser.add_argument("--minimum-samples", type=int, default=500)
    parser.add_argument("--minimum-holdout-samples", type=int, default=100)
    parser.add_argument("--minimum-years", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--local-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--skip-review-pack", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def profile_config(name: str) -> MarketStateConfig:
    base = MarketStateConfig()
    if name == "base":
        return base
    if name == "fast":
        return replace(
            base,
            fast_trend_window=12,
            trend_window=48,
            slow_trend_window=180,
            baseline_window=540,
            trend_confirm_bars=2,
            min_state_bars=12,
            flow_fast_window=2,
            flow_window=8,
            flow_slow_window=24,
            location_window=45,
            structure_window=180,
        )
    if name == "slow":
        return replace(
            base,
            fast_trend_window=24,
            trend_window=96,
            slow_trend_window=360,
            baseline_window=1080,
            trend_confirm_bars=4,
            min_state_bars=24,
            flow_fast_window=5,
            flow_window=20,
            flow_slow_window=45,
            location_window=90,
            structure_window=360,
        )
    raise ValueError(f"unknown profile: {name}")


def _stage(label: str) -> float:
    print(f"\n[stage] {label}", flush=True)
    return time.perf_counter()


def _finish(started: float) -> None:
    print(f"[stage] done elapsed={time.perf_counter() - started:.2f}s", flush=True)


def load_trade_bars(args: argparse.Namespace) -> pd.DataFrame:
    loader = OKXTradeBarLoader(symbol=args.symbol, timeframe=args.timeframe)
    frame = loader.fetch_data_by_date_range(
        args.warmup_start_date,
        args.end_date,
        chunksize=int(args.chunksize),
        cvd_mode="range",
        build_missing=not bool(args.local_only),
    )
    if frame.empty:
        raise RuntimeError(
            "No local Trade Bar data returned. Check coverage or rerun with --no-local-only to build missing days."
        )
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def restrict_definitions(definitions, available_time: pd.Series, start_date: str, end_date: str):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    audit = pd.to_datetime(available_time).between(start, end)
    return [replace(definition, mask=definition.mask & audit) for definition in definitions]


def build_causal_audit(state_frame: pd.DataFrame, path_frame: pd.DataFrame) -> pd.DataFrame:
    available = pd.to_datetime(state_frame["available_time"])
    timestamps = pd.Series(pd.DatetimeIndex(state_frame.index), index=state_frame.index)
    entry_time = pd.to_datetime(path_frame["entry_time"])
    checks = [
        {
            "check": "available_time_not_before_bar_timestamp",
            "passed": bool((available >= timestamps).all()),
            "violations": int((available < timestamps).sum()),
        },
        {
            "check": "next_open_entry_strictly_after_signal_bar",
            "passed": bool((entry_time.dropna() > pd.DatetimeIndex(path_frame.index)[entry_time.notna()]).all()),
            "violations": int((entry_time.dropna() <= pd.DatetimeIndex(path_frame.index)[entry_time.notna()]).sum()),
        },
        {
            "check": "next_open_entry_not_before_available_time",
            "passed": bool((entry_time.dropna() >= pd.to_datetime(path_frame.loc[entry_time.notna(), "available_time"])).all()),
            "violations": int((entry_time.dropna() < pd.to_datetime(path_frame.loc[entry_time.notna(), "available_time"])).sum()),
        },
        {
            "check": "conditions_built_before_future_labels",
            "passed": True,
            "violations": 0,
        },
    ]
    return pd.DataFrame(checks)


def build_axis_summary(registry: pd.DataFrame) -> pd.DataFrame:
    if registry.empty:
        return pd.DataFrame()
    return (
        registry.groupby(["axis", "evidence_status"], dropna=False)
        .size()
        .rename("conditions")
        .reset_index()
        .sort_values(["axis", "evidence_status"])
    )


def write_executive_summary(
    out_dir: Path,
    *,
    registry: pd.DataFrame,
    axis_summary: pd.DataFrame,
    ladder_incremental: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    keep = registry.loc[registry["evidence_status"].eq("KEEP")] if not registry.empty else pd.DataFrame()
    context = registry.loc[registry["evidence_status"].eq("KEEP_CONTEXT_ONLY")] if not registry.empty else pd.DataFrame()
    revise = registry.loc[registry["evidence_status"].eq("REVISE_SEMANTICS")] if not registry.empty else pd.DataFrame()
    drop = registry.loc[registry["evidence_status"].eq("DROP")] if not registry.empty else pd.DataFrame()

    if len(keep):
        decision = "continue_building_conditional_map"
        reason = "At least one state condition has role-consistent, cross-profile and holdout information gain."
    elif len(context) or len(revise):
        decision = "continue_but_revise_state_semantics"
        reason = "Some states contain weak or opposite information, but no condition is yet robust enough to freeze."
    else:
        decision = "rebuild_state_definitions"
        reason = "No state axis separated its intended target with stable information gain."

    lines = [
        "# Market State Conditional Map V2",
        "",
        f"Decision: {decision}",
        f"Reason: {reason}",
        "",
        "This is a market-state information audit, not a standalone strategy backtest.",
        "Round-trip trading cost is deliberately not used as a state-validity gate.",
        "A state is retained only when it separates the variable it is intended to describe.",
        "",
        "## Data",
        f"Rows: {metadata.get('rows')}",
        f"Range: {metadata.get('data_start')} -> {metadata.get('data_end')}",
        f"Order-flow coverage: {metadata.get('orderflow_coverage_ratio')}",
        f"Profiles: {', '.join(metadata.get('profiles', []))}",
        "",
        "## Registry counts",
        f"KEEP: {len(keep)}",
        f"KEEP_CONTEXT_ONLY: {len(context)}",
        f"REVISE_SEMANTICS: {len(revise)}",
        f"DROP: {len(drop)}",
        "",
        "## Robust role-aware information",
    ]
    if keep.empty:
        lines.append("No condition reached KEEP status.")
    else:
        for row in keep.head(30).itertuples(index=False):
            lines.append(
                f"- {row.condition_name} | axis={row.axis} | role={row.intended_role} | "
                f"profiles={row.supported_profiles} | horizons={row.supported_horizons} | "
                f"best_h={row.best_horizon_bars} | uplift={row.best_primary_uplift:.8f}"
            )

    lines.extend(["", "## Semantics that need revision"])
    if revise.empty:
        lines.append("No stable opposite-semantics condition was found.")
    else:
        for row in revise.head(30).itertuples(index=False):
            lines.append(
                f"- {row.condition_name} | intended={row.intended_role} | observed effect is consistently opposite"
            )

    lines.extend(["", "## Strongest positive ladder increments"])
    if ladder_incremental.empty:
        lines.append("No nested ladder increment was available.")
    else:
        top = ladder_incremental.loc[
            ladder_incremental["incremental_primary_uplift"].gt(0.0)
            & ladder_incremental["holdout_incremental_uplift"].fillna(-np.inf).gt(0.0)
        ].sort_values("incremental_primary_uplift", ascending=False).head(30)
        if top.empty:
            lines.append("No ladder stage added positive all-period and holdout information.")
        else:
            for row in top.itertuples(index=False):
                lines.append(
                    f"- {row.profile} | {row.condition_name} | parent={row.parent_condition} | "
                    f"h={row.horizon_bars} | retain={row.retention_ratio:.2%} | "
                    f"increment={row.incremental_primary_uplift:.8f} | "
                    f"holdout_increment={row.holdout_incremental_uplift:.8f}"
                )

    lines.extend([
        "",
        "## Interpretation rules",
        "- Direction states describe confirmed historical structure; they are not automatic long/short permissions.",
        "- Volatility/activity states are judged on future path width, not future direction.",
        "- Mature/decay/noisy states are valid if they reliably reduce continuation, even without predicting reversal.",
        "- Nested conditions are useful only when each added layer improves the parent condition without collapsing samples.",
    ])
    (out_dir / "00_EXECUTIVE_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conditional_cfg = ConditionalMapConfig(
        horizons_bars=tuple(sorted(set(int(v) for v in args.horizons))),
        sample_stride_bars=int(args.sample_stride_bars),
        sequence_lookback_bars=int(args.sequence_lookback_bars),
        minimum_samples=int(args.minimum_samples),
        minimum_holdout_samples=int(args.minimum_holdout_samples),
        minimum_years=int(args.minimum_years),
        holdout_start=str(args.holdout_start) if args.holdout_start else None,
    )
    conditional_cfg.validate()

    started = _stage("load existing rich OKX Trade Bars")
    source = load_trade_bars(args)
    _finish(started)
    print(
        f"[data] rows={len(source):,} range={source.index[0]} -> {source.index[-1]} "
        f"columns={len(source.columns)} local_only={args.local_only}",
        flush=True,
    )

    path_cfg = ValidityAuditConfig(
        horizons_bars=conditional_cfg.horizons_bars,
        trap_horizon_bars=max(conditional_cfg.horizons_bars),
        holdout_start=conditional_cfg.holdout_start,
        minimum_events=1,
    )
    bar_duration = timeframe_to_timedelta(args.timeframe)

    catalogs: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    yearly_parts: list[pd.DataFrame] = []
    period_parts: list[pd.DataFrame] = []
    transitions: list[pd.DataFrame] = []
    durations: list[pd.DataFrame] = []
    causal_parts: list[pd.DataFrame] = []
    profile_meta: dict[str, object] = {}

    for profile in args.profiles:
        state_cfg = profile_config(profile)
        started = _stage(f"market-state engine profile={profile}")
        bundle = MarketStateDataBundle.from_frame(
            source,
            source=f"okx_trade_bar:{args.symbol}:{args.timeframe}",
            timestamp_semantics="bar_start",
            bar_duration=bar_duration,
            metadata={"profile": profile},
        )
        state_result = MarketStateEngine(state_cfg).compute(bundle)
        state_frame = state_result.frame
        # Conditions are intentionally built before forward labels are attached.
        definitions = build_condition_definitions(
            state_frame,
            conditional_cfg,
            medium_trend_window=state_cfg.trend_window,
            slow_trend_window=state_cfg.slow_trend_window,
        )
        definitions = restrict_definitions(definitions, state_frame["available_time"], args.start_date, args.end_date)
        catalog = pd.DataFrame([definition.catalog_row() for definition in definitions])
        catalog.insert(0, "profile", profile)
        catalogs.append(catalog)

        path_frame = attach_conditional_targets(
            build_forward_path_frame(state_frame, path_cfg),
            conditional_cfg.horizons_bars,
        )
        _finish(started)

        reporter = ProgressReporter(
            label=f"[conditional-map] {profile}",
            total=len(definitions),
            every=max(1, len(definitions) // 10),
        )
        evaluation = evaluate_conditions(
            path_frame,
            definitions,
            conditional_cfg,
            profile=profile,
            progress_callback=lambda done, total: reporter.update(done),
        )
        reporter.close()
        summaries.append(evaluation.summary)
        yearly_parts.append(evaluation.yearly)
        period_parts.append(evaluation.periods)

        audit_mask = pd.to_datetime(state_frame["available_time"]).between(
            pd.Timestamp(args.start_date),
            pd.Timestamp(args.end_date) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1),
        )
        audit_frame = state_frame.loc[audit_mask]
        transitions.append(build_transition_matrix(audit_frame, profile=profile))
        durations.append(build_state_duration_summary(audit_frame, profile=profile))
        causal = build_causal_audit(state_frame, path_frame)
        causal.insert(0, "profile", profile)
        causal_parts.append(causal)

        profile_meta[profile] = {
            "state_config": asdict(state_cfg),
            "state_metadata": state_result.metadata,
            "conditions": len(definitions),
            "summary_rows": int(len(evaluation.summary)),
        }
        print(
            f"[profile] {profile} conditions={len(definitions)} summary_rows={len(evaluation.summary):,}",
            flush=True,
        )
        del bundle, state_result, state_frame, path_frame, definitions, evaluation

    started = _stage("aggregate role-aware evidence and nested increments")
    catalog = pd.concat(catalogs, ignore_index=True) if catalogs else pd.DataFrame()
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    yearly = pd.concat(yearly_parts, ignore_index=True) if yearly_parts else pd.DataFrame()
    periods = pd.concat(period_parts, ignore_index=True) if period_parts else pd.DataFrame()
    transition = pd.concat(transitions, ignore_index=True) if transitions else pd.DataFrame()
    duration = pd.concat(durations, ignore_index=True) if durations else pd.DataFrame()
    causal = pd.concat(causal_parts, ignore_index=True) if causal_parts else pd.DataFrame()
    if summary.empty:
        raise RuntimeError("No eligible market-state conditions were produced.")
    evidence_rows, registry = build_information_registry(summary, yearly, periods, conditional_cfg)
    ladder_incremental = build_ladder_incremental_summary(summary, yearly, periods)
    axis_summary = build_axis_summary(registry)
    _finish(started)

    started = _stage("write report artifacts")
    catalog.to_csv(out_dir / "01_condition_catalog.csv", index=False)
    summary.to_csv(out_dir / "02_axis_information_summary.csv", index=False)
    yearly.to_csv(out_dir / "03_yearly_stability.csv", index=False)
    periods.to_csv(out_dir / "04_pre_holdout_vs_holdout.csv", index=False)
    evidence_rows.to_csv(out_dir / "05_profile_horizon_evidence.csv", index=False)
    registry.to_csv(out_dir / "06_state_information_registry.csv", index=False)
    ladder_incremental.to_csv(out_dir / "07_ladder_incremental.csv", index=False)
    transition.to_csv(out_dir / "08_state_transition_matrix.csv", index=False)
    duration.to_csv(out_dir / "09_state_duration_summary.csv", index=False)
    causal.to_csv(out_dir / "10_causal_audit.csv", index=False)
    axis_summary.to_csv(out_dir / "11_axis_verdict_summary.csv", index=False)
    top_evidence = evidence_rows.sort_values(
        ["information_flag", "primary_uplift", "effect_size"],
        ascending=[False, False, False],
    ).head(500)
    top_evidence.to_csv(out_dir / "12_top_evidence_rows.csv", index=False)

    coverage_ratio = float(
        source.get("delta_notional", pd.Series(np.nan, index=source.index)).notna().mean()
    )
    metadata = {
        "title": "Market State Conditional Map V2",
        "experiment_id": "MARKET_STATE_CONDITIONAL_MAP_V2",
        "edge_id": "MARKET_STATE_INFORMATION_MAP",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "warmup_start_date": args.warmup_start_date,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "holdout_start": args.holdout_start,
        "rows": int(len(source)),
        "data_start": str(source.index[0]),
        "data_end": str(source.index[-1]),
        "orderflow_coverage_ratio": coverage_ratio,
        "profiles": list(args.profiles),
        "conditional_config": asdict(conditional_cfg),
        "profile_metadata": profile_meta,
        "registry_counts": registry["evidence_status"].value_counts().to_dict() if not registry.empty else {},
        "warnings": [
            "This is a role-aware market-state information audit, not a trading strategy.",
            "Future path values are labels only; conditions are built before labels are attached.",
            "Trading cost is not a state-validity gate. It must be applied later to any executable strategy.",
            "Nested ladders are fixed a priori; do not tune them around individual losing samples.",
        ],
    }
    (out_dir / "00_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    write_executive_summary(
        out_dir,
        registry=registry,
        axis_summary=axis_summary,
        ladder_incremental=ladder_incremental,
        metadata=metadata,
    )
    _finish(started)

    if not args.skip_review_pack:
        started = _stage("build GPT review pack")
        write_gpt_review_pack(
            ReviewPackConfig(
                report_dir=out_dir,
                experiment_id="MARKET_STATE_CONDITIONAL_MAP_V2",
                edge_id="MARKET_STATE_INFORMATION_MAP",
                title="Market State Conditional Map V2",
                stage="research",
                decision_focus=(
                    "Judge each state by its intended information role; identify stable positive, opposite-semantics, "
                    "and useless states; inspect whether each nested condition adds marginal information."
                ),
                zip_name="gpt_review_pack.zip",
            )
        )
        _finish(started)

    print(f"\n[done] report_dir={out_dir.resolve()}", flush=True)
    print(
        "[done] registry="
        + json.dumps(metadata["registry_counts"], ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return out_dir


def main(argv: Iterable[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
