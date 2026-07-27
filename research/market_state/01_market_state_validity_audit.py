#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Full causal validity audit for Market State Map V0.3.

Purpose
-------
Determine whether static state starts and state transitions have stable forward
path separation after next-open execution and the project's default 0.11%
round-trip cost.  This is a research audit, not a deployable strategy.

Data
----
Uses existing local OKX rich Trade Bars.  No Order Book, OI, Funding or new
external data is required for this first validity pass.
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
from src.market_state.validity_audit import (
    ValidityAuditConfig,
    build_event_definitions,
    build_forward_path_frame,
    build_naive_fixed_horizon_trades,
    build_profile_stability,
    build_verdict,
    extract_event_rows,
    summarize_breakdowns,
    summarize_event_rows,
)
from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack
from src.utils.report import print_full_report


DEFAULT_OUT_DIR = Path("data/reports/research/market_state/01_market_state_validity_audit")


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
    parser.add_argument("--event-cooldown-bars", type=int, default=5)
    parser.add_argument("--transition-lookback-bars", type=int, default=15)
    parser.add_argument("--minimum-events", type=int, default=80)
    parser.add_argument("--round-trip-cost", type=float, default=0.0011)
    parser.add_argument("--diagnostic-horizon", type=int, default=60)
    parser.add_argument("--max-sample-rows-per-group", type=int, default=500)
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


def _print_stage(label: str) -> float:
    print(f"\n[stage] {label}", flush=True)
    return time.perf_counter()


def _finish_stage(started: float) -> None:
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
            "No Trade Bar data returned. Check local data coverage or rerun with --no-local-only to build missing days."
        )
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def restrict_event_definitions(definitions, available_time: pd.Series, start_date: str, end_date: str):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    audit_mask = pd.to_datetime(available_time).between(start, end)
    return [replace(definition, mask=definition.mask & audit_mask) for definition in definitions]


def save_text_summary(
    out_dir: Path,
    *,
    verdict: dict[str, object],
    stability: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    lines = [
        "# Market State Validity Audit",
        "",
        f"Decision: {verdict.get('decision')}",
        f"Reason: {verdict.get('reason')}",
        f"Trend-start direction valid: {verdict.get('trend_start_direction_valid')}",
        f"Transition valid: {verdict.get('transition_valid')}",
        f"Context valid: {verdict.get('context_valid')}",
        f"Robust candidates: {verdict.get('robust_candidate_count')}",
        "",
        "This audit uses next-bar open entry and future bars only as labels.",
        "Static state/event metrics are diagnostic and are not a deployable strategy.",
        "",
        "## Data",
        f"Rows: {metadata.get('rows')}",
        f"Range: {metadata.get('data_start')} -> {metadata.get('data_end')}",
        f"Order-flow coverage: {metadata.get('orderflow_coverage_ratio')}",
        "",
        "## Top robust rows",
    ]
    robust = stability.loc[stability.get("robust_flag", False)].copy() if not stability.empty else pd.DataFrame()
    if robust.empty:
        lines.append("No robust rows passed all checks.")
    else:
        for row in robust.sort_values("mean_excess_return", ascending=False).head(20).itertuples(index=False):
            lines.append(
                f"- {row.profile} | {row.event_name} | h={row.horizon_bars} | "
                f"events={row.events} | net={row.mean_net_return:.6f} | "
                f"excess={row.mean_excess_return:.6f} | PF={row.net_profit_factor:.3f}"
            )
    (out_dir / "00_EXECUTIVE_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_causal_checks(frame: pd.DataFrame, event_rows: pd.DataFrame) -> pd.DataFrame:
    available = pd.to_datetime(frame["available_time"])
    index_series = pd.Series(pd.DatetimeIndex(frame.index), index=frame.index)
    checks = [
        {
            "check": "available_time_not_before_bar_timestamp",
            "passed": bool((available >= index_series).all()),
            "violations": int((available < index_series).sum()),
        },
        {
            "check": "event_entry_strictly_after_signal",
            "passed": bool(event_rows.empty or (pd.to_datetime(event_rows["entry_time"]) > pd.to_datetime(event_rows["signal_time"])).all()),
            "violations": 0 if event_rows.empty else int((pd.to_datetime(event_rows["entry_time"]) <= pd.to_datetime(event_rows["signal_time"])).sum()),
        },
        {
            "check": "event_entry_not_before_available_time",
            "passed": bool(event_rows.empty or (pd.to_datetime(event_rows["entry_time"]) >= pd.to_datetime(event_rows["available_time"])).all()),
            "violations": 0 if event_rows.empty else int((pd.to_datetime(event_rows["entry_time"]) < pd.to_datetime(event_rows["available_time"])).sum()),
        },
    ]
    return pd.DataFrame(checks)


def write_diagnostic_reports(
    event_rows: pd.DataFrame,
    source_frame: pd.DataFrame,
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    specs = [
        (
            "trend_state_start_diagnostic",
            {"trend_up_start", "trend_down_start"},
        ),
        (
            "trade_context_watch_diagnostic",
            {
                "context_long_reversal_watch",
                "context_short_reversal_watch",
                "context_long_continuation_watch",
                "context_short_continuation_watch",
            },
        ),
    ]
    total_days = max(1.0, (source_frame.index[-1] - source_frame.index[0]).total_seconds() / 86400.0)
    for report_name, event_names in specs:
        trades, final_capital = build_naive_fixed_horizon_trades(
            event_rows,
            event_names=event_names,
            horizon_bars=int(args.diagnostic_horizon),
            initial_capital=10_000.0,
        )
        if not trades:
            continue
        print_full_report(
            trades,
            source_frame,
            10_000.0,
            final_capital,
            f"DIAGNOSTIC_NOT_STRATEGY_{report_name}",
            total_days,
            False,
            symbol=args.symbol,
            report_dir=str(out_dir / "diagnostic_reports"),
        )


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_cfg = ValidityAuditConfig(
        horizons_bars=tuple(sorted(set(int(v) for v in args.horizons))),
        event_cooldown_bars=int(args.event_cooldown_bars),
        transition_lookback_bars=int(args.transition_lookback_bars),
        round_trip_cost=float(args.round_trip_cost),
        minimum_events=int(args.minimum_events),
        holdout_start=str(args.holdout_start) if args.holdout_start else None,
    )
    audit_cfg.validate()

    started = _print_stage("load existing rich OKX Trade Bars")
    source = load_trade_bars(args)
    _finish_stage(started)
    print(
        f"[data] rows={len(source):,} range={source.index[0]} -> {source.index[-1]} "
        f"columns={len(source.columns)} local_only={args.local_only}",
        flush=True,
    )

    bar_duration = timeframe_to_timedelta(args.timeframe)
    all_summaries: list[pd.DataFrame] = []
    all_yearly: list[pd.DataFrame] = []
    all_period: list[pd.DataFrame] = []
    sampled_event_rows: list[pd.DataFrame] = []
    trap_examples_parts: list[pd.DataFrame] = []
    diagnostic_event_rows: list[pd.DataFrame] = []
    profile_metadata: dict[str, object] = {}
    base_state_frame: pd.DataFrame | None = None

    diagnostic_names = {
        "trend_up_start",
        "trend_down_start",
        "context_long_reversal_watch",
        "context_short_reversal_watch",
        "context_long_continuation_watch",
        "context_short_continuation_watch",
    }

    for profile in args.profiles:
        started = _print_stage(f"market-state engine profile={profile}")
        state_cfg = profile_config(profile)
        bundle = MarketStateDataBundle.from_frame(
            source,
            source=f"okx_trade_bar:{args.symbol}:{args.timeframe}",
            timestamp_semantics="bar_start",
            bar_duration=bar_duration,
            metadata={"profile": profile},
        )
        result = MarketStateEngine(state_cfg).compute(bundle)
        path_frame = build_forward_path_frame(result.frame, audit_cfg)
        definitions = build_event_definitions(
            path_frame,
            transition_lookback_bars=audit_cfg.transition_lookback_bars,
            event_cooldown_bars=audit_cfg.event_cooldown_bars,
        )
        definitions = restrict_event_definitions(definitions, path_frame["available_time"], args.start_date, args.end_date)
        rows = extract_event_rows(path_frame, definitions, audit_cfg, profile=profile)
        profile_summary = summarize_event_rows(rows, audit_cfg)
        profile_yearly, profile_period = summarize_breakdowns(rows)
        all_summaries.append(profile_summary)
        all_yearly.append(profile_yearly)
        all_period.append(profile_period)

        max_sample = max(0, int(args.max_sample_rows_per_group))
        if max_sample > 0 and not rows.empty:
            sample = (
                rows.sort_values(["profile", "event_name", "horizon_bars", "signal_time"])
                .groupby(["profile", "event_name", "horizon_bars"], group_keys=False)
                .head(max_sample)
            )
            sampled_event_rows.append(sample)
        if not rows.empty:
            traps = rows.loc[rows["trap_flag"]].nsmallest(1_000, "net_return")
            trap_examples_parts.append(traps)
        if profile == "base":
            base_state_frame = result.frame
            diagnostic_event_rows.append(
                rows.loc[
                    rows["event_name"].isin(diagnostic_names)
                    & rows["horizon_bars"].eq(int(args.diagnostic_horizon))
                ].copy()
            )
        profile_metadata[profile] = {
            "state_config": asdict(state_cfg),
            "state_metadata": result.metadata,
            "event_rows": int(len(rows)),
        }
        _finish_stage(started)
        print(f"[profile] {profile} event_rows={len(rows):,}", flush=True)
        del rows, path_frame, result, bundle

    summary = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    yearly = pd.concat(all_yearly, ignore_index=True) if all_yearly else pd.DataFrame()
    period = pd.concat(all_period, ignore_index=True) if all_period else pd.DataFrame()
    event_samples = pd.concat(sampled_event_rows, ignore_index=True) if sampled_event_rows else pd.DataFrame()
    diagnostic_rows = pd.concat(diagnostic_event_rows, ignore_index=True) if diagnostic_event_rows else pd.DataFrame()
    trap_examples = pd.concat(trap_examples_parts, ignore_index=True) if trap_examples_parts else pd.DataFrame()
    if summary.empty:
        raise RuntimeError("No eligible state events were produced in the requested audit range.")

    started = _print_stage("summaries, yearly/holdout stability and verdict")
    stability = build_profile_stability(summary, yearly, period)
    verdict = dict(build_verdict(stability))
    _finish_stage(started)

    started = _print_stage("write report artifacts")
    event_samples.to_csv(out_dir / "01_event_samples.csv", index=False)
    summary.to_csv(out_dir / "02_event_path_summary.csv", index=False)
    yearly.to_csv(out_dir / "03_yearly_breakdown.csv", index=False)
    period.to_csv(out_dir / "04_pre_holdout_vs_holdout.csv", index=False)
    stability.to_csv(out_dir / "05_profile_stability_and_verdict.csv", index=False)

    base_frame = base_state_frame if base_state_frame is not None else source
    causal = build_causal_checks(base_frame, event_samples)
    causal.to_csv(out_dir / "06_causal_audit.csv", index=False)

    trap_examples = trap_examples.sort_values("net_return").head(2_000) if not trap_examples.empty else trap_examples
    trap_examples.to_csv(out_dir / "07_top_bottom_trap_examples.csv", index=False)

    coverage_ratio = float(base_frame.get("orderflow_available", pd.Series(False, index=base_frame.index)).fillna(False).mean())
    metadata = {
        "title": "Market State Validity Audit",
        "experiment_id": "MARKET_STATE_VALIDITY_AUDIT_V1",
        "edge_id": "MARKET_STATE_TRANSITIONS",
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
        "audit_config": asdict(audit_cfg),
        "profiles": profile_metadata,
        "verdict": verdict,
        "warnings": [
            "This is a forward-path/state validity audit, not a deployable strategy.",
            "Future highs/lows/returns are labels only and never feed state construction.",
            "Overlapping event summaries are diagnostic; use the fixed-horizon non-overlap reports only as a naive cost sanity check.",
        ],
    }
    (out_dir / "00_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "08_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    save_text_summary(out_dir, verdict=verdict, stability=stability, metadata=metadata)
    _finish_stage(started)

    started = _print_stage("diagnostic next-open fixed-horizon reports")
    write_diagnostic_reports(diagnostic_rows, source.loc[pd.Timestamp(args.start_date):pd.Timestamp(args.end_date)], args, out_dir)
    _finish_stage(started)

    if not args.skip_review_pack:
        started = _print_stage("build GPT review pack")
        write_gpt_review_pack(
            ReviewPackConfig(
                report_dir=out_dir,
                experiment_id="MARKET_STATE_VALIDITY_AUDIT_V1",
                edge_id="MARKET_STATE_TRANSITIONS",
                title="Market State Validity Audit",
                stage="research",
                decision_focus="Reject static state labels if they do not beat matched baselines; promote only robust transition/context candidates to sequential backtest.",
                zip_name="gpt_review_pack.zip",
            )
        )
        _finish_stage(started)

    print(f"\n[done] report_dir={out_dir.resolve()}", flush=True)
    print(f"[done] decision={verdict.get('decision')} reason={verdict.get('reason')}", flush=True)
    return out_dir


def main(argv: Iterable[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
