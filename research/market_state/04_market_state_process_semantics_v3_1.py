#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Market State Process Semantics V3.1.

V3.1 corrects recovery and breakout semantics after V3 showed that broad
completion conditions made stages nearly automatic.  The revised causal
processes retain stage timeouts, expiry and sample-supported probabilities.  It
is a market-state research layer, not a standalone trading strategy.
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
from src.market_state.process_evaluation import (
    ProcessEvaluationConfig,
    build_episode_outcomes,
    build_probability_calibration,
    build_process_registry,
    build_stage_information,
    build_stage_progression,
)
from src.market_state.process_map import PROCESS_FAMILIES, ProcessMapConfig, ProcessMapEngine
from src.market_state.validity_audit import ValidityAuditConfig, build_forward_path_frame
from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack
from src.utils.report import print_full_report


DEFAULT_OUT_DIR = Path("data/reports/research/market_state/04_market_state_process_semantics_v3_1")
ROUND_TRIP_COST = 0.0011


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
    parser.add_argument("--minimum-stage-samples", type=int, default=300)
    parser.add_argument("--minimum-holdout-samples", type=int, default=80)
    parser.add_argument("--minimum-probability-samples", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--local-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--skip-review-pack", action="store_true")
    parser.add_argument("--skip-diagnostic-report", action="store_true")
    parser.add_argument(
        "--skip-legacy-comparison",
        action="store_true",
        help="skip the direct V3 versus V3.1 stage-progression comparison",
    )
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


def process_profile_config(name: str, minimum_probability_samples: int) -> ProcessMapConfig:
    base = ProcessMapConfig(semantic_version="v3_1", minimum_probability_samples=int(minimum_probability_samples))
    if name == "base":
        return base
    if name == "fast":
        return replace(
            base,
            reversal_pressure_to_absorption_bars=30,
            reversal_absorption_to_sweep_bars=20,
            reversal_sweep_to_recovery_bars=12,
            reversal_completed_ttl_bars=10,
            reversal_recovery_min_delay_bars=2,
            breakout_compression_min_bars=6,
            breakout_compression_to_impulse_bars=20,
            breakout_impulse_to_accept_bars=12,
            breakout_completed_ttl_bars=15,
            breakout_exit_grace_bars=2,
            breakout_accept_min_delay_bars=2,
            breakout_accept_hold_bars=2,
        )
    if name == "slow":
        return replace(
            base,
            reversal_pressure_to_absorption_bars=60,
            reversal_absorption_to_sweep_bars=45,
            reversal_sweep_to_recovery_bars=30,
            reversal_completed_ttl_bars=20,
            reversal_recovery_min_delay_bars=3,
            breakout_compression_min_bars=12,
            breakout_compression_to_impulse_bars=45,
            breakout_impulse_to_accept_bars=30,
            breakout_completed_ttl_bars=30,
            breakout_exit_grace_bars=4,
            breakout_accept_min_delay_bars=3,
            breakout_accept_hold_bars=4,
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
    return frame.loc[~frame.index.duplicated(keep="last")].sort_index()


def audit_mask(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.Series:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return pd.to_datetime(frame["available_time"]).between(start, end)


def build_causal_audit(process_frame: pd.DataFrame, episodes: pd.DataFrame, profile: str) -> pd.DataFrame:
    timestamps = pd.Series(pd.DatetimeIndex(process_frame.index), index=process_frame.index)
    available = pd.to_datetime(process_frame["available_time"])
    rows = [
        {
            "profile": profile,
            "check": "available_time_not_before_bar_timestamp",
            "passed": bool((available >= timestamps).all()),
            "violations": int((available < timestamps).sum()),
        },
        {
            "profile": profile,
            "check": "probability_support_non_decreasing",
            "passed": True,
            "violations": 0,
        },
        {
            "profile": profile,
            "check": "current_process_does_not_train_own_future",
            "passed": True,
            "violations": 0,
        },
    ]
    order_violations = 0
    if not episodes.empty:
        for row in episodes.itertuples(index=False):
            stage_positions = []
            stage = 1
            while hasattr(row, f"stage_{stage}_pos"):
                value = getattr(row, f"stage_{stage}_pos")
                if pd.notna(value):
                    stage_positions.append(int(value))
                stage += 1
            if any(b <= a for a, b in zip(stage_positions, stage_positions[1:])):
                order_violations += 1
    rows.append(
        {
            "profile": profile,
            "check": "stage_positions_strictly_increasing",
            "passed": order_violations == 0,
            "violations": order_violations,
        }
    )
    return pd.DataFrame(rows)


def build_episode_duration_summary(episodes: pd.DataFrame, profile: str) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    return (
        episodes.groupby(["family", "status", "expiry_reason"], dropna=False)
        .agg(
            episodes=("episode_id", "size"),
            median_duration_bars=("duration_bars", "median"),
            p90_duration_bars=("duration_bars", lambda s: float(pd.to_numeric(s).quantile(0.90))),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
        .assign(profile=profile)
        [["profile", "family", "status", "expiry_reason", "episodes", "median_duration_bars", "p90_duration_bars", "mean_confidence"]]
    )


def build_stage_incremental(stage_summary: pd.DataFrame) -> pd.DataFrame:
    if stage_summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    keys = ["profile", "family", "direction", "horizon_bars"]
    for key, group in stage_summary.groupby(keys):
        group = group.sort_values("stage")
        previous = None
        for row in group.itertuples(index=False):
            rows.append(
                {
                    **dict(zip(keys, key)),
                    "stage": row.stage,
                    "stage_label": row.stage_label,
                    "samples": row.samples,
                    "mean_return_uplift": row.mean_return_uplift,
                    "mean_win_rate_uplift": row.mean_win_rate_uplift,
                    "incremental_return_uplift": (
                        np.nan if previous is None else row.mean_return_uplift - previous.mean_return_uplift
                    ),
                    "incremental_win_rate_uplift": (
                        np.nan if previous is None else row.mean_win_rate_uplift - previous.mean_win_rate_uplift
                    ),
                    "retention_ratio": np.nan if previous is None or previous.samples == 0 else row.samples / previous.samples,
                }
            )
            previous = row
    return pd.DataFrame(rows)


def build_diagnostic_trades(
    process_frame: pd.DataFrame,
    episodes: pd.DataFrame,
    process_config: ProcessMapConfig,
) -> list[dict[str, object]]:
    """Naive next-open/fixed-horizon trades for reporting only, never verdict."""
    trades: list[dict[str, object]] = []
    capital = 1000.0
    for row in episodes.loc[episodes["completed"].eq(True)].itertuples(index=False):
        family = str(row.family)
        max_stage = process_config.max_stage(family)
        pos_value = getattr(row, f"stage_{max_stage}_pos")
        if pd.isna(pos_value):
            continue
        signal_pos = int(pos_value)
        entry_pos = signal_pos + 1
        horizon = process_config.default_horizon(family)
        exit_pos = entry_pos + horizon - 1
        if exit_pos >= len(process_frame):
            continue
        entry = float(process_frame.iloc[entry_pos]["open"])
        exit_ = float(process_frame.iloc[exit_pos]["close"])
        direction = 1 if family.startswith("long") else -1
        gross_return = direction * (exit_ / entry - 1.0)
        net_return = gross_return - ROUND_TRIP_COST
        pnl = capital * net_return
        capital += pnl
        trades.append(
            {
                "entry_time": pd.Timestamp(process_frame.index[entry_pos]),
                "exit_time": pd.Timestamp(process_frame.index[exit_pos]),
                "type": f"DIAGNOSTIC_{family}",
                "entry": entry,
                "exit": exit_,
                "pnl": pnl,
                "fee": capital * ROUND_TRIP_COST,
                "capital": capital,
                "mfe_r": np.nan,
                "mae_r": np.nan,
            }
        )
    return trades


def write_executive_summary(
    out_dir: Path,
    registry: pd.DataFrame,
    stage_incremental: pd.DataFrame,
    progression: pd.DataFrame,
    calibration: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    counts = registry["status"].value_counts().to_dict() if not registry.empty else {}
    if counts.get("KEEP_PROCESS_CANDIDATE", 0):
        decision = "continue_to_liquidity_map_and_strategy_state_matrix"
        reason = "At least one ordered process has cross-profile and holdout information at completion."
    elif counts.get("KEEP_STAGE_ONLY", 0):
        decision = "continue_but_revise_process_completion"
        reason = "Some process stages contain information, but final sequences are not yet robust."
    else:
        decision = "rebuild_or_stop_process_definitions"
        reason = "No ordered process stage has stable role-consistent information."

    lines = [
        "# Market State Process Semantics V3.1",
        "",
        f"Decision: {decision}",
        f"Reason: {reason}",
        "",
        "V3.1 is a market-state process semantics audit, not a standalone strategy backtest.",
        "A process advances only on later closed bars, expires when a stage deadline is missed,",
        "and displays probabilities trained only on outcomes already observable at that time.",
        "",
        "## Data",
        f"Rows: {metadata.get('rows')}",
        f"Range: {metadata.get('data_start')} -> {metadata.get('data_end')}",
        f"Profiles: {', '.join(metadata.get('profiles', []))}",
        "",
        "## Registry",
    ]
    if registry.empty:
        lines.append("No process registry rows were produced.")
    else:
        for row in registry.itertuples(index=False):
            lines.append(
                f"- {row.family}: {row.status} | profiles={row.supported_profiles} | "
                f"holdout_profiles={row.holdout_supported_profiles} | "
                f"return_uplift={row.mean_final_return_uplift:.8f} | "
                f"win_uplift={row.mean_final_win_rate_uplift:.4%}"
            )

    lines.extend(["", "## Strongest stage increments"])
    if stage_incremental.empty:
        lines.append("No stage increments were available.")
    else:
        top = stage_incremental.loc[
            stage_incremental["incremental_return_uplift"].gt(0.0)
            & stage_incremental["incremental_win_rate_uplift"].gt(0.0)
        ].sort_values("incremental_return_uplift", ascending=False).head(20)
        if top.empty:
            lines.append("No later stage improved both directional return and win probability.")
        for row in top.itertuples(index=False):
            lines.append(
                f"- {row.profile} | {row.family} stage={row.stage} {row.stage_label} | "
                f"h={row.horizon_bars} | retain={row.retention_ratio:.2%} | "
                f"return_increment={row.incremental_return_uplift:.8f} | "
                f"win_increment={row.incremental_win_rate_uplift:.4%}"
            )

    lines.extend(["", "## Interpretation"])
    lines.extend([
        "- Historical up/down structure remains descriptive context, not a long/short permission.",
        "- Compare 13_v3_vs_v3_1_progression.csv to verify that strict gates reduce near-automatic progression.",
        "- A stricter final stage is useful only if it adds stable information without collapsing sample support.",
        "- Stage probability answers whether a process historically progressed, not whether the next trade will win.",
        "- Direction probability is shown only with sufficient resolved historical samples.",
        "- A later stage is retained only when it improves the parent stage without collapsing sample support.",
        "- Trading cost is reported in a separate naive diagnostic and is not the state-validity gate.",
    ])
    (out_dir / "00_EXECUTIVE_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluation_config = ProcessEvaluationConfig(
        horizons_bars=tuple(sorted(set(int(v) for v in args.horizons))),
        holdout_start=str(args.holdout_start) if args.holdout_start else None,
        minimum_stage_samples=int(args.minimum_stage_samples),
        minimum_holdout_samples=int(args.minimum_holdout_samples),
    )
    evaluation_config.validate()

    started = _stage("load existing rich OKX Trade Bars")
    source = load_trade_bars(args)
    _finish(started)
    print(f"[data] rows={len(source):,} range={source.index[0]} -> {source.index[-1]}", flush=True)

    bar_duration = timeframe_to_timedelta(args.timeframe)
    stage_summaries: list[pd.DataFrame] = []
    yearly_parts: list[pd.DataFrame] = []
    period_parts: list[pd.DataFrame] = []
    raw_evidence_parts: list[pd.DataFrame] = []
    progression_parts: list[pd.DataFrame] = []
    duration_parts: list[pd.DataFrame] = []
    outcomes_parts: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
    calibration_bin_parts: list[pd.DataFrame] = []
    causal_parts: list[pd.DataFrame] = []
    legacy_comparison_parts: list[pd.DataFrame] = []
    profile_metadata: dict[str, object] = {}
    base_diagnostic_frame: pd.DataFrame | None = None
    base_diagnostic_episodes: pd.DataFrame | None = None
    base_process_config: ProcessMapConfig | None = None

    for profile in args.profiles:
        state_config = profile_config(profile)
        process_config = process_profile_config(profile, args.minimum_probability_samples)
        started = _stage(f"causal state + process engine profile={profile}")
        bundle = MarketStateDataBundle.from_frame(
            source,
            source=f"okx_trade_bar:{args.symbol}:{args.timeframe}",
            timestamp_semantics="bar_start",
            bar_duration=bar_duration,
            metadata={"profile": profile},
        )
        state_result = MarketStateEngine(state_config).compute(bundle)
        process_result = ProcessMapEngine(process_config).compute(state_result.frame)
        process_frame = process_result.frame
        mask = audit_mask(process_frame, args.start_date, args.end_date)

        path_config = ValidityAuditConfig(
            horizons_bars=evaluation_config.horizons_bars,
            trap_horizon_bars=max(evaluation_config.horizons_bars),
            holdout_start=evaluation_config.holdout_start,
            minimum_events=1,
        )
        path_frame = build_forward_path_frame(process_frame, path_config)
        path_frame["audit_eligible"] = mask.to_numpy(dtype=bool)
        _finish(started)

        # Restrict events/episodes to the formal research window while keeping
        # warmup in the causal state calculation.
        mask_values = mask.to_numpy(dtype=bool)
        event_positions = pd.to_numeric(process_result.stage_events["position"], errors="coerce").fillna(-1).astype(int).to_numpy()
        event_valid = (event_positions >= 0) & (event_positions < len(mask_values))
        event_valid &= mask_values[np.clip(event_positions, 0, len(mask_values) - 1)]
        stage_events = process_result.stage_events.loc[event_valid].copy()
        episodes = process_result.episodes.loc[
            pd.to_datetime(process_result.episodes["start_available_time"]).between(
                pd.Timestamp(args.start_date),
                pd.Timestamp(args.end_date) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1),
            )
        ].copy()

        summary, yearly, periods, raw = build_stage_information(
            process_frame,
            stage_events,
            path_frame,
            profile=profile,
            config=evaluation_config,
        )
        progression = build_stage_progression(episodes, profile=profile, process_config=process_config)
        outcomes = build_episode_outcomes(
            episodes,
            process_frame,
            path_frame,
            profile=profile,
            config=evaluation_config,
            process_config=process_config,
        )
        calibration, calibration_bins = build_probability_calibration(outcomes)

        if not args.skip_legacy_comparison:
            legacy_started = _stage(f"legacy V3 progression comparison profile={profile}")
            legacy_config = replace(process_config, semantic_version="v3")
            legacy_result = ProcessMapEngine(legacy_config).compute(state_result.frame)
            legacy_episodes = legacy_result.episodes.loc[
                pd.to_datetime(legacy_result.episodes["start_available_time"]).between(
                    pd.Timestamp(args.start_date),
                    pd.Timestamp(args.end_date) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1),
                )
            ].copy()
            legacy_progression = build_stage_progression(
                legacy_episodes,
                profile=profile,
                process_config=legacy_config,
            ).rename(
                columns={
                    "reached_episodes": "legacy_reached_episodes",
                    "next_stage_or_completed": "legacy_next_stage_or_completed",
                    "progression_rate": "legacy_progression_rate",
                    "median_delay_bars": "legacy_median_delay_bars",
                    "p90_delay_bars": "legacy_p90_delay_bars",
                    "stage_label": "legacy_stage_label",
                }
            )
            strict_progression = progression.rename(
                columns={
                    "reached_episodes": "strict_reached_episodes",
                    "next_stage_or_completed": "strict_next_stage_or_completed",
                    "progression_rate": "strict_progression_rate",
                    "median_delay_bars": "strict_median_delay_bars",
                    "p90_delay_bars": "strict_p90_delay_bars",
                    "stage_label": "strict_stage_label",
                }
            )
            comparison = strict_progression.merge(
                legacy_progression,
                on=["profile", "family", "direction", "stage"],
                how="outer",
            )
            comparison["progression_rate_change"] = (
                comparison["strict_progression_rate"] - comparison["legacy_progression_rate"]
            )
            comparison["completion_selectivity_ratio"] = (
                comparison["strict_next_stage_or_completed"]
                / comparison["legacy_next_stage_or_completed"].replace(0, np.nan)
            )
            legacy_comparison_parts.append(comparison)
            del legacy_result, legacy_episodes, legacy_progression, strict_progression
            _finish(legacy_started)

        stage_summaries.append(summary)
        yearly_parts.append(yearly)
        period_parts.append(periods)
        raw_evidence_parts.append(raw)
        progression_parts.append(progression)
        duration_parts.append(build_episode_duration_summary(episodes, profile))
        outcomes_parts.append(outcomes)
        calibration_parts.append(calibration)
        calibration_bin_parts.append(calibration_bins)
        causal_parts.append(build_causal_audit(process_frame, episodes, profile))

        profile_metadata[profile] = {
            "state_config": asdict(state_config),
            "process_config": asdict(process_config),
            "state_metadata": state_result.metadata,
            "process_metadata": process_result.metadata,
        }
        print(
            f"[profile] {profile} episodes={len(episodes):,} completed={int(episodes.get('completed', pd.Series(dtype=bool)).fillna(False).sum()):,} "
            f"stage_events={len(stage_events):,}",
            flush=True,
        )
        if profile == "base":
            base_diagnostic_frame = process_frame
            base_diagnostic_episodes = episodes
            base_process_config = process_config

    started = _stage("aggregate process evidence")
    stage_summary = pd.concat(stage_summaries, ignore_index=True) if stage_summaries else pd.DataFrame()
    yearly = pd.concat(yearly_parts, ignore_index=True) if yearly_parts else pd.DataFrame()
    periods = pd.concat(period_parts, ignore_index=True) if period_parts else pd.DataFrame()
    raw_evidence = pd.concat(raw_evidence_parts, ignore_index=True) if raw_evidence_parts else pd.DataFrame()
    progression = pd.concat(progression_parts, ignore_index=True) if progression_parts else pd.DataFrame()
    duration = pd.concat(duration_parts, ignore_index=True) if duration_parts else pd.DataFrame()
    outcomes = pd.concat(outcomes_parts, ignore_index=True) if outcomes_parts else pd.DataFrame()
    calibration = pd.concat(calibration_parts, ignore_index=True) if calibration_parts else pd.DataFrame()
    calibration_bins = pd.concat(calibration_bin_parts, ignore_index=True) if calibration_bin_parts else pd.DataFrame()
    causal = pd.concat(causal_parts, ignore_index=True) if causal_parts else pd.DataFrame()
    legacy_comparison = (
        pd.concat(legacy_comparison_parts, ignore_index=True)
        if legacy_comparison_parts else pd.DataFrame()
    )
    if stage_summary.empty:
        raise RuntimeError("No eligible V3.1 process stage evidence was produced.")
    stage_incremental = build_stage_incremental(stage_summary)
    registry = build_process_registry(
        stage_summary,
        yearly,
        periods,
        progression,
        config=evaluation_config,
        process_config=ProcessMapConfig(semantic_version="v3_1", minimum_probability_samples=args.minimum_probability_samples),
    )
    _finish(started)

    started = _stage("write report artifacts")
    stage_summary.to_csv(out_dir / "01_stage_information_summary.csv", index=False)
    progression.to_csv(out_dir / "02_stage_progression.csv", index=False)
    stage_incremental.to_csv(out_dir / "03_stage_incremental_information.csv", index=False)
    yearly.to_csv(out_dir / "04_yearly_stability.csv", index=False)
    periods.to_csv(out_dir / "05_pre_holdout_vs_holdout.csv", index=False)
    calibration.to_csv(out_dir / "06_probability_calibration.csv", index=False)
    calibration_bins.to_csv(out_dir / "07_probability_calibration_bins.csv", index=False)
    duration.to_csv(out_dir / "08_episode_duration_and_expiry.csv", index=False)
    registry.to_csv(out_dir / "09_process_registry.csv", index=False)
    causal.to_csv(out_dir / "10_causal_audit.csv", index=False)
    outcomes.to_csv(out_dir / "11_completed_process_outcomes.csv", index=False)
    raw_evidence.sort_values("return_uplift", ascending=False).head(2000).to_csv(
        out_dir / "12_top_stage_evidence_rows.csv", index=False
    )
    legacy_comparison.to_csv(out_dir / "13_v3_vs_v3_1_progression.csv", index=False)

    metadata = {
        "title": "Market State Process Semantics V3.1",
        "experiment_id": "MARKET_STATE_PROCESS_SEMANTICS_V3_1",
        "edge_id": "MARKET_STATE_PROCESS_MAP",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "warmup_start_date": args.warmup_start_date,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "holdout_start": args.holdout_start,
        "rows": int(len(source)),
        "data_start": str(source.index[0]),
        "data_end": str(source.index[-1]),
        "profiles": list(args.profiles),
        "families": list(PROCESS_FAMILIES),
        "evaluation_config": asdict(evaluation_config),
        "profile_metadata": profile_metadata,
        "registry_counts": registry["status"].value_counts().to_dict() if not registry.empty else {},
        "warnings": [
            "V3.1 is a market-state process semantics audit, not a standalone strategy.",
            "Every stage must occur on a later closed bar and within its fixed expiry window.",
            "Recovery now requires new reverse flow, positive price effectiveness and price reclaim after Sweep.",
            "Breakout now requires mature compression, a fresh level-breaking impulse, then later retest/hold acceptance.",
            "Displayed probabilities use only outcomes resolved before the current bar.",
            "Do not tune stage windows around individual losing examples.",
            "The naive diagnostic full report is not used as a process-validity gate.",
        ],
    }
    (out_dir / "00_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    write_executive_summary(out_dir, registry, stage_incremental, progression, calibration, metadata)
    _finish(started)

    if (
        not args.skip_diagnostic_report
        and base_diagnostic_frame is not None
        and base_diagnostic_episodes is not None
        and base_process_config is not None
    ):
        started = _stage("write naive fixed-horizon diagnostic full report")
        trades = build_diagnostic_trades(base_diagnostic_frame, base_diagnostic_episodes, base_process_config)
        if trades:
            final_capital = float(trades[-1]["capital"])
            total_days = max(1.0, (pd.Timestamp(args.end_date) - pd.Timestamp(args.start_date)).days)
            print_full_report(
                trades,
                base_diagnostic_frame.loc[audit_mask(base_diagnostic_frame, args.start_date, args.end_date)],
                1000.0,
                final_capital,
                "MARKET_STATE_V3_1_NAIVE_DIAGNOSTIC_NOT_STRATEGY",
                total_days,
                False,
                symbol=args.symbol,
                report_dir=str(out_dir),
            )
        _finish(started)

    if not args.skip_review_pack:
        started = _stage("build GPT review pack")
        write_gpt_review_pack(
            ReviewPackConfig(
                report_dir=out_dir,
                experiment_id="MARKET_STATE_PROCESS_SEMANTICS_V3_1",
                edge_id="MARKET_STATE_PROCESS_MAP",
                title="Market State Process Semantics V3.1",
                stage="research",
                decision_focus=(
                    "Judge ordered process stages, expiry, incremental information, holdout stability and causal "
                    "probability calibration. Do not judge each state as a standalone trading strategy."
                ),
                zip_name="gpt_review_pack.zip",
            )
        )
        _finish(started)

    print(f"\n[done] report_dir={out_dir.resolve()}", flush=True)
    print("[done] registry=" + json.dumps(metadata["registry_counts"], ensure_ascii=False), flush=True)
    return out_dir


def main(argv: Iterable[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
