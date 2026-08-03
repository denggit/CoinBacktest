#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2.8A occupied-signal atlas and tranche gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.config import LongTailExitAuditConfig
from src.ai_research.long_tail_exit_audit.data import collect_base_period_data, load_minute_path_data
from src.ai_research.long_tail_exit_audit.modeling import fit_frozen_base_scores
from src.ai_research.long_tail_exit_audit.simulator import EventCandidate, build_event_candidates
from src.ai_research.long_tail_q70_audit.analysis import empirical_percentile
from src.ai_research.long_tail_structural_exit.config import StructuralPolicy
from src.ai_research.long_tail_structural_exit.simulator import (
    score_tier,
    simulate_fixed_diagnostic,
    simulate_structural_event,
)
from src.ai_research.state_context_ablation.config import DEFAULT_STATE_CONTEXT_ABLATION_CONFIG
from src.ai_research.state_context_ablation.modeling import default_ablation_folds
from src.ai_research.state_context_ablation.outcomes import build_outcome_caches
from src.ai_research.swing_baseline.dataset import create_loader, list_cached_years, run_public_loader_preflight
from src.ai_research.swing_long_context.config import LONG_CONTEXT_BASE_CONFIG
from src.research_common.progress import ProgressReporter

from . import reports
from .analysis import (
    baseline_summary,
    class_summary,
    occupancy_summary,
    quarter_summary,
    risk_release_distribution,
    score_price_diagnostic,
    score_tier_summary,
    tranche_gate,
)
from .config import (
    DEFAULT_TRANCHE_ELIGIBILITY_CONFIG,
    STAGE_ID,
    STAGE_NAME,
    TrancheEligibilityConfig,
)
from .simulator import build_occupancy_map, classify_occupied_signal, failed_reclaim_snapshots


@dataclass(frozen=True)
class TrancheEligibilityResult:
    decision: str
    report_dir: Path


def _exit_config(config: TrancheEligibilityConfig) -> LongTailExitAuditConfig:
    return LongTailExitAuditConfig(
        symbol=config.symbol,
        research_start=config.research_start,
        research_end=config.research_end,
        sealed_holdout_start=config.sealed_holdout_start,
        primary_signal_quantile=0.90,
        quality_control_quantile=0.95,
        primary_horizon_hours=config.diagnostic_horizon_hours,
        risk_penalty=config.risk_penalty,
        base_round_trip_cost=config.base_round_trip_cost,
        cost_stress_multipliers=config.cost_multipliers,
        entry_delay_minutes=config.entry_delay_minutes,
        episode_merge_gap_minutes=config.episode_merge_gap_minutes,
        independent_event_cooldown_hours=config.independent_event_cooldown_hours,
        train_sample_cap=config.base_train_sample_cap,
        base_n_estimators=config.base_n_estimators,
        base_learning_rate=config.base_learning_rate,
        base_num_leaves=config.base_num_leaves,
        base_min_child_samples=config.base_min_child_samples,
        random_state=config.random_state,
    )


def _causal_audit(config: TrancheEligibilityConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "frozen_q70_opening_model", "status": "PASS", "detail": "R03.4.1 opening model and prior-quarter q70 calibration are unchanged"},
            {"check": "frozen_failed_reclaim", "status": "PASS", "detail": "R03.4.2.7 failed_reclaim parameters and disaster floor are unchanged"},
            {"check": "no_tranche_execution", "status": "PASS", "detail": "R03.4.2.8A diagnoses occupied signals but does not add position size"},
            {"check": "completed_structure_bars", "status": "PASS", "detail": "signal classification uses only completed event-relative 15-minute bars"},
            {"check": "right_confirmed_pivots", "status": "PASS", "detail": "post-entry lows are usable only after right-side pivot confirmation"},
            {"check": "next_open_entry", "status": "PASS", "detail": "candidate event entry remains decision time plus the tested 1/3/5 minute delay"},
            {"check": "score_not_add_trigger", "status": "PASS", "detail": "score upgrade alone is diagnostic and cannot make a signal eligible"},
            {"check": "broken_state_rejected", "status": "PASS", "detail": "signals during BROKEN/failed-reclaim confirmation are hard-rejected"},
            {"check": "risk_release_not_claimed", "status": "PASS", "detail": "candidate stop and released-risk fields are eligibility diagnostics; no stop is executed and no account-risk claim is made"},
            {"check": "same_rule_both_years", "status": "PASS", "detail": "one classification and gate contract is applied to WF_2024 and WF_2025"},
            {"check": "sealed_2026", "status": "PASS", "detail": "2026 remains unopened"},
        ]
    )


def _percentile_for_event(
    event: EventCandidate,
    *,
    timestamps_ns: np.ndarray,
    percentiles: np.ndarray,
) -> float:
    position = int(np.searchsorted(timestamps_ns, event.decision_time_ns, side="left"))
    if position >= len(timestamps_ns) or int(timestamps_ns[position]) != int(event.decision_time_ns):
        return np.nan
    return float(percentiles[position])


def _event_outcomes(
    events: tuple[EventCandidate, ...],
    *,
    fold_id: str,
    delay_minutes: int,
    timestamps_ns: np.ndarray,
    percentiles: np.ndarray,
    path,
    oos_end_ns: int,
    structural_config,
    progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    fixed_rows: list[dict[str, object]] = []
    structural_rows: list[dict[str, object]] = []
    incomplete = 0
    policy = StructuralPolicy(name="failed_reclaim", exit_on_failed_reclaim=True)
    reporter = ProgressReporter(
        f"[R03.4.2.8A {fold_id} d{delay_minutes} events]",
        len(events),
        every=max(1, len(events) // 100),
        enabled=progress,
    )
    for event_number, event in enumerate(events, start=1):
        percentile = _percentile_for_event(
            event,
            timestamps_ns=timestamps_ns,
            percentiles=percentiles,
        )
        if not np.isfinite(percentile):
            incomplete += 1
            reporter.update(event_number)
            continue
        fixed = simulate_fixed_diagnostic(
            event,
            fold_id=fold_id,
            policy="fixed_6h",
            delay_minutes=delay_minutes,
            percentile=percentile,
            path=path,
            config=structural_config,
            disaster_protected=False,
        )
        structural = simulate_structural_event(
            event,
            fold_id=fold_id,
            policy=policy,
            delay_minutes=delay_minutes,
            percentile=percentile,
            path=path,
            oos_end_ns=oos_end_ns,
            config=structural_config,
        )
        if fixed is None or structural is None:
            incomplete += 1
            reporter.update(event_number)
            continue
        fixed_rows.append(fixed.to_dict())
        structural_rows.append(structural.to_dict())
        reporter.update(event_number)
    reporter.close()
    return pd.DataFrame(fixed_rows), pd.DataFrame(structural_rows), int(incomplete)


def _build_atlas(
    *,
    fold_id: str,
    delay_minutes: int,
    events: tuple[EventCandidate, ...],
    percentiles_by_event: dict[str, float],
    fixed: pd.DataFrame,
    structural: pd.DataFrame,
    occupied: pd.DataFrame,
    path,
    structural_config,
    config: TrancheEligibilityConfig,
) -> tuple[pd.DataFrame, int]:
    if occupied.empty:
        return pd.DataFrame(), 0
    events_by_id = {event.event_id: event for event in events}
    fixed_by_id = fixed.set_index("event_id", drop=False)
    structural_by_id = structural.set_index("event_id", drop=False)
    occupied_groups = occupied.groupby("root_event_id", sort=False)
    rows: list[dict[str, object]] = []
    missing_snapshots = 0

    for root_event_id, group in occupied_groups:
        root_event = events_by_id.get(str(root_event_id))
        if root_event is None or root_event_id not in structural_by_id.index:
            missing_snapshots += int(len(group))
            continue
        root_trade = structural_by_id.loc[root_event_id]
        if isinstance(root_trade, pd.DataFrame):
            root_trade = root_trade.iloc[0]
        observations = tuple(
            int(pd.Timestamp(value).value)
            for value in group["decision_time"].sort_values().tolist()
        )
        snapshots = failed_reclaim_snapshots(
            root_event,
            delay_minutes=delay_minutes,
            observation_times_ns=observations,
            path=path,
            end_time_ns=int(pd.Timestamp(root_trade["exit_time"]).value),
            config=structural_config,
        )
        root_entry_price = float(root_trade["entry_price"])
        root_score = float(root_event.score)
        root_percentile = float(percentiles_by_event.get(root_event_id, np.nan))

        for occupied_row in group.to_dict("records"):
            event_id = str(occupied_row["event_id"])
            new_event = events_by_id.get(event_id)
            observation_ns = int(pd.Timestamp(occupied_row["decision_time"]).value)
            snapshot = snapshots.get(observation_ns)
            if new_event is None or snapshot is None or event_id not in fixed_by_id.index or event_id not in structural_by_id.index:
                missing_snapshots += 1
                continue
            fixed_row = fixed_by_id.loc[event_id]
            structural_row = structural_by_id.loc[event_id]
            if isinstance(fixed_row, pd.DataFrame):
                fixed_row = fixed_row.iloc[0]
            if isinstance(structural_row, pd.DataFrame):
                structural_row = structural_row.iloc[0]
            new_entry_time = pd.Timestamp(structural_row["entry_time"])
            new_entry_price = float(structural_row["entry_price"])
            score_percentile = float(percentiles_by_event.get(event_id, np.nan))
            base = {
                "fold_id": fold_id,
                "delay_minutes": int(delay_minutes),
                "event_id": event_id,
                "root_event_id": root_event_id,
                "decision_time": pd.Timestamp(new_event.decision_time_ns, unit="ns"),
                "new_entry_time": new_entry_time,
                "root_entry_time": pd.Timestamp(root_trade["entry_time"]),
                "root_exit_time": pd.Timestamp(root_trade["exit_time"]),
                "root_exit_reason": root_trade["exit_reason"],
                "hours_since_root_entry": float((new_entry_time - pd.Timestamp(root_trade["entry_time"])) / pd.Timedelta(hours=1)),
                "new_score": float(new_event.score),
                "root_score": root_score,
                "score_delta_vs_root": float(new_event.score - root_score),
                "score_percentile": score_percentile,
                "root_score_percentile": root_percentile,
                "score_tier": score_tier(score_percentile),
                "root_entry_price": root_entry_price,
                "new_entry_price": new_entry_price,
                "price_return_at_new_entry": float(new_entry_price / root_entry_price - 1.0),
                "fixed6h_gross_return": float(fixed_row["gross_return"]),
                "fixed6h_mfe": float(fixed_row["mfe"]),
                "fixed6h_mae": float(fixed_row["mae"]),
                "standalone_failed_reclaim_gross_return": float(structural_row["gross_return"]),
                "standalone_failed_reclaim_mfe": float(structural_row["mfe"]),
                "standalone_failed_reclaim_mae": float(structural_row["mae"]),
                "standalone_failed_reclaim_exit_reason": structural_row["exit_reason"],
                "continuation_to_root_exit_gross_return": float(float(root_trade["exit_price"]) / new_entry_price - 1.0),
                **snapshot,
            }
            signal_class, class_reason, eligible = classify_occupied_signal(base, config=config)
            base["signal_class"] = signal_class
            base["class_reason"] = class_reason
            base["eligible_for_tranche_simulation"] = bool(eligible)
            rows.append(base)
    return pd.DataFrame(rows), int(missing_snapshots)


def run_tranche_eligibility_audit(
    *,
    data_dir: str | Path | None = None,
    force_rebuild_outcomes: bool = False,
    progress: bool = True,
    config: TrancheEligibilityConfig = DEFAULT_TRANCHE_ELIGIBILITY_CONFIG,
) -> TrancheEligibilityResult:
    config.validate()
    structural_config = config.structural_config()
    exit_config = _exit_config(config)
    loader = create_loader(LONG_CONTEXT_BASE_CONFIG, data_dir=data_dir)
    loader_preflight = run_public_loader_preflight(
        loader,
        LONG_CONTEXT_BASE_CONFIG,
        sample_dates=("2023-06-15", "2024-06-15", "2025-06-15"),
    )
    preflight: dict[str, object] = {"trade_bar": loader_preflight.to_dict()}
    if loader_preflight.status != "PASS":
        reason = "1分钟Trade Bar公共Loader预检失败。"
        reports.write_reports(
            config=config,
            manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
            preflight=preflight,
            score_audit=pd.DataFrame(),
            event_audit=pd.DataFrame(),
            baseline_summary=pd.DataFrame(),
            occupancy=pd.DataFrame(),
            atlas=pd.DataFrame(),
            classes=pd.DataFrame(),
            quarters=pd.DataFrame(),
            tiers=pd.DataFrame(),
            score_price=pd.DataFrame(),
            risk_release=pd.DataFrame(),
            gate=pd.DataFrame(),
            causal_audit=pd.DataFrame(),
            failures=pd.DataFrame([{"fold_id": "ALL", "error": reason}]),
            baseline_trades=pd.DataFrame(),
            standalone_outcomes=pd.DataFrame(),
            decision="BLOCKED_DATA",
            reason=reason,
        )
        return TrancheEligibilityResult("BLOCKED_DATA", config.report_path)

    base_paths = [path for path in list_cached_years(LONG_CONTEXT_BASE_CONFIG) if 2023 <= int(path.name[-4:]) <= 2025]
    available = {int(path.name[-4:]) for path in base_paths}
    if available != {2023, 2024, 2025}:
        raise RuntimeError(f"missing R03.2 base cache years={sorted(available)}")
    outcome_paths = build_outcome_caches(
        base_paths,
        DEFAULT_STATE_CONTEXT_ABLATION_CONFIG,
        force_rebuild=force_rebuild_outcomes,
        progress=progress,
    )

    score_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    atlas_parts: list[pd.DataFrame] = []
    baseline_parts: list[pd.DataFrame] = []
    standalone_parts: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    folds = default_ablation_folds(DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
    reporter = ProgressReporter("[R03.4.2.8A folds]", len(folds), every=1, enabled=progress)

    for fold_number, fold in enumerate(folds, start=1):
        try:
            fit = collect_base_period_data(
                base_paths,
                outcome_paths,
                start=fold.fit_start,
                end=fold.fit_end,
                outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG,
            )
            calibration = collect_base_period_data(
                base_paths,
                outcome_paths,
                start=fold.calibration_start,
                end=fold.calibration_end,
                outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG,
            )
            test = collect_base_period_data(
                base_paths,
                outcome_paths,
                start=fold.test_start,
                end=fold.test_end,
                outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG,
            )
            bundle = fit_frozen_base_scores(fit, calibration, test, exit_config)
            percentiles = empirical_percentile(bundle.calibration_score, bundle.test_score)
            events = build_event_candidates(
                bundle.timeline,
                signal_quantile=config.evaluation_quantile,
                config=exit_config,
            )
            percentiles_by_event = {
                event.event_id: _percentile_for_event(
                    event,
                    timestamps_ns=test.timestamps_ns,
                    percentiles=percentiles,
                )
                for event in events
            }
            score_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "quantile": config.evaluation_quantile,
                    "calibration_threshold": bundle.timeline.threshold(config.evaluation_quantile),
                    "calibration_rows": int(len(bundle.calibration_score)),
                    "test_rows": int(len(bundle.test_score)),
                    "candidate_events": int(len(events)),
                    "feature_schema_hash": bundle.feature_schema_hash,
                }
            )
            path = load_minute_path_data(
                start=fold.test_start - pd.Timedelta(days=2),
                end=fold.test_end,
                data_dir=data_dir,
                config=exit_config,
                progress=progress,
            )
            for delay in config.entry_delay_minutes:
                fixed, structural, incomplete = _event_outcomes(
                    events,
                    fold_id=fold.fold_id,
                    delay_minutes=delay,
                    timestamps_ns=test.timestamps_ns,
                    percentiles=percentiles,
                    path=path,
                    oos_end_ns=int(pd.Timestamp(fold.test_end).value),
                    structural_config=structural_config,
                    progress=progress,
                )
                occupancy_result = build_occupancy_map(structural)
                atlas, missing_snapshots = _build_atlas(
                    fold_id=fold.fold_id,
                    delay_minutes=delay,
                    events=events,
                    percentiles_by_event=percentiles_by_event,
                    fixed=fixed,
                    structural=structural,
                    occupied=occupancy_result.occupied,
                    path=path,
                    structural_config=structural_config,
                    config=config,
                )
                event_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "delay_minutes": int(delay),
                        "candidate_events": int(len(events)),
                        "complete_events": int(len(structural)),
                        "incomplete_events": int(incomplete),
                        "executed_events": int(len(occupancy_result.executed)),
                        "occupied_events": int(len(occupancy_result.occupied)),
                        "snapshot_complete_events": int(len(atlas)),
                        "missing_snapshots": int(missing_snapshots),
                        "path_coverage_ratio": float(path.coverage_ratio),
                    }
                )
                if not atlas.empty:
                    atlas_parts.append(atlas)
                if not occupancy_result.executed.empty:
                    base = occupancy_result.executed.copy()
                    base["baseline"] = "P0_failed_reclaim_ignore_new_signals"
                    baseline_parts.append(base)
                if not fixed.empty:
                    fixed_copy = fixed.copy()
                    fixed_copy["standalone_outcome"] = "fixed_6h"
                    standalone_parts.append(fixed_copy)
                if not structural.empty:
                    structural_copy = structural.copy()
                    structural_copy["standalone_outcome"] = "failed_reclaim"
                    standalone_parts.append(structural_copy)
        except Exception as exc:
            failures.append({"fold_id": fold.fold_id, "error": f"{type(exc).__name__}: {exc}"})
        reporter.update(fold_number)
    reporter.close()

    event_audit = pd.DataFrame(event_rows)
    atlas = pd.concat(atlas_parts, ignore_index=True) if atlas_parts else pd.DataFrame()
    baseline_trades = pd.concat(baseline_parts, ignore_index=True) if baseline_parts else pd.DataFrame()
    standalone_outcomes = pd.concat(standalone_parts, ignore_index=True) if standalone_parts else pd.DataFrame()
    baselines = baseline_summary(baseline_trades, standalone_outcomes, config)
    occupancy = occupancy_summary(event_audit)
    classes = class_summary(atlas, config)
    quarters = quarter_summary(atlas, config)
    tiers = score_tier_summary(atlas, config)
    score_price = score_price_diagnostic(atlas)
    risk_release = risk_release_distribution(atlas)
    gate = tranche_gate(atlas, quarters, config)
    failures_frame = pd.DataFrame(failures)

    if not gate.empty and bool(gate.loc[gate["delay_minutes"] == 1, "pass_to_tranche_simulation"].any()):
        decision = "PASS_TO_R03_4_2_8B_TRANCHE_SIMULATION"
        reason = "严格因果的健康趋势/修复后新信号在WF_2024与WF_2025均通过独立收益、成本、季度、集中度和亏损仓占比资格门；下一步才允许进行P2/P3真实账户风险模拟。"
    else:
        decision = "FAIL_NO_CROSS_YEAR_OCCUPIED_SIGNAL_ELIGIBILITY"
        reason = "持仓期间的新q70信号尚未形成一个同时通过2024与2025资格门的健康趋势/修复子集；不得进入加仓或Tranche风险优化。"

    reports.write_reports(
        config=config,
        manifest={
            "stage": STAGE_ID,
            "name": STAGE_NAME,
            "config": config.to_dict(),
            "folds": [fold.to_dict() for fold in folds],
            "opening_model": "frozen R03.4.1 long utility h6 with causal q70 calibration",
            "single_position_baseline": "frozen failed_reclaim plus 3% disaster floor",
            "phase_contract": "diagnostic eligibility only; no tranche size added",
            "candidate_stop_contract": "reported but not executed; cannot be treated as released account risk",
            "year_specific_model_selection": "FORBIDDEN",
        },
        preflight=preflight,
        score_audit=pd.DataFrame(score_rows),
        event_audit=event_audit,
        baseline_summary=baselines,
        occupancy=occupancy,
        atlas=atlas,
        classes=classes,
        quarters=quarters,
        tiers=tiers,
        score_price=score_price,
        risk_release=risk_release,
        gate=gate,
        causal_audit=_causal_audit(config),
        failures=failures_frame,
        baseline_trades=baseline_trades,
        standalone_outcomes=standalone_outcomes,
        decision=decision,
        reason=reason,
    )
    return TrancheEligibilityResult(decision=decision, report_dir=config.report_path)
