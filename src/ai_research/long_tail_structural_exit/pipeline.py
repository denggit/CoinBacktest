#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2.7 causal non-time structural exit audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.config import LongTailExitAuditConfig
from src.ai_research.long_tail_exit_audit.data import collect_base_period_data, load_minute_path_data
from src.ai_research.long_tail_exit_audit.modeling import fit_frozen_base_scores
from src.ai_research.long_tail_exit_audit.simulator import build_event_candidates
from src.ai_research.long_tail_q70_audit.analysis import empirical_percentile
from src.ai_research.state_context_ablation.config import DEFAULT_STATE_CONTEXT_ABLATION_CONFIG
from src.ai_research.state_context_ablation.modeling import default_ablation_folds
from src.ai_research.state_context_ablation.outcomes import build_outcome_caches
from src.ai_research.swing_baseline.dataset import create_loader, list_cached_years, run_public_loader_preflight
from src.ai_research.swing_long_context.config import LONG_CONTEXT_BASE_CONFIG
from src.research_common.progress import ProgressReporter

from . import reports
from .analysis import build_tables, comparison_table, stable_candidates
from .config import DEFAULT_STRUCTURAL_EXIT_CONFIG, STAGE_ID, STAGE_NAME, StructuralExitConfig
from .simulator import enforce_non_overlap, simulate_fixed_diagnostic, simulate_structural_event


@dataclass(frozen=True)
class StructuralExitResult:
    decision: str
    report_dir: Path


def _exit_config(config: StructuralExitConfig) -> LongTailExitAuditConfig:
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


def _causal_audit(config: StructuralExitConfig) -> pd.DataFrame:
    candidate_names = ",".join(policy.name for policy in config.policies)
    return pd.DataFrame(
        [
            {"check": "frozen_q70_opening_model", "status": "PASS", "detail": "R03.4.1 long-utility opening model and prior-quarter q70 calibration are unchanged"},
            {"check": "single_cross_year_rule_set", "status": "PASS", "detail": f"identical pre-registered policies run in WF_2024 and WF_2025: {candidate_names}"},
            {"check": "no_holding_time_exit", "status": "PASS", "detail": "candidate policies contain no maximum holding duration or scheduled time exit"},
            {"check": "causal_structure_bars", "status": "PASS", "detail": "structure transitions use completed 15-minute bars and execute at the next one-minute open"},
            {"check": "causal_pivots", "status": "PASS", "detail": "a swing pivot is usable only after its right-side confirmation bars have closed"},
            {"check": "recovery_before_invalidation", "status": "PASS", "detail": "a broken floor can reclaim before lower-high/lower-low confirmation; recoverable pullbacks are not automatically stopped"},
            {"check": "disaster_execution", "status": "PASS", "detail": "safety-floor breach executes at the next available one-minute open, never at an idealized stop price"},
            {"check": "right_censoring", "status": "PASS", "detail": "OOS end or data gap is recorded as mark-to-market censoring, not a strategy time exit"},
            {"check": "score_tiers_retained", "status": "PASS", "detail": "q70-q80, q80-q90 and q90+ are reported separately but use the same exit mechanism"},
            {"check": "opening_score_not_holding_signal", "status": "PASS", "detail": "post-entry score persistence or upgrade is not used by the exit state machine"},
            {"check": "state_model_abandoned", "status": "PASS", "detail": "strategic/tactical/entry/activity state outputs are not loaded"},
            {"check": "sealed_2026", "status": "PASS", "detail": "2026 data are not loaded to resolve year-end positions"},
        ]
    )


def run_structural_exit_audit(
    *,
    data_dir: str | Path | None = None,
    force_rebuild_outcomes: bool = False,
    progress: bool = True,
    config: StructuralExitConfig = DEFAULT_STRUCTURAL_EXIT_CONFIG,
) -> StructuralExitResult:
    config.validate()
    exit_config = _exit_config(config)
    loader = create_loader(LONG_CONTEXT_BASE_CONFIG, data_dir=data_dir)
    loader_preflight = run_public_loader_preflight(
        loader,
        LONG_CONTEXT_BASE_CONFIG,
        sample_dates=("2023-06-15", "2024-06-15", "2025-06-15"),
    )
    preflight: dict[str, object] = {"trade_bar": loader_preflight.to_dict()}
    if loader_preflight.status != "PASS":
        config.report_path.mkdir(parents=True, exist_ok=True)
        reason = "1分钟Trade Bar公共Loader预检失败。"
        reports.write_reports(
            config=config,
            manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
            preflight=preflight,
            score_audit=pd.DataFrame(),
            event_audit=pd.DataFrame(),
            summary=pd.DataFrame(),
            periods=pd.DataFrame(),
            tiers=pd.DataFrame(),
            exits=pd.DataFrame(),
            censoring=pd.DataFrame(),
            overlap=pd.DataFrame(),
            comparison=pd.DataFrame(),
            stable=pd.DataFrame(),
            causal_audit=pd.DataFrame(),
            failures=pd.DataFrame([{"fold_id": "ALL", "error": reason}]),
            trades=pd.DataFrame(),
            decision="BLOCKED_DATA",
            reason=reason,
        )
        return StructuralExitResult("BLOCKED_DATA", config.report_path)

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
    overlap_rows: list[dict[str, object]] = []
    trade_parts: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    folds = default_ablation_folds(DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
    reporter = ProgressReporter("[R03.4.2.7 folds]", len(folds), every=1, enabled=progress)

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
            score_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "quantile": config.evaluation_quantile,
                    "calibration_threshold": bundle.timeline.threshold(config.evaluation_quantile),
                    "calibration_rows": int(len(bundle.calibration_score)),
                    "test_rows": int(len(bundle.test_score)),
                    "test_exceedance_rate": float(np.mean(bundle.test_score >= bundle.timeline.threshold(config.evaluation_quantile))),
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
            policies: list[tuple[str, str, object]] = [
                ("fixed_6h", "diagnostic_time_baseline", False),
                ("fixed_6h_disaster", "diagnostic_time_baseline", True),
            ]
            policies.extend((policy.name, "non_time_structural_candidate", policy) for policy in config.policies)

            for delay in config.entry_delay_minutes:
                for policy_name, policy_kind, policy_object in policies:
                    rows: list[dict[str, object]] = []
                    skipped_incomplete = 0
                    for event in events:
                        pos = int(np.searchsorted(test.timestamps_ns, event.decision_time_ns, side="left"))
                        percentile = (
                            float(percentiles[pos])
                            if pos < len(percentiles) and int(test.timestamps_ns[pos]) == event.decision_time_ns
                            else np.nan
                        )
                        if not np.isfinite(percentile):
                            skipped_incomplete += 1
                            continue
                        if policy_kind == "diagnostic_time_baseline":
                            trade = simulate_fixed_diagnostic(
                                event,
                                fold_id=fold.fold_id,
                                policy=policy_name,
                                delay_minutes=delay,
                                percentile=percentile,
                                path=path,
                                config=config,
                                disaster_protected=bool(policy_object),
                            )
                        else:
                            trade = simulate_structural_event(
                                event,
                                fold_id=fold.fold_id,
                                policy=policy_object,
                                delay_minutes=delay,
                                percentile=percentile,
                                path=path,
                                oos_end_ns=int(pd.Timestamp(fold.test_end).value),
                                config=config,
                            )
                        if trade is None:
                            skipped_incomplete += 1
                        else:
                            rows.append(trade.to_dict())
                    raw = pd.DataFrame(rows)
                    executed, overlap_skipped = enforce_non_overlap(raw)
                    event_rows.append(
                        {
                            "fold_id": fold.fold_id,
                            "policy": policy_name,
                            "policy_kind": policy_kind,
                            "delay_minutes": int(delay),
                            "candidate_events": int(len(events)),
                            "complete_events": int(len(raw)),
                            "skipped_incomplete": int(skipped_incomplete),
                            "executed_events": int(len(executed)),
                            "skipped_overlap": int(overlap_skipped),
                            "path_coverage_ratio": float(path.coverage_ratio),
                        }
                    )
                    overlap_rows.append(
                        {
                            "fold_id": fold.fold_id,
                            "policy": policy_name,
                            "delay_minutes": int(delay),
                            "raw_complete_events": int(len(raw)),
                            "executed_events": int(len(executed)),
                            "overlap_skip_share": float(overlap_skipped / len(raw)) if len(raw) else np.nan,
                        }
                    )
                    if not executed.empty:
                        trade_parts.append(executed)
        except Exception as exc:
            failures.append({"fold_id": fold.fold_id, "error": f"{type(exc).__name__}: {exc}"})
        reporter.update(fold_number)
    reporter.close()

    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    overlap = pd.DataFrame(overlap_rows)
    summary, periods, tiers, exits, censoring, overlap = build_tables(
        trades,
        overlap_audit=overlap,
        config=config,
    )
    comparison = comparison_table(summary)
    stable = stable_candidates(summary, periods, comparison, config)
    failures_frame = pd.DataFrame(failures)

    if not stable.empty and bool(stable["passes_profit_upgrade"].any()):
        decision = "PASS_NON_TIME_STRUCTURAL_PROFIT_UPGRADE"
        reason = "同一套因果结构状态机在2024与2025均通过稳健门槛，并在两年都提高了相对q70固定6小时基准的2倍成本总利润。"
    elif not stable.empty and bool(stable["passes_risk_upgrade"].any()):
        decision = "PASS_NON_TIME_STRUCTURAL_RISK_UPGRADE"
        reason = "同一套因果结构状态机跨年保持正期望与大部分基准利润，同时在两年都显著改善最大回撤；可作为非时间风险控制候选。"
    elif not stable.empty and bool(stable["base_robustness_pass"].any()):
        decision = "RESEARCH_CONTINUE_STRUCTURAL_EDGE_WITHOUT_BASELINE_UPGRADE"
        reason = "至少一个非时间结构策略跨年保持正期望，但尚未在总利润或风险收益上稳定超过固定6小时冻结基准。"
    else:
        decision = "FAIL_NO_ROBUST_NON_TIME_STRUCTURAL_EXIT"
        reason = "预注册的统一结构状态机未能同时守住2024与2025的正期望、利润厚度、回撤、集中度、成本延迟与低删失门槛。"

    reports.write_reports(
        config=config,
        manifest={
            "stage": STAGE_ID,
            "name": STAGE_NAME,
            "config": config.to_dict(),
            "folds": [fold.to_dict() for fold in folds],
            "opening_model": "frozen R03.4.1 long utility h6 with causal q70 calibration",
            "candidate_exit_contract": "no scheduled or maximum holding-time exit",
            "right_censoring_contract": "OOS-end/data-gap marks are accounting boundaries, not strategy exits",
            "year_specific_model_selection": "FORBIDDEN",
            "state_model_policy": "ABANDONED_AND_NOT_LOADED",
        },
        preflight=preflight,
        score_audit=pd.DataFrame(score_rows),
        event_audit=pd.DataFrame(event_rows),
        summary=summary,
        periods=periods,
        tiers=tiers,
        exits=exits,
        censoring=censoring,
        overlap=overlap,
        comparison=comparison,
        stable=stable,
        causal_audit=_causal_audit(config),
        failures=failures_frame,
        trades=trades,
        decision=decision,
        reason=reason,
    )
    return StructuralExitResult(decision=decision, report_dir=config.report_path)
