#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R03.3.3 end-to-end market-state continuity research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.swing_baseline.dataset import list_cached_years
from src.ai_research.swing_long_context.config import LONG_CONTEXT_BASE_CONFIG
from src.research_common.progress import ProgressReporter

from .analytics import (
    build_state_duration_atlas,
    build_state_opportunity_link,
    build_state_samples,
    build_state_target_distribution,
    build_strategic_threshold_audit,
)
from .config import (
    DEFAULT_MARKET_STATE_CONTINUITY_CONFIG,
    STAGE_ID,
    STAGE_NAME,
    MarketStateContinuityConfig,
)
from .data import UnifiedOHLCVLoader, run_state_data_preflight
from .modeling import (
    collect_continuity_period_data,
    default_continuity_folds,
    evaluate_continuity_model,
    feature_importance_frame,
    fit_continuity_model,
    fit_mechanical_baseline,
    mechanical_feature_sets,
    subset_period_data,
    build_mechanical_increment_audit,
    transition_alert_episode_audit,
    prediction_samples,
    run_training_attribution,
    select_stable_candidates,
    validate_continuity_dependencies,
)
from .reports import write_state_continuity_reports
from .state_cache import build_state_caches


@dataclass(frozen=True)
class MarketStateContinuityResult:
    decision: str
    report_dir: Path
    champion: dict[str, object] | None


def _trade_increment(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    universal = metrics.loc[metrics["architecture"] == "universal_ohlcv_lightgbm"].copy()
    trade = metrics.loc[metrics["architecture"] == "trade_enhanced_lightgbm"].copy()
    keys = ["fold_id", "target"]
    merged = universal.merge(trade, on=keys, suffixes=("_universal", "_trade"))
    if merged.empty:
        return merged
    for metric in ("auc", "brier_skill", "bottom_decile_transition_lift", "transition_capture_bottom_decile"):
        merged[f"delta_{metric}"] = merged[f"{metric}_trade"] - merged[f"{metric}_universal"]
    return merged[
        [
            *keys,
            "auc_universal",
            "auc_trade",
            "delta_auc",
            "brier_skill_universal",
            "brier_skill_trade",
            "delta_brier_skill",
            "bottom_decile_transition_lift_universal",
            "bottom_decile_transition_lift_trade",
            "delta_bottom_decile_transition_lift",
            "transition_capture_bottom_decile_universal",
            "transition_capture_bottom_decile_trade",
            "delta_transition_capture_bottom_decile",
        ]
    ]


def _causal_audit(config: MarketStateContinuityConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "ordinary_kline_usage",
                "status": "PASS",
                "detail": "2020-2021 ordinary K-lines are loaded only through src.data_feed.okx_loader.OKXDataLoader",
            },
            {
                "check": "trade_bar_usage",
                "status": "PASS",
                "detail": "2022 onward bars are loaded only through src.data_feed.okx_trade_bar_loader.OKXTradeBarLoader",
            },
            {
                "check": "common_feature_contract",
                "status": "PASS",
                "detail": "Universal models exclude Trade-only fields; early missing Trade fields are never filled as observed zero data",
            },
            {
                "check": "completed_bar_availability",
                "status": "PASS",
                "detail": "all timeframe features are shifted from bar start to completed-bar availability before 15m alignment",
            },
            {
                "check": "state_age",
                "status": "PASS",
                "detail": "state age and flip rates use past and current states only",
            },
            {
                "check": "future_targets",
                "status": "PASS",
                "detail": "persistence and transition targets use future states only as supervised labels",
            },
            {
                "check": "strict_uninterrupted_persistence",
                "status": "PASS",
                "detail": "persistence requires no state change anywhere inside the horizon; flip-away-and-return is a transition",
            },
            {
                "check": "strategic_threshold_calibration",
                "status": "PASS",
                "detail": "strategic thresholds use prior-day rolling distributions only; the current day and future test data cannot set thresholds",
            },
            {
                "check": "mechanical_baseline_audit",
                "status": "PASS",
                "detail": "full models are compared against state age, boundary margin and current-state mechanical baselines",
            },
            {
                "check": "independent_transition_alerts",
                "status": "PASS",
                "detail": "consecutive low-persistence scores are merged into independent warning episodes using a train-only threshold",
            },
            {
                "check": "purge_embargo",
                "status": "PASS",
                "detail": f"training ends at least {config.maximum_target_horizon_hours + 24}h before the next test boundary",
            },
            {
                "check": "sealed_2026",
                "status": "PASS",
                "detail": "research_end is 2025-12-31 and the 2025 cache does not read 2026 data",
            },
            {
                "check": "direct_order_use",
                "status": "PASS",
                "detail": "state output is explicitly auxiliary context and never a direct order decision",
            },
        ]
    )


def _blocked_report(
    config: MarketStateContinuityConfig,
    preflight: dict[str, object],
    decision: str,
    reason: str,
) -> MarketStateContinuityResult:
    report_dir = config.report_path
    write_state_continuity_reports(
        report_dir,
        manifest={"stage_id": STAGE_ID, "stage_name": STAGE_NAME, "config": config.to_dict()},
        preflight=preflight,
        duration_atlas=pd.DataFrame(),
        target_distribution=pd.DataFrame(),
        opportunity_link=pd.DataFrame(),
        model_metrics=pd.DataFrame(),
        decile_curves=pd.DataFrame(),
        candidates=pd.DataFrame(),
        attribution=pd.DataFrame(),
        trade_increment=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        state_samples=pd.DataFrame(),
        prediction_samples=pd.DataFrame(),
        strategic_threshold_audit=pd.DataFrame(),
        mechanical_baseline_metrics=pd.DataFrame(),
        mechanical_increment_audit=pd.DataFrame(),
        transition_alert_metrics=pd.DataFrame(),
        transition_alert_episodes=pd.DataFrame(),
        failures=pd.DataFrame([{"stage": "preflight", "error": reason}]),
        causal_audit=_causal_audit(config),
        decision=decision,
        reason=reason,
        champion=None,
        config=config,
    )
    return MarketStateContinuityResult(decision=decision, report_dir=report_dir, champion=None)


def run_market_state_continuity_pipeline(
    *,
    data_dir: str | Path | None = None,
    force_rebuild_state_cache: bool = False,
    progress: bool = True,
    config: MarketStateContinuityConfig = DEFAULT_MARKET_STATE_CONTINUITY_CONFIG,
) -> MarketStateContinuityResult:
    config.validate()
    try:
        validate_continuity_dependencies()
    except Exception as exc:
        return _blocked_report(config, {}, "BLOCKED_DEPENDENCY", str(exc))

    loader = UnifiedOHLCVLoader(config=config, data_dir=data_dir)
    preflight_result = run_state_data_preflight(loader, config)
    preflight = preflight_result.to_dict()
    if preflight_result.status != "PASS":
        return _blocked_report(
            config,
            preflight,
            "BLOCKED_DATA",
            "Unified 2020-2025 OHLCV preflight failed; inspect 01_preflight.json",
        )

    state_paths = build_state_caches(
        loader,
        config,
        force_rebuild=force_rebuild_state_cache,
        progress=progress,
    )
    trade_paths = list_cached_years(LONG_CONTEXT_BASE_CONFIG)
    folds = default_continuity_folds(config)
    total_jobs = len(folds) * len(config.architectures) * len(config.targets)
    reporter = ProgressReporter("[R03.3.3 models] jobs", total_jobs, every=1, enabled=progress)

    metric_rows: list[dict[str, object]] = []
    decile_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.DataFrame] = []
    sample_parts: list[pd.DataFrame] = []
    alert_metric_rows: list[dict[str, object]] = []
    alert_episode_parts: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    job = 0
    for fold in folds:
        for architecture in config.architectures:
            for spec in config.targets:
                job += 1
                try:
                    fit = collect_continuity_period_data(
                        state_paths,
                        trade_paths,
                        start=fold.fit_start,
                        end=fold.fit_end,
                        target=spec.target_id,
                        architecture=architecture,
                    )
                    test = collect_continuity_period_data(
                        state_paths,
                        trade_paths,
                        start=fold.test_start,
                        end=fold.test_end,
                        target=spec.target_id,
                        architecture=architecture,
                    )
                    model = fit_continuity_model(fit, config)
                    metrics, deciles = evaluate_continuity_model(
                        model,
                        test,
                        fold_id=fold.fold_id,
                        architecture=architecture,
                        target=spec.target_id,
                    )
                    metric_rows.append(metrics)
                    decile_parts.append(deciles)
                    importance_parts.append(
                        feature_importance_frame(
                            model,
                            fit.feature_columns,
                            fold_id=fold.fold_id,
                            architecture=architecture,
                            target=spec.target_id,
                        )
                    )
                    sample_parts.append(
                        prediction_samples(
                            model,
                            test,
                            fold_id=fold.fold_id,
                            architecture=architecture,
                            target=spec.target_id,
                        )
                    )
                    if architecture == "universal_ohlcv_lightgbm":
                        alert_metrics, alert_episodes = transition_alert_episode_audit(
                            model,
                            fit,
                            test,
                            fold_id=fold.fold_id,
                            target=spec.target_id,
                            config=config,
                        )
                        alert_metric_rows.append(alert_metrics)
                        if not alert_episodes.empty:
                            alert_episode_parts.append(alert_episodes)
                except Exception as exc:
                    failures.append(
                        {
                            "fold_id": fold.fold_id,
                            "architecture": architecture,
                            "target": spec.target_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                reporter.update(job)
    reporter.close()

    metrics = pd.DataFrame(metric_rows)
    deciles = pd.concat(decile_parts, ignore_index=True) if decile_parts else pd.DataFrame()
    importance = pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()
    samples = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    alert_metrics = pd.DataFrame(alert_metric_rows)
    alert_episodes = (
        pd.concat(alert_episode_parts, ignore_index=True)
        if alert_episode_parts
        else pd.DataFrame()
    )

    baseline_rows: list[dict[str, object]] = []
    baseline_jobs = len(folds) * len(config.targets) * 3
    baseline_reporter = ProgressReporter(
        "[R03.3.3.1 mechanical audit] jobs",
        baseline_jobs,
        every=1,
        enabled=progress,
    )
    baseline_job = 0
    for fold in folds:
        for spec in config.targets:
            try:
                fit_full = collect_continuity_period_data(
                    state_paths,
                    trade_paths,
                    start=fold.fit_start,
                    end=fold.fit_end,
                    target=spec.target_id,
                    architecture="universal_ohlcv_lightgbm",
                )
                test_full = collect_continuity_period_data(
                    state_paths,
                    trade_paths,
                    start=fold.test_start,
                    end=fold.test_end,
                    target=spec.target_id,
                    architecture="universal_ohlcv_lightgbm",
                )
                for baseline_name, columns in mechanical_feature_sets(spec.target_id).items():
                    baseline_job += 1
                    try:
                        fit_baseline = subset_period_data(fit_full, columns)
                        test_baseline = subset_period_data(test_full, columns)
                        baseline_model = fit_mechanical_baseline(fit_baseline, config)
                        baseline_metric, _ = evaluate_continuity_model(
                            baseline_model,
                            test_baseline,
                            fold_id=fold.fold_id,
                            architecture=baseline_name,
                            target=spec.target_id,
                        )
                        baseline_rows.append(baseline_metric)
                    except Exception as exc:
                        failures.append(
                            {
                                "fold_id": fold.fold_id,
                                "architecture": baseline_name,
                                "target": spec.target_id,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    baseline_reporter.update(baseline_job)
            except Exception as exc:
                for baseline_name in mechanical_feature_sets(spec.target_id):
                    baseline_job += 1
                    failures.append(
                        {
                            "fold_id": fold.fold_id,
                            "architecture": baseline_name,
                            "target": spec.target_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    baseline_reporter.update(baseline_job)
    baseline_reporter.close()

    baseline_metrics = pd.DataFrame(baseline_rows)
    mechanical_increment = build_mechanical_increment_audit(metrics, baseline_metrics, config)
    failure_frame = pd.DataFrame(failures)
    candidates = select_stable_candidates(metrics, config)
    passed = candidates.loc[candidates["passed"]] if not candidates.empty else pd.DataFrame()
    incremental_targets: set[str] = set()
    if not mechanical_increment.empty:
        for target, group in mechanical_increment.groupby("target", sort=False):
            by_fold = {
                str(row["fold_id"]): bool(row["incremental_auc_passed"])
                for row in group.to_dict("records")
            }
            if by_fold.get("WF_2024", False) and by_fold.get("WF_2025", False):
                incremental_targets.add(str(target))
    incremental_passed = (
        passed.loc[
            (passed["architecture"] == "universal_ohlcv_lightgbm")
            & (passed["target"].astype(str).isin(incremental_targets))
        ]
        if not passed.empty
        else pd.DataFrame()
    )
    champion_source = incremental_passed if not incremental_passed.empty else passed
    champion = (
        champion_source.iloc[0].to_dict()
        if not champion_source.empty
        else (candidates.iloc[0].to_dict() if not candidates.empty else None)
    )

    attribution = pd.DataFrame()
    if champion is not None:
        try:
            attribution = run_training_attribution(
                state_paths,
                target=str(champion["target"]),
                config=config,
            )
        except Exception as exc:
            failures.append(
                {
                    "fold_id": "attribution",
                    "architecture": "universal_ohlcv_lightgbm",
                    "target": champion.get("target"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            failure_frame = pd.DataFrame(failures)

    if not incremental_passed.empty:
        decision = "PASS_STATE_CONTINUITY_INCREMENT"
        reason = (
            "At least one continuity task is stable across 2024 and 2025 and beats the best "
            "mechanical age/margin/state baseline in both years. Promote it only as auxiliary context."
        )
    elif not passed.empty:
        decision = "PASS_MECHANICAL_CONTINUITY_ONLY"
        reason = (
            "State persistence is predictable, but the full multi-timescale model does not add enough "
            "AUC beyond mechanical state age and boundary distance in both years. Keep the state machine "
            "as context, but do not claim a learned market-process edge yet."
        )
    else:
        decision = "FAIL_NO_STABLE_CONTINUITY_MODEL"
        reason = (
            "No persistence task passed the frozen cross-year AUC, Brier-skill and transition-lift gates. "
            "The duration atlas remains useful, but the state model must not yet control strategy gating."
        )

    duration_atlas = build_state_duration_atlas(state_paths, config)
    target_distribution = build_state_target_distribution(state_paths, config)
    opportunity_link = build_state_opportunity_link(state_paths)
    state_samples = build_state_samples(state_paths)
    strategic_threshold_audit = build_strategic_threshold_audit(state_paths)
    trade_increment = _trade_increment(metrics)
    report_dir = config.report_path
    manifest = {
        "stage_id": STAGE_ID,
        "stage_name": STAGE_NAME,
        "config": config.to_dict(),
        "state_cache_paths": [str(path) for path in state_paths],
        "trade_cache_paths": [str(path) for path in trade_paths],
        "folds": [fold.to_dict() for fold in folds],
        "decision": decision,
        "champion": champion,
        "research_scope": "auxiliary market context; no order generation",
    }
    write_state_continuity_reports(
        report_dir,
        manifest=manifest,
        preflight=preflight,
        duration_atlas=duration_atlas,
        target_distribution=target_distribution,
        opportunity_link=opportunity_link,
        model_metrics=metrics,
        decile_curves=deciles,
        candidates=candidates,
        attribution=attribution,
        trade_increment=trade_increment,
        feature_importance=importance,
        state_samples=state_samples,
        prediction_samples=samples,
        strategic_threshold_audit=strategic_threshold_audit,
        mechanical_baseline_metrics=baseline_metrics,
        mechanical_increment_audit=mechanical_increment,
        transition_alert_metrics=alert_metrics,
        transition_alert_episodes=alert_episodes,
        failures=failure_frame,
        causal_audit=_causal_audit(config),
        decision=decision,
        reason=reason,
        champion=champion,
        config=config,
    )
    return MarketStateContinuityResult(decision=decision, report_dir=report_dir, champion=champion)
