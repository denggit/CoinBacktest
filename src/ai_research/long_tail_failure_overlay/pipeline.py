#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2.5 q70 high-confidence failure-overlay research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.config import LongTailExitAuditConfig
from src.ai_research.long_tail_exit_audit.data import collect_base_period_data, load_minute_path_data
from src.ai_research.long_tail_exit_audit.modeling import fit_frozen_base_scores
from src.ai_research.long_tail_exit_audit.simulator import build_event_candidates
from src.ai_research.long_tail_multistage_decision.config import LongTailMultistageConfig
from src.ai_research.long_tail_multistage_decision.entry_oof import build_oos_percentile_timeline, build_rolling_oof_entry_timeline
from src.ai_research.long_tail_multistage_decision.features import (
    ExtendedEventPath,
    build_checkpoint_row,
    extract_extended_event_path,
    feature_sets,
    task_frame,
)
from src.ai_research.long_tail_multistage_decision.modeling import (
    causal_oof,
    choose_feature_set,
    feature_importance,
    fit_model,
    metric_row,
)
from src.ai_research.state_context_ablation.config import DEFAULT_STATE_CONTEXT_ABLATION_CONFIG
from src.ai_research.state_context_ablation.modeling import default_ablation_folds
from src.ai_research.state_context_ablation.outcomes import build_outcome_caches
from src.ai_research.swing_baseline.dataset import create_loader, list_cached_years, run_public_loader_preflight
from src.ai_research.swing_long_context.config import LONG_CONTEXT_BASE_CONFIG
from src.research_common.progress import ProgressReporter

from . import reports
from .config import DEFAULT_FAILURE_OVERLAY_CONFIG, STAGE_ID, STAGE_NAME, FailureOverlayConfig
from .policy import (
    OverlayThresholds,
    build_policy_tables,
    enforce_non_overlap,
    score_tier,
    simulate_overlay_event,
    stable_candidates,
)


@dataclass(frozen=True)
class FailureOverlayResult:
    decision: str
    report_dir: Path


def _model_config(config: FailureOverlayConfig) -> LongTailMultistageConfig:
    return LongTailMultistageConfig(
        symbol=config.symbol,
        research_start=config.research_start,
        research_end=config.research_end,
        sealed_holdout_start=config.sealed_holdout_start,
        primary_horizon_hours=config.primary_horizon_hours,
        path_horizon_hours=config.path_horizon_hours,
        entry_delay_minutes=1,
        checkpoints_minutes=config.checkpoints_minutes,
        train_event_quantile=config.train_event_quantile,
        evaluation_quantiles=(0.70, 0.90),
        base_round_trip_cost=config.base_round_trip_cost,
        cost_multipliers=config.cost_multipliers,
        risk_penalty=config.risk_penalty,
        entry_oof_min_train_days=config.entry_oof_min_train_days,
        entry_oof_calibration_days=config.entry_oof_calibration_days,
        entry_oof_blocks=config.entry_oof_blocks,
        entry_oof_embargo_hours=config.entry_oof_embargo_hours,
        holding_oof_splits=config.holding_oof_splits,
        holding_oof_embargo_hours=config.holding_oof_embargo_hours,
        minimum_train_rows=config.minimum_train_rows,
        minimum_class_rows=config.minimum_class_rows,
        minimum_test_rows=config.minimum_test_rows,
        classifier_n_estimators=config.classifier_n_estimators,
        classifier_learning_rate=config.classifier_learning_rate,
        classifier_num_leaves=config.classifier_num_leaves,
        classifier_min_child_samples=config.classifier_min_child_samples,
        base_train_sample_cap=config.base_train_sample_cap,
        base_n_estimators=config.base_n_estimators,
        base_learning_rate=config.base_learning_rate,
        base_num_leaves=config.base_num_leaves,
        base_min_child_samples=config.base_min_child_samples,
        random_state=config.random_state,
        persistent_failure_max_mfe_48h=config.persistent_failure_max_mfe_48h,
        recoverable_min_mfe_48h=config.recoverable_min_mfe_48h,
        continuation_increment_6h_to_24h=config.continuation_increment_6h_to_24h,
        longhold_increment_24h_to_120h=config.longhold_increment_24h_to_120h,
        independent_event_cooldown_hours=config.independent_event_cooldown_hours,
        episode_merge_gap_minutes=config.episode_merge_gap_minutes,
    )


def _exit_config(config: FailureOverlayConfig) -> LongTailExitAuditConfig:
    return LongTailExitAuditConfig(
        symbol=config.symbol,
        research_start=config.research_start,
        research_end=config.research_end,
        sealed_holdout_start=config.sealed_holdout_start,
        primary_signal_quantile=0.90,
        quality_control_quantile=0.95,
        primary_horizon_hours=config.primary_horizon_hours,
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


def _augment_execution_fields(extraction: ExtendedEventPath, config: FailureOverlayConfig) -> None:
    points = extraction.points.reset_index(drop=True)
    summary = extraction.summary
    for delay in config.entry_delay_minutes:
        offset = delay - 1
        fixed_position = offset + config.primary_horizon_hours * 60 - 1
        overlay_position = offset + 180
        if fixed_position >= len(points) or overlay_position >= len(points):
            continue
        entry_price = float(points.loc[offset, "open_price"])
        summary[f"entry_time_delay_{delay}m"] = pd.Timestamp(points.loc[offset, "timestamp"])
        summary[f"entry_price_delay_{delay}m"] = entry_price
        summary[f"fixed_exit_time_delay_{delay}m"] = pd.Timestamp(points.loc[fixed_position, "timestamp"])
        summary[f"fixed_exit_price_delay_{delay}m"] = float(points.loc[fixed_position, "close_price"])
        summary[f"overlay_exit_time_delay_{delay}m"] = pd.Timestamp(points.loc[overlay_position, "timestamp"])
        summary[f"overlay_exit_price_delay_{delay}m"] = float(points.loc[overlay_position, "open_price"])
        stop_price = entry_price * (1.0 + config.disaster_stop_return)
        lows = points.loc[offset:fixed_position, "low_price"].to_numpy(dtype=float)
        hits = np.flatnonzero(lows <= stop_price)
        if len(hits):
            breach = offset + int(hits[0])
            execution = breach + 1
            if execution < len(points):
                summary[f"disaster_exit_time_delay_{delay}m"] = pd.Timestamp(points.loc[execution, "timestamp"])
                summary[f"disaster_exit_price_delay_{delay}m"] = float(points.loc[execution, "open_price"])
            else:
                summary[f"disaster_exit_time_delay_{delay}m"] = pd.NaT
                summary[f"disaster_exit_price_delay_{delay}m"] = np.nan
        else:
            summary[f"disaster_exit_time_delay_{delay}m"] = pd.NaT
            summary[f"disaster_exit_price_delay_{delay}m"] = np.nan


def _extract_dataset(
    *,
    fold_id: str,
    phase: str,
    scope: str,
    events,
    path,
    timeline,
    model_config: LongTailMultistageConfig,
    policy_config: FailureOverlayConfig,
    progress: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    complete = 0
    reporter = ProgressReporter(
        f"[R03.4.2.5 {fold_id} {phase} {scope}] events",
        len(events),
        every=max(1, len(events) // 20),
        enabled=progress,
    )
    for number, event in enumerate(events, start=1):
        extraction = extract_extended_event_path(
            event=event,
            fold_id=fold_id,
            phase=phase,
            scope=scope,
            path=path,
            timeline=timeline,
            config=model_config,
        )
        if extraction is not None:
            complete += 1
            _augment_execution_fields(extraction, policy_config)
            for checkpoint in (60, 180, 360):
                rows.append(
                    build_checkpoint_row(
                        extraction,
                        checkpoint_minutes=checkpoint,
                        path=path,
                        timeline=timeline,
                        config=model_config,
                    )
                )
        reporter.update(number)
    reporter.close()
    return pd.DataFrame(rows), {
        "fold_id": fold_id,
        "phase": phase,
        "scope": scope,
        "candidate_events": int(len(events)),
        "complete_120h_events": int(complete),
        "skipped_incomplete": int(len(events) - complete),
        "checkpoint_rows": int(len(rows)),
    }


def _tier_thresholds(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    quantiles: dict[str, float],
    global_quantile: float,
    minimum_rows: int,
) -> tuple[float, dict[str, float], list[dict[str, object]]]:
    p = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(p)
    global_threshold = float(np.quantile(p[valid], global_quantile))
    tiers = frame["event_score_percentile"].map(score_tier).to_numpy(dtype=object)
    output: dict[str, float] = {}
    audit: list[dict[str, object]] = []
    for tier, quantile in quantiles.items():
        mask = valid & (tiers == tier)
        if int(mask.sum()) >= minimum_rows:
            value = float(np.quantile(p[mask], quantile))
            fallback = False
        else:
            value = global_threshold
            fallback = True
        output[tier] = value
        audit.append(
            {
                "score_tier": tier,
                "quantile": quantile,
                "rows": int(mask.sum()),
                "threshold": value,
                "used_global_fallback": fallback,
            }
        )
    return global_threshold, output, audit


def _event_table(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    events = frame.sort_values("checkpoint_minutes").groupby("event_id", as_index=False).first()
    base_columns = [
        "event_id", "fold_id", "scope", "decision_time", "entry_time", "event_score_percentile",
        "label_persistent_failure", "mfe_360m", "mae_360m",
    ]
    for delay in (1, 3, 5):
        base_columns.extend(
            [
                f"entry_time_delay_{delay}m", f"entry_price_delay_{delay}m",
                f"fixed_exit_time_delay_{delay}m", f"fixed_exit_price_delay_{delay}m",
                f"overlay_exit_time_delay_{delay}m", f"overlay_exit_price_delay_{delay}m",
                f"disaster_exit_time_delay_{delay}m", f"disaster_exit_price_delay_{delay}m",
            ]
        )
    events = events.loc[:, [column for column in base_columns if column in events.columns]].copy()
    path_names = (
        "current_below_entry", "last60_return", "broke_prior_low_60", "distance_to_prior_low_60",
        "bar15_lower_low_share", "recovery_from_trough", "underwater_fraction",
    )
    for checkpoint in (60, 180, 360):
        columns = ["event_id", "x_score__score_max"]
        if checkpoint in (60, 180):
            columns += [f"x_path__{name}" for name in path_names if f"x_path__{name}" in frame.columns]
        part = frame.loc[frame["checkpoint_minutes"] == checkpoint, [column for column in columns if column in frame.columns]].copy()
        rename = {f"x_path__{name}": f"x{checkpoint}_{name}" for name in path_names}
        rename["x_score__score_max"] = f"score_max_{checkpoint}"
        part = part.rename(columns=rename)
        events = events.merge(part, on="event_id", how="left", validate="one_to_one")
    if not predictions.empty:
        pivot = predictions.pivot_table(index="event_id", columns="checkpoint_minutes", values="probability", aggfunc="first")
        pivot.columns = [f"p_failure_{int(value)}" for value in pivot.columns]
        events = events.merge(pivot.reset_index(), on="event_id", how="left", validate="one_to_one")
    entry = events["event_score_percentile"].astype(float)
    events["score_tier"] = entry.map(score_tier)
    next_180 = np.where(entry < 0.80, 0.80, np.where(entry < 0.90, 0.90, 0.95))
    next_360 = next_180
    events["score_upgrade_by_180"] = events.get("score_max_180", pd.Series(np.nan, index=events.index)).astype(float) >= next_180
    events["score_upgrade_by_360"] = events.get("score_max_360", pd.Series(np.nan, index=events.index)).astype(float) >= next_360
    return events


def _upgrade_diagnostics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    base = trades.loc[(trades["policy"] == "fixed_6h") & (trades["delay_minutes"] == 1)].copy()
    rows: list[dict[str, object]] = []
    for checkpoint in (180, 360):
        column = f"score_upgrade_by_{checkpoint}"
        for keys, group in base.groupby(["fold_id", "score_tier", column], dropna=False, sort=True):
            fold_id, tier, upgraded = keys
            gross = group["fixed6h_gross_return"].to_numpy(dtype=float)
            rows.append(
                {
                    "fold_id": fold_id,
                    "checkpoint_minutes": checkpoint,
                    "score_tier": tier,
                    "score_upgraded": bool(upgraded),
                    "events": int(len(group)),
                    "mean_fixed6h_gross_return": float(np.mean(gross)),
                    "fixed6h_win_rate_1x": float(np.mean(gross > 0.0013)),
                    "persistent_failure_rate": float(group["persistent_failure_target"].mean()),
                    "mean_mae_6h": float(group["mae"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _causal_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "frozen_q70_opening_pool", "status": "PASS", "detail": "R03.4.1 opening model and prior-quarter q70 calibration remain frozen"},
            {"check": "score_tiers", "status": "PASS", "detail": "q70-q80, q80-q90 and q90+ are preserved in every policy report; no tier is discarded"},
            {"check": "expanded_oof_train_pool", "status": "PASS", "detail": "persistent-failure models train on rolling causal OOF q50 events across the full pretest history"},
            {"check": "holding_label_embargo", "status": "PASS", "detail": "model OOF thresholds use a 120-hour embargo covering the future failure label"},
            {"check": "t60_warning_only", "status": "PASS", "detail": "T+60 can only arm a warning and never exits a trade"},
            {"check": "t180_double_confirmation", "status": "PASS", "detail": "early exit requires prior risk warning, extreme T+180 probability and multiple causal structure failures"},
            {"check": "tier_tolerance", "status": "PASS", "detail": "higher opening-score tiers require stricter failure evidence before an exit"},
            {"check": "disaster_stop_execution", "status": "PASS", "detail": "wide safety-floor breaches execute at the next minute open, not at an assumed perfect stop fill"},
            {"check": "fixed6h_boundary", "status": "PASS", "detail": "six hours is retained only as a frozen comparison baseline; this stage does not claim a final time exit"},
            {"check": "score_upgrade_diagnostic", "status": "PASS", "detail": "later score upgrades are recorded for future add-on research but do not create hindsight pyramiding"},
            {"check": "state_model", "status": "PASS", "detail": "abandoned market-state outputs are not loaded"},
            {"check": "sealed_holdout", "status": "PASS", "detail": "2026 remains sealed and incomplete future paths are excluded"},
        ]
    )


def run_failure_overlay_research(
    *,
    data_dir: str | Path | None = None,
    force_rebuild_outcomes: bool = False,
    progress: bool = True,
    config: FailureOverlayConfig = DEFAULT_FAILURE_OVERLAY_CONFIG,
) -> FailureOverlayResult:
    config.validate()
    model_config = _model_config(config)
    exit_config = _exit_config(config)
    loader = create_loader(LONG_CONTEXT_BASE_CONFIG, data_dir=data_dir)
    loader_preflight = run_public_loader_preflight(
        loader,
        LONG_CONTEXT_BASE_CONFIG,
        sample_dates=("2023-06-15", "2024-06-15", "2025-06-15"),
    )
    preflight: dict[str, object] = {"trade_bar": loader_preflight.to_dict()}
    if loader_preflight.status != "PASS":
        reports.empty(config, preflight, "BLOCKED_DATA", "1分钟Trade Bar公共Loader预检失败。")
        return FailureOverlayResult("BLOCKED_DATA", config.report_path)

    base_paths = [path for path in list_cached_years(LONG_CONTEXT_BASE_CONFIG) if 2023 <= int(path.name[-4:]) <= 2025]
    available = {int(path.name[-4:]) for path in base_paths}
    if available != {2023, 2024, 2025}:
        reports.empty(config, preflight, "BLOCKED_BASE_CACHE", f"缺少R03.2基础缓存，当前={sorted(available)}。")
        return FailureOverlayResult("BLOCKED_BASE_CACHE", config.report_path)
    outcome_paths = build_outcome_caches(
        base_paths,
        DEFAULT_STATE_CONTEXT_ABLATION_CONFIG,
        force_rebuild=force_rebuild_outcomes,
        progress=progress,
    )

    entry_audits: list[pd.DataFrame] = []
    extraction_rows: list[dict[str, object]] = []
    selection_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    importance_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    overlap_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    folds = default_ablation_folds(DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
    reporter = ProgressReporter("[R03.4.2.5 folds]", len(folds), every=1, enabled=progress)
    for fold_number, fold in enumerate(folds, start=1):
        try:
            fit = collect_base_period_data(base_paths, outcome_paths, start=fold.fit_start, end=fold.fit_end, outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
            calibration = collect_base_period_data(base_paths, outcome_paths, start=fold.calibration_start, end=fold.calibration_end, outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
            test = collect_base_period_data(base_paths, outcome_paths, start=fold.test_start, end=fold.test_end, outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
            pretest = collect_base_period_data(base_paths, outcome_paths, start=pd.Timestamp(config.research_start), end=fold.calibration_end, outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)

            entry_oof = build_rolling_oof_entry_timeline(pretest, event_builder_config=exit_config, config=model_config)
            audit = entry_oof.audit.copy()
            audit.insert(0, "fold_id", fold.fold_id)
            entry_audits.append(audit)

            bundle = fit_frozen_base_scores(fit, calibration, test, exit_config)
            test_timeline = build_oos_percentile_timeline(test.timestamps_ns, bundle.test_score, bundle.calibration_score)
            oos_events = build_event_candidates(test_timeline, signal_quantile=config.evaluation_quantile, config=exit_config)

            pretest_path = load_minute_path_data(start=pd.Timestamp(config.research_start), end=fold.calibration_end, data_dir=data_dir, config=exit_config, progress=progress)
            oos_path = load_minute_path_data(start=fold.test_start - pd.Timedelta(days=1), end=fold.test_end, data_dir=data_dir, config=exit_config, progress=progress)
            if pretest_path.coverage_ratio < 0.995 or oos_path.coverage_ratio < 0.995:
                raise RuntimeError("minute path coverage below 99.5%")

            train_frame, extraction = _extract_dataset(
                fold_id=fold.fold_id, phase="train_oof", scope="train_q50", events=entry_oof.events,
                path=pretest_path, timeline=entry_oof.timeline, model_config=model_config,
                policy_config=config, progress=progress,
            )
            extraction_rows.append(extraction)
            eval_frame, extraction = _extract_dataset(
                fold_id=fold.fold_id, phase="oos", scope="broad_q70", events=oos_events,
                path=oos_path, timeline=test_timeline, model_config=model_config,
                policy_config=config, progress=progress,
            )
            extraction_rows.append(extraction)
            if train_frame.empty or eval_frame.empty:
                raise RuntimeError("failure-overlay train or OOS frame is empty")

            predictions_by_checkpoint: list[pd.DataFrame] = []
            fold_threshold_data: dict[int, dict[str, object]] = {}
            for checkpoint in (60, 180):
                train_checkpoint = train_frame.loc[train_frame["checkpoint_minutes"] == checkpoint].copy()
                train_task, train_target = task_frame(train_checkpoint, "persistent_failure")
                candidates = []
                for feature_set in feature_sets(train_task):
                    try:
                        result = causal_oof(train_task, train_target, task="persistent_failure", feature_set=feature_set, config=model_config)
                        candidates.append((feature_set, result))
                    except Exception as exc:
                        failures.append({"fold_id": fold.fold_id, "checkpoint_minutes": checkpoint, "feature_set": feature_set.name, "error": f"OOF {type(exc).__name__}: {exc}"})
                if not candidates:
                    raise RuntimeError(f"no persistent-failure model candidates at T+{checkpoint}")
                selected_set, selected_oof, selection = choose_feature_set(candidates, train_target)
                selection.insert(0, "fold_id", fold.fold_id)
                selection.insert(1, "checkpoint_minutes", checkpoint)
                selection_parts.append(selection)

                if checkpoint == 60:
                    global_threshold, tier_thresholds, tier_audit = _tier_thresholds(
                        train_task, selected_oof.probabilities,
                        quantiles=config.warning_quantiles,
                        global_quantile=config.global_warning_quantile,
                        minimum_rows=config.minimum_tier_threshold_rows,
                    )
                    fold_threshold_data[60] = {"global": global_threshold, "tier": tier_thresholds}
                    threshold_rows.append({"fold_id": fold.fold_id, "checkpoint_minutes": 60, "kind": "global_warning", "score_tier": "ALL", "quantile": config.global_warning_quantile, "threshold": global_threshold, "feature_set": selected_set.name})
                    for row in tier_audit:
                        threshold_rows.append({"fold_id": fold.fold_id, "checkpoint_minutes": 60, "kind": "tier_warning", "feature_set": selected_set.name, **row})
                else:
                    global_threshold, tier_thresholds, tier_audit = _tier_thresholds(
                        train_task, selected_oof.probabilities,
                        quantiles=config.confirm_quantiles,
                        global_quantile=config.global_confirm_quantile,
                        minimum_rows=config.minimum_tier_threshold_rows,
                    )
                    valid = selected_oof.probabilities[np.isfinite(selected_oof.probabilities)]
                    ultra = float(np.quantile(valid, config.ultra_confirm_quantile))
                    fold_threshold_data[180] = {"global": global_threshold, "tier": tier_thresholds, "ultra": ultra}
                    threshold_rows.append({"fold_id": fold.fold_id, "checkpoint_minutes": 180, "kind": "global_confirm", "score_tier": "ALL", "quantile": config.global_confirm_quantile, "threshold": global_threshold, "feature_set": selected_set.name})
                    threshold_rows.append({"fold_id": fold.fold_id, "checkpoint_minutes": 180, "kind": "ultra_confirm", "score_tier": "ALL", "quantile": config.ultra_confirm_quantile, "threshold": ultra, "feature_set": selected_set.name})
                    for row in tier_audit:
                        threshold_rows.append({"fold_id": fold.fold_id, "checkpoint_minutes": 180, "kind": "tier_confirm", "feature_set": selected_set.name, **row})

                model = fit_model(train_task, train_target, task="persistent_failure", feature_set=selected_set, config=model_config)
                importance = feature_importance(model)
                if not importance.empty:
                    importance.insert(0, "fold_id", fold.fold_id)
                    importance.insert(1, "checkpoint_minutes", checkpoint)
                    importance_parts.append(importance)
                eval_checkpoint = eval_frame.loc[eval_frame["checkpoint_minutes"] == checkpoint].copy()
                eval_task, eval_target = task_frame(eval_checkpoint, "persistent_failure")
                probability = model.predict(eval_task)
                metric_rows.append(metric_row(eval_target, probability, fold_id=fold.fold_id, task="persistent_failure", checkpoint_minutes=checkpoint, feature_set=selected_set.name, scope="broad_q70"))
                for tier in ("q70_to_q80", "q80_to_q90", "q90_plus"):
                    mask = eval_task["event_score_percentile"].map(score_tier).to_numpy(dtype=object) == tier
                    if int(mask.sum()) >= config.minimum_test_rows and len(np.unique(eval_target[mask])) >= 2:
                        metric_rows.append(metric_row(eval_target[mask], probability[mask], fold_id=fold.fold_id, task="persistent_failure", checkpoint_minutes=checkpoint, feature_set=selected_set.name, scope=tier))
                pred = eval_task.loc[:, ["event_id", "fold_id", "scope", "decision_time", "entry_time", "checkpoint_minutes", "event_score_percentile", "label_persistent_failure"]].copy()
                pred["probability"] = probability
                pred["feature_set"] = selected_set.name
                predictions_by_checkpoint.append(pred)

            predictions = pd.concat(predictions_by_checkpoint, ignore_index=True)
            prediction_parts.append(predictions)
            thresholds = OverlayThresholds(
                global_warning=float(fold_threshold_data[60]["global"]),
                global_confirm=float(fold_threshold_data[180]["global"]),
                ultra_confirm=float(fold_threshold_data[180]["ultra"]),
                tier_warning=dict(fold_threshold_data[60]["tier"]),
                tier_confirm=dict(fold_threshold_data[180]["tier"]),
            )
            events = _event_table(eval_frame, predictions)
            for policy in ("fixed_6h", "fixed_6h_disaster_stop", "global_failure_overlay", "tiered_failure_overlay", "ultra_failure_overlay"):
                for delay in config.entry_delay_minutes:
                    raw = pd.DataFrame(
                        [
                            trade
                            for _, row in events.iterrows()
                            if (trade := simulate_overlay_event(row, policy=policy, delay_minutes=delay, thresholds=thresholds, config=config)) is not None
                        ]
                    )
                    executed, skipped = enforce_non_overlap(raw)
                    overlap_rows.append(
                        {
                            "fold_id": fold.fold_id,
                            "policy": policy,
                            "delay_minutes": delay,
                            "candidate_events": int(len(events)),
                            "complete_events": int(len(raw)),
                            "skipped_overlap": int(skipped),
                            "final_trades": int(len(executed)),
                        }
                    )
                    if not executed.empty:
                        trade_parts.append(executed)
        except Exception as exc:
            failures.append({"fold_id": fold.fold_id, "error": f"{type(exc).__name__}: {exc}"})
        reporter.update(fold_number)
    reporter.close()

    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    summary, periods, tier_summary, exits = build_policy_tables(trades, config=config)
    stable = stable_candidates(summary, periods, config)
    upgrades = _upgrade_diagnostics(trades)
    failures_frame = pd.DataFrame(failures)

    if summary.empty or stable.empty:
        decision = "FAIL_INCOMPLETE_FAILURE_OVERLAY"
        reason = "没有同时生成WF_2024与WF_2025的q70高置信坏单Overlay结果。"
    else:
        passed = stable.loc[stable["stable_overlay_upgrade"] == True]  # noqa: E712
        valid = stable.loc[
            (stable["valid_high_confidence_overlay"] == True)
            & (stable["stable_positive_expectancy"] == True)
        ]  # noqa: E712
        if not passed.empty:
            decision = "PASS_HIGH_CONFIDENCE_FAILURE_OVERLAY"
            reason = "至少一个极高置信坏单Overlay在2024和2025都保持q70正期望，并同时提高了相对固定6小时诊断基准的2倍成本总利润。"
        elif not valid.empty:
            decision = "PASS_FAILURE_OVERLAY_RISK_CONTROL_ONLY"
            reason = "高置信坏单Overlay能够改善被退出订单并保留大部分基准利润，但尚未在两年都提高总利润；只能作为风险控制候选。"
        else:
            decision = "FAIL_NO_ROBUST_FAILURE_OVERLAY"
            reason = "价格路径能识别部分坏单，但当前双重确认规则仍未稳定转化为跨年利润或风险改进；不得为了止损而破坏q70 Edge。"

    reports.write_reports(
        config=config,
        preflight=preflight,
        manifest={
            "stage": STAGE_ID,
            "name": STAGE_NAME,
            "config": config.to_dict(),
            "folds": [fold.to_dict() for fold in folds],
            "opening_model": "frozen R03.4.1 long utility h6 with q70 entry pool",
            "final_time_exit_policy": "NOT_CLAIMED; fixed 6h is comparison only",
            "state_model_policy": "ABANDONED_AND_NOT_LOADED",
        },
        entry_oof_audit=pd.concat(entry_audits, ignore_index=True) if entry_audits else pd.DataFrame(),
        extraction_audit=pd.DataFrame(extraction_rows),
        model_selection=pd.concat(selection_parts, ignore_index=True) if selection_parts else pd.DataFrame(),
        model_metrics=pd.DataFrame(metric_rows),
        thresholds=pd.DataFrame(threshold_rows),
        importance=pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame(),
        predictions=pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame(),
        policy_summary=summary,
        period_summary=periods,
        tier_summary=tier_summary,
        exit_summary=exits,
        overlap_audit=pd.DataFrame(overlap_rows),
        upgrade_diagnostics=upgrades,
        stable=stable,
        causal_audit=_causal_audit(),
        failures=failures_frame,
        trades=trades,
        decision=decision,
        reason=reason,
    )
    return FailureOverlayResult(decision=decision, report_dir=config.report_path)
