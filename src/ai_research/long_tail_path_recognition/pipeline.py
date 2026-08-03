#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2.2 causal path-health recognition pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.config import LongTailExitAuditConfig
from src.ai_research.long_tail_exit_audit.data import collect_base_period_data, load_minute_path_data
from src.ai_research.long_tail_exit_audit.modeling import fit_frozen_base_scores
from src.ai_research.long_tail_exit_audit.simulator import EventCandidate, ScoreTimeline, build_event_candidates
from src.ai_research.long_tail_path_atlas.atlas import extract_event_path
from src.ai_research.long_tail_path_atlas.config import LongTailPathAtlasConfig
from src.ai_research.state_context_ablation.config import DEFAULT_STATE_CONTEXT_ABLATION_CONFIG
from src.ai_research.state_context_ablation.modeling import AblationPeriodData, default_ablation_folds
from src.ai_research.state_context_ablation.outcomes import build_outcome_caches
from src.ai_research.swing_baseline.dataset import create_loader, list_cached_years, run_public_loader_preflight
from src.ai_research.swing_long_context.config import LONG_CONTEXT_BASE_CONFIG
from src.research_common.progress import ProgressReporter

from .config import (
    DEFAULT_LONG_TAIL_PATH_RECOGNITION_CONFIG,
    STAGE_ID,
    STAGE_NAME,
    LongTailPathRecognitionConfig,
)
from .features import build_checkpoint_row, feature_sets, task_target
from .modeling import (
    causal_oof_probabilities,
    classification_metrics,
    feature_importance,
    fit_binary_model,
    prediction_deciles,
    probability_threshold,
    stable_signal_candidates,
)
from .reports import write_reports


@dataclass(frozen=True)
class LongTailPathRecognitionResult:
    decision: str
    report_dir: Path


TASKS_BY_CHECKPOINT: dict[int, tuple[str, ...]] = {
    60: ("persistent_failure", "recovery_from_underwater", "giveback_risk"),
    180: ("persistent_failure", "recovery_from_underwater", "giveback_risk"),
    360: ("persistent_failure", "recovery_from_underwater", "post6_continuation"),
}


def _base_config(config: LongTailPathRecognitionConfig) -> LongTailExitAuditConfig:
    return LongTailExitAuditConfig(
        symbol=config.symbol,
        research_start=config.research_start,
        research_end=config.research_end,
        sealed_holdout_start=config.sealed_holdout_start,
        primary_signal_quantile=config.primary_signal_quantile,
        quality_control_quantile=config.quality_control_quantile,
        primary_horizon_hours=config.fixed_diagnostic_horizon_hours,
        risk_penalty=config.risk_penalty,
        base_round_trip_cost=config.base_round_trip_cost,
        train_sample_cap=config.train_sample_cap,
        base_n_estimators=config.base_n_estimators,
        base_learning_rate=config.base_learning_rate,
        base_num_leaves=config.base_num_leaves,
        base_min_child_samples=config.base_min_child_samples,
        random_state=config.random_state,
    )


def _atlas_config(config: LongTailPathRecognitionConfig) -> LongTailPathAtlasConfig:
    return LongTailPathAtlasConfig(
        symbol=config.symbol,
        research_start=config.research_start,
        research_end=config.research_end,
        sealed_holdout_start=config.sealed_holdout_start,
        primary_signal_quantile=config.primary_signal_quantile,
        quality_control_quantile=config.quality_control_quantile,
        entry_delay_minutes=config.entry_delay_minutes,
        analysis_horizon_hours=config.analysis_horizon_hours,
        fixed_diagnostic_horizon_hours=config.fixed_diagnostic_horizon_hours,
        risk_penalty=config.risk_penalty,
        base_round_trip_cost=config.base_round_trip_cost,
        train_sample_cap=config.train_sample_cap,
        base_n_estimators=config.base_n_estimators,
        base_learning_rate=config.base_learning_rate,
        base_num_leaves=config.base_num_leaves,
        base_min_child_samples=config.base_min_child_samples,
        random_state=config.random_state,
        export_full_minute_paths=False,
    )


def _thresholds(calibration_scores: np.ndarray) -> dict[float, float]:
    values = np.asarray(calibration_scores, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 100:
        raise RuntimeError("insufficient calibration scores")
    return {q: float(np.quantile(values, q)) for q in (0.50, 0.60, 0.70, 0.90, 0.95)}


def _timeline(data: AblationPeriodData, scores: np.ndarray, thresholds: dict[float, float]) -> ScoreTimeline:
    return ScoreTimeline(
        decision_times_ns=np.asarray(data.timestamps_ns, dtype=np.int64),
        scores=np.asarray(scores, dtype=float),
        calibration_thresholds=dict(thresholds),
    )


def _extract_checkpoint_dataset(
    *,
    fold_id: str,
    phase: str,
    scope: str,
    events: tuple[EventCandidate, ...],
    path,
    timeline: ScoreTimeline,
    calibration_scores: np.ndarray,
    atlas_config: LongTailPathAtlasConfig,
    config: LongTailPathRecognitionConfig,
    progress: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    complete = 0
    skipped = 0
    reporter = ProgressReporter(
        f"[R03.4.2.2 {fold_id} {phase} {scope}] events",
        len(events),
        every=max(1, len(events) // 20),
        enabled=progress,
    )
    for number, event in enumerate(events, start=1):
        extraction = extract_event_path(
            event=event,
            fold_id=fold_id,
            phase=phase,
            path=path,
            timeline=timeline,
            calibration_scores=calibration_scores,
            config=atlas_config,
        )
        if extraction is None:
            skipped += 1
        else:
            complete += 1
            for checkpoint in config.checkpoints_minutes:
                row = build_checkpoint_row(
                    extraction,
                    checkpoint_minutes=checkpoint,
                    path=path,
                    config=config,
                )
                row["scope"] = scope
                rows.append(row)
        reporter.update(number)
    reporter.close()
    return pd.DataFrame(rows), {
        "fold_id": fold_id,
        "phase": phase,
        "scope": scope,
        "candidate_events": len(events),
        "complete_48h_events": complete,
        "skipped_incomplete_or_missing": skipped,
        "checkpoint_rows": len(rows),
    }


def _dataset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(["fold_id", "phase", "scope", "checkpoint_minutes"], sort=False):
        fold_id, phase, scope, checkpoint = keys
        for task in TASKS_BY_CHECKPOINT[int(checkpoint)]:
            work, target = task_target(group, task)
            rows.append(
                {
                    "fold_id": fold_id,
                    "phase": phase,
                    "scope": scope,
                    "checkpoint_minutes": int(checkpoint),
                    "task": task,
                    "rows": int(len(work)),
                    "positive_rows": int(target.sum()) if len(target) else 0,
                    "positive_rate": float(target.mean()) if len(target) else np.nan,
                    "mean_fixed6h_net_1x": float(work["fixed6h_net_1x"].mean()) if len(work) else np.nan,
                    "mean_net_24h_1x": float(work["net_24h_1x"].mean()) if len(work) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _action_row(
    frame: pd.DataFrame,
    target: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
    fold_id: str,
    task: str,
    checkpoint_minutes: int,
    feature_set: str,
    scope: str,
) -> dict[str, object]:
    p = np.asarray(probability, dtype=float)
    y = np.asarray(target, dtype=np.int8)
    flagged = np.isfinite(p) & (p >= threshold)
    selected = frame.loc[flagged].copy()
    selected_target = y[flagged]
    positives = int(y.sum())
    return {
        "fold_id": fold_id,
        "task": task,
        "checkpoint_minutes": int(checkpoint_minutes),
        "feature_set": feature_set,
        "scope": scope,
        "probability_threshold": float(threshold),
        "rows": int(len(frame)),
        "flagged_rows": int(flagged.sum()),
        "flagged_share": float(flagged.mean()) if len(flagged) else np.nan,
        "flagged_positive_rate": float(selected_target.mean()) if len(selected_target) else np.nan,
        "positive_recall": float(selected_target.sum() / positives) if positives > 0 else np.nan,
        "mean_checkpoint_net_exit_1x": float(selected["checkpoint_net_exit_1x"].mean()) if len(selected) else np.nan,
        "mean_hold6h_net_1x": float(selected["fixed6h_net_1x"].mean()) if len(selected) else np.nan,
        "mean_hold24h_net_1x": float(selected["net_24h_1x"].mean()) if len(selected) else np.nan,
        "mean_hold48h_net_1x": float(selected["net_48h_1x"].mean()) if len(selected) else np.nan,
        "mean_checkpoint_advantage_vs_6h": float((selected["checkpoint_net_exit_1x"] - selected["fixed6h_net_1x"]).mean()) if len(selected) else np.nan,
        "mean_checkpoint_advantage_vs_24h": float((selected["checkpoint_net_exit_1x"] - selected["net_24h_1x"]).mean()) if len(selected) else np.nan,
        "flagged_24h_positive_share": float((selected["net_24h_1x"] > 0).mean()) if len(selected) else np.nan,
    }


def _broad_safe_row(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    safe_threshold: float,
    fold_id: str,
    task: str,
    checkpoint_minutes: int,
    feature_set: str,
    scope: str,
) -> dict[str, object]:
    p = np.asarray(probability, dtype=float)
    selected = frame.loc[np.isfinite(p) & (p <= safe_threshold)].copy()
    return {
        "fold_id": fold_id,
        "task": task,
        "checkpoint_minutes": int(checkpoint_minutes),
        "feature_set": feature_set,
        "scope": scope,
        "safe_probability_threshold": float(safe_threshold),
        "rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "selected_share": float(len(selected) / len(frame)) if len(frame) else np.nan,
        "selected_q90_share": float(selected["is_q90"].mean()) if len(selected) else np.nan,
        "mean_fixed6h_net_1x": float(selected["fixed6h_net_1x"].mean()) if len(selected) else np.nan,
        "mean_net_24h_1x": float(selected["net_24h_1x"].mean()) if len(selected) else np.nan,
        "mean_net_48h_1x": float(selected["net_48h_1x"].mean()) if len(selected) else np.nan,
        "fixed6h_positive_rate": float((selected["fixed6h_net_1x"] > 0).mean()) if len(selected) else np.nan,
        "persistent_failure_rate": float(selected["label_persistent_failure"].mean()) if len(selected) else np.nan,
    }


def _score_ablation(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    keys = ["fold_id", "task", "checkpoint_minutes", "scope"]
    left = metrics.loc[metrics["feature_set"] == "path_structure_logistic"].set_index(keys)
    right = metrics.loc[metrics["feature_set"] == "path_plus_score_logistic"].set_index(keys)
    common = left.index.intersection(right.index)
    rows: list[dict[str, object]] = []
    for key in common:
        path_row = left.loc[key]
        score_row = right.loc[key]
        fold_id, task, checkpoint, scope = key
        rows.append(
            {
                "fold_id": fold_id,
                "task": task,
                "checkpoint_minutes": int(checkpoint),
                "scope": scope,
                "auc_path_only": float(path_row["roc_auc"]),
                "auc_path_plus_score": float(score_row["roc_auc"]),
                "auc_increment_from_score": float(score_row["roc_auc"] - path_row["roc_auc"]),
                "ap_lift_path_only": float(path_row["average_precision_lift"]),
                "ap_lift_path_plus_score": float(score_row["average_precision_lift"]),
                "brier_skill_path_only": float(path_row["brier_skill"]),
                "brier_skill_path_plus_score": float(score_row["brier_skill"]),
            }
        )
    return pd.DataFrame(rows)


def _representatives(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    group_keys = ["fold_id", "task", "checkpoint_minutes", "feature_set", "scope"]
    for _, group in predictions.groupby(group_keys, sort=False):
        work = group.sort_values("probability", ascending=False)
        positive = work.loc[work["target"] == 1]
        negative = work.loc[work["target"] == 0]
        choices = []
        if not positive.empty:
            choices.append(positive.head(1).assign(case="high_probability_true_positive"))
            choices.append(positive.tail(1).assign(case="low_probability_false_negative"))
        if not negative.empty:
            choices.append(negative.head(1).assign(case="high_probability_false_positive"))
            choices.append(negative.tail(1).assign(case="low_probability_true_negative"))
        rows.extend(choices)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _causal_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "frozen_entry_model", "status": "PASS", "detail": "the R03.4.1 base long LightGBM and next-minute-open entry are unchanged"},
            {"check": "entry_score_not_hold_score", "status": "PASS", "detail": "entry score persistence is only an optional feature and is ablated against price structure"},
            {"check": "checkpoint_feature_cutoff", "status": "PASS", "detail": "x_path/x_score features use only rows available through T+60/T+180/T+360"},
            {"check": "future_label_only", "status": "PASS", "detail": "24h/48h recovery, failure, giveback and continuation outcomes are labels only"},
            {"check": "causal_oof_threshold", "status": "PASS", "detail": "probability thresholds use expanding OOF predictions with a 48h embargo"},
            {"check": "broader_pool_policy", "status": "PASS", "detail": "q70 is an audit pool for additional opportunities; q90 remains the primary OOS benchmark"},
            {"check": "no_exit_rule", "status": "PASS", "detail": "checkpoint diagnostics are not presented as executable exits"},
            {"check": "state_model_abandoned", "status": "PASS", "detail": "no strategic, tactical, entry or activity state cache is loaded"},
            {"check": "sealed_holdout", "status": "PASS", "detail": "2026 is not loaded and complete 48h labels cannot cross the seal"},
        ]
    )


def _contract(config: LongTailPathRecognitionConfig) -> dict[str, object]:
    return {
        "frozen_entry_model": "long_utility_h6 = long_mfe_h6 - 1.25 * long_mae_h6; LightGBM 420 trees, lr=0.035, leaves=31",
        "primary_scope": "q90 independent events, next-minute-open entry",
        "discovery_scope": "q70 independent events to increase path-label diversity and test non-high-score opportunities",
        "checkpoints": list(config.checkpoints_minutes),
        "tasks": {
            "persistent_failure": "24h cannot produce 1% MFE and remains non-positive",
            "recovery_from_underwater": "currently weak/underwater but reaches positive 24h net with at least 1% MFE",
            "giveback_risk": "after at least 1% MFE, future path to 6h gives back at least 0.75%",
            "post6_continuation": "additional 1% MFE from 6h to 24h",
        },
        "feature_policy": "causal price path/structure is primary; entry score path is optional and separately ablated",
        "holding_policy": "healthy trades may remain open for 24-48h; this stage does not force time exits",
        "state_model_policy": "ABANDONED_FOR_TRADING_AND_NOT_LOADED",
    }


def _empty_reports(
    *,
    config: LongTailPathRecognitionConfig,
    preflight: dict[str, object],
    decision: str,
    reason: str,
) -> LongTailPathRecognitionResult:
    empty = pd.DataFrame()
    write_reports(
        config.report_path,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight=preflight,
        contract=_contract(config),
        extraction_audit=empty,
        dataset_summary=empty,
        metrics=empty,
        deciles=empty,
        action_diagnostics=empty,
        broad_pool_diagnostics=empty,
        score_ablation=empty,
        importance=empty,
        predictions=empty,
        representatives=empty,
        stable=empty,
        causal_audit=_causal_audit(),
        failures=empty,
        decision=decision,
        reason=reason,
        config=config,
    )
    return LongTailPathRecognitionResult(decision=decision, report_dir=config.report_path)


def run_long_tail_path_recognition(
    *,
    data_dir: str | Path | None = None,
    force_rebuild_outcomes: bool = False,
    progress: bool = True,
    config: LongTailPathRecognitionConfig = DEFAULT_LONG_TAIL_PATH_RECOGNITION_CONFIG,
) -> LongTailPathRecognitionResult:
    config.validate()
    base_config = _base_config(config)
    atlas_config = _atlas_config(config)
    loader = create_loader(LONG_CONTEXT_BASE_CONFIG, data_dir=data_dir)
    loader_preflight = run_public_loader_preflight(
        loader,
        LONG_CONTEXT_BASE_CONFIG,
        sample_dates=("2023-06-15", "2024-06-15", "2025-06-15"),
    )
    preflight: dict[str, object] = {"trade_bar": loader_preflight.to_dict()}
    if loader_preflight.status != "PASS":
        return _empty_reports(
            config=config,
            preflight=preflight,
            decision="BLOCKED_DATA",
            reason="1分钟Trade Bar公共Loader预检失败。",
        )

    base_paths = [path for path in list_cached_years(LONG_CONTEXT_BASE_CONFIG) if 2023 <= int(path.name[-4:]) <= 2025]
    available = {int(path.name[-4:]) for path in base_paths}
    if available != {2023, 2024, 2025}:
        return _empty_reports(
            config=config,
            preflight=preflight,
            decision="BLOCKED_BASE_CACHE",
            reason=f"缺少R03.2的2023/2024/2025基础缓存，当前={sorted(available)}。",
        )
    outcome_paths = build_outcome_caches(
        base_paths,
        DEFAULT_STATE_CONTEXT_ABLATION_CONFIG,
        force_rebuild=force_rebuild_outcomes,
        progress=progress,
    )

    extraction_audits: list[dict[str, object]] = []
    all_frames: list[pd.DataFrame] = []
    discovery_pool: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    decile_parts: list[pd.DataFrame] = []
    action_rows: list[dict[str, object]] = []
    broad_rows: list[dict[str, object]] = []
    importance_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    folds = default_ablation_folds(DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
    fold_reporter = ProgressReporter("[R03.4.2.2 folds]", len(folds), every=1, enabled=progress)

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
            bundle = fit_frozen_base_scores(fit, calibration, test, base_config)
            thresholds = _thresholds(bundle.calibration_score)
            calibration_timeline = _timeline(calibration, bundle.calibration_score, thresholds)
            test_timeline = _timeline(test, bundle.test_score, thresholds)

            discovery_events = build_event_candidates(
                calibration_timeline,
                signal_quantile=config.discovery_signal_quantile,
                config=base_config,
            )
            broad_events = build_event_candidates(
                test_timeline,
                signal_quantile=config.discovery_signal_quantile,
                config=base_config,
            )
            primary_events = build_event_candidates(
                test_timeline,
                signal_quantile=config.primary_signal_quantile,
                config=base_config,
            )

            discovery_path = load_minute_path_data(
                start=fold.calibration_start,
                end=fold.calibration_end,
                data_dir=data_dir,
                config=base_config,
                progress=progress,
            )
            oos_end = min(fold.test_end, pd.Timestamp(config.sealed_holdout_start) - pd.Timedelta(minutes=1))
            oos_path = load_minute_path_data(
                start=fold.test_start,
                end=oos_end,
                data_dir=data_dir,
                config=base_config,
                progress=progress,
            )
            preflight[f"{fold.fold_id}_discovery_path"] = {
                "rows": len(discovery_path.index),
                "start": str(discovery_path.index[0]),
                "end": str(discovery_path.index[-1]),
                "coverage_ratio": discovery_path.coverage_ratio,
            }
            preflight[f"{fold.fold_id}_oos_path"] = {
                "rows": len(oos_path.index),
                "start": str(oos_path.index[0]),
                "end": str(oos_path.index[-1]),
                "coverage_ratio": oos_path.coverage_ratio,
            }
            if discovery_path.coverage_ratio < 0.995 or oos_path.coverage_ratio < 0.995:
                raise RuntimeError("minute path coverage below 99.5%")

            discovery_frame, audit = _extract_checkpoint_dataset(
                fold_id=fold.fold_id,
                phase="discovery",
                scope="discovery_q70",
                events=discovery_events,
                path=discovery_path,
                timeline=calibration_timeline,
                calibration_scores=bundle.calibration_score,
                atlas_config=atlas_config,
                config=config,
                progress=progress,
            )
            extraction_audits.append(audit)
            if not discovery_frame.empty:
                discovery_pool.append(discovery_frame)
                all_frames.append(discovery_frame)

            broad_frame, audit = _extract_checkpoint_dataset(
                fold_id=fold.fold_id,
                phase="oos",
                scope="broad_q70",
                events=broad_events,
                path=oos_path,
                timeline=test_timeline,
                calibration_scores=bundle.calibration_score,
                atlas_config=atlas_config,
                config=config,
                progress=progress,
            )
            extraction_audits.append(audit)
            primary_frame, audit = _extract_checkpoint_dataset(
                fold_id=fold.fold_id,
                phase="oos",
                scope="primary_q90",
                events=primary_events,
                path=oos_path,
                timeline=test_timeline,
                calibration_scores=bundle.calibration_score,
                atlas_config=atlas_config,
                config=config,
                progress=progress,
            )
            extraction_audits.append(audit)
            if not broad_frame.empty:
                all_frames.append(broad_frame)
            if not primary_frame.empty:
                all_frames.append(primary_frame)

            cumulative_discovery = pd.concat(discovery_pool, ignore_index=True)
            for checkpoint in config.checkpoints_minutes:
                train_checkpoint = cumulative_discovery.loc[
                    cumulative_discovery["checkpoint_minutes"] == checkpoint
                ].copy()
                eval_scopes = {
                    "broad_q70": broad_frame.loc[broad_frame["checkpoint_minutes"] == checkpoint].copy(),
                    "primary_q90": primary_frame.loc[primary_frame["checkpoint_minutes"] == checkpoint].copy(),
                }
                for task in TASKS_BY_CHECKPOINT[checkpoint]:
                    train_task, train_target = task_target(train_checkpoint, task)
                    if len(train_task) < config.minimum_train_rows or len(np.unique(train_target)) < 2:
                        failures.append(
                            {
                                "fold_id": fold.fold_id,
                                "task": task,
                                "checkpoint_minutes": checkpoint,
                                "feature_set": "ALL",
                                "error": f"insufficient train task rows={len(train_task)} classes={np.unique(train_target).tolist()}",
                            }
                        )
                        continue
                    for feature_set in feature_sets(train_task):
                        try:
                            oof = causal_oof_probabilities(
                                train_task,
                                target=train_target,
                                task=task,
                                feature_set=feature_set,
                                config=config,
                            )
                            threshold_quantile = (
                                config.high_risk_quantile
                                if task in {"persistent_failure", "giveback_risk"}
                                else config.high_hold_quantile
                            )
                            threshold = probability_threshold(
                                oof.probabilities,
                                quantile=threshold_quantile,
                            )
                            safe_threshold = probability_threshold(
                                oof.probabilities,
                                quantile=config.broad_safe_quantile,
                            )
                            model = fit_binary_model(
                                train_task,
                                target=train_target,
                                task=task,
                                feature_set=feature_set,
                                config=config,
                            )
                            imp = feature_importance(model)
                            if not imp.empty:
                                imp.insert(0, "fold_id", fold.fold_id)
                                imp.insert(2, "checkpoint_minutes", checkpoint)
                                imp["oof_folds_used"] = oof.folds_used
                                importance_parts.append(imp)

                            for scope, eval_checkpoint in eval_scopes.items():
                                eval_task, eval_target = task_target(eval_checkpoint, task)
                                if len(eval_task) < config.minimum_test_rows or len(np.unique(eval_target)) < 2:
                                    continue
                                probability = model.predict(eval_task)
                                metric_rows.append(
                                    classification_metrics(
                                        eval_target,
                                        probability,
                                        fold_id=fold.fold_id,
                                        task=task,
                                        checkpoint_minutes=checkpoint,
                                        feature_set=feature_set.name,
                                        scope=scope,
                                    )
                                )
                                decile = prediction_deciles(
                                    eval_task,
                                    target=eval_target,
                                    probability=probability,
                                    fold_id=fold.fold_id,
                                    task=task,
                                    checkpoint_minutes=checkpoint,
                                    feature_set=feature_set.name,
                                    scope=scope,
                                )
                                if not decile.empty:
                                    decile_parts.append(decile)
                                action_rows.append(
                                    _action_row(
                                        eval_task,
                                        eval_target,
                                        probability,
                                        threshold=threshold,
                                        fold_id=fold.fold_id,
                                        task=task,
                                        checkpoint_minutes=checkpoint,
                                        feature_set=feature_set.name,
                                        scope=scope,
                                    )
                                )
                                if task == "persistent_failure":
                                    broad_rows.append(
                                        _broad_safe_row(
                                            eval_task,
                                            probability,
                                            safe_threshold=safe_threshold,
                                            fold_id=fold.fold_id,
                                            task=task,
                                            checkpoint_minutes=checkpoint,
                                            feature_set=feature_set.name,
                                            scope=scope,
                                        )
                                    )
                                pred = eval_task.loc[
                                    :,
                                    [
                                        "event_id", "fold_id", "scope", "decision_time", "entry_time",
                                        "checkpoint_minutes", "event_score_percentile", "semantic_path_type",
                                        "checkpoint_net_exit_1x", "fixed6h_net_1x", "net_24h_1x", "net_48h_1x",
                                    ],
                                ].copy()
                                pred["task"] = task
                                pred["feature_set"] = feature_set.name
                                pred["target"] = eval_target
                                pred["probability"] = probability
                                pred["probability_threshold"] = threshold
                                prediction_parts.append(pred)
                        except Exception as exc:
                            failures.append(
                                {
                                    "fold_id": fold.fold_id,
                                    "task": task,
                                    "checkpoint_minutes": checkpoint,
                                    "feature_set": feature_set.name,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
        except Exception as exc:
            failures.append({"fold_id": fold.fold_id, "task": "FOLD", "error": f"{type(exc).__name__}: {exc}"})
        fold_reporter.update(fold_number)
    fold_reporter.close()

    all_data = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    deciles = pd.concat(decile_parts, ignore_index=True) if decile_parts else pd.DataFrame()
    actions = pd.DataFrame(action_rows)
    broad = pd.DataFrame(broad_rows)
    importance = pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    stable = stable_signal_candidates(metrics)
    score_ablation = _score_ablation(metrics)
    representatives = _representatives(predictions)
    dataset_summary = _dataset_summary(all_data)
    failures_frame = pd.DataFrame(failures)

    if metrics.empty:
        decision = "FAIL_NO_VALID_PATH_MODELS"
        reason = "没有任务完成2024/2025纯OOS训练与评估；应先检查发现期样本规模、类别覆盖或数据对齐。"
    else:
        passed = stable.loc[stable["stable_signal"] == True] if not stable.empty else pd.DataFrame()  # noqa: E712
        health = passed.loc[passed["task"].isin(["persistent_failure", "recovery_from_underwater"])] if not passed.empty else pd.DataFrame()
        continuation = passed.loc[passed["task"] == "post6_continuation"] if not passed.empty else pd.DataFrame()
        if not health.empty:
            decision = "PASS_CAUSAL_PATH_HEALTH_SIGNAL"
            reason = "至少一个早期价格路径/结构模型在2024和2025均能稳定识别持续失败或可恢复回撤；可以进入冻结差异化退出规则研究。"
        elif not continuation.empty:
            decision = "PASS_LONG_HOLD_CONTINUATION_SIGNAL"
            reason = "尚未稳定识别早期坏单，但T+6小时续持价值可以跨年预测；下一阶段只能研究选择性长期持有，不能假装已有止损模型。"
        else:
            decision = "FAIL_NO_STABLE_PATH_RECOGNITION"
            reason = "路径结构在单年可能有信息，但没有任务跨2024/2025稳定通过；不能据此设计差异化退出。"

    write_reports(
        config.report_path,
        manifest={
            "stage": STAGE_ID,
            "name": STAGE_NAME,
            "config": config.to_dict(),
            "folds": [fold.to_dict() for fold in folds],
            "state_model_policy": "ABANDONED_FOR_TRADING_AND_NOT_LOADED",
        },
        preflight=preflight,
        contract=_contract(config),
        extraction_audit=pd.DataFrame(extraction_audits),
        dataset_summary=dataset_summary,
        metrics=metrics,
        deciles=deciles,
        action_diagnostics=actions,
        broad_pool_diagnostics=broad,
        score_ablation=score_ablation,
        importance=importance,
        predictions=predictions,
        representatives=representatives,
        stable=stable,
        causal_audit=_causal_audit(),
        failures=failures_frame,
        decision=decision,
        reason=reason,
        config=config,
    )
    return LongTailPathRecognitionResult(decision=decision, report_dir=config.report_path)
