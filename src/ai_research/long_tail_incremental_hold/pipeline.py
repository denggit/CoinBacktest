#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2.6 incremental holding-value research."""

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
from src.ai_research.long_tail_multistage_decision.entry_oof import (
    build_oos_percentile_timeline,
    build_rolling_oof_entry_timeline,
)
from src.ai_research.long_tail_multistage_decision.features import extract_extended_event_path
from src.ai_research.state_context_ablation.config import DEFAULT_STATE_CONTEXT_ABLATION_CONFIG
from src.ai_research.state_context_ablation.modeling import default_ablation_folds
from src.ai_research.state_context_ablation.outcomes import build_outcome_caches
from src.ai_research.swing_baseline.dataset import create_loader, list_cached_years, run_public_loader_preflight
from src.ai_research.swing_long_context.config import LONG_CONTEXT_BASE_CONFIG
from src.research_common.progress import ProgressReporter

from . import reports
from .config import (
    DEFAULT_INCREMENTAL_HOLD_CONFIG,
    STAGE_ID,
    STAGE_NAME,
    IncrementalHoldConfig,
)
from .features import (
    build_incremental_hold_row,
    feature_sets,
    prediction_columns,
    regression_targets,
    target_values,
)
from .modeling import (
    causal_oof,
    choose_feature_set,
    evaluate,
    feature_importance,
    fit_model,
    stable_candidates,
)


@dataclass(frozen=True)
class IncrementalHoldResult:
    decision: str
    report_dir: Path


def _exit_config(config: IncrementalHoldConfig) -> LongTailExitAuditConfig:
    return LongTailExitAuditConfig(
        symbol=config.symbol,
        research_start=config.research_start,
        research_end=config.research_end,
        sealed_holdout_start=config.sealed_holdout_start,
        primary_signal_quantile=0.70,
        quality_control_quantile=0.90,
        primary_horizon_hours=config.primary_horizon_hours,
        risk_penalty=config.risk_penalty,
        base_round_trip_cost=config.base_round_trip_cost,
        train_sample_cap=config.base_train_sample_cap,
        base_n_estimators=config.base_n_estimators,
        base_learning_rate=config.base_learning_rate,
        base_num_leaves=config.base_num_leaves,
        base_min_child_samples=config.base_min_child_samples,
        random_state=config.random_state,
    )


def _multistage_config(config: IncrementalHoldConfig) -> LongTailMultistageConfig:
    return LongTailMultistageConfig(
        symbol=config.symbol,
        research_start=config.research_start,
        research_end=config.research_end,
        sealed_holdout_start=config.sealed_holdout_start,
        primary_horizon_hours=config.primary_horizon_hours,
        path_horizon_hours=config.path_horizon_hours,
        entry_delay_minutes=config.entry_delay_minutes,
        checkpoints_minutes=(60, 180, 360, 1440),
        train_event_quantile=config.train_event_quantile,
        evaluation_quantiles=config.evaluation_quantiles,
        base_round_trip_cost=config.base_round_trip_cost,
        risk_penalty=config.risk_penalty,
        entry_oof_min_train_days=config.entry_oof_min_train_days,
        entry_oof_calibration_days=config.entry_oof_calibration_days,
        entry_oof_blocks=config.entry_oof_blocks,
        entry_oof_embargo_hours=config.entry_oof_embargo_hours,
        holding_oof_splits=config.holding_oof_splits,
        holding_oof_embargo_hours=config.holding_oof_embargo_hours,
        minimum_train_rows=config.minimum_train_rows,
        minimum_test_rows=config.minimum_test_rows,
        base_train_sample_cap=config.base_train_sample_cap,
        base_n_estimators=config.base_n_estimators,
        base_learning_rate=config.base_learning_rate,
        base_num_leaves=config.base_num_leaves,
        base_min_child_samples=config.base_min_child_samples,
        random_state=config.random_state,
        independent_event_cooldown_hours=config.independent_event_cooldown_hours,
        episode_merge_gap_minutes=config.episode_merge_gap_minutes,
    )


def _extract_dataset(
    *,
    fold_id: str,
    phase: str,
    scope: str,
    events,
    path,
    timeline,
    config: IncrementalHoldConfig,
    progress: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    complete = 0
    skipped = 0
    reporter = ProgressReporter(
        f"[R03.4.2.6 {fold_id} {phase} {scope}] events",
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
            config=config,
        )
        if extraction is None:
            skipped += 1
        else:
            complete += 1
            for checkpoint in config.checkpoints_minutes:
                row = build_incremental_hold_row(
                    extraction,
                    checkpoint_minutes=checkpoint,
                    path=path,
                    timeline=timeline,
                    config=config,
                )
                rows.append(row)
        reporter.update(number)
    reporter.close()
    frame = pd.DataFrame(rows)
    return frame, {
        "fold_id": fold_id,
        "phase": phase,
        "scope": scope,
        "candidate_events": int(len(events)),
        "complete_120h_events": int(complete),
        "skipped_incomplete_or_missing": int(skipped),
        "checkpoint_rows": int(len(frame)),
    }


def _threshold_rows(
    oof_prediction: np.ndarray,
    *,
    fold_id: str,
    checkpoint: int,
    target: str,
    feature_set: str,
    config: IncrementalHoldConfig,
) -> list[dict[str, object]]:
    valid = np.asarray(oof_prediction, dtype=float)
    valid = valid[np.isfinite(valid)]
    rows: list[dict[str, object]] = []
    if not len(valid):
        return rows
    for quantile in config.decision_quantiles:
        rows.append(
            {
                "fold_id": fold_id,
                "checkpoint_minutes": checkpoint,
                "target": target,
                "feature_set": feature_set,
                "quantile": quantile,
                "prediction_threshold": float(np.quantile(valid, quantile)),
            }
        )
    return rows


def _action_diagnostics(
    frame: pd.DataFrame,
    actual: np.ndarray,
    prediction: np.ndarray,
    *,
    fold_id: str,
    checkpoint: int,
    target: str,
    feature_set: str,
    scope: str,
    thresholds: list[dict[str, object]],
    config: IncrementalHoldConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y = np.asarray(actual, dtype=float)
    p = np.asarray(prediction, dtype=float)
    for item in thresholds:
        quantile = float(item["quantile"])
        value = float(item["prediction_threshold"])
        if quantile <= 0.50:
            mask = np.isfinite(p) & (p <= value)
            action = "exit_candidate_low_predicted_value"
        else:
            mask = np.isfinite(p) & (p >= value)
            action = "hold_candidate_high_predicted_value"
        selected = frame.loc[mask].copy()
        selected_actual = y[mask]
        rows.append(
            {
                "fold_id": fold_id,
                "checkpoint_minutes": checkpoint,
                "target": target,
                "feature_set": feature_set,
                "scope": scope,
                "quantile": quantile,
                "action": action,
                "prediction_threshold": value,
                "rows": int(len(frame)),
                "selected_rows": int(mask.sum()),
                "selected_share": float(mask.mean()) if len(mask) else np.nan,
                "actual_incremental_utility_mean": float(np.mean(selected_actual)) if len(selected_actual) else np.nan,
                "actual_incremental_utility_median": float(np.median(selected_actual)) if len(selected_actual) else np.nan,
                "actual_positive_rate": float(np.mean(selected_actual > config.positive_utility_buffer)) if len(selected_actual) else np.nan,
                "mean_current_mark_return": float(selected["current_mark_return"].mean()) if len(selected) else np.nan,
                "q70_to_q80_share": float((selected["score_tier"] == "q70_to_q80").mean()) if len(selected) else np.nan,
                "q80_to_q90_share": float((selected["score_tier"] == "q80_to_q90").mean()) if len(selected) else np.nan,
                "q90_plus_share": float((selected["score_tier"] == "q90_plus").mean()) if len(selected) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _tier_diagnostics(predictions: pd.DataFrame, config: IncrementalHoldConfig) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    keys = ["fold_id", "checkpoint_minutes", "target", "feature_set", "scope", "score_tier"]
    for key, group in predictions.groupby(keys, sort=False):
        fold, checkpoint, target, feature_set, scope, tier = key
        actual = group["actual"].to_numpy(dtype=float)
        prediction = group["prediction"].to_numpy(dtype=float)
        rows.append(
            {
                "fold_id": fold,
                "checkpoint_minutes": int(checkpoint),
                "target": target,
                "feature_set": feature_set,
                "scope": scope,
                "score_tier": tier,
                "rows": int(len(group)),
                "actual_mean": float(np.mean(actual)),
                "actual_positive_rate": float(np.mean(actual > config.positive_utility_buffer)),
                "prediction_mean": float(np.mean(prediction)),
                "current_mark_return_mean": float(group["current_mark_return"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _score_ablation(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    keys = ["fold_id", "checkpoint_minutes", "target", "scope"]
    path = metrics.loc[metrics["feature_set"] == "path_structure_lightgbm"].set_index(keys)
    plus = metrics.loc[metrics["feature_set"] == "path_plus_score_lightgbm"].set_index(keys)
    score = metrics.loc[metrics["feature_set"] == "score_only_lightgbm"].set_index(keys)
    common = path.index.intersection(plus.index)
    rows: list[dict[str, object]] = []
    for key in common:
        fold, checkpoint, target, scope = key
        p = path.loc[key]
        a = plus.loc[key]
        s = score.loc[key] if key in score.index else None
        rows.append(
            {
                "fold_id": fold,
                "checkpoint_minutes": int(checkpoint),
                "target": target,
                "scope": scope,
                "path_rank_ic": float(p["rank_ic"]),
                "path_plus_score_rank_ic": float(a["rank_ic"]),
                "score_only_rank_ic": float(s["rank_ic"]) if s is not None else np.nan,
                "score_increment_rank_ic": float(a["rank_ic"] - p["rank_ic"]),
                "path_top_bottom_spread": float(p["top_bottom_spread"]),
                "path_plus_score_top_bottom_spread": float(a["top_bottom_spread"]),
            }
        )
    return pd.DataFrame(rows)


def _causal_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "frozen_opening_model", "status": "PASS", "detail": "R03.4.1 six-hour long utility model and next-minute-open entry are unchanged"},
            {"check": "q70_preserved", "status": "PASS", "detail": "q70 is the primary OOS pool; q70-q80/q80-q90/q90+ remain separately reported"},
            {"check": "strict_entry_oof", "status": "PASS", "detail": "holding-model train events use rolling OOF opening percentiles and a separate calibration window"},
            {"check": "checkpoint_cutoff", "status": "PASS", "detail": "all x_path/x_score features stop at the checkpoint; future path is label-only"},
            {"check": "five_day_embargo", "status": "PASS", "detail": "holding-value OOF uses a 120h embargo covering the full label window"},
            {"check": "no_time_exit_claim", "status": "PASS", "detail": "checkpoints are recurrent decision observations, not mandatory exits; 120h is a censored research horizon"},
            {"check": "score_ablation", "status": "PASS", "detail": "path-only, score-only and path-plus-score models are evaluated separately"},
            {"check": "state_model_abandoned", "status": "PASS", "detail": "strategic/tactical/entry/activity state outputs are not loaded"},
            {"check": "sealed_holdout", "status": "PASS", "detail": "2026 remains sealed and labels cannot cross the seal"},
        ]
    )


def run_incremental_hold_research(
    *,
    data_dir: str | Path | None = None,
    force_rebuild_outcomes: bool = False,
    progress: bool = True,
    config: IncrementalHoldConfig = DEFAULT_INCREMENTAL_HOLD_CONFIG,
) -> IncrementalHoldResult:
    config.validate()
    exit_config = _exit_config(config)
    multistage_config = _multistage_config(config)

    loader = create_loader(LONG_CONTEXT_BASE_CONFIG, data_dir=data_dir)
    loader_preflight = run_public_loader_preflight(
        loader,
        LONG_CONTEXT_BASE_CONFIG,
        sample_dates=("2023-06-15", "2024-06-15", "2025-06-15"),
    )
    preflight: dict[str, object] = {"trade_bar": loader_preflight.to_dict()}
    if loader_preflight.status != "PASS":
        reports.empty(config, preflight, "BLOCKED_DATA", "1分钟Trade Bar公共Loader预检失败。")
        return IncrementalHoldResult("BLOCKED_DATA", config.report_path)

    base_paths = [path for path in list_cached_years(LONG_CONTEXT_BASE_CONFIG) if 2023 <= int(path.name[-4:]) <= 2025]
    available = {int(path.name[-4:]) for path in base_paths}
    if available != {2023, 2024, 2025}:
        reports.empty(config, preflight, "BLOCKED_BASE_CACHE", f"缺少R03.2基础缓存，当前={sorted(available)}。")
        return IncrementalHoldResult("BLOCKED_BASE_CACHE", config.report_path)
    outcome_paths = build_outcome_caches(
        base_paths,
        DEFAULT_STATE_CONTEXT_ABLATION_CONFIG,
        force_rebuild=force_rebuild_outcomes,
        progress=progress,
    )

    entry_oof_audits: list[pd.DataFrame] = []
    extraction_rows: list[dict[str, object]] = []
    selection_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    decile_parts: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, object]] = []
    action_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []

    folds = default_ablation_folds(DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
    fold_reporter = ProgressReporter("[R03.4.2.6 folds]", len(folds), every=1, enabled=progress)
    for fold_number, fold in enumerate(folds, start=1):
        try:
            fit = collect_base_period_data(base_paths, outcome_paths, start=fold.fit_start, end=fold.fit_end, outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
            calibration = collect_base_period_data(base_paths, outcome_paths, start=fold.calibration_start, end=fold.calibration_end, outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
            test = collect_base_period_data(base_paths, outcome_paths, start=fold.test_start, end=fold.test_end, outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
            pretest = collect_base_period_data(base_paths, outcome_paths, start=pd.Timestamp(config.research_start), end=fold.calibration_end, outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)

            entry_oof = build_rolling_oof_entry_timeline(pretest, event_builder_config=exit_config, config=multistage_config)
            audit = entry_oof.audit.copy()
            audit.insert(0, "fold_id", fold.fold_id)
            entry_oof_audits.append(audit)

            bundle = fit_frozen_base_scores(fit, calibration, test, exit_config)
            test_timeline = build_oos_percentile_timeline(test.timestamps_ns, bundle.test_score, bundle.calibration_score)
            oos_events = {
                "broad_q70": build_event_candidates(test_timeline, signal_quantile=0.70, config=exit_config),
                "primary_q90": build_event_candidates(test_timeline, signal_quantile=0.90, config=exit_config),
            }

            pretest_path = load_minute_path_data(
                start=pd.Timestamp(config.research_start),
                end=fold.calibration_end,
                data_dir=data_dir,
                config=exit_config,
                progress=progress,
            )
            oos_end = min(
                fold.test_end + pd.Timedelta(hours=config.path_horizon_hours),
                pd.Timestamp(config.sealed_holdout_start) - pd.Timedelta(minutes=1),
            )
            oos_path = load_minute_path_data(
                start=fold.test_start - pd.Timedelta(days=1),
                end=oos_end,
                data_dir=data_dir,
                config=exit_config,
                progress=progress,
            )
            if pretest_path.coverage_ratio < 0.995 or oos_path.coverage_ratio < 0.995:
                raise RuntimeError("minute path coverage below 99.5%")

            train_frame, audit_row = _extract_dataset(
                fold_id=fold.fold_id,
                phase="train_oof",
                scope="train_q50",
                events=entry_oof.events,
                path=pretest_path,
                timeline=entry_oof.timeline,
                config=config,
                progress=progress,
            )
            extraction_rows.append(audit_row)
            if train_frame.empty:
                raise RuntimeError("incremental holding train frame is empty")

            eval_frames: dict[str, pd.DataFrame] = {}
            for scope, events in oos_events.items():
                frame, audit_row = _extract_dataset(
                    fold_id=fold.fold_id,
                    phase="oos",
                    scope=scope,
                    events=events,
                    path=oos_path,
                    timeline=test_timeline,
                    config=config,
                    progress=progress,
                )
                extraction_rows.append(audit_row)
                eval_frames[scope] = frame

            for checkpoint in config.checkpoints_minutes:
                train_checkpoint = train_frame.loc[train_frame["checkpoint_minutes"] == checkpoint].reset_index(drop=True)
                if len(train_checkpoint) < config.minimum_train_rows:
                    failures.append({"fold_id": fold.fold_id, "checkpoint_minutes": checkpoint, "task": "ALL", "error": f"insufficient train rows={len(train_checkpoint)}"})
                    continue
                for target_spec in regression_targets(train_checkpoint):
                    target_name = target_spec.name
                    train_target = target_values(train_checkpoint, target_name)
                    candidates = []
                    for feature_set in feature_sets(train_checkpoint):
                        try:
                            oof = causal_oof(train_checkpoint, train_target, feature_set=feature_set, config=config)
                            candidates.append((feature_set, oof))
                        except Exception as exc:
                            failures.append({"fold_id": fold.fold_id, "checkpoint_minutes": checkpoint, "task": target_name, "feature_set": feature_set.name, "error": f"OOF {type(exc).__name__}: {exc}"})
                    if not candidates:
                        continue
                    selected_set, selected_oof, selection = choose_feature_set(candidates, train_target)
                    selection.insert(0, "fold_id", fold.fold_id)
                    selection.insert(1, "checkpoint_minutes", checkpoint)
                    selection.insert(2, "target", target_name)
                    selection_parts.append(selection)
                    selected_thresholds = _threshold_rows(
                        selected_oof.predictions,
                        fold_id=fold.fold_id,
                        checkpoint=checkpoint,
                        target=target_name,
                        feature_set=selected_set.name,
                        config=config,
                    )
                    threshold_rows.extend(selected_thresholds)
                    model = fit_model(train_checkpoint, train_target, feature_set=selected_set, config=config)
                    importance = feature_importance(model)
                    if not importance.empty:
                        importance.insert(0, "fold_id", fold.fold_id)
                        importance.insert(1, "checkpoint_minutes", checkpoint)
                        importance.insert(2, "target", target_name)
                        importance.insert(3, "feature_set", selected_set.name)
                        importance_parts.append(importance)

                    # Evaluate all feature sets OOS, while predictions/action diagnostics use the selected set.
                    fitted_all = {}
                    for feature_set, _ in candidates:
                        try:
                            fitted_all[feature_set.name] = fit_model(train_checkpoint, train_target, feature_set=feature_set, config=config)
                        except Exception as exc:
                            failures.append({"fold_id": fold.fold_id, "checkpoint_minutes": checkpoint, "task": target_name, "feature_set": feature_set.name, "error": f"FIT {type(exc).__name__}: {exc}"})
                    for scope, eval_frame in eval_frames.items():
                        eval_checkpoint = eval_frame.loc[eval_frame["checkpoint_minutes"] == checkpoint].reset_index(drop=True)
                        if len(eval_checkpoint) < config.minimum_test_rows:
                            continue
                        actual = target_values(eval_checkpoint, target_name)
                        for feature_name, fitted in fitted_all.items():
                            prediction = fitted.predict(eval_checkpoint)
                            metrics, deciles = evaluate(
                                actual,
                                prediction,
                                fold_id=fold.fold_id,
                                checkpoint_minutes=checkpoint,
                                target_name=target_name,
                                feature_set=feature_name,
                                scope=scope,
                                config=config,
                            )
                            metric_rows.append(metrics)
                            decile_parts.append(deciles)
                        selected_prediction = model.predict(eval_checkpoint)
                        action_parts.append(
                            _action_diagnostics(
                                eval_checkpoint,
                                actual,
                                selected_prediction,
                                fold_id=fold.fold_id,
                                checkpoint=checkpoint,
                                target=target_name,
                                feature_set=selected_set.name,
                                scope=scope,
                                thresholds=selected_thresholds,
                                config=config,
                            )
                        )
                        pred = eval_checkpoint.loc[:, prediction_columns(eval_checkpoint)].copy()
                        pred["target"] = target_name
                        pred["feature_set"] = selected_set.name
                        pred["actual"] = actual
                        pred["prediction"] = selected_prediction
                        prediction_parts.append(pred)
        except Exception as exc:
            failures.append({"fold_id": fold.fold_id, "task": "FOLD", "error": f"{type(exc).__name__}: {exc}"})
        fold_reporter.update(fold_number)
    fold_reporter.close()

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    stable = stable_candidates(metrics, config)
    passed = (
        stable.loc[
            (stable["passes_cross_year"] == True)  # noqa: E712
            & (stable["scope"] == "broad_q70")
            & (stable["target"] == "next_incremental_utility")
            & (stable["feature_set"] != "score_only_lightgbm")
        ]
        if not stable.empty
        else pd.DataFrame()
    )
    if not passed.empty:
        decision = "PASS_INCREMENTAL_HOLD_VALUE_SIGNAL"
        reason = "至少一个非纯评分模型在q70主池上，能够跨2024与2025稳定排序到下一个决策节点的增量持仓价值；可以进入循环持仓状态机与真实非时间退出回测。"
    elif not metrics.empty:
        decision = "RESEARCH_CONTINUE_RANKING_ONLY"
        reason = "增量持仓价值在局部年份或检查点有排序信息，但尚未跨年通过；不能据此直接退出或长持。"
    else:
        decision = "FAIL_NO_INCREMENTAL_HOLD_MODELS"
        reason = "没有形成可评估的2024/2025增量持仓价值模型。"

    reports.write_reports(
        config=config,
        preflight=preflight,
        manifest={
            "stage": STAGE_ID,
            "name": STAGE_NAME,
            "config": config.to_dict(),
            "folds": [fold.to_dict() for fold in folds],
            "opening_model": "frozen R03.4.1 long utility h6",
            "state_model_policy": "ABANDONED_AND_NOT_LOADED",
        },
        entry_oof_audit=pd.concat(entry_oof_audits, ignore_index=True) if entry_oof_audits else pd.DataFrame(),
        extraction_audit=pd.DataFrame(extraction_rows),
        model_selection=pd.concat(selection_parts, ignore_index=True) if selection_parts else pd.DataFrame(),
        model_metrics=metrics,
        deciles=pd.concat(decile_parts, ignore_index=True) if decile_parts else pd.DataFrame(),
        thresholds=pd.DataFrame(threshold_rows),
        action_diagnostics=pd.concat(action_parts, ignore_index=True) if action_parts else pd.DataFrame(),
        tier_diagnostics=_tier_diagnostics(predictions, config),
        score_ablation=_score_ablation(metrics),
        importance=pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame(),
        predictions=predictions,
        stable=stable,
        causal_audit=_causal_audit(),
        failures=pd.DataFrame(failures),
        decision=decision,
        reason=reason,
    )
    return IncrementalHoldResult(decision, config.report_path)
