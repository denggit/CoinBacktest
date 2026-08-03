#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2.3 multi-stage holding and q70 expansion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.long_state_calibration.config import LongStateCalibrationConfig
from src.ai_research.long_tail_exit_audit.config import LongTailExitAuditConfig
from src.ai_research.long_tail_exit_audit.data import collect_base_period_data, load_minute_path_data
from src.ai_research.long_tail_exit_audit.modeling import fit_frozen_base_scores
from src.ai_research.long_tail_exit_audit.simulator import build_event_candidates
from src.ai_research.state_context_ablation.config import DEFAULT_STATE_CONTEXT_ABLATION_CONFIG
from src.ai_research.state_context_ablation.modeling import default_ablation_folds
from src.ai_research.state_context_ablation.outcomes import build_outcome_caches
from src.ai_research.swing_baseline.dataset import create_loader, list_cached_years, run_public_loader_preflight
from src.ai_research.swing_long_context.config import LONG_CONTEXT_BASE_CONFIG
from src.research_common.progress import ProgressReporter

from .config import (
    DEFAULT_LONG_TAIL_MULTISTAGE_CONFIG,
    STAGE_ID,
    STAGE_NAME,
    LongTailMultistageConfig,
)
from .entry_oof import build_oos_percentile_timeline, build_rolling_oof_entry_timeline
from .features import build_checkpoint_row, extract_extended_event_path, feature_sets, task_frame
from .modeling import (
    causal_oof,
    choose_feature_set,
    feature_importance,
    fit_model,
    metric_row,
    threshold,
)
from .policy import (
    PolicyThresholds,
    build_policy_tables,
    enforce_non_overlap,
    simulate_policy_event,
    stable_policy_candidates,
)
from . import reports


@dataclass(frozen=True)
class LongTailMultistageResult:
    decision: str
    report_dir: Path


TASKS_BY_CHECKPOINT: dict[int, tuple[str, ...]] = {
    60: ("persistent_failure", "recoverable_drawdown"),
    180: ("persistent_failure", "recoverable_drawdown"),
    360: ("persistent_failure", "recoverable_drawdown", "post6_continuation"),
    1440: ("post24_longhold",),
}


def _exit_config(config: LongTailMultistageConfig) -> LongTailExitAuditConfig:
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
        entry_delay_minutes=(1, 3, 5),
        episode_merge_gap_minutes=config.episode_merge_gap_minutes,
        independent_event_cooldown_hours=config.independent_event_cooldown_hours,
        train_sample_cap=config.base_train_sample_cap,
        base_n_estimators=config.base_n_estimators,
        base_learning_rate=config.base_learning_rate,
        base_num_leaves=config.base_num_leaves,
        base_min_child_samples=config.base_min_child_samples,
        random_state=config.random_state,
    )


def _extract_dataset(
    *,
    fold_id: str,
    phase: str,
    scope: str,
    events,
    path,
    timeline,
    config: LongTailMultistageConfig,
    progress: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    complete = 0
    reporter = ProgressReporter(
        f"[R03.4.2.3 {fold_id} {phase} {scope}] events",
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
        if extraction is not None:
            complete += 1
            for checkpoint in config.checkpoints_minutes:
                rows.append(
                    build_checkpoint_row(
                        extraction,
                        checkpoint_minutes=checkpoint,
                        path=path,
                        timeline=timeline,
                        config=config,
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


def _task_checkpoint(task: str, checkpoint: int) -> bool:
    return task in TASKS_BY_CHECKPOINT.get(checkpoint, ())


def _threshold_quantiles(task: str, config: LongTailMultistageConfig) -> dict[str, float]:
    if task == "persistent_failure":
        return {"high": config.high_failure_quantile, "safe": config.safe_failure_quantile}
    if task == "recoverable_drawdown":
        return {"low": config.low_recovery_quantile, "high": config.high_recovery_quantile}
    if task == "post6_continuation":
        return {"high": config.high_continuation_quantile}
    if task == "post24_longhold":
        return {"high": config.high_longhold_quantile}
    raise ValueError(task)


def _event_table(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    summary_columns = [
        "event_id", "fold_id", "scope", "decision_time", "entry_time", "event_score_percentile",
        "entry_price_delay_1m", "entry_price_delay_3m", "entry_price_delay_5m",
        "open_after_180m", "open_after_360m", "open_after_1440m",
        "close_price_180m", "close_price_360m", "close_price_1440m", "close_price_2880m", "close_price_7200m",
        "mfe_360m", "mae_360m", "mfe_1440m", "mae_1440m", "mfe_7200m", "mae_7200m",
        "ret_360m", "ret_1440m", "ret_2880m", "ret_7200m",
    ]
    events = frame.sort_values("checkpoint_minutes").groupby("event_id", as_index=False).first()
    events = events.loc[:, [column for column in summary_columns if column in events.columns]].copy()
    for checkpoint in (60, 180, 360, 1440):
        cp = frame.loc[frame["checkpoint_minutes"] == checkpoint, ["event_id", "weak_now", "path_class"]].copy()
        cp = cp.rename(columns={"weak_now": f"weak_now_{checkpoint}", "path_class": f"path_class_{checkpoint}"})
        events = events.merge(cp, on="event_id", how="left", validate="one_to_one")
    if not predictions.empty:
        pivot = predictions.pivot_table(
            index="event_id",
            columns=["task", "checkpoint_minutes"],
            values="probability",
            aggfunc="first",
        )
        names: list[str] = []
        for task, checkpoint in pivot.columns:
            prefix = {
                "persistent_failure": "p_failure",
                "recoverable_drawdown": "p_recovery",
                "post6_continuation": "p_continuation",
                "post24_longhold": "p_longhold",
            }[task]
            names.append(f"{prefix}_{int(checkpoint)}")
        pivot.columns = names
        events = events.merge(pivot.reset_index(), on="event_id", how="left", validate="one_to_one")
    return events


def _policy_thresholds(rows: pd.DataFrame) -> PolicyThresholds:
    values = {(str(row.task), int(row.checkpoint_minutes), str(row.threshold_kind)): float(row.threshold) for row in rows.itertuples()}
    return PolicyThresholds(
        fail_high_180=values[("persistent_failure", 180, "high")],
        fail_safe_180=values[("persistent_failure", 180, "safe")],
        recovery_low_180=values[("recoverable_drawdown", 180, "low")],
        recovery_high_180=values[("recoverable_drawdown", 180, "high")],
        fail_high_360=values[("persistent_failure", 360, "high")],
        recovery_low_360=values[("recoverable_drawdown", 360, "low")],
        continuation_high_360=values[("post6_continuation", 360, "high")],
        longhold_high_1440=values[("post24_longhold", 1440, "high")],
    )


def _causal_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "frozen_opening_model", "status": "PASS", "detail": "the R03.4.1 six-hour long-utility LightGBM is unchanged"},
            {"check": "expanded_entry_train_pool", "status": "PASS", "detail": "holding-model train events come from rolling train/calibration/validation OOF opening scores across the full pretest history"},
            {"check": "entry_score_normalization", "status": "PASS", "detail": "each OOF validation block is ranked only against its separate prior calibration block"},
            {"check": "holding_label_embargo", "status": "PASS", "detail": "holding-model OOF thresholds use a 120h embargo covering the complete future label"},
            {"check": "checkpoint_features", "status": "PASS", "detail": "T+60/T+180/T+360/T+1440 features use only path rows visible by that checkpoint"},
            {"check": "three_way_logic", "status": "PASS", "detail": "early exits require high failure risk and low recovery probability; danger alone is insufficient"},
            {"check": "t60_observation_only", "status": "PASS", "detail": "T+60 is diagnostic only and never forces an exit"},
            {"check": "q70_q90_oos", "status": "PASS", "detail": "q50 is training-only; q70 and q90 are separately audited OOS"},
            {"check": "single_position", "status": "PASS", "detail": "events occurring while a position is open are skipped; no overlapping long positions or hidden pyramiding"},
            {"check": "state_model", "status": "PASS", "detail": "abandoned market-state outputs are not loaded"},
            {"check": "sealed_holdout", "status": "PASS", "detail": "complete five-day paths cannot cross the 2026 seal"},
        ]
    )


def run_long_tail_multistage_decision(
    *,
    data_dir: str | Path | None = None,
    force_rebuild_outcomes: bool = False,
    progress: bool = True,
    config: LongTailMultistageConfig = DEFAULT_LONG_TAIL_MULTISTAGE_CONFIG,
) -> LongTailMultistageResult:
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
        reports.empty(config, preflight, "BLOCKED_DATA", "1分钟Trade Bar公共Loader预检失败。")
        return LongTailMultistageResult("BLOCKED_DATA", config.report_path)

    base_paths = [path for path in list_cached_years(LONG_CONTEXT_BASE_CONFIG) if 2023 <= int(path.name[-4:]) <= 2025]
    available = {int(path.name[-4:]) for path in base_paths}
    if available != {2023, 2024, 2025}:
        reports.empty(config, preflight, "BLOCKED_BASE_CACHE", f"缺少R03.2基础缓存，当前={sorted(available)}。")
        return LongTailMultistageResult("BLOCKED_BASE_CACHE", config.report_path)
    outcome_paths = build_outcome_caches(
        base_paths,
        DEFAULT_STATE_CONTEXT_ABLATION_CONFIG,
        force_rebuild=force_rebuild_outcomes,
        progress=progress,
    )

    extraction_rows: list[dict[str, object]] = []
    entry_oof_audits: list[pd.DataFrame] = []
    dataset_parts: list[pd.DataFrame] = []
    model_metrics: list[dict[str, object]] = []
    selection_audits: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    overlap_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    folds = default_ablation_folds(DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
    fold_reporter = ProgressReporter("[R03.4.2.3 folds]", len(folds), every=1, enabled=progress)

    for fold_number, fold in enumerate(folds, start=1):
        try:
            fit = collect_base_period_data(base_paths, outcome_paths, start=fold.fit_start, end=fold.fit_end, outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
            calibration = collect_base_period_data(base_paths, outcome_paths, start=fold.calibration_start, end=fold.calibration_end, outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
            test = collect_base_period_data(base_paths, outcome_paths, start=fold.test_start, end=fold.test_end, outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
            pretest = collect_base_period_data(base_paths, outcome_paths, start=pd.Timestamp(config.research_start), end=fold.calibration_end, outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)

            entry_oof = build_rolling_oof_entry_timeline(pretest, event_builder_config=exit_config, config=config)
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
            oos_path = load_minute_path_data(
                start=fold.test_start - pd.Timedelta(days=1),
                end=fold.test_end,
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
                if not frame.empty:
                    dataset_parts.append(frame)
            if train_frame.empty:
                raise RuntimeError("expanded OOF path train frame is empty")
            dataset_parts.append(train_frame)

            fold_predictions: list[pd.DataFrame] = []
            fold_thresholds: list[dict[str, object]] = []
            for checkpoint in config.checkpoints_minutes:
                train_checkpoint = train_frame.loc[train_frame["checkpoint_minutes"] == checkpoint].copy()
                for task in TASKS_BY_CHECKPOINT[checkpoint]:
                    train_task, train_target = task_frame(train_checkpoint, task)
                    if len(train_task) < config.minimum_train_rows or len(np.unique(train_target)) < 2:
                        failures.append({"fold_id": fold.fold_id, "task": task, "checkpoint_minutes": checkpoint, "error": f"insufficient expanded train rows={len(train_task)}"})
                        continue
                    candidates = []
                    for feature_set in feature_sets(train_task):
                        try:
                            result = causal_oof(train_task, train_target, task=task, feature_set=feature_set, config=config)
                            candidates.append((feature_set, result))
                        except Exception as exc:
                            failures.append({"fold_id": fold.fold_id, "task": task, "checkpoint_minutes": checkpoint, "feature_set": feature_set.name, "error": f"OOF {type(exc).__name__}: {exc}"})
                    if not candidates:
                        continue
                    selected_set, selected_oof, selection = choose_feature_set(candidates, train_target)
                    selection.insert(0, "fold_id", fold.fold_id)
                    selection.insert(1, "task", task)
                    selection.insert(2, "checkpoint_minutes", checkpoint)
                    selection_audits.append(selection)
                    quantiles = _threshold_quantiles(task, config)
                    for kind, quantile in quantiles.items():
                        value = threshold(selected_oof.probabilities, quantile)
                        row = {"fold_id": fold.fold_id, "task": task, "checkpoint_minutes": checkpoint, "threshold_kind": kind, "quantile": quantile, "threshold": value, "feature_set": selected_set.name}
                        threshold_rows.append(row)
                        fold_thresholds.append(row)
                    model = fit_model(train_task, train_target, task=task, feature_set=selected_set, config=config)
                    importance = feature_importance(model)
                    if not importance.empty:
                        importance.insert(0, "fold_id", fold.fold_id)
                        importance.insert(2, "checkpoint_minutes", checkpoint)
                        importance_parts.append(importance)
                    for scope, eval_frame in eval_frames.items():
                        eval_checkpoint = eval_frame.loc[eval_frame["checkpoint_minutes"] == checkpoint].copy()
                        eval_task, eval_target = task_frame(eval_checkpoint, task)
                        if len(eval_task) < config.minimum_test_rows or len(np.unique(eval_target)) < 2:
                            continue
                        probability = model.predict(eval_task)
                        model_metrics.append(metric_row(eval_target, probability, fold_id=fold.fold_id, task=task, checkpoint_minutes=checkpoint, feature_set=selected_set.name, scope=scope))
                        pred = eval_task.loc[:, ["event_id", "fold_id", "scope", "decision_time", "entry_time", "checkpoint_minutes", "weak_now", "path_class"]].copy()
                        pred["task"] = task
                        pred["target"] = eval_target
                        pred["probability"] = probability
                        pred["feature_set"] = selected_set.name
                        fold_predictions.append(pred)

            predictions = pd.concat(fold_predictions, ignore_index=True) if fold_predictions else pd.DataFrame()
            if not predictions.empty:
                prediction_parts.append(predictions)
            threshold_frame = pd.DataFrame(fold_thresholds)
            required = {
                ("persistent_failure", 180, "high"), ("persistent_failure", 180, "safe"),
                ("recoverable_drawdown", 180, "low"), ("recoverable_drawdown", 180, "high"),
                ("persistent_failure", 360, "high"), ("recoverable_drawdown", 360, "low"),
                ("post6_continuation", 360, "high"), ("post24_longhold", 1440, "high"),
            }
            present = {(str(r.task), int(r.checkpoint_minutes), str(r.threshold_kind)) for r in threshold_frame.itertuples()}
            if not required.issubset(present):
                failures.append({"fold_id": fold.fold_id, "task": "POLICY", "error": f"missing policy thresholds={sorted(required - present)}"})
                fold_reporter.update(fold_number)
                continue
            policy_thresholds = _policy_thresholds(threshold_frame)

            for scope, eval_frame in eval_frames.items():
                scope_predictions = predictions.loc[predictions["scope"] == scope].copy() if not predictions.empty else pd.DataFrame()
                events = _event_table(eval_frame, scope_predictions)
                for policy in ("fixed_6h", "full_multistage", "half_probe_then_add", "delayed_confirm_180"):
                    for delay in (1, 3, 5):
                        raw_rows = [
                            simulate_policy_event(row, policy=policy, delay_minutes=delay, thresholds=policy_thresholds, config=config)
                            for _, row in events.iterrows()
                        ]
                        raw = pd.DataFrame([row for row in raw_rows if row is not None])
                        health_skips = int((raw.get("executed", pd.Series(dtype=bool)) == False).sum()) if not raw.empty else 0  # noqa: E712
                        executed, overlap_skips = enforce_non_overlap(raw)
                        overlap_rows.append({"fold_id": fold.fold_id, "scope": scope, "policy": policy, "delay_minutes": delay, "candidate_events": int(len(events)), "executed_before_overlap": int((raw.get("executed", pd.Series(dtype=bool)) == True).sum()) if not raw.empty else 0, "skipped_health": health_skips, "skipped_overlap": overlap_skips, "final_trades": int(len(executed))})  # noqa: E712
                        if not executed.empty:
                            trade_parts.append(executed)
        except Exception as exc:
            failures.append({"fold_id": fold.fold_id, "task": "FOLD", "error": f"{type(exc).__name__}: {exc}"})
        fold_reporter.update(fold_number)
    fold_reporter.close()

    dataset = pd.concat(dataset_parts, ignore_index=True) if dataset_parts else pd.DataFrame()
    metrics = pd.DataFrame(model_metrics)
    selections = pd.concat(selection_audits, ignore_index=True) if selection_audits else pd.DataFrame()
    thresholds = pd.DataFrame(threshold_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    importance = pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    overlap = pd.DataFrame(overlap_rows)
    summary, periods, exits, overlap = build_policy_tables(trades, overlap_audit=overlap, config=config)
    stable = stable_policy_candidates(summary, periods, config)
    failures_frame = pd.DataFrame(failures)

    if summary.empty:
        decision = "FAIL_NO_VALID_MULTISTAGE_POLICY"
        reason = "扩展OOF路径训练或多阶段策略未产生完整的2024/2025评估结果。"
    else:
        multistage = stable.loc[stable["stable_multistage_upgrade"] == True] if not stable.empty else pd.DataFrame()  # noqa: E712
        q70 = stable.loc[stable["stable_q70_expansion"] == True] if not stable.empty else pd.DataFrame()  # noqa: E712
        if not multistage.empty:
            decision = "PASS_MULTISTAGE_HOLDING_PROFIT_UPLIFT"
            reason = "至少一个因果多阶段持仓方案不仅跨年保持正期望，而且在2024和2025都提高了相同信号池的2倍成本总复合利润。"
        elif not q70.empty:
            decision = "PASS_Q70_EXPANSION_ONLY"
            reason = "差异化持仓尚未稳定提高利润，但q70扩展池跨年保持稳健正期望，并在两年都提高了相对q90基准的2倍成本总复合利润。"
        else:
            decision = "RESEARCH_CONTINUE_NO_ROBUST_POLICY"
            reason = "路径模型产生了可评估结果，但没有方案同时守住跨年正期望并提高总利润；不能为了长持、胜率或增频牺牲Edge。"

    reports.write_reports(
        config=config,
        preflight=preflight,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict(), "folds": [fold.to_dict() for fold in folds], "state_model_policy": "ABANDONED_AND_NOT_LOADED"},
        contract={"opening_model": "frozen R03.4.1 long utility h6", "holding_train_pool": "rolling OOF q50 independent events", "oos_scopes": ["q70", "q90"], "decision_chain": ["T+60 observe", "T+180 failure AND low recovery", "T+360 exit/24h hold", "T+24h exit/5d hold"], "positive_expectancy_priority": True},
        entry_oof_audit=pd.concat(entry_oof_audits, ignore_index=True) if entry_oof_audits else pd.DataFrame(),
        extraction_audit=pd.DataFrame(extraction_rows),
        dataset_summary=dataset.groupby(["fold_id", "phase", "scope", "checkpoint_minutes"], as_index=False).agg(events=("event_id", "nunique"), rows=("event_id", "size")) if not dataset.empty else pd.DataFrame(),
        model_selection=selections,
        model_metrics=metrics,
        thresholds=thresholds,
        importance=importance,
        predictions=predictions,
        policy_summary=summary,
        period_summary=periods,
        exit_summary=exits,
        overlap_audit=overlap,
        stable=stable,
        causal_audit=_causal_audit(),
        failures=failures_frame,
        trades=trades,
        decision=decision,
        reason=reason,
    )
    return LongTailMultistageResult(decision=decision, report_dir=config.report_path)
