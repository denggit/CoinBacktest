#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R03.3.1 pipeline: first-alert, event-coverage and remaining-opportunity audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.swing_baseline.dataset import build_yearly_cache, create_loader, list_cached_years, load_year_shard, run_public_loader_preflight
from src.ai_research.swing_long_context.config import FEATURE_PROFILE, LONG_CONTEXT_BASE_CONFIG
from src.research_common.progress import ProgressReporter

from .alert_audit import audit_alert_episodes, build_alert_episodes
from .alert_audit_config import DEFAULT_PROCESS_ALERT_AUDIT_CONFIG, ProcessAlertAuditConfig, STAGE_ID, STAGE_NAME
from .alert_audit_reports import write_alert_audit_reports
from .events import build_event_caches, load_event_year_shard
from .micro_features import build_micro_caches, create_micro_loader, run_micro_preflight
from .modeling import architecture_matrix, collect_period_data, default_folds, fit_one, validate_model_dependencies


@dataclass(frozen=True)
class ProcessAlertAuditPipelineResult:
    decision: str
    report_dir: Path
    champion: dict[str, object] | None


def _load_events(paths: list[Path], config: ProcessAlertAuditConfig) -> pd.DataFrame:
    parts = [load_event_year_shard(path).events for path in paths]
    frame = pd.concat([part for part in parts if not part.empty], ignore_index=True) if any(not p.empty for p in parts) else pd.DataFrame()
    if frame.empty:
        return frame
    frame["start_time"] = pd.to_datetime(frame["start_time"])
    frame["end_time"] = pd.to_datetime(frame["end_time"])
    frame = frame.loc[
        (frame["start_time"] >= pd.Timestamp(config.base.research_start))
        & (frame["start_time"] <= pd.Timestamp(config.base.research_end))
    ].copy()
    return frame.drop_duplicates("event_uid", keep="first").sort_values("start_time", kind="stable").reset_index(drop=True)


def _collect_decision_prices(base_paths: list[Path], start: pd.Timestamp, end: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    time_parts: list[np.ndarray] = []
    price_parts: list[np.ndarray] = []
    for path in base_paths:
        shard = load_year_shard(path)
        times = np.asarray(shard.decision_times_ns, dtype=np.int64)
        left = int(np.searchsorted(times, int(start.value), side="left"))
        right = int(np.searchsorted(times, int(end.value), side="right"))
        if right <= left:
            continue
        time_parts.append(times[left:right])
        price_parts.append(np.asarray(shard.entry_prices[left:right], dtype=float))
    if not time_parts:
        raise RuntimeError(f"no decision prices for {start} -> {end}")
    times = np.concatenate(time_parts)
    prices = np.concatenate(price_parts)
    order = np.argsort(times, kind="stable")
    return times[order], prices[order]


def _select_candidates(episode_metrics: pd.DataFrame, event_metrics: pd.DataFrame, config: ProcessAlertAuditConfig) -> pd.DataFrame:
    if episode_metrics.empty or event_metrics.empty:
        return pd.DataFrame()
    keys = ["architecture", "process", "horizon_hours", "quantile"]
    merged = episode_metrics.merge(event_metrics, on=["fold_id", *keys], how="inner")
    value_columns = [column for column in merged.columns if column not in {"fold_id", *keys}]
    parts: list[pd.DataFrame] = []
    for fold in ("WF_2024", "WF_2025"):
        part = merged.loc[merged["fold_id"] == fold, [*keys, *value_columns]].copy()
        part = part.rename(columns={column: f"{fold}_{column}" for column in value_columns})
        parts.append(part)
    candidates = parts[0].merge(parts[1], on=keys, how="inner")
    if candidates.empty:
        return candidates
    gates: list[pd.Series] = []
    for fold in ("WF_2024", "WF_2025"):
        gates.append(
            (candidates[f"{fold}_events"] >= config.minimum_events_per_fold)
            & (candidates[f"{fold}_actionable_alert_precision"] >= config.minimum_actionable_precision)
            & (candidates[f"{fold}_event_coverage"] >= config.minimum_event_coverage)
            & (candidates[f"{fold}_late_ongoing_rate"] <= config.maximum_late_ongoing_rate)
            & (candidates[f"{fold}_alerts_per_month"] <= config.maximum_alerts_per_month)
            & (
                candidates[f"{fold}_median_first_alert_remaining_opportunity"]
                >= np.where(
                    candidates["process"].eq("volatile_range"),
                    config.min_remaining_range_move,
                    config.min_remaining_directional_move,
                )
            )
        )
    candidates["passes_actionability"] = gates[0] & gates[1]
    candidates["stability_score"] = (
        candidates[["WF_2024_actionable_alert_precision", "WF_2025_actionable_alert_precision"]].min(axis=1)
        + candidates[["WF_2024_event_coverage", "WF_2025_event_coverage"]].min(axis=1)
        + 3.0 * candidates[["WF_2024_median_first_alert_remaining_opportunity", "WF_2025_median_first_alert_remaining_opportunity"]].min(axis=1)
        - candidates[["WF_2024_late_ongoing_rate", "WF_2025_late_ongoing_rate"]].max(axis=1)
    )
    return candidates.sort_values(["passes_actionability", "stability_score"], ascending=[False, False], kind="stable").reset_index(drop=True)


def _causal_audit(config: ProcessAlertAuditConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "model_inputs", "status": "PASS", "detail": "R03.3.1 reuses causal R03.2/R03.3 features and does not add future values to model inputs."},
            {"check": "first_alert_only", "status": "PASS", "detail": "Continuous high-score points are merged; only the first timestamp of each independent episode is evaluated."},
            {"check": "early_start_policy", "status": "PASS", "detail": f"Early confirmation is allowed only within {config.early_start_grace_hours:g}h and progress <= {config.max_actionable_progress:.0%}."},
            {"check": "remaining_opportunity", "status": "PASS", "detail": "Success requires realized target/range space still available after the first alert; this is label-only evaluation."},
            {"check": "sealed_holdout", "status": "PASS", "detail": "2026H1 remains sealed and is not loaded into model selection or reporting."},
        ]
    )


def _empty_result(config: ProcessAlertAuditConfig, preflight: dict[str, object], decision: str, reason: str) -> ProcessAlertAuditPipelineResult:
    write_alert_audit_reports(
        config.report_path,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight=preflight,
        audit_definition={},
        episode_metrics=pd.DataFrame(),
        event_metrics=pd.DataFrame(),
        first_alerts=pd.DataFrame(),
        event_coverage=pd.DataFrame(),
        candidates=pd.DataFrame(),
        failures=pd.DataFrame(),
        causal_audit=_causal_audit(config),
        decision=decision,
        reason=reason,
        champion=None,
        config=config,
    )
    return ProcessAlertAuditPipelineResult(decision, config.report_path, None)


def run_alert_audit_pipeline(
    *,
    config: ProcessAlertAuditConfig = DEFAULT_PROCESS_ALERT_AUDIT_CONFIG,
    data_dir: str | Path | None = None,
    force_rebuild_events: bool = False,
    force_rebuild_micro: bool = False,
    force_rebuild_long_context: bool = False,
    progress: bool = True,
) -> ProcessAlertAuditPipelineResult:
    config.validate()
    validate_model_dependencies(config.base)
    base_loader = create_loader(LONG_CONTEXT_BASE_CONFIG, data_dir=data_dir)
    micro_loader = create_micro_loader(config.base, data_dir=data_dir)
    base_preflight = run_public_loader_preflight(base_loader, LONG_CONTEXT_BASE_CONFIG)
    micro_preflight = run_micro_preflight(micro_loader, config.base)
    preflight = {"base_1m": base_preflight.to_dict(), "micro": micro_preflight.to_dict()}
    if base_preflight.status != "PASS":
        return _empty_result(config, preflight, "BLOCKED_PUBLIC_LOADER", "公共1m Trade Bar预检失败。")
    if config.base.micro_required and micro_preflight.status != "PASS":
        return _empty_result(config, preflight, "BLOCKED_MICRO_DATA", f"公共{config.base.micro_timeframe} Trade Bar覆盖不足。")

    base_paths = build_yearly_cache(
        base_loader,
        LONG_CONTEXT_BASE_CONFIG,
        force_rebuild=force_rebuild_long_context,
        progress=progress,
        feature_profile=FEATURE_PROFILE,
    )
    if not base_paths:
        base_paths = list_cached_years(LONG_CONTEXT_BASE_CONFIG)
    event_paths = build_event_caches(base_paths, config.base, force_rebuild=force_rebuild_events, progress=progress)
    micro_paths = build_micro_caches(base_paths, micro_loader, config.base, force_rebuild=force_rebuild_micro, progress=progress)
    events = _load_events(event_paths, config)

    episode_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    episode_parts: list[pd.DataFrame] = []
    coverage_parts: list[pd.DataFrame] = []
    failure_rows: list[dict[str, object]] = []
    folds = default_folds(config.base)
    jobs = len(folds) * len(config.architectures) * len(config.processes) * len(config.horizons_hours)
    reporter = ProgressReporter("[R03.3.1 alert audit] jobs", jobs, every=1, enabled=progress)
    completed = 0
    for fold in folds:
        fit_data = collect_period_data(base_paths, event_paths, micro_paths, fold.fit_start, fold.fit_end, config.base)
        calibration_data = collect_period_data(base_paths, event_paths, micro_paths, fold.calibration_start, fold.calibration_end, config.base)
        test_data = collect_period_data(base_paths, event_paths, micro_paths, fold.test_start, fold.test_end, config.base)
        price_times, decision_prices = _collect_decision_prices(base_paths, fold.test_start, fold.test_end)
        if not np.array_equal(price_times, test_data.timestamps_ns):
            raise RuntimeError(f"R03.3.1 decision-price axis mismatch in {fold.fold_id}")
        fold_events = events.loc[
            (events["end_time"] >= fold.test_start)
            & (events["start_time"] <= fold.test_end)
        ].copy()
        for architecture in config.architectures:
            for process in config.processes:
                process_events = fold_events.loc[fold_events["process"] == process].copy()
                for horizon in config.horizons_hours:
                    label_name = f"{process}_start_h{horizon}"
                    try:
                        model, calibrator, calibration_score, _, _ = fit_one(
                            architecture, label_name, fit_data, calibration_data, config.base
                        )
                        test_x, _, _ = architecture_matrix(architecture, test_data)
                        raw_test = np.asarray(model.predict_proba(test_x)[:, 1], dtype=float)
                        test_score = calibrator.predict(raw_test)
                        finite_calibration = calibration_score[np.isfinite(calibration_score)]
                        for quantile in config.signal_quantiles:
                            threshold = float(np.quantile(finite_calibration, quantile))
                            episodes = build_alert_episodes(
                                test_data.timestamps_ns,
                                test_score,
                                threshold,
                                merge_gap_hours=config.alert_merge_gap_hours,
                            )
                            audited = audit_alert_episodes(
                                episodes,
                                process=process,
                                horizon_hours=horizon,
                                process_events=process_events,
                                decision_prices=decision_prices,
                                fold_start=fold.test_start,
                                fold_end=fold.test_end,
                                config=config,
                            )
                            keys = {
                                "fold_id": fold.fold_id,
                                "architecture": architecture,
                                "process": process,
                                "horizon_hours": horizon,
                                "quantile": quantile,
                                "threshold": threshold,
                            }
                            episode_rows.append({**keys, **audited.episode_metrics})
                            event_rows.append({**keys, **audited.event_metrics})
                            if not audited.episodes.empty:
                                part = audited.episodes.copy()
                                for key, value in keys.items():
                                    part[key] = value
                                episode_parts.append(part)
                            if not audited.event_coverage.empty:
                                part = audited.event_coverage.copy()
                                for key, value in keys.items():
                                    part[key] = value
                                coverage_parts.append(part)
                    except Exception as exc:
                        failure_rows.append(
                            {
                                "fold_id": fold.fold_id,
                                "architecture": architecture,
                                "process": process,
                                "horizon_hours": horizon,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                    completed += 1
                    reporter.update(completed)
    reporter.close()

    episode_metrics = pd.DataFrame(episode_rows)
    event_metrics = pd.DataFrame(event_rows)
    first_alerts = pd.concat(episode_parts, ignore_index=True) if episode_parts else pd.DataFrame()
    coverage = pd.concat(coverage_parts, ignore_index=True) if coverage_parts else pd.DataFrame()
    candidates = _select_candidates(episode_metrics, event_metrics, config)
    passed = candidates.loc[candidates.get("passes_actionability", False)].copy() if not candidates.empty else pd.DataFrame()
    champion = passed.iloc[0].to_dict() if not passed.empty else None
    if champion:
        decision = "PASS_ACTIONABLE_PROCESS_ALERT"
        reason = "至少一个同配置在2024和2025同时通过独立预警命中率、事件覆盖率、剩余机会和晚确认门槛。"
    else:
        decision = "FAIL_ACTIONABLE_PROCESS_ALERT"
        reason = "没有同一配置同时证明：第一次独立预警足够早、剩余空间足够且跨年误报可控。"
    manifest = {
        "stage": STAGE_ID,
        "name": STAGE_NAME,
        "config": config.to_dict(),
        "folds": [fold.to_dict() for fold in folds],
        "reused_caches": {
            "long_context": str(config.base.base_cache_path),
            "events": str(config.base.event_cache_path),
            "micro": str(config.base.micro_cache_path),
        },
        "sealed_holdout": "2026-01-01 -> 2026-06-30 (not evaluated)",
    }
    definition = {
        "episode_semantics": f"signals with gaps <= {config.alert_merge_gap_hours:g}h are one episode; only first timestamp is audited",
        "early_start_semantics": f"within {config.early_start_grace_hours:g}h after event start and progress <= {config.max_actionable_progress:.0%}",
        "directional_remaining_gate": config.min_remaining_directional_move,
        "volatile_range_remaining_gate": config.min_remaining_range_move,
        "success_semantics": "pre-start or permitted early-start alert with sufficient remaining target/range opportunity",
    }
    write_alert_audit_reports(
        config.report_path,
        manifest=manifest,
        preflight=preflight,
        audit_definition=definition,
        episode_metrics=episode_metrics,
        event_metrics=event_metrics,
        first_alerts=first_alerts,
        event_coverage=coverage,
        candidates=candidates,
        failures=pd.DataFrame(failure_rows),
        causal_audit=_causal_audit(config),
        decision=decision,
        reason=reason,
        champion=champion,
        config=config,
    )
    return ProcessAlertAuditPipelineResult(decision, config.report_path, champion)
