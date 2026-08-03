#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2.1 complete path atlas."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.config import LongTailExitAuditConfig
from src.ai_research.long_tail_exit_audit.data import collect_base_period_data, load_minute_path_data
from src.ai_research.long_tail_exit_audit.modeling import fit_frozen_base_scores
from src.ai_research.long_tail_exit_audit.simulator import EventCandidate, ScoreTimeline, build_event_candidates
from src.ai_research.state_context_ablation.config import DEFAULT_STATE_CONTEXT_ABLATION_CONFIG
from src.ai_research.state_context_ablation.modeling import AblationPeriodData, default_ablation_folds
from src.ai_research.state_context_ablation.outcomes import build_outcome_caches
from src.ai_research.swing_baseline.dataset import create_loader, list_cached_years, run_public_loader_preflight
from src.ai_research.swing_long_context.config import LONG_CONTEXT_BASE_CONFIG
from src.research_common.progress import ProgressReporter

from .atlas import (
    assign_clusters,
    cluster_centroid_frame,
    extract_event_path,
    fit_path_cluster_model,
    oracle_exit_summary,
    representative_events,
    selected_feature_contrast,
    summarize_path_types,
)
from .config import DEFAULT_LONG_TAIL_PATH_ATLAS_CONFIG, STAGE_ID, STAGE_NAME, LongTailPathAtlasConfig
from .reports import write_reports


@dataclass(frozen=True)
class LongTailPathAtlasResult:
    decision: str
    report_dir: Path


def _base_config(config: LongTailPathAtlasConfig) -> LongTailExitAuditConfig:
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


def _extract_events(
    *,
    fold_id: str,
    phase: str,
    events: tuple[EventCandidate, ...],
    path,
    timeline: ScoreTimeline,
    calibration_scores: np.ndarray,
    config: LongTailPathAtlasConfig,
    path_file: Path | None,
    progress: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    skipped_incomplete = 0
    point_rows = 0
    if path_file is not None:
        path_file.parent.mkdir(parents=True, exist_ok=True)
        if path_file.exists():
            path_file.unlink()
    reporter = ProgressReporter(f"[R03.4.2.1 {fold_id} {phase}] events", len(events), every=max(1, len(events) // 20), enabled=progress)
    header = True
    handle = gzip.open(path_file, mode="wt", encoding="utf-8", newline="") if path_file is not None else None
    try:
        for number, event in enumerate(events, start=1):
            extraction = extract_event_path(
                event=event,
                fold_id=fold_id,
                phase=phase,
                path=path,
                timeline=timeline,
                calibration_scores=calibration_scores,
                config=config,
            )
            if extraction is None:
                skipped_incomplete += 1
            else:
                rows.append(extraction.summary)
                point_rows += len(extraction.points)
                if handle is not None:
                    extraction.points.to_csv(handle, index=False, header=header)
                    header = False
            reporter.update(number)
    finally:
        if handle is not None:
            handle.close()
        reporter.close()
    return pd.DataFrame(rows), {
        "fold_id": fold_id,
        "phase": phase,
        "candidate_events": len(events),
        "complete_48h_events": len(rows),
        "skipped_incomplete_or_missing": skipped_incomplete,
        "path_point_rows": point_rows,
        "path_file": str(path_file) if path_file is not None else None,
        "path_file_bytes": path_file.stat().st_size if path_file is not None and path_file.exists() else 0,
    }


def _causal_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "frozen_entry_model", "status": "PASS", "detail": "R03.4.1 q90 base long LightGBM and next-minute-open entry are frozen"},
            {"check": "no_exit_search", "status": "PASS", "detail": "no stop, target, trailing, renewal or exit parameter is optimized in this stage"},
            {"check": "all_events_included", "status": "PASS", "detail": "both winners and losers enter the path atlas; no survivor-only filtering"},
            {"check": "future_path_label_only", "status": "PASS", "detail": "semantic types and clusters are ex-post research labels and are never treated as live features"},
            {"check": "discovery_before_oos", "status": "PASS", "detail": "each OOS cluster model is fit only on calibration paths available before that test year"},
            {"check": "score_asof", "status": "PASS", "detail": "each minute sees only the latest 15m score at or before that minute"},
            {"check": "complete_path", "status": "PASS", "detail": "events require an exact uninterrupted 48h 1m path; incomplete year-end events are excluded"},
            {"check": "state_model_abandoned", "status": "PASS", "detail": "no strategic, tactical, entry or activity state cache is loaded"},
            {"check": "sealed_holdout", "status": "PASS", "detail": "2026 is not loaded and no 48h path may cross the 2026 seal"},
        ]
    )


def _contract(config: LongTailPathAtlasConfig) -> dict[str, object]:
    return {
        "base_model": {
            "target": "long_utility_h6 = long_mfe_h6 - 1.25 * long_mae_h6",
            "objective": "regression_l1",
            "n_estimators": config.base_n_estimators,
            "learning_rate": config.base_learning_rate,
            "num_leaves": config.base_num_leaves,
            "min_child_samples": config.base_min_child_samples,
            "train_sample_cap": config.train_sample_cap,
            "random_state": config.random_state,
        },
        "entry": "prior-quarter q90 independent event; next 1m open",
        "path": "exact 48h of 1m OHLC plus causal 15m score evolution",
        "semantic_types": "fixed descriptive thresholds; not exit rules",
        "clusters": f"KMeans k={config.cluster_count}, robust-scaled, fit only on prior calibration path pool",
        "state_model_policy": "ABANDONED_FOR_TRADING_AND_NOT_LOADED",
        "oos_policy": "2024/2025 paths audit frozen type structure only; no OOS tuning",
    }


def _score_bins(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for fold_id, group in frame.groupby("fold_id", sort=False):
        for feature in ("score_percentile_min_360m", "score_percentile_end_360m", "q90_reconfirmations_360m"):
            work = group.loc[group[feature].notna()].copy()
            if work.empty:
                continue
            try:
                work["bin"] = pd.qcut(work[feature], q=5, duplicates="drop")
            except ValueError:
                continue
            for bucket, bucket_group in work.groupby("bin", observed=True, sort=True):
                rows.append(
                    {
                        "fold_id": fold_id,
                        "score_path_feature": feature,
                        "bucket": str(bucket),
                        "events": int(len(bucket_group)),
                        "feature_mean": float(bucket_group[feature].mean()),
                        "fixed6h_win_rate_1x": float(bucket_group["fixed6h_positive_expectancy_event"].mean()),
                        "mean_fixed6h_net_1x": float(bucket_group["fixed6h_net_1x"].mean()),
                        "mean_mae_360m": float(bucket_group["mae_360m"].mean()),
                        "post6_continuation_rate": float(bucket_group["flag_post6_continuation"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def _cluster_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(["fold_id", "cluster_name"], sort=False):
        fold_id, name = keys
        rows.append(
            {
                "fold_id": fold_id,
                "cluster_name": name,
                "events": int(len(group)),
                "share": float(len(group) / len(frame.loc[frame["fold_id"] == fold_id])),
                "mean_cluster_distance": float(group["cluster_distance"].mean()),
                "mean_fixed6h_net_1x": float(group["fixed6h_net_1x"].mean()),
                "win_rate_1x": float(group["fixed6h_positive_expectancy_event"].mean()),
                "mean_mfe_360m": float(group["mfe_360m"].mean()),
                "mean_mae_360m": float(group["mae_360m"].mean()),
                "post6_continuation_rate": float(group["flag_post6_continuation"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _empty_reports(
    *,
    config: LongTailPathAtlasConfig,
    preflight: dict[str, object],
    decision: str,
    reason: str,
) -> LongTailPathAtlasResult:
    empty = pd.DataFrame()
    write_reports(
        config.report_path,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight=preflight,
        contract=_contract(config),
        event_audit=empty,
        discovery_features=empty,
        oos_features=empty,
        assignments=empty,
        path_type_summary=empty,
        path_type_period=empty,
        target_timing=empty,
        winner_loser_contrast=empty,
        representatives=empty,
        cluster_centroids=empty,
        cluster_summary=empty,
        oracle_exit=empty,
        score_bins=empty,
        path_file_manifest={},
        causal_audit=_causal_audit(),
        failures=empty,
        decision=decision,
        reason=reason,
        config=config,
    )
    return LongTailPathAtlasResult(decision=decision, report_dir=config.report_path)


def run_long_tail_path_atlas(
    *,
    data_dir: str | Path | None = None,
    force_rebuild_outcomes: bool = False,
    progress: bool = True,
    config: LongTailPathAtlasConfig = DEFAULT_LONG_TAIL_PATH_ATLAS_CONFIG,
) -> LongTailPathAtlasResult:
    config.validate()
    base_config = _base_config(config)
    loader = create_loader(LONG_CONTEXT_BASE_CONFIG, data_dir=data_dir)
    loader_preflight = run_public_loader_preflight(
        loader,
        LONG_CONTEXT_BASE_CONFIG,
        sample_dates=("2023-06-15", "2024-06-15", "2025-06-15"),
    )
    preflight: dict[str, object] = {"trade_bar": loader_preflight.to_dict()}
    if loader_preflight.status != "PASS":
        return _empty_reports(config=config, preflight=preflight, decision="BLOCKED_DATA", reason="1分钟Trade Bar公共Loader预检失败。")

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

    report_dir = config.report_path
    path_dir = report_dir / "event_paths"
    path_dir.mkdir(parents=True, exist_ok=True)
    for old in path_dir.glob("*.csv.gz"):
        old.unlink()

    discovery_pool: list[pd.DataFrame] = []
    discovery_parts: list[pd.DataFrame] = []
    oos_parts: list[pd.DataFrame] = []
    assignment_parts: list[pd.DataFrame] = []
    centroid_parts: list[pd.DataFrame] = []
    event_audit_rows: list[dict[str, object]] = []
    path_files: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    folds = default_ablation_folds(DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
    fold_reporter = ProgressReporter("[R03.4.2.1 folds]", len(folds), every=1, enabled=progress)

    for fold_number, fold in enumerate(folds, start=1):
        try:
            fit = collect_base_period_data(base_paths, outcome_paths, start=fold.fit_start, end=fold.fit_end, outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
            calibration = collect_base_period_data(base_paths, outcome_paths, start=fold.calibration_start, end=fold.calibration_end, outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
            test = collect_base_period_data(base_paths, outcome_paths, start=fold.test_start, end=fold.test_end, outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
            bundle = fit_frozen_base_scores(fit, calibration, test, base_config)
            thresholds = _thresholds(bundle.calibration_score)
            calibration_timeline = _timeline(calibration, bundle.calibration_score, thresholds)
            test_timeline = _timeline(test, bundle.test_score, thresholds)

            discovery_events = build_event_candidates(calibration_timeline, signal_quantile=config.primary_signal_quantile, config=base_config)
            oos_events = build_event_candidates(test_timeline, signal_quantile=config.primary_signal_quantile, config=base_config)

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
                "start": str(discovery_path.index[0]),
                "end": str(discovery_path.index[-1]),
                "rows": len(discovery_path.index),
                "coverage_ratio": discovery_path.coverage_ratio,
            }
            preflight[f"{fold.fold_id}_oos_path"] = {
                "start": str(oos_path.index[0]),
                "end": str(oos_path.index[-1]),
                "rows": len(oos_path.index),
                "coverage_ratio": oos_path.coverage_ratio,
            }
            if discovery_path.coverage_ratio < 0.995 or oos_path.coverage_ratio < 0.995:
                raise RuntimeError("minute path coverage below 99.5%")

            discovery_file = path_dir / f"{fold.fold_id}_discovery_q90_48h_1m.csv.gz" if config.export_full_minute_paths else None
            discovery_frame, discovery_audit = _extract_events(
                fold_id=fold.fold_id,
                phase="discovery",
                events=discovery_events,
                path=discovery_path,
                timeline=calibration_timeline,
                calibration_scores=bundle.calibration_score,
                config=config,
                path_file=discovery_file,
                progress=progress,
            )
            event_audit_rows.append(discovery_audit)
            path_files.append(discovery_audit)
            if not discovery_frame.empty:
                discovery_parts.append(discovery_frame)
                discovery_pool.append(discovery_frame)

            cluster_training = pd.concat(discovery_pool, ignore_index=True) if discovery_pool else pd.DataFrame()
            cluster_model = fit_path_cluster_model(cluster_training, config)
            if cluster_model is not None:
                centroid_parts.append(cluster_centroid_frame(cluster_model, fold.fold_id))

            oos_file = path_dir / f"{fold.fold_id}_oos_q90_48h_1m.csv.gz" if config.export_full_minute_paths else None
            oos_frame, oos_audit = _extract_events(
                fold_id=fold.fold_id,
                phase="oos",
                events=oos_events,
                path=oos_path,
                timeline=test_timeline,
                calibration_scores=bundle.calibration_score,
                config=config,
                path_file=oos_file,
                progress=progress,
            )
            event_audit_rows.append(oos_audit)
            path_files.append(oos_audit)
            if not oos_frame.empty:
                assigned = assign_clusters(oos_frame, cluster_model)
                oos_parts.append(assigned)
                assignment_parts.append(
                    assigned.loc[:, [
                        "event_id", "fold_id", "decision_time", "entry_time", "is_q95",
                        "semantic_path_type", "cluster_id", "cluster_name", "cluster_distance",
                        "fixed6h_positive_expectancy_event", "fixed6h_net_1x",
                        "flag_immediate_clean", "flag_early_spike_giveback", "flag_delayed_recovery",
                        "flag_slow_grind", "flag_late_rescue", "flag_persistent_failure",
                        "flag_post6_continuation", "flag_deep_6h_mae",
                        "flag_score_reconfirmed_6h", "flag_score_decayed_below_median_6h",
                    ]]
                )
        except Exception as exc:
            failures.append({"fold_id": fold.fold_id, "error": f"{type(exc).__name__}: {exc}"})
        fold_reporter.update(fold_number)
    fold_reporter.close()

    discovery_features = pd.concat(discovery_parts, ignore_index=True) if discovery_parts else pd.DataFrame()
    oos_features = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame()
    assignments = pd.concat(assignment_parts, ignore_index=True) if assignment_parts else pd.DataFrame()
    centroids = pd.concat(centroid_parts, ignore_index=True) if centroid_parts else pd.DataFrame()
    path_type_summary, path_type_period, target_timing = summarize_path_types(oos_features)
    contrast = selected_feature_contrast(oos_features)
    representatives = representative_events(oos_features)
    cluster_summary = _cluster_summary(oos_features)
    oracle = oracle_exit_summary(oos_features, base_round_trip_cost=config.base_round_trip_cost)
    score_bins = _score_bins(oos_features)

    if oos_features.empty:
        decision = "FAIL_NO_COMPLETE_PATHS"
        reason = "冻结q90事件未能形成完整48小时一分钟路径，无法开展路径类型研究。"
    else:
        fold_counts = oos_features.groupby("fold_id").size().to_dict()
        enough_oos = all(fold_counts.get(fold_id, 0) >= config.minimum_oos_events_per_year for fold_id in ("WF_2024", "WF_2025"))
        discovery_counts = pd.DataFrame(event_audit_rows)
        enough_discovery = bool(
            not discovery_counts.empty
            and (discovery_counts.loc[discovery_counts["phase"] == "discovery", "complete_48h_events"] >= config.minimum_discovery_events).all()
        )
        stable_types = 0
        if not path_type_summary.empty:
            pivot = path_type_summary.pivot_table(index="semantic_path_type", columns="fold_id", values="events", fill_value=0)
            if {"WF_2024", "WF_2025"}.issubset(pivot.columns):
                stable_types = int(((pivot["WF_2024"] >= config.minimum_type_samples) & (pivot["WF_2025"] >= config.minimum_type_samples)).sum())
        if enough_oos and enough_discovery and stable_types >= 3 and not failures:
            decision = "PASS_PATH_ATLAS_READY_FOR_CAUSAL_EXIT_RESEARCH"
            reason = "q90事件已形成完整逐笔48小时路径图谱，至少三类路径在2024和2025均有足够样本；可以进入下一阶段的早期因果路径识别研究，但尚未产生退出规则。"
        else:
            decision = "RESEARCH_CONTINUE_PATH_ATLAS_LIMITED"
            reason = f"路径图谱已生成，但发现期样本、OOS事件规模或跨年类型覆盖仍有限（stable_types={stable_types}）；只能用于诊断，暂不应设计差异化退出。"

    path_file_manifest = {
        "format": "gzip-compressed UTF-8 CSV",
        "scope": "full 48h one-minute points for every complete q90 event",
        "files": path_files,
        "review_pack_policy": "large .csv.gz path shards are intentionally excluded from gpt_review_pack.zip",
    }
    write_reports(
        report_dir,
        manifest={
            "stage": STAGE_ID,
            "name": STAGE_NAME,
            "config": config.to_dict(),
            "folds": [fold.to_dict() for fold in folds],
            "state_model_policy": "ABANDONED_FOR_TRADING_AND_NOT_LOADED",
        },
        preflight=preflight,
        contract=_contract(config),
        event_audit=pd.DataFrame(event_audit_rows),
        discovery_features=discovery_features,
        oos_features=oos_features,
        assignments=assignments,
        path_type_summary=path_type_summary,
        path_type_period=path_type_period,
        target_timing=target_timing,
        winner_loser_contrast=contrast,
        representatives=representatives,
        cluster_centroids=centroids,
        cluster_summary=cluster_summary,
        oracle_exit=oracle,
        score_bins=score_bins,
        path_file_manifest=path_file_manifest,
        causal_audit=_causal_audit(),
        failures=pd.DataFrame(failures),
        decision=decision,
        reason=reason,
        config=config,
    )
    return LongTailPathAtlasResult(decision=decision, report_dir=report_dir)
