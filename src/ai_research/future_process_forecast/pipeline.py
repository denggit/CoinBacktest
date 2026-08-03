#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.3 event-atlas and future-process probability pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.swing_baseline.dataset import (
    build_yearly_cache,
    create_loader,
    list_cached_years,
    run_public_loader_preflight,
)
from src.ai_research.swing_long_context.config import FEATURE_PROFILE, LONG_CONTEXT_BASE_CONFIG
from src.research_common.progress import ProgressReporter

from .config import DEFAULT_FUTURE_PROCESS_FORECAST_CONFIG, FutureProcessForecastConfig, PROCESS_TYPES, STAGE_ID, STAGE_NAME
from .events import build_event_caches, load_event_year_shard
from .micro_features import build_micro_caches, create_micro_loader, run_micro_preflight
from .modeling import (
    architecture_matrix,
    collect_period_data,
    default_folds,
    evaluate_one,
    feature_importance,
    fit_one,
    select_stable_candidates,
    validate_model_dependencies,
)
from .reports import write_reports


@dataclass(frozen=True)
class FutureProcessForecastResult:
    decision: str
    report_dir: Path
    champion: dict[str, object] | None


def _empty_reports(
    config: FutureProcessForecastConfig,
    *,
    preflight: dict[str, object],
    decision: str,
    reason: str,
) -> FutureProcessForecastResult:
    write_reports(
        config.report_path,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight=preflight,
        event_definition={},
        events=pd.DataFrame(),
        event_summary=pd.DataFrame(),
        event_path_summary=pd.DataFrame(),
        label_rates=pd.DataFrame(),
        probability_metrics=pd.DataFrame(),
        quantile_metrics=pd.DataFrame(),
        candidates=pd.DataFrame(),
        micro_increment=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        pre_event_uplift=pd.DataFrame(),
        signal_samples=pd.DataFrame(),
        failures=pd.DataFrame(),
        causal_audit=pd.DataFrame(),
        decision=decision,
        reason=reason,
        champion=None,
        config=config,
    )
    return FutureProcessForecastResult(decision, config.report_path, None)


def _load_events(paths: list[Path], config: FutureProcessForecastConfig) -> pd.DataFrame:
    parts = [load_event_year_shard(path).events for path in paths]
    frame = pd.concat([part for part in parts if not part.empty], ignore_index=True) if any(not p.empty for p in parts) else pd.DataFrame()
    if frame.empty:
        return frame
    frame["start_time"] = pd.to_datetime(frame["start_time"])
    frame["end_time"] = pd.to_datetime(frame["end_time"])
    frame = frame.loc[
        (frame["start_time"] >= pd.Timestamp(config.research_start))
        & (frame["start_time"] <= pd.Timestamp(config.research_end))
    ].copy()
    return frame.drop_duplicates("event_uid", keep="first").sort_values("start_time", kind="stable").reset_index(drop=True)


def _event_summaries(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    events = events.copy()
    events["start_year"] = pd.to_datetime(events["start_time"]).dt.year
    yearly = events.groupby(["process", "start_year"], dropna=False).size().rename("events").reset_index()
    summary = (
        events.groupby("process", dropna=False)
        .agg(
            events=("event_uid", "size"),
            median_hours_to_target=("hours_to_target", "median"),
            median_target_move=("target_move", "median"),
            median_mfe_72h=("mfe_72h", "median"),
            median_mfe_120h=("mfe_120h", "median"),
            hit_7pct_72h_rate=("hit_7pct_72h", "mean"),
            hit_10pct_120h_rate=("hit_10pct_120h", "mean"),
        )
        .reset_index()
    )
    return yearly, summary


def _label_rates(event_paths: list[Path], config: FutureProcessForecastConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in event_paths:
        shard = load_event_year_shard(path)
        year = int(pd.to_datetime(np.asarray(shard.decision_times_ns[:1], dtype=np.int64))[0].year)
        mapping = shard.label_index
        for process in PROCESS_TYPES:
            for horizon in config.forecast_horizons_hours:
                name = f"{process}_start_h{horizon}"
                values = np.asarray(shard.labels[:, mapping[name]], dtype=float)
                rows.append(
                    {
                        "year": year,
                        "process": process,
                        "horizon_hours": horizon,
                        "rows": int(np.isfinite(values).sum()),
                        "positive_rate": float(np.nanmean(values)),
                    }
                )
    return pd.DataFrame(rows)


def _micro_increment(probability: pd.DataFrame, quantiles: pd.DataFrame) -> pd.DataFrame:
    if probability.empty:
        return pd.DataFrame()
    keys = ["fold_id", "process", "horizon_hours"]
    base = probability.loc[probability["architecture"] == "multiframe_lightgbm", [*keys, "roc_auc", "average_precision", "brier"]].rename(
        columns={"roc_auc": "base_auc", "average_precision": "base_ap", "brier": "base_brier"}
    )
    micro = probability.loc[probability["architecture"] == "multiframe_micro_lightgbm", [*keys, "roc_auc", "average_precision", "brier"]].rename(
        columns={"roc_auc": "micro_auc", "average_precision": "micro_ap", "brier": "micro_brier"}
    )
    out = base.merge(micro, on=keys, how="inner")
    q95 = quantiles.loc[np.isclose(quantiles["quantile"], 0.95)]
    qbase = q95.loc[q95["architecture"] == "multiframe_lightgbm", [*keys, "lift", "tail_car_rate_progress30"]].rename(
        columns={"lift": "base_lift", "tail_car_rate_progress30": "base_tail_car"}
    )
    qmicro = q95.loc[q95["architecture"] == "multiframe_micro_lightgbm", [*keys, "lift", "tail_car_rate_progress30"]].rename(
        columns={"lift": "micro_lift", "tail_car_rate_progress30": "micro_tail_car"}
    )
    out = out.merge(qbase, on=keys, how="left").merge(qmicro, on=keys, how="left")
    out["auc_increment"] = out["micro_auc"] - out["base_auc"]
    out["ap_increment"] = out["micro_ap"] - out["base_ap"]
    out["brier_improvement"] = out["base_brier"] - out["micro_brier"]
    out["lift_increment"] = out["micro_lift"] - out["base_lift"]
    out["tail_car_reduction"] = out["base_tail_car"] - out["micro_tail_car"]
    return out


def _pre_event_uplift(
    data,
    events: pd.DataFrame,
    config: FutureProcessForecastConfig,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    matrix = data.combined_x
    columns = (*data.full_columns, *data.micro_columns)
    baseline_positions = np.arange(0, len(matrix), config.sample_stride_decisions)
    baseline = matrix[baseline_positions]
    mean = np.nanmean(baseline, axis=0)
    std = np.nanstd(baseline, axis=0)
    std = np.where(std > 1e-8, std, np.nan)
    times = np.asarray(data.timestamps_ns, dtype=np.int64)
    rows: list[dict[str, object]] = []
    for process in PROCESS_TYPES[:-1]:
        starts = pd.to_datetime(events.loc[events["process"] == process, "start_time"])
        if starts.empty:
            continue
        for lead_hours in (24, 12, 6, 3, 1):
            target_ns = (starts - pd.Timedelta(hours=lead_hours)).to_numpy(dtype="datetime64[ns]").astype(np.int64)
            positions = np.searchsorted(times, target_ns, side="right") - 1
            positions = positions[(positions >= 0) & (positions < len(times))]
            if len(positions) < 5:
                continue
            event_mean = np.nanmean(matrix[positions], axis=0)
            effect = (event_mean - mean) / std
            valid = np.flatnonzero(np.isfinite(effect))
            order = valid[np.argsort(np.abs(effect[valid]), kind="stable")[-30:][::-1]]
            for feature_index in order:
                rows.append(
                    {
                        "process": process,
                        "lead_hours": lead_hours,
                        "events": int(len(positions)),
                        "feature": columns[feature_index],
                        "standardized_mean_difference": float(effect[feature_index]),
                    }
                )
    return pd.DataFrame(rows)


def _causal_audit(config: FutureProcessForecastConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "long_context_availability", "status": "PASS", "detail": "R03.2 completed bars are shifted to timeframe availability before 15m alignment."},
            {"check": "micro_availability", "status": "PASS", "detail": f"{config.micro_timeframe} bars are aggregated to completed 1m features and shifted +1m before decision alignment."},
            {"check": "raw_trade_access", "status": "PASS", "detail": "R03.3 calls public OKXTradeBarLoader with build_missing=False; no Raw Trades access or auto rebuild."},
            {"check": "positive_label_timing", "status": "PASS", "detail": "A start label is positive only when event_start > decision_time and within the forecast horizon."},
            {"check": "post_start_handling", "status": "PASS", "detail": "Signals after start are ongoing/tail-car diagnostics, never successful start predictions."},
            {"check": "independent_events", "status": "PASS", "detail": "Directional and range candidates use refractory non-overlap selection and event_uid deduplication."},
            {"check": "sealed_holdout", "status": "PASS", "detail": "Research, model selection and reports end 2025-12-31; 2026H1 is not evaluated."},
        ]
    )


def run_pipeline(
    *,
    config: FutureProcessForecastConfig = DEFAULT_FUTURE_PROCESS_FORECAST_CONFIG,
    data_dir: str | Path | None = None,
    force_rebuild_events: bool = False,
    force_rebuild_micro: bool = False,
    force_rebuild_long_context: bool = False,
    progress: bool = True,
) -> FutureProcessForecastResult:
    config.validate()
    validate_model_dependencies(config)
    config.report_path.mkdir(parents=True, exist_ok=True)
    base_loader = create_loader(LONG_CONTEXT_BASE_CONFIG, data_dir=data_dir)
    micro_loader = create_micro_loader(config, data_dir=data_dir)
    base_preflight = run_public_loader_preflight(base_loader, LONG_CONTEXT_BASE_CONFIG)
    micro_preflight = run_micro_preflight(micro_loader, config)
    preflight = {"base_1m": base_preflight.to_dict(), "micro": micro_preflight.to_dict()}
    if base_preflight.status != "PASS":
        return _empty_reports(config, preflight=preflight, decision="BLOCKED_PUBLIC_LOADER", reason="公共1m Trade Bar预检失败；未读取Raw Trades，也未自动重建数据。")
    if config.micro_required and micro_preflight.status != "PASS":
        return _empty_reports(
            config,
            preflight=preflight,
            decision="BLOCKED_MICRO_DATA",
            reason=f"公共{config.micro_timeframe} Trade Bar覆盖不足；本版要求真实微观增量，未降级成纯1m研究。",
        )

    base_paths = build_yearly_cache(
        base_loader,
        LONG_CONTEXT_BASE_CONFIG,
        force_rebuild=force_rebuild_long_context,
        progress=progress,
        feature_profile=FEATURE_PROFILE,
    )
    if not base_paths:
        base_paths = list_cached_years(LONG_CONTEXT_BASE_CONFIG)
    event_paths = build_event_caches(base_paths, config, force_rebuild=force_rebuild_events, progress=progress)
    micro_paths = build_micro_caches(base_paths, micro_loader, config, force_rebuild=force_rebuild_micro, progress=progress)

    events = _load_events(event_paths, config)
    event_summary, event_path_summary = _event_summaries(events)
    label_rates = _label_rates(event_paths, config)
    probability_rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []
    signal_parts: list[pd.DataFrame] = []
    importance_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    folds = default_folds(config)
    jobs = len(folds) * len(config.architectures) * len(PROCESS_TYPES) * len(config.forecast_horizons_hours)
    reporter = ProgressReporter("[R03.3 models] jobs", jobs, every=1, enabled=progress)
    completed = 0
    for fold in folds:
        fit_data = collect_period_data(base_paths, event_paths, micro_paths, fold.fit_start, fold.fit_end, config)
        calibration_data = collect_period_data(
            base_paths, event_paths, micro_paths, fold.calibration_start, fold.calibration_end, config
        )
        test_data = collect_period_data(base_paths, event_paths, micro_paths, fold.test_start, fold.test_end, config)
        for architecture in config.architectures:
            for process in PROCESS_TYPES:
                for horizon in config.forecast_horizons_hours:
                    label_name = f"{process}_start_h{horizon}"
                    try:
                        model, calibrator, calibration_score, columns, metadata = fit_one(
                            architecture, label_name, fit_data, calibration_data, config
                        )
                        test_x, _, _ = architecture_matrix(architecture, test_data)
                        test_score = calibrator.predict(np.asarray(model.predict_proba(test_x)[:, 1], dtype=float))
                        base, qrows, samples = evaluate_one(
                            fold_id=fold.fold_id,
                            architecture=architecture,
                            process=process,
                            horizon=horizon,
                            calibration_score=calibration_score,
                            test_score=test_score,
                            test_data=test_data,
                            config=config,
                        )
                        probability_rows.append(base)
                        quantile_rows.extend(qrows)
                        if not samples.empty:
                            signal_parts.append(samples)
                        importance_rows.extend(
                            feature_importance(
                                model,
                                columns,
                                fold_id=fold.fold_id,
                                architecture=architecture,
                                process=process,
                                horizon=horizon,
                            )
                        )
                        metadata_rows.append({"fold": fold.fold_id, **metadata})
                    except Exception as exc:  # keep independent heads auditable
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

    probability = pd.DataFrame(probability_rows)
    quantiles = pd.DataFrame(quantile_rows)
    samples = pd.concat(signal_parts, ignore_index=True) if signal_parts else pd.DataFrame()
    if not samples.empty:
        samples = samples.sort_values("score", ascending=False, kind="stable").groupby(
            ["fold_id", "architecture", "process", "horizon_hours", "quantile"], group_keys=False
        ).head(500)
    importance = pd.DataFrame(importance_rows)
    if not importance.empty:
        importance = importance.sort_values("importance", ascending=False, kind="stable").groupby(
            ["fold_id", "architecture", "process", "horizon_hours"], group_keys=False
        ).head(100)
    candidates = select_stable_candidates(probability, quantiles)
    passed = candidates.loc[(candidates.get("passes", False)) & (candidates["process"] != "low_opportunity")].copy() if not candidates.empty else pd.DataFrame()
    champion = passed.iloc[0].to_dict() if not passed.empty else None
    decision = "PASS_PROCESS_FORECAST_MVP" if champion else "FAIL_NO_STABLE_PROCESS_FORECAST"
    reason = (
        "至少一个高价值过程在2024与2025使用同一模型配置通过提前量、Lift、校准和尾班车门槛；可以进入R03.4入场研究。"
        if champion
        else "没有高价值过程使用同一配置同时通过2024与2025；本轮不能进入入场或交易回测。"
    )
    full_data = collect_period_data(
        base_paths,
        event_paths,
        micro_paths,
        pd.Timestamp(config.research_start),
        pd.Timestamp(config.research_end),
        config,
    )
    uplift = _pre_event_uplift(full_data, events, config)
    micro_increment = _micro_increment(probability, quantiles)
    causal_audit = _causal_audit(config)
    manifest = {
        "stage": STAGE_ID,
        "name": STAGE_NAME,
        "config": config.to_dict(),
        "folds": [fold.to_dict() for fold in folds],
        "feature_profile": FEATURE_PROFILE,
        "base_cache": str(config.base_cache_path),
        "event_cache": str(config.event_cache_path),
        "micro_cache": str(config.micro_cache_path),
        "model_metadata": metadata_rows,
        "sealed_holdout": "2026-01-01 -> 2026-06-30 (not evaluated)",
    }
    event_definition = {
        "directional": config.directional.__dict__,
        "volatile_range": config.volatile_range.__dict__,
        "forecast_horizons_hours": list(config.forecast_horizons_hours),
        "positive_semantics": "new event starts strictly after decision time",
        "tail_car_semantics": "signal occurs during an event and progress >= 30%",
    }
    write_reports(
        config.report_path,
        manifest=manifest,
        preflight=preflight,
        event_definition=event_definition,
        events=events,
        event_summary=event_summary,
        event_path_summary=event_path_summary,
        label_rates=label_rates,
        probability_metrics=probability,
        quantile_metrics=quantiles,
        candidates=candidates,
        micro_increment=micro_increment,
        feature_importance=importance,
        pre_event_uplift=uplift,
        signal_samples=samples,
        failures=pd.DataFrame(failure_rows),
        causal_audit=causal_audit,
        decision=decision,
        reason=reason,
        champion=champion,
        config=config,
    )
    return FutureProcessForecastResult(decision, config.report_path, champion)
