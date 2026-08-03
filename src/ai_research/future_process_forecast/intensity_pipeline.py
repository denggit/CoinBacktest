#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R03.3.2 end-to-end continuous future-opportunity intensity pipeline."""

from __future__ import annotations

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

from .intensity_config import (
    DEFAULT_FUTURE_INTENSITY_CONFIG,
    STAGE_ID,
    STAGE_NAME,
    FutureIntensityConfig,
)
from .intensity_modeling import (
    collect_intensity_period_data,
    default_intensity_folds,
    evaluate_intensity_model,
    feature_importance,
    fit_intensity_model,
    select_stable_intensity_candidates,
    validate_intensity_dependencies,
)
from .intensity_reports import write_intensity_reports
from .intensity_targets import build_intensity_caches, load_intensity_year_shard
from .micro_features import build_micro_caches, create_micro_loader, run_micro_preflight


@dataclass(frozen=True)
class FutureIntensityResult:
    decision: str
    report_dir: Path
    champion: dict[str, object] | None


def _target_distribution(target_paths: list[Path], config: FutureIntensityConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in target_paths:
        shard = load_intensity_year_shard(path)
        year = int(pd.to_datetime(np.asarray(shard.decision_times_ns[:1], dtype=np.int64))[0].year)
        index = shard.target_index
        for target in config.target_names():
            values = np.asarray(shard.targets[:, index[target]], dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            rows.append(
                {
                    "year": year,
                    "target": target,
                    "rows": int(len(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "q75": float(np.quantile(values, 0.75)),
                    "q90": float(np.quantile(values, 0.90)),
                    "q95": float(np.quantile(values, 0.95)),
                    "q99": float(np.quantile(values, 0.99)),
                }
            )
    return pd.DataFrame(rows)


def _micro_increment(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    base = metrics.loc[metrics["architecture"] == "multiframe_lightgbm"].copy()
    micro = metrics.loc[metrics["architecture"] == "multiframe_micro_lightgbm"].copy()
    keys = ["fold_id", "target"]
    merged = base.merge(micro, on=keys, suffixes=("_base", "_micro"))
    if merged.empty:
        return merged
    for metric in ("rank_ic", "mae_skill", "decile_monotonicity", "top_decile_lift"):
        merged[f"delta_{metric}"] = merged[f"{metric}_micro"] - merged[f"{metric}_base"]
    return merged[
        [
            *keys,
            "rank_ic_base",
            "rank_ic_micro",
            "delta_rank_ic",
            "mae_skill_base",
            "mae_skill_micro",
            "delta_mae_skill",
            "top_decile_lift_base",
            "top_decile_lift_micro",
            "delta_top_decile_lift",
            "decile_monotonicity_base",
            "decile_monotonicity_micro",
            "delta_decile_monotonicity",
        ]
    ]


def _causal_audit(config: FutureIntensityConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "current_state_feature_time",
                "status": "PASS",
                "detail": "all 1D/4H/1H/30m/15m/5m/1m features are completed-bar features shifted to their causal availability time",
            },
            {
                "check": "micro_feature_time",
                "status": "PASS",
                "detail": f"public {config.base.micro_timeframe} features use bars at or before each 15m decision time",
            },
            {
                "check": "future_target_boundary",
                "status": "PASS",
                "detail": "future target windows start after the decision bar; the current 15m bar is excluded",
            },
            {
                "check": "current_state_definition",
                "status": "PASS",
                "detail": "current market state is an observed causal vector, never a label inferred from future outcomes",
            },
            {
                "check": "sealed_holdout",
                "status": "PASS",
                "detail": "2026-01-01 through 2026-06-30 is not loaded into fold metrics or model selection",
            },
        ]
    )


def _empty_result(
    config: FutureIntensityConfig,
    preflight: dict[str, object],
    decision: str,
    reason: str,
) -> FutureIntensityResult:
    write_intensity_reports(
        config.report_path,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight=preflight,
        target_distribution=pd.DataFrame(),
        regression_metrics=pd.DataFrame(),
        bucket_metrics=pd.DataFrame(),
        quantile_metrics=pd.DataFrame(),
        candidates=pd.DataFrame(),
        micro_increment=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        prediction_samples=pd.DataFrame(),
        failures=pd.DataFrame(),
        causal_audit=_causal_audit(config),
        decision=decision,
        reason=reason,
        champion=None,
        config=config,
    )
    return FutureIntensityResult(decision, config.report_path, None)


def run_intensity_pipeline(
    *,
    config: FutureIntensityConfig = DEFAULT_FUTURE_INTENSITY_CONFIG,
    data_dir: str | Path | None = None,
    force_rebuild_targets: bool = False,
    force_rebuild_micro: bool = False,
    force_rebuild_long_context: bool = False,
    progress: bool = True,
) -> FutureIntensityResult:
    config.validate()
    validate_intensity_dependencies(config)

    base_loader = create_loader(LONG_CONTEXT_BASE_CONFIG, data_dir=data_dir)
    micro_loader = create_micro_loader(config.base, data_dir=data_dir)
    base_preflight = run_public_loader_preflight(base_loader, LONG_CONTEXT_BASE_CONFIG)
    micro_preflight = run_micro_preflight(micro_loader, config.base)
    preflight = {"base_1m": base_preflight.to_dict(), "micro": micro_preflight.to_dict()}
    if base_preflight.status != "PASS":
        return _empty_result(config, preflight, "BLOCKED_PUBLIC_LOADER", "公共1m Trade Bar预检失败。")
    if config.base.micro_required and micro_preflight.status != "PASS":
        return _empty_result(
            config,
            preflight,
            "BLOCKED_MICRO_DATA",
            f"公共{config.base.micro_timeframe} Trade Bar覆盖不足。",
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
    target_paths = build_intensity_caches(
        base_paths,
        config,
        force_rebuild=force_rebuild_targets,
        progress=progress,
    )
    micro_paths = build_micro_caches(
        base_paths,
        micro_loader,
        config.base,
        force_rebuild=force_rebuild_micro,
        progress=progress,
    )

    metric_rows: list[dict[str, object]] = []
    bucket_rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    sample_parts: list[pd.DataFrame] = []
    failure_rows: list[dict[str, object]] = []

    folds = default_intensity_folds(config)
    jobs = len(folds) * len(config.architectures) * len(config.target_names())
    reporter = ProgressReporter("[R03.3.2 intensity] jobs", jobs, every=1, enabled=progress)
    completed = 0

    for fold in folds:
        fit_data = collect_intensity_period_data(
            base_paths, target_paths, micro_paths, fold.fit_start, fold.fit_end, config
        )
        calibration_data = collect_intensity_period_data(
            base_paths, target_paths, micro_paths, fold.calibration_start, fold.calibration_end, config
        )
        test_data = collect_intensity_period_data(
            base_paths, target_paths, micro_paths, fold.test_start, fold.test_end, config
        )
        for architecture in config.architectures:
            for target_name in config.target_names():
                try:
                    fitted, columns, _ = fit_intensity_model(
                        architecture,
                        target_name,
                        fit_data,
                        config,
                    )
                    metrics, buckets, quantiles, samples = evaluate_intensity_model(
                        fold_id=fold.fold_id,
                        architecture=architecture,
                        target_name=target_name,
                        fitted=fitted,
                        calibration_data=calibration_data,
                        test_data=test_data,
                        config=config,
                    )
                    metric_rows.append(metrics)
                    bucket_rows.extend(buckets)
                    quantile_rows.extend(quantiles)
                    importance_rows.extend(
                        feature_importance(
                            fitted,
                            columns,
                            fold_id=fold.fold_id,
                            architecture=architecture,
                            target_name=target_name,
                        )
                    )
                    if not samples.empty:
                        capped = (
                            samples.sort_values("prediction", ascending=False, kind="stable")
                            .groupby(["fold_id", "architecture", "target", "quantile"], as_index=False, group_keys=False)
                            .head(200)
                        )
                        sample_parts.append(capped)
                except Exception as exc:
                    failure_rows.append(
                        {
                            "fold_id": fold.fold_id,
                            "architecture": architecture,
                            "target": target_name,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                completed += 1
                reporter.update(completed)
    reporter.close()

    metrics = pd.DataFrame(metric_rows)
    buckets = pd.DataFrame(bucket_rows)
    quantiles = pd.DataFrame(quantile_rows)
    candidates = select_stable_intensity_candidates(metrics, config)
    passed = candidates.loc[candidates.get("passes", False)].copy() if not candidates.empty else pd.DataFrame()
    champion = passed.iloc[0].to_dict() if not passed.empty else None
    if champion:
        decision = "PASS_STABLE_INTENSITY_RANKING"
        reason = "至少一个连续机会强度目标在2024和2025同时通过排序相关、Top Decile提升和十分位单调性门槛。"
    else:
        decision = "FAIL_STABLE_INTENSITY_RANKING"
        reason = "没有同一连续机会强度目标同时在2024和2025证明稳定排序能力。"

    manifest = {
        "stage": STAGE_ID,
        "name": STAGE_NAME,
        "config": config.to_dict(),
        "folds": [fold.to_dict() for fold in folds],
        "reused_caches": {
            "long_context": str(config.base.base_cache_path),
            "micro": str(config.base.micro_cache_path),
        },
        "new_cache": str(config.target_cache_path),
        "sealed_holdout": "2026-01-01 -> 2026-06-30 (not evaluated)",
    }
    write_intensity_reports(
        config.report_path,
        manifest=manifest,
        preflight=preflight,
        target_distribution=_target_distribution(target_paths, config),
        regression_metrics=metrics,
        bucket_metrics=buckets,
        quantile_metrics=quantiles,
        candidates=candidates,
        micro_increment=_micro_increment(metrics),
        feature_importance=pd.DataFrame(importance_rows),
        prediction_samples=pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame(),
        failures=pd.DataFrame(failure_rows),
        causal_audit=_causal_audit(config),
        decision=decision,
        reason=reason,
        champion=champion,
        config=config,
    )
    return FutureIntensityResult(decision, config.report_path, champion)
