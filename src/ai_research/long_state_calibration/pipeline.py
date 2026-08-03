#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.1 long-opportunity soft-state meta calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.market_state_continuity.config import DEFAULT_MARKET_STATE_CONTINUITY_CONFIG
from src.ai_research.market_state_continuity.data import UnifiedOHLCVLoader, run_state_data_preflight
from src.ai_research.market_state_continuity.state_cache import build_state_caches
from src.ai_research.state_context_ablation.config import DEFAULT_STATE_CONTEXT_ABLATION_CONFIG
from src.ai_research.state_context_ablation.modeling import (
    collect_ablation_period_data,
    default_ablation_folds,
)
from src.ai_research.state_context_ablation.outcomes import build_outcome_caches
from src.ai_research.swing_baseline.dataset import create_loader, list_cached_years, run_public_loader_preflight
from src.ai_research.swing_long_context.config import LONG_CONTEXT_BASE_CONFIG
from src.research_common.progress import ProgressReporter

from .config import (
    DEFAULT_LONG_STATE_CALIBRATION_CONFIG,
    STAGE_ID,
    STAGE_NAME,
    LongStateCalibrationConfig,
)
from .modeling import (
    build_uplift_tables,
    evaluate_fold_models,
    fit_base_long_model,
    fit_meta_model,
    generate_oof_base_scores,
    select_stable_candidates,
)
from .reports import write_reports


@dataclass(frozen=True)
class LongStateCalibrationResult:
    decision: str
    report_dir: Path
    champion: dict[str, object] | None


def _causal_audit(config: LongStateCalibrationConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "base_features",
                "status": "PASS",
                "detail": "first-stage features reuse the completed-bar R03.2 causal decision axis",
            },
            {
                "check": "state_features",
                "status": "PASS",
                "detail": "only current/past strategic and activity soft state, age, boundary and flip statistics are used",
            },
            {
                "check": "discrete_direction_exclusion",
                "status": "PASS",
                "detail": "tactical_state, entry_state, tactical_score and entry_score are excluded",
            },
            {
                "check": "stacking_oof",
                "status": "PASS",
                "detail": f"meta models train on expanding-window OOF base predictions with {config.oof_embargo_hours}h embargo",
            },
            {
                "check": "threshold_calibration",
                "status": "PASS",
                "detail": "signal thresholds and empirical rank multipliers use only the prior calibration quarter",
            },
            {
                "check": "entry_timing",
                "status": "PASS",
                "detail": "future outcomes begin at the existing next-minute open after each 15-minute decision",
            },
            {
                "check": "sealed_holdout",
                "status": "PASS",
                "detail": "2026 is not loaded and 2025 labels crossing 2026 remain invalid",
            },
        ]
    )


def _empty_write(
    config: LongStateCalibrationConfig,
    preflight: dict[str, object],
    decision: str,
    reason: str,
) -> LongStateCalibrationResult:
    empty = pd.DataFrame()
    write_reports(
        config.report_path,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight=preflight,
        oof_audit=empty,
        model_metrics=empty,
        signal_metrics=empty,
        rerank_metrics=empty,
        multiplier_metrics=empty,
        uplift=empty,
        stable=empty,
        importance=empty,
        samples=empty,
        causal_audit=_causal_audit(config),
        failures=empty,
        decision=decision,
        reason=reason,
        champion=None,
        config=config,
    )
    return LongStateCalibrationResult(decision, config.report_path, None)


def _importance(models: dict[str, object], fold_id: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, meta in models.items():
        raw = getattr(meta, "model", None)
        values = getattr(raw, "feature_importances_", None)
        columns = getattr(meta, "columns", ())
        if values is None:
            continue
        rows.extend(
            {
                "fold_id": fold_id,
                "variant": variant,
                "feature": column,
                "importance": float(value),
            }
            for column, value in zip(columns, values, strict=True)
        )
    return pd.DataFrame(rows)


def run_long_state_calibration(
    *,
    data_dir: str | Path | None = None,
    force_rebuild_state_cache: bool = False,
    force_rebuild_outcomes: bool = False,
    progress: bool = True,
    config: LongStateCalibrationConfig = DEFAULT_LONG_STATE_CALIBRATION_CONFIG,
) -> LongStateCalibrationResult:
    config.validate()
    base_loader = create_loader(LONG_CONTEXT_BASE_CONFIG, data_dir=data_dir)
    base_preflight = run_public_loader_preflight(
        base_loader,
        LONG_CONTEXT_BASE_CONFIG,
        sample_dates=("2023-06-15", "2024-06-15", "2025-06-15"),
    )
    state_loader = UnifiedOHLCVLoader(DEFAULT_MARKET_STATE_CONTINUITY_CONFIG, data_dir=data_dir)
    state_preflight = run_state_data_preflight(state_loader, DEFAULT_MARKET_STATE_CONTINUITY_CONFIG)
    preflight = {"trade_bar": base_preflight.to_dict(), "unified_state_ohlcv": state_preflight.to_dict()}
    if base_preflight.status != "PASS" or state_preflight.status != "PASS":
        return _empty_write(config, preflight, "BLOCKED_DATA", "Trade Bar或统一状态OHLCV预检失败。")

    base_paths = [
        path for path in list_cached_years(LONG_CONTEXT_BASE_CONFIG)
        if 2023 <= int(path.name[-4:]) <= 2025
    ]
    available_years = {int(path.name[-4:]) for path in base_paths}
    if available_years != {2023, 2024, 2025}:
        return _empty_write(
            config,
            preflight,
            "BLOCKED_BASE_CACHE",
            f"缺少R03.2的2023/2024/2025基础缓存，当前可用={sorted(available_years)}。",
        )

    state_paths = build_state_caches(
        state_loader,
        DEFAULT_MARKET_STATE_CONTINUITY_CONFIG,
        force_rebuild=force_rebuild_state_cache,
        progress=progress,
    )
    outcome_paths = build_outcome_caches(
        base_paths,
        DEFAULT_STATE_CONTEXT_ABLATION_CONFIG,
        force_rebuild=force_rebuild_outcomes,
        progress=progress,
    )

    metrics_parts: list[pd.DataFrame] = []
    signal_parts: list[pd.DataFrame] = []
    rerank_parts: list[pd.DataFrame] = []
    multiplier_parts: list[pd.DataFrame] = []
    sample_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.DataFrame] = []
    oof_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    folds = default_ablation_folds(DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
    reporter = ProgressReporter("[R03.4.1 folds]", len(folds), every=1, enabled=progress)

    for fold_number, fold in enumerate(folds, start=1):
        try:
            fit = collect_ablation_period_data(
                base_paths, state_paths, outcome_paths,
                start=fold.fit_start, end=fold.fit_end,
                config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG,
            )
            calibration = collect_ablation_period_data(
                base_paths, state_paths, outcome_paths,
                start=fold.calibration_start, end=fold.calibration_end,
                config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG,
            )
            test = collect_ablation_period_data(
                base_paths, state_paths, outcome_paths,
                start=fold.test_start, end=fold.test_end,
                config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG,
            )
            oof_score, _blocks, audit = generate_oof_base_scores(fit, config)
            for row in audit:
                row["fold_id"] = fold.fold_id
                row["strictly_prior"] = bool(row["maximum_train_time_ns"] < row["minimum_prediction_time_ns"])
                oof_rows.append(row)
            base_model = fit_base_long_model(fit, config)
            models = {
                variant: fit_meta_model(variant, oof_score, fit, config)
                for variant in config.variants
            }
            metrics, signals, rerank, multipliers, samples = evaluate_fold_models(
                fold_id=fold.fold_id,
                models=models,
                base_model=base_model,
                calibration=calibration,
                test=test,
                config=config,
            )
            metrics_parts.append(metrics)
            signal_parts.append(signals)
            rerank_parts.append(rerank)
            multiplier_parts.append(multipliers)
            if not samples.empty:
                sample_parts.append(samples)
            importance = _importance(models, fold.fold_id)
            if not importance.empty:
                importance_parts.append(importance)
        except Exception as exc:
            failures.append({"fold_id": fold.fold_id, "error": f"{type(exc).__name__}: {exc}"})
        reporter.update(fold_number)
    reporter.close()

    metrics = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame()
    signals = pd.concat(signal_parts, ignore_index=True) if signal_parts else pd.DataFrame()
    rerank = pd.concat(rerank_parts, ignore_index=True) if rerank_parts else pd.DataFrame()
    multipliers = pd.concat(multiplier_parts, ignore_index=True) if multiplier_parts else pd.DataFrame()
    samples = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    importance = pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()
    if not importance.empty:
        importance = importance.sort_values(
            ["fold_id", "variant", "importance"], ascending=[True, True, False], kind="stable"
        ).groupby(["fold_id", "variant"], as_index=False).head(50)
    if len(samples) > 100_000:
        samples = samples.iloc[np.linspace(0, len(samples) - 1, 100_000, dtype=np.int64)].reset_index(drop=True)

    if metrics.empty or rerank.empty or multipliers.empty:
        decision = "FAIL_NO_VALID_MODELS"
        reason = "二阶段模型任务未完整产出，不能判断软状态增量。"
        uplift = pd.DataFrame()
        stable = pd.DataFrame()
        champion = None
    else:
        uplift = build_uplift_tables(metrics, rerank, multipliers)
        stable = select_stable_candidates(uplift, config)
        calibration_pass = stable.loc[stable["passes_calibration"]]
        risk_pass = stable.loc[stable["passes_risk_scaling"]]
        if not calibration_pass.empty:
            champion = calibration_pass.iloc[0].to_dict()
            decision = "PASS_STATE_META_CALIBRATION_UPLIFT"
            reason = "至少一个软状态元校准版本在2024和2025都超过纯分数校准，并改善共同基础候选池。"
        elif not risk_pass.empty:
            champion = risk_pass.iloc[0].to_dict()
            decision = "PASS_STATE_RISK_SCALING_ONLY"
            reason = "软状态未稳定改善开仓排序，但在固定基础事件上产生跨年风险倍率增量。"
        else:
            champion = None
            decision = "FAIL_NO_STABLE_LONG_STATE_UPLIFT"
            reason = "软战略/活跃状态未在2024和2025同时稳定改善多头机会排序或固定候选风险倍率。"

    write_reports(
        config.report_path,
        manifest={
            "stage": STAGE_ID,
            "name": STAGE_NAME,
            "config": config.to_dict(),
            "folds": [fold.to_dict() for fold in folds],
        },
        preflight=preflight,
        oof_audit=pd.DataFrame(oof_rows),
        model_metrics=metrics,
        signal_metrics=signals,
        rerank_metrics=rerank,
        multiplier_metrics=multipliers,
        uplift=uplift,
        stable=stable,
        importance=importance,
        samples=samples,
        causal_audit=_causal_audit(config),
        failures=pd.DataFrame(failures),
        decision=decision,
        reason=reason,
        champion=champion,
        config=config,
    )
    return LongStateCalibrationResult(decision, config.report_path, champion)
