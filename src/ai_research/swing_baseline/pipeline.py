#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03 medium-horizon swing research pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter

from .backtest import build_market_path, evaluate_prediction_scenarios, score_thresholds
from .config import DEFAULT_SWING_BASELINE_CONFIG, SwingBaselineConfig, SwingTargetSpec
from .dataset import (
    build_yearly_cache,
    create_loader,
    list_cached_years,
    load_year_shard,
    run_public_loader_preflight,
)
from .modeling import (
    ARCHITECTURES,
    collect_period_data,
    default_folds,
    feature_importance_rows,
    fit_model_bundle,
    probability_metrics,
    validate_model_dependencies,
)
from .reports import write_reports


@dataclass(frozen=True)
class SwingPipelineResult:
    decision: str
    report_dir: Path
    champion: dict[str, object] | None
    holdout: dict[str, object] | None


def _python_scalar(value: object) -> object:
    return value.item() if hasattr(value, "item") else value


def _row_dict(row: pd.Series) -> dict[str, object]:
    return {key: _python_scalar(value) for key, value in row.to_dict().items()}


def _target_by_id(config: SwingBaselineConfig, target_id: str) -> SwingTargetSpec:
    for target in config.target_specs:
        if target.target_id == target_id:
            return target
    raise KeyError(target_id)


def _label_balance(paths: Iterable[Path], config: SwingBaselineConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        shard = load_year_shard(path)
        label_map = shard.label_index
        year = int(pd.Timestamp(int(shard.decision_times_ns[0])).year)
        for target in config.target_specs:
            for direction in ("long", "short"):
                name = f"{target.target_id}_{direction}_quality"
                values = np.asarray(shard.labels[:, label_map[name]], dtype=float)
                valid = np.isfinite(values)
                rows.append(
                    {
                        "year": year,
                        "target_id": target.target_id,
                        "direction": direction,
                        "rows": int(valid.sum()),
                        "positive_rate": float(np.mean(values[valid])) if valid.any() else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def _attach_thresholds(
    scenarios: pd.DataFrame,
    thresholds: dict[float, tuple[float, float]],
) -> pd.DataFrame:
    if scenarios.empty:
        return scenarios
    out = scenarios.copy()
    out["long_threshold"] = out["quantile"].map(lambda value: thresholds[float(value)][0])
    out["short_threshold"] = out["quantile"].map(lambda value: thresholds[float(value)][1])
    return out


def _select_validation_champion(
    scenarios: pd.DataFrame,
    config: SwingBaselineConfig,
) -> dict[str, object] | None:
    if scenarios.empty:
        return None
    validation = scenarios.loc[scenarios["fold_id"] == "WF_2025"].copy()
    keys = ["architecture", "target_id", "quantile"]
    base = validation.loc[
        (validation["delay_minutes"] == config.execution_delay_minutes)
        & (validation["cost_multiplier"] == 1.0)
    ].copy()
    cost2 = validation.loc[
        (validation["delay_minutes"] == config.execution_delay_minutes)
        & (validation["cost_multiplier"] == 2.0)
    ][[*keys, "total_return"]].rename(columns={"total_return": "return_2x"})
    delay5 = validation.loc[
        (validation["delay_minutes"] == max(config.delay_scenarios_minutes))
        & (validation["cost_multiplier"] == 1.0)
    ][[*keys, "total_return"]].rename(columns={"total_return": "return_delay5"})
    candidates = base.merge(cost2, on=keys, how="left").merge(delay5, on=keys, how="left")
    candidates = candidates.loc[
        (candidates["trades"] >= 24)
        & (candidates["mean_net_return"] > 0)
        & (candidates["profit_factor"] > 1.20)
        & (candidates["total_return"] > 0)
        & (candidates["max_drawdown"] > -0.20)
        & (candidates["positive_quarter_ratio"] >= 0.50)
        & (candidates["return_2x"] > 0)
        & (candidates["return_delay5"] > 0)
        & (candidates["top5_removed_total_return"] > 0)
    ].copy()
    if candidates.empty:
        return None

    def neighbour_stable(row: pd.Series) -> bool:
        peers = base.loc[
            (base["architecture"] == row["architecture"])
            & (base["target_id"] == row["target_id"])
            & (base["quantile"] != row["quantile"])
        ].copy()
        if peers.empty:
            return False
        peers["distance"] = (peers["quantile"] - float(row["quantile"])).abs()
        neighbour = peers.sort_values("distance", kind="stable").iloc[0]
        return bool(
            int(neighbour["trades"]) >= 12
            and float(neighbour["mean_net_return"]) > 0
            and float(neighbour["profit_factor"]) > 1.0
            and float(neighbour["total_return"]) > 0
        )

    candidates["neighbour_stable"] = candidates.apply(neighbour_stable, axis=1)
    candidates = candidates.loc[candidates["neighbour_stable"]].copy()
    if candidates.empty:
        return None
    candidates["robust_score"] = (
        candidates["total_return"]
        + candidates["return_2x"]
        + candidates["return_delay5"]
        + candidates["top5_removed_total_return"]
        - candidates["max_drawdown"].abs()
    )
    row = candidates.sort_values(
        ["robust_score", "profit_factor", "trades"],
        ascending=[False, False, False],
        kind="stable",
    ).iloc[0]
    return _row_dict(row)


def _stress_enriched_row(
    scenarios: pd.DataFrame,
    *,
    fold_id: str,
    architecture: str,
    target_id: str,
    quantile: float,
    config: SwingBaselineConfig,
) -> dict[str, object] | None:
    scope = scenarios.loc[
        (scenarios["fold_id"] == fold_id)
        & (scenarios["architecture"] == architecture)
        & (scenarios["target_id"] == target_id)
        & (scenarios["quantile"] == quantile)
    ]
    base = scope.loc[
        (scope["delay_minutes"] == config.execution_delay_minutes)
        & (scope["cost_multiplier"] == 1.0)
    ]
    if base.empty:
        return None
    row = _row_dict(base.iloc[0])
    stress_cost = scope.loc[
        (scope["delay_minutes"] == config.execution_delay_minutes)
        & (scope["cost_multiplier"] == 2.0)
    ]
    stress_delay = scope.loc[
        (scope["delay_minutes"] == max(config.delay_scenarios_minutes))
        & (scope["cost_multiplier"] == 1.0)
    ]
    row["return_2x"] = float(stress_cost.iloc[0]["total_return"]) if not stress_cost.empty else float("nan")
    row["return_delay5"] = float(stress_delay.iloc[0]["total_return"]) if not stress_delay.empty else float("nan")
    return row


def _holdout_passes(row: dict[str, object] | None) -> bool:
    if not row:
        return False
    return bool(
        int(row.get("trades", 0)) >= 12
        and float(row.get("mean_net_return", 0.0)) > 0
        and float(row.get("profit_factor", 0.0)) > 1.15
        and float(row.get("total_return", 0.0)) > 0
        and float(row.get("max_drawdown", -1.0)) > -0.20
        and float(row.get("return_2x", 0.0)) > 0
        and float(row.get("return_delay5", 0.0)) > 0
        and float(row.get("top5_removed_total_return", 0.0)) > 0
    )


def _cache_manifest(config: SwingBaselineConfig) -> dict[str, object]:
    path = config.cache_path / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_pipeline(
    *,
    config: SwingBaselineConfig = DEFAULT_SWING_BASELINE_CONFIG,
    architectures: Iterable[str] = ARCHITECTURES,
    force_rebuild_cache: bool = False,
    data_dir: str | Path | None = None,
    progress: bool = True,
) -> SwingPipelineResult:
    config.validate()
    architectures = tuple(dict.fromkeys(architectures))
    dependency_status = validate_model_dependencies(architectures)
    config.report_path.mkdir(parents=True, exist_ok=True)
    loader = create_loader(config, data_dir=data_dir)
    preflight = run_public_loader_preflight(loader, config)
    if preflight.status != "PASS":
        reason = "公共1m Trade Bar Loader 的轻量预检未通过，R03未访问Raw文件，也未尝试自动重建数据。"
        write_reports(
            config.report_path,
            run_manifest={"config": config.to_dict(), "architectures": list(architectures), "dependencies": dependency_status},
            preflight=preflight.to_dict(),
            cache_manifest={},
            label_balance=pd.DataFrame(),
            prediction_metrics=pd.DataFrame(),
            scenario_summaries=pd.DataFrame(),
            trades=pd.DataFrame(),
            feature_importance=pd.DataFrame(),
            champion=None,
            holdout=None,
            decision="BLOCKED_PUBLIC_LOADER",
            reason=reason,
            config=config,
        )
        return SwingPipelineResult("BLOCKED_PUBLIC_LOADER", config.report_path, None, None)

    cache_paths = build_yearly_cache(
        loader,
        config,
        force_rebuild=force_rebuild_cache,
        progress=progress,
    )
    if not cache_paths:
        cache_paths = list_cached_years(config)
    first_shard = load_year_shard(cache_paths[0])
    feature_columns = first_shard.full_feature_columns
    folds = default_folds(config)
    prediction_rows: list[dict[str, object]] = []
    scenario_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, object]] = []
    model_metadata: list[dict[str, object]] = []
    model_failures: list[dict[str, object]] = []

    jobs = len(architectures) * len(config.target_specs) * 2
    reporter = ProgressReporter("[R03 models] validation jobs", jobs, every=1, enabled=progress)
    completed = 0
    for fold in folds[:2]:
        market_path = build_market_path(
            cache_paths,
            fold.test_start,
            fold.test_end + pd.Timedelta(hours=config.max_hold_hours + 1),
        )
        for target in config.target_specs:
            long_label = f"{target.target_id}_long_quality"
            short_label = f"{target.target_id}_short_quality"
            calibration = collect_period_data(
                cache_paths,
                fold.calibration_start,
                fold.calibration_end,
                label_names=(long_label, short_label),
            )
            test = collect_period_data(
                cache_paths,
                fold.test_start,
                fold.test_end,
                label_names=(long_label, short_label),
            )
            for architecture in architectures:
                try:
                    bundle, metadata = fit_model_bundle(architecture, target, cache_paths, fold, config)
                    model_metadata.append(metadata)
                    calibration_scores = bundle.predict(calibration.high_x, calibration.full_x)
                    test_scores = bundle.predict(test.high_x, test.full_x)
                    thresholds = score_thresholds(
                        calibration_scores["score_long"],
                        calibration_scores["score_short"],
                        config.signal_quantiles,
                    )
                    for direction, label_name in (("long", long_label), ("short", short_label)):
                        metrics = probability_metrics(test.labels[label_name], test_scores[f"score_{direction}"])
                        prediction_rows.append(
                            {
                                "fold_id": fold.fold_id,
                                "architecture": architecture,
                                "target_id": target.target_id,
                                "direction": direction,
                                "status": "PASS",
                                "error": "",
                                **metrics.to_dict(),
                            }
                        )
                    scenarios, trades = evaluate_prediction_scenarios(
                        fold_id=fold.fold_id,
                        architecture=architecture,
                        target=target,
                        period=test,
                        feature_columns=feature_columns,
                        score_long=test_scores["score_long"],
                        score_short=test_scores["score_short"],
                        thresholds_by_quantile=thresholds,
                        market_path=market_path,
                        config=config,
                    )
                    scenario_frames.append(_attach_thresholds(scenarios, thresholds))
                    if not trades.empty:
                        trade_frames.append(trades)
                    importance_rows.extend(feature_importance_rows(bundle))
                except (RuntimeError, ValueError) as exc:
                    model_failures.append(
                        {
                            "fold_id": fold.fold_id,
                            "architecture": architecture,
                            "target_id": target.target_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    for direction in ("long", "short"):
                        prediction_rows.append(
                            {
                                "fold_id": fold.fold_id,
                                "architecture": architecture,
                                "target_id": target.target_id,
                                "direction": direction,
                                "status": "SKIPPED",
                                "error": str(exc),
                                "rows": 0,
                                "positive_rate": float("nan"),
                                "roc_auc": float("nan"),
                                "average_precision": float("nan"),
                                "brier": float("nan"),
                                "score_mean": float("nan"),
                                "score_std": float("nan"),
                            }
                        )
                finally:
                    completed += 1
                    reporter.update(completed)
    reporter.close()

    scenario_summaries = pd.concat(scenario_frames, ignore_index=True) if scenario_frames else pd.DataFrame()
    champion = _select_validation_champion(scenario_summaries, config)
    holdout_row: dict[str, object] | None = None

    if champion is not None:
        holdout_fold = folds[2]
        architecture = str(champion["architecture"])
        target = _target_by_id(config, str(champion["target_id"]))
        quantile = float(champion["quantile"])
        bundle, metadata = fit_model_bundle(architecture, target, cache_paths, holdout_fold, config)
        model_metadata.append(metadata)
        long_label = f"{target.target_id}_long_quality"
        short_label = f"{target.target_id}_short_quality"
        calibration = collect_period_data(
            cache_paths,
            holdout_fold.calibration_start,
            holdout_fold.calibration_end,
            label_names=(long_label, short_label),
        )
        test = collect_period_data(
            cache_paths,
            holdout_fold.test_start,
            holdout_fold.test_end,
            label_names=(long_label, short_label),
        )
        calibration_scores = bundle.predict(calibration.high_x, calibration.full_x)
        test_scores = bundle.predict(test.high_x, test.full_x)
        thresholds = score_thresholds(
            calibration_scores["score_long"],
            calibration_scores["score_short"],
            (quantile,),
        )
        for direction, label_name in (("long", long_label), ("short", short_label)):
            metrics = probability_metrics(test.labels[label_name], test_scores[f"score_{direction}"])
            prediction_rows.append(
                {
                    "fold_id": holdout_fold.fold_id,
                    "architecture": architecture,
                    "target_id": target.target_id,
                    "direction": direction,
                    **metrics.to_dict(),
                }
            )
        market_path = build_market_path(
            cache_paths,
            holdout_fold.test_start,
            holdout_fold.test_end + pd.Timedelta(hours=config.max_hold_hours + 1),
        )
        scenarios, trades = evaluate_prediction_scenarios(
            fold_id=holdout_fold.fold_id,
            architecture=architecture,
            target=target,
            period=test,
            feature_columns=feature_columns,
            score_long=test_scores["score_long"],
            score_short=test_scores["score_short"],
            thresholds_by_quantile=thresholds,
            market_path=market_path,
            config=config,
        )
        scenarios = _attach_thresholds(scenarios, thresholds)
        scenario_summaries = pd.concat([scenario_summaries, scenarios], ignore_index=True)
        if not trades.empty:
            trade_frames.append(trades)
        importance_rows.extend(feature_importance_rows(bundle))
        holdout_row = _stress_enriched_row(
            scenario_summaries,
            fold_id=holdout_fold.fold_id,
            architecture=architecture,
            target_id=target.target_id,
            quantile=quantile,
            config=config,
        )

    if champion is None:
        decision = "FAIL_VALIDATION"
        reason = (
            "2025验证期没有候选同时通过交易数、PF、2x成本、5分钟延迟、回撤、季度稳定性、"
            "去掉前5大盈利和相邻分位数稳定性门槛；因此没有查看R03的2026冠军结果来救模型。"
        )
    elif _holdout_passes(holdout_row):
        decision = "PASS_SWING_EDGE"
        reason = "2025验证冠军在2026锁定样本外再次通过结构退出、真实成本和延迟压力，可以保留为Swing主Sleeve候选。"
    else:
        decision = "FAIL_LOCKED_HOLDOUT"
        reason = "2025验证期存在候选，但在2026锁定样本外未通过成本、延迟、左尾或稳定性门槛，暂不具备实盘资格。"

    run_manifest = {
        "stage": "R03",
        "name": "Medium-horizon swing direction and entry baseline",
        "config": config.to_dict(),
        "architectures": list(architectures),
        "dependencies": dependency_status,
        "folds": [fold.to_dict() for fold in folds],
        "model_metadata": model_metadata,
        "model_failures": model_failures,
        "holdout_disclosure": (
            "2026H1 was already observed at project level in R01. R03 does not use it for model, target, "
            "quantile, or threshold selection; it is a locked out-of-sample benchmark, not a virgin holdout."
        ),
    }
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    write_reports(
        config.report_path,
        run_manifest=run_manifest,
        preflight=preflight.to_dict(),
        cache_manifest=_cache_manifest(config),
        label_balance=_label_balance(cache_paths, config),
        prediction_metrics=pd.DataFrame(prediction_rows),
        scenario_summaries=scenario_summaries,
        trades=all_trades,
        feature_importance=pd.DataFrame(importance_rows),
        champion=champion,
        holdout=holdout_row,
        decision=decision,
        reason=reason,
        config=config,
    )
    return SwingPipelineResult(decision, config.report_path, champion, holdout_row)
