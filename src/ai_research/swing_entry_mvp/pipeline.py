#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R03.1 exact-label swing entry MVP pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.swing_baseline.backtest import build_market_path
from src.ai_research.swing_baseline.dataset import (
    build_yearly_cache,
    create_loader,
    list_cached_years,
    load_year_shard,
    run_public_loader_preflight,
)
from src.ai_research.swing_baseline.modeling import (
    default_folds,
    feature_importance_rows,
    fit_model_bundle_from_period,
    probability_metrics,
    validate_model_dependencies,
)
from src.research_common.progress import ProgressReporter

from .backtest import evaluate_scenarios, score_thresholds
from .config import DEFAULT_SWING_ENTRY_MVP_CONFIG, SwingEntryMvpConfig
from .outcomes import (
    build_outcome_overlays,
    collect_exact_period_data,
    overlay_manifest_rows,
    validate_outcome_dependency,
)
from .reports import write_reports


@dataclass(frozen=True)
class SwingEntryMvpResult:
    decision: str
    report_dir: Path
    champion: dict[str, object] | None
    holdout: dict[str, object] | None


ReportWriter = Callable[..., dict[str, Path]]


def _scalar(value: object) -> object:
    return value.item() if hasattr(value, "item") else value


def _row_dict(row: pd.Series) -> dict[str, object]:
    return {key: _scalar(value) for key, value in row.to_dict().items()}


def _base_scope(scenarios: pd.DataFrame, fold_id: str, config: SwingEntryMvpConfig) -> pd.DataFrame:
    return scenarios.loc[
        (scenarios["fold_id"] == fold_id)
        & (scenarios["delay_minutes"] == config.base.execution_delay_minutes)
        & (scenarios["cost_multiplier"] == 1.0)
    ].copy()


def _stress_table(scenarios: pd.DataFrame, fold_id: str, config: SwingEntryMvpConfig) -> pd.DataFrame:
    keys = ["architecture", "target_id", "direction", "exit_policy", "quantile"]
    base = _base_scope(scenarios, fold_id, config)
    cost2 = scenarios.loc[
        (scenarios["fold_id"] == fold_id)
        & (scenarios["delay_minutes"] == config.base.execution_delay_minutes)
        & (scenarios["cost_multiplier"] == 2.0)
    ][[*keys, "total_return"]].rename(columns={"total_return": "return_2x"})
    delay = scenarios.loc[
        (scenarios["fold_id"] == fold_id)
        & (scenarios["delay_minutes"] == max(config.base.delay_scenarios_minutes))
        & (scenarios["cost_multiplier"] == 1.0)
    ][[*keys, "total_return"]].rename(columns={"total_return": "return_delay_max"})
    return base.merge(cost2, on=keys, how="left").merge(delay, on=keys, how="left")


def _select_validation_champion(
    scenarios: pd.DataFrame,
    config: SwingEntryMvpConfig,
) -> dict[str, object] | None:
    if scenarios.empty:
        return None
    keys = ["architecture", "target_id", "direction", "exit_policy", "quantile"]
    dev = _stress_table(scenarios, "WF_2024", config)
    validation = _stress_table(scenarios, "WF_2025", config)
    if validation.empty or dev.empty:
        return None
    support = dev[
        [
            *keys,
            "trades",
            "profit_factor",
            "total_return",
            "max_drawdown",
            "target_hit_rate",
        ]
    ].rename(
        columns={
            "trades": "dev_trades",
            "profit_factor": "dev_profit_factor",
            "total_return": "dev_total_return",
            "max_drawdown": "dev_max_drawdown",
            "target_hit_rate": "dev_target_hit_rate",
        }
    )
    candidates = validation.merge(support, on=keys, how="left")
    candidates = candidates.loc[
        (candidates["trades"] >= 12)
        & (candidates["mean_net_return"] > 0)
        & (candidates["profit_factor"] > 1.20)
        & (candidates["total_return"] > 0)
        & (candidates["max_drawdown"] > -0.20)
        & (candidates["positive_quarter_ratio"] >= 0.50)
        & (candidates["return_2x"] > 0)
        & (candidates["return_delay_max"] > 0)
        & (candidates["top3_removed_total_return"] > 0)
        & (candidates["target_hit_rate"] >= 0.25)
        & (candidates["dev_trades"] >= 8)
        & (candidates["dev_profit_factor"] > 1.0)
        & (candidates["dev_total_return"] > 0)
        & (candidates["dev_max_drawdown"] > -0.25)
    ].copy()
    if candidates.empty:
        return None

    base_2025 = _base_scope(scenarios, "WF_2025", config)

    def neighbour_stable(row: pd.Series) -> bool:
        peers = base_2025.loc[
            (base_2025["architecture"] == row["architecture"])
            & (base_2025["target_id"] == row["target_id"])
            & (base_2025["direction"] == row["direction"])
            & (base_2025["exit_policy"] == row["exit_policy"])
            & (base_2025["quantile"] != row["quantile"])
        ].copy()
        if peers.empty:
            return False
        peers["distance"] = (peers["quantile"] - float(row["quantile"])).abs()
        neighbour = peers.sort_values("distance", kind="stable").iloc[0]
        return bool(
            int(neighbour["trades"]) >= 8
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
        + candidates["return_delay_max"]
        + candidates["top3_removed_total_return"]
        + 0.5 * candidates["dev_total_return"]
        - candidates["max_drawdown"].abs()
        - 0.5 * candidates["dev_max_drawdown"].abs()
    )
    row = candidates.sort_values(
        ["robust_score", "profit_factor", "target_hit_rate", "trades"],
        ascending=[False, False, False, False],
        kind="stable",
    ).iloc[0]
    return _row_dict(row)


def _stress_enriched_holdout(
    scenarios: pd.DataFrame,
    champion: dict[str, object],
    config: SwingEntryMvpConfig,
) -> dict[str, object] | None:
    scope = scenarios.loc[
        (scenarios["fold_id"] == "WF_2026")
        & (scenarios["architecture"] == champion["architecture"])
        & (scenarios["target_id"] == champion["target_id"])
        & (scenarios["direction"] == champion["direction"])
        & (scenarios["exit_policy"] == champion["exit_policy"])
        & (scenarios["quantile"] == champion["quantile"])
    ]
    base = scope.loc[
        (scope["delay_minutes"] == config.base.execution_delay_minutes)
        & (scope["cost_multiplier"] == 1.0)
    ]
    if base.empty:
        return None
    row = _row_dict(base.iloc[0])
    cost2 = scope.loc[
        (scope["delay_minutes"] == config.base.execution_delay_minutes)
        & (scope["cost_multiplier"] == 2.0)
    ]
    delay = scope.loc[
        (scope["delay_minutes"] == max(config.base.delay_scenarios_minutes))
        & (scope["cost_multiplier"] == 1.0)
    ]
    row["return_2x"] = float(cost2.iloc[0]["total_return"]) if not cost2.empty else float("nan")
    row["return_delay_max"] = float(delay.iloc[0]["total_return"]) if not delay.empty else float("nan")
    return row


def _holdout_passes(row: dict[str, object] | None) -> bool:
    if not row:
        return False
    return bool(
        int(row.get("trades", 0)) >= 6
        and float(row.get("mean_net_return", 0.0)) > 0
        and float(row.get("profit_factor", 0.0)) > 1.15
        and float(row.get("total_return", 0.0)) > 0
        and float(row.get("max_drawdown", -1.0)) > -0.20
        and float(row.get("return_2x", 0.0)) > 0
        and float(row.get("return_delay_max", 0.0)) > 0
        and float(row.get("target_hit_rate", 0.0)) >= 0.20
    )


def _target(config: SwingEntryMvpConfig, target_id: str):
    for target in config.base.target_specs:
        if target.target_id == target_id:
            return target
    raise KeyError(target_id)


def _cache_manifest(config: SwingEntryMvpConfig) -> dict[str, object]:
    path = config.base.cache_path / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_pipeline(
    *,
    config: SwingEntryMvpConfig = DEFAULT_SWING_ENTRY_MVP_CONFIG,
    force_rebuild_exact_labels: bool = False,
    force_rebuild_base_cache: bool = False,
    data_dir: str | Path | None = None,
    progress: bool = True,
    feature_profile: str = "r03_multiframe_v1",
    stage_id: str = "R03.1",
    stage_name: str = "Exact-path 3%-5% swing entry MVP",
    report_writer: ReportWriter = write_reports,
    pass_decision: str = "PASS_SWING_ENTRY_MVP",
) -> SwingEntryMvpResult:
    config.validate()
    model_dependencies = validate_model_dependencies(config.architectures)
    outcome_dependencies = validate_outcome_dependency()
    config.report_path.mkdir(parents=True, exist_ok=True)
    loader = create_loader(config.base, data_dir=data_dir)
    preflight = run_public_loader_preflight(loader, config.base)
    if preflight.status != "PASS":
        reason = f"公共1m Trade Bar Loader轻量预检未通过；{stage_id}未访问Raw Trades，也未自动重建数据。"
        report_writer(
            config.report_path,
            run_manifest={"stage": stage_id, "name": stage_name, "feature_profile": feature_profile, "config": config.to_dict()},
            preflight=preflight.to_dict(),
            base_cache_manifest={},
            exact_label_diagnostics=pd.DataFrame(),
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
        return SwingEntryMvpResult("BLOCKED_PUBLIC_LOADER", config.report_path, None, None)

    shard_paths = build_yearly_cache(
        loader,
        config.base,
        force_rebuild=force_rebuild_base_cache,
        progress=progress,
        feature_profile=feature_profile,
    )
    if not shard_paths:
        shard_paths = list_cached_years(config.base)
    overlay_paths = build_outcome_overlays(
        shard_paths,
        config,
        force_rebuild=force_rebuild_exact_labels,
        progress=progress,
    )
    first_shard = load_year_shard(shard_paths[0])
    folds = default_folds(config.base)
    prediction_rows: list[dict[str, object]] = []
    scenario_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, object]] = []
    model_metadata: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    jobs = 2 * len(config.base.target_specs) * len(config.architectures)
    reporter = ProgressReporter(f"[{stage_id} models] validation jobs", jobs, every=1, enabled=progress)
    completed = 0
    for fold in folds[:2]:
        market_path = build_market_path(
            shard_paths,
            fold.test_start,
            fold.test_end + pd.Timedelta(hours=config.base.max_hold_hours + 1),
        )
        for target in config.base.target_specs:
            fit = collect_exact_period_data(shard_paths, overlay_paths, fold.fit_start, fold.fit_end, target=target)
            calibration = collect_exact_period_data(
                shard_paths, overlay_paths, fold.calibration_start, fold.calibration_end, target=target
            )
            test = collect_exact_period_data(shard_paths, overlay_paths, fold.test_start, fold.test_end, target=target)
            for architecture in config.architectures:
                try:
                    bundle, metadata = fit_model_bundle_from_period(
                        architecture,
                        target,
                        fit,
                        high_columns=first_shard.high_feature_columns,
                        full_columns=first_shard.full_feature_columns,
                        config=config.base,
                        metadata_context={"fold": fold.to_dict(), "label_definition": "exact_target_before_adverse", "feature_profile": feature_profile},
                    )
                    model_metadata.append(metadata)
                    calibration_scores = bundle.predict(calibration.high_x, calibration.full_x)
                    test_scores = bundle.predict(test.high_x, test.full_x)
                    thresholds = {
                        "long": score_thresholds(calibration_scores["score_long"], config.base.signal_quantiles),
                        "short": score_thresholds(calibration_scores["score_short"], config.base.signal_quantiles),
                    }
                    for direction in config.direction_modes:
                        label = f"{target.target_id}_{direction}_quality"
                        metrics = probability_metrics(test.labels[label], test_scores[f"score_{direction}"])
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
                    scenarios, trades = evaluate_scenarios(
                        fold_id=fold.fold_id,
                        architecture=architecture,
                        target=target,
                        period=test,
                        score_long=test_scores["score_long"],
                        score_short=test_scores["score_short"],
                        thresholds=thresholds,
                        market_path=market_path,
                        config=config,
                    )
                    scenario_frames.append(scenarios)
                    if not trades.empty:
                        trade_frames.append(trades)
                    importance_rows.extend(feature_importance_rows(bundle))
                except (RuntimeError, ValueError) as exc:
                    failures.append(
                        {
                            "fold_id": fold.fold_id,
                            "architecture": architecture,
                            "target_id": target.target_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                finally:
                    completed += 1
                    reporter.update(completed)
    reporter.close()

    scenarios = pd.concat(scenario_frames, ignore_index=True) if scenario_frames else pd.DataFrame()
    champion = _select_validation_champion(scenarios, config)
    holdout: dict[str, object] | None = None

    if champion is not None:
        fold = folds[2]
        target = _target(config, str(champion["target_id"]))
        fit = collect_exact_period_data(shard_paths, overlay_paths, fold.fit_start, fold.fit_end, target=target)
        calibration = collect_exact_period_data(
            shard_paths, overlay_paths, fold.calibration_start, fold.calibration_end, target=target
        )
        test = collect_exact_period_data(shard_paths, overlay_paths, fold.test_start, fold.test_end, target=target)
        bundle, metadata = fit_model_bundle_from_period(
            str(champion["architecture"]),
            target,
            fit,
            high_columns=first_shard.high_feature_columns,
            full_columns=first_shard.full_feature_columns,
            config=config.base,
            metadata_context={"fold": fold.to_dict(), "label_definition": "exact_target_before_adverse", "feature_profile": feature_profile},
        )
        model_metadata.append(metadata)
        cal_scores = bundle.predict(calibration.high_x, calibration.full_x)
        test_scores = bundle.predict(test.high_x, test.full_x)
        direction = str(champion["direction"])
        quantile = float(champion["quantile"])
        thresholds = {
            "long": score_thresholds(cal_scores["score_long"], (quantile,)),
            "short": score_thresholds(cal_scores["score_short"], (quantile,)),
        }
        metrics = probability_metrics(
            test.labels[f"{target.target_id}_{direction}_quality"],
            test_scores[f"score_{direction}"],
        )
        prediction_rows.append(
            {
                "fold_id": fold.fold_id,
                "architecture": champion["architecture"],
                "target_id": target.target_id,
                "direction": direction,
                "status": "PASS",
                "error": "",
                **metrics.to_dict(),
            }
        )
        selected_policy = next(policy for policy in config.exit_policies if policy.policy_id == champion["exit_policy"])
        holdout_config = SwingEntryMvpConfig(
            base=config.base,
            exact_label_cache_dir=config.exact_label_cache_dir,
            report_dir=config.report_dir,
            architectures=(str(champion["architecture"]),),
            direction_modes=(direction,),
            exit_policies=(selected_policy,),
            score_margin=config.score_margin,
            protection_trigger_fraction=config.protection_trigger_fraction,
            locked_profit_fraction=config.locked_profit_fraction,
            cooldown_minutes=config.cooldown_minutes,
            same_bar_policy=config.same_bar_policy,
        )
        market_path = build_market_path(
            shard_paths,
            fold.test_start,
            fold.test_end + pd.Timedelta(hours=config.base.max_hold_hours + 1),
        )
        holdout_scenarios, holdout_trades = evaluate_scenarios(
            fold_id=fold.fold_id,
            architecture=str(champion["architecture"]),
            target=target,
            period=test,
            score_long=test_scores["score_long"],
            score_short=test_scores["score_short"],
            thresholds=thresholds,
            market_path=market_path,
            config=holdout_config,
        )
        scenarios = pd.concat([scenarios, holdout_scenarios], ignore_index=True)
        if not holdout_trades.empty:
            trade_frames.append(holdout_trades)
        importance_rows.extend(feature_importance_rows(bundle))
        holdout = _stress_enriched_holdout(scenarios, champion, config)

    if champion is None:
        decision = "FAIL_VALIDATION"
        reason = (
            "没有方向/目标/退出规则组合同时在2024开发样本外与2025验证期通过真实成本、2x成本、"
            f"最多5分钟延迟、左尾、季度稳定性和相邻阈值门槛。{stage_id}因此没有用2026结果救模型。"
        )
    elif _holdout_passes(holdout):
        decision = pass_decision
        reason = (
            "同一方向的3%/5%开仓模型在2024、2025和2026锁定样本外均通过目标先于风险线的"
            "真实路径回放及成本压力，可进入模型导出与AetherEdge影子实盘MVP阶段。"
        )
    else:
        decision = "FAIL_LOCKED_HOLDOUT"
        reason = "2024/2025存在候选，但2026锁定样本外未通过，不能迁移实盘。"

    run_manifest = {
        "stage": stage_id,
        "name": stage_name,
        "feature_profile": feature_profile,
        "config": config.to_dict(),
        "dependencies": {**model_dependencies, **outcome_dependencies},
        "folds": [fold.to_dict() for fold in folds],
        "model_metadata": model_metadata,
        "model_failures": failures,
        "key_contracts": [
            "No minimum holding duration.",
            "Exact target-before-adverse labels; same-minute ambiguity is adverse-first.",
            "No 15m trend invalidation or model-reversal exit.",
            "Long and short are validated independently; a one-sided MVP is allowed.",
            "2026 is not used for candidate selection.",
        ],
    }
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    report_writer(
        config.report_path,
        run_manifest=run_manifest,
        preflight=preflight.to_dict(),
        base_cache_manifest=_cache_manifest(config),
        exact_label_diagnostics=pd.DataFrame(overlay_manifest_rows(overlay_paths)),
        prediction_metrics=pd.DataFrame(prediction_rows),
        scenario_summaries=scenarios,
        trades=all_trades,
        feature_importance=pd.DataFrame(importance_rows),
        champion=champion,
        holdout=holdout,
        decision=decision,
        reason=reason,
        config=config,
    )
    return SwingEntryMvpResult(decision, config.report_path, champion, holdout)
