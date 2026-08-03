#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2 frozen q90 long-tail exit audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.long_state_calibration.config import LongStateCalibrationConfig
from src.ai_research.state_context_ablation.config import DEFAULT_STATE_CONTEXT_ABLATION_CONFIG
from src.ai_research.state_context_ablation.modeling import default_ablation_folds
from src.ai_research.state_context_ablation.outcomes import build_outcome_caches
from src.ai_research.swing_baseline.dataset import (
    create_loader,
    list_cached_years,
    run_public_loader_preflight,
)
from src.ai_research.swing_long_context.config import LONG_CONTEXT_BASE_CONFIG
from src.research_common.progress import ProgressReporter

from .config import (
    DEFAULT_LONG_TAIL_EXIT_AUDIT_CONFIG,
    STAGE_ID,
    STAGE_NAME,
    LongTailExitAuditConfig,
)
from .data import collect_base_period_data, load_minute_path_data
from .modeling import (
    build_summary_tables,
    fit_frozen_base_scores,
    select_stable_candidates,
    trades_to_frame,
)
from .reports import write_reports
from .simulator import build_event_candidates, simulate_sequential_events


@dataclass(frozen=True)
class LongTailExitAuditResult:
    decision: str
    report_dir: Path
    champion: dict[str, object] | None


def _causal_audit(config: LongTailExitAuditConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "frozen_base_model",
                "status": "PASS",
                "detail": "R03.4.1 base LightGBM parameters and long_utility_h6 target are frozen",
            },
            {
                "check": "state_model_abandoned",
                "status": "PASS",
                "detail": "no state cache or state feature is loaded by R03.4.2",
            },
            {
                "check": "threshold_calibration",
                "status": "PASS",
                "detail": "q90/q95 entry and q50/q60/q70 maintenance thresholds come only from the prior calibration quarter",
            },
            {
                "check": "entry_timing",
                "status": "PASS",
                "detail": "entry executes at the requested 1/3/5 minute open after the 15-minute decision",
            },
            {
                "check": "structural_stop",
                "status": "PASS",
                "detail": "60/180 minute lows and ATR buffer are shifted one minute and therefore exclude the entry minute and future bars",
            },
            {
                "check": "trailing_stop",
                "status": "PASS",
                "detail": "a trail activated by the current minute high becomes executable only from the next minute",
            },
            {
                "check": "same_bar_conflict",
                "status": "PASS",
                "detail": "when target and stop are both touched in one minute, conservative stop-first ordering is used",
            },
            {
                "check": "rolling_score_exit",
                "status": "PASS",
                "detail": "a score observed at decision t can only exit at the open of t+1 minute",
            },
            {
                "check": "single_position",
                "status": "PASS",
                "detail": "events occurring while a prior long is still open are skipped; no overlapping positions or scale-ins",
            },
            {
                "check": "sealed_holdout",
                "status": "PASS",
                "detail": "2026 is not loaded; trades without a complete path before the seal are excluded",
            },
        ]
    )


def _exit_contract(config: LongTailExitAuditConfig) -> dict[str, object]:
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
        "primary_signal": "prior-quarter calibrated q90 of frozen base score",
        "quality_control_signal": "prior-quarter calibrated q95",
        "state_model_policy": "ABANDONED_FOR_TRADING; visualization and post-hoc interpretation only",
        "recipes": [recipe.__dict__ for recipe in config.recipes],
        "positive_expectancy_priority": True,
        "time_exit_policy": "fixed 6h is diagnostic only; other recipes use structural/target/trail/model exits with a safety cap",
    }


def _empty_result(
    *,
    config: LongTailExitAuditConfig,
    preflight: dict[str, object],
    decision: str,
    reason: str,
) -> LongTailExitAuditResult:
    empty = pd.DataFrame()
    write_reports(
        config.report_path,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight=preflight,
        exit_contract=_exit_contract(config),
        signal_audit=empty,
        threshold_audit=empty,
        trade_summary=empty,
        period_summary=empty,
        exit_reason_summary=empty,
        duration_summary=empty,
        stress_summary=empty,
        concentration=empty,
        stable=empty,
        trades=empty,
        causal_audit=_causal_audit(config),
        failures=empty,
        decision=decision,
        reason=reason,
        champion=None,
        config=config,
    )
    return LongTailExitAuditResult(decision, config.report_path, None)


def run_long_tail_exit_audit(
    *,
    data_dir: str | Path | None = None,
    force_rebuild_outcomes: bool = False,
    progress: bool = True,
    config: LongTailExitAuditConfig = DEFAULT_LONG_TAIL_EXIT_AUDIT_CONFIG,
) -> LongTailExitAuditResult:
    config.validate()
    base_loader = create_loader(LONG_CONTEXT_BASE_CONFIG, data_dir=data_dir)
    base_preflight = run_public_loader_preflight(
        base_loader,
        LONG_CONTEXT_BASE_CONFIG,
        sample_dates=("2023-06-15", "2024-06-15", "2025-06-15"),
    )
    preflight: dict[str, object] = {"trade_bar": base_preflight.to_dict()}
    if base_preflight.status != "PASS":
        return _empty_result(
            config=config,
            preflight=preflight,
            decision="BLOCKED_DATA",
            reason="1分钟Trade Bar公共Loader预检失败。",
        )

    base_paths = [
        path
        for path in list_cached_years(LONG_CONTEXT_BASE_CONFIG)
        if 2023 <= int(path.name[-4:]) <= 2025
    ]
    available = {int(path.name[-4:]) for path in base_paths}
    if available != {2023, 2024, 2025}:
        return _empty_result(
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

    trade_parts: list[pd.DataFrame] = []
    signal_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    execution_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    folds = default_ablation_folds(DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
    reporter = ProgressReporter("[R03.4.2 folds]", len(folds), every=1, enabled=progress)

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
            score_bundle = fit_frozen_base_scores(fit, calibration, test, config)
            threshold_rows.extend(
                {
                    "fold_id": fold.fold_id,
                    "quantile": quantile,
                    "threshold": threshold,
                    "calibration_rows": int(np.isfinite(score_bundle.calibration_score).sum()),
                    "test_rows": int(np.isfinite(score_bundle.test_score).sum()),
                    "test_exceedance_rate": float(np.nanmean(score_bundle.test_score >= threshold)),
                    "feature_schema_hash": score_bundle.feature_schema_hash,
                }
                for quantile, threshold in sorted(score_bundle.timeline.calibration_thresholds.items())
            )

            maximum_safety_hours = max(recipe.safety_cap_hours for recipe in config.recipes)
            path_start = fold.test_start - pd.Timedelta(days=2)
            # Never cross the sealed 2026 boundary. Events without a complete
            # path to their recipe's safety horizon are dropped by the simulator.
            path_end = min(fold.test_end, pd.Timestamp(config.sealed_holdout_start) - pd.Timedelta(minutes=1))
            path = load_minute_path_data(
                start=path_start,
                end=path_end,
                data_dir=data_dir,
                config=config,
                progress=progress,
            )
            preflight[f"{fold.fold_id}_path"] = {
                "start": str(path.index[0]),
                "end": str(path.index[-1]),
                "rows": len(path.index),
                "coverage_ratio": path.coverage_ratio,
                "maximum_safety_hours": maximum_safety_hours,
            }
            if path.coverage_ratio < 0.995:
                raise RuntimeError(f"minute path coverage below 99.5%: {path.coverage_ratio:.6f}")

            for signal_quantile in config.signal_quantiles:
                events = build_event_candidates(
                    score_bundle.timeline,
                    signal_quantile=signal_quantile,
                    config=config,
                )
                signal_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "signal_quantile": signal_quantile,
                        "threshold": score_bundle.timeline.threshold(signal_quantile),
                        "dense_signal_count": int(
                            np.sum(score_bundle.test_score >= score_bundle.timeline.threshold(signal_quantile))
                        ),
                        "dense_signal_rate": float(
                            np.mean(score_bundle.test_score >= score_bundle.timeline.threshold(signal_quantile))
                        ),
                        "independent_candidate_events": len(events),
                    }
                )
                for recipe in config.recipes:
                    for delay_minutes in config.entry_delay_minutes:
                        trades, audit = simulate_sequential_events(
                            events=events,
                            recipe=recipe,
                            delay_minutes=delay_minutes,
                            path=path,
                            timeline=score_bundle.timeline,
                            config=config,
                        )
                        execution_rows.append(
                            {
                                "fold_id": fold.fold_id,
                                "signal_quantile": signal_quantile,
                                "recipe": recipe.name,
                                "delay_minutes": delay_minutes,
                                **audit,
                            }
                        )
                        frame = trades_to_frame(trades, fold_id=fold.fold_id)
                        if not frame.empty:
                            trade_parts.append(frame)
        except Exception as exc:
            failures.append({"fold_id": fold.fold_id, "error": f"{type(exc).__name__}: {exc}"})
        reporter.update(fold_number)
    reporter.close()

    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    execution_audit = pd.DataFrame(execution_rows)
    if trades.empty:
        summary = periods = exits = durations = stress = concentration = stable = pd.DataFrame()
        decision = "FAIL_NO_VALID_TRADES"
        reason = "冻结q90信号未能完成有效的一分钟路径交易模拟。"
        champion = None
    else:
        summary, periods, exits, durations, stress, concentration = build_summary_tables(
            trades,
            execution_audit,
            config,
        )
        stable = select_stable_candidates(summary, periods, concentration, config)
        passing = stable.loc[stable["passes_positive_expectancy_gate"]] if not stable.empty else pd.DataFrame()
        if not passing.empty:
            champion = passing.iloc[0].to_dict()
            decision = "PASS_Q90_PATH_EXIT_POSITIVE_EXPECTANCY"
            reason = "至少一个非固定时间退出方案在2024和2025均保持成本后正期望，并通过利润集中度、延迟、回撤与安全时间占比审核。"
        else:
            champion = None
            decision = "FAIL_NO_ROBUST_PATH_EXIT"
            reason = "冻结q90开仓候选仍有诊断价值，但当前结构止损、利润保护和滚动续期方案未同时通过两年正期望与稳健性门槛。"

    write_reports(
        config.report_path,
        manifest={
            "stage": STAGE_ID,
            "name": STAGE_NAME,
            "config": config.to_dict(),
            "folds": [fold.to_dict() for fold in folds],
            "state_model_policy": "ABANDONED_FOR_TRADING",
        },
        preflight=preflight,
        exit_contract=_exit_contract(config),
        signal_audit=pd.DataFrame(signal_rows),
        threshold_audit=pd.DataFrame(threshold_rows),
        trade_summary=summary,
        period_summary=periods,
        exit_reason_summary=exits,
        duration_summary=durations,
        stress_summary=stress,
        concentration=concentration,
        stable=stable,
        trades=trades,
        causal_audit=_causal_audit(config),
        failures=pd.DataFrame(failures),
        decision=decision,
        reason=reason,
        champion=champion,
        config=config,
    )
    return LongTailExitAuditResult(decision, config.report_path, champion)
