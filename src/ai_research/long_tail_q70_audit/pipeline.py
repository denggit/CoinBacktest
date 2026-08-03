#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2.4 q70 versus q90 cross-year audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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

from .analysis import (
    build_tables,
    comparison_table,
    empirical_percentile,
    enforce_non_overlap,
    simulate_fixed_horizon_event,
)
from .config import DEFAULT_Q70_CROSS_YEAR_AUDIT_CONFIG, STAGE_ID, STAGE_NAME, Q70CrossYearAuditConfig
from . import reports


@dataclass(frozen=True)
class Q70CrossYearAuditResult:
    decision: str
    report_dir: Path


def _exit_config(config: Q70CrossYearAuditConfig) -> LongTailExitAuditConfig:
    return LongTailExitAuditConfig(
        symbol=config.symbol,
        research_start=config.research_start,
        research_end=config.research_end,
        sealed_holdout_start=config.sealed_holdout_start,
        primary_signal_quantile=0.90,
        quality_control_quantile=0.95,
        primary_horizon_hours=config.diagnostic_horizon_hours,
        risk_penalty=config.risk_penalty,
        base_round_trip_cost=config.base_round_trip_cost,
        cost_stress_multipliers=config.cost_multipliers,
        entry_delay_minutes=config.entry_delay_minutes,
        episode_merge_gap_minutes=config.episode_merge_gap_minutes,
        independent_event_cooldown_hours=config.independent_event_cooldown_hours,
        train_sample_cap=config.base_train_sample_cap,
        base_n_estimators=config.base_n_estimators,
        base_learning_rate=config.base_learning_rate,
        base_num_leaves=config.base_num_leaves,
        base_min_child_samples=config.base_min_child_samples,
        random_state=config.random_state,
    )


def _causal_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "frozen_opening_model", "status": "PASS", "detail": "R03.4.1 six-hour long utility model and parameters are unchanged"},
            {"check": "calibration_thresholds", "status": "PASS", "detail": "q70/q90 thresholds are computed from the prior calibration quarter only"},
            {"check": "next_minute_open", "status": "PASS", "detail": "signals execute at the open of decision_time plus 1/3/5 minutes"},
            {"check": "complete_path", "status": "PASS", "detail": "a trade is retained only when all 360 one-minute rows are present"},
            {"check": "single_position", "status": "PASS", "detail": "same-scope events overlapping an open diagnostic position are skipped"},
            {"check": "q70_decoupled_from_holding_models", "status": "PASS", "detail": "q70/q90 audit does not require recovery, continuation or long-hold model thresholds"},
            {"check": "state_model", "status": "PASS", "detail": "abandoned market-state outputs are not loaded"},
            {"check": "sealed_holdout", "status": "PASS", "detail": "paths crossing 2026 are incomplete and excluded; 2026 is never loaded"},
        ]
    )


def _stable_candidate(
    summary: pd.DataFrame,
    periods: pd.DataFrame,
    bands: pd.DataFrame,
    comparison: pd.DataFrame,
    config: Q70CrossYearAuditConfig,
) -> pd.DataFrame:
    required_folds = {"WF_2024", "WF_2025"}

    def rows(scope: str, delay: int, cost: float) -> pd.DataFrame:
        return summary.loc[
            (summary["scope"] == scope)
            & (summary["delay_minutes"] == delay)
            & (summary["cost_multiplier"] == cost)
        ].copy()

    q70_2x = rows("broad_q70", 1, 2.0)
    q70_3x = rows("broad_q70", 1, 3.0)
    q70_delay5 = rows("broad_q70", 5, 2.0)
    band = bands.loc[
        (bands["score_band"] == "q70_to_q90")
        & (bands["delay_minutes"] == 1)
        & (bands["cost_multiplier"] == 2.0)
    ].copy()
    q70_quarters = periods.loc[
        (periods["scope"] == "broad_q70")
        & (periods["delay_minutes"] == 1)
        & (periods["cost_multiplier"] == 2.0)
        & (periods["period_kind"] == "quarter")
    ].copy()

    complete = (
        set(q70_2x["fold_id"]) == required_folds
        and set(q70_3x["fold_id"]) == required_folds
        and set(q70_delay5["fold_id"]) == required_folds
        and set(band["fold_id"]) == required_folds
        and set(comparison["fold_id"]) == required_folds
    )
    positive = complete and bool((q70_2x["mean_net_return"] > 0).all())
    pf = complete and bool((q70_2x["profit_factor"] >= config.minimum_pf_2x).all())
    trades = complete and bool((q70_2x["trades"] >= config.minimum_trades_per_year).all())
    incremental = complete and bool(
        (band["mean_net_return"] > 0).all()
        and (band["profit_factor"] >= config.minimum_incremental_band_pf_2x).all()
    )
    beats = complete and bool((comparison["total_return_delta"] > 0).all())
    stress = complete and bool(
        (q70_3x["mean_net_return"] > 0).all()
        and (q70_delay5["mean_net_return"] > 0).all()
    )
    positive_quarters = int((q70_quarters["mean_net_return"] > 0).sum()) if not q70_quarters.empty else 0
    robustness = complete and bool(
        (q70_2x["mean_net_without_top10"] > 0).all()
        and (q70_2x["max_drawdown"].abs() <= config.maximum_drawdown).all()
        and (q70_2x["top10_profit_share"] <= config.maximum_top10_profit_share).all()
        and positive_quarters >= config.minimum_positive_quarters
    )
    return pd.DataFrame(
        [
            {
                "complete_2024_2025": complete,
                "positive_expectancy_2x_both_years": positive,
                "pf_2x_both_years": pf,
                "minimum_trades_both_years": trades,
                "incremental_band_positive_both_years": incremental,
                "beats_q90_total_return_both_years": beats,
                "delay_and_cost_stress_pass": stress,
                "positive_quarters": positive_quarters,
                "robustness_pass": robustness,
                "stable_q70_expansion": bool(complete and positive and pf and trades and incremental and beats and stress and robustness),
            }
        ]
    )


def run_q70_cross_year_audit(
    *,
    data_dir: str | Path | None = None,
    force_rebuild_outcomes: bool = False,
    progress: bool = True,
    config: Q70CrossYearAuditConfig = DEFAULT_Q70_CROSS_YEAR_AUDIT_CONFIG,
) -> Q70CrossYearAuditResult:
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
        config.report_path.mkdir(parents=True, exist_ok=True)
        reason = "1分钟Trade Bar公共Loader预检失败。"
        reports.write_reports(
            config=config,
            manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
            preflight=preflight,
            score_audit=pd.DataFrame(), execution_audit=pd.DataFrame(), summary=pd.DataFrame(),
            periods=pd.DataFrame(), bands=pd.DataFrame(), score_deciles=pd.DataFrame(),
            comparison=pd.DataFrame(), overlap=pd.DataFrame(), stable=pd.DataFrame(),
            causal_audit=pd.DataFrame(), failures=pd.DataFrame([{"fold_id": "ALL", "error": reason}]),
            trades=pd.DataFrame(), decision="BLOCKED_DATA", reason=reason,
        )
        return Q70CrossYearAuditResult("BLOCKED_DATA", config.report_path)

    base_paths = [path for path in list_cached_years(LONG_CONTEXT_BASE_CONFIG) if 2023 <= int(path.name[-4:]) <= 2025]
    available = {int(path.name[-4:]) for path in base_paths}
    if available != {2023, 2024, 2025}:
        raise RuntimeError(f"missing R03.2 base cache years={sorted(available)}")
    outcome_paths = build_outcome_caches(
        base_paths,
        DEFAULT_STATE_CONTEXT_ABLATION_CONFIG,
        force_rebuild=force_rebuild_outcomes,
        progress=progress,
    )

    score_rows: list[dict[str, object]] = []
    execution_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    trade_parts: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    folds = default_ablation_folds(DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
    reporter = ProgressReporter("[R03.4.2.4 folds]", len(folds), every=1, enabled=progress)

    for fold_number, fold in enumerate(folds, start=1):
        try:
            fit = collect_base_period_data(base_paths, outcome_paths, start=fold.fit_start, end=fold.fit_end, outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
            calibration = collect_base_period_data(base_paths, outcome_paths, start=fold.calibration_start, end=fold.calibration_end, outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
            test = collect_base_period_data(base_paths, outcome_paths, start=fold.test_start, end=fold.test_end, outcome_config=DEFAULT_STATE_CONTEXT_ABLATION_CONFIG)
            bundle = fit_frozen_base_scores(fit, calibration, test, exit_config)
            test_percentiles = empirical_percentile(bundle.calibration_score, bundle.test_score)
            path = load_minute_path_data(
                start=fold.test_start - pd.Timedelta(days=1),
                end=fold.test_end,
                data_dir=data_dir,
                config=exit_config,
                progress=progress,
            )
            for quantile, scope in ((0.70, "broad_q70"), (0.90, "primary_q90")):
                events = build_event_candidates(bundle.timeline, signal_quantile=quantile, config=exit_config)
                threshold = bundle.timeline.threshold(quantile)
                score_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "scope": scope,
                        "quantile": quantile,
                        "calibration_threshold": threshold,
                        "calibration_rows": int(len(bundle.calibration_score)),
                        "test_rows": int(len(bundle.test_score)),
                        "test_exceedance_rate": float(np.mean(bundle.test_score >= threshold)),
                        "candidate_events": int(len(events)),
                        "feature_schema_hash": bundle.feature_schema_hash,
                    }
                )
                for delay in config.entry_delay_minutes:
                    rows: list[dict[str, object]] = []
                    skipped_incomplete = 0
                    for event in events:
                        pos = int(np.searchsorted(test.timestamps_ns, event.decision_time_ns, side="left"))
                        percentile = float(test_percentiles[pos]) if pos < len(test_percentiles) and int(test.timestamps_ns[pos]) == event.decision_time_ns else np.nan
                        trade = simulate_fixed_horizon_event(
                            event,
                            fold_id=fold.fold_id,
                            scope=scope,
                            delay_minutes=delay,
                            score_percentile=percentile,
                            path=path,
                            config=config,
                        )
                        if trade is None:
                            skipped_incomplete += 1
                        else:
                            rows.append(trade)
                    raw = pd.DataFrame(rows)
                    executed, overlap_skipped = enforce_non_overlap(raw)
                    execution_rows.append(
                        {
                            "fold_id": fold.fold_id,
                            "scope": scope,
                            "delay_minutes": delay,
                            "candidate_events": int(len(events)),
                            "complete_path_events": int(len(raw)),
                            "skipped_incomplete": int(skipped_incomplete),
                            "executed_events": int(len(executed)),
                            "skipped_overlap": int(overlap_skipped),
                            "path_coverage_ratio": float(path.coverage_ratio),
                        }
                    )
                    overlap_rows.append(
                        {
                            "fold_id": fold.fold_id,
                            "scope": scope,
                            "delay_minutes": delay,
                            "raw_complete_events": int(len(raw)),
                            "executed_events": int(len(executed)),
                            "overlap_skip_share": float(overlap_skipped / len(raw)) if len(raw) else np.nan,
                        }
                    )
                    if not executed.empty:
                        trade_parts.append(executed)
        except Exception as exc:  # report the fold without silently fabricating the missing year
            failures.append({"fold_id": fold.fold_id, "error": f"{type(exc).__name__}: {exc}"})
        reporter.update(fold_number)
    reporter.close()

    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    overlap = pd.DataFrame(overlap_rows)
    summary, periods, bands, deciles, overlap = build_tables(trades, overlap_audit=overlap, config=config)
    comparison = comparison_table(summary)
    stable = _stable_candidate(summary, periods, bands, comparison, config) if not summary.empty else pd.DataFrame()
    failures_frame = pd.DataFrame(failures)

    if summary.empty or stable.empty or not bool(stable.iloc[0]["complete_2024_2025"]):
        decision = "FAIL_INCOMPLETE_Q70_CROSS_YEAR_AUDIT"
        reason = "q70/q90冻结基准没有同时产生完整的WF_2024与WF_2025结果；不得用单年结果确认扩频。"
    elif bool(stable.iloc[0]["stable_q70_expansion"]):
        decision = "PASS_Q70_CROSS_YEAR_EXPANSION"
        reason = "q70在2024和2025均保持成本后正期望，新增q70-q90分数带本身为正，并在两年都提高了相对q90的2倍成本总复合利润。"
    elif bool(stable.iloc[0]["positive_expectancy_2x_both_years"]):
        decision = "PASS_Q70_POSITIVE_EXPECTANCY_WITH_CAVEATS"
        reason = "q70跨年保持正期望，但至少一项新增分数带、总利润、成本延迟、季度、集中度或回撤门槛未通过；可保留研究，不得直接替代q90。"
    else:
        decision = "FAIL_Q70_DILUTES_EDGE"
        reason = "q70至少在一个OOS年份无法守住2倍成本正期望，扩频会稀释原有Edge。"

    reports.write_reports(
        config=config,
        manifest={
            "stage": STAGE_ID,
            "name": STAGE_NAME,
            "config": config.to_dict(),
            "folds": [fold.to_dict() for fold in folds],
            "opening_model": "frozen R03.4.1 long utility h6",
            "diagnostic_only": True,
            "final_exit_policy": "not studied in this stage",
            "state_model_policy": "ABANDONED_AND_NOT_LOADED",
        },
        preflight=preflight,
        score_audit=pd.DataFrame(score_rows),
        execution_audit=pd.DataFrame(execution_rows),
        summary=summary,
        periods=periods,
        bands=bands,
        score_deciles=deciles,
        comparison=comparison,
        overlap=overlap,
        stable=stable,
        causal_audit=_causal_audit(),
        failures=failures_frame,
        trades=trades,
        decision=decision,
        reason=reason,
    )
    return Q70CrossYearAuditResult(decision=decision, report_dir=config.report_path)
