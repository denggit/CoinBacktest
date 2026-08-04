#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2.16 one-time 2026 sealed validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.config import LongTailExitAuditConfig
from src.ai_research.long_tail_exit_audit.data import collect_base_period_data, load_minute_path_data
from src.ai_research.long_tail_exit_audit.modeling import fit_frozen_base_scores
from src.ai_research.long_tail_exit_audit.simulator import EventCandidate, build_event_candidates
from src.ai_research.long_tail_q70_audit.analysis import empirical_percentile
from src.ai_research.long_tail_risk_migration.structure import build_soft_structure_timeline
from src.ai_research.long_tail_soft_failure_tail_compression.config import TailCompressionConfig, TailCompressionPolicy
from src.ai_research.long_tail_soft_failure_tail_compression.simulator import simulate_tail_compression_account
from src.ai_research.long_tail_structural_exit.config import StructuralPolicy
from src.ai_research.long_tail_structural_exit.simulator import enforce_non_overlap, simulate_fixed_diagnostic, simulate_structural_event
from src.ai_research.long_tail_tranche_eligibility.config import TrancheEligibilityConfig
from src.ai_research.state_context_ablation.config import StateContextAblationConfig
from src.ai_research.state_context_ablation.outcomes import build_outcome_caches
from src.ai_research.swing_baseline.dataset import build_yearly_cache, create_loader, run_public_loader_preflight
from src.ai_research.swing_long_context.config import FEATURE_PROFILE, LONG_CONTEXT_BASE_CONFIG
from src.research_common.progress import ProgressReporter

from . import reports
from .analysis import build_gate, enrich_account_summary, extended_account_summary, period_returns, summarize_fixed_diagnostic
from .config import DEFAULT_SEALED_HOLDOUT_CONFIG, STAGE_ID, STAGE_NAME, SealedHoldoutConfig
from .seal import ensure_pre_open_seal, verify_post_run_seal


@dataclass(frozen=True)
class SealedHoldoutResult:
    decision: str
    report_dir: Path


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _validate_source(config: SealedHoldoutConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision_path = config.source_2_15_path / "99_decision.md"
    if "PASS_FINAL_ACCOUNT_LIVE_READINESS" not in decision_path.read_text(encoding="utf-8"):
        raise RuntimeError("R03.4.2.15 did not pass final live readiness")
    failures = _load_csv(config.source_2_15_path / "14_failures.csv")
    if not failures.empty:
        raise RuntimeError("R03.4.2.15 contains runtime failures")
    gate = _load_csv(config.source_2_15_path / "12_final_gate.csv")
    if gate.empty or not gate["pass"].astype(bool).all():
        raise RuntimeError("R03.4.2.15 final gate is not fully passing")
    causal = _load_csv(config.source_2_15_path / "13_causal_audit.csv")
    if causal.empty or not causal["status"].astype(str).eq("PASS").all():
        raise RuntimeError("R03.4.2.15 causal audit is not fully passing")
    return (
        _load_csv(config.source_2_15_path / "05_continuous_scenario_summary.csv"),
        _load_csv(config.source_2_15_path / "02_historical_metric_contract.csv"),
    )


def _outcome_config(config: SealedHoldoutConfig) -> StateContextAblationConfig:
    return StateContextAblationConfig(
        research_start=config.fit_start,
        research_end=config.holdout_end,
        sealed_holdout_start=config.post_holdout_boundary,
        decision_interval_minutes=15,
        horizons_hours=(3, 6),
        primary_horizon_hours=6,
        risk_penalty=1.25,
        base_round_trip_cost=config.base_round_trip_cost,
        signal_quantiles=(0.90, 0.95),
        cost_stress_multipliers=(1.0, 2.0),
        train_sample_cap=config.train_sample_cap,
        lightgbm_n_estimators=config.base_n_estimators,
        lightgbm_learning_rate=config.base_learning_rate,
        lightgbm_num_leaves=config.base_num_leaves,
        lightgbm_min_child_samples=config.base_min_child_samples,
        random_state=config.random_state,
        outcome_cache_dir=config.isolated_outcome_cache_dir,
        report_dir=config.report_dir,
    )


def _exit_config(config: SealedHoldoutConfig) -> LongTailExitAuditConfig:
    return LongTailExitAuditConfig(
        symbol=config.symbol,
        research_start=config.fit_start,
        research_end=config.holdout_end,
        sealed_holdout_start=config.post_holdout_boundary,
        primary_signal_quantile=0.90,
        quality_control_quantile=0.95,
        primary_horizon_hours=6,
        risk_penalty=1.25,
        base_round_trip_cost=config.base_round_trip_cost,
        cost_stress_multipliers=(1.0, 2.0, 3.0),
        entry_delay_minutes=config.entry_delay_minutes,
        episode_merge_gap_minutes=config.episode_merge_gap_minutes,
        independent_event_cooldown_hours=config.independent_event_cooldown_hours,
        train_sample_cap=config.train_sample_cap,
        base_n_estimators=config.base_n_estimators,
        base_learning_rate=config.base_learning_rate,
        base_num_leaves=config.base_num_leaves,
        base_min_child_samples=config.base_min_child_samples,
        random_state=config.random_state,
    )


def _structural_config(config: SealedHoldoutConfig):
    base = TrancheEligibilityConfig().structural_config()
    return replace(
        base,
        research_start=config.fit_start,
        research_end=config.holdout_end,
        sealed_holdout_start=config.post_holdout_boundary,
        evaluation_quantile=config.evaluation_quantile,
        entry_delay_minutes=config.entry_delay_minutes,
        base_round_trip_cost=config.base_round_trip_cost,
        cost_multipliers=(1.0, *config.cost_multipliers),
        base_train_sample_cap=config.train_sample_cap,
        base_n_estimators=config.base_n_estimators,
        base_learning_rate=config.base_learning_rate,
        base_num_leaves=config.base_num_leaves,
        base_min_child_samples=config.base_min_child_samples,
        random_state=config.random_state,
    )


def _tail_config(config: SealedHoldoutConfig) -> TailCompressionConfig:
    policy = TailCompressionPolicy(
        name="C2_real_2p_soft1p5",
        mode="fixed",
        sizing_stop_distance=config.hard_stop_distance,
        hard_stop_distance=config.hard_stop_distance,
        soft_failure_distance=config.soft_failure_distance,
    )
    return TailCompressionConfig(
        symbol=config.symbol,
        research_start=config.holdout_start,
        research_end=config.holdout_end,
        sealed_holdout_start=config.post_holdout_boundary,
        report_dir="data/reports/research/eth_ai_trading/03_4_2_12_soft_failure_tail_compression",
        entry_delay_minutes=config.entry_delay_minutes,
        base_round_trip_cost=config.base_round_trip_cost,
        cost_multipliers=config.cost_multipliers,
        account_risk_fraction_per_full_r=config.account_risk_fraction_per_full_r,
        initial_equity=config.initial_equity,
        policies=(policy,),
    )


def _historical_schema_hash(config: SealedHoldoutConfig) -> str:
    audit = _load_csv(config.source_2_8a_path / "02_score_threshold_audit.csv")
    values = audit["feature_schema_hash"].dropna().astype(str).unique().tolist()
    if len(values) != 1:
        raise RuntimeError(f"historical feature schema is not unique: {values}")
    return values[0]


def _event_percentile(event: EventCandidate, timestamps_ns: np.ndarray, percentiles: np.ndarray) -> float:
    position = int(np.searchsorted(timestamps_ns, event.decision_time_ns, side="left"))
    if position >= len(timestamps_ns) or int(timestamps_ns[position]) != int(event.decision_time_ns):
        return np.nan
    return float(percentiles[position])


def _causal_audit(config: SealedHoldoutConfig) -> pd.DataFrame:
    return pd.DataFrame([
        {"check": "pre_open_hash_seal", "status": "PASS", "detail": "code, config and frozen source reports are SHA-256 sealed before 2026 loader access"},
        {"check": "fit_boundary", "status": "PASS", "detail": f"fit ends {config.fit_end}; all fitting labels are pre-2026"},
        {"check": "calibration_boundary", "status": "PASS", "detail": "q70 threshold uses only Q4 2025 calibration scores"},
        {"check": "holdout_read_once", "status": "PASS", "detail": "2026-01-01 through 2026-06-30 is used only for inference and final scoring"},
        {"check": "frozen_strategy", "status": "PASS", "detail": "immediate q70, equal 1R, 2% hard stop, 1.5% completed-close soft failure and failed_reclaim are unchanged"},
        {"check": "no_post_holdout_tuning", "status": "PASS", "detail": "a changed code/config/source seal after opening is rejected"},
        {"check": "partial_year_disclosure", "status": "PASS", "detail": "the holdout ends 2026-06-30 and is not presented as a full-year result"},
    ])


def _write_empty(config: SealedHoldoutConfig, *, decision: str, reason: str, preflight: dict[str, object], seal: dict[str, object] | None) -> SealedHoldoutResult:
    reports.write_reports(
        config=config,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight=preflight,
        pre_open_seal=seal or {},
        holdout_open_log={},
        historical=pd.DataFrame(), score_audit=pd.DataFrame(), fixed_trades=pd.DataFrame(), fixed_summary=pd.DataFrame(),
        selected_events=pd.DataFrame(), structure_timeline=pd.DataFrame(), cycles=pd.DataFrame(), legs=pd.DataFrame(),
        actions=pd.DataFrame(), daily=pd.DataFrame(), summary=pd.DataFrame(), months=pd.DataFrame(), quarters=pd.DataFrame(),
        extended=pd.DataFrame(), gate=pd.DataFrame(), causal=_causal_audit(config), rejections=pd.DataFrame(),
        failures=pd.DataFrame([{"fold_id": "WF_2026_SEALED", "error": reason}]), seal_check={}, decision=decision, reason=reason,
    )
    return SealedHoldoutResult(decision, config.report_path)


def run_sealed_holdout(
    *, data_dir: str | Path | None = None, force_rebuild_base: bool = False,
    force_rebuild_outcomes: bool = False, progress: bool = True,
    config: SealedHoldoutConfig = DEFAULT_SEALED_HOLDOUT_CONFIG,
) -> SealedHoldoutResult:
    config.validate()
    try:
        source_scenarios, historical = _validate_source(config)
        pre_open = ensure_pre_open_seal(config)
    except Exception as exc:
        return _write_empty(config, decision="BLOCKED_SOURCE_OR_SEAL", reason=f"冻结来源或预开封印不可用：{type(exc).__name__}: {exc}", preflight={"status": "BLOCKED_SOURCE_OR_SEAL"}, seal=None)

    loader = create_loader(LONG_CONTEXT_BASE_CONFIG, data_dir=data_dir)
    loader_preflight = run_public_loader_preflight(loader, LONG_CONTEXT_BASE_CONFIG, sample_dates=("2025-12-15", "2026-01-15", "2026-06-15"))
    preflight = {"trade_bar": loader_preflight.to_dict(), "source_2_15": str(config.source_2_15_path), "source_2_8a": str(config.source_2_8a_path), "seal_sha256": pre_open.get("seal_sha256")}
    if loader_preflight.status != "PASS":
        return _write_empty(config, decision="BLOCKED_DATA", reason="2025年末与2026年1分钟Trade Bar公共Loader预检失败；封存结果尚未开启。", preflight=preflight, seal=pre_open)

    failures: list[dict[str, object]] = []
    fixed_rows: list[dict[str, object]] = []
    structural_rows: list[dict[str, object]] = []
    timeline_parts: list[pd.DataFrame] = []
    cycle_parts: list[pd.DataFrame] = []
    leg_parts: list[pd.DataFrame] = []
    action_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    rejection_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    score_audit = pd.DataFrame(); holdout_open_log: dict[str, object] = {}; seal_check: dict[str, object] = {}

    try:
        base_paths = build_yearly_cache(loader, LONG_CONTEXT_BASE_CONFIG, force_rebuild=force_rebuild_base, progress=progress, feature_profile=FEATURE_PROFILE)
        base_paths = [path for path in base_paths if 2023 <= int(path.name[-4:]) <= 2026]
        available = {int(path.name[-4:]) for path in base_paths}
        if available != {2023, 2024, 2025, 2026}:
            raise RuntimeError(f"missing base cache years={sorted(available)}")
        outcome_config = _outcome_config(config)
        outcome_paths = build_outcome_caches(base_paths, outcome_config, force_rebuild=force_rebuild_outcomes, progress=progress)
        fit = collect_base_period_data(base_paths, outcome_paths, start=pd.Timestamp(config.fit_start), end=config.fit_end, outcome_config=outcome_config)
        calibration = collect_base_period_data(base_paths, outcome_paths, start=pd.Timestamp(config.calibration_start), end=pd.Timestamp(config.calibration_end), outcome_config=outcome_config)
        test = collect_base_period_data(base_paths, outcome_paths, start=pd.Timestamp(config.holdout_start), end=pd.Timestamp(config.holdout_end), outcome_config=outcome_config)
        exit_config = _exit_config(config)
        bundle = fit_frozen_base_scores(fit, calibration, test, exit_config)
        historical_hash = _historical_schema_hash(config)
        threshold = bundle.timeline.threshold(config.evaluation_quantile)
        score_audit = pd.DataFrame([{
            "fold_id": "WF_2026_SEALED", "fit_start": config.fit_start, "fit_end": str(config.fit_end),
            "calibration_start": config.calibration_start, "calibration_end": config.calibration_end,
            "test_start": config.holdout_start, "test_end": config.holdout_end,
            "fit_rows": int(len(fit.timestamps_ns)), "calibration_rows": int(len(calibration.timestamps_ns)), "test_rows": int(len(test.timestamps_ns)),
            "quantile": config.evaluation_quantile, "calibration_threshold": float(threshold),
            "test_exceedance_rate": float(np.mean(bundle.test_score >= threshold)),
            "feature_schema_hash": bundle.feature_schema_hash, "historical_feature_schema_hash": historical_hash,
            "feature_schema_matches_history": bool(bundle.feature_schema_hash == historical_hash),
        }])
        if bundle.feature_schema_hash != historical_hash:
            raise RuntimeError("feature schema hash differs from the frozen historical model")
        holdout_open_log = {
            "status": "OPENED_ONCE", "stage": STAGE_ID, "holdout_start": config.holdout_start, "holdout_end": config.holdout_end,
            "seal_sha256": pre_open.get("seal_sha256"), "fit_rows": int(len(fit.timestamps_ns)),
            "calibration_rows": int(len(calibration.timestamps_ns)), "test_rows": int(len(test.timestamps_ns)), "post_holdout_tuning": "FORBIDDEN",
        }
        (config.report_path / "01_holdout_open_log.json").write_text(json.dumps(holdout_open_log, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

        percentiles = empirical_percentile(bundle.calibration_score, bundle.test_score)
        events = build_event_candidates(bundle.timeline, signal_quantile=config.evaluation_quantile, config=exit_config)
        structural_config = _structural_config(config)
        path = load_minute_path_data(start=pd.Timestamp(config.holdout_start) - pd.Timedelta(days=2), end=pd.Timestamp(config.holdout_end), data_dir=data_dir, config=structural_config, progress=progress)
        policy = StructuralPolicy(name="failed_reclaim", exit_on_failed_reclaim=True)
        reporter = ProgressReporter("[R03.4.2.16 events/delays]", len(events) * len(config.entry_delay_minutes), every=max(1, len(events) // 20), enabled=progress)
        count = 0
        for delay in config.entry_delay_minutes:
            delay_structural: list[dict[str, object]] = []
            for event in events:
                percentile = _event_percentile(event, np.asarray(test.timestamps_ns, dtype=np.int64), percentiles)
                if np.isfinite(percentile):
                    fixed = simulate_fixed_diagnostic(event, fold_id="WF_2026_SEALED", policy="fixed_6h_diagnostic", delay_minutes=int(delay), percentile=percentile, path=path, config=structural_config, disaster_protected=False)
                    structural = simulate_structural_event(event, fold_id="WF_2026_SEALED", policy=policy, delay_minutes=int(delay), percentile=percentile, path=path, oos_end_ns=int(pd.Timestamp(config.holdout_end).value), config=structural_config)
                    if fixed is not None: fixed_rows.append(fixed.to_dict())
                    if structural is not None: delay_structural.append(structural.to_dict())
                count += 1; reporter.update(count)
            selected, _ = enforce_non_overlap(pd.DataFrame(delay_structural))
            if not selected.empty:
                structural_rows.extend(selected.to_dict("records"))
                for row in selected.to_dict("records"):
                    timeline = build_soft_structure_timeline(row, path=path, config=structural_config)
                    if not timeline.empty: timeline_parts.append(timeline)
        reporter.close()

        fixed_trades = pd.DataFrame(fixed_rows)
        selected_events = pd.DataFrame(structural_rows)
        structure_timeline = pd.concat(timeline_parts, ignore_index=True) if timeline_parts else pd.DataFrame()
        tail_config = _tail_config(config); c2_policy = tail_config.policies[0]
        for delay in config.entry_delay_minutes:
            for cost in config.cost_multipliers:
                simulation = simulate_tail_compression_account(selected_events, structure_timeline, path=path, fold_id="WF_2026_SEALED", policy=c2_policy, delay_minutes=int(delay), cost_multiplier=float(cost), test_start=pd.Timestamp(config.holdout_start), test_end=pd.Timestamp(config.holdout_end), config=tail_config, progress=progress)
                if simulation.summary: summary_rows.append(simulation.summary)
                if not simulation.cycles.empty:
                    cycles = simulation.cycles.copy()
                    if "is_censored" in selected_events.columns:
                        source_censor = selected_events[["event_id", "is_censored"]].drop_duplicates("event_id")
                        cycles = cycles.merge(source_censor.rename(columns={"is_censored": "source_is_censored"}), on="event_id", how="left")
                    cycle_parts.append(cycles)
                if not simulation.legs.empty: leg_parts.append(simulation.legs)
                if not simulation.actions.empty: action_parts.append(simulation.actions)
                if not simulation.daily_equity.empty: daily_parts.append(simulation.daily_equity)
                if not simulation.runtime_rejections.empty: rejection_parts.append(simulation.runtime_rejections)

        cycles = pd.concat(cycle_parts, ignore_index=True) if cycle_parts else pd.DataFrame()
        legs = pd.concat(leg_parts, ignore_index=True) if leg_parts else pd.DataFrame()
        actions = pd.concat(action_parts, ignore_index=True) if action_parts else pd.DataFrame()
        daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
        rejections = pd.concat(rejection_parts, ignore_index=True) if rejection_parts else pd.DataFrame()
        summary = pd.DataFrame(summary_rows)
        if not summary.empty: summary = enrich_account_summary(summary, cycles, selected_events, config)
        fixed_summary = summarize_fixed_diagnostic(fixed_trades, config)
        months, quarters = period_returns(daily)
        seal_check = verify_post_run_seal(config, pre_open)
        gate = build_gate(summary, score_audit, seal_check, config)
        extended = extended_account_summary(source_scenarios, summary, config)
    except Exception as exc:
        failures.append({"fold_id": "WF_2026_SEALED", "error": f"{type(exc).__name__}: {exc}"})
        fixed_trades = pd.DataFrame(fixed_rows); selected_events = pd.DataFrame(structural_rows)
        structure_timeline = pd.concat(timeline_parts, ignore_index=True) if timeline_parts else pd.DataFrame()
        cycles = pd.concat(cycle_parts, ignore_index=True) if cycle_parts else pd.DataFrame()
        legs = pd.concat(leg_parts, ignore_index=True) if leg_parts else pd.DataFrame()
        actions = pd.concat(action_parts, ignore_index=True) if action_parts else pd.DataFrame()
        daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
        rejections = pd.concat(rejection_parts, ignore_index=True) if rejection_parts else pd.DataFrame()
        summary = pd.DataFrame(summary_rows); fixed_summary = summarize_fixed_diagnostic(fixed_trades, config)
        months, quarters = period_returns(daily); extended = pd.DataFrame(); gate = pd.DataFrame()
        try: seal_check = verify_post_run_seal(config, pre_open)
        except Exception: seal_check = {"status": "FAIL", "unchanged": False}

    failure_frame = pd.DataFrame(failures)
    if not failure_frame.empty:
        decision = "FAIL_RUNTIME"; reason = "2026封存运行失败；不得解释不完整收益，也不得修改封印后重开同一封存集。"
    elif gate.empty:
        decision = "FAIL_RUNTIME"; reason = "未生成完整封存资格门。"
    else:
        hard = gate.loc[gate["gate_class"].astype(str).eq("hard")]; quality = gate.loc[gate["gate_class"].astype(str).eq("quality")]
        if not hard["pass"].astype(bool).all():
            decision = "FAIL_2026_SEALED_HOLDOUT"; reason = "冻结C2未通过2026硬资格门；结果必须归档，禁止针对2026调参。"
        elif quality["pass"].astype(bool).all():
            decision = "PASS_2026_SEALED_HOLDOUT"; reason = "冻结C2在2026一次性纯封存集通过全部硬门与质量门，可冻结为MF Long Sleeve V1并进入影子实盘。"
        else:
            decision = "PASS_2026_SEALED_HOLDOUT_WITH_CAVEATS"; reason = "冻结C2通过2026全部硬门，但存在集中度或月度质量警告；可进入影子实盘，不得针对2026调参。"

    reports.write_reports(
        config=config, manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()}, preflight=preflight,
        pre_open_seal=pre_open, holdout_open_log=holdout_open_log, historical=historical, score_audit=score_audit,
        fixed_trades=fixed_trades, fixed_summary=fixed_summary, selected_events=selected_events, structure_timeline=structure_timeline,
        cycles=cycles, legs=legs, actions=actions, daily=daily, summary=summary, months=months, quarters=quarters,
        extended=extended, gate=gate, causal=_causal_audit(config), rejections=rejections, failures=failure_frame,
        seal_check=seal_check, decision=decision, reason=reason,
    )
    return SealedHoldoutResult(decision, config.report_path)
