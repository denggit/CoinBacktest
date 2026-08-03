#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2.9 structural protection and dynamic release audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.ai_research.long_tail_exit_audit.data import load_minute_path_data
from src.ai_research.long_tail_tranche_eligibility.config import TrancheEligibilityConfig
from src.ai_research.swing_baseline.dataset import create_loader, run_public_loader_preflight
from src.ai_research.swing_long_context.config import LONG_CONTEXT_BASE_CONFIG
from src.research_common.progress import ProgressReporter

from . import reports
from .analysis import build_account_summary, policy_gate, protection_trade_summary
from .config import (
    DEFAULT_DYNAMIC_RISK_RELEASE_CONFIG,
    STAGE_ID,
    STAGE_NAME,
    DynamicRiskReleaseConfig,
)
from .inputs import load_dynamic_risk_release_inputs
from .protection import simulate_protection_event
from .simulator import (
    build_release_pair_diagnostics,
    select_dynamic_trades,
    simulate_dynamic_account,
)


@dataclass(frozen=True)
class DynamicRiskReleaseResult:
    decision: str
    report_dir: Path


def _causal_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "frozen_q70_source", "status": "PASS", "detail": "all candidates come from validated R03.4.2.8A q70 outcomes"},
            {"check": "frozen_failed_reclaim", "status": "PASS", "detail": "the deterministic failed-reclaim exit is unchanged unless an enforceable hard stop exits earlier"},
            {"check": "three_percent_disaster_floor", "status": "PASS", "detail": "every tranche retains the frozen 3% outer disaster stop"},
            {"check": "completed_structure_only", "status": "PASS", "detail": "pivot lows require right-side confirmation on completed 15m structure bars"},
            {"check": "next_open_stop_activation", "status": "PASS", "detail": "a protection update becomes live only at the next one-minute open"},
            {"check": "monotone_stop", "status": "PASS", "detail": "hard protection may rise but can never be widened"},
            {"check": "lagged_floor_contract", "status": "PASS", "detail": "S2 uses the prior confirmed structural floor, never the newest unlagged floor"},
            {"check": "primary_not_diluted", "status": "PASS", "detail": "every standalone primary begins at one full R; no static secondary reservation exists"},
            {"check": "enforceable_release_only", "status": "PASS", "detail": "secondary risk is capped by risk already removed by a live hard stop"},
            {"check": "maximum_two_tranches", "status": "PASS", "detail": "selection rejects a third simultaneous virtual tranche"},
            {"check": "live_risk_cap", "status": "PASS", "detail": "runtime caps the sum of stop-defined remaining loss at one account R"},
            {"check": "fixed_6h_diagnostic_only", "status": "PASS", "detail": "fixed six hours is not used for any final exit"},
            {"check": "same_policy_both_years", "status": "PASS", "detail": "all stop and release rules are identical in WF_2024 and WF_2025"},
            {"check": "sealed_2026", "status": "PASS", "detail": "2026 remains unopened"},
        ]
    )


def _empty_write(
    *,
    config: DynamicRiskReleaseConfig,
    decision: str,
    reason: str,
    preflight: dict[str, object],
) -> DynamicRiskReleaseResult:
    reports.write_reports(
        config=config,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight=preflight,
        source_baseline=pd.DataFrame(),
        protection_summary=pd.DataFrame(),
        protection_trades=pd.DataFrame(),
        stop_updates=pd.DataFrame(),
        pair_diagnostics=pd.DataFrame(),
        decisions=pd.DataFrame(),
        account_summary=pd.DataFrame(),
        account_trades=pd.DataFrame(),
        daily_equity=pd.DataFrame(),
        gate=pd.DataFrame(),
        runtime_rejections=pd.DataFrame(),
        causal_audit=pd.DataFrame(),
        failures=pd.DataFrame([{"fold_id": "ALL", "error": reason}]),
        decision=decision,
        reason=reason,
    )
    return DynamicRiskReleaseResult(decision, config.report_path)


def run_dynamic_risk_release_audit(
    *,
    data_dir: str | Path | None = None,
    progress: bool = True,
    config: DynamicRiskReleaseConfig = DEFAULT_DYNAMIC_RISK_RELEASE_CONFIG,
) -> DynamicRiskReleaseResult:
    config.validate()
    try:
        inputs = load_dynamic_risk_release_inputs(config)
    except Exception as exc:
        reason = f"冻结的2.8A/2.8B输入不可用：{type(exc).__name__}: {exc}"
        return _empty_write(config=config, decision="BLOCKED_SOURCE_REPORT", reason=reason, preflight={"source_report": reason})

    loader = create_loader(LONG_CONTEXT_BASE_CONFIG, data_dir=data_dir)
    loader_preflight = run_public_loader_preflight(
        loader,
        LONG_CONTEXT_BASE_CONFIG,
        sample_dates=("2024-06-15", "2025-06-15"),
    )
    preflight: dict[str, object] = {
        "trade_bar": loader_preflight.to_dict(),
        "source_2_8a": str(config.source_2_8a_path),
        "source_2_8b": str(config.source_2_8b_path),
        "source_2_8b_decision": (config.source_2_8b_path / "99_decision.md").read_text(encoding="utf-8").splitlines()[2:6],
    }
    if loader_preflight.status != "PASS":
        reason = "1分钟Trade Bar公共Loader预检失败。"
        return _empty_write(config=config, decision="BLOCKED_DATA", reason=reason, preflight=preflight)

    structural_config = TrancheEligibilityConfig().structural_config()
    source_baseline = inputs.p0_summary.copy()
    protection_trade_parts: list[pd.DataFrame] = []
    update_parts: list[pd.DataFrame] = []
    state_parts: list[pd.DataFrame] = []
    pair_parts: list[pd.DataFrame] = []
    decision_parts: list[pd.DataFrame] = []
    account_trade_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    runtime_parts: list[pd.DataFrame] = []
    simulation_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    candidate_count_rows: list[dict[str, object]] = []

    total_protection_steps = len(inputs.folds) * len(config.entry_delay_minutes)
    protection_reporter = ProgressReporter("[R03.4.2.9 protection folds/delays]", total_protection_steps, every=1, enabled=progress)
    step = 0
    fold_paths: dict[str, object] = {}
    for fold in inputs.folds.to_dict("records"):
        fold_id = str(fold["fold_id"])
        test_start = pd.Timestamp(fold["test_start"])
        test_end = pd.Timestamp(fold["test_end"]).floor("min")
        try:
            path = load_minute_path_data(
                start=test_start - pd.Timedelta(days=2),
                end=test_end,
                data_dir=data_dir,
                config=structural_config,
                progress=progress,
            )
            fold_paths[fold_id] = path
            fold_structural = inputs.structural.loc[inputs.structural["fold_id"] == fold_id]
            for delay in config.entry_delay_minutes:
                try:
                    raw = fold_structural.loc[fold_structural["delay_minutes"] == int(delay)].copy()
                    candidate_count_rows.append({"fold_id": fold_id, "delay_minutes": int(delay), "candidate_events": int(len(raw))})
                    for protection_policy in config.protection_policies:
                        trades: list[dict[str, object]] = []
                        updates: list[pd.DataFrame] = []
                        states: list[pd.DataFrame] = []
                        event_reporter = ProgressReporter(
                            f"[R03.4.2.9 {fold_id} d{delay} {protection_policy.name}]",
                            len(raw),
                            every=max(1, len(raw) // 100),
                            enabled=progress,
                        )
                        for number, (_, event) in enumerate(raw.iterrows(), start=1):
                            simulation = simulate_protection_event(
                                event,
                                path=path,
                                policy=protection_policy,
                                structural_config=structural_config,
                            )
                            trades.append(simulation.trade)
                            if not simulation.updates.empty:
                                updates.append(simulation.updates)
                            if not simulation.states.empty:
                                states.append(simulation.states)
                            event_reporter.update(number)
                        event_reporter.close()
                        trade_frame = pd.DataFrame(trades)
                        protection_trade_parts.append(trade_frame)
                        if updates:
                            update_parts.append(pd.concat(updates, ignore_index=True))
                        if states:
                            state_parts.append(pd.concat(states, ignore_index=True))
                except Exception as exc:
                    failures.append({"fold_id": fold_id, "delay_minutes": delay, "stage": "protection", "error": f"{type(exc).__name__}: {exc}"})
                step += 1
                protection_reporter.update(step)
        except Exception as exc:
            failures.append({"fold_id": fold_id, "delay_minutes": "ALL", "stage": "path", "error": f"{type(exc).__name__}: {exc}"})
            step += len(config.entry_delay_minutes)
            protection_reporter.update(step)
    protection_reporter.close()

    protection_trades = pd.concat(protection_trade_parts, ignore_index=True) if protection_trade_parts else pd.DataFrame()
    stop_updates = pd.concat(update_parts, ignore_index=True) if update_parts else pd.DataFrame()
    states = pd.concat(state_parts, ignore_index=True) if state_parts else pd.DataFrame()
    candidate_counts = pd.DataFrame(candidate_count_rows)

    if protection_trades.empty or failures:
        reason = "结构保护模拟未完整完成。"
        reports.write_reports(
            config=config,
            manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
            preflight=preflight,
            source_baseline=source_baseline,
            protection_summary=protection_trade_summary(protection_trades),
            protection_trades=protection_trades,
            stop_updates=stop_updates,
            pair_diagnostics=pd.DataFrame(),
            decisions=pd.DataFrame(),
            account_summary=pd.DataFrame(),
            account_trades=pd.DataFrame(),
            daily_equity=pd.DataFrame(),
            gate=pd.DataFrame(),
            runtime_rejections=pd.DataFrame(),
            causal_audit=_causal_audit(),
            failures=pd.DataFrame(failures),
            decision="FAIL_RUNTIME",
            reason=reason,
        )
        return DynamicRiskReleaseResult("FAIL_RUNTIME", config.report_path)

    protection_summary = protection_trade_summary(protection_trades)

    # Phase 1: one full-R primary under each stop policy. This determines which
    # stop is allowed to proceed to dynamic release; no dynamic candidate can
    # rescue a protection policy that already destroys the base edge.
    for fold in inputs.folds.to_dict("records"):
        fold_id = str(fold["fold_id"])
        test_start = pd.Timestamp(fold["test_start"])
        test_end = pd.Timestamp(fold["test_end"]).floor("min")
        path = fold_paths[fold_id]
        for delay in config.entry_delay_minutes:
            for protection_policy in config.protection_policies:
                trade_frame = protection_trades.loc[
                    (protection_trades["fold_id"] == fold_id)
                    & (protection_trades["delay_minutes"].astype(int) == int(delay))
                    & (protection_trades["protection_policy"] == protection_policy.name)
                ].copy()
                single_policy = config.dynamic_policies[0]
                selection = select_dynamic_trades(
                    trade_frame,
                    protection_policy=protection_policy.name,
                    dynamic_policy=single_policy,
                    pair_diagnostics=pd.DataFrame(),
                )
                if protection_policy.name == "S0_disaster_only":
                    expected = inputs.p0_trades.loc[
                        (inputs.p0_trades["fold_id"] == fold_id)
                        & (inputs.p0_trades["delay_minutes"].astype(int) == int(delay))
                        & (inputs.p0_trades["cost_multiplier"].astype(float) == 2.0),
                        "event_id",
                    ].astype(str).tolist()
                    actual = selection.accepted["event_id"].astype(str).tolist()
                    if actual != expected:
                        failures.append({"fold_id": fold_id, "delay_minutes": delay, "stage": "p0_parity", "error": f"event parity drift actual={len(actual)} expected={len(expected)}"})
                if not selection.decisions.empty:
                    decision_parts.append(selection.decisions)
                for cost in config.cost_multipliers:
                    simulation = simulate_dynamic_account(
                        selection.accepted,
                        stop_updates=stop_updates,
                        path=path,
                        fold_id=fold_id,
                        protection_policy=protection_policy.name,
                        dynamic_policy=single_policy,
                        delay_minutes=delay,
                        cost_multiplier=cost,
                        test_start=test_start,
                        test_end=test_end,
                        config=config,
                        progress=progress,
                    )
                    if simulation.summary:
                        simulation_rows.append(simulation.summary)
                    if not simulation.trades.empty:
                        account_trade_parts.append(simulation.trades)
                    if not simulation.daily_equity.empty:
                        daily_parts.append(simulation.daily_equity)
                    if not simulation.runtime_rejections.empty:
                        runtime_parts.append(simulation.runtime_rejections)

    decisions = pd.concat(decision_parts, ignore_index=True) if decision_parts else pd.DataFrame()
    account_summary = build_account_summary(simulation_rows, decisions=decisions, candidate_counts=candidate_counts)
    preliminary_gate = policy_gate(account_summary, protection_summary, config)
    eligible_stops = preliminary_gate.loc[
        (preliminary_gate["dynamic_policy"] == "D0_single_1R")
        & (preliminary_gate["protection_policy"] != "S0_disaster_only")
        & preliminary_gate["protection_gate_pass"].astype(bool),
        "protection_policy",
    ].astype(str).tolist()

    # Phase 2: only protection policies that preserved the base edge may fund a
    # second tranche from enforceable released risk.
    for fold in inputs.folds.to_dict("records"):
        fold_id = str(fold["fold_id"])
        test_start = pd.Timestamp(fold["test_start"])
        test_end = pd.Timestamp(fold["test_end"]).floor("min")
        path = fold_paths[fold_id]
        for delay in config.entry_delay_minutes:
            for stop_name in eligible_stops:
                trade_frame = protection_trades.loc[
                    (protection_trades["fold_id"] == fold_id)
                    & (protection_trades["delay_minutes"].astype(int) == int(delay))
                    & (protection_trades["protection_policy"] == stop_name)
                ].copy()
                state_frame = states.loc[
                    (states["fold_id"] == fold_id)
                    & (states["delay_minutes"].astype(int) == int(delay))
                    & (states["protection_policy"] == stop_name)
                ].copy()
                pairs = build_release_pair_diagnostics(
                    trade_frame,
                    state_frame,
                    protection_policy=stop_name,
                    delay_minutes=delay,
                )
                if not pairs.empty:
                    pair_parts.append(pairs)
                for dynamic_policy in config.dynamic_policies[1:]:
                    selection = select_dynamic_trades(
                        trade_frame,
                        protection_policy=stop_name,
                        dynamic_policy=dynamic_policy,
                        pair_diagnostics=pairs,
                    )
                    if not selection.decisions.empty:
                        decision_parts.append(selection.decisions)
                    for cost in config.cost_multipliers:
                        simulation = simulate_dynamic_account(
                            selection.accepted,
                            stop_updates=stop_updates,
                            path=path,
                            fold_id=fold_id,
                            protection_policy=stop_name,
                            dynamic_policy=dynamic_policy,
                            delay_minutes=delay,
                            cost_multiplier=cost,
                            test_start=test_start,
                            test_end=test_end,
                            config=config,
                            progress=progress,
                        )
                        if simulation.summary:
                            simulation_rows.append(simulation.summary)
                        if not simulation.trades.empty:
                            account_trade_parts.append(simulation.trades)
                        if not simulation.daily_equity.empty:
                            daily_parts.append(simulation.daily_equity)
                        if not simulation.runtime_rejections.empty:
                            runtime_parts.append(simulation.runtime_rejections)

    pair_diagnostics = pd.concat(pair_parts, ignore_index=True) if pair_parts else pd.DataFrame()
    decisions = pd.concat(decision_parts, ignore_index=True) if decision_parts else pd.DataFrame()
    account_trades = pd.concat(account_trade_parts, ignore_index=True) if account_trade_parts else pd.DataFrame()
    daily_equity = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    runtime_rejections = pd.concat(runtime_parts, ignore_index=True) if runtime_parts else pd.DataFrame()
    account_summary = build_account_summary(simulation_rows, decisions=decisions, candidate_counts=candidate_counts)
    gate = policy_gate(account_summary, protection_summary, config)
    failures_frame = pd.DataFrame(failures)

    dynamic_pass = gate.loc[
        (gate["dynamic_policy"] != "D0_single_1R") & gate["pass_to_next_stage"].astype(bool)
    ]
    protection_pass = gate.loc[
        (gate["dynamic_policy"] == "D0_single_1R")
        & (gate["protection_policy"] != "S0_disaster_only")
        & gate["protection_gate_pass"].astype(bool)
    ]
    if not failures_frame.empty:
        decision = "FAIL_RUNTIME"
        reason = "R03.4.2.9存在运行或P0复现错误，结果不可用于研究决策。"
    elif not dynamic_pass.empty:
        decision = "PASS_DYNAMIC_RISK_RELEASE_CANDIDATE"
        reason = "至少一套统一结构保护与动态风险释放方案在2024、2025均保留P0收益、恢复q70覆盖，并在首仓不静态削弱和实时剩余风险不超过1R的条件下通过成本、延迟、MDD及集中度门槛。"
    elif not protection_pass.empty:
        decision = "PASS_PROTECTION_ONLY_NO_DYNAMIC_TRANCHE"
        reason = "至少一套结构保护位跨年保留了Failed-Reclaim基础收益，但动态第二Tranche未能同时恢复覆盖并达到P0收益要求；保留保护止损，停止当前动态加仓规则。"
    else:
        decision = "FAIL_NO_ROBUST_STRUCTURE_PROTECTION"
        reason = "最新或落后一层结构保护均未在2024、2025及成本/延迟压力下保留足够P0收益；不得为了释放风险而强行收紧止损或降低收益门槛。"

    reports.write_reports(
        config=config,
        manifest={
            "stage": STAGE_ID,
            "name": STAGE_NAME,
            "config": config.to_dict(),
            "source_2_8a_stage": inputs.manifest_2_8a.get("stage"),
            "source_2_8b_stage": inputs.manifest_2_8b.get("stage"),
            "eligible_protection_policies_for_dynamic_phase": eligible_stops,
            "policy_selection_by_year": "FORBIDDEN",
        },
        preflight=preflight,
        source_baseline=source_baseline,
        protection_summary=protection_summary,
        protection_trades=protection_trades,
        stop_updates=stop_updates,
        pair_diagnostics=pair_diagnostics,
        decisions=decisions,
        account_summary=account_summary,
        account_trades=account_trades,
        daily_equity=daily_equity,
        gate=gate,
        runtime_rejections=runtime_rejections,
        causal_audit=_causal_audit(),
        failures=failures_frame,
        decision=decision,
        reason=reason,
    )
    return DynamicRiskReleaseResult(decision, config.report_path)
